import os, json, pickle, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit, GroupKFold, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


print("=" * 58)
print("  VOLTX 2.0 — ML Training Pipeline (Tuned + Safe)")
print("=" * 58)


print("\n[1/5] Loading data...")
DATA_PATH = "data/tneb_smart_meter_readings.csv"
df = pd.read_csv(DATA_PATH)
print(f"   → {len(df):,} rows loaded")


need_cols = ["meter_id", "anomaly_label"]
for c in need_cols:
    if c not in df.columns:
        raise ValueError(f"Missing required column: {c}")


print("\n[2/5] Engineering features...")

# Fill missing 
for col in ["kwh_consumed", "voltage_volts", "current_amps", "power_factor"]:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# Per-meter baseline
stats = df.groupby("meter_id")["kwh_consumed"].agg(
    mean_kwh="mean", std_kwh="std", median_kwh="median"
).reset_index()
stats["std_kwh"] = stats["std_kwh"].fillna(0.1)
df = df.merge(stats, on="meter_id", how="left")

df["kwh_deviation"]     = df["kwh_consumed"] - df["mean_kwh"]
df["kwh_deviation_pct"] = (df["kwh_deviation"] / df["mean_kwh"].clip(lower=0.1)) * 100
df["z_score"]           = (df["kwh_consumed"] - df["mean_kwh"]) / df["std_kwh"].clip(lower=0.1)
df["load_utilisation"]  = (df["kwh_consumed"] / (df["sanctioned_load_kw"] * 24)) * 100
df["reactive_ratio"]    = df["reactive_power_kvar"] / df["apparent_power_kva"].clip(lower=0.001)
df["current_per_kwh"]   = df["current_amps"] / df["kwh_consumed"].clip(lower=0.1)

# Rule flags
df["flag_spike"]       = (df["z_score"] > 3.0).astype(int)
df["flag_low_pf"]      = (df["power_factor"] < 0.75).astype(int)
df["flag_voltage"]     = ((df["voltage_volts"] < 185) | (df["voltage_volts"] > 255)).astype(int)
df["flag_low_use"]     = (df["z_score"] < -2.5).astype(int)
df["flag_overload"]    = (df["load_utilisation"] > 120).astype(int)
df["rule_flag_count"]  = (
    df["flag_spike"] + df["flag_low_pf"] + df["flag_voltage"] +
    df["flag_low_use"] + df["flag_overload"]
)

FEATURES = [
    "kwh_consumed","voltage_volts","current_amps","power_factor",
    "apparent_power_kva","reactive_power_kvar","sanctioned_load_kw",
    "mean_kwh","std_kwh","median_kwh",
    "kwh_deviation","kwh_deviation_pct","z_score",
    "load_utilisation","reactive_ratio","current_per_kwh",
    "flag_spike","flag_low_pf","flag_voltage","flag_low_use","flag_overload","rule_flag_count",
    "is_weekend","month",
]

# X/y/groups
X = df[FEATURES].copy()
y = df["anomaly_label"].astype(int).copy()
groups = df["meter_id"].astype(str).values

print(f"   → {len(FEATURES)} features ready")


print("\n[3/5] Creating group-safe Train/Val/Test splits...")

# 20% TEST
gss1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
trainval_idx, test_idx = next(gss1.split(X, y, groups=groups))

X_trainval, X_test = X.iloc[trainval_idx], X.iloc[test_idx]
y_trainval, y_test = y.iloc[trainval_idx], y.iloc[test_idx]
groups_trainval = groups[trainval_idx]

# 20% of trainval => VAL (~16% of total)
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, val_idx = next(gss2.split(X_trainval, y_trainval, groups=groups_trainval))

X_train, X_val = X_trainval.iloc[train_idx], X_trainval.iloc[val_idx]
y_train, y_val = y_trainval.iloc[train_idx], y_trainval.iloc[val_idx]
groups_train = groups_trainval[train_idx]

print(f"   → Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
print(f"   → Anomaly rate (Train/Val/Test): "
      f"{y_train.mean():.3f} / {y_val.mean():.3f} / {y_test.mean():.3f}")


def metric_pack(y_true, y_pred, y_prob):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.0
    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
    }

def print_metrics(tag, pack):
    print(f"   {tag:18s} Acc={pack['accuracy']:.4f}  "
          f"Prec={pack['precision']:.4f}  Rec={pack['recall']:.4f}  "
          f"F1={pack['f1_score']:.4f}  AUC={pack['roc_auc']:.4f}")


print("\n[4/5] Training + tuning models (no leakage)...")

# Preprocess 
preprocess = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler()), #outlier
])

# ── Random Forest (tuned) ────────────────────────────────
rf_pipe = Pipeline([
    ("prep", preprocess),
    ("clf", RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

rf_params = {
    "clf__n_estimators": [150, 200, 300, 450],
    "clf__max_depth": [6, 8, 10, 12, 16, None],
    "clf__min_samples_split": [2, 5, 10, 20],
    "clf__min_samples_leaf": [2, 4, 6, 10],
    "clf__max_features": ["sqrt", "log2", 0.5, 0.7],
}

cv = GroupKFold(n_splits=5)
rf_search = RandomizedSearchCV(
    rf_pipe,
    rf_params,
    n_iter=25,
    scoring="f1",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=0
)
rf_search.fit(X_train, y_train, clf__sample_weight=None, groups=groups_train)
rf_best = rf_search.best_estimator_
print(f"   → RF best params: {rf_search.best_params_}")

# ── Gradient Boosting (tuned) ────────────────────────────
gb_pipe = Pipeline([
    ("prep", preprocess),
    ("clf", GradientBoostingClassifier(random_state=42))
])

gb_params = {
    "clf__n_estimators": [80, 120, 150, 200],
    "clf__learning_rate": [0.03, 0.05, 0.08, 0.12],
    "clf__max_depth": [2, 3, 4, 5],
    "clf__subsample": [0.6, 0.75, 0.85, 1.0],
    "clf__min_samples_leaf": [2, 4, 6, 10]
}

gb_search = RandomizedSearchCV(
    gb_pipe,
    gb_params,
    n_iter=25,
    scoring="f1",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=0
)
gb_search.fit(X_train, y_train, groups=groups_train)
gb_best = gb_search.best_estimator_
print(f"   → GB best params: {gb_search.best_params_}")

# ── Logistic Regression
lr_pipe = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ))
])
lr_pipe.fit(X_train, y_train)

# ── Isolation Forest (unsupervised)
prep_fit = preprocess.fit(X_train)
X_train_s = prep_fit.transform(X_train)
X_all_s   = prep_fit.transform(X)  # for scoring later
iso = IsolationForest(
    n_estimators=300,
    contamination=float(y.mean()),  # approx anomaly rate
    random_state=42,
    n_jobs=-1
)
iso.fit(X_train_s)


print("\n[5/5] Evaluating models (VAL then TEST) + scoring full dataset...")

results = {}

def eval_pipe(name, pipe, X_split, y_split):
    prob = pipe.predict_proba(X_split)[:, 1]
    pred = (prob >= 0.5).astype(int)
    pack = metric_pack(y_split, pred, prob)
    return pack

# Validate (
rf_val = eval_pipe("Random Forest", rf_best, X_val, y_val)
gb_val = eval_pipe("Gradient Boosting", gb_best, X_val, y_val)
lr_val = eval_pipe("Logistic Regression", lr_pipe, X_val, y_val)

print_metrics("RF (VAL)", rf_val)
print_metrics("GB (VAL)", gb_val)
print_metrics("LR (VAL)", lr_val)

# Test 
rf_test = eval_pipe("Random Forest", rf_best, X_test, y_test)
gb_test = eval_pipe("Gradient Boosting", gb_best, X_test, y_test)
lr_test = eval_pipe("Logistic Regression", lr_pipe, X_test, y_test)

# Isolation forest test 
iso_raw_all = iso.predict(X_all_s)
iso_pred_all = np.where(iso_raw_all == -1, 1, 0)
iso_scores = -iso.decision_function(X_all_s)
iso_norm = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-9)
iso_pred_test = iso_pred_all[test_idx]
iso_prob_test = iso_norm[test_idx]
iso_test = metric_pack(y_test, iso_pred_test, iso_prob_test)

print("\n   ── FINAL TEST METRICS (report these) ──")
print_metrics("Isolation (TEST)", iso_test)
print_metrics("RF (TEST)", rf_test)
print_metrics("GB (TEST)", gb_test)
print_metrics("LR (TEST)", lr_test)

results["Isolation Forest"] = {"model":"Isolation Forest", **iso_test}
results["Random Forest"] = {"model":"Random Forest", **rf_test}
results["Gradient Boosting"] = {"model":"Gradient Boosting", **gb_test}
results["Logistic Regression"] = {"model":"Logistic Regression", **lr_test}

# Pick best model by TEST F1
best_name = max(results.keys(), key=lambda k: results[k]["f1_score"])
print(f"\n✅ Best by TEST F1: {best_name} (F1={results[best_name]['f1_score']:.4f})")


rf_best.fit(X_trainval, y_trainval)
gb_best.fit(X_trainval, y_trainval)

# Scores on full dataset (for dashboard)
rf_all = rf_best.predict_proba(X)[:, 1]
gb_all = gb_best.predict_proba(X)[:, 1]

df["iso_score"] = np.round(iso_norm, 4)
df["rf_score"]  = np.round(rf_all, 4)
df["gb_score"]  = np.round(gb_all, 4)

df["final_risk_score"] = (
    0.35 * df["iso_score"] + 0.35 * df["gb_score"] + 0.30 * df["rf_score"]
).clip(0, 1).round(4)

df["risk_band"] = pd.cut(
    df["final_risk_score"],
    bins=[0, 0.30, 0.60, 1.01],
    labels=["Low", "Medium", "High"],
    right=False
)



df["estimated_loss_rs"] = np.where(
    df["risk_band"] == "High",
    (df["mean_kwh"] - df["kwh_consumed"].clip(lower=0)) * df["tariff_rs_per_kwh"],
    0
)
df["estimated_loss_rs"] = pd.Series(df["estimated_loss_rs"]).clip(lower=0).round(2)

df["action_required"] = df["risk_band"].map({
    "High": "INSPECT IMMEDIATELY",
    "Medium": "MONITOR CLOSELY",
    "Low": "NORMAL"
})

# Save outputs
os.makedirs("data", exist_ok=True)
df.to_csv("data/tneb_scored_results.csv", index=False)

pd.DataFrame(list(results.values())).to_csv("data/model_comparison.csv", index=False)
with open("data/model_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Save models + preprocessors
with open("data/iso_model.pkl", "wb") as f: pickle.dump(iso, f)
with open("data/rf_model.pkl", "wb") as f: pickle.dump(rf_best, f)
with open("data/gb_model.pkl", "wb") as f: pickle.dump(gb_best, f)
with open("data/lr_model.pkl", "wb") as f: pickle.dump(lr_pipe, f)
with open("data/preprocess.pkl", "wb") as f: pickle.dump(preprocess, f)

high = (df["risk_band"] == "High").sum()
loss = df["estimated_loss_rs"].sum()

print(f"\n✅ Done! Scored {len(df):,} rows")
print(f"   High-risk rows  : {high:,}")
print(f"   Est. total loss : ₹{loss:,.2f}")
print(f"   Saved → data/tneb_scored_results.csv")
print("=" * 58)