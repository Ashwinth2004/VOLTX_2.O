"""
VOLTX 2.0 — Streamlit Dashboard
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json, os, warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="VOLTX 2.0 | TNEB Theft Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CLEAN WHITE CSS ───────────────────────────────────────
st.markdown("""
<style>
body, .stApp { background:#f8fafc; font-family: system-ui, -apple-system, sans-serif; }
section[data-testid="stSidebar"] { background:#fff; border-right:1px solid #e2e8f0; }
[data-testid="stMetric"] {
    background:#fff; border:1px solid #e2e8f0; border-radius:10px;
    padding:16px 18px !important;
}
[data-testid="stMetricLabel"] { font-size:11px !important; color:#64748b !important;
    text-transform:uppercase; letter-spacing:0.5px; }
[data-testid="stMetricValue"] { font-size:26px !important; font-weight:800 !important; }
[data-testid="stTabs"] button { font-size:13px !important; }
h1 { font-size:24px !important; font-weight:800 !important; color:#0f172a !important; }
h2 { font-size:16px !important; font-weight:700 !important; color:#0f172a !important; }
h3 { font-size:14px !important; font-weight:700 !important; color:#0f172a !important; }
.stButton>button {
    background:#1d4ed8 !important; color:#fff !important;
    border:none !important; border-radius:7px !important;
    font-weight:600 !important; font-size:12px !important;
}
[data-testid="stDataFrame"] { border:1px solid #e2e8f0 !important; border-radius:8px; }
</style>
""", unsafe_allow_html=True)

# ── PLOTLY CLEAN THEME ───────────────────────────────────
LAYOUT = dict(
    paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
    font=dict(family="system-ui", color="#374151", size=11),
    margin=dict(l=10, r=10, t=35, b=10),
    legend=dict(bgcolor="#f8fafc", bordercolor="#e2e8f0", borderwidth=1),
)
GRID = dict(gridcolor="#f1f5f9", zerolinecolor="#e2e8f0")
C = {"High":"#dc2626","Medium":"#d97706","Low":"#16a34a",
     "p1":"#1d4ed8","p2":"#7c3aed","p3":"#059669"}

# ── LOAD DATA ────────────────────────────────────────────
@st.cache_data
def load_data():
    p = "data/tneb_scored_results.csv"
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    df["reading_date"] = pd.to_datetime(df["reading_date"])
    df["risk_band"] = df["risk_band"].astype(str)
    return df

@st.cache_data
def load_model_results():
    p = "data/model_results.json"
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)

df = load_data()
model_results = load_model_results()

# ── SIDEBAR ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ VOLTX 2.0")
    st.caption("TNEB Smart Meter Theft Detection")
    st.markdown("---")

    if df is not None:
        zones = ["All"] + sorted(df["zone"].unique().tolist())
        sel_zone = st.selectbox("📍 Zone", zones)

        areas = ["All"]
        if sel_zone != "All":
            areas += sorted(df[df["zone"] == sel_zone]["area"].unique().tolist())
        else:
            areas += sorted(df["area"].unique().tolist())
        sel_area = st.selectbox("🏘 Area", areas)

        sel_risk = st.selectbox("⚠️ Risk Band", ["All", "High", "Medium", "Low"])

        d_min = df["reading_date"].min().date()
        d_max = df["reading_date"].max().date()
        sel_dates = st.date_input("📅 Date Range", value=(d_min, d_max),
                                   min_value=d_min, max_value=d_max)

        st.markdown("---")
        st.markdown("**📊 Dataset**")
        st.markdown(f"- Rows: **{len(df):,}**")
        st.markdown(f"- Meters: **{df['meter_id'].nunique():,}**")
        st.markdown(f"- Period: **{d_min}** → **{d_max}**")
        st.markdown("---")
        st.markdown("**🏢 Zones**")
        for z in df["zone"].unique():
            n = df[df["zone"]==z]["meter_id"].nunique()
            st.markdown(f"- {z}: `{n}` meters")

# ── FILTER ───────────────────────────────────────────────
if df is not None:
    dff = df.copy()
    if sel_zone != "All":  dff = dff[dff["zone"] == sel_zone]
    if sel_area != "All":  dff = dff[dff["area"] == sel_area]
    if sel_risk != "All":  dff = dff[dff["risk_band"] == sel_risk]
    if len(sel_dates) == 2:
        dff = dff[(dff["reading_date"].dt.date >= sel_dates[0]) &
                  (dff["reading_date"].dt.date <= sel_dates[1])]

# ── MAIN PAGE ────────────────────────────────────────────
st.title("⚡ VOLTX 2.0 — Electricity Theft Detection")
st.caption("TNEB Smart Meter Analytics | Chennai · Avadi · Tiruvallur")
st.markdown("---")

if df is None:
    st.error("⚠️  Scored data not found. Run training first:")
    st.code("python train_models.py", language="bash")
    st.stop()

# ── KPIs ─────────────────────────────────────────────────
high_m  = dff[dff["risk_band"]=="High"]["meter_id"].nunique()
tot_m   = dff["meter_id"].nunique()
anom_r  = dff["anomaly_label"].mean() * 100
tot_l   = dff["estimated_loss_rs"].sum()
med_m   = dff[dff["risk_band"]=="Medium"]["meter_id"].nunique()

c1,c2,c3,c4,c5 = st.columns(5)
with c1: st.metric("📋 Total Readings",  f"{len(dff):,}")
with c2: st.metric("🔴 High Risk",       f"{high_m}", delta=f"{high_m/tot_m*100:.1f}% of meters", delta_color="inverse")
with c3: st.metric("⚠️ Anomaly Rate",   f"{anom_r:.1f}%")
with c4: st.metric("💸 Revenue Loss",   f"₹{tot_l/100000:.2f} L")
with c5: st.metric("🟡 Medium Risk",    f"{med_m}")

st.markdown("---")

# ── TABS ─────────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "🗺️ Theft Map",
    "📊 Overview",
    "🤖 Model Compare",
    "🔍 Meter Drilldown",
    "🚨 Alerts",
])

# ════════════════════════════════════════════════════════
#  TAB 1 — THEFT MAP
# ════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Which Areas Have the Highest Theft?")
    st.caption("Area-level risk map. Red = high theft concentration.")

    area_risk = dff.groupby(["zone","area"]).agg(
        avg_risk       = ("final_risk_score","mean"),
        high_meters    = ("meter_id", lambda x: (dff.loc[x.index,"risk_band"]=="High").sum()),
        total_meters   = ("meter_id","nunique"),
        total_loss     = ("estimated_loss_rs","sum"),
        anomaly_count  = ("anomaly_label","sum"),
    ).reset_index()
    area_risk["risk_pct"] = (area_risk["high_meters"] / area_risk["total_meters"] * 100).round(1)

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = px.treemap(
            area_risk, path=["zone","area"],
            values="total_meters", color="avg_risk",
            color_continuous_scale=[[0,"#dcfce7"],[0.4,"#fef9c3"],[0.7,"#fee2e2"],[1,"#dc2626"]],
            hover_data={"avg_risk":":.3f","high_meters":True,"total_loss":":,.0f"},
            title="Area Risk Map — Size=No. of Meters | Color=Avg Risk Score",
        )
        # ✅ makes zoom/click still readable, avoids “empty” feel
        fig.update_traces(
            textinfo="label+percent parent",
            insidetextfont=dict(size=12),
            marker=dict(line=dict(width=1, color="#ffffff")),
            pathbar=dict(visible=True),
        )
        fig.update_layout(**LAYOUT, height=400, coloraxis_colorbar=dict(title="Risk"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Top 10 Highest Risk Areas**")
        top = area_risk.sort_values("avg_risk", ascending=False).head(10)
        for _, r in top.iterrows():
            color = "#dc2626" if r["avg_risk"]>0.60 else "#d97706" if r["avg_risk"]>0.30 else "#16a34a"
            st.markdown(f"""
            <div style="background:#fff;border:1px solid #e2e8f0;border-left:3px solid {color};
                        border-radius:6px;padding:8px 12px;margin:4px 0">
                <div style="font-weight:700;font-size:13px;color:#0f172a">{r['area']}</div>
                <div style="font-size:11px;color:#94a3b8">{r['zone']}</div>
                <div style="display:flex;justify-content:space-between;margin-top:4px">
                    <span style="font-size:12px;color:{color};font-weight:700">Risk: {r['avg_risk']:.3f}</span>
                    <span style="font-size:11px;color:#d97706">₹{r['total_loss']:,.0f}</span>
                </div>
            </div>""", unsafe_allow_html=True)

    # Feeder bars
    st.markdown("### Feeder Zone Risk")
    feeder_risk = dff.groupby("feeder_id").agg(
        avg_risk = ("final_risk_score","mean"),
        high     = ("meter_id", lambda x: (dff.loc[x.index,"risk_band"]=="High").sum()),
        meters   = ("meter_id","nunique"),
        loss     = ("estimated_loss_rs","sum"),
    ).reset_index().sort_values("avg_risk", ascending=False).head(20)

    fig2 = px.bar(feeder_risk, x="feeder_id", y="avg_risk",
                  color="loss", color_continuous_scale=["#fef9c3","#dc2626"],
                  title="Top 20 Feeders — Risk Score (color = revenue loss)")
    fig2.update_layout(**LAYOUT, height=300,
                       xaxis=dict(tickangle=-45, **GRID), yaxis=dict(**GRID))
    st.plotly_chart(fig2, use_container_width=True)

# ════════════════════════════════════════════════════════
#  TAB 2 — OVERVIEW
# ════════════════════════════════════════════════════════
with tab2:
    st.markdown("### System Overview")

    c1, c2, c3 = st.columns(3)

    with c1:
        rc = dff["risk_band"].value_counts().reset_index()
        rc.columns = ["band","count"]
        fig = go.Figure(go.Pie(
            labels=rc["band"], values=rc["count"], hole=0.55,
            marker_colors=[C.get(b,"#aaa") for b in rc["band"]],
            textfont=dict(size=11),
        ))
        fig.update_layout(**LAYOUT, height=260, title="Risk Distribution")
        fig.add_annotation(text=f"<b>{len(dff):,}</b>", x=0.5, y=0.5,
                           font=dict(size=13, color="#0f172a"), showarrow=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        at = dff[dff["anomaly_label"]==1].groupby("anomaly_type").size().reset_index(name="count")
        at = at[at["anomaly_type"]!="Normal"].sort_values("count")
        fig = px.bar(at, x="count", y="anomaly_type", orientation="h",
                     color="count", color_continuous_scale=["#fef2f2","#dc2626"],
                     title="Anomaly Types Detected")
        fig.update_layout(**LAYOUT, height=260, showlegend=False,
                          coloraxis_showscale=False,
                          xaxis=dict(**GRID), yaxis=dict(title="",**GRID))
        st.plotly_chart(fig, use_container_width=True)

    with c3:
        zs = dff.groupby("zone").agg(
            anomaly_rate=("anomaly_label","mean"),
            avg_risk=("final_risk_score","mean"),
        ).reset_index()
        fig = go.Figure()
        colors = [C["p1"], C["p2"], C["p3"]]
        for i, row in zs.iterrows():
            fig.add_trace(go.Bar(name=row["zone"].split()[0],
                                 x=["Anomaly Rate","Avg Risk"],
                                 y=[row["anomaly_rate"]*100, row["avg_risk"]*100],
                                 marker_color=colors[i % 3]))
        fig.update_layout(**LAYOUT, height=260, barmode="group",
                          title="Zone Comparison (%)",
                          yaxis=dict(**GRID), xaxis=dict(**GRID))
        st.plotly_chart(fig, use_container_width=True)

    daily = dff.groupby("reading_date").agg(
        anomalies    = ("anomaly_label","sum"),
        avg_risk     = ("final_risk_score","mean"),
        total_loss   = ("estimated_loss_rs","sum"),
    ).reset_index()

    fig = make_subplots(specs=[[{"secondary_y":True}]])
    fig.add_trace(go.Scatter(x=daily["reading_date"], y=daily["anomalies"],
                             name="Anomalies/day", line=dict(color="#dc2626",width=2),
                             fill="tozeroy", fillcolor="rgba(220,38,38,0.06)"),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=daily["reading_date"], y=daily["avg_risk"],
                             name="Avg Risk", line=dict(color="#d97706",width=1.5,dash="dot")),
                  secondary_y=True)
    fig.update_layout(**LAYOUT, height=280, title="Daily Anomaly Count & Avg Risk Score")
    fig.update_xaxes(**GRID)
    fig.update_yaxes(title_text="Anomalies", **GRID, secondary_y=False)
    fig.update_yaxes(title_text="Risk Score", **GRID, secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        samp = dff.sample(min(3000,len(dff)), random_state=42)
        fig = px.scatter(samp, x="kwh_consumed", y="power_factor",
                         color="risk_band",
                         color_discrete_map={"High":"#dc2626","Medium":"#d97706","Low":"#16a34a"},
                         opacity=0.4, title="kWh vs Power Factor by Risk Band",
                         hover_data=["meter_id","area"])
        fig.update_layout(**LAYOUT, height=300,
                          xaxis=dict(**GRID), yaxis=dict(**GRID))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        rf_df = pd.DataFrame({
            "Rule":  ["Spike (z>3)","Low PF","Voltage Anom","Low Use","Overload"],
            "Count": [dff["flag_spike"].sum(), dff["flag_low_pf"].sum(),
                      dff["flag_voltage"].sum(), dff["flag_low_use"].sum(),
                      dff["flag_overload"].sum()],
        }).sort_values("Count")
        fig = px.bar(rf_df, x="Count", y="Rule", orientation="h",
                     color="Count", color_continuous_scale=["#dbeafe","#1d4ed8"],
                     title="Rule Flag Counts")
        fig.update_layout(**LAYOUT, height=300, showlegend=False,
                          coloraxis_showscale=False,
                          xaxis=dict(**GRID), yaxis=dict(title="",**GRID))
        st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════
#  TAB 3 — MODEL COMPARE
# ════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Multi-Model Performance Comparison")
    st.caption("4 models trained on the same dataset. Best model used for final scoring.")

    if model_results:
        mdf = pd.DataFrame(list(model_results.values()))
        metrics = ["accuracy","precision","recall","f1_score","roc_auc"]
        labels  = ["Accuracy","Precision","Recall","F1","AUC"]
        model_colors = [C["p1"], C["p3"], "#d97706", "#7c3aed"]

        c1, c2 = st.columns([3, 2])

        with c1:
            fig = go.Figure()
            for i, (_, row) in enumerate(mdf.iterrows()):
                vals = [row[m] for m in metrics] + [row[metrics[0]]]
                fig.add_trace(go.Scatterpolar(
                    r=vals, theta=labels+[labels[0]],
                    name=row["model"],
                    line=dict(color=model_colors[i],width=2),
                    fill="toself", fillcolor=model_colors[i], opacity=0.15,
                ))
            fig.update_layout(
                **LAYOUT, height=380,
                polar=dict(
                    bgcolor="#f8fafc",
                    radialaxis=dict(visible=True, range=[0,1],
                                   gridcolor="#e2e8f0",
                                   tickfont=dict(size=8,color="#94a3b8")),
                    angularaxis=dict(gridcolor="#e2e8f0",
                                    tickfont=dict(size=10,color="#374151")),
                ),
                title="Model Performance Radar",
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("**Metrics Table**")
            disp = mdf[["model"]+metrics].copy()
            disp.columns = ["Model"]+labels
            for lbl in labels:
                disp[lbl] = disp[lbl].apply(lambda x: f"{x:.4f}")
            st.dataframe(disp.set_index("Model"), use_container_width=True, height=190)

            best = mdf.loc[mdf["f1_score"].idxmax()]
            st.success(f"🏆 **Best Model:** {best['model']}  \nF1={best['f1_score']:.4f} | AUC={best['roc_auc']:.4f}")

        fig = go.Figure()
        colors2 = ["#1d4ed8","#16a34a","#d97706","#dc2626","#7c3aed"]
        for m, lbl, color in zip(metrics, labels, colors2):
            fig.add_trace(go.Bar(name=lbl, x=mdf["model"], y=mdf[m],
                                 text=[f"{v:.3f}" for v in mdf[m]],
                                 textposition="outside", textfont=dict(size=9),
                                 marker_color=color))

        # ✅ FIX: no duplicate legend kwargs
        fig.update_layout(
            **LAYOUT, height=320, barmode="group",
            title="Side-by-Side Metric Comparison",
            yaxis=dict(range=[0,1.1],**GRID),
            xaxis=dict(**GRID),
        )
        fig.update_layout(legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### How Each Model Works")
        c1,c2,c3,c4 = st.columns(4)
        info = [
            ("Isolation Forest","Unsupervised","#1d4ed8","No labels needed. Builds random trees and isolates anomalies — easy-to-isolate points are outliers. Best for unknown theft patterns."),
            ("Random Forest","Supervised","#16a34a","200 decision trees vote on each reading. Very high accuracy when trained on labelled theft data."),
            ("Gradient Boosting","Supervised","#d97706","Trees built sequentially — each corrects errors of the last. Achieves highest F1. Catches subtle multi-feature patterns."),
            ("Logistic Regression","Baseline","#7c3aed","Simple and fast. Acts as baseline. Shows direction of each feature's impact. Good for quick real-time scoring."),
        ]
        for col, (name, typ, color, desc) in zip([c1,c2,c3,c4], info):
            with col:
                st.markdown(f"""
                <div style="background:#fff;border:1px solid #e2e8f0;
                            border-top:3px solid {color};border-radius:8px;
                            padding:14px;height:180px">
                    <div style="font-size:9px;color:{color};letter-spacing:1px">{typ.upper()}</div>
                    <div style="font-weight:700;font-size:13px;margin:6px 0">{name}</div>
                    <div style="font-size:10px;color:#64748b;line-height:1.5">{desc}</div>
                </div>""", unsafe_allow_html=True)
    else:
        st.warning("Run `python train_models.py` first to generate model results.")

# ════════════════════════════════════════════════════════
#  TAB 4 — METER DRILLDOWN
# ════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Individual Meter Analysis")

    c1, c2 = st.columns([1,3])
    with c1:
        risk_meters = dff[dff["risk_band"].isin(["High","Medium"])].sort_values("final_risk_score", ascending=False)
        opts = risk_meters["meter_id"].unique()[:100].tolist()
        if not opts:
            opts = dff["meter_id"].unique()[:50].tolist()
        sel_m = st.selectbox("Select Meter", opts)

    md = dff[dff["meter_id"]==sel_m].sort_values("reading_date")
    if len(md):
        mi = md.iloc[-1]
        rb = str(mi.get("risk_band","Low"))
        rc = {"High":"#dc2626","Medium":"#d97706","Low":"#16a34a"}.get(rb,"#94a3b8")
        rc_bg = {"High":"#fef2f2","Medium":"#fffbeb","Low":"#f0fdf4"}.get(rb,"#f8fafc")

        with c2:
            st.markdown(f"""
            <div style="background:{rc_bg};border:1px solid {rc}44;border-radius:8px;
                        padding:12px 16px;display:flex;justify-content:space-between">
                <div>
                    <span style="font-size:16px;font-weight:800;color:#0f172a">{sel_m}</span>
                    <span style="font-size:12px;color:#94a3b8;margin-left:10px">{mi.get('area','')}, {mi.get('zone','')} | {mi.get('feeder_id','')}</span>
                </div>
                <div style="text-align:right">
                    <span style="background:{rc};color:#fff;border-radius:5px;
                                 padding:4px 12px;font-size:12px;font-weight:700">{rb} RISK</span>
                    <div style="font-size:11px;color:#94a3b8;margin-top:4px">Score: {mi.get('final_risk_score',0):.4f}</div>
                </div>
            </div>""", unsafe_allow_html=True)

        c1,c2,c3,c4,c5 = st.columns(5)
        with c1: st.metric("Avg kWh/day",   f"{md['kwh_consumed'].mean():.2f}")
        with c2: st.metric("Power Factor",  f"{md['power_factor'].mean():.3f}")
        with c3: st.metric("Avg Voltage",   f"{md['voltage_volts'].mean():.1f}V")
        with c4: st.metric("Total Loss",    f"₹{md['estimated_loss_rs'].sum():.2f}")
        with c5: st.metric("Anomaly Days",  f"{int(md['anomaly_label'].sum())}/{len(md)}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=md["reading_date"], y=md["mean_kwh"],
                                 name="Expected kWh",
                                 line=dict(color=C["p1"],width=1.5,dash="dot"), opacity=0.7))
        fig.add_trace(go.Scatter(x=md["reading_date"], y=md["kwh_consumed"],
                                 name="Actual kWh", line=dict(color=C["p3"],width=2),
                                 fill="tozeroy", fillcolor="rgba(5,150,105,0.05)"))
        anom_rows = md[md["anomaly_label"]==1]
        if len(anom_rows):
            fig.add_trace(go.Scatter(x=anom_rows["reading_date"], y=anom_rows["kwh_consumed"],
                                     mode="markers", name="⚠ Anomaly",
                                     marker=dict(color="#dc2626",size=10,symbol="x",
                                                 line=dict(color="#dc2626",width=2))))
        fig.update_layout(**LAYOUT, height=300,
                          title=f"Consumption Pattern — {sel_m}",
                          xaxis=dict(**GRID), yaxis=dict(title="kWh",**GRID))
        st.plotly_chart(fig, use_container_width=True)

        if len(anom_rows):
            st.markdown(f"**⚠️ {len(anom_rows)} Anomaly Days Detected**")
            show = [c for c in ["reading_date","kwh_consumed","mean_kwh","z_score",
                                 "power_factor","voltage_volts","anomaly_type",
                                 "risk_band","final_risk_score"] if c in anom_rows.columns]
            st.dataframe(anom_rows[show], use_container_width=True, height=200)

# ════════════════════════════════════════════════════════
#  TAB 5 — ALERTS
# ════════════════════════════════════════════════════════
with tab5:
    st.markdown("### 🚨 High Risk Alerts — Inspection Required")
    high_df = dff[dff["risk_band"]=="High"].copy()

    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div style="background:#fef2f2;border:1px solid #fca5a544;border-top:3px solid #dc2626;border-radius:10px;padding:16px">
            <div style="font-size:10px;color:#dc2626;font-weight:600;letter-spacing:1px">HIGH RISK METERS</div>
            <div style="font-size:28px;font-weight:800;color:#dc2626;margin:8px 0">{high_df['meter_id'].nunique()}</div>
            <div style="font-size:11px;color:#94a3b8">Immediate inspection needed</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div style="background:#fffbeb;border:1px solid #fcd34d44;border-top:3px solid #d97706;border-radius:10px;padding:16px">
            <div style="font-size:10px;color:#d97706;font-weight:600;letter-spacing:1px">TOTAL REVENUE LOSS</div>
            <div style="font-size:28px;font-weight:800;color:#d97706;margin:8px 0">₹{high_df['estimated_loss_rs'].sum():,.0f}</div>
            <div style="font-size:11px;color:#94a3b8">Recoverable with action</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        top_zone = high_df.groupby("zone")["meter_id"].nunique().idxmax() if len(high_df) else "N/A"
        st.markdown(f"""
        <div style="background:#f5f3ff;border:1px solid #c4b5fd44;border-top:3px solid #7c3aed;border-radius:10px;padding:16px">
            <div style="font-size:10px;color:#7c3aed;font-weight:600;letter-spacing:1px">HIGHEST RISK ZONE</div>
            <div style="font-size:20px;font-weight:800;color:#7c3aed;margin:8px 0">{top_zone}</div>
            <div style="font-size:11px;color:#94a3b8">Most meters flagged</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    alert_df = (high_df.groupby("meter_id").agg(
        zone           = ("zone","first"),
        area           = ("area","first"),
        feeder_id      = ("feeder_id","first"),
        connection_type= ("connection_type","first"),
        avg_risk       = ("final_risk_score","mean"),
        total_loss     = ("estimated_loss_rs","sum"),
        anomaly_days   = ("anomaly_label","sum"),
        total_days     = ("reading_date","count"),
        anomaly_type   = ("anomaly_type", lambda x: x[x!="Normal"].mode()[0] if (x!="Normal").any() else "Unknown"),
    ).reset_index().sort_values("avg_risk", ascending=False))
    alert_df["anomaly_freq_pct"] = (alert_df["anomaly_days"]/alert_df["total_days"]*100).round(1)
    alert_df["avg_risk"]   = alert_df["avg_risk"].round(4)
    alert_df["total_loss"] = alert_df["total_loss"].round(2)

    st.markdown("#### 🔴 Top 10 Critical Meters")
    for _, row in alert_df.head(10).iterrows():
        c1,c2,c3,c4 = st.columns([3,2,2,1])
        with c1:
            st.markdown(f"""
            <div style="background:#fff;border:1px solid #fca5a544;border-left:3px solid #dc2626;
                        border-radius:7px;padding:10px 14px">
                <div style="font-weight:700;font-size:13px">{row['meter_id']}</div>
                <div style="font-size:11px;color:#94a3b8">{row['area']}, {row['zone']} | {row['feeder_id']}</div>
                <span style="font-size:10px;background:#fef2f2;color:#dc2626;
                             border-radius:3px;padding:1px 7px;font-weight:600">{row.get('anomaly_type','Unknown')}</span>
            </div>""", unsafe_allow_html=True)
        with c2: st.metric("Risk Score",   f"{row['avg_risk']:.4f}")
        with c3: st.metric("Revenue Loss", f"₹{row['total_loss']:,.2f}")
        with c4: st.metric("Anom %",       f"{row['anomaly_freq_pct']}%")

    st.markdown("---")
    st.markdown("#### 📋 Full Alert Table")
    show_cols = ["meter_id","zone","area","feeder_id","anomaly_type",
                 "avg_risk","total_loss","anomaly_freq_pct"]
    avail = [c for c in show_cols if c in alert_df.columns]
    st.dataframe(alert_df[avail], use_container_width=True, height=380)

    c1,c2 = st.columns(2)
    with c1:
        st.download_button("📥 Download Alert List",
                           alert_df.to_csv(index=False).encode(),
                           "voltx2_high_risk_alerts.csv", "text/csv")
    with c2:
        st.download_button("📥 Download Full Dataset",
                           dff.to_csv(index=False).encode(),
                           "voltx2_full_data.csv", "text/csv")

st.markdown("---")
st.caption("VOLTX 2.0 · PySpark + Isolation Forest + Random Forest + Gradient Boosting + Logistic Regression · TNEB Tamil Nadu")