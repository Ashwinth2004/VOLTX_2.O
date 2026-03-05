"""
VOLTX 2.0 — Streamlit Dashboard (Dark Theme)
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

# ── DARK THEME CSS ────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"], .stApp {
    background: #0d1117 !important;
    color: #e6edf3 !important;
    font-family: 'Inter', system-ui, sans-serif !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #161b22 !important;
    border-right: 1px solid #21262d !important;
}
section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
section[data-testid="stSidebar"] .stSelectbox label { color: #8b949e !important; }

/* Metrics */
[data-testid="stMetric"] {
    background: #161b22 !important;
    border: 1px solid #21262d !important;
    border-radius: 10px !important;
    padding: 16px 18px !important;
}
[data-testid="stMetricLabel"] {
    font-size: 10px !important; color: #8b949e !important;
    text-transform: uppercase !important; letter-spacing: 1px !important;
}
[data-testid="stMetricValue"] {
    font-size: 24px !important; font-weight: 800 !important;
    color: #e6edf3 !important;
}
[data-testid="stMetricDelta"] { font-size: 11px !important; }

/* Tabs */
[data-testid="stTabs"] { background: transparent !important; }
[data-testid="stTabs"] button {
    font-size: 12px !important; font-weight: 500 !important;
    color: #8b949e !important; background: transparent !important;
    border: none !important; letter-spacing: 0.3px !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #58a6ff !important;
    border-bottom: 2px solid #58a6ff !important;
    background: transparent !important;
}
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid #21262d !important;
}

/* Headings */
h1 { font-size: 24px !important; font-weight: 800 !important; color: #e6edf3 !important; }
h2 { font-size: 17px !important; font-weight: 700 !important; color: #e6edf3 !important; }
h3 { font-size: 14px !important; font-weight: 600 !important; color: #c9d1d9 !important; }

/* Selectbox */
.stSelectbox > div > div {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #c9d1d9 !important;
    border-radius: 7px !important;
}

/* Buttons */
.stButton > button {
    background: #1f6feb !important; color: #fff !important;
    border: none !important; border-radius: 7px !important;
    font-weight: 600 !important; font-size: 12px !important;
    padding: 8px 16px !important;
}
.stButton > button:hover { background: #388bfd !important; }

/* Download button */
.stDownloadButton > button {
    background: #21262d !important; color: #58a6ff !important;
    border: 1px solid #30363d !important; border-radius: 7px !important;
    font-weight: 600 !important; font-size: 12px !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    background: #161b22 !important;
    border: 1px solid #21262d !important;
    border-radius: 8px !important;
}

/* Caption */
.stCaption { color: #8b949e !important; font-size: 12px !important; }

/* Divider */
hr { border-color: #21262d !important; }

/* Success / warning boxes */
[data-testid="stAlert"] {
    background: #1c2128 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #c9d1d9 !important;
}

/* Date input */
.stDateInput input {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #c9d1d9 !important;
    border-radius: 7px !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #484f58; }
</style>
""", unsafe_allow_html=True)

# ── PLOTLY DARK THEME ────────────────────────────────────
LAYOUT = dict(
    paper_bgcolor="#161b22",
    plot_bgcolor="#161b22",
    font=dict(family="Inter, system-ui", color="#8b949e", size=11),
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(bgcolor="#1c2128", bordercolor="#30363d", borderwidth=1,
                font=dict(color="#c9d1d9")),
)
GRID = dict(gridcolor="#21262d", zerolinecolor="#30363d")
AXIS = dict(color="#8b949e", tickfont=dict(color="#8b949e"),
            title_font=dict(color="#8b949e"))

C = {
    "High":   "#f85149",
    "Medium": "#e3b341",
    "Low":    "#3fb950",
    "p1":     "#58a6ff",
    "p2":     "#bc8cff",
    "p3":     "#3fb950",
    "p4":     "#e3b341",
}

# ── REVENUE LOSS FIX FUNCTION ─────────────────────────────
def recalculate_loss(df):
    """
    Fix: Revenue loss calculation per anomaly type.

    - Bypass / Tamper / Reverse / Night Theft / Illegal:
        Consumer uses real power but meter records less.
        Loss = (mean_kwh - kwh_consumed) × tariff  [positive when kwh < mean]

    - Abnormal Spike:
        Consumer uses FAR MORE than sanctioned load.
        Loss = (kwh_consumed - mean_kwh) × tariff
        (unbilled excess because meter is manipulated to show spike then reset)

    - Low Power Factor:
        Utility loses due to reactive power —
        Loss = reactive_power_kvar × tariff × 0.3  (approx penalty)
    """
    df = df.copy()

    spike_mask  = df["anomaly_type"] == "Abnormal Consumption Spike"
    lpf_mask    = df["anomaly_type"] == "Low Power Factor Loss"
    under_mask  = df["anomaly_label"] == 1  # all other theft types

    loss = np.zeros(len(df))

    # Under-recording theft (bypass, tamper, reverse, night, illegal)
    under_only = under_mask & ~spike_mask & ~lpf_mask
    loss = np.where(
        under_only,
        np.maximum(0, (df["mean_kwh"] - df["kwh_consumed"]) * df["tariff_rs_per_kwh"]),
        loss
    )

    # Spike anomaly — excess consumption beyond mean
    loss = np.where(
        spike_mask,
        np.maximum(0, (df["kwh_consumed"] - df["mean_kwh"]) * df["tariff_rs_per_kwh"]),
        loss
    )

    # Low PF — reactive power penalty
    if "reactive_power_kvar" in df.columns:
        loss = np.where(
            lpf_mask,
            np.maximum(0, df["reactive_power_kvar"] * df["tariff_rs_per_kwh"] * 0.30),
            loss
        )
    else:
        loss = np.where(
            lpf_mask,
            np.maximum(0, df["mean_kwh"] * df["tariff_rs_per_kwh"] * 0.20),
            loss
        )

    df["estimated_loss_rs"] = loss.round(2)
    return df


# ── LOAD DATA ────────────────────────────────────────────
@st.cache_data
def load_data():
    for p in ["data/tneb_scored_results.csv", "tneb_scored_results.csv"]:
        if os.path.exists(p):
            df = pd.read_csv(p)
            df["reading_date"] = pd.to_datetime(df["reading_date"])
            df["risk_band"] = df["risk_band"].astype(str)
            df = recalculate_loss(df)   # ← apply revenue fix
            return df
    return None

@st.cache_data
def load_model_results():
    for p in ["data/model_results.json", "model_results.json"]:
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    return None

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
            n = df[df["zone"] == z]["meter_id"].nunique()
            st.markdown(f"- {z}: `{n}` meters")
        st.markdown("---")
        st.markdown("**🔧 Anomaly Key**")
        st.markdown("""
- 🔴 **Bypass** — meter skipped
- 🟠 **Tampered** — slowed by magnet
- 🟡 **Illegal** — hooked before meter
- 🔵 **Reverse** — meter runs backwards
- ⚡ **Spike** — excess unbilled load
- 📉 **Low PF** — reactive loss
- 🌙 **Night Theft** — 2–5AM pattern
        """)

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
st.caption("TNEB Smart Meter Analytics | Chennai Urban · Chennai Suburban · Tiruvallur District")
st.markdown("---")

if df is None:
    st.error("⚠️  Data not found. Run the pipeline first:")
    st.code("python generate_csv.py\npython train_models.py", language="bash")
    st.stop()

# ── KPIs ─────────────────────────────────────────────────
high_m = dff[dff["risk_band"] == "High"]["meter_id"].nunique()
tot_m  = dff["meter_id"].nunique()
anom_r = dff["anomaly_label"].mean() * 100
tot_l  = dff["estimated_loss_rs"].sum()
med_m  = dff[dff["risk_band"] == "Medium"]["meter_id"].nunique()

c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.metric("📋 Total Readings", f"{len(dff):,}")
with c2: st.metric("🔴 High Risk",      f"{high_m}",
                   delta=f"{high_m/max(tot_m,1)*100:.1f}% of meters",
                   delta_color="inverse")
with c3: st.metric("⚠️ Anomaly Rate",  f"{anom_r:.1f}%")
with c4: st.metric("💸 Revenue Loss",  f"₹{tot_l/100000:.2f} L")
with c5: st.metric("🟡 Medium Risk",   f"{med_m}")

st.markdown("---")

# ── TABS ─────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️  Theft Map",
    "📊  Overview",
    "🤖  Model Compare",
    "🔍  Meter Drilldown",
    "🚨  Alerts",
])

# ═══════════════════════════════════════════════════════════
#  TAB 1 — THEFT MAP
# ═══════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Which Areas & Feeders Have the Highest Theft?")
    st.caption("Size = number of meters | Color = average risk score. Click a zone to drill down.")

    area_risk = dff.groupby(["zone", "area"]).agg(
        avg_risk      = ("final_risk_score", "mean"),
        high_meters   = ("meter_id", lambda x: (dff.loc[x.index, "risk_band"] == "High").sum()),
        total_meters  = ("meter_id", "nunique"),
        total_loss    = ("estimated_loss_rs", "sum"),
        anomaly_count = ("anomaly_label", "sum"),
    ).reset_index()
    area_risk["risk_pct"] = (area_risk["high_meters"] / area_risk["total_meters"] * 100).round(1)

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = px.treemap(
            area_risk, path=["zone", "area"],
            values="total_meters", color="avg_risk",
            color_continuous_scale=[[0, "#0d4429"], [0.35, "#1a7f37"],
                                     [0.65, "#9e6a03"], [1, "#f85149"]],
            hover_data={"avg_risk": ":.3f", "high_meters": True,
                        "total_loss": ":,.0f", "risk_pct": ":.1f"},
            title="Area Risk Map",
        )
        fig.update_traces(
            textinfo="label+percent parent",
            insidetextfont=dict(size=12, color="#e6edf3"),
            marker=dict(line=dict(width=1, color="#0d1117")),
            pathbar=dict(visible=True),
        )
        fig.update_layout(**LAYOUT, height=420,
                          coloraxis_colorbar=dict(
                              title="Risk", tickfont=dict(color="#8b949e"),
                              title_font=dict(color="#8b949e")))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**🔴 Top Risk Areas**")
        top = area_risk.sort_values("avg_risk", ascending=False).head(12)
        for _, r in top.iterrows():
            color = C["High"] if r["avg_risk"] > 0.60 else C["Medium"] if r["avg_risk"] > 0.30 else C["Low"]
            st.markdown(f"""
            <div style="background:#161b22;border:1px solid #21262d;border-left:3px solid {color};
                        border-radius:6px;padding:8px 12px;margin:4px 0">
                <div style="font-weight:700;font-size:13px;color:#e6edf3">{r['area']}</div>
                <div style="font-size:10px;color:#8b949e">{r['zone']}</div>
                <div style="display:flex;justify-content:space-between;margin-top:5px">
                    <span style="font-size:12px;color:{color};font-weight:700;font-family:JetBrains Mono,monospace">
                        {r['avg_risk']:.3f}</span>
                    <span style="font-size:11px;color:{C['Medium']}">
                        ₹{r['total_loss']:,.0f} loss</span>
                </div>
                <div style="margin-top:4px;height:3px;background:#21262d;border-radius:2px">
                    <div style="width:{min(r['avg_risk']*100,100):.0f}%;height:100%;
                                background:{color};border-radius:2px"></div>
                </div>
            </div>""", unsafe_allow_html=True)

    # Feeder risk
    st.markdown("### 🔌 Feeder Zone Risk Analysis")
    c1, c2 = st.columns(2)

    feeder_risk = dff.groupby("feeder_id").agg(
        avg_risk = ("final_risk_score", "mean"),
        high     = ("meter_id", lambda x: (dff.loc[x.index, "risk_band"] == "High").sum()),
        meters   = ("meter_id", "nunique"),
        loss     = ("estimated_loss_rs", "sum"),
    ).reset_index().sort_values("avg_risk", ascending=False).head(20)

    with c1:
        fig2 = px.bar(
            feeder_risk, x="feeder_id", y="avg_risk",
            color="loss", color_continuous_scale=["#1a7f37", "#9e6a03", "#f85149"],
            title="Top 20 Feeders — Avg Risk Score (color = loss)",
        )
        fig2.update_layout(**LAYOUT, height=320,
                           xaxis=dict(tickangle=-45, **GRID, **AXIS),
                           yaxis=dict(title="Risk Score", **GRID, **AXIS),
                           coloraxis_colorbar=dict(title="Loss ₹",
                                                    tickfont=dict(color="#8b949e")))
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        fig3 = px.scatter(
            feeder_risk, x="meters", y="avg_risk",
            size="loss", color="high",
            color_continuous_scale=["#1f6feb", "#f85149"],
            hover_name="feeder_id",
            title="Feeder: Meters vs Risk (bubble = loss, color = high-risk count)",
            labels={"meters": "No. of Meters", "avg_risk": "Avg Risk Score",
                    "high": "High-Risk Count"},
        )
        fig3.update_layout(**LAYOUT, height=320,
                           xaxis=dict(**GRID, **AXIS), yaxis=dict(**GRID, **AXIS),
                           coloraxis_colorbar=dict(tickfont=dict(color="#8b949e")))
        st.plotly_chart(fig3, use_container_width=True)

# ═══════════════════════════════════════════════════════════
#  TAB 2 — OVERVIEW
# ═══════════════════════════════════════════════════════════
with tab2:
    st.markdown("### System Overview")

    c1, c2, c3 = st.columns(3)

    with c1:
        rc = dff["risk_band"].value_counts().reset_index()
        rc.columns = ["band", "count"]
        fig = go.Figure(go.Pie(
            labels=rc["band"], values=rc["count"], hole=0.58,
            marker_colors=[C.get(b, "#8b949e") for b in rc["band"]],
            textfont=dict(size=11, color="#e6edf3"),
            textinfo="label+percent",
        ))
        fig.update_layout(**LAYOUT, height=280, title="Risk Distribution",
                          title_font=dict(color="#c9d1d9"))
        fig.add_annotation(text=f"<b style='color:#e6edf3'>{len(dff):,}</b><br><span style='color:#8b949e;font-size:10px'>Readings</span>",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=13, color="#e6edf3"))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        at = dff[dff["anomaly_label"] == 1].groupby("anomaly_type").size().reset_index(name="count")
        at = at[at["anomaly_type"] != "Normal"].sort_values("count")
        fig = px.bar(at, x="count", y="anomaly_type", orientation="h",
                     color="count",
                     color_continuous_scale=["#9e6a03", "#f85149"],
                     title="Anomaly Types Detected")
        fig.update_layout(**LAYOUT, height=280, showlegend=False,
                          coloraxis_showscale=False, title_font=dict(color="#c9d1d9"),
                          xaxis=dict(**GRID, **AXIS), yaxis=dict(title="", **AXIS))
        st.plotly_chart(fig, use_container_width=True)

    with c3:
        zs = dff.groupby("zone").agg(
            anomaly_rate = ("anomaly_label", "mean"),
            avg_risk     = ("final_risk_score", "mean"),
        ).reset_index()
        fig = go.Figure()
        colors = [C["p1"], C["p2"], C["p3"]]
        for i, row in zs.iterrows():
            fig.add_trace(go.Bar(
                name=row["zone"].split()[0],
                x=["Anomaly Rate %", "Avg Risk %"],
                y=[row["anomaly_rate"] * 100, row["avg_risk"] * 100],
                marker_color=colors[i % 3],
            ))
        fig.update_layout(**LAYOUT, height=280, barmode="group",
                          title="Zone Comparison", title_font=dict(color="#c9d1d9"),
                          yaxis=dict(title="%", **GRID, **AXIS),
                          xaxis=dict(**GRID, **AXIS))
        st.plotly_chart(fig, use_container_width=True)

    # Daily trend
    daily = dff.groupby("reading_date").agg(
        anomalies  = ("anomaly_label", "sum"),
        avg_risk   = ("final_risk_score", "mean"),
        total_loss = ("estimated_loss_rs", "sum"),
    ).reset_index()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=daily["reading_date"], y=daily["anomalies"],
        name="Anomalies/day", line=dict(color=C["High"], width=2),
        fill="tozeroy", fillcolor="rgba(248,81,73,0.08)"),
        secondary_y=False)
    fig.add_trace(go.Scatter(
        x=daily["reading_date"], y=daily["avg_risk"],
        name="Avg Risk Score", line=dict(color=C["Medium"], width=1.5, dash="dot")),
        secondary_y=True)
    fig.add_trace(go.Bar(
        x=daily["reading_date"], y=daily["total_loss"],
        name="Daily Loss ₹", marker_color="rgba(88,166,255,0.15)",
        marker_line_color=C["p1"], marker_line_width=0.5),
        secondary_y=False)
    fig.update_layout(**LAYOUT, height=300,
                      title="Daily Anomaly Count, Risk Score & Revenue Loss",
                      title_font=dict(color="#c9d1d9"))
    fig.update_xaxes(**GRID, **AXIS)
    fig.update_yaxes(title_text="Anomalies / Loss ₹", **GRID, **AXIS, secondary_y=False)
    fig.update_yaxes(title_text="Risk Score", **GRID, **AXIS, secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        samp = dff.sample(min(3000, len(dff)), random_state=42)
        fig = px.scatter(
            samp, x="kwh_consumed", y="power_factor",
            color="risk_band",
            color_discrete_map={k: C[k] for k in C if k in ["High","Medium","Low"]},
            opacity=0.5, title="kWh Consumed vs Power Factor",
            hover_data=["meter_id", "area", "anomaly_type"],
        )
        fig.update_layout(**LAYOUT, height=310,
                          title_font=dict(color="#c9d1d9"),
                          xaxis=dict(title="kWh/day", **GRID, **AXIS),
                          yaxis=dict(title="Power Factor", **GRID, **AXIS))
        fig.update_traces(marker=dict(size=5))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Loss by anomaly type — using fixed calculation
        loss_by_type = dff[dff["anomaly_label"] == 1].groupby("anomaly_type").agg(
            total_loss = ("estimated_loss_rs", "sum"),
            count      = ("meter_id", "count"),
        ).reset_index().sort_values("total_loss", ascending=True)

        fig = px.bar(
            loss_by_type, x="total_loss", y="anomaly_type", orientation="h",
            color="count",
            color_continuous_scale=["#1f6feb", "#bc8cff"],
            title="Revenue Loss by Anomaly Type (₹)",
            text=loss_by_type["total_loss"].apply(lambda x: f"₹{x:,.0f}"),
        )
        fig.update_traces(textposition="outside", textfont=dict(size=9, color="#8b949e"))
        fig.update_layout(**LAYOUT, height=310, showlegend=False,
                          coloraxis_showscale=False,
                          title_font=dict(color="#c9d1d9"),
                          xaxis=dict(title="Total Loss ₹", **GRID, **AXIS),
                          yaxis=dict(title="", **AXIS))
        st.plotly_chart(fig, use_container_width=True)

    # Rule flags + connection type
    c1, c2 = st.columns(2)
    with c1:
        rf_df = pd.DataFrame({
            "Rule":  ["Spike (z>3)", "Low PF", "Voltage Anom", "Low Use", "Overload"],
            "Count": [dff["flag_spike"].sum(), dff["flag_low_pf"].sum(),
                      dff["flag_voltage"].sum(), dff["flag_low_use"].sum(),
                      dff["flag_overload"].sum()],
        }).sort_values("Count")
        fig = px.bar(rf_df, x="Count", y="Rule", orientation="h",
                     color="Count",
                     color_continuous_scale=["#1f6feb", "#58a6ff"],
                     title="Rule-Based Flag Counts")
        fig.update_layout(**LAYOUT, height=280, showlegend=False,
                          coloraxis_showscale=False,
                          title_font=dict(color="#c9d1d9"),
                          xaxis=dict(**GRID, **AXIS),
                          yaxis=dict(title="", **AXIS))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        ct = dff[dff["anomaly_label"] == 1].groupby("connection_type").agg(
            count = ("meter_id", "count"),
            loss  = ("estimated_loss_rs", "sum"),
        ).reset_index().sort_values("loss", ascending=True)
        fig = px.bar(ct, x="loss", y="connection_type", orientation="h",
                     color="count",
                     color_continuous_scale=["#3fb950", "#e3b341"],
                     title="Revenue Loss by Connection Type (₹)",
                     text=ct["loss"].apply(lambda x: f"₹{x:,.0f}"))
        fig.update_traces(textposition="outside", textfont=dict(size=9, color="#8b949e"))
        fig.update_layout(**LAYOUT, height=280, showlegend=False,
                          coloraxis_showscale=False,
                          title_font=dict(color="#c9d1d9"),
                          xaxis=dict(title="Total Loss ₹", **GRID, **AXIS),
                          yaxis=dict(title="", **AXIS))
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════
#  TAB 3 — MODEL COMPARE
# ═══════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Multi-Model Performance Comparison")
    st.caption("4 models trained on the same dataset. Ensemble of all 3 supervised models used for final risk score.")

    if model_results:
        mdf = pd.DataFrame(list(model_results.values()))
        metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
        labels  = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
        model_colors = [C["p1"], C["p3"], C["p4"], C["p2"]]

        c1, c2 = st.columns([3, 2])

        with c1:
            fig = go.Figure()
            for i, (_, row) in enumerate(mdf.iterrows()):
                vals = [row[m] for m in metrics] + [row[metrics[0]]]
                fig.add_trace(go.Scatterpolar(
                    r=vals, theta=labels + [labels[0]],
                    name=row["model"],
                    line=dict(color=model_colors[i], width=2),
                    fill="toself", fillcolor=model_colors[i], opacity=0.12,
                ))
            fig.update_layout(
                **LAYOUT, height=400,
                polar=dict(
                    bgcolor="#1c2128",
                    radialaxis=dict(visible=True, range=[0, 1],
                                    gridcolor="#30363d",
                                    tickfont=dict(size=8, color="#8b949e"),
                                    linecolor="#30363d"),
                    angularaxis=dict(gridcolor="#30363d",
                                     tickfont=dict(size=10, color="#c9d1d9"),
                                     linecolor="#30363d"),
                ),
                title="Model Performance Radar", title_font=dict(color="#c9d1d9"),
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("**📊 Metrics Table**")
            disp = mdf[["model"] + metrics].copy()
            disp.columns = ["Model"] + labels
            for lbl in labels:
                disp[lbl] = disp[lbl].apply(lambda x: f"{x:.4f}")
            st.dataframe(disp.set_index("Model"), use_container_width=True, height=200)

            best = mdf.loc[mdf["f1_score"].idxmax()]
            st.success(
                f"🏆 **Best Model: {best['model']}**\n\n"
                f"F1 = {best['f1_score']:.4f} · AUC = {best['roc_auc']:.4f}"
            )

            # Score summary box
            st.markdown("""
            <div style="background:#1c2128;border:1px solid #30363d;border-radius:8px;padding:12px 14px;margin-top:8px">
                <div style="font-size:10px;color:#8b949e;letter-spacing:1px;margin-bottom:6px">ENSEMBLE SCORING</div>
                <div style="font-size:12px;color:#c9d1d9">
                    Final Risk = <span style="color:#58a6ff">35% Isolation Forest</span><br>
                    + <span style="color:#3fb950">35% Gradient Boosting</span><br>
                    + <span style="color:#e3b341">30% Random Forest</span>
                </div>
            </div>""", unsafe_allow_html=True)

        # Bar comparison
        fig = go.Figure()
        bar_colors = [C["p1"], C["p3"], C["p4"], C["High"], C["p2"]]
        for m, lbl, color in zip(metrics, labels, bar_colors):
            fig.add_trace(go.Bar(
                name=lbl, x=mdf["model"], y=mdf[m],
                text=[f"{v:.3f}" for v in mdf[m]],
                textposition="outside",
                textfont=dict(size=9, color="#8b949e"),
                marker_color=color,
            ))
        fig.update_layout(**LAYOUT, height=340, barmode="group",
                          title="Side-by-Side Metric Comparison",
                          title_font=dict(color="#c9d1d9"),
                          yaxis=dict(range=[0, 1.15], **GRID, **AXIS),
                          xaxis=dict(**GRID, **AXIS))
        fig.update_layout(legend=dict(orientation="h", y=1.12,
                                       font=dict(color="#c9d1d9", size=10)))
        st.plotly_chart(fig, use_container_width=True)

        # How each works
        st.markdown("### How Each Model Works")
        c1, c2, c3, c4 = st.columns(4)
        info = [
            ("Isolation Forest", "Unsupervised", C["p1"],
             "No labels needed. Builds random trees and isolates anomalies — easy-to-isolate points are flagged. Best for unknown theft patterns."),
            ("Random Forest", "Supervised", C["p3"],
             "200 decision trees vote on each reading. Highly accurate when trained on labelled theft data. Also gives feature importances."),
            ("Gradient Boosting", "Supervised", C["p4"],
             "Trees built sequentially — each corrects errors of the last. Achieves highest F1 score. Catches subtle multi-feature patterns."),
            ("Logistic Regression", "Baseline", C["p2"],
             "Simple and fast baseline. Shows direction and magnitude of each feature's contribution. Interpretable by non-technical users."),
        ]
        for col, (name, typ, color, desc) in zip([c1, c2, c3, c4], info):
            with col:
                st.markdown(f"""
                <div style="background:#161b22;border:1px solid #21262d;
                            border-top:3px solid {color};border-radius:8px;
                            padding:14px;min-height:190px">
                    <div style="font-size:9px;color:{color};letter-spacing:1.5px;font-weight:600">{typ.upper()}</div>
                    <div style="font-weight:700;font-size:13px;margin:8px 0;color:#e6edf3">{name}</div>
                    <div style="font-size:11px;color:#8b949e;line-height:1.6">{desc}</div>
                </div>""", unsafe_allow_html=True)

    else:
        st.warning("⚠️ Run `python train_models.py` first to generate model results.")

# ═══════════════════════════════════════════════════════════
#  TAB 4 — METER DRILLDOWN
# ═══════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Individual Meter Analysis")

    c1, c2 = st.columns([1, 3])
    with c1:
        risk_meters = dff[dff["risk_band"].isin(["High", "Medium"])].sort_values(
            "final_risk_score", ascending=False)
        opts = risk_meters["meter_id"].unique()[:150].tolist()
        if not opts:
            opts = dff["meter_id"].unique()[:50].tolist()
        sel_m = st.selectbox("Select Meter ID", opts)

    md = dff[dff["meter_id"] == sel_m].sort_values("reading_date")

    if len(md):
        mi  = md.iloc[-1]
        rb  = str(mi.get("risk_band", "Low"))
        rc  = {"High": C["High"], "Medium": C["Medium"], "Low": C["Low"]}.get(rb, "#8b949e")
        rbg = {"High": "rgba(248,81,73,0.08)", "Medium": "rgba(227,179,65,0.08)",
               "Low":  "rgba(63,185,80,0.08)"}.get(rb, "rgba(255,255,255,0.04)")

        with c2:
            st.markdown(f"""
            <div style="background:{rbg};border:1px solid {rc}44;border-radius:8px;
                        padding:12px 16px;display:flex;justify-content:space-between;align-items:center">
                <div>
                    <span style="font-size:16px;font-weight:800;color:#e6edf3;font-family:JetBrains Mono,monospace">{sel_m}</span>
                    <span style="font-size:12px;color:#8b949e;margin-left:10px">
                        {mi.get('area','')} · {mi.get('zone','')} · {mi.get('feeder_id','')}
                    </span>
                </div>
                <div style="text-align:right">
                    <span style="background:{rc};color:#fff;border-radius:5px;
                                 padding:4px 14px;font-size:12px;font-weight:700">{rb} RISK</span>
                    <div style="font-size:11px;color:#8b949e;margin-top:4px">
                        Score: <b style="color:{rc}">{mi.get('final_risk_score',0):.4f}</b>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1: st.metric("Avg kWh/day",  f"{md['kwh_consumed'].mean():.2f}")
        with c2: st.metric("Expected kWh", f"{md['mean_kwh'].mean():.2f}")
        with c3: st.metric("Power Factor", f"{md['power_factor'].mean():.3f}")
        with c4: st.metric("Avg Voltage",  f"{md['voltage_volts'].mean():.1f}V")
        with c5: st.metric("Total Loss",   f"₹{md['estimated_loss_rs'].sum():.2f}")
        with c6: st.metric("Anomaly Days", f"{int(md['anomaly_label'].sum())}/{len(md)}")

        c_left, c_right = st.columns([2, 1])
        with c_left:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=md["reading_date"], y=md["mean_kwh"],
                name="Expected kWh",
                line=dict(color=C["p1"], width=1.5, dash="dot"), opacity=0.7))
            fig.add_trace(go.Scatter(
                x=md["reading_date"], y=md["kwh_consumed"],
                name="Actual kWh",
                line=dict(color=C["p3"], width=2),
                fill="tozeroy", fillcolor="rgba(63,185,80,0.06)"))
            anom_rows = md[md["anomaly_label"] == 1]
            if len(anom_rows):
                fig.add_trace(go.Scatter(
                    x=anom_rows["reading_date"], y=anom_rows["kwh_consumed"],
                    mode="markers", name="⚠ Anomaly",
                    marker=dict(color=C["High"], size=10, symbol="x",
                                line=dict(color=C["High"], width=2))))
            fig.update_layout(**LAYOUT, height=320,
                              title=f"Consumption Pattern — {sel_m}",
                              title_font=dict(color="#c9d1d9"),
                              xaxis=dict(title="Date", **GRID, **AXIS),
                              yaxis=dict(title="kWh/day", **GRID, **AXIS))
            st.plotly_chart(fig, use_container_width=True)

        with c_right:
            # Power factor trend
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=md["reading_date"], y=md["power_factor"],
                name="Power Factor", line=dict(color=C["p2"], width=1.5),
                fill="tozeroy", fillcolor="rgba(188,140,255,0.06)"))
            fig2.add_hline(y=0.75, line_dash="dot", line_color=C["High"],
                           annotation_text="Min PF 0.75",
                           annotation_font_color=C["High"])
            fig2.update_layout(**LAYOUT, height=320,
                               title="Power Factor Trend",
                               title_font=dict(color="#c9d1d9"),
                               xaxis=dict(title="Date", **GRID, **AXIS),
                               yaxis=dict(title="PF", range=[0.3, 1.05], **GRID, **AXIS))
            st.plotly_chart(fig2, use_container_width=True)

        # Risk score and voltage
        c_left2, c_right2 = st.columns([2, 1])
        with c_left2:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=md["reading_date"], y=md["final_risk_score"],
                name="Risk Score",
                line=dict(color=C["High"], width=1.5),
                fill="tozeroy", fillcolor="rgba(248,81,73,0.06)"))
            fig3.add_hline(y=0.60, line_dash="dot", line_color=C["Medium"],
                           annotation_text="High Threshold 0.60",
                           annotation_font_color=C["Medium"])
            fig3.update_layout(**LAYOUT, height=240,
                               title="Daily Risk Score",
                               title_font=dict(color="#c9d1d9"),
                               xaxis=dict(**GRID, **AXIS),
                               yaxis=dict(title="Score", range=[0, 1.05], **GRID, **AXIS))
            st.plotly_chart(fig3, use_container_width=True)

        with c_right2:
            # Meter info card
            st.markdown(f"""
            <div style="background:#161b22;border:1px solid #21262d;border-radius:8px;padding:14px">
                <div style="font-size:10px;color:#8b949e;letter-spacing:1px;margin-bottom:10px">METER DETAILS</div>
                <div style="font-size:11px;color:#c9d1d9;line-height:2">
                    <b style="color:#8b949e">Connection</b><br>{mi.get('connection_type','N/A')}<br>
                    <b style="color:#8b949e">Feeder</b><br>{mi.get('feeder_id','N/A')}<br>
                    <b style="color:#8b949e">Tariff</b><br>₹{mi.get('tariff_rs_per_kwh','N/A')}/kWh<br>
                    <b style="color:#8b949e">Sanctioned Load</b><br>{mi.get('sanctioned_load_kw','N/A')} kW<br>
                    <b style="color:#8b949e">Anomaly Type</b><br>
                    <span style="color:{C['High']}">{mi.get('anomaly_type','Normal')}</span>
                </div>
            </div>""", unsafe_allow_html=True)

        # Anomaly table
        if len(anom_rows):
            st.markdown(f"**⚠️ {len(anom_rows)} Anomaly Days Detected for {sel_m}**")
            show = [c for c in ["reading_date", "kwh_consumed", "mean_kwh", "z_score",
                                  "power_factor", "voltage_volts", "anomaly_type",
                                  "risk_band", "final_risk_score", "estimated_loss_rs"]
                    if c in anom_rows.columns]
            st.dataframe(anom_rows[show].sort_values("final_risk_score", ascending=False),
                         use_container_width=True, height=220)

# ═══════════════════════════════════════════════════════════
#  TAB 5 — ALERTS
# ═══════════════════════════════════════════════════════════
with tab5:
    st.markdown("### 🚨 High Risk Alerts — Inspection Required")
    high_df = dff[dff["risk_band"] == "High"].copy()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div style="background:rgba(248,81,73,0.08);border:1px solid rgba(248,81,73,0.25);
                    border-top:3px solid {C['High']};border-radius:10px;padding:16px">
            <div style="font-size:10px;color:{C['High']};font-weight:700;letter-spacing:1px">HIGH RISK METERS</div>
            <div style="font-size:32px;font-weight:800;color:{C['High']};margin:8px 0;font-family:JetBrains Mono,monospace">
                {high_df['meter_id'].nunique()}</div>
            <div style="font-size:11px;color:#8b949e">Immediate inspection needed</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div style="background:rgba(227,179,65,0.08);border:1px solid rgba(227,179,65,0.25);
                    border-top:3px solid {C['Medium']};border-radius:10px;padding:16px">
            <div style="font-size:10px;color:{C['Medium']};font-weight:700;letter-spacing:1px">TOTAL REVENUE LOSS</div>
            <div style="font-size:32px;font-weight:800;color:{C['Medium']};margin:8px 0;font-family:JetBrains Mono,monospace">
                ₹{high_df['estimated_loss_rs'].sum():,.0f}</div>
            <div style="font-size:11px;color:#8b949e">Recoverable with field action</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        top_zone = high_df.groupby("zone")["meter_id"].nunique().idxmax() if len(high_df) else "N/A"
        st.markdown(f"""
        <div style="background:rgba(188,140,255,0.08);border:1px solid rgba(188,140,255,0.25);
                    border-top:3px solid {C['p2']};border-radius:10px;padding:16px">
            <div style="font-size:10px;color:{C['p2']};font-weight:700;letter-spacing:1px">HIGHEST RISK ZONE</div>
            <div style="font-size:22px;font-weight:800;color:{C['p2']};margin:8px 0">{top_zone}</div>
            <div style="font-size:11px;color:#8b949e">Most meters flagged</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Build alert summary per meter
    alert_df = (
        high_df.groupby("meter_id").agg(
            zone            = ("zone", "first"),
            area            = ("area", "first"),
            feeder_id       = ("feeder_id", "first"),
            connection_type = ("connection_type", "first"),
            avg_risk        = ("final_risk_score", "mean"),
            total_loss      = ("estimated_loss_rs", "sum"),
            anomaly_days    = ("anomaly_label", "sum"),
            total_days      = ("reading_date", "count"),
            anomaly_type    = ("anomaly_type",
                               lambda x: x[x != "Normal"].mode()[0]
                               if (x != "Normal").any() else "Unknown"),
            avg_kwh         = ("kwh_consumed", "mean"),
            expected_kwh    = ("mean_kwh", "mean"),
        )
        .reset_index()
        .sort_values("avg_risk", ascending=False)
    )
    alert_df["anomaly_freq_pct"] = (alert_df["anomaly_days"] / alert_df["total_days"] * 100).round(1)
    alert_df["avg_risk"]   = alert_df["avg_risk"].round(4)
    alert_df["total_loss"] = alert_df["total_loss"].round(2)

    # Chart: loss by area
    col1, col2 = st.columns(2)
    with col1:
        area_loss = alert_df.groupby("area").agg(
            total_loss = ("total_loss", "sum"),
            meters     = ("meter_id", "count"),
        ).reset_index().sort_values("total_loss", ascending=True).tail(12)
        fig = px.bar(area_loss, x="total_loss", y="area", orientation="h",
                     color="meters",
                     color_continuous_scale=["#1f6feb", "#f85149"],
                     title="Revenue Loss by Area (High-Risk Meters)",
                     text=area_loss["total_loss"].apply(lambda x: f"₹{x:,.0f}"))
        fig.update_traces(textposition="outside", textfont=dict(size=9, color="#8b949e"))
        fig.update_layout(**LAYOUT, height=360,
                          title_font=dict(color="#c9d1d9"),
                          xaxis=dict(title="Total Loss ₹", **GRID, **AXIS),
                          yaxis=dict(title="", **AXIS),
                          coloraxis_colorbar=dict(title="Meters",
                                                   tickfont=dict(color="#8b949e")))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        type_loss = alert_df.groupby("anomaly_type").agg(
            total_loss = ("total_loss", "sum"),
            count      = ("meter_id", "count"),
        ).reset_index()
        fig2 = px.pie(type_loss, names="anomaly_type", values="total_loss",
                      hole=0.45, title="Loss Share by Anomaly Type",
                      color_discrete_sequence=[C["High"], C["Medium"], C["p1"],
                                               C["p2"], C["p3"], C["p4"], "#ff7b72"])
        fig2.update_traces(textfont=dict(size=10, color="#e6edf3"),
                           textinfo="label+percent")
        fig2.update_layout(**LAYOUT, height=360, title_font=dict(color="#c9d1d9"))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### 🔴 Top 15 Critical Meters")
    for _, row in alert_df.head(15).iterrows():
        loss_color = C["Medium"] if row["total_loss"] > 0 else "#8b949e"
        c1, c2, c3, c4 = st.columns([3, 2, 2, 1])
        with c1:
            st.markdown(f"""
            <div style="background:#161b22;border:1px solid rgba(248,81,73,0.2);
                        border-left:3px solid {C['High']};border-radius:7px;padding:10px 14px">
                <div style="font-weight:700;font-size:13px;color:#e6edf3;
                            font-family:JetBrains Mono,monospace">{row['meter_id']}</div>
                <div style="font-size:11px;color:#8b949e;margin-top:2px">
                    {row['area']} · {row['zone']} · {row['feeder_id']}</div>
                <div style="margin-top:5px">
                    <span style="font-size:10px;background:rgba(248,81,73,0.15);
                                 color:{C['High']};border-radius:4px;
                                 padding:2px 8px;font-weight:600">
                        {row.get('anomaly_type','Unknown')}
                    </span>
                    <span style="font-size:10px;color:#8b949e;margin-left:8px">
                        {row.get('connection_type','N/A')}
                    </span>
                </div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.metric("Risk Score",   f"{row['avg_risk']:.4f}")
        with c3:
            st.metric("Revenue Loss", f"₹{row['total_loss']:,.2f}")
        with c4:
            st.metric("Anom %",       f"{row['anomaly_freq_pct']}%")

    st.markdown("---")

    # Full table
    st.markdown("#### 📋 Full High-Risk Alert Table")
    show_cols = ["meter_id", "zone", "area", "feeder_id", "connection_type",
                 "anomaly_type", "avg_risk", "total_loss",
                 "anomaly_freq_pct", "avg_kwh", "expected_kwh"]
    avail = [c for c in show_cols if c in alert_df.columns]
    st.dataframe(alert_df[avail], use_container_width=True, height=400)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "📥 Download Alert List (CSV)",
            alert_df.to_csv(index=False).encode(),
            "voltx2_high_risk_alerts.csv", "text/csv")
    with c2:
        st.download_button(
            "📥 Download Full Filtered Dataset",
            dff.to_csv(index=False).encode(),
            "voltx2_filtered_data.csv", "text/csv")

# ── FOOTER ───────────────────────────────────────────────
st.markdown("---")
st.caption(
    "VOLTX 2.0 · Isolation Forest + Random Forest + Gradient Boosting + "
    "Logistic Regression · TNEB Tamil Nadu · Chennai · Avadi · Tiruvallur"
)