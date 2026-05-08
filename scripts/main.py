import os
import sys
import io
import json
import tempfile
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from datetime import datetime
from PIL import Image
from sklearn.metrics import confusion_matrix

from app.utils.predictor import predict_image

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="HelioSight AI",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# LOAD METADATA
# ─────────────────────────────────────────────────────────────

with open("models/model_meta.json") as f:
    meta = json.load(f)

CLASSES = meta["classes"]

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"], .stApp {
    background:
radial-gradient(circle at top left, rgba(37,99,235,0.12), transparent 25%),
radial-gradient(circle at top right, rgba(14,165,233,0.08), transparent 20%),
#050816 !important;
    color: #d1d1d1;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    -webkit-font-smoothing: antialiased;
}

#MainMenu, footer, header { visibility: hidden; }

.block-container {
    max-width: 1320px !important;
    padding: 0.5rem 1.2rem 2rem 1.2rem !important;
}

/* ── Sidebar ──────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #080808 !important;
    border-right: 1px solid #141414 !important;
    min-width: 170px !important;
    max-width: 170px !important;
}

section[data-testid="stSidebar"] > div {
    padding: 1.5rem 1rem !important;
}

[data-testid="stRadio"] > label { display: none !important; }

[data-testid="stRadio"] div[role="radiogroup"] {
    display: flex; flex-direction: column; gap: 2px;
}

[data-testid="stRadio"] label[data-baseweb="radio"] {
    padding: 0.45rem 0.6rem !important;
    border-radius: 5px !important;
    cursor: pointer;
    transition: background 0.1s;
}

[data-testid="stRadio"] label[data-baseweb="radio"]:hover {
    background: rgba(255,255,255,0.04) !important;
}

[data-testid="stRadio"] label[data-baseweb="radio"] p {
    font-size: 0.78rem !important;
    color: #555 !important;
    font-weight: 450 !important;
    letter-spacing: 0.01em !important;
    line-height: 1 !important;
}

[data-testid="stRadio"] [aria-checked="true"] p {
    color: #e8e8e8 !important;
    font-weight: 500 !important;
}

[data-testid="stRadio"] [aria-checked="true"] {
    background: rgba(255,255,255,0.06) !important;
}

/* ── KPI cards ────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #0d0d0d !important;
    border: 1px solid #1a1a1a !important;
    border-radius: 7px !important;
    padding: 0.65rem 0.85rem !important;
    transition: border-color 0.15s;
}

[data-testid="stMetric"]:hover { border-color: #252525 !important; }

[data-testid="stMetric"] label {
    font-size: 0.61rem !important;
    font-weight: 500 !important;
    color: #3e3e3e !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    line-height: 1 !important;
}

[data-testid="stMetricValue"] {
    font-size: 1.1rem !important;
    font-weight: 650 !important;
    color: #ececec !important;
    letter-spacing: -0.025em !important;
    line-height: 1.2 !important;
}

[data-testid="stMetric"] {
    background: linear-gradient(180deg, #0f1722 0%, #0b1017 100%) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 14px !important;
    padding: 0.65rem 0.85rem !important;
    box-shadow: 0 4px 18px rgba(0,0,0,0.35);
}

[data-testid="stMetricValue"] {
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: #f8fafc !important;
}

[data-testid="stMetric"] label {
    color: #7c8aa0 !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em !important;
}

/* ── Top bar ──────────────────────────────────────────────── */
.hs-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.5rem 0 1rem 0;
    border-bottom: 1px solid #111;
    margin-bottom: 1.1rem;
}

.hs-page-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #f8fafc;
    letter-spacing: -0.03em;
}

.hs-page-sub {
    font-size: 0.7rem;
    color: #303030;
    margin-top: 0.1rem;
}

/* ── Cards ────────────────────────────────────────────────── */
.hs-card {
    background: linear-gradient(180deg, #0d1117 0%, #0b0f14 100%);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 0.8rem 0.9rem;
    height: 100%;
    backdrop-filter: blur(10px);
    transition: all 0.2s ease;
    box-shadow: 0 4px 18px rgba(0,0,0,0.35);
}

.hs-card:hover {
    transform: translateY(-2px);
    border-color: rgba(96,165,250,0.25);
}

.hs-card-hd {
    font-size: 0.6rem;
    font-weight: 600;
    color: #333;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding-bottom: 0.6rem;
    margin-bottom: 0.7rem;
    border-bottom: 1px solid #111;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.hs-card-hd-tag {
    font-size: 0.58rem;
    background: #141414;
    border: 1px solid #1e1e1e;
    border-radius: 3px;
    padding: 0.12rem 0.4rem;
    color: #383838;
    text-transform: none;
    letter-spacing: 0;
    font-weight: 400;
}

/* ── Spec rows ────────────────────────────────────────────── */
.hs-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.4rem 0;
    border-bottom: 1px solid #0f0f0f;
}
.hs-row:last-child { border-bottom: none; }
.hs-rk { font-size: 0.7rem; color: #404040; }
.hs-rv { font-size: 0.7rem; color: #b8b8b8; font-weight: 500; }

/* ── Dot ──────────────────────────────────────────────────── */
.hs-dot {
    width: 5px; height: 5px;
    border-radius: 50%;
    background: #22c55e;
    display: inline-block;
    box-shadow: 0 0 5px rgba(34,197,94,0.7);
    margin-right: 0.28rem;
}

/* ── Pills ────────────────────────────────────────────────── */
.hs-pill {
    display: inline-block;
    padding: 0.13rem 0.4rem;
    border-radius: 3px;
    font-size: 0.6rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    line-height: 1.5;
}
.p-critical { background: #280909; color: #f87171; border: 1px solid #3d1111; }
.p-high     { background: #291300; color: #fb923c; border: 1px solid #3e2000; }
.p-medium   { background: #091324; color: #60a5fa; border: 1px solid #102040; }
.p-low      { background: #071510; color: #34d399; border: 1px solid #0e2c1e; }
.p-normal   { background: #111; color: #555; border: 1px solid #1c1c1c; }

/* ── Brand ────────────────────────────────────────────────── */
.hs-brand {
    display: flex; align-items: center; gap: 0.55rem;
    padding-bottom: 1.2rem;
    margin-bottom: 0.9rem;
    border-bottom: 1px solid #141414;
}
.hs-brand-icon {
    width: 24px; height: 24px;
    background: linear-gradient(140deg, #f59e0b, #ef4444);
    border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; font-weight: 800; color: #000;
    flex-shrink: 0;
    box-shadow: 0 2px 10px rgba(245,158,11,0.2);
}
.hs-brand-name {
    font-size: 0.88rem; font-weight: 700;
    color: #e8e8e8; letter-spacing: -0.025em;
}
.hs-brand-ver { font-size: 0.58rem; color: #2a2a2a; }

.hs-nav-label {
    font-size: 0.57rem; font-weight: 600;
    color: #242424; text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 1rem 0 0.4rem 0.5rem;
}

/* ── Upload ───────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: #0a0a0a !important;
    border: 1px dashed #1a1a1a !important;
    border-radius: 8px !important;
    transition: border-color 0.15s;
}

[data-testid="stFileUploader"]:hover { border-color: #282828 !important; }

[data-testid="stFileUploader"] section { padding: 1.25rem !important; }

[data-testid="stFileUploader"] label {
    color: #383838 !important;
    font-size: 0.75rem !important;
}

/* ── Inference result ─────────────────────────────────────── */
.hs-result {
    background: #0d0d0d;
    border: 1px solid #161616;
    border-radius: 8px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.7rem;
    position: relative;
    overflow: hidden;
}

.hs-result::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 2px; height: 100%;
    background: linear-gradient(180deg, #f59e0b, #ef4444);
}

.hs-result-label {
    font-size: 1.25rem; font-weight: 700;
    letter-spacing: -0.03em; line-height: 1.2;
    margin-bottom: 0.4rem;
}

.hs-result-sub {
    display: flex; align-items: center;
    gap: 0.7rem; font-size: 0.7rem; color: #383838;
}

.hs-conf-val { color: #86efac; font-weight: 600; }

/* ── Dataframe ────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid #161616 !important;
    border-radius: 6px !important;
    overflow: hidden;
}

/* ── Button ───────────────────────────────────────────────── */
.stButton > button,
[data-testid="stDownloadButton"] > button {
    background: #111 !important;
    color: #aaa !important;
    border: 1px solid #1e1e1e !important;
    border-radius: 5px !important;
    padding: 0.35rem 0.8rem !important;
    font-size: 0.7rem !important;
    font-weight: 500 !important;
    transition: all 0.1s !important;
    box-shadow: none !important;
    font-family: 'Inter', sans-serif !important;
}

.stButton > button:hover,
[data-testid="stDownloadButton"] > button:hover {
    background: #181818 !important;
    border-color: #282828 !important;
    color: #ccc !important;
}

[data-testid="stImage"] img {
    border-radius: 16px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    box-shadow: 0 8px 28px rgba(0,0,0,0.45) !important;
}

.hs-sep { height: 1px; background: #0f0f0f; margin: 0.75rem 0; }

.hs-arch-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.65rem 1rem;
}
.hs-arch-k {
    font-size: 0.6rem; color: #303030;
    text-transform: uppercase; letter-spacing: 0.08em;
    margin-bottom: 0.18rem;
}
.hs-arch-v { font-size: 0.78rem; color: #c0c0c0; font-weight: 500; }

section.main > div {
    padding-top: 0rem !important;
}

.element-container {
    margin-bottom: 0.45rem !important;
}

div[data-testid="stVerticalBlock"] > div:has(.hs-card) {
    gap: 0.65rem !important;
}
</style>
""", unsafe_allow_html=True)



# ─────────────────────────────────────────────────────────────
# DEFECT CONFIG
# ─────────────────────────────────────────────────────────────

DEFECT_COLORS = {
    "Cell":            "#f59e0b",
    "Cell-Multi":      "#f97316",
    "Cracking":        "#ef4444",
    "Diode":           "#8b5cf6",
    "Diode-Multi":     "#7c3aed",
    "Hot-Spot":        "#dc2626",
    "Hot-Spot-Multi":  "#b91c1c",
    "No-Anomaly":      "#22c55e",
    "Offline-Module":  "#6b7280",
    "Shadowing":       "#3b82f6",
    "Soiling":         "#ca8a04",
    "Vegetation":      "#16a34a",
}

SEVERITY_MAP = {
    "No-Anomaly":     "Normal",
    "Soiling":        "Low",
    "Shadowing":      "Medium",
    "Vegetation":     "Medium",
    "Cell":           "Medium",
    "Cell-Multi":     "High",
    "Cracking":       "High",
    "Diode":          "High",
    "Diode-Multi":    "Critical",
    "Hot-Spot":       "Critical",
    "Hot-Spot-Multi": "Critical",
    "Offline-Module": "Critical",
}

ACTION_MAP = {
    "No-Anomaly":     "No maintenance required. System operating normally.",
    "Soiling":        "Schedule panel cleaning within 7 days.",
    "Shadowing":      "Inspect and remove surrounding obstructions.",
    "Vegetation":     "Trim vegetation encroaching on panel area.",
    "Cell":           "Monitor affected cell region. Flag for next service cycle.",
    "Cell-Multi":     "Replace affected cell string. Schedule within 2 weeks.",
    "Cracking":       "Replace cracked module. Take offline if structural risk present.",
    "Diode":          "Inspect bypass diode circuitry. Electrical team required.",
    "Diode-Multi":    "Urgent: multiple diode failures detected.",
    "Hot-Spot":       "Immediate thermal inspection. Isolate panel from array.",
    "Hot-Spot-Multi": "Critical overheating. Take offline immediately.",
    "Offline-Module": "Module disconnected from array. Reconnect or replace.",
}

PILL_MAP = {
    "Normal": "p-normal", "Low": "p-low",
    "Medium": "p-medium", "High": "p-high", "Critical": "p-critical",
}

CLASS_COUNTS = {
    "No-Anomaly": 10000, "Cell": 1877, "Vegetation": 1639, "Diode": 1499,
    "Cell-Multi": 1288, "Shadowing": 1056, "Cracking": 940,
    "Offline-Module": 827, "Hot-Spot": 249, "Hot-Spot-Multi": 246,
    "Soiling": 204, "Diode-Multi": 175,
}

SEVERITY_COUNTS = {
    "Normal": 10000, "Low": 204, "Medium": 3734, "High": 3139, "Critical": 1318,
}

SEV_COLORS     = {"Normal":"#181818","Low":"#0d2b1a","Medium":"#0d1e35","High":"#2a1200","Critical":"#2a0808"}
SEV_TXTCOLORS  = {"Normal":"#505050","Low":"#4ade80","Medium":"#60a5fa","High":"#fb923c","Critical":"#f87171"}

# ─────────────────────────────────────────────────────────────
# PLOTLY HELPERS
# ─────────────────────────────────────────────────────────────

def base_layout(**kw):
    d = dict(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", size=11, color="#404040"),
        hovermode="closest",
        margin=dict(l=0, r=10, t=6, b=0),
        showlegend=False,
    )
    d.update(kw)
    return d

XAXIS_H = dict(showgrid=True, gridwidth=1, gridcolor="#0e0e0e",
               zeroline=False, showline=False, showticklabels=False)

YAXIS_H = dict(showgrid=False, zeroline=False, showline=False,
               tickfont=dict(size=10.5, color="#555"))

XAXIS_V = dict(showgrid=False, zeroline=False, showline=False,
               tickfont=dict(size=9.5, color="#444"))

YAXIS_V = dict(showgrid=False, zeroline=False, showline=False, showticklabels=False)

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def topbar(title, subtitle, tag=None):
    tag_html = (f'<span style="font-size:0.63rem;color:#282828;background:#0f0f0f;'
                f'border:1px solid #1a1a1a;border-radius:3px;padding:0.13rem 0.45rem;'
                f'font-weight:500;">{tag}</span>') if tag else ""
    st.markdown(textwrap.dedent(f"""
<div class="hs-topbar">
    <div>
        <div class="hs-page-title">{title}</div>
        <div class="hs-page-sub">{subtitle}</div>
    </div>
    <div style="display:flex;align-items:center;gap:0.5rem;">
        {tag_html}
        <span style="font-size:0.63rem;color:#202020;">{datetime.now().strftime('%H:%M · %d %b %Y')}</span>
    </div>
</div>
"""), unsafe_allow_html=True)

def hex_rgba(hex_col, alpha):
    r, g, b = int(hex_col[1:3],16), int(hex_col[3:5],16), int(hex_col[5:7],16)
    return f"rgba({r},{g},{b},{alpha})"

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(textwrap.dedent(f"""
<div class="hs-brand">
    <div class="hs-brand-icon">H</div>
    <div>
        <div class="hs-brand-name">HelioSight</div>
        <div class="hs-brand-ver">v2.0 · AI Platform</div>
    </div>
</div>
"""), unsafe_allow_html=True)

    st.markdown('<div class="hs-nav-label">Navigation</div>', unsafe_allow_html=True)

    page = st.radio("nav", ["Dashboard", "Inspection", "Model Insights", "Reports"],
                    label_visibility="collapsed")

    st.markdown('<div class="hs-sep" style="margin:1.2rem 0;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="hs-nav-label">System Status</div>', unsafe_allow_html=True)

    st.markdown(textwrap.dedent(f"""
<div style="margin-top:0.3rem;">
    <div class="hs-row"><span class="hs-rk">Status</span>
        <span class="hs-rv"><span class="hs-dot"></span>Online</span></div>
    <div class="hs-row"><span class="hs-rk">Model</span>
        <span class="hs-rv">{meta['model']}</span></div>
    <div class="hs-row"><span class="hs-rk">Accuracy</span>
        <span class="hs-rv" style="color:#4ade80;">{meta['test_accuracy']*100:.1f}%</span></div>
    <div class="hs-row"><span class="hs-rk">Classes</span>
        <span class="hs-rv">{len(CLASSES)}</span></div>
    <div class="hs-row"><span class="hs-rk">Samples</span>
        <span class="hs-rv">20,000</span></div>
</div>
"""), unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ═════════════════════════════════════════════════════════════

if page == "Dashboard":

    topbar("Solar Defect Detection",
           "Real-time AI thermal inspection analytics · Drone imagery pipeline", "Live")

    # KPIs
    k1, k2, k3, k4, k5 = st.columns(5, gap="small")
    k1.metric("Total Images", "20,000")
    k2.metric("Defect Classes", "12")
    k3.metric("Model Accuracy", f"{meta['test_accuracy']*100:.1f}%")
    k4.metric("Architecture", meta["model"])
    k5.metric("Resolution", "128 × 128")

    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

    col_main, col_side = st.columns([1.8, 1], gap="medium")

    # ── Bar chart ─────────────────────────────────────────────
    with col_main:
        st.markdown('<div class="hs-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="hs-card-hd">Defect Distribution'
            '<span class="hs-card-hd-tag">12 classes · 20K samples</span></div>',
            unsafe_allow_html=True)

        sorted_items = sorted(CLASS_COUNTS.items(), key=lambda x: x[1], reverse=False)
        labels = [k for k, v in sorted_items]
        values = [v for k, v in sorted_items]
        colors = [hex_rgba(DEFECT_COLORS.get(k, "#444"), 0.8) for k in labels]

        fig_bar = go.Figure(go.Bar(
            y=labels, x=values,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            width=0.62,
            text=[f"{v:,}" for v in values],
            textposition="outside",
            textfont=dict(size=9.5, color="#383838", family="Inter"),
            hovertemplate="<b>%{y}</b>  %{x:,}<extra></extra>",
            cliponaxis=False,
        ))

        fig_bar.update_layout(**base_layout(
            height=290,
            xaxis=dict(**XAXIS_H, range=[0, max(values) * 1.2]),
            yaxis=dict(**YAXIS_H),
        ))

        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with col_side:
        # ── Severity vertical bar ─────────────────────────────
        st.markdown('<div class="hs-card" style="margin-bottom:0.7rem;">', unsafe_allow_html=True)
        st.markdown('<div class="hs-card-hd">Severity Breakdown</div>', unsafe_allow_html=True)

        sev_labels = list(SEVERITY_COUNTS.keys())
        sev_values = list(SEVERITY_COUNTS.values())

        fig_sev = go.Figure(go.Bar(
            x=sev_labels, y=sev_values,
            marker=dict(
                color=[SEV_COLORS[k] for k in sev_labels],
                line=dict(color=[SEV_TXTCOLORS[k] for k in sev_labels], width=1),
            ),
            width=0.5,
            text=[f"{v:,}" for v in sev_values],
            textposition="outside",
            textfont=dict(size=9, color=[SEV_TXTCOLORS[k] for k in sev_labels]),
            hovertemplate="<b>%{x}</b>: %{y:,}<extra></extra>",
        ))

        fig_sev.update_layout(**base_layout(
            height=170,
            margin=dict(l=0, r=0, t=6, b=0),
            xaxis=dict(**XAXIS_V),
            yaxis=dict(**YAXIS_V, range=[0, max(sev_values) * 1.25]),
        ))

        st.plotly_chart(fig_sev, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

        # ── System card ───────────────────────────────────────
        st.markdown(textwrap.dedent(f"""
<div class="hs-card">
    <div class="hs-card-hd">System Overview</div>
    <div class="hs-row"><span class="hs-rk">Engine</span><span class="hs-rv">CNN Inference</span></div>
    <div class="hs-row"><span class="hs-rk">Pipeline</span>
        <span class="hs-rv"><span class="hs-dot"></span>Active</span></div>
    <div class="hs-row"><span class="hs-rk">Resolution</span><span class="hs-rv">128 × 128 px</span></div>
    <div class="hs-row"><span class="hs-rk">Source</span><span class="hs-rv">Drone Thermal</span></div>
    <div class="hs-row"><span class="hs-rk">Format</span><span class="hs-rv">Infrared (IR)</span></div>
    <div class="hs-row"><span class="hs-rk">Accuracy</span>
        <span class="hs-rv" style="color:#4ade80;">{meta['test_accuracy']*100:.2f}%</span></div>
</div>
"""), unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
# PAGE 2 — INSPECTION
# ═════════════════════════════════════════════════════════════

elif page == "Inspection":

    topbar("Thermal Inspection",
           "Upload drone-captured infrared imagery for AI defect classification")

    uploaded_file = st.file_uploader(
        "Drop thermal image here — JPG / JPEG / PNG",
        type=["jpg", "jpeg", "png"],
        label_visibility="visible",
    )

    if uploaded_file is not None:
        col_img, col_out = st.columns([1, 1.05], gap="medium")

        with col_img:
            st.markdown('<div class="hs-card">', unsafe_allow_html=True)
            st.markdown('<div class="hs-card-hd">Input Image</div>', unsafe_allow_html=True)
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
            w, h = image.size
            st.markdown(textwrap.dedent(f"""
<div style="display:flex;gap:1.25rem;margin-top:0.65rem;padding-top:0.55rem;border-top:1px solid #0f0f0f;">
    <div>
        <div style="font-size:0.58rem;color:#282828;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.15rem;">Width</div>
        <div style="font-size:0.75rem;color:#888;font-weight:500;">{w}px</div>
    </div>
    <div>
        <div style="font-size:0.58rem;color:#282828;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.15rem;">Height</div>
        <div style="font-size:0.75rem;color:#888;font-weight:500;">{h}px</div>
    </div>
    <div>
        <div style="font-size:0.58rem;color:#282828;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.15rem;">Mode</div>
        <div style="font-size:0.75rem;color:#888;font-weight:500;">{image.mode}</div>
    </div>
    <div>
        <div style="font-size:0.58rem;color:#282828;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.15rem;">File</div>
        <div style="font-size:0.75rem;color:#888;font-weight:500;">{uploaded_file.name[:16]}</div>
    </div>
</div>
"""), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Inference
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file.getvalue())
            temp_path = tmp.name

        result      = predict_image(temp_path)
        os.remove(temp_path)

        predicted_label = result["label"]
        confidence      = result["confidence"]
        severity        = SEVERITY_MAP.get(predicted_label, "Review")
        action          = ACTION_MAP.get(predicted_label, "Review inspection results.")
        pill_cls        = PILL_MAP.get(severity, "p-normal")
        defect_color    = DEFECT_COLORS.get(predicted_label, "#888")

        with col_out:
            # Result header
            st.markdown(textwrap.dedent(f"""
<div class="hs-result">
    <div style="font-size:0.58rem;color:#242424;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.45rem;font-weight:600;">Detection Result</div>
    <div class="hs-result-label" style="color:{defect_color};">{predicted_label}</div>
    <div class="hs-result-sub">
        <span class="hs-conf-val">{confidence*100:.1f}% confidence</span>
        <span>·</span>
        <span class="hs-pill {pill_cls}">{severity}</span>
    </div>
</div>
"""), unsafe_allow_html=True)

            # Action
            st.markdown(textwrap.dedent(f"""
<div class="hs-card" style="margin-bottom:0.7rem;">
    <div class="hs-card-hd">Recommended Action</div>
    <p style="font-size:0.76rem;color:#808080;line-height:1.65;margin:0;">{action}</p>
</div>
"""), unsafe_allow_html=True)

            # Confidence chart
            st.markdown('<div class="hs-card">', unsafe_allow_html=True)
            st.markdown('<div class="hs-card-hd">Class Probabilities</div>', unsafe_allow_html=True)

            prob_df = pd.DataFrame({
                "Class": list(result["probabilities"].keys()),
                "Probability": list(result["probabilities"].values()),
            }).sort_values("Probability", ascending=True)

            bar_colors_p = [
                hex_rgba(DEFECT_COLORS.get(c, "#444444"), 0.85 if c == predicted_label else 0.18)
                for c in prob_df["Class"]
            ]

            fig_prob = go.Figure(go.Bar(
                y=prob_df["Class"],
                x=prob_df["Probability"],
                orientation="h",
                marker=dict(color=bar_colors_p, line=dict(width=0)),
                width=0.58,
                text=[f"{v:.1%}" if v >= 0.01 else "" for v in prob_df["Probability"]],
                textposition="outside",
                textfont=dict(size=9, color="#383838"),
                hovertemplate="<b>%{y}</b>: %{x:.2%}<extra></extra>",
                cliponaxis=False,
            ))

            fig_prob.update_layout(**base_layout(
                height=max(240, len(prob_df) * 22),
                xaxis=dict(**XAXIS_H, range=[0, 1.2]),
                yaxis=dict(**YAXIS_H),
            ))

            st.plotly_chart(fig_prob, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.markdown(textwrap.dedent("""
<div style="
    margin-top:1rem; border:1px dashed #141414; border-radius:10px;
    padding:4rem 2rem; text-align:center; background:rgba(255,255,255,0.008);
">
    <div style="font-size:1.5rem;margin-bottom:0.6rem;opacity:0.12;">⬡</div>
    <div style="font-size:0.75rem;color:#262626;line-height:1.9;">
        Upload a thermal image to begin AI defect classification<br>
        <span style="font-size:0.65rem;color:#1e1e1e;">JPG · JPEG · PNG</span>
    </div>
</div>
"""), unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
# PAGE 3 — MODEL INSIGHTS
# ═════════════════════════════════════════════════════════════

elif page == "Model Insights":

    topbar("Model Insights",
           "Training analytics, architecture specification, and performance metrics")

    m1, m2, m3, m4 = st.columns(4, gap="small")
    m1.metric("Architecture", meta["model"])
    m2.metric("Input Shape", "128 × 128 × 3")
    m3.metric("Output Classes", len(CLASSES))
    m4.metric("Test Accuracy", f"{meta['test_accuracy']*100:.2f}%")

    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2, gap="medium")

    with col_l:
        st.markdown('<div class="hs-card">', unsafe_allow_html=True)
        st.markdown('<div class="hs-card-hd">Training Accuracy</div>', unsafe_allow_html=True)
        st.image("models/training_accuracy.png", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="hs-card">', unsafe_allow_html=True)
        st.markdown('<div class="hs-card-hd">Training Loss</div>', unsafe_allow_html=True)
        st.image("models/training_loss.png", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)

    st.markdown(textwrap.dedent(f"""
<div class="hs-card">
    <div class="hs-card-hd">Architecture Overview</div>
    <div class="hs-arch-grid">
        <div><div class="hs-arch-k">Base Model</div><div class="hs-arch-v">{meta['model']}</div></div>
        <div><div class="hs-arch-k">Input Shape</div><div class="hs-arch-v">128 × 128 × 3</div></div>
        <div><div class="hs-arch-k">Classifier</div><div class="hs-arch-v">Dense + Softmax</div></div>
        <div><div class="hs-arch-k">Output</div><div class="hs-arch-v">{len(CLASSES)} Classes</div></div>
        <div><div class="hs-arch-k">Preprocessing</div><div class="hs-arch-v">ImageNet Norm.</div></div>
        <div><div class="hs-arch-k">Augmentation</div><div class="hs-arch-v">Flip · Rotate · Zoom</div></div>
        <div><div class="hs-arch-k">Dataset Size</div><div class="hs-arch-v">20,000 Samples</div></div>
        <div><div class="hs-arch-k">Test Accuracy</div>
            <div class="hs-arch-v" style="color:#4ade80;">{meta['test_accuracy']*100:.2f}%</div></div>
    </div>
</div>
"""), unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
# PAGE 4 — REPORTS
# ═════════════════════════════════════════════════════════════

elif page == "Reports":

    topbar("Inspection Reports",
           "Aggregated analytics, model performance summary, and export")

    r_left, r_right = st.columns([1.35, 1], gap="medium")

    with r_left:
        st.markdown('<div class="hs-card">', unsafe_allow_html=True)
        st.markdown('<div class="hs-card-hd">Performance Summary</div>', unsafe_allow_html=True)

        report_data = {
            "Metric": [
                "Total Inspected Images", "Defect Categories", "Model Architecture",
                "Test Accuracy", "Critical Defect Types", "Inspection Method",
                "Image Source", "Report Generated",
            ],
            "Value": [
                "20,000", "12", meta["model"],
                f"{meta['test_accuracy']*100:.2f}%",
                "Hot-Spot, Hot-Spot-Multi, Cracking, Diode-Multi",
                "Drone Thermal Imagery", "Infrared (IR) Camera",
                datetime.now().strftime("%Y-%m-%d %H:%M"),
            ],
        }

        report_df = pd.DataFrame(report_data)

        st.dataframe(
            report_df, use_container_width=True, hide_index=True,
            column_config={
                "Metric": st.column_config.TextColumn("Metric", width="medium"),
                "Value":  st.column_config.TextColumn("Value",  width="large"),
            },
        )

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        csv = report_df.to_csv(index=False)
        st.download_button(
            "↓  Export CSV",
            csv,
            file_name=f"heliosight_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

    with r_right:
        st.markdown('<div class="hs-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="hs-card-hd">Defect Priority Index'
            '<span class="hs-card-hd-tag">12 classes</span></div>',
            unsafe_allow_html=True)

        defect_priority = [
            ("Hot-Spot-Multi",  "Critical", 246),
            ("Hot-Spot",        "Critical", 249),
            ("Diode-Multi",     "Critical", 175),
            ("Offline-Module",  "Critical", 827),
            ("Cracking",        "High",     940),
            ("Cell-Multi",      "High",    1288),
            ("Diode",           "High",    1499),
            ("Shadowing",       "Medium",  1056),
            ("Cell",            "Medium",  1877),
            ("Vegetation",      "Medium",  1639),
            ("Soiling",         "Low",      204),
            ("No-Anomaly",      "Normal", 10000),
        ]

        for name, sev, count in defect_priority:
            pill_cls = PILL_MAP.get(sev, "p-normal")
            st.markdown(textwrap.dedent(f"""
<div class="hs-row">
    <span class="hs-rk">{name}</span>
    <div style="display:flex;align-items:center;gap:0.6rem;">
        <span style="font-size:0.63rem;color:#222;font-variant-numeric:tabular-nums;">{count:,}</span>
        <span class="hs-pill {pill_cls}">{sev}</span>
    </div>
</div>
"""), unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(f"""
<div class="hs-card" style="margin-top:0.7rem;">
    <div class="hs-card-hd">Notes</div>
    <p style="font-size:0.7rem;color:#2e2e2e;line-height:1.8;margin:0;">
        Report reflects model performance across all 20,000 training samples.
        Upload individual thermal images on the Inspection page for per-panel results.
        Generated {datetime.now().strftime('%d %b %Y, %H:%M')}.
    </p>
</div>
""", unsafe_allow_html=True)