import os
import sys
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from datetime import datetime
from PIL import Image

from app.utils.predictor import predict_image

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="HelioSight AI",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# METADATA
# ─────────────────────────────────────────────────────────────

with open("models/model_meta.json") as f:
    meta = json.load(f)

CLASSES = meta["classes"]

# ─────────────────────────────────────────────────────────────
# DESIGN TOKENS
# ─────────────────────────────────────────────────────────────

BG          = "#06070F"
BG_CARD     = "#0D0F1C"
BG_CARD2    = "#111326"
BORDER      = "rgba(255,255,255,0.07)"
BORDER_HL   = "rgba(255,255,255,0.14)"
BLUE        = "#3B82F6"
BLUE_DIM    = "rgba(59,130,246,0.12)"
GREEN       = "#22C55E"
GREEN_DIM   = "rgba(34,197,94,0.12)"
AMBER       = "#F59E0B"
RED         = "#EF4444"
TEXT_P      = "#F1F5F9"   # primary
TEXT_S      = "#64748B"   # secondary
TEXT_T      = "#334155"   # tertiary
FONT        = "'Inter', -apple-system, BlinkMacSystemFont, sans-serif"

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

html, body, [class*="css"], .stApp {{
    background-color: {BG} !important;
    font-family: {FONT};
    color: {TEXT_P};
    -webkit-font-smoothing: antialiased;
}}

#MainMenu, footer, header {{ visibility: hidden; }}

/* ── Layout ─────────────────────────────────────────── */
.block-container {{
    max-width: 1480px !important;
    padding: 0 2rem 4rem 2rem !important;
}}

/* ── Sidebar ─────────────────────────────────────────── */
section[data-testid="stSidebar"] {{
    background: #080A14 !important;
    border-right: 1px solid {BORDER} !important;
    min-width: 220px !important;
    max-width: 220px !important;
}}

section[data-testid="stSidebar"] > div {{
    padding: 1.75rem 1.25rem !important;
}}

/* Radio buttons */
[data-testid="stRadio"] > label {{ display: none !important; }}

[data-testid="stRadio"] div[role="radiogroup"] {{
    display: flex; flex-direction: column; gap: 3px;
}}

[data-testid="stRadio"] label[data-baseweb="radio"] {{
    padding: 0.5rem 0.75rem !important;
    border-radius: 8px !important;
    cursor: pointer;
    transition: all 0.15s ease;
    border: 1px solid transparent !important;
}}

[data-testid="stRadio"] label[data-baseweb="radio"]:hover {{
    background: rgba(59,130,246,0.08) !important;
    border-color: rgba(59,130,246,0.15) !important;
}}

[data-testid="stRadio"] label[data-baseweb="radio"] p {{
    font-size: 0.82rem !important;
    color: {TEXT_S} !important;
    font-weight: 500 !important;
    letter-spacing: 0.01em !important;
    line-height: 1 !important;
}}

[data-testid="stRadio"] [aria-checked="true"] p {{
    color: {TEXT_P} !important;
    font-weight: 600 !important;
}}

[data-testid="stRadio"] [aria-checked="true"] {{
    background: rgba(59,130,246,0.1) !important;
    border-color: rgba(59,130,246,0.2) !important;
}}

/* ── KPI Metrics ─────────────────────────────────────── */
[data-testid="stMetric"] {{
    background: {BG_CARD} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 12px !important;
    padding: 1rem 1.1rem 0.85rem 1.1rem !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
    position: relative;
    overflow: hidden;
}}

[data-testid="stMetric"]::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(59,130,246,0.4), transparent);
}}

[data-testid="stMetric"]:hover {{
    border-color: {BORDER_HL} !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}}

[data-testid="stMetric"] label {{
    font-size: 0.65rem !important;
    font-weight: 600 !important;
    color: {TEXT_S} !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    line-height: 1 !important;
}}

[data-testid="stMetricValue"] {{
    font-size: 1.35rem !important;
    font-weight: 700 !important;
    color: {TEXT_P} !important;
    letter-spacing: -0.03em !important;
    line-height: 1.25 !important;
}}

[data-testid="stMetricDelta"] {{ display: none !important; }}

/* ── Cards ───────────────────────────────────────────── */
.hs-card {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 1.1rem 1.25rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s ease;
}}

.hs-card:hover {{ border-color: {BORDER_HL}; }}

.hs-card-glow {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 1.1rem 1.25rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 0 0 1px rgba(59,130,246,0.05), 0 8px 40px rgba(0,0,0,0.3);
}}

.hs-card-title {{
    font-size: 0.65rem;
    font-weight: 700;
    color: {TEXT_S};
    text-transform: uppercase;
    letter-spacing: 0.1em;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 0.75rem;
    margin-bottom: 0.85rem;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}}

.hs-badge {{
    font-size: 0.6rem;
    font-weight: 600;
    padding: 0.18rem 0.5rem;
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    line-height: 1.6;
    display: inline-block;
}}

.badge-blue     {{ background: rgba(59,130,246,0.15); color: #60A5FA; border: 1px solid rgba(59,130,246,0.25); }}
.badge-green    {{ background: rgba(34,197,94,0.12);  color: #4ADE80; border: 1px solid rgba(34,197,94,0.22); }}
.badge-amber    {{ background: rgba(245,158,11,0.12); color: #FCD34D; border: 1px solid rgba(245,158,11,0.22); }}
.badge-red      {{ background: rgba(239,68,68,0.12);  color: #F87171; border: 1px solid rgba(239,68,68,0.22); }}
.badge-orange   {{ background: rgba(249,115,22,0.12); color: #FB923C; border: 1px solid rgba(249,115,22,0.22); }}
.badge-slate    {{ background: rgba(100,116,139,0.12);color: #94A3B8; border: 1px solid rgba(100,116,139,0.22); }}

/* ── Status row ──────────────────────────────────────── */
.hs-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.45rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    font-size: 0.78rem;
}}
.hs-row:last-child {{ border-bottom: none; }}
.hs-row-k {{ color: {TEXT_S}; font-weight: 400; }}
.hs-row-v {{ color: {TEXT_P}; font-weight: 500; }}

/* ── Pulse dot ───────────────────────────────────────── */
.hs-pulse {{
    width: 6px; height: 6px;
    border-radius: 50%;
    background: {GREEN};
    display: inline-block;
    box-shadow: 0 0 8px rgba(34,197,94,0.8);
    margin-right: 0.35rem;
    animation: pulse-anim 2s ease-in-out infinite;
}}

@keyframes pulse-anim {{
    0%, 100% {{ box-shadow: 0 0 6px rgba(34,197,94,0.7); }}
    50%       {{ box-shadow: 0 0 14px rgba(34,197,94,1.0); }}
}}

/* ── Severity pills ──────────────────────────────────── */
.sev-normal   {{ background:rgba(100,116,139,0.15); color:#94A3B8; border:1px solid rgba(100,116,139,0.3); border-radius:6px; padding:0.2rem 0.55rem; font-size:0.68rem; font-weight:600; text-transform:uppercase; letter-spacing:0.06em; }}
.sev-low      {{ background:rgba(34,197,94,0.12);  color:#4ADE80; border:1px solid rgba(34,197,94,0.3);  border-radius:6px; padding:0.2rem 0.55rem; font-size:0.68rem; font-weight:600; text-transform:uppercase; letter-spacing:0.06em; }}
.sev-medium   {{ background:rgba(59,130,246,0.12); color:#60A5FA; border:1px solid rgba(59,130,246,0.3); border-radius:6px; padding:0.2rem 0.55rem; font-size:0.68rem; font-weight:600; text-transform:uppercase; letter-spacing:0.06em; }}
.sev-high     {{ background:rgba(249,115,22,0.12); color:#FB923C; border:1px solid rgba(249,115,22,0.3); border-radius:6px; padding:0.2rem 0.55rem; font-size:0.68rem; font-weight:600; text-transform:uppercase; letter-spacing:0.06em; }}
.sev-critical {{ background:rgba(239,68,68,0.12);  color:#F87171; border:1px solid rgba(239,68,68,0.3);  border-radius:6px; padding:0.2rem 0.55rem; font-size:0.68rem; font-weight:600; text-transform:uppercase; letter-spacing:0.06em; }}

/* ── Top-bar ─────────────────────────────────────────── */
.hs-topbar {{
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    padding: 1.25rem 0 1rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 1.25rem;
}}

.hs-topbar-title {{
    font-size: 1.05rem;
    font-weight: 700;
    color: {TEXT_P};
    letter-spacing: -0.025em;
    line-height: 1.2;
}}

.hs-topbar-sub {{
    font-size: 0.72rem;
    color: {TEXT_T};
    margin-top: 0.18rem;
    font-weight: 400;
}}

.hs-topbar-ts {{
    font-size: 0.66rem;
    color: {TEXT_T};
    font-weight: 400;
    font-variant-numeric: tabular-nums;
}}

/* ── Hero ─────────────────────────────────────────────── */
.hs-hero {{
    background: linear-gradient(135deg, #0D1226 0%, #0F1B3D 40%, #0D1226 100%);
    border: 1px solid rgba(59,130,246,0.18);
    border-radius: 16px;
    padding: 2.25rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 0 60px rgba(59,130,246,0.06), 0 8px 40px rgba(0,0,0,0.4);
}}

.hs-hero::before {{
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(59,130,246,0.1) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}}

.hs-hero::after {{
    content: '';
    position: absolute;
    bottom: -80px; left: -40px;
    width: 300px; height: 200px;
    background: radial-gradient(circle, rgba(34,197,94,0.06) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}}

.hs-hero-logo {{
    font-size: 2.2rem;
    font-weight: 800;
    color: {TEXT_P};
    letter-spacing: -0.04em;
    line-height: 1;
    margin-bottom: 0.5rem;
}}

.hs-hero-logo span {{
    background: linear-gradient(135deg, #60A5FA, #34D399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}

.hs-hero-sub {{
    font-size: 0.88rem;
    color: #4B6A9B;
    font-weight: 400;
    line-height: 1.6;
    max-width: 540px;
}}

.hs-hero-chips {{
    display: flex;
    gap: 0.5rem;
    margin-top: 1.1rem;
    flex-wrap: wrap;
}}

.hs-chip {{
    font-size: 0.65rem;
    font-weight: 600;
    padding: 0.25rem 0.6rem;
    border-radius: 20px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    border: 1px solid;
}}

.chip-blue  {{ background:rgba(59,130,246,0.1);  color:#60A5FA;  border-color:rgba(59,130,246,0.2);  }}
.chip-green {{ background:rgba(34,197,94,0.1);   color:#4ADE80;  border-color:rgba(34,197,94,0.2);   }}
.chip-amber {{ background:rgba(245,158,11,0.1);  color:#FCD34D;  border-color:rgba(245,158,11,0.2);  }}

/* ── Capability item ─────────────────────────────────── */
.cap-item {{
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.65rem 0.85rem;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.04);
    border-radius: 10px;
    margin-bottom: 0.5rem;
    transition: border-color 0.15s, background 0.15s;
}}

.cap-item:hover {{
    background: rgba(59,130,246,0.05);
    border-color: rgba(59,130,246,0.15);
}}

.cap-dot {{
    width: 6px; height: 6px; border-radius: 50%;
    background: {BLUE}; flex-shrink: 0;
    box-shadow: 0 0 6px rgba(59,130,246,0.6);
}}

.cap-text {{ font-size: 0.8rem; color: #94A3B8; font-weight: 500; }}

/* ── Upload zone ─────────────────────────────────────── */
[data-testid="stFileUploader"] {{
    background: rgba(13,15,28,0.8) !important;
    border: 1px dashed rgba(59,130,246,0.2) !important;
    border-radius: 14px !important;
    transition: border-color 0.2s ease !important;
}}

[data-testid="stFileUploader"]:hover {{
    border-color: rgba(59,130,246,0.4) !important;
}}

[data-testid="stFileUploader"] section {{ padding: 1.5rem !important; }}

[data-testid="stFileUploader"] label {{
    color: {TEXT_S} !important;
    font-size: 0.8rem !important;
}}

/* ── Detection result ────────────────────────────────── */
.hs-detect {{
    background: linear-gradient(135deg, rgba(13,15,28,0.95), rgba(15,20,45,0.95));
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: 14px;
    padding: 1.25rem 1.4rem;
    margin-bottom: 0.85rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 32px rgba(0,0,0,0.3);
}}

.hs-detect::before {{
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    border-radius: 14px 0 0 14px;
}}

.detect-label-text {{
    font-size: 0.6rem;
    font-weight: 700;
    color: {TEXT_T};
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.5rem;
}}

.detect-name {{
    font-size: 1.6rem;
    font-weight: 800;
    letter-spacing: -0.035em;
    line-height: 1.15;
    margin-bottom: 0.5rem;
}}

.detect-meta {{
    display: flex;
    align-items: center;
    gap: 0.85rem;
    font-size: 0.75rem;
    color: {TEXT_S};
    flex-wrap: wrap;
}}

.detect-conf {{ color: #4ADE80; font-weight: 700; font-size: 0.8rem; }}

/* ── Progress bar custom ─────────────────────────────── */
.hs-bar-wrap {{
    margin-bottom: 0.6rem;
}}

.hs-bar-hdr {{
    display: flex;
    justify-content: space-between;
    font-size: 0.72rem;
    color: {TEXT_S};
    margin-bottom: 0.3rem;
    font-weight: 500;
}}

.hs-bar-track {{
    height: 4px;
    background: rgba(255,255,255,0.06);
    border-radius: 4px;
    overflow: hidden;
}}

.hs-bar-fill {{
    height: 100%;
    border-radius: 4px;
    transition: width 0.4s ease;
}}

/* ── Image ───────────────────────────────────────────── */
[data-testid="stImage"] img {{
    border-radius: 10px !important;
    border: 1px solid {BORDER} !important;
}}

/* ── Buttons ─────────────────────────────────────────── */
.stButton > button,
[data-testid="stDownloadButton"] > button {{
    background: rgba(59,130,246,0.12) !important;
    color: #60A5FA !important;
    border: 1px solid rgba(59,130,246,0.25) !important;
    border-radius: 8px !important;
    padding: 0.45rem 1rem !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    transition: all 0.15s ease !important;
    box-shadow: none !important;
    font-family: {FONT} !important;
    letter-spacing: 0.01em;
}}

.stButton > button:hover,
[data-testid="stDownloadButton"] > button:hover {{
    background: rgba(59,130,246,0.2) !important;
    border-color: rgba(59,130,246,0.4) !important;
    box-shadow: 0 4px 16px rgba(59,130,246,0.15) !important;
}}

/* ── Dataframe ───────────────────────────────────────── */
[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
    overflow: hidden;
}}

/* ── Sidebar brand ───────────────────────────────────── */
.sb-brand {{
    display: flex; align-items: center; gap: 0.6rem;
    padding-bottom: 1.25rem;
    margin-bottom: 0.75rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}}

.sb-icon {{
    width: 28px; height: 28px;
    background: linear-gradient(135deg, {AMBER}, {RED});
    border-radius: 7px;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; font-weight: 800; color: #000;
    flex-shrink: 0;
    box-shadow: 0 4px 12px rgba(245,158,11,0.3);
}}

.sb-name {{
    font-size: 0.9rem; font-weight: 700;
    color: {TEXT_P}; letter-spacing: -0.02em;
}}

.sb-ver {{
    font-size: 0.6rem;
    color: {TEXT_T};
    font-weight: 400;
    margin-top: 0.05rem;
}}

.sb-section {{
    font-size: 0.58rem;
    font-weight: 700;
    color: {TEXT_T};
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin: 1.1rem 0 0.4rem 0.25rem;
}}

/* ── Info box ────────────────────────────────────────── */
[data-testid="stInfo"] {{
    background: rgba(59,130,246,0.08) !important;
    border: 1px solid rgba(59,130,246,0.2) !important;
    border-radius: 10px !important;
    color: #93C5FD !important;
    font-size: 0.8rem !important;
}}

/* ── Section divider ─────────────────────────────────── */
.hs-divider {{
    height: 1px;
    background: rgba(255,255,255,0.05);
    margin: 1.25rem 0;
}}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# DEFECT CONFIG
# ─────────────────────────────────────────────────────────────

DEFECT_COLORS = {
    "Cell":           "#F59E0B",
    "Cell-Multi":     "#F97316",
    "Cracking":       "#EF4444",
    "Diode":          "#8B5CF6",
    "Diode-Multi":    "#7C3AED",
    "Hot-Spot":       "#DC2626",
    "Hot-Spot-Multi": "#B91C1C",
    "No-Anomaly":     "#22C55E",
    "Offline-Module": "#6B7280",
    "Shadowing":      "#3B82F6",
    "Soiling":        "#CA8A04",
    "Vegetation":     "#16A34A",
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
    "Cell":           "Monitor affected cell region — flag for next service cycle.",
    "Cell-Multi":     "Replace affected cell string. Schedule within 2 weeks.",
    "Cracking":       "Replace cracked module. Take offline if structural risk present.",
    "Diode":          "Inspect bypass diode circuitry. Electrical team required.",
    "Diode-Multi":    "Urgent: multiple diode failures detected. Dispatch maintenance.",
    "Hot-Spot":       "Immediate thermal inspection required. Isolate panel from array.",
    "Hot-Spot-Multi": "Critical overheating. Take offline immediately.",
    "Offline-Module": "Module disconnected from array. Reconnect or replace.",
}

SEV_CLASS = {
    "Normal":   "sev-normal",
    "Low":      "sev-low",
    "Medium":   "sev-medium",
    "High":     "sev-high",
    "Critical": "sev-critical",
}

SEV_DETECT_COLOR = {
    "Normal":   "#4ADE80",
    "Low":      "#4ADE80",
    "Medium":   "#60A5FA",
    "High":     "#FB923C",
    "Critical": "#F87171",
}

CLASS_COUNTS = {
    "No-Anomaly": 10000, "Cell": 1877, "Vegetation": 1639, "Diode": 1499,
    "Cell-Multi": 1288,  "Shadowing": 1056, "Cracking": 940,
    "Offline-Module": 827, "Hot-Spot": 249, "Hot-Spot-Multi": 246,
    "Soiling": 204, "Diode-Multi": 175,
}

SEV_COUNTS  = {"Normal": 10000, "Low": 204, "Medium": 3734, "High": 3139, "Critical": 1318}
SEV_COLORS  = {"Normal": "#1E293B","Low": "#14532D","Medium": "#1E3A5F","High": "#7C2D12","Critical": "#7F1D1D"}
SEV_TXTCOL  = {"Normal": "#475569","Low": "#4ADE80","Medium": "#60A5FA","High": "#FB923C","Critical": "#F87171"}

# ─────────────────────────────────────────────────────────────
# PLOTLY HELPERS
# ─────────────────────────────────────────────────────────────

_PLOT_BASE = dict(
    plot_bgcolor  = "rgba(0,0,0,0)",
    paper_bgcolor = "rgba(0,0,0,0)",
    font = dict(family="Inter, system-ui, sans-serif", size=11, color="#475569"),
    hovermode = "closest",
    showlegend = False,
    margin = dict(l=0, r=12, t=8, b=0),
)

_XAXIS_H = dict(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.04)",
                zeroline=False, showline=False, showticklabels=False)
_YAXIS_H = dict(showgrid=False, zeroline=False, showline=False,
                tickfont=dict(size=11, color="#64748B"))
_XAXIS_V = dict(showgrid=False, zeroline=False, showline=False,
                tickfont=dict(size=10, color="#475569"))
_YAXIS_V = dict(showgrid=False, zeroline=False, showline=False, showticklabels=False)

def plotly_layout(**kw):
    d = dict(**_PLOT_BASE)
    d.update(kw)
    return d

def hex_rgba(h, a):
    r, g, b = int(h[1:3],16), int(h[3:5],16), int(h[5:7],16)
    return f"rgba({r},{g},{b},{a})"

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

with st.sidebar:

    st.markdown("""
<div class="sb-brand">
    <div class="sb-icon">☀</div>
    <div>
        <div class="sb-name">HelioSight</div>
        <div class="sb-ver">AI Platform · v2.0</div>
    </div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="sb-section">Navigation</div>', unsafe_allow_html=True)

    page = st.radio(
        "nav",
        ["Dashboard", "Thermal Inspection", "Model Insights", "Reports"],
        label_visibility="collapsed",
    )

    st.markdown('<div class="hs-divider" style="margin:1.25rem 0;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-section">System</div>', unsafe_allow_html=True)

    st.markdown(f"""
<div style="margin-top:0.4rem;">
    <div class="hs-row">
        <span class="hs-row-k">Status</span>
        <span class="hs-row-v"><span class="hs-pulse"></span>Online</span>
    </div>
    <div class="hs-row">
        <span class="hs-row-k">Model</span>
        <span class="hs-row-v">{meta['model']}</span>
    </div>
    <div class="hs-row">
        <span class="hs-row-k">Accuracy</span>
        <span class="hs-row-v" style="color:#4ADE80;">{meta['test_accuracy']*100:.1f}%</span>
    </div>
    <div class="hs-row">
        <span class="hs-row-k">Classes</span>
        <span class="hs-row-v">{len(CLASSES)}</span>
    </div>
    <div class="hs-row">
        <span class="hs-row-k">Dataset</span>
        <span class="hs-row-v">20,000</span>
    </div>
    <div class="hs-row">
        <span class="hs-row-k">Input</span>
        <span class="hs-row-v">128 × 128</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TOPBAR HELPER
# ─────────────────────────────────────────────────────────────

def topbar(title, subtitle, tag_html=""):
    st.markdown(f"""
<div class="hs-topbar">
    <div>
        <div class="hs-topbar-title">{title}</div>
        <div class="hs-topbar-sub">{subtitle}</div>
    </div>
    <div style="display:flex;align-items:center;gap:0.6rem;">
        {tag_html}
        <span class="hs-topbar-ts">{datetime.now().strftime('%H:%M · %d %b %Y')}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ═════════════════════════════════════════════════════════════

if page == "Dashboard":

    # ── Hero ─────────────────────────────────────────────────
    st.markdown(f"""
<div class="hs-hero">
    <div class="hs-hero-logo">Helio<span>Sight</span> AI</div>
    <div class="hs-hero-sub">
        AI-powered solar panel defect intelligence platform for drone thermal inspection workflows —
        real-time classification across 12 defect categories.
    </div>
    <div class="hs-hero-chips">
        <span class="hs-chip chip-blue">MobileNetV2</span>
        <span class="hs-chip chip-green">{meta['test_accuracy']*100:.1f}% Accuracy</span>
        <span class="hs-chip chip-amber">20K Images</span>
        <span class="hs-chip chip-blue">12 Classes</span>
        <span class="hs-chip chip-green">Live Inference</span>
    </div>
</div>
""", unsafe_allow_html=True)

    # ── KPI row ───────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5, gap="small")
    k1.metric("Total Images", "20,000")
    k2.metric("Defect Classes", "12")
    k3.metric("Model Accuracy", f"{meta['test_accuracy']*100:.1f}%")
    k4.metric("Architecture", meta["model"])
    k5.metric("Resolution", "128 × 128")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── Main grid: chart + right panels ──────────────────────
    col_l, col_r = st.columns([1.75, 1], gap="medium")

    with col_l:
        st.markdown('<div class="hs-card">', unsafe_allow_html=True)
        st.markdown('<div class="hs-card-title">Defect Distribution <span class="hs-badge badge-slate">12 classes · 20K samples</span></div>', unsafe_allow_html=True)

        sorted_items = sorted(CLASS_COUNTS.items(), key=lambda x: x[1])
        labels  = [k for k, v in sorted_items]
        values  = [v for k, v in sorted_items]
        colors  = [hex_rgba(DEFECT_COLORS.get(k, "#6B7280"), 0.85) for k in labels]

        fig_bar = go.Figure(go.Bar(
            y=labels, x=values, orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            width=0.65,
            text=[f"{v:,}" for v in values],
            textposition="outside",
            textfont=dict(size=9.5, color="#475569", family="Inter"),
            hovertemplate="<b>%{y}</b><br>Count: %{x:,}<extra></extra>",
            cliponaxis=False,
        ))

        fig_bar.update_layout(**plotly_layout(
            height=370,
            xaxis=dict(**_XAXIS_H, range=[0, max(values)*1.2]),
            yaxis=dict(**_YAXIS_H),
        ))

        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        # Severity breakdown
        st.markdown('<div class="hs-card" style="margin-bottom:0.75rem;">', unsafe_allow_html=True)
        st.markdown('<div class="hs-card-title">Severity Breakdown</div>', unsafe_allow_html=True)

        fig_sev = go.Figure(go.Bar(
            x=list(SEV_COUNTS.keys()),
            y=list(SEV_COUNTS.values()),
            marker=dict(
                color=[SEV_COLORS[k] for k in SEV_COUNTS],
                line=dict(color=[SEV_TXTCOL[k] for k in SEV_COUNTS], width=1),
            ),
            width=0.52,
            text=[f"{v:,}" for v in SEV_COUNTS.values()],
            textposition="outside",
            textfont=dict(size=9, color=[SEV_TXTCOL[k] for k in SEV_COUNTS]),
            hovertemplate="<b>%{x}</b>: %{y:,}<extra></extra>",
        ))

        fig_sev.update_layout(**plotly_layout(
            height=205,
            margin=dict(l=0, r=0, t=6, b=0),
            xaxis=dict(**_XAXIS_V),
            yaxis=dict(**_YAXIS_V, range=[0, max(SEV_COUNTS.values())*1.28]),
        ))

        st.plotly_chart(fig_sev, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

        # Capabilities list
        st.markdown('<div class="hs-card">', unsafe_allow_html=True)
        st.markdown('<div class="hs-card-title">AI Capabilities</div>', unsafe_allow_html=True)

        caps = [
            "Real thermal drone imagery",
            "MobileNetV2 transfer learning",
            "12 solar defect categories",
            "Real-time CNN inference",
            "Confidence score output",
            "Renewable energy analytics",
        ]

        for cap in caps:
            st.markdown(f"""
<div class="cap-item">
    <div class="cap-dot"></div>
    <span class="cap-text">{cap}</span>
</div>
""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
# PAGE 2 — THERMAL INSPECTION
# ═════════════════════════════════════════════════════════════

elif page == "Thermal Inspection":

    topbar(
        "Thermal Inspection",
        "Upload drone-captured infrared imagery for real-time AI defect classification",
        '<span class="hs-badge badge-green"><span class="hs-pulse" style="width:4px;height:4px;margin-right:0.3rem;"></span>Model Active</span>'
    )

    uploaded_file = st.file_uploader(
        "Drop thermal image here — JPG / JPEG / PNG",
        type=["jpg", "jpeg", "png"],
        label_visibility="visible",
    )

    if uploaded_file is not None:

        col_img, col_out = st.columns([1, 1.1], gap="medium")

        with col_img:
            st.markdown('<div class="hs-card">', unsafe_allow_html=True)
            st.markdown('<div class="hs-card-title">Input Image</div>', unsafe_allow_html=True)
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
            w, h = image.size
            st.markdown(f"""
<div style="display:flex;gap:1.5rem;margin-top:0.75rem;padding-top:0.65rem;border-top:1px solid rgba(255,255,255,0.05);">
    <div><div style="font-size:0.6rem;color:#334155;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.15rem;">Width</div><div style="font-size:0.78rem;color:#94A3B8;font-weight:500;">{w}px</div></div>
    <div><div style="font-size:0.6rem;color:#334155;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.15rem;">Height</div><div style="font-size:0.78rem;color:#94A3B8;font-weight:500;">{h}px</div></div>
    <div><div style="font-size:0.6rem;color:#334155;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.15rem;">Mode</div><div style="font-size:0.78rem;color:#94A3B8;font-weight:500;">{image.mode}</div></div>
    <div><div style="font-size:0.6rem;color:#334155;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.15rem;">File</div><div style="font-size:0.78rem;color:#94A3B8;font-weight:500;">{uploaded_file.name[:16]}</div></div>
</div>
""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Run inference ─────────────────────────────────────
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file.getvalue())
            temp_path = tmp.name

        result          = predict_image(temp_path)
        os.remove(temp_path)

        predicted_label = result["label"]
        confidence      = result["confidence"]
        severity        = SEVERITY_MAP.get(predicted_label, "Review")
        action          = ACTION_MAP.get(predicted_label, "Review inspection results.")
        sev_cls         = SEV_CLASS.get(severity, "sev-normal")
        sev_color       = SEV_DETECT_COLOR.get(severity, "#60A5FA")
        defect_color    = DEFECT_COLORS.get(predicted_label, "#888")

        with col_out:

            # Result header
            st.markdown(f"""
<div class="hs-detect">
    <div class="detect-label-text">Detection Result</div>
    <div class="detect-name" style="color:{defect_color};">{predicted_label}</div>
    <div class="detect-meta">
        <span class="detect-conf">{confidence*100:.1f}% confidence</span>
        <span>·</span>
        <span class="{sev_cls}">{severity}</span>
    </div>
</div>
""", unsafe_allow_html=True)

            # Recommended action
            st.markdown(f"""
<div class="hs-card" style="margin-bottom:0.75rem;">
    <div class="hs-card-title">Recommended Action</div>
    <p style="font-size:0.8rem;color:#94A3B8;line-height:1.7;margin:0;">{action}</p>
</div>
""", unsafe_allow_html=True)

            # Confidence breakdown — Plotly horizontal bar
            st.markdown('<div class="hs-card">', unsafe_allow_html=True)
            st.markdown('<div class="hs-card-title">Class Probabilities</div>', unsafe_allow_html=True)

            prob_df = pd.DataFrame({
                "Class":       list(result["probabilities"].keys()),
                "Probability": list(result["probabilities"].values()),
            }).sort_values("Probability", ascending=True)

            bar_colors_p = [
                hex_rgba(DEFECT_COLORS.get(c, "#6B7280"), 0.85 if c == predicted_label else 0.2)
                for c in prob_df["Class"]
            ]

            fig_prob = go.Figure(go.Bar(
                y=prob_df["Class"],
                x=prob_df["Probability"],
                orientation="h",
                marker=dict(color=bar_colors_p, line=dict(width=0)),
                width=0.6,
                text=[f"{v:.1%}" if v >= 0.01 else "" for v in prob_df["Probability"]],
                textposition="outside",
                textfont=dict(size=9, color="#475569"),
                hovertemplate="<b>%{y}</b>: %{x:.2%}<extra></extra>",
                cliponaxis=False,
            ))

            fig_prob.update_layout(**plotly_layout(
                height=max(250, len(prob_df) * 23),
                xaxis=dict(**_XAXIS_H, range=[0, 1.22]),
                yaxis=dict(**_YAXIS_H),
            ))

            st.plotly_chart(fig_prob, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.markdown("""
<div style="
    margin-top:1.25rem;
    border:1px dashed rgba(59,130,246,0.15);
    border-radius:14px;
    padding:4rem 2rem;
    text-align:center;
    background:rgba(59,130,246,0.02);
">
    <div style="font-size:2rem;margin-bottom:0.75rem;opacity:0.2;">⬡</div>
    <div style="font-size:0.82rem;color:#334155;line-height:1.9;">
        Upload a thermal drone image to begin AI defect classification<br>
        <span style="font-size:0.7rem;color:#1E293B;">JPG · JPEG · PNG supported</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
# PAGE 3 — MODEL INSIGHTS
# ═════════════════════════════════════════════════════════════

elif page == "Model Insights":

    topbar("Model Insights",
           "Training analytics, architecture specification, and performance metrics")

    m1, m2, m3, m4 = st.columns(4, gap="small")
    m1.metric("Architecture", meta["model"])
    m2.metric("Input Shape", "128 , 128 , 3")
    m3.metric("Output Classes", len(CLASSES))
    m4.metric("Test Accuracy", f"{meta['test_accuracy']*100:.2f}%")

    st.markdown("<div style='height:0.9rem'></div>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2, gap="medium")

    with col_l:
        st.markdown('<div class="hs-card">', unsafe_allow_html=True)
        st.markdown('<div class="hs-card-title">Training Accuracy</div>', unsafe_allow_html=True)
        st.image("models/training_accuracy.png", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="hs-card">', unsafe_allow_html=True)
        st.markdown('<div class="hs-card-title">Training Loss</div>', unsafe_allow_html=True)
        st.image("models/training_loss.png", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:0.35rem'></div>", unsafe_allow_html=True)

    st.markdown(f"""
<div class="hs-card">
    <div class="hs-card-title">Architecture Overview</div>
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.75rem 1.5rem;">
        <div>
            <div style="font-size:0.62rem;color:#334155;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.2rem;">Base Model</div>
            <div style="font-size:0.85rem;color:#CBD5E1;font-weight:500;">{meta['model']}</div>
        </div>
        <div>
            <div style="font-size:0.62rem;color:#334155;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.2rem;">Input Shape</div>
            <div style="font-size:0.85rem;color:#CBD5E1;font-weight:500;">128 × 128 × 3</div>
        </div>
        <div>
            <div style="font-size:0.62rem;color:#334155;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.2rem;">Classifier</div>
            <div style="font-size:0.85rem;color:#CBD5E1;font-weight:500;">Dense + Softmax</div>
        </div>
        <div>
            <div style="font-size:0.62rem;color:#334155;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.2rem;">Output</div>
            <div style="font-size:0.85rem;color:#CBD5E1;font-weight:500;">{len(CLASSES)} Classes</div>
        </div>
        <div>
            <div style="font-size:0.62rem;color:#334155;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.2rem;">Preprocessing</div>
            <div style="font-size:0.85rem;color:#CBD5E1;font-weight:500;">ImageNet Norm.</div>
        </div>
        <div>
            <div style="font-size:0.62rem;color:#334155;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.2rem;">Augmentation</div>
            <div style="font-size:0.85rem;color:#CBD5E1;font-weight:500;">Flip · Rotate · Zoom</div>
        </div>
        <div>
            <div style="font-size:0.62rem;color:#334155;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.2rem;">Dataset</div>
            <div style="font-size:0.85rem;color:#CBD5E1;font-weight:500;">20,000 Samples</div>
        </div>
        <div>
            <div style="font-size:0.62rem;color:#334155;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.2rem;">Test Accuracy</div>
            <div style="font-size:0.85rem;font-weight:600;color:#4ADE80;">{meta['test_accuracy']*100:.2f}%</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
# PAGE 4 — REPORTS
# ═════════════════════════════════════════════════════════════

elif page == "Reports":

    topbar("Inspection Reports",
           "Aggregated analytics, model performance summary, and CSV export")

    r_l, r_r = st.columns([1.35, 1], gap="medium")

    with r_l:
        st.markdown('<div class="hs-card">', unsafe_allow_html=True)
        st.markdown('<div class="hs-card-title">Performance Summary</div>', unsafe_allow_html=True)

        report_data = {
            "Metric": [
                "Total Inspected Images", "Defect Categories", "Model Architecture",
                "Test Accuracy", "Critical Defect Types",
                "Inspection Method", "Image Source", "Report Generated",
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
        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

        csv = report_df.to_csv(index=False)
        st.download_button(
            "↓  Export CSV Report",
            csv,
            file_name=f"heliosight_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

    with r_r:
        st.markdown('<div class="hs-card">', unsafe_allow_html=True)
        st.markdown('<div class="hs-card-title">Defect Priority Index <span class="hs-badge badge-slate">12 classes</span></div>', unsafe_allow_html=True)

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
            sev_cls = SEV_CLASS.get(sev, "sev-normal")
            st.markdown(f"""
<div class="hs-row">
    <span class="hs-row-k">{name}</span>
    <div style="display:flex;align-items:center;gap:0.75rem;">
        <span style="font-size:0.68rem;color:#334155;font-variant-numeric:tabular-nums;">{count:,}</span>
        <span class="{sev_cls}">{sev}</span>
    </div>
</div>
""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(f"""
<div class="hs-card" style="margin-top:0.75rem;">
    <div class="hs-card-title">Report Notes</div>
    <p style="font-size:0.75rem;color:#334155;line-height:1.8;margin:0;">
        Metrics reflect model performance across all 20,000 training samples.
        Upload individual thermal images on the Inspection page for per-panel results.
        <br><br>
        <span style="color:#1E3A5F;">Generated: {datetime.now().strftime('%d %b %Y, %H:%M UTC')}</span>
    </p>
</div>
""", unsafe_allow_html=True)