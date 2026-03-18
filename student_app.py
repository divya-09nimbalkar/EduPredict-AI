import io
import os
import re
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import bcrypt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from fpdf import FPDF
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings("ignore")


APP_TITLE   = "EduPredict AI"
APP_TAGLINE = "Student Intelligence Platform"
DB_PATH     = os.path.join(os.path.dirname(__file__), "app_users.db")
EMAIL_RE    = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


# ═══════════════════════════════════════════════════════════════
#  DATA CLASSES
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class AuthUser:
    user_id: int
    name: str
    email: str


@dataclass
class TrainArtifacts:
    pipeline: Pipeline
    feature_columns: List[str]
    target_column: str
    classes_: List[str]
    model_name: str
    metrics: Dict[str, float]


# ═══════════════════════════════════════════════════════════════
#  PREMIUM CSS INJECTION
# ═══════════════════════════════════════════════════════════════

def _inject_css() -> None:
    st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
/* ── Reset & Base ───────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

#MainMenu, footer { visibility: hidden; }
header { visibility: visible; }

html, body {
  font-family: 'DM Sans', sans-serif;
  background: #050810;
  color: #e2e8f4;
}

/* ── Page canvas ───────────────────────────────────────────── */
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(ellipse 1400px 700px at 0% -10%, rgba(14,165,233,0.13) 0%, transparent 60%),
    radial-gradient(ellipse 1000px 600px at 100% 15%, rgba(99,102,241,0.12) 0%, transparent 55%),
    radial-gradient(ellipse 800px 500px at 50% 100%, rgba(16,185,129,0.07) 0%, transparent 55%),
    linear-gradient(180deg, #050810 0%, #080c18 100%) !important;
  min-height: 100vh;
}

.block-container {
  padding-top: 2rem !important;
  padding-bottom: 3rem !important;
  max-width: 1320px !important;
}

/* ── Sidebar ────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: rgba(5,8,16,0.92) !important;
  border-right: 1px solid rgba(99,102,241,0.15) !important;
  backdrop-filter: blur(20px);
}
[data-testid="stSidebar"] > div { padding-top: 1.5rem; }
[data-testid="stSidebar"] * { color: rgba(226,232,244,0.90) !important; }
[data-testid="stSidebar"] .stMarkdown h3 {
  font-family: 'Syne', sans-serif !important;
  font-size: 11px !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
  color: rgba(99,102,241,0.85) !important;
  margin-top: 1.4rem !important;
}

/* ── Typography ────────────────────────────────────────────── */
h1, h2, h3, h4, .heroTitle { font-family: 'Syne', sans-serif !important; }
p, label, div, span { font-family: 'DM Sans', sans-serif; }
code, pre, .mono { font-family: 'JetBrains Mono', monospace !important; }

/* ── Tabs ───────────────────────────────────────────────────── */
div[data-baseweb="tab-list"] {
  gap: 4px !important;
  background: rgba(5,8,16,0.6) !important;
  border: 1px solid rgba(99,102,241,0.15) !important;
  border-radius: 14px !important;
  padding: 4px !important;
  backdrop-filter: blur(10px);
}
button[role="tab"] {
  border-radius: 10px !important;
  border: none !important;
  background: transparent !important;
  color: rgba(148,163,200,0.75) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  padding: 8px 16px !important;
  transition: all 0.2s ease !important;
}
button[role="tab"]:hover {
  color: rgba(226,232,244,0.95) !important;
  background: rgba(99,102,241,0.12) !important;
}
button[role="tab"][aria-selected="true"] {
  background: linear-gradient(135deg, rgba(99,102,241,0.35), rgba(14,165,233,0.25)) !important;
  color: #ffffff !important;
  font-weight: 600 !important;
  box-shadow: 0 2px 12px rgba(99,102,241,0.20) !important;
}

/* ── Buttons ────────────────────────────────────────────────── */
div.stButton > button {
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 600 !important;
  font-size: 13px !important;
  letter-spacing: 0.02em !important;
  border-radius: 10px !important;
  background: linear-gradient(135deg, #6366f1, #0ea5e9) !important;
  color: #ffffff !important;
  border: none !important;
  padding: 10px 20px !important;
  box-shadow: 0 4px 20px rgba(99,102,241,0.30) !important;
  transition: all 0.25s ease !important;
}
div.stButton > button:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 8px 28px rgba(99,102,241,0.40) !important;
}
div.stButton > button:active { transform: translateY(0px) !important; }
div.stButton > button:disabled {
  opacity: 0.45 !important;
  transform: none !important;
  cursor: not-allowed !important;
}
/* Danger/secondary variant */
div.stButton > button[kind="secondary"] {
  background: rgba(99,102,241,0.12) !important;
  border: 1px solid rgba(99,102,241,0.25) !important;
  color: rgba(226,232,244,0.85) !important;
  box-shadow: none !important;
}

/* ── Inputs ──────────────────────────────────────────────────── */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div,
div[data-baseweb="textarea"] > div {
  border-radius: 10px !important;
  background: rgba(8,12,24,0.70) !important;
  border: 1px solid rgba(99,102,241,0.18) !important;
  transition: border-color 0.2s ease !important;
}
div[data-baseweb="input"] > div:focus-within,
div[data-baseweb="select"] > div:focus-within {
  border-color: rgba(99,102,241,0.55) !important;
  box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
}
div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea {
  color: rgba(226,232,244,0.95) !important;
  font-family: 'DM Sans', sans-serif !important;
}
div[data-baseweb="select"] span { color: rgba(226,232,244,0.92) !important; }

/* ── Metrics ─────────────────────────────────────────────────── */
[data-testid="stMetricValue"] {
  font-family: 'Syne', sans-serif !important;
  font-size: 28px !important;
  font-weight: 800 !important;
  color: #ffffff !important;
}
[data-testid="stMetricLabel"] {
  font-family: 'DM Sans', sans-serif !important;
  color: rgba(148,163,200,0.80) !important;
  font-size: 12px !important;
  text-transform: uppercase !important;
  letter-spacing: 0.08em !important;
}
[data-testid="metric-container"] {
  background: rgba(8,12,24,0.65) !important;
  border: 1px solid rgba(99,102,241,0.18) !important;
  border-radius: 16px !important;
  padding: 18px 20px !important;
  backdrop-filter: blur(10px) !important;
}

/* ── Dataframe ───────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
  border-radius: 14px !important;
  overflow: hidden !important;
  border: 1px solid rgba(99,102,241,0.15) !important;
}

/* ── Alerts / callouts ──────────────────────────────────────── */
[data-testid="stAlert"] {
  border-radius: 12px !important;
  border-left-width: 3px !important;
  font-family: 'DM Sans', sans-serif !important;
}

/* ── Sliders ────────────────────────────────────────────────── */
div[data-baseweb="slider"] [data-testid="stSlider"] div { color: rgba(226,232,244,0.85) !important; }

/* ── Spinner ─────────────────────────────────────────────────── */
[data-testid="stSpinner"] { color: #6366f1 !important; }

/* ── Download buttons — fully dark-themed ───────────────────── */
div.stDownloadButton > button {
  font-family: 'DM Sans', sans-serif !important;
  font-size: 13px !important;
  font-weight: 600 !important;
  letter-spacing: 0.01em !important;
  border-radius: 12px !important;
  padding: 12px 16px !important;
  width: 100% !important;
  text-align: left !important;
  box-shadow: none !important;
  transition: all 0.22s ease !important;
  background: rgba(14,165,233,0.12) !important;
  border: 1px solid rgba(14,165,233,0.35) !important;
  color: #7dd3fc !important;
}
div.stDownloadButton > button:hover {
  background: rgba(14,165,233,0.24) !important;
  border-color: rgba(14,165,233,0.65) !important;
  color: #ffffff !important;
  box-shadow: 0 4px 22px rgba(14,165,233,0.25) !important;
  transform: translateX(4px) !important;
}
div.stDownloadButton > button p,
div.stDownloadButton > button span {
  color: inherit !important;
}
/* 2nd download = violet */
div.stDownloadButton + div.stDownloadButton > button,
div.stDownloadButton:nth-child(2) > button {
  background: rgba(99,102,241,0.12) !important;
  border-color: rgba(99,102,241,0.35) !important;
  color: #a5b4fc !important;
}
div.stDownloadButton + div.stDownloadButton > button:hover,
div.stDownloadButton:nth-child(2) > button:hover {
  background: rgba(99,102,241,0.24) !important;
  border-color: rgba(99,102,241,0.65) !important;
  color: #ffffff !important;
  box-shadow: 0 4px 22px rgba(99,102,241,0.25) !important;
}
/* 3rd download = emerald */
div.stDownloadButton + div.stDownloadButton + div.stDownloadButton > button,
div.stDownloadButton:nth-child(3) > button {
  background: rgba(16,185,129,0.10) !important;
  border-color: rgba(16,185,129,0.32) !important;
  color: #6ee7b7 !important;
}
div.stDownloadButton + div.stDownloadButton + div.stDownloadButton > button:hover,
div.stDownloadButton:nth-child(3) > button:hover {
  background: rgba(16,185,129,0.22) !important;
  border-color: rgba(16,185,129,0.58) !important;
  color: #ffffff !important;
  box-shadow: 0 4px 22px rgba(16,185,129,0.22) !important;
}

/* ── Sidebar spacing ─────────────────────────────────────────── */
[data-testid="stSidebar"] .stDownloadButton { margin-bottom: 8px !important; }
[data-testid="stSidebar"] .stButton         { margin-bottom: 4px !important; }

/* ── Custom Component Styles ──────────────────────────────────── */

/* Hero landing */
.landing-bg {
  position: relative;
  overflow: hidden;
  border-radius: 24px;
  background: linear-gradient(135deg, rgba(8,12,28,0.95), rgba(5,8,16,0.90));
  border: 1px solid rgba(99,102,241,0.20);
  padding: 52px 48px;
  margin-bottom: 24px;
}
.landing-bg::before {
  content: '';
  position: absolute;
  top: -60px; right: -60px;
  width: 350px; height: 350px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(99,102,241,0.18), transparent 70%);
  pointer-events: none;
}
.landing-bg::after {
  content: '';
  position: absolute;
  bottom: -40px; left: 30%;
  width: 280px; height: 280px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(14,165,233,0.12), transparent 70%);
  pointer-events: none;
}
.hero-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: rgba(99,102,241,0.15);
  border: 1px solid rgba(99,102,241,0.30);
  border-radius: 999px;
  padding: 5px 14px;
  font-size: 12px;
  font-weight: 500;
  color: rgba(165,170,255,0.90);
  letter-spacing: 0.05em;
  margin-bottom: 18px;
}
.hero-title {
  font-family: 'Syne', sans-serif;
  font-size: 52px;
  font-weight: 800;
  line-height: 1.08;
  letter-spacing: -0.025em;
  background: linear-gradient(135deg, #ffffff 30%, rgba(165,170,255,0.90) 60%, rgba(14,165,233,0.85) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin: 0 0 14px 0;
}
.hero-sub {
  font-size: 16px;
  line-height: 1.65;
  color: rgba(172,183,210,0.85);
  max-width: 540px;
  margin: 0 0 28px 0;
}
.feature-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 12px;
  margin-top: 4px;
}
.feature-item {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  background: rgba(99,102,241,0.07);
  border: 1px solid rgba(99,102,241,0.12);
  border-radius: 12px;
  padding: 12px 14px;
  font-size: 13px;
  color: rgba(200,210,235,0.88);
  line-height: 1.4;
}
.feature-icon {
  font-size: 16px;
  margin-top: 1px;
  flex-shrink: 0;
}

/* Auth card */
.auth-card {
  background: rgba(8,12,24,0.80);
  border: 1px solid rgba(99,102,241,0.22);
  border-radius: 20px;
  padding: 32px 28px;
  backdrop-filter: blur(20px);
  box-shadow: 0 20px 60px rgba(0,0,0,0.40), 0 0 0 1px rgba(99,102,241,0.10) inset;
}
.auth-title {
  font-family: 'Syne', sans-serif;
  font-size: 20px;
  font-weight: 700;
  color: #fff;
  margin-bottom: 4px;
}
.auth-sub {
  font-size: 13px;
  color: rgba(148,163,200,0.70);
  margin-bottom: 20px;
}

/* KPI cards */
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
  margin-bottom: 6px;
}
.kpi-card {
  background: rgba(8,12,24,0.70);
  border: 1px solid rgba(99,102,241,0.16);
  border-radius: 18px;
  padding: 20px 20px 16px;
  backdrop-filter: blur(12px);
  position: relative;
  overflow: hidden;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
.kpi-card:hover {
  border-color: rgba(99,102,241,0.35);
  box-shadow: 0 8px 32px rgba(99,102,241,0.12);
}
.kpi-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: var(--kpi-accent, linear-gradient(90deg, #6366f1, #0ea5e9));
  border-radius: 18px 18px 0 0;
}
.kpi-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
.kpi-label {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.10em;
  text-transform: uppercase;
  color: rgba(148,163,200,0.70);
}
.kpi-icon {
  width: 36px; height: 36px;
  border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  font-size: 17px;
  background: rgba(99,102,241,0.15);
  border: 1px solid rgba(99,102,241,0.20);
}
.kpi-value {
  font-family: 'Syne', sans-serif;
  font-size: 30px;
  font-weight: 800;
  letter-spacing: -0.02em;
  color: #fff;
  line-height: 1.1;
  margin-bottom: 4px;
}
.kpi-hint {
  font-size: 12px;
  color: rgba(148,163,200,0.60);
}

/* Section cards */
.section-card {
  background: rgba(8,12,24,0.60);
  border: 1px solid rgba(99,102,241,0.13);
  border-radius: 18px;
  padding: 22px 22px 18px;
  backdrop-filter: blur(8px);
  margin-bottom: 4px;
}
.section-title {
  font-family: 'Syne', sans-serif;
  font-size: 16px;
  font-weight: 700;
  color: #fff;
  margin: 0 0 4px 0;
}
.section-sub {
  font-size: 12px;
  color: rgba(148,163,200,0.65);
  margin: 0 0 14px 0;
}

/* Page header */
.page-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 20px;
  padding-bottom: 18px;
  border-bottom: 1px solid rgba(99,102,241,0.12);
}
.page-title {
  font-family: 'Syne', sans-serif;
  font-size: 26px;
  font-weight: 800;
  letter-spacing: -0.015em;
  color: #fff;
}
.page-sub {
  font-size: 13px;
  color: rgba(148,163,200,0.72);
  margin-top: 2px;
}

/* Pills & badges */
.pill {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 4px 12px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}
.pill-blue  { background: rgba(14,165,233,0.15); border:1px solid rgba(14,165,233,0.28); color:rgba(125,211,252,0.90); }
.pill-violet{ background: rgba(99,102,241,0.15); border:1px solid rgba(99,102,241,0.28); color:rgba(165,170,255,0.90); }
.pill-green { background: rgba(16,185,129,0.12); border:1px solid rgba(16,185,129,0.25); color:rgba(110,231,183,0.90); }
.pill-amber { background: rgba(245,158,11,0.12); border:1px solid rgba(245,158,11,0.25); color:rgba(252,211,77,0.90); }
.pill-red   { background: rgba(239,68,68,0.12);  border:1px solid rgba(239,68,68,0.25);  color:rgba(252,165,165,0.90); }

/* Divider */
.divider {
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(99,102,241,0.30), transparent);
  margin: 18px 0;
}

/* Sidebar profile card */
.profile-card {
  background: rgba(99,102,241,0.10);
  border: 1px solid rgba(99,102,241,0.20);
  border-radius: 14px;
  padding: 14px 16px;
  margin-bottom: 4px;
}
.profile-name { font-family:'Syne',sans-serif; font-weight:700; font-size:15px; color:#fff; }
.profile-email { font-size:12px; color:rgba(148,163,200,0.65); margin-top:2px; }
.profile-badge {
  margin-top: 8px;
  display: inline-flex;
  align-items: center;
  gap: 4px;
  background: rgba(16,185,129,0.12);
  border:1px solid rgba(16,185,129,0.22);
  border-radius: 999px;
  padding: 3px 10px;
  font-size: 11px;
  color: rgba(110,231,183,0.85);
  font-weight: 600;
  letter-spacing: 0.04em;
}

/* Result prediction card */
.result-card {
  background: linear-gradient(135deg, rgba(16,185,129,0.10), rgba(14,165,233,0.07));
  border: 1px solid rgba(16,185,129,0.25);
  border-radius: 18px;
  padding: 28px 28px;
  text-align: center;
}
.result-outcome {
  font-family: 'Syne', sans-serif;
  font-size: 36px;
  font-weight: 800;
  color: #10b981;
  margin: 8px 0 4px;
}
.result-label { font-size:13px; color:rgba(148,163,200,0.70); text-transform:uppercase; letter-spacing:0.08em; }
.result-confidence { font-size:14px; color:rgba(200,210,235,0.80); margin-top: 6px; }

/* Train step indicator */
.step-row { display:flex; gap:8px; align-items:center; margin-bottom:18px; }
.step { display:flex; align-items:center; gap:8px; padding:10px 16px; border-radius:12px; font-size:13px; font-weight:500; }
.step-active { background:rgba(99,102,241,0.18); border:1px solid rgba(99,102,241,0.35); color:#a5b4fc; }
.step-done   { background:rgba(16,185,129,0.10); border:1px solid rgba(16,185,129,0.22); color:#6ee7b7; }
.step-idle   { background:rgba(8,12,24,0.50);     border:1px solid rgba(99,102,241,0.10); color:rgba(148,163,200,0.55); }

/* Model accuracy badge */
.acc-badge {
  display:inline-flex; align-items:center; gap:6px;
  background: linear-gradient(135deg,rgba(99,102,241,0.25),rgba(14,165,233,0.18));
  border:1px solid rgba(99,102,241,0.30);
  border-radius:12px; padding:8px 16px;
  font-family:'Syne',sans-serif; font-size:20px; font-weight:800; color:#fff;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  PLOT STYLING
# ═══════════════════════════════════════════════════════════════

BRAND_COLORS = [
    "#6366f1", "#0ea5e9", "#10b981", "#f59e0b",
    "#ec4899", "#8b5cf6", "#14b8a6", "#f97316",
]

def _style_fig(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color="rgba(200,210,235,0.88)", size=12),
        title_font=dict(family="Syne, sans-serif", size=15, color="#fff"),
        height=height,
        margin=dict(l=16, r=16, t=44, b=16),
        legend=dict(
            bgcolor="rgba(5,8,16,0.60)",
            bordercolor="rgba(99,102,241,0.18)",
            borderwidth=1,
            font=dict(size=11),
        ),
        xaxis=dict(gridcolor="rgba(99,102,241,0.08)", zerolinecolor="rgba(99,102,241,0.15)"),
        yaxis=dict(gridcolor="rgba(99,102,241,0.08)", zerolinecolor="rgba(99,102,241,0.15)"),
        colorway=BRAND_COLORS,
    )
    return fig


# ═══════════════════════════════════════════════════════════════
#  DATABASE & AUTH
# ═══════════════════════════════════════════════════════════════

def _db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash BLOB NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    con.commit()
    return con


def _hash_pw(pw: str) -> bytes:
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt(rounds=12))

def _check_pw(pw: str, h: bytes) -> bool:
    try:
        return bcrypt.checkpw(pw.encode(), h)
    except Exception:
        return False


def _create_user(name: str, email: str, pw: str) -> Tuple[bool, str]:
    name  = (name or "").strip()
    email = (email or "").strip().lower()
    if len(name) < 2:        return False, "Name too short (min 2 chars)."
    if not EMAIL_RE.match(email): return False, "Invalid email address."
    if len(pw) < 8:          return False, "Password needs at least 8 characters."
    con = _db()
    try:
        con.execute("INSERT INTO users(name,email,password_hash) VALUES(?,?,?)",
                    (name, email, _hash_pw(pw)))
        con.commit()
        return True, "Account created — please log in."
    except sqlite3.IntegrityError:
        return False, "Email already registered."
    finally:
        con.close()


def _login(email: str, pw: str) -> Tuple[bool, str, Optional[AuthUser]]:
    email = (email or "").strip().lower()
    if not EMAIL_RE.match(email): return False, "Invalid email.", None
    con = _db()
    try:
        row = con.execute(
            "SELECT id,name,email,password_hash FROM users WHERE email=?", (email,)
        ).fetchone()
        if not row:              return False, "Account not found — please sign up.", None
        uid, nm, em, ph = row
        if not _check_pw(pw, ph): return False, "Incorrect password.", None
        return True, "Welcome back!", AuthUser(int(uid), str(nm), str(em))
    finally:
        con.close()


def _ensure_state() -> None:
    if "auth_user"  not in st.session_state: st.session_state.auth_user  = None
    if "auth_page"  not in st.session_state: st.session_state.auth_page  = "Login"

def _logout() -> None:
    for k in ["auth_user","auth_page","artifacts","holdout","last_metrics"]:
        st.session_state.pop(k, None)
    st.session_state.auth_page = "Login"


# ═══════════════════════════════════════════════════════════════
#  LANDING / AUTH PAGE
# ═══════════════════════════════════════════════════════════════

def _render_auth_landing() -> None:
    _ensure_state()

    left, right = st.columns([1.35, 1.0], gap="large")

    with left:
        st.markdown(f"""
<div class="landing-bg">
  <div class="hero-badge">⚡ Powered by Ensemble ML</div>
  <div class="hero-title">{APP_TITLE}</div>
  <div style="font-family:'Syne',sans-serif;font-size:14px;letter-spacing:0.12em;text-transform:uppercase;color:rgba(99,102,241,0.80);margin-bottom:14px;">{APP_TAGLINE}</div>
  <div class="hero-sub">
    Predict every student's career trajectory with interpretable AI — placement, higher studies, entrepreneurship, or early intervention. Built for educators, institutions, and career counselors.
  </div>
  <div class="feature-grid">
    <div class="feature-item"><span class="feature-icon">🧠</span><div><strong>4 ML models</strong><br>RF, Logistic, GBM, SVM</div></div>
    <div class="feature-item"><span class="feature-icon">📊</span><div><strong>Rich EDA</strong><br>Interactive visual analytics</div></div>
    <div class="feature-item"><span class="feature-icon">🔍</span><div><strong>Explainability</strong><br>Permutation feature importance</div></div>
    <div class="feature-item"><span class="feature-icon">📄</span><div><strong>PDF &amp; HTML exports</strong><br>Share reports instantly</div></div>
    <div class="feature-item"><span class="feature-icon">🎯</span><div><strong>Per-student predictions</strong><br>With class probabilities</div></div>
    <div class="feature-item"><span class="feature-icon">🔐</span><div><strong>Secure auth</strong><br>bcrypt-hashed passwords</div></div>
  </div>
</div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        mode = st.radio("", ["Login", "Create account"],
                        index=0 if st.session_state.auth_page == "Login" else 1,
                        horizontal=True, label_visibility="collapsed")
        st.session_state.auth_page = mode
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        if mode == "Login":
            st.markdown('<div class="auth-title">Welcome back 👋</div>', unsafe_allow_html=True)
            st.markdown('<div class="auth-sub">Sign in to your EduPredict workspace.</div>', unsafe_allow_html=True)
            email = st.text_input("Email address", placeholder="you@institution.edu")
            pw    = st.text_input("Password", type="password", placeholder="••••••••")
            if st.button("Sign in →", type="primary", use_container_width=True):
                ok, msg, user = _login(email, pw)
                if ok:
                    st.session_state.auth_user = user
                    st.rerun()
                else:
                    st.error(msg)
        else:
            st.markdown('<div class="auth-title">Get started free 🚀</div>', unsafe_allow_html=True)
            st.markdown('<div class="auth-sub">Create your EduPredict account in seconds.</div>', unsafe_allow_html=True)
            name  = st.text_input("Full name", placeholder="Dr. Jane Smith")
            email = st.text_input("Email address", placeholder="you@institution.edu")
            pw    = st.text_input("Password", type="password", placeholder="Min 8 characters")
            pw2   = st.text_input("Confirm password", type="password", placeholder="Repeat password")
            if st.button("Create account →", type="primary", use_container_width=True):
                if pw != pw2:
                    st.error("Passwords don't match.")
                else:
                    ok, msg = _create_user(name, email, pw)
                    if ok:
                        st.success(msg)
                        st.session_state.auth_page = "Login"
                        st.rerun()
                    else:
                        st.error(msg)

        st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  DEMO DATASET
# ═══════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def make_demo_dataset(n: int = 800, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    gender     = rng.choice(["Female","Male","Other"], n, p=[0.47,0.50,0.03])
    department = rng.choice(["CSE","IT","ECE","ME","CE","AI&DS"], n, p=[0.25,0.17,0.16,0.16,0.12,0.14])
    year       = rng.choice(["1st","2nd","3rd","4th"], n, p=[0.20,0.25,0.28,0.27])
    attendance = np.clip(rng.normal(78,12,n), 35,100).round(1)
    cgpa       = np.clip(rng.normal(7.2,0.9,n), 4.0,10.0).round(2)
    backlogs   = np.clip(rng.poisson(0.7,n),   0,8).astype(int)
    internships= np.clip(rng.poisson(0.8,n),   0,6).astype(int)
    projects   = np.clip(rng.poisson(1.6,n),   0,10).astype(int)
    coding     = np.clip(rng.normal(62,18,n),  0,100).round(0)
    comm       = np.clip(rng.normal(60,16,n),  0,100).round(0)
    aptitude   = np.clip(rng.normal(58,17,n),  0,100).round(0)
    extra      = rng.choice(["Low","Medium","High"], n, p=[0.35,0.45,0.20])
    leadership = rng.choice(["No","Yes"],             n, p=[0.70,0.30])
    certifications= np.clip(rng.poisson(1.2,n), 0,8).astype(int)

    readiness = (
        0.35*(cgpa-6.0) + 0.015*(attendance-70)
        + 0.020*(coding-50) + 0.012*(comm-50) + 0.012*(aptitude-50)
        + 0.25*internships + 0.12*projects - 0.40*backlogs
        + 0.10*certifications + rng.normal(0,0.8,n)
    )
    readiness += np.where(extra=="High", 0.25, 0.0)
    readiness += np.where(leadership=="Yes", 0.18, 0.0)

    y = np.where(readiness>=2.15,"Placed",
        np.where(readiness>=1.15,"Higher Studies",
        np.where(readiness>=0.25,"Entrepreneur","Needs Support")))

    return pd.DataFrame({
        "Gender":gender,"Department":department,"Year_of_Study":year,
        "Attendance_%":attendance,"CGPA":cgpa,"Backlogs":backlogs,
        "Internships":internships,"Projects":projects,
        "Certifications":certifications,
        "Coding_Score":coding.astype(int),
        "Communication_Score":comm.astype(int),
        "Aptitude_Score":aptitude.astype(int),
        "Extracurricular":extra,"Leadership":leadership,
        "Future_Outcome":y,
    })


# ═══════════════════════════════════════════════════════════════
#  ML HELPERS
# ═══════════════════════════════════════════════════════════════

def _dataset_health(df: pd.DataFrame, target_col: str) -> Dict[str,Any]:
    miss  = int(df.isna().sum().sum())
    pct   = float(miss/max(int(df.size),1)*100)
    y     = df[target_col].astype(str)
    cnts  = y.value_counts(dropna=False)
    maj   = float(cnts.max()/max(len(y),1)*100) if len(cnts) else 0.0
    return {"missing_cells":miss,"missing_pct":pct,"classes":int(y.nunique(dropna=False)),"majority_pct":maj}


def _target_split_plan(y: pd.Series, cv_folds: int) -> Dict[str,Any]:
    ys   = y.astype(str); cnts = ys.value_counts()
    nc   = int(ys.nunique()); mc = int(cnts.min()) if len(cnts) else 0
    can_s= nc>1 and mc>=2
    mcv  = int(min(cv_folds,mc)) if nc>1 else 0
    return {"n_classes":nc,"min_class":mc,"can_stratify":can_s,
            "can_cv":mcv>=2,"cv_splits":mcv if mcv>=2 else None,
            "rare_classes":cnts[cnts<2].index.tolist()[:12]}


def _build_pipeline(X: pd.DataFrame, model_name: str) -> Pipeline:
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat = [c for c in X.columns if c not in num]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())]), num),
        ("cat", Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("ohe",OneHotEncoder(handle_unknown="ignore"))]), cat),
    ], remainder="drop")
    models = {
        "RandomForest":       RandomForestClassifier(n_estimators=500,min_samples_leaf=2,random_state=42,n_jobs=-1),
        "GradientBoosting":   GradientBoostingClassifier(n_estimators=300,learning_rate=0.08,max_depth=4,random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=2000,solver="lbfgs",C=1.0),
        "SVM":                SVC(probability=True,kernel="rbf",C=2.0,gamma="scale",random_state=42),
    }
    return Pipeline([("pre",pre),("model",models.get(model_name, models["RandomForest"]))])


def _infer_target(df: pd.DataFrame) -> Optional[str]:
    for c in ["Future_Outcome","future_outcome","Outcome","outcome","Target","target","Placed","placed"]:
        if c in df.columns: return c
    return None


def _train(df: pd.DataFrame, target_col: str, model_name: str,
           test_size: float) -> Tuple[TrainArtifacts, Any, Any, Any]:
    X = df.drop(columns=[target_col]); y = df[target_col].astype(str)
    plan = _target_split_plan(y,5)
    strat = y if plan["can_stratify"] else None
    X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=test_size,random_state=42,stratify=strat)
    pipe = _build_pipeline(X_tr, model_name)
    pipe.fit(X_tr, y_tr)
    y_pr = pipe.predict(X_te)
    cl   = list(map(str, getattr(pipe.named_steps["model"],"classes_",np.unique(y))))
    met  = {
        "accuracy": float(accuracy_score(y_te,y_pr)),
        "f1":       float(f1_score(y_te,y_pr,average="weighted",zero_division=0)),
        "precision":float(precision_score(y_te,y_pr,average="weighted",zero_division=0)),
        "recall":   float(recall_score(y_te,y_pr,average="weighted",zero_division=0)),
    }
    art = TrainArtifacts(pipe, list(X.columns), target_col, cl, model_name, met)
    return art, X_te, y_te, y_pr


def _feat_importance(art: TrainArtifacts, df: pd.DataFrame, n: int=14) -> pd.DataFrame:
    X = df.drop(columns=[art.target_column]); y = df[art.target_column].astype(str)
    plan = _target_split_plan(y,5)
    strat = y if plan["can_stratify"] else None
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.25,random_state=42,stratify=strat)
    art.pipeline.fit(Xtr,ytr)
    r = permutation_importance(art.pipeline,Xte,yte,n_repeats=12,random_state=42,scoring="accuracy",n_jobs=1)
    return (pd.DataFrame({"feature":X.columns,"importance":r.importances_mean,"std":r.importances_std})
              .sort_values("importance",ascending=False).head(n).reset_index(drop=True))


# ═══════════════════════════════════════════════════════════════
#  REPORT GENERATION
# ═══════════════════════════════════════════════════════════════

def _safe(text: str) -> str:
    """Replace characters outside latin-1 so fpdf built-in fonts never crash."""
    return (text
        .replace("\u2014", "-")
        .replace("\u2013", "-")
        .replace("\u00b7", ".")
        .replace("\u2022", "*")
        .encode("latin-1", errors="replace")
        .decode("latin-1"))


def _mpl_charts_b64(df, art, y_test, y_pred):
    """
    Build all report charts using matplotlib only (no kaleido / plotly needed).
    Returns list of (title, base64_png_string).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    import io as _io, base64, warnings
    warnings.filterwarnings("ignore")

    PALETTE = ["#6366f1","#0ea5e9","#10b981","#f59e0b","#ec4899","#8b5cf6","#14b8a6","#f97316"]
    BG   = "#f8fafc"
    CARD = "#ffffff"
    TEXT = "#0f172a"
    SUB  = "#64748b"

    def _b64(fig):
        buf = _io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        return b64

    def _style(ax, title="", xlabel="", ylabel=""):
        ax.set_facecolor(CARD)
        ax.tick_params(colors=SUB, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#e2e8f0")
        if title:
            ax.set_title(title, fontsize=10, fontweight="bold", color=TEXT, pad=8)
        if xlabel: ax.set_xlabel(xlabel, fontsize=8, color=SUB)
        if ylabel: ax.set_ylabel(ylabel, fontsize=8, color=SUB)
        ax.grid(axis="y", color="#f1f5f9", linewidth=0.8)
        ax.set_axisbelow(True)

    charts = []
    y_s   = df[art.target_column].astype(str)
    classes = sorted(y_s.unique().tolist())

    # ── 1. Outcome Distribution (donut) ───────────────────────────
    try:
        fig, ax = plt.subplots(figsize=(5.2, 3.8), facecolor=CARD)
        counts = y_s.value_counts()
        wedges, texts, autotexts = ax.pie(
            counts.values,
            labels=counts.index,
            autopct="%1.1f%%",
            colors=PALETTE[:len(counts)],
            startangle=90,
            wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2),
            pctdistance=0.78,
        )
        for t in texts:    t.set_fontsize(8);  t.set_color(TEXT)
        for t in autotexts: t.set_fontsize(7.5); t.set_color("white"); t.set_fontweight("bold")
        ax.set_title("Outcome Distribution", fontsize=10, fontweight="bold", color=TEXT, pad=10)
        fig.tight_layout()
        charts.append(("Outcome Distribution", _b64(fig)))
    except Exception: pass

    # ── 2. Confusion Matrix ────────────────────────────────────────
    try:
        labels_u = sorted(list(set(list(y_test) + list(y_pred))))
        cm = confusion_matrix(y_test, y_pred, labels=labels_u)
        fig, ax = plt.subplots(figsize=(5.2, 3.8), facecolor=CARD)
        ax.set_facecolor(CARD)
        im = ax.imshow(cm, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(labels_u))); ax.set_yticks(range(len(labels_u)))
        ax.set_xticklabels(labels_u, fontsize=7, rotation=20, ha="right", color=TEXT)
        ax.set_yticklabels(labels_u, fontsize=7, color=TEXT)
        ax.set_xlabel("Predicted", fontsize=8, color=SUB)
        ax.set_ylabel("Actual",    fontsize=8, color=SUB)
        ax.set_title("Confusion Matrix", fontsize=10, fontweight="bold", color=TEXT, pad=8)
        for i in range(len(labels_u)):
            for j in range(len(labels_u)):
                ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                        fontsize=11, fontweight="bold",
                        color="white" if cm[i,j] > cm.max()/2 else TEXT)
        for spine in ax.spines.values(): spine.set_visible(False)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        fig.tight_layout()
        charts.append(("Confusion Matrix", _b64(fig)))
    except Exception: pass

    # ── 3. Per-class Precision / Recall / F1 ──────────────────────
    try:
        p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, labels=classes, zero_division=0)
        x = np.arange(len(classes)); w = 0.25
        fig, ax = plt.subplots(figsize=(6.0, 3.8), facecolor=CARD)
        ax.set_facecolor(CARD)
        ax.bar(x - w, p,  w, label="Precision", color=PALETTE[0], alpha=0.88)
        ax.bar(x,     r,  w, label="Recall",    color=PALETTE[1], alpha=0.88)
        ax.bar(x + w, f1, w, label="F1",        color=PALETTE[2], alpha=0.88)
        ax.set_xticks(x); ax.set_xticklabels(classes, fontsize=8, color=TEXT, rotation=15, ha="right")
        ax.set_ylim(0, 1.12); ax.set_yticks([0,.25,.5,.75,1.0])
        ax.yaxis.set_tick_params(labelsize=7, colors=SUB)
        ax.legend(fontsize=7, framealpha=0.6, loc="upper right")
        _style(ax, "Per-Class Precision / Recall / F1")
        fig.tight_layout()
        charts.append(("Per-Class Metrics", _b64(fig)))
    except Exception: pass

    # ── 4. Feature Importance ──────────────────────────────────────
    try:
        model = art.pipeline.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            num_cols = [c for c in art.feature_columns if pd.api.types.is_numeric_dtype(df[c])]
            cat_cols = [c for c in art.feature_columns if c not in num_cols]
            pre = art.pipeline.named_steps["pre"]
            ohe = pre.named_transformers_["cat"].named_steps["ohe"]
            cat_names = list(ohe.get_feature_names_out(cat_cols))
            feat_names = num_cols + cat_names
            imp = model.feature_importances_
            if len(feat_names) == len(imp):
                imp_df = (pd.DataFrame({"feature": feat_names, "importance": imp})
                            .sort_values("importance", ascending=False).head(12)
                            .sort_values("importance"))
                fig, ax = plt.subplots(figsize=(6.0, 3.8), facecolor=CARD)
                ax.set_facecolor(CARD)
                colors = [PALETTE[i % len(PALETTE)] for i in range(len(imp_df))]
                bars = ax.barh(imp_df["feature"], imp_df["importance"], color=colors, alpha=0.88, height=0.65)
                ax.set_xlabel("Importance (Gini)", fontsize=8, color=SUB)
                ax.tick_params(axis="y", labelsize=7, colors=TEXT)
                ax.tick_params(axis="x", labelsize=7, colors=SUB)
                for spine in ["top","right"]: ax.spines[spine].set_visible(False)
                ax.spines["left"].set_color("#e2e8f0")
                ax.spines["bottom"].set_color("#e2e8f0")
                ax.set_title("Top Feature Importances", fontsize=10, fontweight="bold", color=TEXT, pad=8)
                fig.tight_layout()
                charts.append(("Feature Importance", _b64(fig)))
    except Exception: pass

    # ── 5. CGPA vs Coding Score scatter ───────────────────────────
    try:
        if "CGPA" in df.columns and "Coding_Score" in df.columns:
            fig, ax = plt.subplots(figsize=(5.2, 3.8), facecolor=CARD)
            ax.set_facecolor(CARD)
            for i, cls in enumerate(classes):
                mask = y_s == cls
                ax.scatter(df.loc[mask, "CGPA"], df.loc[mask, "Coding_Score"],
                           label=cls, color=PALETTE[i % len(PALETTE)],
                           alpha=0.60, s=18, linewidths=0)
            ax.legend(fontsize=7, framealpha=0.7, loc="upper left")
            _style(ax, "CGPA vs Coding Score", "CGPA", "Coding Score")
            for spine in ["top","right"]: ax.spines[spine].set_visible(False)
            fig.tight_layout()
            charts.append(("CGPA vs Coding Score", _b64(fig)))
    except Exception: pass

    # ── 6. Numeric correlation heatmap ────────────────────────────
    try:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != art.target_column]
        if len(num_cols) >= 3:
            corr = df[num_cols].corr(numeric_only=True)
            fig, ax = plt.subplots(figsize=(6.0, 4.2), facecolor=CARD)
            ax.set_facecolor(CARD)
            import matplotlib.cm as mcm
            im = ax.imshow(corr.values, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
            ax.set_xticks(range(len(num_cols))); ax.set_yticks(range(len(num_cols)))
            ax.set_xticklabels(num_cols, fontsize=6, rotation=35, ha="right", color=TEXT)
            ax.set_yticklabels(num_cols, fontsize=6, color=TEXT)
            for i in range(len(num_cols)):
                for j in range(len(num_cols)):
                    ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center",
                            fontsize=5.5,
                            color="white" if abs(corr.values[i,j]) > 0.5 else TEXT)
            for spine in ax.spines.values(): spine.set_visible(False)
            ax.set_title("Numeric Feature Correlation", fontsize=10, fontweight="bold", color=TEXT, pad=8)
            fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
            fig.tight_layout()
            charts.append(("Correlation Heatmap", _b64(fig)))
    except Exception: pass

    # ── 7. Class score distribution (box per outcome) ─────────────
    try:
        score_col = next((c for c in ["CGPA","Coding_Score","Aptitude_Score"] if c in df.columns), None)
        if score_col:
            fig, ax = plt.subplots(figsize=(5.2, 3.8), facecolor=CARD)
            ax.set_facecolor(CARD)
            data_by_class = [df.loc[y_s == cls, score_col].dropna().values for cls in classes]
            bp = ax.boxplot(data_by_class, patch_artist=True, notch=False,
                            medianprops=dict(color="white", linewidth=2),
                            whiskerprops=dict(color=SUB),
                            capprops=dict(color=SUB),
                            flierprops=dict(marker="o", markersize=3, alpha=0.4, color=SUB))
            for patch, color in zip(bp["boxes"], PALETTE):
                patch.set_facecolor(color); patch.set_alpha(0.80)
            ax.set_xticklabels(classes, fontsize=7.5, rotation=15, ha="right", color=TEXT)
            _style(ax, f"{score_col} Distribution by Outcome", ylabel=score_col)
            for spine in ["top","right"]: ax.spines[spine].set_visible(False)
            fig.tight_layout()
            charts.append((f"{score_col} by Outcome", _b64(fig)))
    except Exception: pass

    return charts


def _html_report(user, art, df, y_test, y_pred) -> str:
    health = _dataset_health(df, art.target_column)
    rep    = classification_report(y_test, y_pred, zero_division=0)
    m      = art.metrics
    ts     = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")

    # Build chart rows
    chart_html = ""
    try:
        charts = _mpl_charts_b64(df, art, y_test, y_pred)
        rows = []
        for i in range(0, len(charts), 2):
            pair = charts[i:i+2]
            cells = "".join(
                f'<div style="flex:1;min-width:0;">'
                f'<div style="font-size:12px;font-weight:700;color:#1e293b;margin-bottom:8px;">{name}</div>'
                f'<img src="data:image/png;base64,{b64}" '
                f'style="width:100%;border-radius:10px;border:1px solid #e2e8f0;display:block;"/>'
                f'</div>'
                for name, b64 in pair
            )
            rows.append(f'<div style="display:flex;gap:18px;margin-bottom:24px;">{cells}</div>')
        chart_html = "\n".join(rows)
    except Exception as e:
        chart_html = f'<p style="color:#ef4444;padding:12px;background:#fef2f2;border-radius:8px;">Chart error: {e}</p>'

    classes_badges = "".join(
        f'<span style="display:inline-block;padding:4px 12px;border-radius:999px;'
        f'font-size:11px;font-weight:700;background:#ede9fe;color:#6366f1;margin:3px;">{c}</span>'
        for c in art.classes_
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>EduPredict AI - Model Report</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500;600&display=swap');
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{font-family:'DM Sans','Segoe UI',sans-serif;background:#f1f5f9;color:#0f172a;line-height:1.5;}}

  /* Header */
  .hdr{{background:linear-gradient(135deg,#4f46e5 0%,#0ea5e9 100%);padding:38px 48px 30px;color:#fff;position:relative;overflow:hidden;}}
  .hdr::before{{content:'';position:absolute;right:-80px;top:-80px;width:320px;height:320px;border-radius:50%;background:rgba(255,255,255,0.06);}}
  .hdr::after{{content:'';position:absolute;left:30%;bottom:-60px;width:200px;height:200px;border-radius:50%;background:rgba(255,255,255,0.04);}}
  .hdr-logo{{font-family:'Syne',sans-serif;font-size:30px;font-weight:800;letter-spacing:-0.02em;}}
  .hdr-tag{{font-size:12px;opacity:0.75;margin-top:3px;letter-spacing:0.08em;text-transform:uppercase;}}
  .hdr-chips{{display:flex;gap:10px;margin-top:20px;flex-wrap:wrap;}}
  .chip{{background:rgba(255,255,255,0.15);border:1px solid rgba(255,255,255,0.25);border-radius:999px;padding:5px 14px;font-size:12px;font-weight:600;}}

  /* Body */
  .body{{padding:36px 48px 48px;max-width:1160px;margin:0 auto;}}

  /* Section */
  .sec-title{{font-family:'Syne',sans-serif;font-size:15px;font-weight:800;color:#1e293b;
    margin:32px 0 16px;padding-bottom:8px;border-bottom:2px solid #e2e8f0;
    display:flex;align-items:center;gap:8px;}}

  /* KPI grid */
  .kpi-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:8px;}}
  .kpi{{background:#fff;border:1px solid #e2e8f0;border-radius:14px;padding:18px 18px 14px;
    box-shadow:0 2px 10px rgba(0,0,0,.05);position:relative;overflow:hidden;}}
  .kpi::before{{content:'';position:absolute;top:0;left:0;right:0;height:3px;
    background:var(--acc,linear-gradient(90deg,#6366f1,#0ea5e9));}}
  .kpi-k{{font-size:10px;text-transform:uppercase;letter-spacing:.10em;color:#64748b;font-weight:700;}}
  .kpi-v{{font-size:26px;font-weight:800;color:#0f172a;margin-top:6px;}}

  /* Classification report */
  pre{{background:#0f172a;color:#e2e8f0;padding:20px 24px;border-radius:12px;
    font-family:'Courier New',monospace;font-size:11.5px;line-height:1.7;overflow-x:auto;}}

  /* Footer */
  footer{{text-align:center;padding:24px 48px;font-size:12px;color:#94a3b8;
    border-top:1px solid #e2e8f0;margin-top:40px;}}
</style>
</head>
<body>

<div class="hdr">
  <div class="hdr-logo">EduPredict AI</div>
  <div class="hdr-tag">Student Intelligence Platform &mdash; Model Report</div>
  <div class="hdr-chips">
    <span class="chip">&#128100; {user.name}</span>
    <span class="chip">&#129302; {art.model_name}</span>
    <span class="chip">&#127919; Target: {art.target_column}</span>
    <span class="chip">&#128336; {ts}</span>
  </div>
</div>

<div class="body">

  <div class="sec-title">&#9889; Performance Summary</div>
  <div class="kpi-grid">
    <div class="kpi" style="--acc:linear-gradient(90deg,#6366f1,#8b5cf6)">
      <div class="kpi-k">Students</div><div class="kpi-v">{len(df):,}</div></div>
    <div class="kpi" style="--acc:linear-gradient(90deg,#0ea5e9,#14b8a6)">
      <div class="kpi-k">Classes</div><div class="kpi-v">{health['classes']}</div></div>
    <div class="kpi" style="--acc:linear-gradient(90deg,#10b981,#0ea5e9)">
      <div class="kpi-k">Accuracy</div>
      <div class="kpi-v" style="color:#4f46e5">{m['accuracy']:.3f}</div></div>
    <div class="kpi" style="--acc:linear-gradient(90deg,#f59e0b,#ec4899)">
      <div class="kpi-k">F1 Weighted</div>
      <div class="kpi-v" style="color:#0ea5e9">{m['f1']:.3f}</div></div>
    <div class="kpi"><div class="kpi-k">Precision</div>
      <div class="kpi-v">{m['precision']:.3f}</div></div>
    <div class="kpi"><div class="kpi-k">Recall</div>
      <div class="kpi-v">{m['recall']:.3f}</div></div>
    <div class="kpi"><div class="kpi-k">Missing Cells</div>
      <div class="kpi-v">{health['missing_cells']:,}</div></div>
    <div class="kpi"><div class="kpi-k">Features</div>
      <div class="kpi-v">{len(art.feature_columns)}</div></div>
  </div>

  <div class="sec-title">&#128202; Visual Analytics</div>
  {chart_html}

  <div class="sec-title">&#128203; Classification Report</div>
  <pre>{rep}</pre>

  <div class="sec-title">&#127991; Classes Detected</div>
  <div style="margin-top:4px;">{classes_badges}</div>

</div>
<footer>Generated by <strong>EduPredict AI</strong> &nbsp;&middot;&nbsp; {ts} &nbsp;&middot;&nbsp; {user.email}</footer>
</body>
</html>"""


def _pdf_report(user, art, df, y_test, y_pred) -> bytes:
    import tempfile, os as _os
    m      = art.metrics
    health = _dataset_health(df, art.target_column)
    rep    = classification_report(y_test, y_pred, zero_division=0)

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_page()

    # ── Header banner ──────────────────────────────────────────
    pdf.set_fill_color(79, 70, 229)
    pdf.rect(0, 0, 210, 36, style="F")
    pdf.set_fill_color(14, 165, 233)
    pdf.rect(150, 0, 60, 36, style="F")
    pdf.set_font("Helvetica", "B", 17)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(14, 9)
    pdf.cell(0, 10, "EduPredict AI  -  Model Report")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_xy(14, 23)
    pdf.cell(0, 6, _safe(
        f"User: {user.name}  |  Model: {art.model_name}  |  "
        f"{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}"
    ))

    # ── Summary metrics (2-row card grid) ─────────────────────
    pdf.set_text_color(15, 23, 42)
    pdf.set_y(44)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_fill_color(241, 245, 249)
    pdf.cell(0, 7, "  Performance Summary", ln=True, fill=True)
    pdf.ln(3)

    metrics_data = [
        ("Students",      f"{len(df):,}",              "#6366f1"),
        ("Classes",       str(health["classes"]),       "#0ea5e9"),
        ("Accuracy",      f"{m['accuracy']:.4f}",       "#10b981"),
        ("F1 Weighted",   f"{m['f1']:.4f}",             "#f59e0b"),
        ("Precision",     f"{m['precision']:.4f}",      "#6366f1"),
        ("Recall",        f"{m['recall']:.4f}",         "#0ea5e9"),
        ("Missing Cells", f"{health['missing_cells']:,}","#94a3b8"),
        ("Features",      str(len(art.feature_columns)),"#94a3b8"),
    ]

    col_w = 45.5; row_h = 17; cols = 4; left = 14
    for idx, (k, v, accent_hex) in enumerate(metrics_data):
        col = idx % cols
        if col == 0:
            card_y = pdf.get_y()
        x = left + col * col_w
        # card background
        pdf.set_fill_color(248, 250, 252)
        pdf.set_draw_color(226, 232, 240)
        pdf.rect(x, card_y, col_w - 1, row_h, style="FD")
        # accent top line
        r, g, b = int(accent_hex[1:3],16), int(accent_hex[3:5],16), int(accent_hex[5:7],16)
        pdf.set_fill_color(r, g, b)
        pdf.rect(x, card_y, col_w - 1, 1.2, style="F")
        # label
        pdf.set_font("Helvetica", "", 6.5)
        pdf.set_text_color(100, 116, 139)
        pdf.set_xy(x + 2, card_y + 2.5)
        pdf.cell(col_w - 4, 3.5, _safe(k.upper()))
        # value
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(15, 23, 42)
        pdf.set_xy(x + 2, card_y + 8)
        pdf.cell(col_w - 4, 6, _safe(v))
        if col == cols - 1:
            pdf.set_y(card_y + row_h + 1)

    pdf.ln(4)

    # ── Charts via matplotlib ──────────────────────────────────
    try:
        charts = _mpl_charts_b64(df, art, y_test, y_pred)
        tmpfiles = []

        pdf.set_font("Helvetica", "B", 11)
        pdf.set_fill_color(241, 245, 249)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 7, "  Visual Analytics", ln=True, fill=True)
        pdf.ln(3)

        img_w = 88; img_h = 55; gap = 5; lx = 14

        # decode and write tmp files, then insert 2-per-row
        it = iter(charts)
        for name_l, b64_l in it:
            import base64 as _b64mod
            tmp_l = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp_l.write(_b64mod.b64decode(b64_l))
            tmp_l.close(); tmpfiles.append(tmp_l.name)

            try:
                name_r, b64_r = next(it)
                tmp_r = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                tmp_r.write(_b64mod.b64decode(b64_r))
                tmp_r.close(); tmpfiles.append(tmp_r.name)
            except StopIteration:
                tmp_r = None; name_r = ""

            row_y = pdf.get_y()
            # chart title + image — left
            pdf.set_font("Helvetica", "B", 7.5)
            pdf.set_text_color(30, 41, 59)
            pdf.set_xy(lx, row_y)
            pdf.cell(img_w, 5, _safe(name_l))
            pdf.image(tmp_l.name, x=lx, y=row_y + 5, w=img_w, h=img_h)
            # right
            if tmp_r:
                rx = lx + img_w + gap
                pdf.set_xy(rx, row_y)
                pdf.cell(img_w, 5, _safe(name_r))
                pdf.image(tmp_r.name, x=rx, y=row_y + 5, w=img_w, h=img_h)

            pdf.set_y(row_y + img_h + 9)

        for f in tmpfiles:
            try: _os.unlink(f)
            except Exception: pass

    except Exception as e:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(100, 116, 139)
        pdf.multi_cell(0, 5, _safe(f"Chart error: {str(e)[:180]}"))

    # ── Classification report ──────────────────────────────────
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_fill_color(241, 245, 249)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 7, "  Classification Report", ln=True, fill=True)
    pdf.ln(2)
    pdf.set_font("Courier", "", 7.5)
    pdf.set_text_color(30, 41, 59)
    pdf.multi_cell(0, 4.0, _safe(rep))

    # ── Footer ─────────────────────────────────────────────────
    pdf.set_y(-13)
    pdf.set_font("Helvetica", "", 7.5)
    pdf.set_text_color(148, 163, 184)
    pdf.cell(0, 5,
        _safe(f"EduPredict AI  |  {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}  |  {user.email}"),
        align="C"
    )

    out = pdf.output(dest="S")
    return out.encode("latin-1") if isinstance(out, str) else bytes(out)


# ═══════════════════════════════════════════════════════════════
#  UI COMPONENTS
# ═══════════════════════════════════════════════════════════════

def _kpi_html(label,value,hint,icon,accent="linear-gradient(90deg,#6366f1,#0ea5e9)") -> str:
    return f"""
<div class="kpi-card" style="--kpi-accent:{accent}">
  <div class="kpi-header">
    <div class="kpi-label">{label}</div>
    <div class="kpi-icon">{icon}</div>
  </div>
  <div class="kpi-value">{value}</div>
  <div class="kpi-hint">{hint}</div>
</div>"""


def _metric_row(y_true, y_pred) -> None:
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Accuracy",        f"{acc:.3f}")
    c2.metric("F1  (weighted)",  f"{f1:.3f}")
    c3.metric("Precision",       f"{prec:.3f}")
    c4.metric("Recall",          f"{rec:.3f}")


def _confusion_fig(y_true, y_pred, labels) -> go.Figure:
    cm  = confusion_matrix(y_true, y_pred, labels=labels)
    fig = px.imshow(cm, x=labels, y=labels,
                    color_continuous_scale=[[0,"rgba(5,8,16,1)"],[0.5,"rgba(99,102,241,0.60)"],[1,"rgba(14,165,233,1)"]],
                    text_auto=True, aspect="auto", title="Confusion Matrix")
    fig.update_traces(textfont=dict(size=13,color="#fff"))
    return _style_fig(fig, 400)


# ═══════════════════════════════════════════════════════════════
#  MAIN APP
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    st.set_page_config(
        page_title=f"{APP_TITLE} · {APP_TAGLINE}",
        page_icon="🎓", layout="wide",
        initial_sidebar_state="expanded"
    )
    _inject_css()
    _ensure_state()

    # ── Auth gate ──────────────────────────────────────────────
    if st.session_state.auth_user is None:
        _render_auth_landing()
        return

    user: AuthUser = st.session_state.auth_user

    # ── Sidebar ────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f"""
<div style="padding:12px 0 6px;">
  <div style="font-family:'Syne',sans-serif;font-size:18px;font-weight:800;
       background:linear-gradient(135deg,#6366f1,#0ea5e9);
       -webkit-background-clip:text;-webkit-text-fill-color:transparent;
       background-clip:text;">🎓 {APP_TITLE}</div>
  <div style="font-size:10px;letter-spacing:0.12em;text-transform:uppercase;
       color:rgba(99,102,241,0.75);margin-top:2px;">{APP_TAGLINE}</div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

        st.markdown("### 👤 Account")
        trained = "artifacts" in st.session_state
        st.markdown(f"""
<div class="profile-card">
  <div class="profile-name">{user.name}</div>
  <div class="profile-email">{user.email}</div>
  <div class="profile-badge">{'🟢 Model trained' if trained else '⚪ No model yet'}</div>
</div>
""", unsafe_allow_html=True)
        st.button("Sign out", on_click=_logout, use_container_width=True)

        st.markdown("### 📂 Data source")
        source = st.radio("", ["Demo dataset (800 students)","Upload your CSV"],
                          label_visibility="collapsed")

        st.markdown("### 🤖 Algorithm")
        model_name = st.selectbox(
            "", ["RandomForest","GradientBoosting","LogisticRegression","SVM"],
            label_visibility="collapsed"
        )
        model_desc = {
            "RandomForest":     "Best overall · handles mixed types well",
            "GradientBoosting": "Often top accuracy · slower training",
            "LogisticRegression":"Fast, interpretable · linear boundaries",
            "SVM":              "Powerful · slow on large datasets",
        }
        st.caption(model_desc.get(model_name,""))

        if "holdout" in st.session_state:
            Xte,yte,ype = st.session_state["holdout"]
            art: TrainArtifacts = st.session_state["artifacts"]
            df_side = make_demo_dataset() if "Upload" not in source else st.session_state.get("uploaded_df", make_demo_dataset())
            html_b = _html_report(user, art, df_side, yte, ype).encode()
            pdf_b  = _pdf_report(user, art, df_side, yte, ype)
            preds_dl = Xte.copy(); preds_dl["y_true"]=yte.to_numpy(); preds_dl["y_pred"]=ype
            bio = io.BytesIO(); preds_dl.to_csv(bio,index=False)

            st.markdown("""
<div style="margin-top:18px;margin-bottom:10px;">
  <div style="font-size:11px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;
       color:rgba(99,102,241,0.80);margin-bottom:12px;display:flex;align-items:center;gap:7px;">
    <span style="display:inline-block;width:18px;height:18px;border-radius:6px;
         background:rgba(99,102,241,0.18);text-align:center;line-height:18px;font-size:10px;">📤</span>
    Export Reports
  </div>
  <div style="background:rgba(5,8,16,0.55);border:1px solid rgba(99,102,241,0.14);
       border-radius:16px;padding:14px 12px;display:flex;flex-direction:column;gap:8px;">
    <div style="font-size:11px;color:rgba(148,163,200,0.55);margin-bottom:2px;padding-left:2px;">
      Model trained &nbsp;·&nbsp; 3 formats available
    </div>
""", unsafe_allow_html=True)

            st.download_button(
                "🌐  HTML Report",
                html_b, "edupredict_report.html", "text/html",
                use_container_width=True,
                help="Full interactive report in your browser"
            )
            st.download_button(
                "📄  PDF Report",
                pdf_b, "edupredict_report.pdf", "application/pdf",
                use_container_width=True,
                help="Printable PDF summary"
            )
            st.download_button(
                "📊  Holdout Predictions CSV",
                bio.getvalue(), "holdout_preds.csv", "text/csv",
                use_container_width=True,
                help="Raw predictions on test set"
            )
            st.markdown("</div></div>", unsafe_allow_html=True)

    # ── Load data ──────────────────────────────────────────────
    if "Upload" in source:
        up = st.file_uploader("Upload student CSV", type=["csv"], label_visibility="collapsed")
        if up is None:
            st.info("📂  Upload a CSV to get started, or switch to the demo dataset in the sidebar.")
            st.stop()
        df = pd.read_csv(up)
        st.session_state["uploaded_df"] = df
    else:
        df = make_demo_dataset()

    if df.empty:
        st.error("Dataset is empty."); st.stop()

    # ── Target column ──────────────────────────────────────────
    inferred    = _infer_target(df)
    default_idx = list(df.columns).index(inferred) if inferred in df.columns else len(df.columns)-1
    with st.sidebar:
        st.markdown("### 🎯 Target column")
        target_col = st.selectbox("", df.columns, index=default_idx, label_visibility="collapsed")
        st.caption("Must be a categorical outcome (Placed / Higher Studies / …)")

    # ── Page header ────────────────────────────────────────────
    h_left, h_right = st.columns([3,1])
    with h_left:
        st.markdown(f"""
<div class="page-header">
  <div>
    <div class="page-title">Student Intelligence Dashboard</div>
    <div class="page-sub">Predictive analytics for academic outcomes &nbsp;·&nbsp; {len(df):,} students loaded</div>
  </div>
</div>""", unsafe_allow_html=True)
    with h_right:
        st.markdown(f"""
<div style="text-align:right;padding-top:6px;">
  <span class="pill pill-violet">🤖 {model_name}</span>&nbsp;
  <span class="pill {'pill-green' if 'artifacts' in st.session_state else 'pill-amber'}">
    {'✓ Trained' if 'artifacts' in st.session_state else '○ Untrained'}
  </span>
</div>""", unsafe_allow_html=True)

    # ── Tabs ───────────────────────────────────────────────────
    tabs = st.tabs(["🏠  Overview","📊  Explore","🔬  Train & Evaluate","🎯  Predict","💡  Insights"])

    health = _dataset_health(df, target_col)

    # ─────────────────── TAB 0: OVERVIEW ──────────────────────
    with tabs[0]:
        trained = "artifacts" in st.session_state and "holdout" in st.session_state
        last_acc = st.session_state.get("last_metrics",{}).get("accuracy","—")
        acc_str  = f"{last_acc:.3f}" if isinstance(last_acc,float) else str(last_acc)

        kpis = f"""
<div class="kpi-grid">
{_kpi_html("Total students",f"{len(df):,}","rows in dataset","👥","linear-gradient(90deg,#6366f1,#8b5cf6)")}
{_kpi_html("Features",f"{df.shape[1]-1}","predictors (excl. target)","🧩","linear-gradient(90deg,#0ea5e9,#14b8a6)")}
{_kpi_html("Data quality",f"{100-health['missing_pct']:.1f}%",f"{health['missing_cells']:,} missing cells","🛡️","linear-gradient(90deg,#10b981,#0ea5e9)")}
{_kpi_html("Holdout accuracy",acc_str,"train model to update","🤖","linear-gradient(90deg,#f59e0b,#ec4899)")}
</div>"""
        st.markdown(kpis, unsafe_allow_html=True)
        st.write("")

        # Row 1 — donut + scatter
        c1, c2 = st.columns([1.1,0.9], gap="medium")
        with c1:
            st.markdown('<div class="section-card"><div class="section-title">Outcome distribution</div><div class="section-sub">Share of each predicted outcome class</div>', unsafe_allow_html=True)
            y_s = df[target_col].astype(str)
            cnts = y_s.value_counts().reset_index(); cnts.columns=["class","count"]
            fig = px.pie(cnts, names="class", values="count", hole=0.60,
                         color_discrete_sequence=BRAND_COLORS)
            fig.update_traces(textinfo="percent+label",
                              textfont=dict(family="DM Sans",size=12),
                              marker=dict(line=dict(color="rgba(5,8,16,0.9)",width=2)))
            fig.update_layout(showlegend=True)
            st.plotly_chart(_style_fig(fig, 340), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="section-card"><div class="section-title">CGPA vs Coding Score</div><div class="section-sub">Coloured by outcome class</div>', unsafe_allow_html=True)
            if "CGPA" in df.columns and "Coding_Score" in df.columns:
                fig2 = px.scatter(df, x="CGPA", y="Coding_Score", color=target_col,
                                  opacity=0.72, size_max=8,
                                  color_discrete_sequence=BRAND_COLORS)
                fig2.update_traces(marker=dict(size=5))
                st.plotly_chart(_style_fig(fig2, 340), use_container_width=True)
            else:
                num_c = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c!=target_col]
                if len(num_c) >= 2:
                    fig2 = px.scatter(df, x=num_c[0], y=num_c[1], color=target_col,
                                      opacity=0.72, color_discrete_sequence=BRAND_COLORS)
                    st.plotly_chart(_style_fig(fig2, 340), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.write("")
        # Row 2 — corr heatmap + box
        c3, c4 = st.columns(2, gap="medium")
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c!=target_col]
        with c3:
            st.markdown('<div class="section-card"><div class="section-title">Correlation matrix</div><div class="section-sub">Numeric feature relationships</div>', unsafe_allow_html=True)
            if len(num_cols) >= 2:
                corr = df[num_cols].corr(numeric_only=True)
                fig3 = px.imshow(corr, color_continuous_scale="RdBu", zmin=-1, zmax=1, aspect="auto")
                st.plotly_chart(_style_fig(fig3, 360), use_container_width=True)
            else:
                st.caption("Not enough numeric columns.")
            st.markdown('</div>', unsafe_allow_html=True)

        with c4:
            st.markdown('<div class="section-card"><div class="section-title">Feature vs outcome</div><div class="section-sub">Box distribution by class</div>', unsafe_allow_html=True)
            if num_cols:
                pick = st.selectbox("Feature", num_cols, key="ov_box")
                fig4 = px.box(df, x=target_col, y=pick, points="outliers",
                              color=target_col, color_discrete_sequence=BRAND_COLORS)
                fig4.update_layout(showlegend=False)
                st.plotly_chart(_style_fig(fig4, 320), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Row 3 — categorical breakdown
        cat_cols = [c for c in df.columns if c not in num_cols and c!=target_col]
        if cat_cols:
            st.write("")
            st.markdown('<div class="section-card"><div class="section-title">Category breakdown</div><div class="section-sub">Proportion of outcomes by categorical feature</div>', unsafe_allow_html=True)
            pick_cat = st.selectbox("Categorical feature", cat_cols, key="ov_cat")
            ct = pd.crosstab(df[pick_cat].astype(str), df[target_col].astype(str), normalize="index").reset_index()
            ct_m = ct.melt(id_vars=[pick_cat], var_name="Outcome", value_name="Share")
            fig5 = px.bar(ct_m, x=pick_cat, y="Share", color="Outcome", barmode="stack",
                          color_discrete_sequence=BRAND_COLORS)
            fig5.update_layout(yaxis_tickformat=".0%")
            st.plotly_chart(_style_fig(fig5, 340), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ─────────────────── TAB 1: EXPLORE ───────────────────────
    with tabs[1]:
        st.markdown('<div class="section-card"><div class="section-title">Dataset preview</div><div class="section-sub">First 40 rows</div>', unsafe_allow_html=True)
        st.dataframe(df.head(40), use_container_width=True, height=340)
        st.markdown('</div>', unsafe_allow_html=True)

        st.write("")
        e1, e2 = st.columns(2, gap="medium")
        with e1:
            st.markdown('<div class="section-card"><div class="section-title">Numeric stats</div>', unsafe_allow_html=True)
            st.dataframe(df[num_cols].describe().T.round(2), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with e2:
            st.markdown('<div class="section-card"><div class="section-title">Class counts</div>', unsafe_allow_html=True)
            cc = df[target_col].value_counts().reset_index(); cc.columns=["class","count"]
            fig_bar = px.bar(cc, x="class", y="count", color="class",
                             color_discrete_sequence=BRAND_COLORS)
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(_style_fig(fig_bar, 280), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        if num_cols:
            st.write("")
            st.markdown('<div class="section-card"><div class="section-title">Distribution explorer</div><div class="section-sub">Histogram with outcome overlay</div>', unsafe_allow_html=True)
            col_pick = st.selectbox("Numeric feature", num_cols, key="exp_hist")
            fig_h = px.histogram(df, x=col_pick, color=target_col, barmode="overlay",
                                 nbins=35, opacity=0.78, color_discrete_sequence=BRAND_COLORS)
            st.plotly_chart(_style_fig(fig_h, 320), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.write("")
            st.markdown('<div class="section-card"><div class="section-title">Violin plot</div>', unsafe_allow_html=True)
            v_pick = st.selectbox("Feature for violin", num_cols, key="exp_violin")
            fig_v = px.violin(df, x=target_col, y=v_pick, color=target_col,
                              box=True, points="outliers", color_discrete_sequence=BRAND_COLORS)
            fig_v.update_layout(showlegend=False)
            st.plotly_chart(_style_fig(fig_v, 340), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ─────────────────── TAB 2: TRAIN ─────────────────────────
    with tabs[2]:
        st.markdown('<div class="section-card"><div class="section-title">Training configuration</div><div class="section-sub">Adjust hyperparameters, then hit Train</div>', unsafe_allow_html=True)
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            test_size = st.slider("Holdout test size", 0.10, 0.40, 0.20, 0.05,
                                  help="Fraction of data reserved for evaluation")
        with col_b:
            cv_folds  = st.slider("Cross-validation folds", 3, 8, 5, 1)
        with col_c:
            st.markdown("<br>", unsafe_allow_html=True)
            run_cv = st.checkbox("Run CV after training", value=True)
        st.markdown('</div>', unsafe_allow_html=True)

        y_prev = df[target_col].astype(str)
        plan   = _target_split_plan(y_prev, cv_folds)
        if plan["n_classes"] > max(30, int(0.25*len(y_prev))):
            st.warning("⚠️ The selected target column has very high cardinality — it looks like a numeric ID/score, not a category. Please choose a categorical target.")
        if not plan["can_stratify"] and plan["n_classes"]>1:
            st.info("ℹ️ Some classes have < 2 samples. Stratified splitting will be disabled automatically.")

        st.write("")
        if st.button("🚀  Train model", type="primary", use_container_width=True):
            with st.spinner(f"Training {model_name}…"):
                try:
                    art, X_te, y_te, y_pr = _train(df, target_col, model_name, test_size)
                    st.session_state["artifacts"]    = art
                    st.session_state["holdout"]      = (X_te, y_te, y_pr)
                    st.session_state["last_metrics"] = art.metrics
                    st.success(f"✅  Training complete  ·  Holdout accuracy: **{art.metrics['accuracy']:.3f}**  ·  F1: **{art.metrics['f1']:.3f}**")
                except ValueError as e:
                    st.error(f"Training failed: {e}")
                    if plan["rare_classes"]:
                        st.caption(f"Rare classes (< 2 samples): {plan['rare_classes']}")
                    st.stop()

        if "holdout" in st.session_state:
            X_te, y_te, y_pr = st.session_state["holdout"]
            art: TrainArtifacts = st.session_state["artifacts"]
            labels = sorted(pd.unique(y_te.astype(str)).tolist())

            st.write("")
            _metric_row(y_te.to_numpy(), np.array(y_pr, dtype=str))

            st.write("")
            cm_col, rep_col = st.columns([1.1, 0.9], gap="medium")
            with cm_col:
                st.plotly_chart(_confusion_fig(y_te.to_numpy(), np.array(y_pr,dtype=str), labels),
                                use_container_width=True)
            with rep_col:
                with st.expander("📋  Classification report", expanded=True):
                    st.code(classification_report(y_te, y_pr, zero_division=0),
                            language=None)

            if run_cv:
                with st.expander("🔁  Cross-validation results", expanded=False):
                    if not plan["can_cv"]:
                        st.info("CV needs ≥ 2 samples per class. Reduce cardinality or add data.")
                    else:
                        with st.spinner("Running cross-validation…"):
                            X  = df.drop(columns=[target_col])
                            y  = df[target_col].astype(str)
                            pp = _build_pipeline(X, model_name)
                            cv = StratifiedKFold(n_splits=int(plan["cv_splits"]), shuffle=True, random_state=42)
                            y_cv = cross_val_predict(pp, X, y, cv=cv)
                        cv1, cv2, cv3 = st.columns(3)
                        cv1.metric("CV Accuracy", f"{accuracy_score(y,y_cv):.3f}")
                        cv2.metric("CV F1 (weighted)", f"{f1_score(y,y_cv,average='weighted',zero_division=0):.3f}")
                        cv3.metric("CV folds", str(plan["cv_splits"]))

    # ─────────────────── TAB 3: PREDICT ───────────────────────
    with tabs[3]:
        if "artifacts" not in st.session_state:
            st.warning("⚠️ Train the model first in the **Train & Evaluate** tab.")
            st.stop()

        art: TrainArtifacts = st.session_state["artifacts"]
        X_feat = df[art.feature_columns]

        st.markdown('<div class="section-card"><div class="section-title">Student profile input</div><div class="section-sub">Fill in student attributes to predict their future outcome</div>', unsafe_allow_html=True)
        form_cols = st.columns(3)
        input_row: Dict[str,Any] = {}
        for i, col in enumerate(art.feature_columns):
            with form_cols[i % 3]:
                s = X_feat[col]
                if pd.api.types.is_numeric_dtype(s):
                    vmin    = float(np.nanmin(s.to_numpy(dtype=float)))
                    vmax    = float(np.nanmax(s.to_numpy(dtype=float)))
                    default = float(np.nanmedian(s.to_numpy(dtype=float)))
                    step    = 1.0 if (vmax-vmin)>10 else 0.1
                    input_row[col] = st.number_input(col, vmin, vmax, default, step, key=f"pr_{col}")
                else:
                    opts = sorted(pd.Series(s.astype(str).unique()).dropna().tolist())
                    input_row[col] = st.selectbox(col, opts, key=f"pr_{col}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.write("")
        if st.button("🎯  Predict outcome", type="primary", use_container_width=True):
            X_in = pd.DataFrame([input_row], columns=art.feature_columns)
            pred = art.pipeline.predict(X_in)[0]
            proba= art.pipeline.predict_proba(X_in)[0] if hasattr(art.pipeline.named_steps["model"],"predict_proba") else None

            col_res, col_prob = st.columns([0.45, 0.55], gap="large")
            with col_res:
                outcome_colors = {
                    "Placed":"#10b981","Higher Studies":"#0ea5e9",
                    "Entrepreneur":"#f59e0b","Needs Support":"#ef4444"
                }
                col = outcome_colors.get(str(pred),"#6366f1")
                st.markdown(f"""
<div class="result-card" style="border-color:rgba({','.join(str(int(col.lstrip('#')[i:i+2],16)) for i in (0,2,4))},0.35);">
  <div class="result-label">Predicted outcome</div>
  <div class="result-outcome" style="color:{col};">{pred}</div>
  {'<div class="result-confidence">Confidence: <strong>' + f"{max(proba)*100:.1f}%" + '</strong></div>' if proba is not None else ''}
</div>""", unsafe_allow_html=True)

            with col_prob:
                if proba is not None:
                    pdf = pd.DataFrame({"class":art.classes_,"probability":proba}).sort_values("probability",ascending=True)
                    fig_p = px.bar(pdf, x="probability", y="class", orientation="h",
                                   color="probability",
                                   color_continuous_scale=[[0,"rgba(99,102,241,0.30)"],[1,"rgba(14,165,233,0.95)"]],
                                   text=pdf["probability"].map(lambda v:f"{v:.1%}"))
                    fig_p.update_traces(textposition="outside", textfont=dict(size=12))
                    fig_p.update_layout(coloraxis_showscale=False, xaxis_tickformat=".0%", xaxis_range=[0,1.05])
                    st.plotly_chart(_style_fig(fig_p, 280), use_container_width=True)

    # ─────────────────── TAB 4: INSIGHTS ──────────────────────
    with tabs[4]:
        if "artifacts" not in st.session_state:
            st.warning("⚠️ Train the model first in the **Train & Evaluate** tab.")
            st.stop()

        art: TrainArtifacts = st.session_state["artifacts"]

        st.markdown('<div class="section-card"><div class="section-title">Permutation feature importance</div><div class="section-sub">How much does each feature affect model accuracy? (higher = more important)</div>', unsafe_allow_html=True)
        with st.spinner("Computing permutation importance…"):
            try:
                imp = _feat_importance(art, df)
                fig_imp = px.bar(
                    imp.sort_values("importance"),
                    x="importance", y="feature", orientation="h",
                    error_x="std",
                    color="importance",
                    color_continuous_scale=[[0,"rgba(99,102,241,0.40)"],[1,"rgba(14,165,233,0.95)"]],
                    title="Permutation Feature Importance (mean accuracy drop)"
                )
                fig_imp.update_layout(coloraxis_showscale=False)
                st.plotly_chart(_style_fig(fig_imp, 440), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

                st.write("")
                st.markdown('<div class="section-card"><div class="section-title">Top feature deep-dive</div><div class="section-sub">See how the most important feature distributes across outcomes</div>', unsafe_allow_html=True)
                top_feat = imp.iloc[0]["feature"] if not imp.empty else None
                if top_feat and top_feat in df.columns:
                    if pd.api.types.is_numeric_dtype(df[top_feat]):
                        fig_top = px.histogram(df, x=top_feat, color=target_col, barmode="overlay",
                                               nbins=30, opacity=0.80, color_discrete_sequence=BRAND_COLORS,
                                               title=f"Distribution of '{top_feat}' by outcome")
                    else:
                        ct2 = pd.crosstab(df[top_feat].astype(str), df[target_col].astype(str), normalize="index").reset_index()
                        ct2_m = ct2.melt(id_vars=[top_feat], var_name="Outcome", value_name="Share")
                        fig_top = px.bar(ct2_m, x=top_feat, y="Share", color="Outcome", barmode="stack",
                                         color_discrete_sequence=BRAND_COLORS,
                                         title=f"'{top_feat}' vs Outcome")
                        fig_top.update_layout(yaxis_tickformat=".0%")
                    st.plotly_chart(_style_fig(fig_top, 320), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Could not compute importance: {e}")

        # Model info card
        st.write("")
        m = art.metrics
        st.markdown(f"""
<div class="section-card">
  <div class="section-title">Model summary</div>
  <div class="section-sub">Trained configuration and performance snapshot</div>
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:10px;">
    <div style="background:rgba(99,102,241,0.10);border:1px solid rgba(99,102,241,0.18);border-radius:12px;padding:12px 14px;">
      <div style="font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:rgba(148,163,200,.65);">Algorithm</div>
      <div style="font-family:'Syne',sans-serif;font-size:16px;font-weight:800;color:#fff;margin-top:4px;">{art.model_name}</div>
    </div>
    <div style="background:rgba(99,102,241,0.10);border:1px solid rgba(99,102,241,0.18);border-radius:12px;padding:12px 14px;">
      <div style="font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:rgba(148,163,200,.65);">Accuracy</div>
      <div style="font-family:'Syne',sans-serif;font-size:16px;font-weight:800;color:#10b981;margin-top:4px;">{m['accuracy']:.3f}</div>
    </div>
    <div style="background:rgba(99,102,241,0.10);border:1px solid rgba(99,102,241,0.18);border-radius:12px;padding:12px 14px;">
      <div style="font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:rgba(148,163,200,.65);">F1 Weighted</div>
      <div style="font-family:'Syne',sans-serif;font-size:16px;font-weight:800;color:#0ea5e9;margin-top:4px;">{m['f1']:.3f}</div>
    </div>
    <div style="background:rgba(99,102,241,0.10);border:1px solid rgba(99,102,241,0.18);border-radius:12px;padding:12px 14px;">
      <div style="font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:rgba(148,163,200,.65);">Features used</div>
      <div style="font-family:'Syne',sans-serif;font-size:16px;font-weight:800;color:#f59e0b;margin-top:4px;">{len(art.feature_columns)}</div>
    </div>
  </div>
  <div style="margin-top:12px;font-size:13px;color:rgba(148,163,200,.65);">
    Classes: {' · '.join(f'<span class="pill pill-violet">{c}</span>' for c in art.classes_)}
  </div>
</div>
""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()