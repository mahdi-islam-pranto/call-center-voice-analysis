"""
Agent Quality Check — Streamlit Frontend
Supports both Pre-Issuance (Outbound) and Inbound call QC, batch upload of
multiple recordings processed one by one against the FastAPI backend, and a
detailed per-call review view for QC analysts to read evaluations and decide
on counselling.
"""

import streamlit as st
import streamlit.components.v1 as components
import requests
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MetLife AQC — Agent Quality Check",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://localhost:8000"

# ─────────────────────────────────────────────
# CALL TYPE CONFIG
# ─────────────────────────────────────────────
CALL_TYPES = {
    "outbound": {
        "label": "Pre-Issuance (Outbound)",
        "endpoint": "/analyze",
        "max_total": 45,
        "criteria_order": [
            "Greetings", "Caller Authentication", "Telephony Etiquette", "Pronunciation",
            "Script Following", "Handling Time", "Complaint Handling",
            "Attentiveness / Focus", "Closing",
        ],
        "icons": {
            "Greetings": "👋", "Caller Authentication": "🔐", "Telephony Etiquette": "🎙️",
            "Pronunciation": "🗣️", "Script Following": "📋", "Handling Time": "⏱️",
            "Complaint Handling": "🤝", "Attentiveness / Focus": "🎯", "Closing": "✅",
        },
        "has_issue_summary": False,
    },
    "inbound": {
        "label": "Inbound",
        "endpoint": "/analyze-inbound",
        "max_total": 60,
        "criteria_order": [
            "Greetings", "Caller Authentication", "Telephony Etiquette", "Pronunciation",
            "Issue Identification", "Information Accuracy", "Issue Resolution", "Handling Time",
            "Complaint Handling", "FCR (First Call Resolution)", "Attentiveness / Focus", "Closing",
        ],
        "icons": {
            "Greetings": "👋", "Caller Authentication": "🔐", "Telephony Etiquette": "🎙️",
            "Pronunciation": "🗣️", "Issue Identification": "🔎", "Information Accuracy": "📑",
            "Issue Resolution": "🛠️", "Handling Time": "⏱️", "Complaint Handling": "🤝",
            "FCR (First Call Resolution)": "🔁", "Attentiveness / Focus": "🎯", "Closing": "✅",
        },
        "has_issue_summary": True,
    },
}

SCORE_COLORS = {5: "#2ECC71", 4: "#82E0AA", 3: "#F4D03F", 2: "#F39C12", 1: "#E74C3C", 0: "#922B21"}

# ─────────────────────────────────────────────
# CUSTOM CSS (same palette/style as the original AQC frontend)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #0D1B2A; color: #E8F0F7; }

/* ── Top toolbar/header (was showing as a plain white bar) ── */
header[data-testid="stHeader"] {
    background: #0D1B2A !important;
}
header[data-testid="stHeader"] * { color: #C8D8E8 !important; fill: #C8D8E8 !important; }
[data-testid="stToolbar"] { background: transparent !important; }
[data-testid="stDecoration"] { background: #00A8E8 !important; }
#MainMenu { visibility: hidden; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A1628 0%, #1E3A5F 100%);
    border-right: 1px solid #1E3A5F;
}
[data-testid="stSidebar"] * { color: #C8D8E8 !important; }

/* ── Buttons everywhere (sidebar AND main area — Previous/Next/Download etc.) ── */
.stButton button, .stDownloadButton button {
    background: #00A8E8 !important; color: #fff !important; border: none !important;
    font-weight: 600 !important; border-radius: 8px !important; padding: 0.5rem 1rem !important;
    width: 100%;
}
.stButton button:hover, .stDownloadButton button:hover {
    background: #0090CC !important; transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0,168,232,0.3) !important;
}
.stButton button:disabled {
    background: #1E3A5F !important; color: #5A7DA0 !important; opacity: 1 !important;
    box-shadow: none !important;
}

/* ── Selectbox / dropdown popovers (were defaulting to light theme) ── */
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: #162B45 !important; color: #E8F0F7 !important; border-color: #2A4F7C !important;
}
div[data-baseweb="popover"] ul, div[data-baseweb="menu"] {
    background: #162B45 !important; color: #E8F0F7 !important;
}
div[data-baseweb="popover"] li:hover { background: #1E3A5F !important; }

/* ── Dataframe / table (batch overview) ── */
[data-testid="stDataFrame"] { background: #162B45 !important; border-radius: 8px; }
[data-testid="stDataFrame"] * { color: #E8F0F7 !important; }
[data-testid="stElementToolbar"] { background: transparent !important; }

/* ── Metrics ── */
[data-testid="stMetric"] { background: #162B45; border: 1px solid #2A4F7C; border-radius: 8px; padding: 0.6rem 0.9rem; }
[data-testid="stMetricValue"] { color: #FFFFFF !important; }
[data-testid="stMetricLabel"] { color: #7BAFD4 !important; }

/* ── Checkbox/radio labels in main area ── */
.stCheckbox, .stRadio { color: #C8D8E8 !important; }

.aqc-header {
    background: linear-gradient(135deg, #1E3A5F 0%, #0D2847 100%);
    border: 1px solid #2A4F7C; border-radius: 12px; padding: 1.4rem 2rem;
    margin-bottom: 1.5rem; display: flex; align-items: center; gap: 1rem;
}
.aqc-header-title { font-size: 1.6rem; font-weight: 700; color: #FFFFFF; letter-spacing: -0.02em; }
.aqc-header-sub { font-size: 0.85rem; color: #7BAFD4; margin-top: 2px; }
.metlife-badge {
    background: #00A8E8; color: #fff; font-size: 0.7rem; font-weight: 700;
    padding: 3px 10px; border-radius: 20px; letter-spacing: 0.08em; text-transform: uppercase;
}
.type-badge {
    background: #1E3A5F; border: 1px solid #00A8E8; color: #00A8E8; font-size: 0.7rem;
    font-weight: 700; padding: 3px 10px; border-radius: 20px; letter-spacing: 0.05em;
    text-transform: uppercase; margin-left: 6px;
}

.result-slot-header {
    background: #1E3A5F; border: 1px solid #2A4F7C; border-bottom: 3px solid #00A8E8;
    border-radius: 10px 10px 0 0; padding: 0.9rem 1.2rem; margin-bottom: 0;
}
.slot-label { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: #00A8E8; }
.slot-agent { font-size: 1.1rem; font-weight: 600; color: #FFFFFF; margin-top: 2px; }
.slot-meta { font-size: 0.78rem; color: #7BAFD4; margin-top: 2px; }

.score-summary {
    background: #162B45; border: 1px solid #2A4F7C; border-radius: 0 0 10px 10px;
    padding: 1rem 1.2rem; margin-bottom: 1rem;
}
.big-score { font-family: 'JetBrains Mono', monospace; font-size: 2.8rem; font-weight: 700; line-height: 1; }
.big-score.pass { color: #2ECC71; }
.big-score.fail { color: #FF6B6B; }
.score-label { font-size: 0.75rem; color: #7BAFD4; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 4px; }

.counselling-badge-yes {
    background: rgba(255,107,107,0.15); border: 1px solid #FF6B6B; color: #FF6B6B;
    border-radius: 6px; padding: 5px 12px; font-size: 0.8rem; font-weight: 600;
    display: inline-block; margin-top: 8px;
}
.counselling-badge-no {
    background: rgba(46,204,113,0.12); border: 1px solid #2ECC71; color: #2ECC71;
    border-radius: 6px; padding: 5px 12px; font-size: 0.8rem; font-weight: 600;
    display: inline-block; margin-top: 8px;
}
.flag-badge {
    background: rgba(244,208,63,0.14); border: 1px solid #F4D03F; color: #F4D03F;
    border-radius: 6px; padding: 5px 12px; font-size: 0.8rem; font-weight: 600;
    display: inline-block; margin-top: 8px; margin-left: 8px;
}

.summary-box {
    background: #0D2036; border: 1px solid #2A4F7C; border-radius: 8px;
    padding: 0.9rem 1.1rem; font-size: 0.85rem; color: #A8C4DC; line-height: 1.7; margin-top: 0.5rem;
}
.issue-box {
    background: #0D2036; border: 1px solid #00A8E8; border-radius: 8px;
    padding: 0.8rem 1.1rem; font-size: 0.85rem; color: #E8F0F7; line-height: 1.6; margin-bottom: 0.8rem;
}

.section-label {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase;
    color: #00A8E8; margin: 1.4rem 0 0.6rem 0; padding-bottom: 4px; border-bottom: 1px solid #1E3A5F;
}

.stTabs [data-baseweb="tab-list"] { background: #162B45 !important; border-radius: 8px; padding: 4px; gap: 4px; }
.stTabs [data-baseweb="tab"] {
    background: transparent !important; color: #7BAFD4 !important; border-radius: 6px !important;
    font-size: 0.85rem !important; font-weight: 500 !important; padding: 0.4rem 1.1rem !important;
}
.stTabs [aria-selected="true"] { background: #00A8E8 !important; color: #fff !important; font-weight: 600 !important; }

[data-testid="stFileUploader"] { background: #162B45; border: 1px dashed #2A4F7C; border-radius: 10px; padding: 0.5rem; }
.streamlit-expanderHeader { background: #162B45 !important; color: #C8D8E8 !important; border-radius: 6px !important; font-size: 0.85rem !important; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0D1B2A; }
::-webkit-scrollbar-thumb { background: #2A4F7C; border-radius: 3px; }

.empty-state {
    background: #162B45; border: 1px dashed #2A4F7C; border-radius: 12px;
    padding: 2.5rem; text-align: center; color: #7BAFD4;
}
.empty-icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
.empty-title { font-size: 1rem; font-weight: 600; color: #C8D8E8; }
.empty-sub { font-size: 0.82rem; margin-top: 4px; }

.queue-row {
    background: #162B45; border: 1px solid #2A4F7C; border-radius: 8px;
    padding: 0.6rem 1rem; margin-bottom: 0.4rem; font-size: 0.85rem; color: #C8D8E8;
}
.error-box {
    background: rgba(255,107,107,0.1); border: 1px solid #FF6B6B; color: #FF6B6B;
    border-radius: 8px; padding: 0.7rem 1rem; font-size: 0.82rem; margin-bottom: 0.4rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "batch_results" not in st.session_state:
    st.session_state.batch_results = []   # list of dicts: {filename, call_type, result, timestamp}
if "batch_errors" not in st.session_state:
    st.session_state.batch_errors = []    # list of dicts: {filename, error}
if "selected_idx" not in st.session_state:
    st.session_state.selected_idx = 0


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def call_api(call_type_key: str, audio_bytes: bytes, filename: str, double_check: bool) -> dict:
    endpoint = CALL_TYPES[call_type_key]["endpoint"]
    files = {"audio": (filename, audio_bytes, "audio/mpeg")}
    params = {"double_check": double_check}
    resp = requests.post(f"{API_URL}{endpoint}", files=files, params=params, timeout=180)
    resp.raise_for_status()
    return resp.json()


def build_criterion_card_html(c: dict, icons: dict) -> str:
    score = c["score"]
    bar_pct = int(score / 5 * 100)
    sc_color = SCORE_COLORS.get(score, "#00A8E8")
    icon = icons.get(c["name"], "")
    justification = c.get("justification", "").replace("<", "&lt;").replace(">", "&gt;")
    evidence = c.get("evidence", "").replace("<", "&lt;").replace(">", "&gt;")
    name = c["name"].replace("<", "&lt;").replace(">", "&gt;")

    evidence_html = ""
    if evidence:
        evidence_html = (
            f'<div style="font-size:12px;color:#7BAFD4;margin-top:6px;line-height:1.6;">'
            f'<b style="color:#00A8E8;">Evidence:</b> {evidence}</div>'
        )

    return (
        f'<div style="background:#162B45;border:1px solid #2A4F7C;border-left:4px solid {sc_color};'
        f'border-radius:0 8px 8px 0;padding:12px 16px;margin-bottom:9px;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
        f'<span style="font-size:14px;font-weight:600;color:#C8D8E8;">{icon} {name}</span>'
        f'<span style="font-family:monospace;font-size:13px;font-weight:700;padding:2px 10px;'
        f'border-radius:20px;background:#0D1B2A;color:{sc_color};border:1px solid {sc_color};">'
        f'{score}/5</span></div>'
        f'<div style="background:#0D1B2A;border-radius:4px;height:5px;margin-top:8px;overflow:hidden;">'
        f'<div style="width:{bar_pct}%;height:100%;border-radius:4px;background:{sc_color};"></div></div>'
        f'{evidence_html}'
        f'<div style="font-size:12px;color:#8BA7C2;margin-top:6px;line-height:1.6;">{justification}</div>'
        f'</div>'
    )


def render_all_criteria(criteria_scores: list, icons: dict):
    """
    Render criterion cards via components.html() (raw iframe) so multi-line
    inline styles aren't stripped by Streamlit's HTML sanitizer.
    """
    cards_html = "".join(build_criterion_card_html(c, icons) for c in criteria_scores)
    full_html = (
        "<!DOCTYPE html><html><head><style>"
        "body{margin:0;padding:0;background:transparent;font-family:Inter,sans-serif;}"
        "</style></head><body>" + cards_html + "</body></html>"
    )
    components.html(full_html, height=len(criteria_scores) * 128, scrolling=False)


def make_gauge(pct: float) -> go.Figure:
    color = "#2ECC71" if pct >= 75 else "#FF6B6B"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number=dict(suffix="%", font=dict(size=28, color=color, family="JetBrains Mono")),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor="#2A4F7C", tickfont=dict(color="#7BAFD4", size=9)),
            bar=dict(color=color),
            bgcolor="#0D1B2A", borderwidth=1, bordercolor="#2A4F7C",
            steps=[dict(range=[0, 75], color="#1A2F47"), dict(range=[75, 100], color="#1A3530")],
            threshold=dict(line=dict(color="#F4D03F", width=3), thickness=0.85, value=75),
        ),
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=10, b=10),
                       height=170, font=dict(color="#C8D8E8"))
    return fig


def make_overview_bar(entries: list) -> go.Figure:
    names = [e["filename"] for e in entries]
    pcts = [e["result"]["percentage"] for e in entries]
    colors = ["#2ECC71" if p >= 75 else "#FF6B6B" for p in pcts]
    fig = go.Figure(go.Bar(
        x=names, y=pcts, marker_color=colors, text=[f"{p:.1f}%" for p in pcts],
        textposition="outside", textfont=dict(color="#E8F0F7", size=11),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#C8D8E8"),
        xaxis=dict(gridcolor="#1E3A5F", tickangle=-25, tickfont=dict(size=10)),
        yaxis=dict(gridcolor="#1E3A5F", range=[0, 105], title="Score %"),
        margin=dict(l=20, r=20, t=20, b=80), height=340, showlegend=False,
    )
    return fig


def render_result_header(result: dict, call_type_key: str, filename: str, timestamp: str):
    agent = result.get("agent_name") or "Unknown Agent"
    duration = result.get("call_duration_note") or "—"
    language = result.get("call_language_note")
    pct = result["percentage"]
    total = result["total_marks_obtained"]
    possible = result["total_marks_possible"]
    pass_class = "pass" if pct >= 75 else "fail"
    counselling = result["needs_counselling"]

    badges = (
        '<span class="counselling-badge-yes">⚠ Counselling Required</span>'
        if counselling else
        '<span class="counselling-badge-no">✓ No Counselling Needed</span>'
    )
    if result.get("low_confidence_flag"):
        badges += '<span class="flag-badge">🔍 Low Confidence — Review Audio</span>'
    if result.get("score_variance_flag"):
        badges += '<span class="flag-badge">⚖️ Score Variance — Double-Check Recommended</span>'

    meta_line = f"📁 {filename} &nbsp;·&nbsp; 🕐 {timestamp} &nbsp;·&nbsp; ⏱ {duration}"
    if language:
        meta_line += f" &nbsp;·&nbsp; 🗣 {language}"

    st.markdown(f"""
    <div class="result-slot-header">
        <div class="slot-label">{CALL_TYPES[call_type_key]['label']}</div>
        <div class="slot-agent">🎧 {agent}</div>
        <div class="slot-meta">{meta_line}</div>
    </div>
    <div class="score-summary">
        <div style="display:flex;gap:2rem;align-items:flex-start;flex-wrap:wrap;">
            <div>
                <div class="big-score {pass_class}">{pct:.1f}%</div>
                <div class="score-label">Overall Score</div>
            </div>
            <div>
                <div class="big-score {pass_class}">{total}<span style="font-size:1.2rem;color:#7BAFD4;">/{possible}</span></div>
                <div class="score-label">Marks Obtained</div>
            </div>
            <div style="align-self:center;">{badges}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_full_result(entry: dict):
    result = entry["result"]
    call_type_key = entry["call_type"]
    icons = CALL_TYPES[call_type_key]["icons"]

    render_result_header(result, call_type_key, entry["filename"], entry["timestamp"])

    col_gauge, col_summary = st.columns([1, 2])
    with col_gauge:
        st.plotly_chart(make_gauge(result["percentage"]), use_container_width=True,
                         key=f"gauge_{entry['filename']}_{entry['timestamp']}")
    with col_summary:
        if CALL_TYPES[call_type_key]["has_issue_summary"] and result.get("customer_issue_summary"):
            st.markdown(
                f'<div class="issue-box">📞 <b style="color:#00A8E8;">Customer called about:</b> '
                f'{result["customer_issue_summary"]}</div>', unsafe_allow_html=True,
            )
        st.markdown('<div class="section-label">Overall Summary</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="summary-box">{result["overall_summary"]}</div>', unsafe_allow_html=True)

        if result["needs_counselling"] and result.get("counselling_reason"):
            st.markdown(
                f'<div class="summary-box" style="border-color:#FF6B6B;margin-top:6px;">'
                f'⚠️ <b style="color:#FF6B6B;">Counselling Reason:</b> {result["counselling_reason"]}</div>',
                unsafe_allow_html=True,
            )
        if result.get("low_confidence_flag") and result.get("low_confidence_reason"):
            st.markdown(
                f'<div class="summary-box" style="border-color:#F4D03F;margin-top:6px;">'
                f'🔍 <b style="color:#F4D03F;">Low Confidence:</b> {result["low_confidence_reason"]}</div>',
                unsafe_allow_html=True,
            )
        if result.get("score_variance_flag") and result.get("score_variance_note"):
            st.markdown(
                f'<div class="summary-box" style="border-color:#F4D03F;margin-top:6px;">'
                f'⚖️ <b style="color:#F4D03F;">Score Variance:</b> {result["score_variance_note"]}</div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-label">Criteria Breakdown (with evidence)</div>', unsafe_allow_html=True)
    render_all_criteria(result["criteria_scores"], icons)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 0.3rem 0 1.2rem 0;">
        <div style="font-size:1.15rem;font-weight:700;color:#FFFFFF;">🎧 AQC System</div>
        <div style="font-size:0.75rem;color:#7BAFD4;margin-top:2px;">MetLife Bangladesh · Call Quality</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Call Type</div>', unsafe_allow_html=True)
    call_type_key = st.radio(
        "Select the type of call recording(s) you're evaluating:",
        options=list(CALL_TYPES.keys()),
        format_func=lambda k: CALL_TYPES[k]["label"],
        key="call_type_select",
        label_visibility="collapsed",
    )

    st.markdown('<div class="section-label">Upload Recordings</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload one or more call recordings",
        type=["mp3", "wav", "ogg", "m4a", "aac", "flac", "webm"],
        accept_multiple_files=True,
        key="batch_uploader",
    )

    double_check = st.checkbox(
        "Double-check pass (2x API cost)",
        value=False,
        help="Runs each recording through the model twice and flags any criterion where the "
             "two passes disagree by 2+ points, so you know which scores to verify manually.",
    )

    if uploaded_files:
        st.markdown(f'<div class="queue-row">📥 {len(uploaded_files)} file(s) ready to analyze</div>',
                    unsafe_allow_html=True)
        if st.button(f"▶ Analyze All ({len(uploaded_files)})", key="btn_analyze_all"):
            st.session_state.batch_results = []
            st.session_state.batch_errors = []

            progress = st.progress(0.0, text="Starting…")
            total_files = len(uploaded_files)

            for i, f in enumerate(uploaded_files):
                progress.progress(i / total_files, text=f"Analyzing {f.name} ({i+1}/{total_files})…")
                try:
                    audio_bytes = f.getvalue()
                    result = call_api(call_type_key, audio_bytes, f.name, double_check)
                    st.session_state.batch_results.append({
                        "filename": f.name,
                        "call_type": call_type_key,
                        "result": result,
                        "timestamp": datetime.now().strftime("%d %b %Y · %H:%M"),
                    })
                except Exception as e:
                    st.session_state.batch_errors.append({"filename": f.name, "error": str(e)})

            progress.progress(1.0, text="Done.")
            st.session_state.selected_idx = 0
            n_ok = len(st.session_state.batch_results)
            n_err = len(st.session_state.batch_errors)
            if n_err == 0:
                st.success(f"Analyzed {n_ok} recording(s) ✓")
            else:
                st.warning(f"Analyzed {n_ok} recording(s), {n_err} failed — see below.")

    if st.session_state.batch_errors:
        for e in st.session_state.batch_errors:
            st.markdown(f'<div class="error-box">❌ <b>{e["filename"]}</b>: {e["error"]}</div>',
                        unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑 Clear All Results", use_container_width=True):
        st.session_state.batch_results = []
        st.session_state.batch_errors = []
        st.session_state.selected_idx = 0
        st.rerun()

    st.markdown("---")
    cfg = CALL_TYPES[call_type_key]
    st.markdown(f"""
    <div style="font-size:0.72rem;color:#4A6A8A;line-height:1.8;">
        <b style="color:#7BAFD4;">API Endpoint</b><br>
        <code style="color:#00A8E8;font-size:0.7rem;">{API_URL}{cfg['endpoint']}</code><br><br>
        <b style="color:#7BAFD4;">Counselling Threshold</b><br>
        Score &lt; 75% or any criterion ≤ 1<br><br>
        <b style="color:#7BAFD4;">Max Score</b><br>
        {cfg['max_total']} marks ({len(cfg['criteria_order'])} × 5)
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
current_label = CALL_TYPES[call_type_key]["label"]
st.markdown(f"""
<div class="aqc-header">
    <div>
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
            <div class="aqc-header-title">Agent Quality Check</div>
            <span class="metlife-badge">MetLife</span>
            <span class="type-badge">{current_label}</span>
        </div>
        <div class="aqc-header-sub">AI-powered call recording evaluation · Batch upload · Bangla & English support</div>
    </div>
</div>
""", unsafe_allow_html=True)

entries = st.session_state.batch_results

tab_review, tab_overview = st.tabs(["🔍 Review Calls", "📊 Batch Overview"])

# ── TAB: REVIEW ──────────────────────────────
with tab_review:
    if not entries:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🎧</div>
            <div class="empty-title">No recordings analyzed yet</div>
            <div class="empty-sub">Choose a call type, upload one or more recordings in the sidebar, and click
            <b>Analyze All</b> to begin.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        labels = []
        for e in entries:
            r = e["result"]
            agent = r.get("agent_name") or "Unknown Agent"
            flag = " ⚠️" if r["needs_counselling"] else ""
            labels.append(f"{e['filename']} — {agent} ({r['percentage']:.1f}%){flag}")

        st.session_state.selected_idx = st.selectbox(
            "Select a recording to review in detail:",
            options=list(range(len(entries))),
            format_func=lambda i: labels[i],
            index=min(st.session_state.selected_idx, len(entries) - 1),
        )
        render_full_result(entries[st.session_state.selected_idx])

        st.markdown("---")
        c_prev, c_pos, c_next = st.columns([1, 3, 1])
        with c_prev:
            if st.button("← Previous", disabled=st.session_state.selected_idx == 0):
                st.session_state.selected_idx -= 1
                st.rerun()
        with c_pos:
            st.markdown(
                f"<div style='text-align:center;color:#7BAFD4;font-size:0.82rem;padding-top:6px;'>"
                f"Recording {st.session_state.selected_idx + 1} of {len(entries)}</div>",
                unsafe_allow_html=True,
            )
        with c_next:
            if st.button("Next →", disabled=st.session_state.selected_idx == len(entries) - 1):
                st.session_state.selected_idx += 1
                st.rerun()

# ── TAB: BATCH OVERVIEW ──────────────────────
with tab_overview:
    if not entries:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">📊</div>
            <div class="empty-title">Nothing to summarize yet</div>
            <div class="empty-sub">Analyze a batch of recordings to see them all here at once.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="section-label">All Recordings — Score Overview</div>', unsafe_allow_html=True)
        st.plotly_chart(make_overview_bar(entries), use_container_width=True, key="overview_bar")

        st.markdown('<div class="section-label">Full Batch Table</div>', unsafe_allow_html=True)

        def rating_label(pct: float) -> str:
            if pct >= 90:
                return "🟢 Excellent"
            if pct >= 75:
                return "🟢 Good"
            if pct >= 60:
                return "🟡 Fair"
            return "🔴 Poor"

        # Union of every criterion name seen across the batch, in the order each
        # call type defines them — handles a batch that mixes inbound + outbound.
        all_criteria = []
        for cfg in CALL_TYPES.values():
            for c in cfg["criteria_order"]:
                if c not in all_criteria:
                    all_criteria.append(c)

        rows = []
        for e in entries:
            r = e["result"]
            criteria_map = {cs["name"]: cs["score"] for cs in r["criteria_scores"]}
            row = {
                "File": e["filename"],
                "Call Type": CALL_TYPES[e["call_type"]]["label"],
                "Agent": r.get("agent_name") or "Unknown",
            }
            for c in all_criteria:
                row[c] = criteria_map.get(c, "—")
            row["Total Marks"] = f"{r['total_marks_obtained']}/{r['total_marks_possible']}"
            row["Score %"] = r["percentage"]
            row["Call Rating"] = rating_label(r["percentage"])
            row["Counselling"] = "⚠ Required" if r["needs_counselling"] else "✓ Not Needed"
            row["Low Confidence"] = "🔍 Yes" if r.get("low_confidence_flag") else "—"
            row["Analyzed At"] = e["timestamp"]
            rows.append(row)
        df = pd.DataFrame(rows)

        def style_pct(val):
            try:
                v = float(val)
                return "color:#2ECC71;font-weight:bold" if v >= 75 else "color:#FF6B6B;font-weight:bold"
            except Exception:
                return ""

        def style_criterion(val):
            try:
                v = int(val)
                return f"color:{SCORE_COLORS.get(v, '#C8D8E8')};font-weight:600"
            except Exception:
                return "color:#5A7DA0"

        styled = (
            df.style
            .apply(lambda col: [style_pct(v) for v in col], subset=["Score %"])
            .apply(lambda col: [style_criterion(v) for v in col], subset=all_criteria)
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)
        st.caption(
            "Criterion columns show that call type's actual 0–5 score, or '—' if that "
            "criterion doesn't apply to the call type (e.g. inbound-only criteria for an "
            "outbound call)."
        )

        n_flagged = sum(1 for e in entries if e["result"]["needs_counselling"])
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Total Recordings", len(entries))
        with m2:
            avg_pct = sum(e["result"]["percentage"] for e in entries) / len(entries)
            st.metric("Average Score", f"{avg_pct:.1f}%")
        with m3:
            st.metric("Flagged for Counselling", f"{n_flagged}/{len(entries)}")

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download Batch Results (CSV)", data=csv,
                            file_name=f"aqc_batch_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv")