"""
Agent Quality Check — Streamlit Frontend
Interacts with the FastAPI /analyze endpoint and supports side-by-side comparison of two QC results.
"""

import streamlit as st
import streamlit.components.v1 as components
import requests
import plotly.graph_objects as go
import plotly.express as px
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

# API_URL = st.secrets.get("API_URL", "http://localhost:8000")
API_URL = "http://localhost:8000"

CRITERIA_ORDER = [
    "Greetings",
    "Caller Authentication",
    "Telephony Etiquette",
    "Pronunciation",
    "Script Following",
    "Handling Time",
    "Complaint Handling",
    "Attentiveness / Focus",
    "Closing",
]

CRITERIA_ICONS = {
    "Greetings": "👋",
    "Caller Authentication": "🔐",
    "Telephony Etiquette": "🎙️",
    "Pronunciation": "🗣️",
    "Script Following": "📋",
    "Handling Time": "⏱️",
    "Complaint Handling": "🤝",
    "Attentiveness / Focus": "🎯",
    "Closing": "✅",
}

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Background ── */
.stApp {
    background-color: #0D1B2A;
    color: #E8F0F7;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A1628 0%, #1E3A5F 100%);
    border-right: 1px solid #1E3A5F;
}
[data-testid="stSidebar"] * {
    color: #C8D8E8 !important;
}
[data-testid="stSidebar"] .stButton button {
    background: #00A8E8 !important;
    color: #fff !important;
    border: none !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 0.5rem 1rem !important;
    width: 100%;
}
[data-testid="stSidebar"] .stButton button:hover {
    background: #0090CC !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0,168,232,0.3) !important;
}

/* ── Header ── */
.aqc-header {
    background: linear-gradient(135deg, #1E3A5F 0%, #0D2847 100%);
    border: 1px solid #2A4F7C;
    border-radius: 12px;
    padding: 1.4rem 2rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.aqc-header-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: -0.02em;
}
.aqc-header-sub {
    font-size: 0.85rem;
    color: #7BAFD4;
    margin-top: 2px;
}
.metlife-badge {
    background: #00A8E8;
    color: #fff;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── Result card header ── */
.result-slot-header {
    background: #1E3A5F;
    border: 1px solid #2A4F7C;
    border-bottom: 3px solid #00A8E8;
    border-radius: 10px 10px 0 0;
    padding: 0.9rem 1.2rem;
    margin-bottom: 0;
}
.slot-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #00A8E8;
}
.slot-agent {
    font-size: 1.1rem;
    font-weight: 600;
    color: #FFFFFF;
    margin-top: 2px;
}
.slot-meta {
    font-size: 0.78rem;
    color: #7BAFD4;
    margin-top: 2px;
}

/* ── Score summary cards ── */
.score-summary {
    background: #162B45;
    border: 1px solid #2A4F7C;
    border-radius: 0 0 10px 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
}
.big-score {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.8rem;
    font-weight: 700;
    line-height: 1;
}
.big-score.pass { color: #2ECC71; }
.big-score.fail { color: #FF6B6B; }
.score-label {
    font-size: 0.75rem;
    color: #7BAFD4;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
}
.counselling-badge-yes {
    background: rgba(255,107,107,0.15);
    border: 1px solid #FF6B6B;
    color: #FF6B6B;
    border-radius: 6px;
    padding: 5px 12px;
    font-size: 0.8rem;
    font-weight: 600;
    display: inline-block;
    margin-top: 8px;
}
.counselling-badge-no {
    background: rgba(46,204,113,0.12);
    border: 1px solid #2ECC71;
    color: #2ECC71;
    border-radius: 6px;
    padding: 5px 12px;
    font-size: 0.8rem;
    font-weight: 600;
    display: inline-block;
    margin-top: 8px;
}

/* ── Criterion cards ── */
.criterion-card {
    background: #162B45;
    border: 1px solid #2A4F7C;
    border-left: 4px solid #00A8E8;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin-bottom: 0.55rem;
    transition: border-left-color 0.2s;
}
.criterion-card.score-5 { border-left-color: #2ECC71; }
.criterion-card.score-4 { border-left-color: #82E0AA; }
.criterion-card.score-3 { border-left-color: #F4D03F; }
.criterion-card.score-2 { border-left-color: #F39C12; }
.criterion-card.score-1 { border-left-color: #E74C3C; }
.criterion-card.score-0 { border-left-color: #922B21; }

.crit-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.crit-name {
    font-size: 0.88rem;
    font-weight: 600;
    color: #C8D8E8;
}
.crit-score-pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    font-weight: 700;
    padding: 2px 10px;
    border-radius: 20px;
    background: #0D1B2A;
}
.crit-score-pill.s5 { color: #2ECC71; border: 1px solid #2ECC71; }
.crit-score-pill.s4 { color: #82E0AA; border: 1px solid #82E0AA; }
.crit-score-pill.s3 { color: #F4D03F; border: 1px solid #F4D03F; }
.crit-score-pill.s2 { color: #F39C12; border: 1px solid #F39C12; }
.crit-score-pill.s1 { color: #E74C3C; border: 1px solid #E74C3C; }
.crit-score-pill.s0 { color: #922B21; border: 1px solid #922B21; }

.crit-bar-bg {
    background: #0D1B2A;
    border-radius: 4px;
    height: 5px;
    margin-top: 7px;
    overflow: hidden;
}
.crit-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.5s ease;
}
.crit-justification {
    font-size: 0.77rem;
    color: #8BA7C2;
    margin-top: 6px;
    line-height: 1.5;
}

/* ── Comparison diff badge ── */
.diff-better { color: #2ECC71; font-weight: 700; font-size: 0.8rem; }
.diff-worse  { color: #FF6B6B; font-weight: 700; font-size: 0.8rem; }
.diff-same   { color: #7BAFD4; font-weight: 600; font-size: 0.8rem; }

/* ── Summary box ── */
.summary-box {
    background: #0D2036;
    border: 1px solid #2A4F7C;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    font-size: 0.83rem;
    color: #A8C4DC;
    line-height: 1.7;
    margin-top: 0.5rem;
}

/* ── Section divider ── */
.section-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #00A8E8;
    margin: 1.4rem 0 0.6rem 0;
    padding-bottom: 4px;
    border-bottom: 1px solid #1E3A5F;
}

/* ── Tabs override ── */
.stTabs [data-baseweb="tab-list"] {
    background: #162B45 !important;
    border-radius: 8px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #7BAFD4 !important;
    border-radius: 6px !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 0.4rem 1.1rem !important;
}
.stTabs [aria-selected="true"] {
    background: #00A8E8 !important;
    color: #fff !important;
    font-weight: 600 !important;
}

/* ── Upload area ── */
[data-testid="stFileUploader"] {
    background: #162B45;
    border: 1px dashed #2A4F7C;
    border-radius: 10px;
    padding: 0.5rem;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #162B45 !important;
    color: #C8D8E8 !important;
    border-radius: 6px !important;
    font-size: 0.85rem !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0D1B2A; }
::-webkit-scrollbar-thumb { background: #2A4F7C; border-radius: 3px; }

/* ── Empty state ── */
.empty-state {
    background: #162B45;
    border: 1px dashed #2A4F7C;
    border-radius: 12px;
    padding: 2.5rem;
    text-align: center;
    color: #7BAFD4;
}
.empty-icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
.empty-title { font-size: 1rem; font-weight: 600; color: #C8D8E8; }
.empty-sub { font-size: 0.82rem; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "result_a" not in st.session_state:
    st.session_state.result_a = None
if "result_a_filename" not in st.session_state:
    st.session_state.result_a_filename = None
if "result_a_time" not in st.session_state:
    st.session_state.result_a_time = None
if "result_b" not in st.session_state:
    st.session_state.result_b = None
if "result_b_filename" not in st.session_state:
    st.session_state.result_b_filename = None
if "result_b_time" not in st.session_state:
    st.session_state.result_b_time = None


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def score_color(score: int) -> str:
    return ["#922B21","#E74C3C","#F39C12","#F4D03F","#82E0AA","#2ECC71"][score]

def score_bar_color(score: int) -> str:
    colors = ["#922B21","#E74C3C","#F39C12","#F4D03F","#82E0AA","#2ECC71"]
    return colors[score]

def get_scores_dict(result: dict) -> dict:
    return {c["name"]: c["score"] for c in result["criteria_scores"]}

def call_api(audio_bytes: bytes, filename: str) -> dict:
    files = {"audio": (filename, audio_bytes, "audio/mpeg")}
    resp = requests.post(f"{API_URL}/analyze", files=files, timeout=120)
    resp.raise_for_status()
    return resp.json()

def build_criterion_card_html(c: dict, show_diff: int | None = None) -> str:
    """Return the HTML string for one criterion card."""
    score = c["score"]
    bar_pct = int(score / 5 * 100)

    score_colors = {5: "#2ECC71", 4: "#82E0AA", 3: "#F4D03F", 2: "#F39C12", 1: "#E74C3C", 0: "#922B21"}
    sc_color = score_colors.get(score, "#00A8E8")
    bl_color = sc_color

    diff_html = ""
    if show_diff is not None:
        delta = score - show_diff
        if delta > 0:
            diff_html = f'<span style="color:#2ECC71;font-weight:700;font-size:13px;">&#9650; +{delta}</span>'
        elif delta < 0:
            diff_html = f'<span style="color:#FF6B6B;font-weight:700;font-size:13px;">&#9660; {delta}</span>'
        else:
            diff_html = f'<span style="color:#7BAFD4;font-weight:600;font-size:13px;">= same</span>'

    icon = CRITERIA_ICONS.get(c["name"], "")
    justification = c["justification"].replace("<", "&lt;").replace(">", "&gt;")
    name = c["name"].replace("<", "&lt;").replace(">", "&gt;")

    return (
        f'<div style="background:#162B45;border:1px solid #2A4F7C;border-left:4px solid {bl_color};'
        f'border-radius:0 8px 8px 0;padding:12px 16px;margin-bottom:8px;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
        f'<span style="font-size:14px;font-weight:600;color:#C8D8E8;">{icon} {name}</span>'
        f'<span style="display:flex;gap:8px;align-items:center;">'
        f'{diff_html}'
        f'<span style="font-family:monospace;font-size:13px;font-weight:700;padding:2px 10px;'
        f'border-radius:20px;background:#0D1B2A;color:{sc_color};border:1px solid {sc_color};">'
        f'{score}/5</span></span></div>'
        f'<div style="background:#0D1B2A;border-radius:4px;height:5px;margin-top:8px;overflow:hidden;">'
        f'<div style="width:{bar_pct}%;height:100%;border-radius:4px;background:{sc_color};"></div></div>'
        f'<div style="font-size:12px;color:#8BA7C2;margin-top:8px;line-height:1.6;">{justification}</div>'
        f'</div>'
    )


def render_all_criteria(criteria_scores: list, compare_scores: dict | None = None):
    """
    Render all 9 criterion cards using st.components.v1.html() which renders raw HTML
    in an iframe — completely bypassing Streamlit's HTML sanitizer (bleach) that strips
    multi-line style attributes and causes raw tag leakage.
    """
    cards_html = ""
    for c in criteria_scores:
        diff = compare_scores.get(c["name"]) if compare_scores else None
        cards_html += build_criterion_card_html(c, show_diff=diff)

    full_html = (
        "<!DOCTYPE html><html><head>"
        "<style>"
        "body{margin:0;padding:0;background:transparent;font-family:Inter,sans-serif;}"
        "</style></head>"
        f"<body>{cards_html}</body></html>"
    )
    # Height: 9 cards × ~115px each
    components.html(full_html, height=9 * 115, scrolling=False)

def render_result_header(result: dict, slot_label: str, filename: str, timestamp: str):
    agent = result.get("agent_name") or "Unknown Agent"
    duration = result.get("call_duration_note") or "—"
    pct = result["percentage"]
    total = result["total_marks_obtained"]
    possible = result["total_marks_possible"]
    pass_class = "pass" if pct >= 75 else "fail"
    counselling = result["needs_counselling"]
    badge = (
        '<span class="counselling-badge-yes">⚠ Counselling Required</span>'
        if counselling else
        '<span class="counselling-badge-no">✓ No Counselling Needed</span>'
    )

    st.markdown(f"""
    <div class="result-slot-header">
        <div class="slot-label">{slot_label}</div>
        <div class="slot-agent">🎧 {agent}</div>
        <div class="slot-meta">📁 {filename} &nbsp;·&nbsp; 🕐 {timestamp} &nbsp;·&nbsp; ⏱ {duration}</div>
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
            <div style="align-self:center;">{badge}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def make_radar_chart(results: list[dict], labels: list[str]) -> go.Figure:
    categories = CRITERIA_ORDER
    fig = go.Figure()
    colors = ["#00A8E8", "#F4D03F", "#2ECC71"]
    for i, (result, label) in enumerate(zip(results, labels)):
        scores_d = get_scores_dict(result)
        vals = [scores_d.get(c, 0) for c in categories]
        vals_closed = vals + [vals[0]]
        cats_closed = categories + [categories[0]]
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatterpolar(
            r=vals_closed,
            theta=cats_closed,
            fill='toself',
            name=label,
            line=dict(color=color, width=2.5),
            fillcolor=color.replace("#", "rgba(").rstrip(")") if False else color,
            opacity=0.85,
        ))
        # Use fillcolor properly
        fig.data[i].fillcolor = f"rgba{tuple(int(color.lstrip('#')[j:j+2], 16) for j in (0,2,4)) + (0.18,)}"

    fig.update_layout(
        polar=dict(
            bgcolor="#162B45",
            radialaxis=dict(
                visible=True, range=[0, 5],
                gridcolor="#2A4F7C", linecolor="#2A4F7C",
                tickfont=dict(color="#7BAFD4", size=10),
                tickvals=[1,2,3,4,5],
            ),
            angularaxis=dict(
                gridcolor="#2A4F7C", linecolor="#2A4F7C",
                tickfont=dict(color="#C8D8E8", size=11),
            ),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            font=dict(color="#C8D8E8", size=12),
            bgcolor="rgba(22,43,69,0.8)",
            bordercolor="#2A4F7C",
            borderwidth=1,
        ),
        margin=dict(l=40, r=40, t=30, b=30),
        height=380,
    )
    return fig


def make_bar_chart(results: list[dict], labels: list[str]) -> go.Figure:
    fig = go.Figure()
    colors = ["#00A8E8", "#F4D03F"]
    for i, (result, label) in enumerate(zip(results, labels)):
        scores_d = get_scores_dict(result)
        vals = [scores_d.get(c, 0) for c in CRITERIA_ORDER]
        fig.add_trace(go.Bar(
            name=label,
            x=[f"{CRITERIA_ICONS.get(c,'')} {c}" for c in CRITERIA_ORDER],
            y=vals,
            marker_color=colors[i % len(colors)],
            opacity=0.88,
            text=vals,
            textposition="outside",
            textfont=dict(color="#E8F0F7", size=11),
        ))
    fig.update_layout(
        barmode='group',
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#C8D8E8"),
        xaxis=dict(gridcolor="#1E3A5F", tickangle=-30, tickfont=dict(size=10)),
        yaxis=dict(gridcolor="#1E3A5F", range=[0, 6], tickvals=[0,1,2,3,4,5]),
        legend=dict(bgcolor="rgba(22,43,69,0.8)", bordercolor="#2A4F7C", borderwidth=1),
        margin=dict(l=20, r=20, t=20, b=60),
        height=320,
    )
    return fig


def make_gauge(pct: float) -> go.Figure:
    color = "#2ECC71" if pct >= 75 else "#FF6B6B"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number=dict(suffix="%", font=dict(size=28, color=color, family="JetBrains Mono")),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor="#2A4F7C",
                      tickfont=dict(color="#7BAFD4", size=9)),
            bar=dict(color=color),
            bgcolor="#0D1B2A",
            borderwidth=1,
            bordercolor="#2A4F7C",
            steps=[
                dict(range=[0, 75], color="#1A2F47"),
                dict(range=[75, 100], color="#1A3530"),
            ],
            threshold=dict(line=dict(color="#F4D03F", width=3), thickness=0.85, value=75),
        ),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=10, b=10),
        height=180,
        font=dict(color="#C8D8E8"),
    )
    return fig


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

    st.markdown('<div class="section-label">Slot A — Primary Analysis</div>', unsafe_allow_html=True)
    audio_a = st.file_uploader("Upload Call Recording (A)", type=["mp3","wav","ogg","m4a","aac","flac","webm"], key="upload_a")
    if audio_a:
        if st.button("▶ Analyze Recording A", key="btn_a"):
            with st.spinner("Analyzing recording A…"):
                try:
                    result = call_api(audio_a.read(), audio_a.name)
                    st.session_state.result_a = result
                    st.session_state.result_a_filename = audio_a.name
                    st.session_state.result_a_time = datetime.now().strftime("%d %b %Y · %H:%M")
                    st.success("Recording A analyzed ✓")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.markdown('<div class="section-label">Slot B — Comparison Analysis</div>', unsafe_allow_html=True)
    audio_b = st.file_uploader("Upload Call Recording (B)", type=["mp3","wav","ogg","m4a","aac","flac","webm"], key="upload_b")
    if audio_b:
        if st.button("▶ Analyze Recording B", key="btn_b"):
            with st.spinner("Analyzing recording B…"):
                try:
                    result = call_api(audio_b.read(), audio_b.name)
                    st.session_state.result_b = result
                    st.session_state.result_b_filename = audio_b.name
                    st.session_state.result_b_time = datetime.now().strftime("%d %b %Y · %H:%M")
                    st.success("Recording B analyzed ✓")
                except Exception as e:
                    st.error(f"Error: {e}")

    # Clear buttons
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear A", use_container_width=True):
            st.session_state.result_a = None
            st.session_state.result_a_filename = None
            st.session_state.result_a_time = None
            st.rerun()
    with col2:
        if st.button("Clear B", use_container_width=True):
            st.session_state.result_b = None
            st.session_state.result_b_filename = None
            st.session_state.result_b_time = None
            st.rerun()

    st.markdown("---")
    st.markdown(f"""
    <div style="font-size:0.72rem;color:#4A6A8A;line-height:1.8;">
        <b style="color:#7BAFD4;">API Endpoint</b><br>
        <code style="color:#00A8E8;font-size:0.7rem;">{API_URL}</code><br><br>
        <b style="color:#7BAFD4;">Counselling Threshold</b><br>
        Score &lt; 75% or any criterion ≤ 1<br><br>
        <b style="color:#7BAFD4;">Max Score</b><br>
        45 marks (9 × 5)
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
st.markdown("""
<div class="aqc-header">
    <div>
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
            <div class="aqc-header-title">Agent Quality Check</div>
            <span class="metlife-badge">MetLife</span>
        </div>
        <div class="aqc-header-sub">AI-powered call recording evaluation · 9 quality criteria · Bangla & English support</div>
    </div>
</div>
""", unsafe_allow_html=True)

result_a = st.session_state.result_a
result_b = st.session_state.result_b
has_a = result_a is not None
has_b = result_b is not None

# ── TABS ──────────────────────────────────────
tab_labels = ["📊 Results"]
if has_a and has_b:
    tab_labels.append("⚖️ Comparison")
tab_labels.append("ℹ️ About")

tabs = st.tabs(tab_labels)
tab_results = tabs[0]
tab_compare = tabs[1] if (has_a and has_b) else None
tab_about = tabs[-1]

# ─────────────────────────────────────────────
# TAB: RESULTS
# ─────────────────────────────────────────────
with tab_results:
    if not has_a and not has_b:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🎧</div>
            <div class="empty-title">No recordings analyzed yet</div>
            <div class="empty-sub">Upload a call recording in the sidebar and click <b>Analyze</b> to begin.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        if has_a and has_b:
            cols = st.columns([1, 1])
        else:
            cols = [st.container()]

        # ── Slot A ──
        if has_a:
            with cols[0]:
                render_result_header(
                    result_a, "Slot A · Primary",
                    st.session_state.result_a_filename,
                    st.session_state.result_a_time,
                )

                # Gauge
                st.plotly_chart(make_gauge(result_a["percentage"]), use_container_width=True, key="gauge_a")

                # Overall summary
                st.markdown('<div class="section-label">Overall Summary</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="summary-box">{result_a["overall_summary"]}</div>', unsafe_allow_html=True)

                if result_a["needs_counselling"] and result_a.get("counselling_reason"):
                    st.markdown(f'<div class="summary-box" style="border-color:#FF6B6B;margin-top:6px;">⚠️ <b style="color:#FF6B6B;">Counselling Reason:</b> {result_a["counselling_reason"]}</div>', unsafe_allow_html=True)

                # Criteria cards — all rendered in one st.markdown call
                st.markdown('<div class="section-label">Criteria Breakdown</div>', unsafe_allow_html=True)
                scores_b_dict = get_scores_dict(result_b) if has_b else None
                render_all_criteria(result_a["criteria_scores"], compare_scores=scores_b_dict)

        # ── Slot B ──
        if has_b:
            with cols[1] if has_a else cols[0]:
                render_result_header(
                    result_b, "Slot B · Comparison",
                    st.session_state.result_b_filename,
                    st.session_state.result_b_time,
                )

                st.plotly_chart(make_gauge(result_b["percentage"]), use_container_width=True, key="gauge_b")

                st.markdown('<div class="section-label">Overall Summary</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="summary-box">{result_b["overall_summary"]}</div>', unsafe_allow_html=True)

                if result_b["needs_counselling"] and result_b.get("counselling_reason"):
                    st.markdown(f'<div class="summary-box" style="border-color:#FF6B6B;margin-top:6px;">⚠️ <b style="color:#FF6B6B;">Counselling Reason:</b> {result_b["counselling_reason"]}</div>', unsafe_allow_html=True)

                st.markdown('<div class="section-label">Criteria Breakdown</div>', unsafe_allow_html=True)
                scores_a_dict = get_scores_dict(result_a) if has_a else None
                render_all_criteria(result_b["criteria_scores"], compare_scores=scores_a_dict)

        # ── Only A, no B: show note ──
        if has_a and not has_b:
            st.markdown("""
            <div class="empty-state" style="margin-top:1rem;">
                <div class="empty-icon">⚖️</div>
                <div class="empty-title">Compare with a second recording</div>
                <div class="empty-sub">Upload Recording B in the sidebar to enable side-by-side comparison and score delta indicators.</div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TAB: COMPARISON
# ─────────────────────────────────────────────
if has_a and has_b and tab_compare is not None:
    with tab_compare:
        agent_a = result_a.get("agent_name") or "Recording A"
        agent_b = result_b.get("agent_name") or "Recording B"
        label_a = f"{agent_a} (A)"
        label_b = f"{agent_b} (B)"

        st.markdown('<div class="section-label">Performance Radar — Head to Head</div>', unsafe_allow_html=True)
        st.plotly_chart(make_radar_chart([result_a, result_b], [label_a, label_b]),
                        use_container_width=True, key="radar_cmp")

        st.markdown('<div class="section-label">Score by Criterion — Side by Side</div>', unsafe_allow_html=True)
        st.plotly_chart(make_bar_chart([result_a, result_b], [label_a, label_b]),
                        use_container_width=True, key="bar_cmp")

        # ── Delta table ──
        st.markdown('<div class="section-label">Score Delta Table</div>', unsafe_allow_html=True)
        scores_a = get_scores_dict(result_a)
        scores_b = get_scores_dict(result_b)
        rows = []
        for c in CRITERIA_ORDER:
            sa = scores_a.get(c, 0)
            sb = scores_b.get(c, 0)
            delta = sa - sb
            delta_str = f"+{delta}" if delta > 0 else str(delta)
            rows.append({
                "Criterion": f"{CRITERIA_ICONS.get(c,'')} {c}",
                f"{label_a}": sa,
                f"{label_b}": sb,
                "Delta (A−B)": delta_str,
                "Winner": label_a if sa > sb else (label_b if sb > sa else "Tie"),
            })
        df = pd.DataFrame(rows)

        # Style the dataframe
        def style_delta(val):
            try:
                v = int(val)
                if v > 0: return "color: #2ECC71; font-weight: bold"
                if v < 0: return "color: #FF6B6B; font-weight: bold"
                return "color: #7BAFD4"
            except: return ""

        styled = df.style.apply(lambda col: [style_delta(v) for v in col], subset=["Delta (A−B)"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # ── Summary comparison metrics ──
        st.markdown('<div class="section-label">Summary Comparison</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Score A", f"{result_a['percentage']:.1f}%",
                      delta=f"{result_a['percentage'] - result_b['percentage']:+.1f}%")
        with m2:
            st.metric("Score B", f"{result_b['percentage']:.1f}%",
                      delta=f"{result_b['percentage'] - result_a['percentage']:+.1f}%")
        with m3:
            wins_a = sum(1 for c in CRITERIA_ORDER if scores_a.get(c, 0) > scores_b.get(c, 0))
            wins_b = sum(1 for c in CRITERIA_ORDER if scores_b.get(c, 0) > scores_a.get(c, 0))
            st.metric("Criteria Won (A)", f"{wins_a}/9")
        with m4:
            st.metric("Criteria Won (B)", f"{wins_b}/9")

        # ── Counselling comparison ──
        st.markdown('<div class="section-label">Counselling Decision</div>', unsafe_allow_html=True)
        cc1, cc2 = st.columns(2)
        with cc1:
            if result_a["needs_counselling"]:
                st.markdown(f'<div class="summary-box" style="border-color:#FF6B6B;"><b style="color:#FF6B6B;">⚠ {label_a}: Counselling Required</b><br><span style="color:#8BA7C2;font-size:0.8rem;">{result_a.get("counselling_reason","")}</span></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="summary-box" style="border-color:#2ECC71;"><b style="color:#2ECC71;">✓ {label_a}: No Counselling Needed</b></div>', unsafe_allow_html=True)
        with cc2:
            if result_b["needs_counselling"]:
                st.markdown(f'<div class="summary-box" style="border-color:#FF6B6B;"><b style="color:#FF6B6B;">⚠ {label_b}: Counselling Required</b><br><span style="color:#8BA7C2;font-size:0.8rem;">{result_b.get("counselling_reason","")}</span></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="summary-box" style="border-color:#2ECC71;"><b style="color:#2ECC71;">✓ {label_b}: No Counselling Needed</b></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TAB: ABOUT
# ─────────────────────────────────────────────
with tab_about:
    st.markdown("""
    <div class="summary-box" style="font-size:0.88rem;line-height:2;">
        <b style="color:#00A8E8;font-size:1rem;">Agent Quality Check (AQC) System</b><br><br>
        This tool uses <b style="color:#C8D8E8;">Gemini 2.5 Flash Lite</b> (multimodal AI) to analyze MetLife Bangladesh
        call center agent recordings and score them against <b style="color:#C8D8E8;">9 quality criteria</b>.<br><br>

        <b style="color:#C8D8E8;">How to use:</b><br>
        1. Upload a call recording (MP3, WAV, OGG, M4A etc.) in <b>Slot A</b> via the sidebar<br>
        2. Click <b>Analyze Recording A</b> — the AI will evaluate the full call<br>
        3. Optionally upload a second recording in <b>Slot B</b> to compare results<br>
        4. Switch to the <b>⚖️ Comparison</b> tab for side-by-side charts and delta analysis<br><br>

        <b style="color:#C8D8E8;">Criteria (each scored 0–5):</b><br>
        👋 Greetings &nbsp;·&nbsp; 🔐 Caller Authentication &nbsp;·&nbsp; 🎙️ Telephony Etiquette
        &nbsp;·&nbsp; 🗣️ Pronunciation &nbsp;·&nbsp; 📋 Script Following &nbsp;·&nbsp; ⏱️ Handling Time
        &nbsp;·&nbsp; 🤝 Complaint Handling &nbsp;·&nbsp; 🎯 Attentiveness / Focus &nbsp;·&nbsp; ✅ Closing<br><br>

        <b style="color:#C8D8E8;">Counselling Threshold:</b><br>
        An agent is flagged for counselling if their total score is <b style="color:#FF6B6B;">below 75%</b>
        or if any single criterion scores <b style="color:#FF6B6B;">0 or 1</b>.<br><br>

        <b style="color:#C8D8E8;">Language Support:</b> Bangla 🇧🇩, English 🇬🇧, or mixed — the AI handles both natively.
    </div>
    """, unsafe_allow_html=True)