"""
Agent Performance QC — Streamlit Frontend (General Purpose / Manual Criteria)
Interacts with the FastAPI /agent-performance endpoint (Gemini native-audio backend)
and supports side-by-side comparison of two agents' evaluations.
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
    page_title="AQC — Agent Performance QC",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_API_URL = "http://localhost:8000/agent-performance"
SCORE_MAX = 10  # matches qc_chain_manual.py SCORE_MAX

DEFAULT_METRICS = [
    "Opening Greetings",
    "Active Listening",
    "Check Resource",
    "Hold",
    "Correct Info",
    "Complete Info",
    "Empathy/Tone",
    "Taking Ownership",
    "Further Assistance",
    "Ending Greetings",
    "Slang Usage",
]

METRIC_ICONS = {
    "Opening Greetings": "👋",
    "Active Listening": "👂",
    "Check Resource": "📚",
    "Hold": "⏸️",
    "Correct Info": "✅",
    "Complete Info": "📋",
    "Empathy/Tone": "❤️",
    "Taking Ownership": "🤝",
    "Further Assistance": "🙋",
    "Ending Greetings": "🏁",
    "Slang Usage": "🗣️",
}
DEFAULT_ICON = "📌"

# ─────────────────────────────────────────────
# CUSTOM CSS (same visual language as the MetLife dashboard)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background-color: #0D1B2A; color: #E8F0F7; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A1628 0%, #1E3A5F 100%);
    border-right: 1px solid #1E3A5F;
}
[data-testid="stSidebar"] * { color: #C8D8E8 !important; }
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
.aqc-header-title { font-size: 1.6rem; font-weight: 700; color: #FFFFFF; letter-spacing: -0.02em; }
.aqc-header-sub { font-size: 0.85rem; color: #7BAFD4; margin-top: 2px; }
.general-badge {
    background: #F4A825;
    color: #1a1200;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.result-slot-header {
    background: #1E3A5F;
    border: 1px solid #2A4F7C;
    border-bottom: 3px solid #00A8E8;
    border-radius: 10px 10px 0 0;
    padding: 0.9rem 1.2rem;
    margin-bottom: 0;
}
.slot-label { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: #00A8E8; }
.slot-agent { font-size: 1.1rem; font-weight: 600; color: #FFFFFF; margin-top: 2px; }
.slot-meta { font-size: 0.78rem; color: #7BAFD4; margin-top: 2px; }

.score-summary {
    background: #162B45;
    border: 1px solid #2A4F7C;
    border-radius: 0 0 10px 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
}
.big-score { font-family: 'JetBrains Mono', monospace; font-size: 2.8rem; font-weight: 700; line-height: 1; }
.big-score.pass { color: #2ECC71; }
.big-score.fail { color: #FF6B6B; }
.score-label { font-size: 0.75rem; color: #7BAFD4; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 4px; }

.flag-badge-yes {
    background: rgba(255,107,107,0.15); border: 1px solid #FF6B6B; color: #FF6B6B;
    border-radius: 6px; padding: 5px 12px; font-size: 0.8rem; font-weight: 600;
    display: inline-block; margin-top: 8px;
}
.flag-badge-no {
    background: rgba(46,204,113,0.12); border: 1px solid #2ECC71; color: #2ECC71;
    border-radius: 6px; padding: 5px 12px; font-size: 0.8rem; font-weight: 600;
    display: inline-block; margin-top: 8px;
}

.diff-better { color: #2ECC71; font-weight: 700; font-size: 0.8rem; }
.diff-worse  { color: #FF6B6B; font-weight: 700; font-size: 0.8rem; }
.diff-same   { color: #7BAFD4; font-weight: 600; font-size: 0.8rem; }

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

.url-pill {
    display: inline-block; background: #0D2036; border: 1px solid #2A4F7C; color: #7BAFD4;
    border-radius: 6px; padding: 2px 8px; font-size: 0.72rem; margin: 2px 4px 2px 0;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
for key in [
    "result_a", "result_a_urls", "result_a_time", "result_a_extra",
    "result_b", "result_b_urls", "result_b_time", "result_b_extra",
]:
    if key not in st.session_state:
        st.session_state[key] = None


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def score_to_color(score: int) -> str:
    """Buckets follow the same rubric described in qc_chain_manual.py's system prompt."""
    if score >= 9:
        return "#2ECC71"   # Excellent
    if score >= 7:
        return "#82E0AA"   # Good
    if score >= 5:
        return "#F4D03F"   # Average
    if score >= 3:
        return "#F39C12"   # Poor
    if score >= 1:
        return "#E74C3C"   # Very poor
    return "#922B21"       # Not performed


def get_scores_dict(agent_perf: dict) -> dict:
    return {m["metric_name"]: m["score"] for m in agent_perf["metrics"]}


def call_api(endpoint: str, urls: list, performance_types: list, timeout: int = 120) -> dict:
    payload = {"paths": urls, "performance_types": performance_types}
    resp = requests.post(endpoint, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def build_metric_card_html(m: dict, show_diff: int | None = None) -> str:
    score = m["score"]
    bar_pct = int(score / SCORE_MAX * 100)
    color = score_to_color(score)

    diff_html = ""
    if show_diff is not None:
        delta = score - show_diff
        if delta > 0:
            diff_html = f'<span style="color:#2ECC71;font-weight:700;font-size:13px;">&#9650; +{delta}</span>'
        elif delta < 0:
            diff_html = f'<span style="color:#FF6B6B;font-weight:700;font-size:13px;">&#9660; {delta}</span>'
        else:
            diff_html = '<span style="color:#7BAFD4;font-weight:600;font-size:13px;">= same</span>'

    icon = METRIC_ICONS.get(m["metric_name"], DEFAULT_ICON)
    observation = m["observation"].replace("<", "&lt;").replace(">", "&gt;")
    name = m["metric_name"].replace("<", "&lt;").replace(">", "&gt;")

    return (
        f'<div style="background:#162B45;border:1px solid #2A4F7C;border-left:4px solid {color};'
        f'border-radius:0 8px 8px 0;padding:12px 16px;margin-bottom:8px;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
        f'<span style="font-size:14px;font-weight:600;color:#C8D8E8;">{icon} {name}</span>'
        f'<span style="display:flex;gap:8px;align-items:center;">'
        f'{diff_html}'
        f'<span style="font-family:monospace;font-size:13px;font-weight:700;padding:2px 10px;'
        f'border-radius:20px;background:#0D1B2A;color:{color};border:1px solid {color};">'
        f'{score}/{SCORE_MAX}</span></span></div>'
        f'<div style="background:#0D1B2A;border-radius:4px;height:5px;margin-top:8px;overflow:hidden;">'
        f'<div style="width:{bar_pct}%;height:100%;border-radius:4px;background:{color};"></div></div>'
        f'<div style="font-size:12px;color:#8BA7C2;margin-top:8px;line-height:1.6;">{observation}</div>'
        f'</div>'
    )


def render_all_metrics(metrics: list, compare_scores: dict | None = None):
    """Rendered via components.html (iframe) to avoid Streamlit's HTML sanitizer stripping styles."""
    cards_html = ""
    for m in metrics:
        diff = compare_scores.get(m["metric_name"]) if compare_scores else None
        cards_html += build_metric_card_html(m, show_diff=diff)

    full_html = (
        "<!DOCTYPE html><html><head>"
        "<style>body{margin:0;padding:0;background:transparent;font-family:Inter,sans-serif;}</style>"
        f"</head><body>{cards_html}</body></html>"
    )
    components.html(full_html, height=max(1, len(metrics)) * 115, scrolling=False)


def render_result_header(agent_perf: dict, slot_label: str, urls: list, timestamp: str, flag_threshold: float):
    agent = agent_perf.get("agent_name") or "Unknown Agent"
    pct = agent_perf["percentage"]
    total = agent_perf["total_score"]
    possible = agent_perf["max_possible_score"]
    pass_class = "pass" if pct >= flag_threshold else "fail"
    flagged = pct < flag_threshold or any(m["score"] <= 2 for m in agent_perf["metrics"])
    badge = (
        '<span class="flag-badge-yes">⚠ Flagged for Review</span>'
        if flagged else
        '<span class="flag-badge-no">✓ No Issues Flagged</span>'
    )
    url_count = len(urls) if urls else 0

    st.markdown(f"""
    <div class="result-slot-header">
        <div class="slot-label">{slot_label}</div>
        <div class="slot-agent">🎧 {agent}</div>
        <div class="slot-meta">🔗 {url_count} recording(s) &nbsp;·&nbsp; 🕐 {timestamp} &nbsp;·&nbsp; 🎙️ Gemini native audio</div>
    </div>
    <div class="score-summary">
        <div style="display:flex;gap:2rem;align-items:flex-start;flex-wrap:wrap;">
            <div>
                <div class="big-score {pass_class}">{pct:.1f}%</div>
                <div class="score-label">Overall Score</div>
            </div>
            <div>
                <div class="big-score {pass_class}">{total}<span style="font-size:1.2rem;color:#7BAFD4;">/{possible}</span></div>
                <div class="score-label">Points Obtained</div>
            </div>
            <div style="align-self:center;">{badge}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    return flagged


def make_radar_chart(perf_a: dict, perf_b: dict, label_a: str, label_b: str) -> go.Figure:
    scores_a = get_scores_dict(perf_a)
    scores_b = get_scores_dict(perf_b)
    # Union of criteria, A's order first, then any B-only criteria appended
    categories = list(scores_a.keys()) + [c for c in scores_b.keys() if c not in scores_a]

    fig = go.Figure()
    for scores_d, label, color in [(scores_a, label_a, "#00A8E8"), (scores_b, label_b, "#F4D03F")]:
        vals = [scores_d.get(c, 0) for c in categories]
        vals_closed = vals + [vals[0]] if vals else vals
        cats_closed = categories + [categories[0]] if categories else categories
        fig.add_trace(go.Scatterpolar(
            r=vals_closed, theta=cats_closed, fill='toself', name=label,
            line=dict(color=color, width=2.5), opacity=0.85,
        ))
    for i, color in enumerate(["#00A8E8", "#F4D03F"]):
        rgb = tuple(int(color.lstrip('#')[j:j+2], 16) for j in (0, 2, 4))
        fig.data[i].fillcolor = f"rgba{rgb + (0.18,)}"

    fig.update_layout(
        polar=dict(
            bgcolor="#162B45",
            radialaxis=dict(visible=True, range=[0, SCORE_MAX], gridcolor="#2A4F7C", linecolor="#2A4F7C",
                             tickfont=dict(color="#7BAFD4", size=10)),
            angularaxis=dict(gridcolor="#2A4F7C", linecolor="#2A4F7C", tickfont=dict(color="#C8D8E8", size=11)),
        ),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color="#C8D8E8", size=12), bgcolor="rgba(22,43,69,0.8)", bordercolor="#2A4F7C", borderwidth=1),
        margin=dict(l=40, r=40, t=30, b=30), height=380,
    )
    return fig


def make_bar_chart(perf_a: dict, perf_b: dict, label_a: str, label_b: str) -> go.Figure:
    scores_a = get_scores_dict(perf_a)
    scores_b = get_scores_dict(perf_b)
    categories = list(scores_a.keys()) + [c for c in scores_b.keys() if c not in scores_a]

    fig = go.Figure()
    for scores_d, label, color in [(scores_a, label_a, "#00A8E8"), (scores_b, label_b, "#F4D03F")]:
        vals = [scores_d.get(c, 0) for c in categories]
        fig.add_trace(go.Bar(
            name=label,
            x=[f"{METRIC_ICONS.get(c, DEFAULT_ICON)} {c}" for c in categories],
            y=vals, marker_color=color, opacity=0.88, text=vals,
            textposition="outside", textfont=dict(color="#E8F0F7", size=11),
        ))
    fig.update_layout(
        barmode='group', paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#C8D8E8"),
        xaxis=dict(gridcolor="#1E3A5F", tickangle=-30, tickfont=dict(size=10)),
        yaxis=dict(gridcolor="#1E3A5F", range=[0, SCORE_MAX + 1], tickvals=list(range(0, SCORE_MAX + 1))),
        legend=dict(bgcolor="rgba(22,43,69,0.8)", bordercolor="#2A4F7C", borderwidth=1),
        margin=dict(l=20, r=20, t=20, b=60), height=320,
    )
    return fig


def make_gauge(pct: float, threshold: float) -> go.Figure:
    color = "#2ECC71" if pct >= threshold else "#FF6B6B"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=pct,
        number=dict(suffix="%", font=dict(size=28, color=color, family="JetBrains Mono")),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor="#2A4F7C", tickfont=dict(color="#7BAFD4", size=9)),
            bar=dict(color=color), bgcolor="#0D1B2A", borderwidth=1, bordercolor="#2A4F7C",
            steps=[dict(range=[0, threshold], color="#1A2F47"), dict(range=[threshold, 100], color="#1A3530")],
            threshold=dict(line=dict(color="#F4D03F", width=3), thickness=0.85, value=threshold),
        ),
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=10, b=10), height=180, font=dict(color="#C8D8E8"))
    return fig


def parse_urls(text: str, uploaded_file) -> list:
    if uploaded_file and not text:
        try:
            text = uploaded_file.getvalue().decode("utf-8")
        except Exception:
            text = uploaded_file.getvalue().decode("latin-1")
    return [u.strip() for u in text.splitlines() if u.strip()]


def run_slot(slot_key: str, urls: list, criteria: list, endpoint: str, timeout: int):
    with st.spinner(f"Analyzing {len(urls)} recording(s)…"):
        try:
            resp_json = call_api(endpoint, urls, criteria, timeout=timeout)
            st.session_state[f"result_{slot_key}"] = resp_json["agent_performance"]
            st.session_state[f"result_{slot_key}_urls"] = urls
            st.session_state[f"result_{slot_key}_time"] = datetime.now().strftime("%d %b %Y · %H:%M")
            st.session_state[f"result_{slot_key}_extra"] = {
                "audio_analysis_bearer": resp_json.get("audio_analysis_bearer"),
                "failed_urls": resp_json.get("failed_urls") or [],
            }
            st.success(f"Slot {slot_key.upper()} analyzed ✓")
        except requests.exceptions.HTTPError as e:
            detail = ""
            try:
                detail = e.response.json().get("detail", "")
            except Exception:
                pass
            st.error(f"API error: {detail or e}")
        except Exception as e:
            st.error(f"Error: {e}")


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 0.3rem 0 1.2rem 0;">
        <div style="font-size:1.15rem;font-weight:700;color:#FFFFFF;">🎧 AQC System</div>
        <div style="font-size:0.75rem;color:#7BAFD4;margin-top:2px;">General Purpose · Manual Criteria · Native Audio</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("⚙️ Connection settings", expanded=False):
        api_url = st.text_input("API URL", value=DEFAULT_API_URL)
        timeout = st.number_input("Request timeout (s)", min_value=10, max_value=300, value=120)
        flag_threshold = st.slider("Flag-for-review threshold (%)", min_value=50, max_value=95, value=70, step=5)

    st.markdown('<div class="section-label">Slot A — Primary Agent</div>', unsafe_allow_html=True)
    urls_text_a = st.text_area(
        "Call recording URLs (one per line)", height=110, key="urls_a",
        placeholder="https://.../call1.mp3\nhttps://.../call2.mp3",
    )
    file_a = st.file_uploader("Or upload a .txt file with one URL per line", type=["txt"], key="file_a")
    criteria_a = st.multiselect(
        "Evaluation criteria", options=DEFAULT_METRICS, default=DEFAULT_METRICS[:6], key="criteria_a",
    )
    custom_a = st.text_input("Add custom criterion (optional)", key="custom_a")
    if custom_a and custom_a not in criteria_a:
        criteria_a = criteria_a + [custom_a]

    if st.button("▶ Analyze Agent A", key="btn_a"):
        parsed_urls_a = parse_urls(urls_text_a, file_a)
        if not parsed_urls_a:
            st.warning("Provide at least one audio URL for Slot A.")
        elif not criteria_a:
            st.warning("Select at least one evaluation criterion for Slot A.")
        else:
            run_slot("a", parsed_urls_a, criteria_a, api_url, timeout)

    st.markdown('<div class="section-label">Slot B — Comparison Agent</div>', unsafe_allow_html=True)
    urls_text_b = st.text_area(
        "Call recording URLs (one per line)", height=110, key="urls_b",
        placeholder="https://.../call1.mp3\nhttps://.../call2.mp3",
    )
    file_b = st.file_uploader("Or upload a .txt file with one URL per line", type=["txt"], key="file_b")
    criteria_b = st.multiselect(
        "Evaluation criteria", options=DEFAULT_METRICS, default=DEFAULT_METRICS[:6], key="criteria_b",
    )
    custom_b = st.text_input("Add custom criterion (optional)", key="custom_b")
    if custom_b and custom_b not in criteria_b:
        criteria_b = criteria_b + [custom_b]

    if st.button("▶ Analyze Agent B", key="btn_b"):
        parsed_urls_b = parse_urls(urls_text_b, file_b)
        if not parsed_urls_b:
            st.warning("Provide at least one audio URL for Slot B.")
        elif not criteria_b:
            st.warning("Select at least one evaluation criterion for Slot B.")
        else:
            run_slot("b", parsed_urls_b, criteria_b, api_url, timeout)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear A", use_container_width=True):
            for k in ["result_a", "result_a_urls", "result_a_time", "result_a_extra"]:
                st.session_state[k] = None
            st.rerun()
    with col2:
        if st.button("Clear B", use_container_width=True):
            for k in ["result_b", "result_b_urls", "result_b_time", "result_b_extra"]:
                st.session_state[k] = None
            st.rerun()

    st.markdown("---")
    st.markdown(f"""
    <div style="font-size:0.72rem;color:#4A6A8A;line-height:1.8;">
        <b style="color:#7BAFD4;">API Endpoint</b><br>
        <code style="color:#00A8E8;font-size:0.7rem;word-break:break-all;">{api_url}</code><br><br>
        <b style="color:#7BAFD4;">Flag Threshold</b><br>
        Below {flag_threshold}% or any criterion ≤ 2/{SCORE_MAX}<br><br>
        <b style="color:#7BAFD4;">Score Scale</b><br>
        0–{SCORE_MAX} per criterion
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
st.markdown("""
<div class="aqc-header">
    <div>
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
            <div class="aqc-header-title">Agent Performance QC</div>
            <span class="general-badge">General Purpose</span>
        </div>
        <div class="aqc-header-sub">AI-powered evaluation · your own criteria · Gemini native audio understanding · multi-call analysis</div>
    </div>
</div>
""", unsafe_allow_html=True)

result_a = st.session_state.result_a
result_b = st.session_state.result_b
has_a = result_a is not None
has_b = result_b is not None

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
            <div class="empty-title">No agents analyzed yet</div>
            <div class="empty-sub">Add recording URLs and criteria in the sidebar, then click <b>Analyze</b> to begin.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        cols = st.columns([1, 1]) if (has_a and has_b) else [st.container()]

        if has_a:
            with cols[0]:
                render_result_header(
                    result_a, "Slot A · Primary",
                    st.session_state.result_a_urls, st.session_state.result_a_time, flag_threshold,
                )
                st.plotly_chart(make_gauge(result_a["percentage"], flag_threshold), use_container_width=True, key="gauge_a")

                extra_a = st.session_state.result_a_extra or {}
                if extra_a.get("failed_urls"):
                    st.warning(f"{len(extra_a['failed_urls'])} URL(s) could not be downloaded and were skipped:\n\n" +
                               "\n".join(f"- {u}" for u in extra_a["failed_urls"]))

                st.markdown('<div class="section-label">Performance Summary</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="summary-box">{result_a["performance_summary"]}</div>', unsafe_allow_html=True)

                sc1, sc2 = st.columns(2)
                with sc1:
                    if result_a.get("strengths"):
                        st.markdown('<div class="section-label">Strengths</div>', unsafe_allow_html=True)
                        st.markdown('<div class="summary-box" style="border-color:#2ECC71;">' +
                                    "<br>".join(f"✓ {s}" for s in result_a["strengths"]) + '</div>', unsafe_allow_html=True)
                with sc2:
                    if result_a.get("weaknesses"):
                        st.markdown('<div class="section-label">Weaknesses</div>', unsafe_allow_html=True)
                        st.markdown('<div class="summary-box" style="border-color:#FF6B6B;">' +
                                    "<br>".join(f"✗ {w}" for w in result_a["weaknesses"]) + '</div>', unsafe_allow_html=True)

                if result_a.get("improvement_suggestions"):
                    st.markdown('<div class="section-label">Improvement Suggestions</div>', unsafe_allow_html=True)
                    st.markdown('<div class="summary-box">' +
                                "<br>".join(f"{i}. {s}" for i, s in enumerate(result_a["improvement_suggestions"], 1)) +
                                '</div>', unsafe_allow_html=True)

                st.markdown('<div class="section-label">Criteria Breakdown</div>', unsafe_allow_html=True)
                scores_b_dict = get_scores_dict(result_b) if has_b else None
                render_all_metrics(result_a["metrics"], compare_scores=scores_b_dict)

        if has_b:
            with cols[1] if has_a else cols[0]:
                render_result_header(
                    result_b, "Slot B · Comparison",
                    st.session_state.result_b_urls, st.session_state.result_b_time, flag_threshold,
                )
                st.plotly_chart(make_gauge(result_b["percentage"], flag_threshold), use_container_width=True, key="gauge_b")

                extra_b = st.session_state.result_b_extra or {}
                if extra_b.get("failed_urls"):
                    st.warning(f"{len(extra_b['failed_urls'])} URL(s) could not be downloaded and were skipped:\n\n" +
                               "\n".join(f"- {u}" for u in extra_b["failed_urls"]))

                st.markdown('<div class="section-label">Performance Summary</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="summary-box">{result_b["performance_summary"]}</div>', unsafe_allow_html=True)

                sc1, sc2 = st.columns(2)
                with sc1:
                    if result_b.get("strengths"):
                        st.markdown('<div class="section-label">Strengths</div>', unsafe_allow_html=True)
                        st.markdown('<div class="summary-box" style="border-color:#2ECC71;">' +
                                    "<br>".join(f"✓ {s}" for s in result_b["strengths"]) + '</div>', unsafe_allow_html=True)
                with sc2:
                    if result_b.get("weaknesses"):
                        st.markdown('<div class="section-label">Weaknesses</div>', unsafe_allow_html=True)
                        st.markdown('<div class="summary-box" style="border-color:#FF6B6B;">' +
                                    "<br>".join(f"✗ {w}" for w in result_b["weaknesses"]) + '</div>', unsafe_allow_html=True)

                if result_b.get("improvement_suggestions"):
                    st.markdown('<div class="section-label">Improvement Suggestions</div>', unsafe_allow_html=True)
                    st.markdown('<div class="summary-box">' +
                                "<br>".join(f"{i}. {s}" for i, s in enumerate(result_b["improvement_suggestions"], 1)) +
                                '</div>', unsafe_allow_html=True)

                st.markdown('<div class="section-label">Criteria Breakdown</div>', unsafe_allow_html=True)
                scores_a_dict = get_scores_dict(result_a) if has_a else None
                render_all_metrics(result_b["metrics"], compare_scores=scores_a_dict)

        if has_a and not has_b:
            st.markdown("""
            <div class="empty-state" style="margin-top:1rem;">
                <div class="empty-icon">⚖️</div>
                <div class="empty-title">Compare with a second agent</div>
                <div class="empty-sub">Fill in Slot B in the sidebar to enable side-by-side comparison and score delta indicators.</div>
            </div>
            """, unsafe_allow_html=True)

        if has_a and has_b and get_scores_dict(result_a).keys() != get_scores_dict(result_b).keys():
            st.info("Slot A and Slot B used different criteria sets — comparison charts will show 0 for any criterion an agent wasn't scored on.")


# ─────────────────────────────────────────────
# TAB: COMPARISON
# ─────────────────────────────────────────────
if has_a and has_b and tab_compare is not None:
    with tab_compare:
        agent_a = result_a.get("agent_name") or "Recording A"
        agent_b = result_b.get("agent_name") or "Recording B"
        label_a, label_b = f"{agent_a} (A)", f"{agent_b} (B)"

        st.markdown('<div class="section-label">Performance Radar — Head to Head</div>', unsafe_allow_html=True)
        st.plotly_chart(make_radar_chart(result_a, result_b, label_a, label_b), use_container_width=True, key="radar_cmp")

        st.markdown('<div class="section-label">Score by Criterion — Side by Side</div>', unsafe_allow_html=True)
        st.plotly_chart(make_bar_chart(result_a, result_b, label_a, label_b), use_container_width=True, key="bar_cmp")

        st.markdown('<div class="section-label">Score Delta Table</div>', unsafe_allow_html=True)
        scores_a = get_scores_dict(result_a)
        scores_b = get_scores_dict(result_b)
        all_criteria = list(scores_a.keys()) + [c for c in scores_b.keys() if c not in scores_a]
        rows = []
        for c in all_criteria:
            sa, sb = scores_a.get(c, 0), scores_b.get(c, 0)
            delta = sa - sb
            rows.append({
                "Criterion": f"{METRIC_ICONS.get(c, DEFAULT_ICON)} {c}",
                label_a: sa, label_b: sb,
                "Delta (A−B)": f"+{delta}" if delta > 0 else str(delta),
                "Winner": label_a if sa > sb else (label_b if sb > sa else "Tie"),
            })
        df = pd.DataFrame(rows)

        def style_delta(val):
            try:
                v = int(val)
                if v > 0: return "color: #2ECC71; font-weight: bold"
                if v < 0: return "color: #FF6B6B; font-weight: bold"
                return "color: #7BAFD4"
            except Exception:
                return ""

        styled = df.style.apply(lambda col: [style_delta(v) for v in col], subset=["Delta (A−B)"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        st.markdown('<div class="section-label">Summary Comparison</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Score A", f"{result_a['percentage']:.1f}%", delta=f"{result_a['percentage'] - result_b['percentage']:+.1f}%")
        with m2:
            st.metric("Score B", f"{result_b['percentage']:.1f}%", delta=f"{result_b['percentage'] - result_a['percentage']:+.1f}%")
        with m3:
            wins_a = sum(1 for c in all_criteria if scores_a.get(c, 0) > scores_b.get(c, 0))
            st.metric("Criteria Won (A)", f"{wins_a}/{len(all_criteria)}")
        with m4:
            wins_b = sum(1 for c in all_criteria if scores_b.get(c, 0) > scores_a.get(c, 0))
            st.metric("Criteria Won (B)", f"{wins_b}/{len(all_criteria)}")

        st.markdown('<div class="section-label">Review Flag Comparison</div>', unsafe_allow_html=True)
        flagged_a = result_a["percentage"] < flag_threshold or any(m["score"] <= 2 for m in result_a["metrics"])
        flagged_b = result_b["percentage"] < flag_threshold or any(m["score"] <= 2 for m in result_b["metrics"])
        cc1, cc2 = st.columns(2)
        with cc1:
            if flagged_a:
                st.markdown(f'<div class="summary-box" style="border-color:#FF6B6B;"><b style="color:#FF6B6B;">⚠ {label_a}: Flagged for Review</b></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="summary-box" style="border-color:#2ECC71;"><b style="color:#2ECC71;">✓ {label_a}: No Issues Flagged</b></div>', unsafe_allow_html=True)
        with cc2:
            if flagged_b:
                st.markdown(f'<div class="summary-box" style="border-color:#FF6B6B;"><b style="color:#FF6B6B;">⚠ {label_b}: Flagged for Review</b></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="summary-box" style="border-color:#2ECC71;"><b style="color:#2ECC71;">✓ {label_b}: No Issues Flagged</b></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TAB: ABOUT
# ─────────────────────────────────────────────
with tab_about:
    st.markdown(f"""
    <div class="summary-box" style="font-size:0.88rem;line-height:2;">
        <b style="color:#00A8E8;font-size:1rem;">Agent Performance QC — General Purpose</b><br><br>
        This tool sends call recordings directly to Gemini as <b style="color:#C8D8E8;">native audio</b> —
        no separate transcription step — so scoring accounts for tone, pacing, and emotion, not just
        the words spoken. Unlike the fixed MetLife rubric, <b style="color:#C8D8E8;">you choose the criteria</b>.<br><br>

        <b style="color:#C8D8E8;">How to use:</b><br>
        1. Paste one or more call recording URLs for an agent into <b>Slot A</b> in the sidebar<br>
        2. Choose the evaluation criteria (or add your own custom criterion)<br>
        3. Click <b>Analyze Agent A</b> — Gemini evaluates all recordings together, looking for patterns<br>
        4. Optionally fill in <b>Slot B</b> with a second agent (or the same agent later) to compare<br>
        5. Switch to the <b>⚖️ Comparison</b> tab for side-by-side charts and score deltas<br><br>

        <b style="color:#C8D8E8;">Scoring:</b> each criterion is scored 0–{SCORE_MAX}
        (0-2 Very Poor · 3-4 Poor · 5-6 Average · 7-8 Good · 9-{SCORE_MAX} Excellent).
        Totals and percentages are computed server-side from the raw scores, never trusted from the model directly.<br><br>

        <b style="color:#C8D8E8;">Review Flag:</b> an agent is flagged if their overall score falls below your
        chosen threshold (sidebar) or if any single criterion scores 2 or lower.<br><br>

        <b style="color:#C8D8E8;">Note:</b> if URLs fail to download, they're skipped and listed under
        the result — the evaluation still runs on whatever recordings succeeded.
    </div>
    """, unsafe_allow_html=True)