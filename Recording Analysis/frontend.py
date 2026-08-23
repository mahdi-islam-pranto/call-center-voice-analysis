"""
Recording Analysis — Streamlit Frontend
Upload call recordings and get AI-powered structured analysis including
summaries, sentiment, keywords, action items, and more.
"""

import streamlit as st
import streamlit.components.v1 as components
import requests
import time
from datetime import datetime

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Recording Analysis — AI Call Analyzer",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_API_URL = "http://localhost:8000/crm/analyze-call"

# ─────────────────────────────────────────────
# CUSTOM CSS
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

.ra-header {
    background: linear-gradient(135deg, #1E3A5F 0%, #0D2847 100%);
    border: 1px solid #2A4F7C;
    border-radius: 12px;
    padding: 1.4rem 2rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.ra-header-title { font-size: 1.6rem; font-weight: 700; color: #FFFFFF; letter-spacing: -0.02em; }
.ra-header-sub { font-size: 0.85rem; color: #7BAFD4; margin-top: 2px; }
.ra-badge {
    background: #00A8E8;
    color: #fff;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.section-label {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase;
    color: #00A8E8; margin: 1.4rem 0 0.6rem 0; padding-bottom: 4px; border-bottom: 1px solid #1E3A5F;
}

.card {
    background: #162B45;
    border: 1px solid #2A4F7C;
    border-radius: 10px;
    padding: 1.2rem;
    margin-bottom: 0.8rem;
}
.card-accent {
    border-left: 4px solid #00A8E8;
}

.summary-box {
    background: #0D2036;
    border: 1px solid #2A4F7C;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    font-size: 0.85rem;
    color: #A8C4DC;
    line-height: 1.8;
    margin-top: 0.5rem;
}

.sentiment-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.04em;
}
.sentiment-positive { background: rgba(46,204,113,0.15); border: 1px solid #2ECC71; color: #2ECC71; }
.sentiment-neutral { background: rgba(123,175,212,0.15); border: 1px solid #7BAFD4; color: #7BAFD4; }
.sentiment-negative { background: rgba(255,107,107,0.15); border: 1px solid #FF6B6B; color: #FF6B6B; }
.sentiment-mixed { background: rgba(244,168,37,0.15); border: 1px solid #F4A825; color: #F4A825; }

.outcome-badge {
    display: inline-block;
    background: rgba(0,168,232,0.12);
    border: 1px solid #00A8E8;
    color: #00A8E8;
    border-radius: 6px;
    padding: 5px 12px;
    font-size: 0.8rem;
    font-weight: 600;
}

.followup-yes {
    background: rgba(255,107,107,0.12); border: 1px solid #FF6B6B; color: #FF6B6B;
    border-radius: 6px; padding: 5px 12px; font-size: 0.8rem; font-weight: 600;
    display: inline-block;
}
.followup-no {
    background: rgba(46,204,113,0.12); border: 1px solid #2ECC71; color: #2ECC71;
    border-radius: 6px; padding: 5px 12px; font-size: 0.8rem; font-weight: 600;
    display: inline-block;
}

.keyword-pill {
    display: inline-block;
    background: #0D2036;
    border: 1px solid #2A4F7C;
    color: #7BAFD4;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.75rem;
    margin: 3px 4px 3px 0;
}

.list-item {
    padding: 6px 0;
    font-size: 0.85rem;
    color: #A8C4DC;
    line-height: 1.6;
    border-bottom: 1px solid rgba(42,79,124,0.4);
}
.list-item:last-child { border-bottom: none; }

.token-stat {
    text-align: center;
    padding: 0.8rem;
}
.token-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.3rem;
    font-weight: 700;
    color: #00A8E8;
}
.token-label {
    font-size: 0.7rem;
    color: #7BAFD4;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 2px;
}

.empty-state {
    background: #162B45; border: 1px dashed #2A4F7C; border-radius: 12px;
    padding: 2.5rem; text-align: center; color: #7BAFD4;
}
.empty-icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
.empty-title { font-size: 1rem; font-weight: 600; color: #C8D8E8; }
.empty-sub { font-size: 0.82rem; margin-top: 4px; }

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
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "analysis_time" not in st.session_state:
    st.session_state.analysis_time = None
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def get_sentiment_color(sentiment: str) -> str:
    return {
        "positive": "#2ECC71",
        "neutral": "#7BAFD4",
        "negative": "#FF6B6B",
        "mixed": "#F4A825",
    }.get(sentiment, "#7BAFD4")


def build_keyword_pills_html(keywords: list) -> str:
    pills = "".join(
        f'<span class="keyword-pill">{kw}</span>' for kw in keywords
    )
    return f'<div style="line-height:2.2;">{pills}</div>'


def build_list_html(items: list, icon: str = "•") -> str:
    if not items:
        return '<div style="color:#4A6A8A;font-size:0.82rem;font-style:italic;">None identified</div>'
    rows = "".join(
        f'<div class="list-item">{icon} {item}</div>' for item in items
    )
    return rows


def call_analysis_api(api_url: str, audio_bytes: bytes, filename: str, mime_type: str,
                      language: str, context: dict, timeout: int) -> dict:
    """Call the Recording Analysis API."""
    files = {"audio_file": (filename, audio_bytes, mime_type)}
    data = {"language": language}
    data.update({k: v for k, v in context.items() if v})

    resp = requests.post(api_url, files=files, data=data, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def render_result(result: dict):
    """Render the full analysis result."""
    analysis = result.get("analysis", {})
    token_usage = result.get("token_usage")

    # ── Top Row: Sentiment + Outcome + Follow-up ──
    st.markdown('<div class="section-label">Call Overview</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        sentiment = analysis.get("customer_sentiment", "neutral")
        color = get_sentiment_color(sentiment)
        st.markdown(f"""
        <div class="card">
            <div style="font-size:0.7rem;color:#7BAFD4;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">Customer Sentiment</div>
            <div class="sentiment-badge sentiment-{sentiment}">{sentiment.upper()}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        outcome = analysis.get("call_outcome", "—")
        st.markdown(f"""
        <div class="card">
            <div style="font-size:0.7rem;color:#7BAFD4;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">Call Outcome</div>
            <div class="outcome-badge">{outcome}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        follow_up = analysis.get("follow_up_required", False)
        followup_badge = '<span class="followup-yes">⚠ FOLLOW-UP REQUIRED</span>' if follow_up else '<span class="followup-no">✓ NO FOLLOW-UP NEEDED</span>'
        st.markdown(f"""
        <div class="card">
            <div style="font-size:0.7rem;color:#7BAFD4;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">Follow-up Status</div>
            {followup_badge}
        </div>
        """, unsafe_allow_html=True)

    # ── Summary ──
    st.markdown('<div class="section-label">Call Summary</div>', unsafe_allow_html=True)
    summary = analysis.get("summary", "No summary available.")
    st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)

    # ── Keywords ──
    keywords = analysis.get("keywords", [])
    if keywords:
        st.markdown('<div class="section-label">Keywords</div>', unsafe_allow_html=True)
        st.markdown(build_keyword_pills_html(keywords), unsafe_allow_html=True)

    # ── Two Column Layout for Details ──
    left, right = st.columns(2)

    with left:
        # ── Pain Points ──
        st.markdown('<div class="section-label">Customer Pain Points</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="card">{build_list_html(analysis.get("customer_pain_points", []), "🔴")}</div>', unsafe_allow_html=True)

        # ── Products/Services Discussed ──
        st.markdown('<div class="section-label">Products / Services Discussed</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="card">{build_list_html(analysis.get("products_services_discussed", []), "📦")}</div>', unsafe_allow_html=True)

        # ── Follow-up Notes ──
        follow_up_notes = analysis.get("follow_up_notes")
        if follow_up_notes:
            st.markdown('<div class="section-label">Follow-up Notes</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="card card-accent" style="border-left-color:#F4A825;"><div style="font-size:0.85rem;color:#A8C4DC;line-height:1.7;">{follow_up_notes}</div></div>', unsafe_allow_html=True)

    with right:
        # ── Objections ──
        st.markdown('<div class="section-label">Objections Raised</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="card">{build_list_html(analysis.get("objections_raised", []), "⚠️")}</div>', unsafe_allow_html=True)

        # ── Action Items ──
        st.markdown('<div class="section-label">Action Items</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="card">{build_list_html(analysis.get("action_items", []), "✅")}</div>', unsafe_allow_html=True)

        # ── Important Notes ──
        important_notes = analysis.get("important_notes")
        if important_notes:
            st.markdown('<div class="section-label">Important Notes</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="card card-accent" style="border-left-color:#FF6B6B;"><div style="font-size:0.85rem;color:#A8C4DC;line-height:1.7;">{important_notes}</div></div>', unsafe_allow_html=True)

    # ── Token Usage ──
    if token_usage:
        st.markdown('<div class="section-label">Token Usage</div>', unsafe_allow_html=True)
        tc1, tc2, tc3, tc4 = st.columns(4)
        with tc1:
            st.markdown(f"""
            <div class="card token-stat">
                <div class="token-value">{token_usage.get("input_tokens", 0):,}</div>
                <div class="token-label">Input Tokens</div>
            </div>
            """, unsafe_allow_html=True)
        with tc2:
            st.markdown(f"""
            <div class="card token-stat">
                <div class="token-value">{token_usage.get("output_tokens", 0):,}</div>
                <div class="token-label">Output Tokens</div>
            </div>
            """, unsafe_allow_html=True)
        with tc3:
            st.markdown(f"""
            <div class="card token-stat">
                <div class="token-value">{token_usage.get("total_tokens", 0):,}</div>
                <div class="token-label">Total Tokens</div>
            </div>
            """, unsafe_allow_html=True)
        with tc4:
            audio_tok = token_usage.get("audio_tokens")
            audio_tok_display = f"{audio_tok:,}" if audio_tok else "—"
            st.markdown(f"""
            <div class="card token-stat">
                <div class="token-value">{audio_tok_display}</div>
                <div class="token-label">Audio Tokens</div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 0.3rem 0 1.2rem 0;">
        <div style="font-size:1.15rem;font-weight:700;color:#FFFFFF;">🎙️ Recording Analyzer</div>
        <div style="font-size:0.75rem;color:#7BAFD4;margin-top:2px;">AI-Powered Call Analysis</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("⚙️ Connection Settings", expanded=False):
        api_url = st.text_input("API URL", value=DEFAULT_API_URL, placeholder="http://localhost:8000/crm/analyze-call")
        timeout = st.number_input("Request Timeout (s)", min_value=30, max_value=600, value=180)

    st.markdown('<div class="section-label">Upload Recording</div>', unsafe_allow_html=True)

    audio_file = st.file_uploader(
        "Upload Audio File",
        type=["mp3", "wav", "ogg", "m4a", "flac", "wma", "aac"],
        help="Supported formats: MP3, WAV, OGG, M4A, FLAC, WMA, AAC. Max size: 200MB",
    )

    if audio_file:
        file_size_mb = audio_file.size / (1024 * 1024)
        st.info(f"📎 **{audio_file.name}** ({file_size_mb:.1f} MB)")

        if file_size_mb > 200:
            st.error("File exceeds 200MB limit. Please upload a smaller file.")
        else:
            st.audio(audio_file, format=audio_file.type)

    st.markdown('<div class="section-label">Output Language</div>', unsafe_allow_html=True)

    language = st.radio(
        "Summary Language",
        options=["English", "Bangla"],
        index=0,
        horizontal=True,
        help="The analysis text will be written in this language regardless of the audio language.",
    )
    language_code = "en" if language == "English" else "bn"

    st.markdown('<div class="section-label">CRM Context (Optional)</div>', unsafe_allow_html=True)

    with st.expander("📋 Add Context from CRM", expanded=False):
        agent_name = st.text_input("Agent Name", placeholder="e.g. John Smith")
        customer_name = st.text_input("Customer Name", placeholder="e.g. Jane Doe")
        call_direction = st.selectbox("Call Direction", options=["— Select —", "Outbound", "Inbound"], index=0)
        deal_or_lead_id = st.text_input("Deal / Lead ID", placeholder="e.g. DEAL-12345")
        product_or_service = st.text_input("Product / Service", placeholder="e.g. Enterprise CRM Suite")

    st.markdown("---")

    can_analyze = (
        audio_file is not None
        and (audio_file.size / (1024 * 1024)) <= 200
    )

    analyze_clicked = st.button(
        "▶ Analyze Recording",
        disabled=not can_analyze,
        use_container_width=True,
    )

    st.markdown("---")

    if st.session_state.analysis_result:
        if st.button("🗑️ Clear Result", use_container_width=True):
            st.session_state.analysis_result = None
            st.session_state.analysis_time = None
            st.session_state.uploaded_filename = None
            st.rerun()

    st.markdown(f"""
    <div style="font-size:0.72rem;color:#4A6A8A;line-height:1.8;margin-top:0.5rem;">
        <b style="color:#7BAFD4;">API Endpoint</b><br>
        <code style="color:#00A8E8;font-size:0.7rem;word-break:break-all;">{api_url}</code><br><br>
        <b style="color:#7BAFD4;">Max File Size</b><br>
        200 MB<br><br>
        <b style="color:#7BAFD4;">Supported Formats</b><br>
        MP3, WAV, OGG, M4A, FLAC, WMA, AAC
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
st.markdown("""
<div class="ra-header">
    <div>
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
            <div class="ra-header-title">Recording Analysis</div>
            <span class="ra-badge">AI-Powered</span>
        </div>
        <div class="ra-header-sub">Upload a call recording → get structured insights · sentiment · keywords · action items · more</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Handle Analysis ──
if analyze_clicked and audio_file:
    # Validate
    file_size_mb = audio_file.size / (1024 * 1024)
    if file_size_mb > 200:
        st.error("File exceeds 200MB limit.")
        st.stop()

    context = {
        "agent_name": agent_name if agent_name else None,
        "customer_name": customer_name if customer_name else None,
        "call_direction": call_direction if call_direction != "— Select —" else None,
        "deal_or_lead_id": deal_or_lead_id if deal_or_lead_id else None,
        "product_or_service": product_or_service if product_or_service else None,
    }

    audio_bytes = audio_file.read()
    mime_type = audio_file.type or "audio/mpeg"

    start_time = time.time()

    with st.spinner("🎙️ Analyzing recording... This may take a minute."):
        try:
            result = call_analysis_api(
                api_url=api_url,
                audio_bytes=audio_bytes,
                filename=audio_file.name,
                mime_type=mime_type,
                language=language_code,
                context=context,
                timeout=timeout,
            )
            elapsed = time.time() - start_time
            st.session_state.analysis_result = result
            st.session_state.analysis_time = datetime.now().strftime("%d %b %Y · %H:%M")
            st.session_state.uploaded_filename = audio_file.name
            st.success(f"Analysis complete in {elapsed:.1f}s")
        except requests.exceptions.Timeout:
            st.error("Request timed out. The audio may be too long or the server is slow. Try again or increase the timeout.")
            st.stop()
        except requests.exceptions.HTTPError as e:
            detail = ""
            try:
                detail = e.response.json().get("detail", "")
            except Exception:
                detail = e.text[:300]
            st.error(f"API Error ({e.response.status_code}): {detail}")
            st.stop()
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the API server. Make sure the backend is running.")
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            st.stop()


# ── Display Results ──
result = st.session_state.analysis_result

if result:
    tabs = st.tabs(["📊 Analysis Results", "📄 Raw JSON"])

    with tabs[0]:
        # Metadata bar
        filename = st.session_state.uploaded_filename or "recording"
        analysis_time = st.session_state.analysis_time or "—"
        lang_label = "English" if result.get("language") == "en" else "Bangla"

        st.markdown(f"""
        <div style="background:#162B45;border:1px solid #2A4F7C;border-radius:8px;padding:0.6rem 1rem;margin-bottom:1rem;display:flex;gap:2rem;font-size:0.78rem;color:#7BAFD4;">
            <span>📎 <b style="color:#C8D8E8;">{filename}</b></span>
            <span>🌐 Output: <b style="color:#00A8E8;">{lang_label}</b></span>
            <span>🕐 {analysis_time}</span>
        </div>
        """, unsafe_allow_html=True)

        render_result(result)

    with tabs[1]:
        st.json(result)

else:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">🎙️</div>
        <div class="empty-title">No recording analyzed yet</div>
        <div class="empty-sub">Upload an audio file in the sidebar and click <b>Analyze Recording</b> to begin.</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ABOUT / HOW IT WORKS
# ─────────────────────────────────────────────
with st.expander("ℹ️ How It Works", expanded=False):
    st.markdown("""
    <div style="font-size:0.85rem;color:#A8C4DC;line-height:1.9;">
    <b style="color:#00A8E8;">1. Upload</b> — Select an audio file from your computer (MP3, WAV, OGG, M4A, etc.).<br>
    <b style="color:#00A8E8;">2. Configure</b> — Choose the output language and optionally add CRM context (agent name, customer, deal ID, etc.).<br>
    <b style="color:#00A8E8;">3. Analyze</b> — The audio is sent to Google's Gemini AI which listens to the full recording and extracts structured insights.<br>
    <b style="color:#00A8E8;">4. Results</b> — Get a comprehensive analysis including summary, sentiment, keywords, pain points, objections, action items, and follow-up recommendations.<br><br>

    <b style="color:#C8D8E8;">Language Support:</b><br>
    The call audio can be in Bangla, English, or a code-mixed blend of both. The output analysis is written in whichever language you select, regardless of the audio language.<br><br>

    <b style="color:#C8D8E8;">Privacy:</b><br>
    Audio files are temporarily uploaded to Google's servers for analysis and automatically deleted immediately after processing. No audio data is stored permanently.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.write("Run locally:")
st.code("pip install streamlit requests\nstreamlit run \"Recording Analysis/frontend.py\"", language="bash")
