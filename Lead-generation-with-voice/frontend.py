import streamlit as st
import requests
import tempfile
import os
import json
import time
from datetime import datetime

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Lead Detection AI",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# STYLES
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .stApp { background-color: #FFFFFF; }
        section[data-testid="stSidebar"] { background-color: #F7F7F9; border-right: 1px solid #E5E7EB; }
        section[data-testid="stSidebar"] * { color: #1F2430; }

        .hero {
            background: linear-gradient(120deg, #7C3AED 0%, #4338CA 100%);
            padding: 2.2rem 2.5rem;
            border-radius: 18px;
            margin-bottom: 1.6rem;
            color: white;
        }
        .hero h1 { margin: 0 0 0.3rem 0; font-size: 2.1rem; color: white; }
        .hero p { margin: 0; opacity: 0.92; font-size: 1.02rem; }

        .metric-card {
            background: #FAFAFC;
            border: 1px solid #E5E7EB;
            border-radius: 14px;
            padding: 1.1rem 1.3rem;
            text-align: center;
        }
        .metric-card .label { color: #6B7280; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.04em; }
        .metric-card .value { font-size: 1.7rem; font-weight: 700; color: #1F2430; margin-top: 0.2rem; }

        .badge {
            display: inline-block;
            padding: 0.35rem 1rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.95rem;
            letter-spacing: 0.01em;
        }
        .badge-strong     { background: #DCFCE7; color: #15803D; border: 1px solid #4ADE80; }
        .badge-potential  { background: #DBEAFE; color: #1D4ED8; border: 1px solid #60A5FA; }
        .badge-weak       { background: #FEF3C7; color: #B45309; border: 1px solid #FBBF24; }
        .badge-not        { background: #FEE2E2; color: #B91C1C; border: 1px solid #F87171; }

        .chip {
            display: inline-block;
            background: #EEF2FF;
            color: #4338CA;
            border: 1px solid #C7D2FE;
            padding: 0.25rem 0.75rem;
            margin: 0.2rem 0.3rem 0.2rem 0;
            border-radius: 999px;
            font-size: 0.85rem;
        }

        .justification-box {
            background: #FAFAFC;
            border-left: 4px solid #7C3AED;
            border-radius: 8px;
            padding: 1rem 1.2rem;
            color: #1F2430;
            font-size: 0.98rem;
            line-height: 1.5;
        }

        .section-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #1F2430;
            margin: 1.4rem 0 0.6rem 0;
        }

        .history-row {
            background: #FAFAFC;
            border: 1px solid #E5E7EB;
            border-radius: 10px;
            padding: 0.7rem 1rem;
            margin-bottom: 0.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ----------------------------------------------------------------------------
# HERO HEADER
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>📦 Call Center Lead Detection AI</h1>
        <p>Upload or link a call recording — the AI listens, understands Bangla &amp; English,
        and tells you whether the caller is a sales-ready lead, with full justification.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# SIDEBAR — SETTINGS
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    api_url = st.text_input(
        "FastAPI endpoint URL",
        value="http://138.252.115.100:6002/find_lead",
        placeholder="http://138.252.115.100:6002/find_lead",
    )
    api_key = st.text_input("Optional API key (Bearer)", value="", type="password")
    timeout = st.number_input("Request timeout (s)", min_value=10, max_value=300, value=60)

    st.markdown("---")
    st.header("🎯 Quick Demo Samples")
    st.caption("Pre-loaded call recordings for a fast, reliable demo.")

    demo_samples = {
        "— Select a sample —": "",
        "✅ Strong Lead (price inquiry)": "https://103.204.81.3/RECORDINGS/MP3/20260615-100751_FEDEX_TN_FEDEX_louis_8801862901607-all.mp3",
        "❌ Not a Lead (irrelevant query)": "https://103.204.81.3/RECORDINGS/MP3/20260615-091435_FEDEX_TN__louis_01712174109-all.mp3",
        "❌ Not a Lead (tracking/delivery status)": "https://103.204.81.3/RECORDINGS/MP3/20260615-101255_FEDEX_TN_FEDEX_IMPORT_bijoy_8801923430572-all.mp3",
    }
    chosen_sample = st.selectbox("Choose a sample recording", list(demo_samples.keys()))

    st.markdown("---")
    if st.session_state.history:
        st.header("📊 Session Summary")
        strong = sum(1 for h in st.session_state.history if h["lead_quality"] == "Strong Lead")
        potential = sum(1 for h in st.session_state.history if h["lead_quality"] == "Potential Lead")
        weak = sum(1 for h in st.session_state.history if h["lead_quality"] == "Weak Interest")
        not_lead = sum(1 for h in st.session_state.history if h["lead_quality"] == "Not a Lead")
        st.write(f"🟢 Strong: **{strong}**  |  🔵 Potential: **{potential}**")
        st.write(f"🟡 Weak: **{weak}**  |  🔴 Not a lead: **{not_lead}**")
        if st.button("🗑️ Clear history"):
            st.session_state.history = []
            st.rerun()

# ----------------------------------------------------------------------------
# MAIN — INPUT
# ----------------------------------------------------------------------------
tab_analyze, tab_history = st.tabs(["🔍 Analyze Call", "🗂️ History"])

with tab_analyze:
    st.markdown('<div class="section-title">1. Provide the audio</div>', unsafe_allow_html=True)

    input_mode = st.radio(
        "Input mode",
        options=["Provide public URL (recommended)", "Upload local file"],
        horizontal=True,
        label_visibility="collapsed",
    )

    audio_url = None
    uploaded_file = None

    if input_mode.startswith("Provide public URL"):
        default_url = demo_samples.get(chosen_sample, "")
        audio_url = st.text_input(
            "Audio file public URL",
            value=default_url,
            placeholder="https://example.com/path/to/audio.mp3",
        )
        if audio_url:
            st.audio(audio_url)
    else:
        uploaded_file = st.file_uploader(
            "Upload an audio file (wav, mp3, m4a, ogg)", type=["wav", "mp3", "m4a", "ogg"]
        )
        if uploaded_file:
            st.audio(uploaded_file.read(), start_time=0)
            uploaded_file.seek(0)

    st.markdown('<div class="section-title">2. Run the analysis</div>', unsafe_allow_html=True)
    run = st.button("🚀 Analyze Recording", type="primary", use_container_width=False)

    # ------------------------------------------------------------------
    # API CALLS
    # ------------------------------------------------------------------
    def call_api_file(file_obj, filename, content_type):
        files = {"file": (filename, file_obj, content_type)}
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            resp = requests.post(api_url, files=files, headers=headers, timeout=timeout)
        except Exception as e:
            return {"error": f"Request failed: {e}"}
        try:
            return resp.json()
        except Exception:
            return {"error": f"Non-JSON response: {resp.status_code} - {resp.text[:200]}"}

    def call_api_url(audio_url_value: str):
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {"audio_url": audio_url_value}
        try:
            resp = requests.post(api_url, json=payload, headers=headers, timeout=timeout)
        except Exception as e:
            return {"error": f"Request failed: {e}"}
        try:
            return resp.json()
        except Exception:
            return {"error": f"Non-JSON response: {resp.status_code} - {resp.text[:200]}"}

    # ------------------------------------------------------------------
    # RESULT RENDERING
    # ------------------------------------------------------------------
    BADGE_CLASS = {
        "Strong Lead": ("badge-strong", "🟢"),
        "Potential Lead": ("badge-potential", "🔵"),
        "Weak Interest": ("badge-weak", "🟡"),
        "Not a Lead": ("badge-not", "🔴"),
    }

    def _display_result(result, elapsed):
        if not result:
            st.error("Empty response from API.")
            return
        if isinstance(result, dict) and result.get("error"):
            st.error(result.get("error"))
            return

        try:
            parsed = result if not isinstance(result, str) else json.loads(result)
        except Exception:
            parsed = {"raw": result}

        is_potential_lead = parsed.get("is_potential_lead") or parsed.get("is_lead")
        justification = parsed.get("justification") or parsed.get("reason") or ""
        lead_quality = parsed.get("lead_quality", "Not a Lead")
        keywords_matched = (
            parsed.get("keywords_matched") or parsed.get("keywords") or parsed.get("matched_keywords") or []
        )
        tokens_used = parsed.get("tokens_used", 0)

        badge_class, emoji = BADGE_CLASS.get(lead_quality, ("badge-not", "⚪"))

        st.markdown('<div class="section-title">Result</div>', unsafe_allow_html=True)

        # Top row: badge + metrics (tokens_used is intentionally excluded here — it lives in History)
        c1, c2, c3 = st.columns([1.8, 1, 1])
        with c1:
            st.markdown(
                f'<span class="badge {badge_class}">{emoji} {lead_quality}</span>',
                unsafe_allow_html=True,
            )
            st.write("")
            verdict = "Yes ✅" if is_potential_lead else "No ❌"
            st.markdown(f"**Potential Lead:** {verdict}")
        with c2:
            st.markdown(
                f'<div class="metric-card"><div class="label">Response Time</div>'
                f'<div class="value">{elapsed:.1f}s</div></div>',
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f'<div class="metric-card"><div class="label">Keywords Found</div>'
                f'<div class="value">{len(keywords_matched)}</div></div>',
                unsafe_allow_html=True,
            )

        # Justification
        if justification:
            st.markdown('<div class="section-title">Why the AI decided this</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="justification-box">{justification}</div>', unsafe_allow_html=True)

        # Keywords
        if keywords_matched:
            st.markdown('<div class="section-title">Matched Keywords / Phrases</div>', unsafe_allow_html=True)
            chips_html = "".join(f'<span class="chip">{kw}</span>' for kw in keywords_matched)
            st.markdown(chips_html, unsafe_allow_html=True)

        # Raw JSON
        with st.expander("🔎 View full API response (JSON)"):
            st.json(parsed)

        # Save to history
        st.session_state.history.insert(
            0,
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source": audio_url if audio_url else (uploaded_file.name if uploaded_file else "n/a"),
                "lead_quality": lead_quality,
                "is_potential_lead": bool(is_potential_lead),
                "keywords": keywords_matched,
                "justification": justification,
                "tokens_used": tokens_used,
                "elapsed": elapsed,
            },
        )

    if run:
        if input_mode.startswith("Provide public URL"):
            if not audio_url:
                st.warning("Please provide a public audio URL, or pick a sample from the sidebar.")
            else:
                start = time.time()
                with st.spinner("🎧 Listening to the call and analyzing intent..."):
                    result = call_api_url(audio_url)
                _display_result(result, time.time() - start)
        else:
            if not uploaded_file:
                st.warning("Please upload an audio file first.")
            else:
                start = time.time()
                with st.spinner("📤 Uploading and analyzing..."):
                    with tempfile.NamedTemporaryFile(delete=False) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp.flush()
                        tmp_name = tmp.name
                    with open(tmp_name, "rb") as f:
                        result = call_api_file(f, uploaded_file.name, uploaded_file.type or "audio/wav")
                    try:
                        os.remove(tmp_name)
                    except Exception:
                        pass
                _display_result(result, time.time() - start)

    st.markdown("---")
    with st.expander("ℹ️ Tips for a smooth demo"):
        st.write("- Make sure the FastAPI server is running and reachable at the configured URL.")
        st.write("- Use the **Quick Demo Samples** in the sidebar for a reliable, repeatable walkthrough with clients.")
        st.write("- The app expects JSON fields: `is_potential_lead`, `justification`, `lead_quality`, `keywords_matched`, `tokens_used`.")

# ----------------------------------------------------------------------------
# HISTORY TAB
# ----------------------------------------------------------------------------
with tab_history:
    st.markdown('<div class="section-title">Session History</div>', unsafe_allow_html=True)
    if not st.session_state.history:
        st.info("No calls analyzed yet in this session. Run an analysis in the **Analyze Call** tab.")
    else:
        for item in st.session_state.history:
            badge_class, emoji = BADGE_CLASS.get(item["lead_quality"], ("badge-not", "⚪"))
            st.markdown(
                f"""
                <div class="history-row">
                    <span class="badge {badge_class}">{emoji} {item['lead_quality']}</span>
                    &nbsp;&nbsp;<b>{item['timestamp']}</b>
                    &nbsp;&nbsp;<span style="color:#9AA0B4;">{item['source']}</span>
                    &nbsp;&nbsp;<span style="color:#9AA0B4;">⏱ {item['elapsed']:.1f}s · 🔢 {item['tokens_used']} tokens</span>
                </div>
                """,
                unsafe_allow_html=True,
            )