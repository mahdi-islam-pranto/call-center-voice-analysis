import streamlit as st
import requests
import tempfile
import os
import json


st.set_page_config(page_title="Lead Detection Demo", layout="centered")

st.title("Call Center Lead Detection — Demo")
st.markdown(
    "Upload a call recording and the demo will call your FastAPI endpoint to determine whether the customer is a potential lead, with justification, lead quality, and matched keywords."
)

with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("FastAPI endpoint URL", value="http://localhost:8000/find_lead", placeholder="http://localhost:8000/find_lead")
    api_key = st.text_input("Optional API key (Bearer)", value="", type="password")
    timeout = st.number_input("Request timeout (s)", min_value=10, max_value=300, value=60)

st.subheader("1) Provide audio")

# Let the user choose whether to provide a public URL or upload a local file
input_mode = st.radio("Input mode", options=["Provide public URL (recommended)", "Upload local file"])

audio_url = None
uploaded_file = None

if input_mode.startswith("Provide public URL"):
    audio_url = st.text_input("Audio file public URL", value="", placeholder="https://example.com/path/to/audio.mp3")
    if audio_url:
        st.write("URL to be sent to the API:")
        st.write(audio_url)
else:
    uploaded_file = st.file_uploader("Upload an audio file (wav, mp3, m4a, ogg)", type=["wav", "mp3", "m4a", "ogg"]) 
    if uploaded_file:
        st.audio(uploaded_file.read(), start_time=0)
        # Rewind the buffer for later POST
        uploaded_file.seek(0)

st.markdown("---")
st.subheader("2) Analyze")
cols = st.columns([1, 1, 2])
run = cols[2].button("Analyze Recording")

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
        return {"error": f"Non-JSON response: {resp.status_code} - {resp.text[:100]}"}


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


def _display_result(result):
    if not result:
        st.error("Empty response from API.")
        return
    if isinstance(result, dict) and result.get("error"):
        st.error(result.get("error"))
        return

    st.success("Analysis received")
    try:
        parsed = result if not isinstance(result, str) else json.loads(result)
    except Exception:
        parsed = {"raw": result}

    is_potential_lead = parsed.get("is_potential_lead") or parsed.get("is_lead")
    justification = parsed.get("justification") or parsed.get("reason")
    lead_quality = parsed.get("lead_quality")
    keywords_matched = parsed.get("keywords_matched") or parsed.get("keywords") or parsed.get("matched_keywords")
    transcript = parsed.get("transcript")

    if is_potential_lead is not None:
        if isinstance(is_potential_lead, bool):
            st.write("**Potential Lead:**", "Yes" if is_potential_lead else "No")
        else:
            st.write("**Potential Lead:**", is_potential_lead)

    if lead_quality is not None:
        st.write("**Lead Quality:**", lead_quality)

    if justification:
        st.write("**Justification:**")
        st.info(justification)

    if keywords_matched:
        st.write("**Matched Keywords:**")
        if isinstance(keywords_matched, (list, tuple)):
            st.write(", ".join(map(str, keywords_matched)))
        else:
            st.write(keywords_matched)

    if transcript:
        st.write("**Transcript (partial):**")
        st.write(transcript)

    st.markdown("**Full JSON response:**")
    st.json(parsed)


if run:
    # If user chose URL mode, require a URL
    if input_mode.startswith("Provide public URL"):
        if not audio_url:
            st.warning("Please provide a public audio URL to send to the API.")
        else:
            with st.spinner("Sending URL to API and analyzing..."):
                result = call_api_url(audio_url)
            _display_result(result)
    else:
        # file upload path — note: your FastAPI currently expects a URL; this will POST multipart/form-data
        if not uploaded_file:
            st.warning("Please upload an audio file first.")
        else:
            with st.spinner("Uploading and analyzing (multipart)..."):
                # Save to a temporary file and send it
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp.flush()
                    tmp_name = tmp.name

                # Re-open for binary streaming
                with open(tmp_name, "rb") as f:
                    result = call_api_file(f, uploaded_file.name, uploaded_file.type or "audio/wav")

                try:
                    os.remove(tmp_name)
                except Exception:
                    pass

            _display_result(result)

st.markdown("---")
st.write("Tips:")
st.write("- Ensure your FastAPI server is running and reachable at the configured URL.")
st.write("- The demo expects the endpoint to return JSON fields such as `is_potential_lead`, `justification`, `lead_quality`, and `keywords_matched`. If your API returns different keys, update the Streamlit app to match.")
