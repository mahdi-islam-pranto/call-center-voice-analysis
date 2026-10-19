from datetime import datetime

import pandas as pd
import requests
import streamlit as st

from excel_report import build_excel_report

SCORE_MAX = 10

st.set_page_config(page_title="Agent Performance Demo", layout="centered")

st.title("Agent Performance Evaluator — Demo")
st.markdown(
    "Upload call recordings (.mp3 / .wav) or paste public audio URLs, choose performance types, "
    "and download the evaluation as an Excel report."
)

with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("API URL", value="http://138.252.115.100:6027", placeholder="http://localhost:8000", help="Server address; `/agent-performance` at the end is optional.")
    api_key = st.text_input("Optional API key (Bearer)", value="", type="password")
    timeout = st.number_input("Request timeout (s)", min_value=10, max_value=1800, value=300)

st.subheader("1) Call Recordings")
tab_upload, tab_urls = st.tabs(["Upload audio files", "Recording URLs"])

with tab_upload:
    audio_files = st.file_uploader(
        "Upload call recordings (.mp3, .wav)",
        type=["mp3", "wav"],
        accept_multiple_files=True,
    ) or []
    if audio_files:
        with st.expander(f"{len(audio_files)} file(s) selected — preview", expanded=False):
            for f in audio_files:
                st.write(f"**{f.name}** — {f.size / (1024 * 1024):.2f} MB")
                st.audio(f)

with tab_urls:
    urls_text = st.text_area("Audio URLs (one per line)", placeholder="https://.../file1.mp3\nhttps://.../file2.mp3", height=150)

    uploaded = st.file_uploader("Or upload a .txt file with one URL per line", type=["txt"])
    if uploaded and not urls_text:
        try:
            urls_text = uploaded.getvalue().decode("utf-8")
        except Exception:
            urls_text = uploaded.getvalue().decode("latin-1")

    paths = [u.strip() for u in urls_text.splitlines() if u.strip()]
    if paths:
        st.write(f"Detected {len(paths)} URL(s)")
        for i, p in enumerate(paths, 1):
            st.write(f"{i}. {p}")

total_recordings = len(audio_files) + len(paths)
if total_recordings:
    st.info(f"Total recordings to evaluate: **{total_recordings}** ({len(audio_files)} uploaded, {len(paths)} URL)")

st.subheader("2) Performance Types")
default_types = ["Opening Greetings", "Active Listening", "Check Resource", "Hold", "Correct Info", "Complete Info", "Empathy/Tone", "Taking Ownership", "Further Assistance", "Ending Greetings", "Slang Usage"]
selected = st.multiselect("Choose performance types to evaluate", options=default_types, default=["Empathy/Tone", "Active Listening"])
extra_type = st.text_input("Add custom performance type (optional)")

if extra_type:
    if extra_type not in selected:
        selected.append(extra_type)

st.markdown("---")


def _endpoint_url() -> str:
    # Accept either the server root (http://host:port) or the full /agent-performance URL in the sidebar.
    base = api_url.strip().rstrip("/")
    if base.endswith("/agent-performance"):
        base = base[: -len("/agent-performance")]
    return f"{base}/agent-performance/upload" if audio_files else f"{base}/agent-performance"


def _call_api(url: str) -> requests.Response:
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    if audio_files:
        files = [("files", (f.name, f.getvalue(), f.type or "application/octet-stream")) for f in audio_files]
        data = {"performance_types": selected, "paths": paths}
        return requests.post(url, files=files, data=data, headers=headers, timeout=timeout)

    payload = {"paths": paths, "performance_types": selected}
    return requests.post(url, json=payload, headers=headers, timeout=timeout)


if st.button("Run Evaluation", type="primary"):
    if not total_recordings:
        st.warning("Please upload at least one recording or provide a public audio URL.")
    elif not selected:
        st.warning("Please select at least one performance type.")
    else:
        resp = None
        endpoint = _endpoint_url()
        with st.spinner(f"Evaluating {total_recordings} recording(s)... this can take a few minutes."):
            try:
                resp = _call_api(endpoint)
            except Exception as e:
                st.error(f"Request failed: {e}")

        if resp is None:
            st.stop()

        try:
            result = resp.json()
        except Exception:
            st.error(f"Non-JSON response from `{endpoint}`: {resp.status_code} - {resp.text[:300]}")
            st.stop()

        if not resp.ok:
            detail = result.get("detail") if isinstance(result, dict) else result
            st.error(f"API error ({resp.status_code}) from `{endpoint}`: {detail}")
            if resp.status_code == 404:
                st.info(
                    "This API server does not have that endpoint. If you uploaded files, the server must run the "
                    "latest `main.py` (with `/agent-performance/upload`) — redeploy it, or point the API URL in "
                    "the sidebar to a server that has it (e.g. `http://localhost:8000`)."
                )
            st.stop()

        # Keep results across reruns (the download button triggers a rerun).
        ran_at = datetime.now()
        st.session_state["qc_result"] = result
        st.session_state["qc_ran_at"] = ran_at
        st.session_state["qc_excel"] = build_excel_report(result, generated_at=ran_at)


def _normalize_metrics(metrics_raw):
    # Metrics may be a list, or a dict with numeric keys; normalize to list of dicts
    if metrics_raw is None:
        return []
    if isinstance(metrics_raw, list):
        return metrics_raw
    if isinstance(metrics_raw, dict):
        try:
            return [metrics_raw[k] for k in sorted(metrics_raw.keys(), key=lambda x: int(x) if str(x).isdigit() else x)]
        except Exception:
            return list(metrics_raw.values())
    return []


def _fmt(score) -> str:
    return f"{score:g}" if isinstance(score, (int, float)) else str(score)


def display_agent_performance(resp_json):
    ap = resp_json.get("agent_performance") if isinstance(resp_json, dict) else None
    if not ap:
        # fallback: if top-level contains metrics
        ap = resp_json

    metrics = _normalize_metrics(ap.get("metrics"))
    calls = ap.get("call_evaluations") or []
    total = ap.get("total_score")
    max_possible = ap.get("max_possible_score")
    failed = resp_json.get("failed_urls") or []
    reasons = ap.get("failed_reasons") or {}

    if failed:
        st.warning("Could not evaluate:\n" + "\n".join(f"- {f}" + (f" — {reasons[f]}" if f in reasons else "") for f in failed))

    st.markdown("**Agent Performance Overview**")
    if ap.get("agent_name"):
        st.write(f"Agent: **{ap['agent_name']}**")
    if len(calls) > 1:
        st.caption(f"Scores below are averages across {len(calls)} recordings.")
    if total is not None and max_possible:
        st.write(f"Score: **{_fmt(total)} / {max_possible}** ({ap.get('percentage', 0)}%)")
        st.progress(min(max(float(total) / float(max_possible), 0.0), 1.0))

    if ap.get("performance_summary"):
        with st.expander("Performance Summary", expanded=True):
            st.write(ap["performance_summary"])

    if metrics:
        st.markdown("**Metrics**")
        cols = st.columns(2)
        for i, m in enumerate(metrics):
            name = m.get("metric_name") or m.get("name") or m.get("metric")
            score = m.get("score")
            obs = m.get("observation") or m.get("notes")
            with cols[i % 2]:
                st.subheader(f"{name}")
                if score is not None:
                    pct = float(score) / SCORE_MAX
                    st.metric(label="Score", value=f"{_fmt(score)}/{SCORE_MAX}", delta=f"{int(pct * 100)}%")
                    st.progress(min(max(pct, 0.0), 1.0))
                if obs:
                    st.write(obs)

    for title, items, numbered in [
        ("Strengths", ap.get("strengths"), False),
        ("Weaknesses", ap.get("weaknesses"), False),
        ("Improvement Suggestions", ap.get("improvement_suggestions"), True),
    ]:
        if items:
            st.markdown(f"**{title}**")
            for i, item in enumerate(items, 1):
                st.write(f"{i}. {item}" if numbered else f"- {item}")

    if len(calls) > 1:
        st.markdown("---")
        st.markdown("**Per-Recording Results**")
        overview = pd.DataFrame(
            [
                {"File": c["file_name"], **{m["metric_name"]: m["score"] for m in c["metrics"]}, "Percentage": c["percentage"]}
                for c in calls
            ]
        )
        st.dataframe(overview, hide_index=True, use_container_width=True)
        for c in calls:
            with st.expander(f"{c['file_name']} — {c['total_score']}/{c['max_possible_score']} ({c['percentage']}%)"):
                st.write(c.get("call_summary", ""))
                st.dataframe(
                    pd.DataFrame(
                        [{"Performance Type": m["metric_name"], "Score": m["score"], "Justification": m["observation"]} for m in c["metrics"]]
                    ),
                    hide_index=True,
                    use_container_width=True,
                )

    if resp_json.get("audio_analysis_bearer"):
        st.caption(f"Analysis: {resp_json['audio_analysis_bearer']}")


if "qc_result" in st.session_state:
    result = st.session_state["qc_result"]
    st.success("Results received")

    st.download_button(
        "Download Excel report",
        data=st.session_state["qc_excel"],
        file_name=f"agent_qc_report_{st.session_state['qc_ran_at']:%Y%m%d_%H%M%S}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )

    display_agent_performance(result)

    # Always provide full JSON in an expander for advanced users
    with st.expander("Full JSON response", expanded=False):
        st.json(result)

st.markdown("---")
st.write("Run locally:")
st.code("pip install -r requirements.txt\nuvicorn main:app --port 8000\nstreamlit run frontend.py", language="bash")
st.write("Adjust `API URL` in the sidebar to point to your running API.")
