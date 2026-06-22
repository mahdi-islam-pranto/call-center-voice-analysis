import streamlit as st
import requests
import json

st.set_page_config(page_title="Agent Performance Demo", layout="centered")

st.title("Agent Performance Evaluator — Demo")
st.markdown(
    "Paste multiple public audio URLs (one per line) and choose performance types to evaluate agent performance."
)

with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("API URL", value="http://localhost:8000/agent-performance", placeholder="http://localhost:8000/agent-performance")
    api_key = st.text_input("Optional API key (Bearer)", value="", type="password")
    timeout = st.number_input("Request timeout (s)", min_value=10, max_value=300, value=60)

st.subheader("1) Audio URLs")
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

st.subheader("2) Performance Types")
default_types = ["Opening Greetings", "Active Listening", "Check Resource", "Hold", "Correct Info", "Complete Info", "Empathy/Tone", "Taking Ownership", "Further Assistance", "Ending Greetings", "Slang Usage"]
selected = st.multiselect("Choose performance types to evaluate", options=default_types, default=["Empathy/Tone", "Active Listening"])
extra_type = st.text_input("Add custom performance type (optional)")

if extra_type:
    if extra_type not in selected:
        selected.append(extra_type)

st.markdown("---")

if st.button("Run Evaluation"):
    if not paths:
        st.warning("Please provide at least one public audio URL.")
    elif not selected:
        st.warning("Please select at least one performance type.")
    else:
        payload = {"paths": paths, "performance_types": selected}
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        with st.spinner("Sending request to API and waiting for results..."):
            try:
                resp = requests.post(api_url, json=payload, headers=headers, timeout=timeout)
            except Exception as e:
                st.error(f"Request failed: {e}")
                resp = None

        if resp is None:
            st.stop()

        try:
            result = resp.json()
        except Exception:
            st.error(f"Non-JSON response: {resp.status_code} - {resp.text[:300]}")
            st.stop()

        st.success("Results received")
        # Display structured agent performance if present
        def _normalize_metrics(metrics_raw):
            # Metrics may be a list, or a dict with numeric keys; normalize to list of dicts
            if metrics_raw is None:
                return []
            if isinstance(metrics_raw, list):
                return metrics_raw
            if isinstance(metrics_raw, dict):
                # convert numeric-keyed dict to list
                try:
                    # sort keys if numeric-like
                    items = []
                    for k in sorted(metrics_raw.keys(), key=lambda x: int(x) if str(x).isdigit() else x):
                        items.append(metrics_raw[k])
                    return items
                except Exception:
                    return list(metrics_raw.values())
            return []

        def display_agent_performance(resp_json):
            ap = resp_json.get("agent_performance") if isinstance(resp_json, dict) else None
            if not ap:
                # fallback: if top-level contains metrics
                ap = resp_json

            metrics = _normalize_metrics(ap.get("metrics") if isinstance(ap, dict) else None)
            total = ap.get("total_score") if isinstance(ap, dict) else None
            max_possible = ap.get("max_possible_score") if isinstance(ap, dict) else None
            summary = ap.get("performance_summary") if isinstance(ap, dict) else None
            strengths = ap.get("strengths") if isinstance(ap, dict) else None
            weaknesses = ap.get("weaknesses") if isinstance(ap, dict) else None
            suggestions = ap.get("improvement_suggestions") if isinstance(ap, dict) else None

            st.markdown("**Agent Performance Overview**")
            if total is not None and max_possible is not None:
                pct = float(total) / float(max_possible) if max_possible else 0
                st.write(f"Score: **{total} / {max_possible}**")
                st.progress(min(max(pct, 0.0), 1.0))
            elif total is not None:
                st.write(f"Score: **{total}**")

            if summary:
                with st.expander("Performance Summary", expanded=True):
                    st.write(summary)

            if metrics:
                st.markdown("**Metrics**")
                # display metrics in two columns grid
                cols = st.columns(2)
                for i, m in enumerate(metrics):
                    col = cols[i % 2]
                    name = m.get("metric_name") or m.get("name") or m.get("metric")
                    score = m.get("score")
                    obs = m.get("observation") or m.get("notes")
                    with col:
                        st.subheader(f"{name}")
                        if score is not None:
                            # assume score is out of 10 if max_possible >=10 or unknown
                            display_max = 10
                            if max_possible:
                                # heuristics: if there are n metrics, per-metric max = max_possible / n
                                try:
                                    n = max(1, len(metrics))
                                    per_max = float(max_possible) / n
                                    if per_max >= 1:
                                        display_max = per_max
                                except Exception:
                                    display_max = 10
                            pct = float(score) / float(display_max) if display_max else 0
                            st.metric(label="Score", value=f"{score}/{int(display_max)}", delta=f"{int(pct*100)}%")
                            st.progress(min(max(pct, 0.0), 1.0))
                        if obs:
                            st.write(obs)

            if strengths:
                st.markdown("**Strengths**")
                for s in strengths:
                    st.write(f"- {s}")

            if weaknesses:
                st.markdown("**Weaknesses**")
                for w in weaknesses:
                    st.write(f"- {w}")

            if suggestions:
                st.markdown("**Improvement Suggestions**")
                for i, s in enumerate(suggestions, 1):
                    st.write(f"{i}. {s}")

            # optional metadata
            if isinstance(resp_json, dict):
                tb = resp_json.get("transcript_bearer") or resp_json.get("transcript_provider")
                sb = resp_json.get("summary_bearer") or resp_json.get("summary_provider")
                if tb or sb:
                    st.markdown("**Sources**")
                    if tb:
                        st.write(f"Transcript provider: {tb}")
                    # if sb:
                    #     st.write(f"Summary provider: {sb}")

        display_agent_performance(result)

        # Always provide full JSON in an expander for advanced users
        with st.expander("Full JSON response", expanded=False):
            st.json(result)

st.markdown("---")
st.write("Run locally:")
st.code("pip install streamlit requests\nstreamlit run Agent-performance/streamlit_app.py", language="bash")
st.write("Adjust `API URL` in the sidebar to point to your running API.")
