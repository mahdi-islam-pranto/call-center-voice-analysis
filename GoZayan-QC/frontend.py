"""
GoZayaan Call QC portal (Streamlit).

1. Upload a batch of recordings, tag each with its agent, run the QC job (progress per file).
2. Team dashboard: KPIs, agent leaderboard, parameter heatmap.
3. Call review: flags, evidence, transcript, claims - and QA overrides with a comment.
4. Agent coaching briefs.
5. Export: Excel (final ratings), JSON (re-open later), QA labels CSV (for evaluate_against_human.py).
"""
import csv
import io
import json
import time
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from excel_report import build_excel_report
from scoring import apply_overrides

st.set_page_config(page_title="GoZayaan Call QC", page_icon="🎧", layout="wide")

RATING_ICON = {"Good": "🟢", "Average": "🟡", "Bad": "🔴", "N/A": "⚪", None: "⚫"}
SEVERITY_BOX = {"high": st.error, "medium": st.warning, "low": st.info}

st.markdown("""
<style>
.qc-title { font-size: 1.8rem; font-weight: 700; margin-bottom: 0; }
.qc-sub { opacity: 0.7; margin-top: 0; }
.qc-quote { border-left: 3px solid #1f77b4; padding: 0.2rem 0.6rem; margin: 0.2rem 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Sidebar / API
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("API URL", value="http://localhost:8000").strip().rstrip("/")
    timeout = st.number_input("Max wait for a batch (s)", min_value=60, max_value=7200, value=1800, step=60)
    double_check = st.toggle(
        "Consistency check", value=False,
        help="Scores every call twice and flags parameters where the two passes disagree. About 1.6x cost.",
    )


@st.cache_data(ttl=300, show_spinner=False)
def load_config(url: str):
    resp = requests.get(f"{url}/qc/config", timeout=15)
    resp.raise_for_status()
    return resp.json()


try:
    CFG = load_config(api_url)
except Exception as e:
    CFG = None
    st.sidebar.error(f"Cannot reach the API at {api_url}: {e}")

brand = CFG["brand_name"] if CFG else "GoZayaan"
st.markdown(f'<p class="qc-title">🎧 {brand} Call QC</p>', unsafe_allow_html=True)
st.markdown('<p class="qc-sub">10-parameter QA scorecard · Good 1 / Average 0.5 / Bad 0 · evidence-checked AI evaluation with QA review</p>', unsafe_allow_html=True)

if CFG:
    with st.sidebar.expander("Scorecard reference"):
        for p in CFG["parameters"]:
            st.markdown(f"**{p['name']}**  \n{p['question']}")
        st.caption(f"N/A policy: {CFG['na_policy']} · Pass mark: {CFG['pass_mark']}/10 · Critical errors: {CFG['critical_error_policy']}")
        st.caption(f"Models: {', '.join(f'{k}: {v}' for k, v in CFG['models'].items())}")

st.session_state.setdefault("reviews", {})

# ─────────────────────────────────────────────
# New batch
# ─────────────────────────────────────────────
with st.expander("➕ New QC batch", expanded="report" not in st.session_state):
    tab_upload, tab_urls, tab_open = st.tabs(["Upload recordings", "Recording URLs", "Open saved report"])

    with tab_upload:
        files = st.file_uploader("Call recordings", type=["mp3", "wav", "m4a", "ogg"], accept_multiple_files=True) or []
        agent_names = []
        if files:
            st.caption("Assign each recording to its agent (leave blank to detect the name from the call).")
            editor_key = "agents_" + str(abs(hash(tuple(f.name for f in files))))
            table = st.data_editor(
                pd.DataFrame({"File": [f.name for f in files], "Size (MB)": [round(f.size / 1048576, 2) for f in files], "Agent": [""] * len(files)}),
                disabled=["File", "Size (MB)"], hide_index=True, width="stretch", key=editor_key,
            )
            agent_names = [str(a or "").strip() for a in table["Agent"].tolist()]

    with tab_urls:
        urls_text = st.text_area("Public recording URLs (one per line)", height=120)
        urls = [u.strip() for u in urls_text.splitlines() if u.strip()]

    with tab_open:
        saved = st.file_uploader("Report JSON downloaded earlier from this portal", type=["json"], key="saved_json")
        if saved and st.button("Open report"):
            data = json.loads(saved.getvalue())
            st.session_state["report"] = data
            # Restore QA decisions stored in the JSON so a review can continue where it stopped.
            st.session_state["reviews"] = data.get("qa_reviews", {})
            st.session_state["audio"] = {}
            st.rerun()

    total = len(files) + len(urls)
    if st.button(f"Run QC on {total} recording(s)", type="primary", disabled=not total or not CFG):
        try:
            resp = requests.post(
                f"{api_url}/qc/jobs",
                files=[("files", (f.name, f.getvalue(), f.type or "application/octet-stream")) for f in files] or None,
                data={"agent_names": agent_names, "urls": urls, "double_check": str(double_check).lower()},
                timeout=300,
            )
            resp.raise_for_status()
            job_id = resp.json()["job_id"]
        except Exception as e:
            detail = getattr(e, "response", None)
            st.error(f"Could not start the job: {detail.text if detail is not None else e}")
            st.stop()

        bar = st.progress(0.0, text="Starting...")
        started = time.time()
        job = {}
        while time.time() - started < timeout:
            job = requests.get(f"{api_url}/qc/jobs/{job_id}", timeout=30).json()
            if job.get("status") != "running":
                break
            done = job.get("done", 0)
            bar.progress(done / max(job.get("total", 1), 1), text=f"Evaluated {done}/{job.get('total')} · last: {job.get('last_file') or '-'}")
            time.sleep(2)
        if job.get("status") == "completed":
            bar.progress(1.0, text="Done")
            st.session_state["report"] = job["result"]
            st.session_state["reviews"] = {}
            st.session_state["audio"] = {f.name: f for f in files}
            st.rerun()
        elif job.get("status") == "failed":
            st.error(f"QC job failed: {job.get('error')}")
        else:
            st.error("Timed out waiting for the job; it may still be running on the server.")

if "report" not in st.session_state:
    st.info("Upload recordings and run a QC batch to see results.")
    st.stop()

raw_report = st.session_state["report"]
reviews = st.session_state["reviews"]
report = apply_overrides(raw_report, reviews)
cfg = report.get("config") or CFG or {}
params = cfg.get("parameters", [])
calls = report["calls"]

if report.get("failed"):
    st.warning("Could not evaluate:\n" + "\n".join(f"- {k}: {v}" for k, v in report["failed"].items()))


def _mmss(seconds):
    if seconds is None:
        return "-"
    seconds = int(round(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


tab_team, tab_call, tab_agent, tab_export = st.tabs(["📊 Team dashboard", "🎧 Call review", "🧑‍🏫 Agent coaching", "⬇️ Export"])

# ─────────────────────────────────────────────
# Team dashboard
# ─────────────────────────────────────────────
with tab_team:
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Calls", len(calls))
    k2.metric("Team average", f"{report['team_average_score']}/10")
    k3.metric("Pass rate", f"{report['team_pass_rate']}%", help=f"Score ≥ {cfg.get('pass_mark')} and no critical error")
    k4.metric("Need review", sum(c["needs_review"] and not c.get("reviewed") for c in calls), help="Flagged and not yet reviewed")
    k5.metric("Critical errors", sum(len(c["critical_errors"]) for c in calls if not c.get("critical_dismissed")))

    agents = report["agents"]
    st.subheader("Agent leaderboard")
    st.dataframe(pd.DataFrame([{
        "Agent": a["agent_name"], "Calls": a["calls"], "Avg score /10": a["average_score"], "Pass rate %": a["pass_rate"],
        "Need review": a["calls_needing_review"], "Critical errors": a["critical_errors"],
    } for a in agents]), hide_index=True, width="stretch")

    if agents and params:
        st.subheader("Parameter heatmap")
        st.caption("Average points per parameter (1.0 = always Good, 0 = always Bad; N/A calls excluded).")
        z = [[pa["average_points"] for pa in a["parameter_averages"]] for a in agents]
        fig = go.Figure(go.Heatmap(
            z=z, x=[p["name"] for p in params], y=[a["agent_name"] for a in agents],
            zmin=0, zmax=1, colorscale="RdYlGn", text=[[("" if v is None else f"{v:.2f}") for v in row] for row in z],
            texttemplate="%{text}", hovertemplate="%{y} · %{x}: %{z}<extra></extra>",
        ))
        fig.update_layout(height=120 + 40 * len(agents), margin=dict(l=10, r=10, t=10, b=10), xaxis_tickangle=-30)
        st.plotly_chart(fig, width="stretch")

    st.subheader("All calls")
    st.dataframe(pd.DataFrame([{
        "Call": c["call_id"], "File": c["file_name"], "Agent": c["agent_name"],
        "Category": c["forensics"]["call_category"].replace("_", " "), "Duration": _mmss(c.get("duration_seconds")),
        "Score": c["score"], "Band": c["band"], "Passed": "✅" if c["passed"] else "❌",
        "Review": "✔ reviewed" if c.get("reviewed") else ("⚠ needed" if c["needs_review"] else ""),
        "Critical": ", ".join(e["error_key"] for e in c["critical_errors"]),
        **{p["name"]: RATING_ICON.get(cp.get("final_rating")) for p, cp in zip(params, c["parameters"])},
    } for c in calls]), hide_index=True, width="stretch")

# ─────────────────────────────────────────────
# Call review
# ─────────────────────────────────────────────
with tab_call:
    only_flagged = st.checkbox("Show only calls that need review", value=False)
    shown = [c for c in calls if c["needs_review"] or not only_flagged]
    if not shown:
        st.success("No calls need review.")
    else:
        labels = {
            c["call_id"]: f"{c['call_id']} · {c['file_name']} · {c['agent_name']} · {c['score']}/10"
            + (" · ⚠" if c["needs_review"] and not c.get("reviewed") else "") + (" · ✔" if c.get("reviewed") else "")
            for c in shown
        }
        call_id = st.selectbox("Call", list(labels), format_func=labels.get)
        call = next(c for c in calls if c["call_id"] == call_id)
        f = call["forensics"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Score", f"{call['score']}/10", delta=None if call["score"] == call["ai_score"] else f"AI: {call['ai_score']}")
        m2.metric("Band", call["band"])
        m3.metric("Passed", "Yes" if call["passed"] else "No")
        m4.metric("Duration", _mmss(call.get("duration_seconds")))
        st.markdown(
            f"**Agent:** {call['agent_name']} ({call['agent_name_source']}) · **Direction:** {f['call_direction']} · "
            f"**Category:** {f['call_category'].replace('_', ' ')} · **Language:** {f['primary_language']}  \n"
            f"**Customer sentiment:** {f['customer_sentiment_start']} → {f['customer_sentiment_end']} · "
            f"**Outcome:** {f['resolution_status'].replace('_', ' ')} — {f['resolution_note']}"
        )
        audio = (st.session_state.get("audio") or {}).get(call["file_name"])
        if audio is not None:
            st.audio(audio)

        for flag in call["review_flags"]:
            SEVERITY_BOX[flag["severity"]](flag["reason"], icon="🚩" if flag["severity"] == "high" else None)

        st.markdown(f"**Customer need:** {f['customer_intent']}")
        st.markdown(f"**Summary:** {call['call_summary']}")
        c1, c2 = st.columns(2)
        c1.markdown("**What went well**\n" + "\n".join(f"- {s}" for s in call["what_went_well"]))
        c2.markdown("**Improvement areas**\n" + "\n".join(f"- {s}" for s in call["improvement_areas"]))

        sub_params, sub_transcript, sub_claims, sub_voice = st.tabs(["Parameters", "Transcript & events", "Agent claims", "Voice delivery"])

        with sub_params:
            for p in call["parameters"]:
                final, ai_rating = p["final_rating"], p["ai_rating"]
                title = f"{RATING_ICON.get(final)} {p['name']} — {final or 'not rated'}"
                if final != ai_rating:
                    title += f" (AI: {ai_rating})"
                title += f" · confidence {p['confidence']}"
                if p.get("second_opinion") and p["second_opinion"] != ai_rating:
                    title += f" · 2nd pass: {p['second_opinion']}"
                with st.expander(title, expanded=final in ("Bad", None) or p["confidence"] == "low"):
                    st.write(p["justification"])
                    for e in p["evidence"]:
                        mark = "✓" if e["verified"] else "✗ not found in transcript"
                        st.markdown(f'<div class="qc-quote"><b>[{e["t"]}] {e["speaker"]}:</b> {e["quote"]} <small>({mark})</small></div>', unsafe_allow_html=True)
                    st.dataframe(pd.DataFrame([{"Item": i["item_id"], "Status": i["status"], "Note": i["note"]} for i in p["checklist"]]),
                                 hide_index=True, width="stretch")
                    if p.get("coaching_tip"):
                        st.info(f"Coaching tip: {p['coaching_tip']}")

        with sub_transcript:
            events_by_time = {}
            for e in f["events"]:
                events_by_time.setdefault(e["t"], []).append(f"{e['type'].replace('_', ' ')}" + (f" ({e['duration_seconds']}s)" if e.get("duration_seconds") else ""))
            st.dataframe(pd.DataFrame([{
                "Time": t["t"], "Speaker": t["speaker"], "Text": t["text"], "Event": "; ".join(events_by_time.get(t["t"], [])),
            } for t in f["transcript"]]), hide_index=True, width="stretch", height=420)
            if f["events"]:
                st.markdown("**Events**")
                st.dataframe(pd.DataFrame([{"Time": e["t"], "Event": e["type"].replace("_", " "), "Seconds": e.get("duration_seconds"), "Detail": e["detail"]}
                                           for e in f["events"]]), hide_index=True, width="stretch")

        with sub_claims:
            checks = {cc["claim_index"]: cc for cc in call["claim_checks"]}
            if not f["agent_claims"]:
                st.caption("The agent made no factual claims about prices, rules or timelines.")
            else:
                st.dataframe(pd.DataFrame([{
                    "Time": cl["t"], "Topic": cl["topic"].replace("_", " "), "Claim": cl["claim"],
                    "Check": checks.get(i, {}).get("verdict", "not checked").replace("_", " "), "Note": checks.get(i, {}).get("note", ""),
                } for i, cl in enumerate(f["agent_claims"])]), hide_index=True, width="stretch")
                st.caption("Claims can only be checked against what is written in the knowledge base (knowledge/gozayaan_knowledge.md).")

        with sub_voice:
            d = f["delivery"]
            st.markdown(
                f"- **Tone:** opening *{d['tone_opening']}*, middle *{d['tone_middle']}*, closing *{d['tone_closing']}*\n"
                f"- **Pace:** {d['pace']}\n- **Clarity:** {d['clarity']}\n- **Accent & register:** {d['accent_and_register']}\n"
                f"- **Filler words:** {d['filler_words']}" + (f" ({', '.join(d['filler_examples'])})" if d.get("filler_examples") else "")
                + f"\n- **Audio quality:** {f['audio_quality']}" + (f" — {f['audio_quality_note']}" if f.get("audio_quality_note") else "")
            )

        st.divider()
        st.subheader("QA review")
        existing = reviews.get(call_id, {})
        na_allowed = {p["key"]: p["na_allowed"] for p in params}
        with st.form(f"review_{call_id}"):
            edited = st.data_editor(
                pd.DataFrame([{"Parameter": p["name"], "AI rating": p["ai_rating"], "Final rating": p["final_rating"]} for p in call["parameters"]]),
                column_config={"Final rating": st.column_config.SelectboxColumn(options=["Good", "Average", "Bad", "N/A"], required=True)},
                disabled=["Parameter", "AI rating"], hide_index=True, width="stretch",
            )
            dismiss = False
            if call["critical_errors"]:
                dismiss = st.checkbox("Dismiss the critical error(s) as false positives", value=existing.get("dismiss_critical", False))
            comment = st.text_area("Reviewer comment", value=existing.get("comment", ""))
            if st.form_submit_button("Save review", type="primary"):
                changed, rejected = {}, []
                for p, final in zip(call["parameters"], edited["Final rating"].tolist()):
                    if final == "N/A" and not na_allowed.get(p["key"]):
                        rejected.append(p["name"])
                    elif final != p["ai_rating"]:
                        changed[p["key"]] = final
                if rejected:
                    st.error(f"N/A is not allowed for: {', '.join(rejected)}. Nothing saved.")
                else:
                    reviews[call_id] = {"ratings": changed, "dismiss_critical": dismiss, "comment": comment,
                                        "reviewed_at": datetime.now().isoformat(timespec="seconds")}
                    st.rerun()

# ─────────────────────────────────────────────
# Agent coaching
# ─────────────────────────────────────────────
with tab_agent:
    for a in report["agents"]:
        with st.expander(f"{a['agent_name']} — {a['average_score']}/10 over {a['calls']} call(s) · pass rate {a['pass_rate']}%", expanded=len(report["agents"]) == 1):
            pa = [x for x in a["parameter_averages"] if x["average_points"] is not None]
            fig = go.Figure(go.Bar(
                x=[x["average_points"] for x in pa], y=[x["name"] for x in pa], orientation="h",
                marker_color=["#2E9E5B" if x["average_points"] >= 0.85 else "#E3A21A" if x["average_points"] >= 0.6 else "#D64545" for x in pa],
                hovertemplate="%{y}: %{x}<extra></extra>",
            ))
            fig.update_layout(height=60 + 28 * len(pa), margin=dict(l=10, r=10, t=10, b=10), xaxis_range=[0, 1])
            st.plotly_chart(fig, width="stretch", key=f"bar_{a['agent_name']}")
            coaching = a.get("coaching")
            if coaching:
                st.write(coaching["performance_summary"])
                c1, c2 = st.columns(2)
                c1.markdown("**Consistent strengths**\n" + "\n".join(f"- {s}" for s in coaching["consistent_strengths"]))
                c2.markdown("**Recurring issues**\n" + "\n".join(f"- {s}" for s in coaching["recurring_issues"]))
                st.markdown("**Coaching plan**\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(coaching["coaching_plan"], 1)))
            else:
                agent_calls = [c for c in calls if c["agent_name"] == a["agent_name"]]
                st.caption("Single call (or unnamed agent) - showing the call's own improvement areas.")
                st.markdown("\n".join(f"- {s}" for c in agent_calls for s in c["improvement_areas"]))

# ─────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────
with tab_export:
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    st.download_button(
        "Download Excel report", data=build_excel_report(report),
        file_name=f"gozayaan_qc_{stamp}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary",
    )
    st.caption("Uses the final ratings (QA overrides applied); the Parameter Detail sheet shows AI vs final.")

    saved_report = dict(raw_report, qa_reviews=reviews)
    st.download_button("Download JSON (re-open later in 'Open saved report')", data=json.dumps(saved_report, ensure_ascii=False, indent=1),
                       file_name=f"gozayaan_qc_{stamp}.json", mime="application/json")

    reviewed = [c for c in calls if c.get("reviewed")]
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["file_name", "agent", *[p["name"] for p in params]])
    for c in reviewed:
        writer.writerow([c["file_name"], c["agent_name"], *[p["final_rating"] for p in c["parameters"]]])
    st.download_button(f"Download QA labels CSV ({len(reviewed)} reviewed calls)", data=buffer.getvalue().encode("utf-8-sig"),
                       file_name=f"gozayaan_qa_labels_{stamp}.csv", mime="text/csv", disabled=not reviewed)
    st.caption("Final human-confirmed ratings of the calls you reviewed. Use it with "
               "`python evaluate_against_human.py --report <json> --human <csv>` to measure AI vs human agreement.")

    if report.get("usage"):
        u = report["usage"]
        st.caption(f"Tokens used: input {u.get('input_tokens', 0):,} · output {u.get('output_tokens', 0):,} · total {u.get('total_tokens', 0):,}")
