"""
GoZayaan QC evaluation pipeline.

Per call:
  Pass 1  (audio)              -> CallForensics: transcript, events, agent claims, voice delivery
  Pass 2  (audio + pass-1 notes + knowledge base) -> ScoringLLMOutput: checklist-based ratings with quotes
  [Pass 2b, optional]          -> second independent scoring pass; disagreements go to review
  Code                          -> quote verification, consistency rules, score, band, review flags

Per batch:
  Calls grouped by agent -> averages in code + one cheap text-only coaching brief per agent.
"""
import asyncio
import base64
import json
import logging
import os
import re
from collections import defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Awaitable, Callable, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

import config
from audio import AudioSource
from prompts import build_forensics_prompt, build_rollup_prompt, build_scoring_prompt
from schemas import (
    AgentCoachingLLMOutput,
    AgentSummary,
    BatchReport,
    CallForensics,
    CallResult,
    ChecklistResult,
    ParameterAverage,
    ParameterResult,
    ReviewFlag,
    ScoringLLMOutput,
    VerifiedEvidence,
)
from scoring import compute_score, rating_breakdown

logger = logging.getLogger("gozayaan_qc")

ProgressCallback = Callable[[int, int, str], Awaitable[None]]

# Gemini 3 models are tuned for the default temperature (1.0); Google warns lower values can cause
# looping, which shows up as repeated lines in long transcripts. Leave unset unless you have tested.
TEMPERATURE = os.environ.get("QC_TEMPERATURE", "").strip()


class EvaluationFailedError(Exception):
    """Raised when not a single recording could be evaluated (quota, billing, outage...)."""


# ---------------------------------------------------------------------------
# LLM plumbing
# ---------------------------------------------------------------------------


def _build_llm(model: str, thinking_level: str = "") -> ChatGoogleGenerativeAI:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set.")
    kwargs: dict[str, Any] = {"model": model, "google_api_key": api_key}
    if thinking_level:
        kwargs["thinking_level"] = thinking_level
    if TEMPERATURE:
        kwargs["temperature"] = float(TEMPERATURE)
    return ChatGoogleGenerativeAI(**kwargs)


def _usage(raw_message) -> dict[str, int]:
    usage = getattr(raw_message, "usage_metadata", None) or {}
    return {k: v for k, v in usage.items() if isinstance(v, int)}


def _add_usage(total: dict[str, int], usage: dict[str, int]) -> dict[str, int]:
    for key, value in usage.items():
        total[key] = total.get(key, 0) + value
    return total


def short_error(e: Exception) -> str:
    """Pulls the human-readable part out of verbose Gemini errors."""
    text = str(e)
    message = re.search(r"'message': '(.*?)'", text)
    if message:
        status = re.search(r"\(([A-Z_]+)\)", text)
        return f"{status.group(1) + ': ' if status else ''}{message.group(1).strip()}"
    return text[:300]


async def _invoke_structured(llm: ChatGoogleGenerativeAI, schema, messages: list, label: str):
    """
    Structured call with a retry when the reply is not schema-conformant.
    API-level errors (429 / 503) are already retried inside ChatGoogleGenerativeAI.
    """
    structured = llm.with_structured_output(schema, include_raw=True)
    usage: dict[str, int] = {}
    error = None
    for _ in range(config.LLM_RETRIES + 1):
        result = await structured.ainvoke(messages)
        _add_usage(usage, _usage(result["raw"]))
        if result["parsed"] is not None:
            return result["parsed"], usage
        error = result.get("parsing_error")
        logger.warning("[%s] reply was not schema-conformant, retrying: %r", label, error)
    raise ValueError(f"{label}: Gemini did not return a schema-conformant response ({error!r}).")


def _audio_part(audio: AudioSource) -> dict:
    return {"type": "media", "mime_type": audio.mime_type, "data": base64.standard_b64encode(audio.data).decode("utf-8")}


def _fmt_seconds(seconds: Optional[float]) -> str:
    if seconds is None:
        return "unknown"
    seconds = int(round(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _parse_ts(ts: str) -> Optional[int]:
    parts = re.findall(r"\d+", ts or "")
    if not parts or len(parts) > 3:
        return None
    seconds = 0
    for part in parts:
        seconds = seconds * 60 + int(part)
    return seconds


# ---------------------------------------------------------------------------
# Pass 1 / Pass 2
# ---------------------------------------------------------------------------


async def run_forensics(audio: AudioSource) -> tuple[CallForensics, dict]:
    llm = _build_llm(config.FORENSICS_MODEL, config.FORENSICS_THINKING)
    messages = [
        SystemMessage(content=build_forensics_prompt()),
        HumanMessage(content=[
            _audio_part(audio),
            {"type": "text", "text": (
                f"Recording file: {audio.file_name}. Measured duration: {_fmt_seconds(audio.duration_seconds)}. "
                "Produce the complete call notes for this recording."
            )},
        ]),
    ]
    return await _invoke_structured(llm, CallForensics, messages, f"{audio.file_name} / forensics")


async def run_scoring(audio: AudioSource, forensics: CallForensics) -> tuple[ScoringLLMOutput, dict]:
    llm = _build_llm(config.SCORING_MODEL, config.SCORING_THINKING)
    messages = [
        SystemMessage(content=build_scoring_prompt()),
        HumanMessage(content=[
            _audio_part(audio),
            {"type": "text", "text": (
                f"Recording file: {audio.file_name}. Measured duration: {_fmt_seconds(audio.duration_seconds)}.\n\n"
                "CALL NOTES (from the listening analyst):\n"
                f"{forensics.model_dump_json()}\n\n"
                "Evaluate this call following your instructions."
            )},
        ]),
    ]
    return await _invoke_structured(llm, ScoringLLMOutput, messages, f"{audio.file_name} / scoring")


# ---------------------------------------------------------------------------
# Evidence verification
# ---------------------------------------------------------------------------

_SEPARATORS = re.compile(r"[\s.,!?।৷;:\"'“”‘’\-–—()\[\]…]+")


def _norm(text: str) -> str:
    return _SEPARATORS.sub(" ", (text or "").lower()).strip()


def _quote_found(quote: str, transcript_texts: list[str]) -> bool:
    """Exact (normalised) substring match, else >= 80% of the quote matched in blocks of 4+ chars within one turn."""
    q = _norm(quote)
    if len(q) < 3:
        return False
    if q in " ".join(transcript_texts):
        return True
    best = 0
    for text in transcript_texts:
        blocks = SequenceMatcher(None, q, text, autojunk=False).get_matching_blocks()
        best = max(best, sum(b.size for b in blocks if b.size >= 4))
    return best / len(q) >= 0.8


def _verify_evidence(evaluation, transcript_texts: list[str], duration: Optional[float]) -> list[VerifiedEvidence]:
    verified = []
    for ev in evaluation.evidence:
        ts = _parse_ts(ev.t)
        in_range = ts is None or duration is None or ts <= duration + 5
        verified.append(VerifiedEvidence(**ev.model_dump(), verified=in_range and _quote_found(ev.quote, transcript_texts)))
    return verified


# ---------------------------------------------------------------------------
# Post-processing: alignment, consistency rules, score
# ---------------------------------------------------------------------------


def _align_checklist(param: config.QAParameter, returned: list[ChecklistResult], flags: list[ReviewFlag]) -> list[ChecklistResult]:
    by_id = {c.item_id.strip().upper(): c for c in returned}
    aligned = []
    for item_id, _ in param.checklist:
        item = by_id.get(item_id)
        if item is None:
            flags.append(ReviewFlag(severity="low", reason=f"{param.name}: checklist item {item_id} was not assessed by the AI."))
            item = ChecklistResult(item_id=item_id, status="N/A", note="Not assessed by the AI.")
        aligned.append(item)
    return aligned


def _consistency_flags(
    forensics: CallForensics, scoring: ScoringLLMOutput, ratings: dict[str, Optional[str]]
) -> list[ReviewFlag]:
    """Cross-checks ratings against what pass 1 observed. These are the disagreements a human QA would catch."""
    flags: list[ReviewFlag] = []
    events = defaultdict(list)
    for e in forensics.events:
        events[e.type].append(e.t)

    def at(kind: str) -> str:
        return ", ".join(events[kind][:3])

    if ratings.get("no_assistance") == "Good" and (events["colleague_consultation"] or events["transfer"]):
        when = at("colleague_consultation") or at("transfer")
        flags.append(ReviewFlag(severity="medium", reason=f"No Assistance rated Good, but a colleague consultation / transfer was heard at {when}."))
    if ratings.get("first_contact_resolution") == "Good" and forensics.resolution_status in ("unresolved", "partially_resolved"):
        flags.append(ReviewFlag(severity="medium", reason=f"First Contact Resolution rated Good, but the call outcome was noted as {forensics.resolution_status}."))
    if ratings.get("first_contact_resolution") == "Bad" and forensics.resolution_status == "resolved":
        flags.append(ReviewFlag(severity="low", reason="First Contact Resolution rated Bad, but the call outcome was noted as resolved."))
    if ratings.get("escalation") == "N/A" and (
        events["customer_asks_for_supervisor"] or events["escalation_by_agent"] or forensics.resolution_status == "escalated"
    ):
        flags.append(ReviewFlag(severity="medium", reason="Escalation rated N/A, but an escalation or supervisor request occurred in the call."))
    if events["customer_asks_for_supervisor"] and ratings.get("escalation") == "Good" and not (events["transfer"] or events["escalation_by_agent"]):
        flags.append(ReviewFlag(severity="medium", reason=f"Customer asked for a supervisor at {at('customer_asks_for_supervisor')} but no transfer / escalation was noted, yet Escalation is Good."))
    if ratings.get("politeness_empathy") == "Good" and events["customer_frustration"] and not events["empathy_or_apology"]:
        flags.append(ReviewFlag(severity="low", reason=f"Customer frustration at {at('customer_frustration')} with no empathy statement noted, yet Politeness and Empathy is Good."))
    if ratings.get("active_listening") == "Good" and (
        len(events["customer_repeats_information"]) >= 2 or len(events["agent_interrupts_customer"]) >= 2
    ):
        flags.append(ReviewFlag(severity="medium", reason="Active Listening rated Good despite repeated interruptions or the customer repeating information."))
    if ratings.get("product_knowledge") == "Good" and any(
        c.verdict in ("contradicts_knowledge_base", "self_contradiction") for c in scoring.claim_checks
    ):
        flags.append(ReviewFlag(severity="high", reason="Product/Service Knowledge rated Good although an agent claim contradicts the knowledge base or itself."))
    if events["sensitive_data_request"] and not any(c.error_key == "sensitive_data" for c in scoring.critical_errors):
        flags.append(ReviewFlag(severity="high", reason=f"Agent may have asked for card / OTP / PIN data at {at('sensitive_data_request')} - verify; not reported as a critical error."))
    long_holds = [e for e in forensics.events if e.type == "hold" and (e.duration_seconds or 0) > config.LONG_HOLD_SECONDS]
    if long_holds:
        flags.append(ReviewFlag(severity="low", reason=f"Long hold(s) over {config.LONG_HOLD_SECONDS}s at {', '.join(e.t for e in long_holds[:3])}."))
    return flags


def build_call_result(
    call_id: str,
    audio: AudioSource,
    forensics: CallForensics,
    scoring: ScoringLLMOutput,
    second: Optional[ScoringLLMOutput],
    usage: dict[str, int],
) -> CallResult:
    flags: list[ReviewFlag] = []
    transcript_texts = [_norm(t.text) for t in forensics.transcript]
    by_key = {p.parameter_key.strip().lower(): p for p in scoring.parameters}
    second_by_key = {p.parameter_key.strip().lower(): p for p in (second.parameters if second else [])}

    # Critical errors: keep only known keys.
    critical = [c for c in scoring.critical_errors if c.error_key in config.CRITICAL_ERRORS]
    unknown = [c.error_key for c in scoring.critical_errors if c.error_key not in config.CRITICAL_ERRORS]
    if unknown:
        flags.append(ReviewFlag(severity="medium", reason=f"AI reported unrecognised critical error(s): {', '.join(unknown)}."))
    for c in critical:
        flags.append(ReviewFlag(severity="high", reason=f"Critical error at {c.t}: {config.CRITICAL_ERRORS[c.error_key]} {c.evidence}"))

    results: list[ParameterResult] = []
    for index, param in enumerate(config.PARAMETERS):
        ev = by_key.get(param.key)
        if ev is None and len(scoring.parameters) == len(config.PARAMETERS):
            ev = scoring.parameters[index]  # model mangled the key but kept the order
        if ev is None:
            flags.append(ReviewFlag(severity="high", reason=f"{param.name}: the AI returned no evaluation - rate manually."))
            results.append(ParameterResult(
                key=param.key, name=param.name, rating=None, points=0.0, justification="Not evaluated by the AI.",
                confidence="low", checklist=[], evidence=[],
            ))
            continue

        rating = ev.rating
        justification = ev.justification
        if rating == "N/A" and not param.na_allowed:
            rating = "Average"
            justification = f"[N/A is not allowed for this parameter - set to Average pending review] {justification}"
            flags.append(ReviewFlag(severity="medium", reason=f"{param.name}: AI answered N/A, which is not allowed; set to Average."))
        if param.key == "qa_satisfaction" and critical and rating != "Bad":
            rating = "Bad"
            justification = f"[Set to Bad because a critical error was reported] {justification}"

        checklist = _align_checklist(param, ev.checklist, flags)
        statuses = [c.status for c in checklist]
        if rating == "Good" and "Not met" in statuses:
            flags.append(ReviewFlag(severity="low", reason=f"{param.name}: rated Good while a checklist item is Not met."))
        if rating == "Bad" and statuses and all(s in ("Met", "N/A") for s in statuses):
            flags.append(ReviewFlag(severity="low", reason=f"{param.name}: rated Bad while every checklist item is Met."))

        evidence = _verify_evidence(ev, transcript_texts, audio.duration_seconds)
        unverified = [e for e in evidence if not e.verified]
        if unverified and rating in ("Bad", "Average") and len(unverified) == len(evidence):
            flags.append(ReviewFlag(severity="medium", reason=f"{param.name}: rated {rating}, but none of its quoted evidence was found in the transcript."))
        elif unverified:
            flags.append(ReviewFlag(severity="low", reason=f"{param.name}: {len(unverified)} quote(s) could not be matched to the transcript."))
        if ev.confidence == "low":
            flags.append(ReviewFlag(severity="medium", reason=f"{param.name}: AI confidence is low."))

        second_rating = second_by_key[param.key].rating if param.key in second_by_key else None
        if second_rating == "N/A" and not param.na_allowed:
            second_rating = "Average"  # same correction as the first pass, so it is compared like for like
        if second and second_rating != rating and not (param.key == "qa_satisfaction" and critical):
            gap = {"Good", "Bad"} == {rating, second_rating}
            flags.append(ReviewFlag(
                severity="high" if gap else "medium",
                reason=f"{param.name}: consistency check disagrees ({rating} vs {second_rating}).",
            ))

        if rating == "N/A":
            points = max(config.RATING_POINTS.values()) if config.NA_POLICY == "full_credit" else None
        else:
            points = config.RATING_POINTS[rating]
        results.append(ParameterResult(
            key=param.key, name=param.name, rating=rating, points=points, justification=justification,
            coaching_tip=ev.coaching_tip if rating != "Good" else None, confidence=ev.confidence,
            checklist=checklist, evidence=evidence, second_opinion=second_rating,
        ))

    ratings = {r.key: r.rating for r in results}
    flags.extend(_consistency_flags(forensics, scoring, ratings))
    if forensics.audio_quality == "poor":
        flags.append(ReviewFlag(severity="medium", reason=f"Poor audio quality: {forensics.audio_quality_note or 'hard to hear'}."))
    if not forensics.recording_complete:
        flags.append(ReviewFlag(severity="medium", reason="Recording appears incomplete (starts mid-call or is cut off)."))
    if scoring.reviewer_attention:
        flags.append(ReviewFlag(severity="low", reason=f"AI note: {scoring.reviewer_attention}"))

    score = compute_score(
        [r.rating for r in results], config.RATING_POINTS, config.NA_POLICY,
        critical_error=bool(critical), critical_error_policy=config.CRITICAL_ERROR_POLICY,
    )

    if audio.agent_name:
        agent_name, agent_source = audio.agent_name, "provided"
    elif forensics.agent_name:
        agent_name, agent_source = forensics.agent_name.strip().title(), "detected"
    else:
        agent_name, agent_source = "Unknown agent", "unknown"

    severity_order = {"high": 0, "medium": 1, "low": 2}
    flags.sort(key=lambda f: severity_order[f.severity])

    return CallResult(
        call_id=call_id,
        file_name=audio.file_name,
        source=audio.source,
        agent_name=agent_name,
        agent_name_source=agent_source,
        duration_seconds=audio.duration_seconds,
        forensics=forensics,
        claim_checks=scoring.claim_checks,
        parameters=results,
        critical_errors=critical,
        score=score["score"],
        out_of=score["out_of"],
        percentage=score["percentage"],
        band=config.score_band(score["score"]),
        passed=score["score"] >= config.PASS_MARK and not critical,
        call_summary=scoring.call_summary,
        what_went_well=scoring.what_went_well,
        improvement_areas=scoring.improvement_areas,
        needs_review=any(f.severity in ("high", "medium") for f in flags),
        review_flags=flags,
        usage=usage,
    )

# Evaluates a single call and returns the result.
async def evaluate_call(call_id: str, audio: AudioSource, double_check: bool = False) -> CallResult:
    forensics, usage = await run_forensics(audio)
    passes = [run_scoring(audio, forensics)] + ([run_scoring(audio, forensics)] if double_check else [])
    scored = await asyncio.gather(*passes)
    for _, pass_usage in scored:
        _add_usage(usage, pass_usage)
    second = scored[1][0] if double_check else None
    return build_call_result(call_id, audio, forensics, scored[0][0], second, usage)


# ---------------------------------------------------------------------------
# Agent roll-up
# ---------------------------------------------------------------------------


def _parameter_averages(calls: list[CallResult]) -> list[ParameterAverage]:
    return [
        ParameterAverage(
            key=param.key, name=param.name,
            **rating_breakdown([c.parameters[i].rating for c in calls], config.RATING_POINTS),
        )
        for i, param in enumerate(config.PARAMETERS)
    ]


async def _coaching_brief(agent: str, calls: list[CallResult]) -> tuple[Optional[AgentCoachingLLMOutput], dict]:
    payload = [
        {
            "file_name": c.file_name,
            "category": c.forensics.call_category,
            "score": f"{c.score}/{c.out_of}",
            "critical_errors": [e.error_key for e in c.critical_errors],
            "ratings": {p.name: {"rating": p.rating, "why": p.justification, "tip": p.coaching_tip} for p in c.parameters},
            "what_went_well": c.what_went_well,
            "improvement_areas": c.improvement_areas,
        }
        for c in calls
    ]
    llm = _build_llm(config.ROLLUP_MODEL, "low")
    messages = [
        SystemMessage(content=build_rollup_prompt(agent, len(calls))),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
    ]
    try:
        return await _invoke_structured(llm, AgentCoachingLLMOutput, messages, f"{agent} / coaching")
    except Exception as e:
        logger.warning("Coaching brief failed for %s: %s", agent, e)
        return None, {}


async def build_agent_summaries(calls: list[CallResult]) -> tuple[list[AgentSummary], dict]:
    by_agent: dict[str, list[CallResult]] = defaultdict(list)
    for call in calls:
        by_agent[call.agent_name].append(call)

    # One text-only coaching brief per named agent with 2+ calls (single calls already carry their own tips).
    needs_brief = [a for a, cs in by_agent.items() if len(cs) > 1 and a != "Unknown agent"]
    briefs = await asyncio.gather(*(_coaching_brief(a, by_agent[a]) for a in needs_brief))
    coaching = dict(zip(needs_brief, briefs))

    usage: dict[str, int] = {}
    summaries = []
    for agent, agent_calls in by_agent.items():
        brief, brief_usage = coaching.get(agent, (None, {}))
        _add_usage(usage, brief_usage)
        summaries.append(AgentSummary(
            agent_name=agent,
            calls=len(agent_calls),
            average_score=round(sum(c.score for c in agent_calls) / len(agent_calls), 2),
            pass_rate=round(sum(c.passed for c in agent_calls) / len(agent_calls) * 100, 1),
            calls_needing_review=sum(c.needs_review for c in agent_calls),
            critical_errors=sum(len(c.critical_errors) for c in agent_calls),
            parameter_averages=_parameter_averages(agent_calls),
            coaching=brief,
        ))
    summaries.sort(key=lambda s: s.average_score, reverse=True)
    return summaries, usage


# ---------------------------------------------------------------------------
# Batch entry point
# ---------------------------------------------------------------------------


async def run_batch(
    audios: list[AudioSource],
    failed: Optional[dict[str, str]] = None,
    double_check: bool = False,
    progress: Optional[ProgressCallback] = None,
) -> BatchReport:
    failed = dict(failed or {})
    if not audios:
        raise ValueError(f"No audio could be loaded. {failed}")

    semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_CALLS)
    done = 0

    async def evaluate(index: int, audio: AudioSource):
        nonlocal done
        async with semaphore:
            try:
                return await evaluate_call(f"C{index + 1:03d}", audio, double_check)
            except Exception as e:
                logger.exception("Evaluation failed for %s", audio.file_name)
                return e
            finally:
                done += 1
                if progress:
                    await progress(done, len(audios), audio.file_name)

    results = await asyncio.gather(*(evaluate(i, a) for i, a in enumerate(audios)))

    calls: list[CallResult] = []
    usage: dict[str, int] = {}
    for audio, result in zip(audios, results):
        if isinstance(result, Exception):
            failed[audio.source if audio.source != "upload" else audio.file_name] = f"Evaluation failed: {short_error(result)}"
        else:
            calls.append(result)
            _add_usage(usage, result.usage)

    if not calls:
        raise EvaluationFailedError(f"None of the recordings could be evaluated. {' | '.join(f'{k}: {v}' for k, v in failed.items())}")

    agents, rollup_usage = await build_agent_summaries(calls)
    _add_usage(usage, rollup_usage)

    return BatchReport(
        generated_at=datetime.now().isoformat(timespec="seconds"),
        brand_name=config.BRAND_NAME,
        calls=calls,
        agents=agents,
        team_average_score=round(sum(c.score for c in calls) / len(calls), 2),
        team_pass_rate=round(sum(c.passed for c in calls) / len(calls) * 100, 1),
        failed=failed,
        models={"forensics": config.FORENSICS_MODEL, "scoring": config.SCORING_MODEL, "rollup": config.ROLLUP_MODEL},
        usage=usage,
        config=config.public_config(),
    )
