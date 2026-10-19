import os
import re
import json
import base64
import asyncio
import mimetypes
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlparse, unquote

import httpx
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Default metrics used only when the caller (Streamlit UI / API consumer)
# does not supply their own list of evaluation criteria.
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
]

SCORE_MAX = 10  # keep parity with the existing manual-QC scoring scale

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")

# How many recordings are sent to Gemini at the same time.
MAX_CONCURRENT_CALLS = int(os.environ.get("QC_MAX_CONCURRENCY", "5"))

# Audio is sent inline (base64) to Gemini, so keep each file within the inline request limit.
MAX_UPLOAD_MB = int(os.environ.get("QC_MAX_UPLOAD_MB", "20"))
MAX_UPLOAD_FILES = int(os.environ.get("QC_MAX_UPLOAD_FILES", "20"))

ALLOWED_UPLOAD_EXTENSIONS = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
}

EXTENSION_TO_MIME = {
    ".mp3": "audio/mpeg",
    ".mpeg": "audio/mpeg",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
    ".mp4": "audio/mp4",
    ".flac": "audio/flac",
    ".aac": "audio/aac",
    ".webm": "audio/webm",
}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class CriteriaScore(BaseModel):
    metric_name: str = Field(description="Name of the evaluation metric, exactly as given in the criteria list.")
    score: int = Field(ge=0, le=SCORE_MAX, description=f"Score for this metric, 0-{SCORE_MAX}.")
    observation: str = Field(
        description="1-3 sentence explanation citing specific evidence heard in the call. "
        "If the behavior never came up in the call, say so explicitly."
    )


class AgentPerformanceLLMOutput(BaseModel):
    """
    Exactly what we ask Gemini to produce for ONE call. total_score /
    max_possible_score are deliberately NOT trusted from the model - we compute
    them ourselves afterwards for consistency, same approach used in the
    MetLife QC tool.
    """
    agent_name: Optional[str] = Field(default=None, description="Detected agent name, or null if not identifiable.")
    metrics: list[CriteriaScore] = Field(
        description="One entry per requested evaluation metric, in the same order they were provided."
    )
    performance_summary: str = Field(description="2-4 sentence professional analysis of this call.")
    strengths: list[str] = Field(description="Key strengths observed in this call.")
    weaknesses: list[str] = Field(description="Key weaknesses observed in this call.")
    improvement_suggestions: list[str] = Field(description="Actionable, specific training recommendations.")


class MetricObservation(BaseModel):
    metric_name: str = Field(description="Name of the evaluation metric, exactly as given.")
    observation: str = Field(description="1-3 sentence cross-call observation for this metric.")


class OverallLLMOutput(BaseModel):
    """
    Text-only roll-up written from the already-scored calls. Scores are
    averaged in code, the model only writes the narrative.
    """
    agent_name: Optional[str] = Field(default=None, description="Agent name if consistently identified, else null.")
    metric_observations: list[MetricObservation] = Field(
        description="One entry per evaluation metric, in the same order they were provided."
    )
    performance_summary: str = Field(description="2-4 sentence professional overall analysis across all calls.")
    strengths: list[str] = Field(description="Strengths repeated across calls.")
    weaknesses: list[str] = Field(description="Weaknesses repeated across calls.")
    improvement_suggestions: list[str] = Field(description="Actionable, specific training recommendations.")


class AverageScore(BaseModel):
    metric_name: str
    score: float = Field(description=f"Average score across all evaluated calls, 0-{SCORE_MAX}.")
    observation: str


class CallEvaluation(BaseModel):
    file_name: str
    source: str = Field(description="Original URL, or 'upload' for uploaded files.")
    agent_name: Optional[str] = None
    metrics: list[CriteriaScore]
    total_score: int
    max_possible_score: int
    percentage: float
    call_summary: str
    strengths: list[str]
    weaknesses: list[str]
    improvement_suggestions: list[str]


class AgentPerformanceReport(BaseModel):
    agent_name: Optional[str] = None
    metrics: list[AverageScore]
    total_score: float
    max_possible_score: int
    percentage: float
    performance_summary: str
    strengths: list[str]
    weaknesses: list[str]
    improvement_suggestions: list[str]
    call_evaluations: list[CallEvaluation] = Field(
        default_factory=list, description="Per-recording scores and justifications."
    )
    usage_metadata: Any
    failed_urls: list[str] = Field(
        default_factory=list, description="URLs / file names that could not be downloaded or evaluated."
    )
    failed_reasons: dict[str, str] = Field(
        default_factory=dict, description="Why each entry in failed_urls failed."
    )


class EvaluationFailedError(Exception):
    """Raised when not a single recording could be evaluated (e.g. Gemini quota/billing errors)."""


def _short_error(e: Exception) -> str:
    """Pulls the human-readable part out of verbose Gemini / httpx errors."""
    text = str(e)
    message = re.search(r"'message': '(.*?)'", text)
    if message:
        status = re.search(r"\(([A-Z_]+)\)", text)
        return f"{status.group(1) + ': ' if status else ''}{message.group(1).strip()}"
    return text[:300]


@dataclass
class AudioSource:
    file_name: str
    source: str
    data: bytes
    mime_type: str


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """
You are a Senior Call Center Quality Assurance (QA) Auditor.

You will be given ONE call recording (raw audio, not a transcript). Listen to the audio
directly - use tone of voice, pacing, hesitation, interruptions, and emotional cues in addition
to the words spoken, since these carry information a plain transcript would lose.

GENERAL RULES:
- Base every judgment only on what you actually hear in the recording.
- Do NOT assume or hallucinate missing information.
- If a behavior is not observed in the call, say so explicitly in the observation and score it low
  only if the behavior was expected to occur (e.g. hold procedure was never used because no hold
  was needed should NOT be penalized - use judgment).
- Be objective, analytical, and professional.

SCORING RULE:
Each metric is scored from 0 to {score_max}.
0-2  = Very Poor / Not performed
3-4  = Poor
5-6  = Average
7-8  = Good
9-{score_max} = Excellent

METRICS TO EVALUATE (evaluate ONLY these, in this exact order, using this exact metric_name):
{metrics_block}

OUTPUT INSTRUCTIONS:
Return your evaluation using the structured schema provided to you (`AgentPerformanceLLMOutput`).
Provide exactly {n_metrics} entries in `metrics`, one per metric listed above, in the same order.
Do NOT compute total_score or max_possible_score yourself - that is handled outside the model.
"""

OVERALL_PROMPT_TEMPLATE = """
You are a Senior Call Center Quality Assurance (QA) Auditor.

You will be given the QA evaluations of {n_calls} calls handled by the SAME agent. Each call has
already been scored metric by metric. Write the overall assessment of the agent across all calls.

RULES:
- Use only the evaluations provided. Do NOT invent evidence.
- Focus on PATTERNS across calls (repeated strengths or repeated mistakes), and mention call
  file names when a point applies to specific calls only.
- Do NOT re-score anything - the scores are final.

METRICS (return one observation per metric, in this exact order, using this exact metric_name):
{metrics_block}

Return your answer using the structured schema provided to you (`OverallLLMOutput`).
"""


def _metrics_block(metrics: list[str]) -> str:
    return "\n".join(f"{i}. {m}" for i, m in enumerate(metrics, start=1))


def _build_system_prompt(metrics: list[str]) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        score_max=SCORE_MAX,
        metrics_block=_metrics_block(metrics),
        n_metrics=len(metrics),
    )


def _build_overall_prompt(metrics: list[str], n_calls: int) -> str:
    return OVERALL_PROMPT_TEMPLATE.format(metrics_block=_metrics_block(metrics), n_calls=n_calls)


# ---------------------------------------------------------------------------
# Audio inputs (URLs + uploads)
# ---------------------------------------------------------------------------


def _guess_mime_type(url: str, content_type_header: Optional[str]) -> str:
    if content_type_header and content_type_header.startswith("audio/"):
        return content_type_header.split(";")[0].strip()

    path = urlparse(url).path
    ext = os.path.splitext(path)[1].lower()
    if ext in EXTENSION_TO_MIME:
        return EXTENSION_TO_MIME[ext]

    guessed, _ = mimetypes.guess_type(path)
    if guessed and guessed.startswith("audio/"):
        return guessed

    return "audio/mpeg"  # reasonable default; Gemini is tolerant of this for mp3-like data


def _file_name_from_url(url: str) -> str:
    return unquote(os.path.basename(urlparse(url).path)) or url


def _looks_like_audio(ext: str, data: bytes) -> bool:
    if ext == ".wav":
        return data[:4] == b"RIFF" and data[8:12] == b"WAVE"
    if ext == ".mp3":
        # ID3 tag, or a bare MPEG frame sync
        return data[:3] == b"ID3" or (len(data) > 1 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0)
    return False


def build_upload_source(file_name: str, data: bytes) -> AudioSource:
    """Validates an uploaded recording and wraps it as an AudioSource. Raises ValueError if invalid."""
    file_name = os.path.basename(file_name or "recording")
    ext = os.path.splitext(file_name)[1].lower()
    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        raise ValueError(f"'{file_name}': only .mp3 and .wav files are supported.")
    if not data:
        raise ValueError(f"'{file_name}' is empty.")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"'{file_name}' is larger than the {MAX_UPLOAD_MB} MB limit.")
    if not _looks_like_audio(ext, data):
        raise ValueError(f"'{file_name}' does not look like a valid {ext[1:].upper()} file.")
    return AudioSource(file_name=file_name, source="upload", data=data, mime_type=ALLOWED_UPLOAD_EXTENSIONS[ext])


async def _fetch_audio(client: httpx.AsyncClient, url: str) -> tuple[str, Optional[AudioSource], Optional[str]]:
    """Returns (url, AudioSource_or_None, error_or_None). None means the download failed - caller decides how to handle that."""
    try:
        resp = await client.get(url, timeout=60.0, follow_redirects=True)
        resp.raise_for_status()
        mime_type = _guess_mime_type(url, resp.headers.get("content-type"))
        return url, AudioSource(file_name=_file_name_from_url(url), source=url, data=resp.content, mime_type=mime_type), None
    except Exception as e:
        print(f"Failed to download audio from {url}: {e}")
        return url, None, f"Download failed: {_short_error(e)}"


async def fetch_all_audio(urls: list[str]) -> tuple[list[AudioSource], dict[str, str]]:
    """
    Downloads all URLs concurrently.
    Returns (successful=[AudioSource, ...], failed={url: reason, ...}).
    """
    if not urls:
        return [], {}

    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*(_fetch_audio(client, url) for url in urls))

    successful = [audio for _, audio, _ in results if audio is not None]
    failed = {url: error for url, audio, error in results if audio is None}
    return successful, failed


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def _align_metrics(returned: list[CriteriaScore], requested: list[str]) -> list[CriteriaScore]:
    """
    Forces the model output onto the requested metric names / order and clamps
    scores. Falls back to name matching if the model returned a different count.
    """
    if len(returned) == len(requested):
        pairs = list(zip(requested, returned))
    else:
        by_name = {m.metric_name.strip().lower(): m for m in returned}
        pairs = [(name, by_name.get(name.strip().lower())) for name in requested]

    aligned = []
    for name, m in pairs:
        if m is None:
            aligned.append(
                CriteriaScore(metric_name=name, score=0, observation="The model did not return a score for this metric.")
            )
        else:
            aligned.append(
                CriteriaScore(metric_name=name, score=max(0, min(SCORE_MAX, m.score)), observation=m.observation)
            )
    return aligned


def _describe_failures(failed: dict[str, str]) -> str:
    return " | ".join(f"{name}: {reason}" for name, reason in failed.items())


def _sum_usage(usages: list[Any]) -> Optional[dict]:
    total: dict = {}
    for usage in usages:
        if not isinstance(usage, dict):
            continue
        for key, value in usage.items():
            if isinstance(value, int):
                total[key] = total.get(key, 0) + value
    return total or None


async def _evaluate_call(
    structured_llm,
    system_message: SystemMessage,
    metrics: list[str],
    audio: AudioSource,
) -> tuple[CallEvaluation, Any]:
    human_message = HumanMessage(
        content=[
            {
                "type": "media",
                "mime_type": audio.mime_type,
                "data": base64.standard_b64encode(audio.data).decode("utf-8"),
            },
            {
                "type": "text",
                "text": (
                    "Above is one call recording. Evaluate the agent's performance on this call "
                    "according to the metrics in your instructions."
                ),
            },
        ]
    )

    result = await structured_llm.ainvoke([system_message, human_message])
    raw_message = result["raw"]
    parsed: Optional[AgentPerformanceLLMOutput] = result["parsed"]
    usage_metadata = getattr(raw_message, "usage_metadata", None)
    print(f"[{audio.file_name}] usage metadata: {usage_metadata}")

    if parsed is None:
        raise ValueError(
            f"Gemini did not return a schema-conformant response. "
            f"parsing_error={result.get('parsing_error')!r}, raw_content={raw_message.content!r}"
        )

    # Compute totals ourselves - do not trust the model's arithmetic.
    scores = _align_metrics(parsed.metrics, metrics)
    total_score = sum(m.score for m in scores)
    max_possible_score = len(scores) * SCORE_MAX
    percentage = round((total_score / max_possible_score) * 100, 2) if max_possible_score else 0.0

    evaluation = CallEvaluation(
        file_name=audio.file_name,
        source=audio.source,
        agent_name=parsed.agent_name,
        metrics=scores,
        total_score=total_score,
        max_possible_score=max_possible_score,
        percentage=percentage,
        call_summary=parsed.performance_summary,
        strengths=parsed.strengths,
        weaknesses=parsed.weaknesses,
        improvement_suggestions=parsed.improvement_suggestions,
    )
    return evaluation, usage_metadata


async def _summarize_overall(llm, metrics: list[str], calls: list[CallEvaluation]) -> tuple[OverallLLMOutput, Any]:
    structured_llm = llm.with_structured_output(OverallLLMOutput, include_raw=True)
    calls_payload = [
        {
            "file_name": c.file_name,
            "agent_name": c.agent_name,
            "percentage": c.percentage,
            "metrics": [m.model_dump() for m in c.metrics],
            "call_summary": c.call_summary,
            "strengths": c.strengths,
            "weaknesses": c.weaknesses,
        }
        for c in calls
    ]

    result = await structured_llm.ainvoke(
        [
            SystemMessage(content=_build_overall_prompt(metrics, len(calls))),
            HumanMessage(content=json.dumps(calls_payload, ensure_ascii=False, indent=2)),
        ]
    )
    parsed: Optional[OverallLLMOutput] = result["parsed"]
    usage_metadata = getattr(result["raw"], "usage_metadata", None)
    if parsed is None:
        raise ValueError(f"Overall summary was not schema-conformant: {result.get('parsing_error')!r}")
    return parsed, usage_metadata


def _fallback_overall(calls: list[CallEvaluation], metrics: list[str], error: Exception) -> OverallLLMOutput:
    """Used when the roll-up call fails, so the per-call results are not lost."""
    return OverallLLMOutput(
        agent_name=next((c.agent_name for c in calls if c.agent_name), None),
        metric_observations=[
            MetricObservation(
                metric_name=name,
                observation=" | ".join(f"{c.file_name}: {c.metrics[i].observation}" for c in calls),
            )
            for i, name in enumerate(metrics)
        ],
        performance_summary=f"Overall narrative could not be generated ({error}). See the per-call results.",
        strengths=list(dict.fromkeys(s for c in calls for s in c.strengths)),
        weaknesses=list(dict.fromkeys(w for c in calls for w in c.weaknesses)),
        improvement_suggestions=list(dict.fromkeys(s for c in calls for s in c.improvement_suggestions)),
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def run_quality_check(
    urls: Optional[list[str]] = None,
    performance_types: Optional[list[str]] = None,
    uploads: Optional[list[AudioSource]] = None,
) -> AgentPerformanceReport:
    """
    Evaluates every recording (downloaded URLs + uploaded files) independently
    and concurrently with Gemini's native audio understanding, then rolls the
    per-call scores up into one agent-level report. Per-call scores are what
    the Excel export uses; averages are computed in code, not by the model.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set.")

    metrics = performance_types if performance_types else DEFAULT_METRICS

    downloaded, failed = await fetch_all_audio(urls or [])
    audios = list(uploads or []) + downloaded
    if not audios:
        raise ValueError(f"No audio could be loaded. {_describe_failures(failed)}")

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=api_key,
        temperature=0.1,
    )
    structured_llm = llm.with_structured_output(AgentPerformanceLLMOutput, include_raw=True)
    system_message = SystemMessage(content=_build_system_prompt(metrics))
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_CALLS)

    async def evaluate(audio: AudioSource):
        async with semaphore:
            return await _evaluate_call(structured_llm, system_message, metrics, audio)

    results = await asyncio.gather(*(evaluate(a) for a in audios), return_exceptions=True)

    calls: list[CallEvaluation] = []
    usages: list[Any] = []
    for audio, result in zip(audios, results):
        if isinstance(result, Exception):
            print(f"Evaluation failed for {audio.file_name}: {result}")
            failed[audio.source if audio.source != "upload" else audio.file_name] = f"Evaluation failed: {_short_error(result)}"
            continue
        calls.append(result[0])
        usages.append(result[1])

    if not calls:
        raise EvaluationFailedError(f"None of the recordings could be evaluated. {_describe_failures(failed)}")

    # ------------------------------------------------------------------
    # Roll-up: averages in code, narrative from the model (single call needs no roll-up).
    # ------------------------------------------------------------------
    if len(calls) == 1:
        only = calls[0]
        overall = OverallLLMOutput(
            agent_name=only.agent_name,
            metric_observations=[MetricObservation(metric_name=m.metric_name, observation=m.observation) for m in only.metrics],
            performance_summary=only.call_summary,
            strengths=only.strengths,
            weaknesses=only.weaknesses,
            improvement_suggestions=only.improvement_suggestions,
        )
    else:
        try:
            overall, usage = await _summarize_overall(llm, metrics, calls)
            usages.append(usage)
        except Exception as e:
            print(f"Overall summary failed: {e}")
            overall = _fallback_overall(calls, metrics, e)

    observations = {o.metric_name.strip().lower(): o.observation for o in overall.metric_observations}
    average_scores = [
        AverageScore(
            metric_name=name,
            score=round(sum(c.metrics[i].score for c in calls) / len(calls), 2),
            observation=observations.get(name.strip().lower(), ""),
        )
        for i, name in enumerate(metrics)
    ]

    total_score = round(sum(m.score for m in average_scores), 2)
    max_possible_score = len(average_scores) * SCORE_MAX
    percentage = round((total_score / max_possible_score) * 100, 2) if max_possible_score else 0.0

    return AgentPerformanceReport(
        agent_name=overall.agent_name,
        metrics=average_scores,
        total_score=total_score,
        max_possible_score=max_possible_score,
        percentage=percentage,
        performance_summary=overall.performance_summary,
        strengths=overall.strengths,
        weaknesses=overall.weaknesses,
        improvement_suggestions=overall.improvement_suggestions,
        call_evaluations=calls,
        usage_metadata=_sum_usage(usages),
        failed_urls=list(failed),
        failed_reasons=failed,
    )
