import os
import base64
import asyncio
import mimetypes
from typing import Any, Optional
from urllib.parse import urlparse

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
        description="1-3 sentence explanation citing specific evidence heard across the call(s). "
        "If the behavior never came up in any call, say so explicitly."
    )


class AgentPerformanceLLMOutput(BaseModel):
    """
    Exactly what we ask Gemini to produce. total_score / max_possible_score are
    deliberately NOT trusted from the model - we compute them ourselves
    afterwards for consistency, same approach used in the MetLife QC tool.
    """
    agent_name: Optional[str] = Field(default=None, description="Detected agent name, or null if not identifiable.")
    metrics: list[CriteriaScore] = Field(
        description="One entry per requested evaluation metric, in the same order they were provided."
    )
    performance_summary: str = Field(description="2-4 sentence professional overall analysis.")
    strengths: list[str] = Field(description="Key strengths observed, ideally patterns repeated across calls.")
    weaknesses: list[str] = Field(description="Key weaknesses observed, ideally patterns repeated across calls.")
    improvement_suggestions: list[str] = Field(description="Actionable, specific training recommendations.")


class AgentPerformanceReport(BaseModel):
    agent_name: Optional[str] = None
    metrics: list[CriteriaScore]
    total_score: int
    max_possible_score: int
    percentage: float
    performance_summary: str
    strengths: list[str]
    weaknesses: list[str]
    improvement_suggestions: list[str]
    usage_metadata: Any
    failed_urls: list[str] = Field(default_factory=list, description="URLs that could not be downloaded/evaluated.")


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """
You are a Senior Call Center Quality Assurance (QA) Auditor.

You will be given one or more call recordings (raw audio, not a transcript) handled by the
SAME agent. Listen to the audio directly - use tone of voice, pacing, hesitation, interruptions,
and emotional cues in addition to the words spoken, since these carry information a plain
transcript would lose.

GENERAL RULES:
- Base every judgment only on what you actually hear in the recordings.
- Do NOT assume or hallucinate missing information.
- If a behavior is not observed in any call, say so explicitly in the observation and score it low
  only if the behavior was expected to occur (e.g. hold procedure was never used because no hold
  was needed should NOT be penalized - use judgment).
- Look for patterns ACROSS calls (repeated strengths or repeated mistakes), not just a single call.
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


def _build_system_prompt(metrics: list[str]) -> str:
    metrics_block = "\n".join(f"{i}. {m}" for i, m in enumerate(metrics, start=1))
    return SYSTEM_PROMPT_TEMPLATE.format(
        score_max=SCORE_MAX,
        metrics_block=metrics_block,
        n_metrics=len(metrics),
    )


# ---------------------------------------------------------------------------
# Audio fetching
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


async def _fetch_audio(client: httpx.AsyncClient, url: str) -> tuple[str, Optional[bytes], Optional[str]]:
    """
    Returns (url, audio_bytes_or_None, mime_type_or_None).
    audio_bytes is None if the download failed - caller decides how to handle that.
    """
    try:
        resp = await client.get(url, timeout=60.0, follow_redirects=True)
        resp.raise_for_status()
        mime_type = _guess_mime_type(url, resp.headers.get("content-type"))
        return url, resp.content, mime_type
    except Exception as e:
        print(f"Failed to download audio from {url}: {e}")
        return url, None, None


async def fetch_all_audio(urls: list[str]) -> tuple[list[tuple[str, bytes, str]], list[str]]:
    """
    Downloads all URLs concurrently.
    Returns (successful=[(url, bytes, mime_type), ...], failed_urls=[...]).
    """
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*(_fetch_audio(client, url) for url in urls))

    successful = [(url, data, mime) for url, data, mime in results if data is not None]
    failed = [url for url, data, _ in results if data is None]
    return successful, failed


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def run_quality_check(
    urls: list[str],
    performance_types: Optional[list[str]] = None,
) -> AgentPerformanceReport:
    """
    Downloads every recording, sends them all to Gemini in a single request as
    inline audio (so the model can reason about patterns across calls using the
    actual audio - not a transcript), and returns a structured, arithmetic-checked
    performance report.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set.")

    metrics = performance_types if performance_types else DEFAULT_METRICS

    successful, failed = await fetch_all_audio(urls)
    if not successful:
        raise ValueError(f"Could not download any of the provided audio URLs. Failed: {failed}")

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        google_api_key=api_key,
        temperature=0.1,
    )
    structured_llm = llm.with_structured_output(AgentPerformanceLLMOutput, include_raw=True)

    content_parts: list[dict] = []
    for url, audio_bytes, mime_type in successful:
        content_parts.append(
            {
                "type": "media",
                "mime_type": mime_type,
                "data": base64.standard_b64encode(audio_bytes).decode("utf-8"),
            }
        )
    content_parts.append(
        {
            "type": "text",
            "text": (
                f"Above are {len(successful)} call recording(s) from the same agent. "
                "Evaluate the agent's performance across all of them according to the "
                "metrics in your instructions."
            ),
        }
    )

    human_message = HumanMessage(content=content_parts)
    system_message = SystemMessage(content=_build_system_prompt(metrics))

    result = await structured_llm.ainvoke([system_message, human_message])

    raw_message = result["raw"]
    parsed: Optional[AgentPerformanceLLMOutput] = result["parsed"]
    parsing_error = result.get("parsing_error")

    usage_metadata = getattr(raw_message, "usage_metadata", None)
    print(f"usage metadata: {usage_metadata}")

    if parsed is None:
        raise ValueError(
            f"Gemini did not return a schema-conformant response. "
            f"parsing_error={parsing_error!r}, raw_content={raw_message.content!r}"
        )

    # ------------------------------------------------------------------
    # Compute totals ourselves - do not trust the model's arithmetic.
    # ------------------------------------------------------------------
    clamped_metrics = [
        CriteriaScore(
            metric_name=m.metric_name,
            score=max(0, min(SCORE_MAX, m.score)),
            observation=m.observation,
        )
        for m in parsed.metrics
    ]

    total_score = sum(m.score for m in clamped_metrics)
    max_possible_score = len(clamped_metrics) * SCORE_MAX
    percentage = round((total_score / max_possible_score) * 100, 2) if max_possible_score else 0.0

    return AgentPerformanceReport(
        agent_name=parsed.agent_name,
        metrics=clamped_metrics,
        total_score=total_score,
        max_possible_score=max_possible_score,
        percentage=percentage,
        performance_summary=parsed.performance_summary,
        strengths=parsed.strengths,
        weaknesses=parsed.weaknesses,
        improvement_suggestions=parsed.improvement_suggestions,
        usage_metadata=usage_metadata,
        failed_urls=failed,
    )