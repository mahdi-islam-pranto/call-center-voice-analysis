"""
Customer Call Transcript Generator

POST /transcribe with a call recording URL -> the audio is downloaded, sent to
an LLM through LangChain (Gemini 3.1 Flash-Lite by default), and the transcript
is returned split into what the agent said and what the customer said, together
with the token usage and cost of the request.

The model is chosen with TRANSCRIPT_MODEL in .env ("provider:model"), so it
can be switched without touching the code.
"""
import asyncio
import base64
import io
import math
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Optional
from urllib.parse import unquote, urlparse

import httpx
import mutagen
import uvicorn
from mutagen.aac import AAC
from mutagen.aiff import AIFF
from mutagen.flac import FLAC
from mutagen.mp3 import MP3
from mutagen.mp4 import MP4
from mutagen.oggvorbis import OggVorbis
from mutagen.wave import WAVE
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, HttpUrl

from prompts import TRANSCRIPT_SYSTEM_PROMPT, TRANSCRIPT_USER_PROMPT

load_dotenv()

# ---------------------------------------------------------------------------
# Settings (overridable from .env)
# ---------------------------------------------------------------------------

# Any LangChain chat model as "provider:model". The model must accept audio input.
LLM_MODEL = os.environ.get("TRANSCRIPT_MODEL", "google_genai:gemini-3.1-flash-lite")
# minimal | low | medium | high (blank = model default). Gemini calls this thinking_level.
REASONING_EFFORT = os.environ.get("TRANSCRIPT_REASONING_EFFORT", "low")
# Extra attempts when the model's reply is not a valid transcript.
# (Rate limits and temporary API errors are retried separately by LangChain.)
LLM_RETRIES = int(os.environ.get("TRANSCRIPT_LLM_RETRIES", "1"))
MAX_AUDIO_MB = int(os.environ.get("TRANSCRIPT_MAX_AUDIO_MB", "20"))
DOWNLOAD_TIMEOUT_SECONDS = float(os.environ.get("TRANSCRIPT_DOWNLOAD_TIMEOUT", "60"))
# Batch: most URLs per request, and how many recordings are processed at the same time.
MAX_BATCH_URLS = int(os.environ.get("TRANSCRIPT_MAX_BATCH_URLS", "50"))
MAX_CONCURRENCY = int(os.environ.get("TRANSCRIPT_MAX_CONCURRENCY", "4"))
# Comma-separated hosts whose SSL certificate is NOT checked when downloading,
# e.g. an internal recording server reached by IP whose certificate is issued for a domain name.
SSL_SKIP_VERIFY_HOSTS = {
    h.strip().lower() for h in os.environ.get("TRANSCRIPT_SSL_SKIP_VERIFY_HOSTS", "").split(",") if h.strip()
}

# Prices in USD per 1 million tokens. Defaults are Google's paid-tier prices for
# gemini-3.1-flash-lite; update them if you change TRANSCRIPT_MODEL.
PRICE_TEXT_INPUT_PER_1M = float(os.environ.get("TRANSCRIPT_PRICE_TEXT_INPUT_PER_1M", "0.25"))
PRICE_AUDIO_INPUT_PER_1M = float(os.environ.get("TRANSCRIPT_PRICE_AUDIO_INPUT_PER_1M", "0.50"))
PRICE_OUTPUT_PER_1M = float(os.environ.get("TRANSCRIPT_PRICE_OUTPUT_PER_1M", "1.50"))  # includes thinking tokens
# Gemini counts 32 tokens per second of audio. Used to split input tokens into audio vs text,
# since LangChain reports only the input total and audio input is priced differently.
AUDIO_TOKENS_PER_SECOND = float(os.environ.get("TRANSCRIPT_AUDIO_TOKENS_PER_SECOND", "32"))

# Extension -> MIME type, used when the file's bytes and headers don't say.
AUDIO_EXTENSIONS = {
    ".mp3": "audio/mp3",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".aiff": "audio/aiff",
}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class TranscriptRequest(BaseModel):
    audio_url: HttpUrl = Field(description="Direct link to the audio file (mp3, wav, m4a, aac, ogg, flac, aiff).")


class Segment(BaseModel):
    speaker: Literal["Agent", "Customer"] = Field(description="Who is speaking: the call-centre Agent or the Customer.")
    start: str = Field(description="When the segment starts, MM:SS from the start of the recording.")
    text: str = Field(description="Verbatim text of what was said.")


class TranscriptOutput(BaseModel):
    """The structured output the model is asked to fill."""

    language: str = Field(description='Main language(s) spoken, e.g. "Bangla, English".')
    segments: list[Segment]


class SplitTranscript(BaseModel):
    agent: str = Field(description='Everything the agent said, one "[MM:SS] text" line per segment.')
    customer: str = Field(description='Everything the customer said, one "[MM:SS] text" line per segment.')


class TokenUsage(BaseModel):
    input_tokens: int = Field(description="All input tokens (audio + prompt text).")
    audio_input_tokens: int = Field(description="Input tokens for the audio, estimated from its duration.")
    text_input_tokens: int = Field(description="Input tokens for the prompt text (input_tokens - audio_input_tokens).")
    output_tokens: int = Field(description="Output tokens, including thinking tokens.")
    thinking_tokens: int = Field(description="Part of output_tokens spent on thinking.")
    total_tokens: int
    model_calls: int = Field(description="Model calls made (more than 1 when an invalid reply was retried; all are billed).")
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float


class TranscriptResponse(BaseModel):
    audio_url: str
    file_name: str
    mime_type: str
    size_bytes: int
    duration_seconds: Optional[float] = Field(description="Recording length, if it could be measured.")
    model: str
    language: str
    segments: list[Segment] = Field(description="The whole conversation in order, each line labelled Agent or Customer.")
    transcript: SplitTranscript
    usage: TokenUsage
    processing_seconds: float


class BatchTranscriptRequest(BaseModel):
    audio_urls: list[HttpUrl] = Field(min_length=1, description="Direct links to the audio files; they can be on different servers.")


class BatchItem(BaseModel):
    audio_url: str
    status: Literal["ok", "failed"]
    result: Optional[TranscriptResponse] = None
    error: Optional[str] = None


class BatchTranscriptResponse(BaseModel):
    total: int
    succeeded: int
    failed: int
    usage: TokenUsage = Field(description="Tokens and cost added up over the recordings that succeeded.")
    items: list[BatchItem] = Field(description="One item per URL, in the same order as the request.")
    processing_seconds: float


# ---------------------------------------------------------------------------
# Audio download
# ---------------------------------------------------------------------------


class DownloadError(Exception):
    pass


@dataclass
class DownloadedAudio:
    file_name: str
    mime_type: str
    data: bytes
    duration_seconds: Optional[float] = None


# mutagen reader per MIME type. Auto-detection alone misses MP3s without an ID3 tag
# (common on call-recording servers), so the format we detected is used first.
_MUTAGEN_READERS = {
    "audio/mp3": MP3,
    "audio/wav": WAVE,
    "audio/mp4": MP4,
    "audio/aac": AAC,
    "audio/ogg": OggVorbis,
    "audio/flac": FLAC,
    "audio/aiff": AIFF,
}


def measure_duration(data: bytes, mime_type: str) -> Optional[float]:
    """Recording length in seconds, or None if the format can't be read."""
    for reader in (_MUTAGEN_READERS.get(mime_type), mutagen.File):
        if reader is None:
            continue
        try:
            audio = reader(io.BytesIO(data))
            if audio is not None and audio.info.length:
                return round(audio.info.length, 1)
        except Exception:
            pass
    return None


def _sniff_mime_type(data: bytes) -> Optional[str]:
    """Identifies the audio format from the file's first bytes."""
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "audio/wav"
    if data[:3] == b"ID3":
        return "audio/mp3"
    if data[:4] == b"OggS":
        return "audio/ogg"
    if data[:4] == b"fLaC":
        return "audio/flac"
    if data[:4] == b"FORM" and data[8:12] in (b"AIFF", b"AIFC"):
        return "audio/aiff"
    if data[4:8] == b"ftyp":
        return "audio/mp4"
    if len(data) > 1 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0:
        # MPEG frame sync: layer bits 00 means AAC (ADTS), otherwise MP3.
        return "audio/aac" if (data[1] & 0x06) == 0 else "audio/mp3"
    return None


def _detect_mime_type(data: bytes, url: str, content_type: str) -> Optional[str]:
    sniffed = _sniff_mime_type(data)
    if sniffed:
        return sniffed
    content_type = content_type.split(";")[0].strip().lower()
    if content_type.startswith("audio/"):
        return content_type
    return AUDIO_EXTENSIONS.get(os.path.splitext(urlparse(url).path)[1].lower())


async def download_audio(url: str) -> DownloadedAudio:
    """Streams the file into memory, enforcing the size limit, and checks that it is audio."""
    max_bytes = MAX_AUDIO_MB * 1024 * 1024
    timeout = httpx.Timeout(DOWNLOAD_TIMEOUT_SECONDS, connect=10.0)
    verify_ssl = (urlparse(url).hostname or "").lower() not in SSL_SKIP_VERIFY_HOSTS
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=timeout, verify=verify_ssl) as client:
            async with client.stream("GET", url) as resp:
                if resp.status_code >= 400:
                    raise DownloadError(f"The audio URL returned HTTP {resp.status_code}.")
                declared = resp.headers.get("content-length", "")
                if declared.isdigit() and int(declared) > max_bytes:
                    raise DownloadError(f"The audio is larger than the {MAX_AUDIO_MB} MB limit.")

                chunks, size = [], 0
                async for chunk in resp.aiter_bytes():
                    size += len(chunk)
                    if size > max_bytes:
                        raise DownloadError(f"The audio is larger than the {MAX_AUDIO_MB} MB limit.")
                    chunks.append(chunk)

                final_url = str(resp.url)
                content_type = resp.headers.get("content-type", "")
    except httpx.HTTPError as e:
        raise DownloadError(f"Could not download the audio: {e.__class__.__name__}: {e}") from e

    data = b"".join(chunks)
    if not data:
        raise DownloadError("The audio URL returned an empty file.")

    mime_type = _detect_mime_type(data, final_url, content_type)
    if not mime_type:
        raise DownloadError(
            f"The URL did not return an audio file (content-type: {content_type or 'unknown'}). "
            "Use a direct download link, not a web page."
        )

    file_name = unquote(os.path.basename(urlparse(final_url).path)) or "audio"
    return DownloadedAudio(
        file_name=file_name, mime_type=mime_type, data=data, duration_seconds=measure_duration(data, mime_type)
    )


# ---------------------------------------------------------------------------
# Transcription (LangChain)
# ---------------------------------------------------------------------------


# Finish reasons meaning "stopped at the output limit" (Gemini: MAX_TOKENS, OpenAI: length).
TRUNCATED_FINISH_REASONS = ("MAX_TOKENS", "LENGTH")


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """
    Builds the chat model once. The API key is read from the provider's usual
    environment variable (GOOGLE_API_KEY for Gemini, OPENAI_API_KEY for OpenAI, ...).
    """
    kwargs = {"max_retries": 3}
    if REASONING_EFFORT:
        kwargs["reasoning_effort"] = REASONING_EFFORT
    return init_chat_model(LLM_MODEL, **kwargs)


def build_messages(audio: DownloadedAudio) -> list:
    return [
        SystemMessage(content=TRANSCRIPT_SYSTEM_PROMPT),
        HumanMessage(content=[
            # LangChain's standard audio block: each provider package converts it to its own format.
            {
                "type": "audio",
                "base64": base64.b64encode(audio.data).decode("ascii"),
                "mime_type": audio.mime_type,
            },
            {"type": "text", "text": TRANSCRIPT_USER_PROMPT},
        ]),
    ]


async def transcribe_audio(audio: DownloadedAudio) -> tuple[TranscriptOutput, TokenUsage]:
    structured_llm = get_llm().with_structured_output(TranscriptOutput, include_raw=True)
    messages = build_messages(audio)

    raw_usages = []  # every attempt is billed, so all of them count toward the cost
    error = None
    for _ in range(LLM_RETRIES + 1):
        result = await structured_llm.ainvoke(messages)
        raw_usages.append(result["raw"].usage_metadata or {})
        # LangChain auto-closes unfinished JSON, so a reply cut off at the output
        # limit would otherwise come back as a shorter "valid" transcript.
        finish_reason = str(result["raw"].response_metadata.get("finish_reason", "")).upper()
        if finish_reason.endswith(TRUNCATED_FINISH_REASONS):
            raise RuntimeError(
                "The transcript was cut off because it reached the model's output limit. "
                "The recording is too long for one request."
            )
        if result["parsed"] is not None:
            return result["parsed"], calculate_usage(raw_usages, audio.duration_seconds)
        error = result.get("parsing_error")
    raise RuntimeError(f"The model did not return a valid transcript ({error!r}).")


def _usd(tokens: int, price_per_1m: float) -> float:
    return tokens * price_per_1m / 1_000_000


def calculate_usage(raw_usages: list[dict], duration_seconds: Optional[float]) -> TokenUsage:
    """
    Adds up LangChain's usage_metadata over all model calls and prices it.
    Audio and text input have different prices, so the audio share is estimated
    from the duration. If the duration is unknown, all input is priced as audio
    (the higher rate), so the cost is never under-reported.
    """
    calls = len(raw_usages)
    input_tokens = sum(u.get("input_tokens", 0) for u in raw_usages)
    output_tokens = sum(u.get("output_tokens", 0) for u in raw_usages)
    thinking_tokens = sum((u.get("output_token_details") or {}).get("reasoning", 0) for u in raw_usages)
    total_tokens = sum(u.get("total_tokens", 0) for u in raw_usages) or input_tokens + output_tokens

    if duration_seconds:
        audio_tokens = min(input_tokens, calls * math.ceil(duration_seconds * AUDIO_TOKENS_PER_SECOND))
    else:
        audio_tokens = input_tokens
    text_tokens = input_tokens - audio_tokens

    input_cost = _usd(audio_tokens, PRICE_AUDIO_INPUT_PER_1M) + _usd(text_tokens, PRICE_TEXT_INPUT_PER_1M)
    output_cost = _usd(output_tokens, PRICE_OUTPUT_PER_1M)
    return TokenUsage(
        input_tokens=input_tokens,
        audio_input_tokens=audio_tokens,
        text_input_tokens=text_tokens,
        output_tokens=output_tokens,
        thinking_tokens=thinking_tokens,
        total_tokens=total_tokens,
        model_calls=calls,
        input_cost_usd=round(input_cost, 6),
        output_cost_usd=round(output_cost, 6),
        total_cost_usd=round(input_cost + output_cost, 6),
    )


def sum_usage(usages: list[TokenUsage]) -> TokenUsage:
    totals = {field: sum(getattr(u, field) for u in usages) for field in TokenUsage.model_fields}
    for field in ("input_cost_usd", "output_cost_usd", "total_cost_usd"):
        totals[field] = round(totals[field], 6)
    return TokenUsage(**totals)


def split_transcript(segments: list[Segment]) -> SplitTranscript:
    def lines(speaker: str) -> str:
        return "\n".join(f"[{s.start}] {s.text}" for s in segments if s.speaker == speaker)

    return SplitTranscript(agent=lines("Agent"), customer=lines("Customer"))


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Customer Call Transcript Generator",
    description=(
        "Downloads a call recording from a URL and returns separate Agent and Customer transcripts "
        "generated by an LLM via LangChain, with token usage and cost."
    ),
    version="1.0.0",
)


class TranscriptionError(Exception):
    pass


# Shared by all requests, so parallel batches together never exceed MAX_CONCURRENCY model calls.
_concurrency = asyncio.Semaphore(MAX_CONCURRENCY)


async def process_url(audio_url: str) -> TranscriptResponse:
    """Download + transcribe one URL. Raises DownloadError or TranscriptionError."""
    started = time.perf_counter()
    async with _concurrency:
        audio = await download_audio(audio_url)
        try:
            result, usage = await transcribe_audio(audio)
        except Exception as e:
            raise TranscriptionError(f"Transcription failed: {str(e)[:300]}") from e

    return TranscriptResponse(
        audio_url=audio_url,
        file_name=audio.file_name,
        mime_type=audio.mime_type,
        size_bytes=len(audio.data),
        duration_seconds=audio.duration_seconds,
        model=LLM_MODEL,
        language=result.language,
        segments=result.segments,
        transcript=split_transcript(result.segments),
        usage=usage,
        processing_seconds=round(time.perf_counter() - started, 2),
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model": LLM_MODEL}


@app.post("/transcribe", response_model=TranscriptResponse)
async def transcribe(request: TranscriptRequest):
    try:
        return await process_url(str(request.audio_url))
    except DownloadError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except TranscriptionError as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/transcribe/batch", response_model=BatchTranscriptResponse)
async def transcribe_batch(request: BatchTranscriptRequest):
    """
    Transcribes many recordings (from any number of servers) in one request.
    A failed URL does not stop the others: it comes back with status "failed" and the reason.
    """
    if len(request.audio_urls) > MAX_BATCH_URLS:
        raise HTTPException(status_code=400, detail=f"At most {MAX_BATCH_URLS} URLs per batch.")
    started = time.perf_counter()

    async def run_one(url: str) -> BatchItem:
        try:
            return BatchItem(audio_url=url, status="ok", result=await process_url(url))
        except (DownloadError, TranscriptionError) as e:
            return BatchItem(audio_url=url, status="failed", error=str(e))

    items = await asyncio.gather(*(run_one(str(u)) for u in request.audio_urls))
    succeeded = sum(item.status == "ok" for item in items)
    return BatchTranscriptResponse(
        total=len(items),
        succeeded=succeeded,
        failed=len(items) - succeeded,
        usage=sum_usage([item.result.usage for item in items if item.result]),
        items=items,
        processing_seconds=round(time.perf_counter() - started, 2),
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
