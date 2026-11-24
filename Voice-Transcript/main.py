"""
Voice Transcript Generator

POST /transcribe with an audio URL -> the audio is downloaded, sent to an
LLM through LangChain (Gemini 3.1 Flash-Lite by default), and a
speaker-labelled transcript is returned.

The model is chosen with TRANSCRIPT_MODEL in .env ("provider:model"), so it
can be switched without touching the code.
"""
import asyncio
import base64
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Optional
from urllib.parse import unquote, urlparse

import httpx
import uvicorn
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
    speaker: str = Field(description='Speaker label, e.g. "Agent", "Customer", "Speaker 1".')
    start: str = Field(description="When the segment starts, MM:SS from the start of the recording.")
    text: str = Field(description="Verbatim text of what was said.")


class TranscriptOutput(BaseModel):
    """The structured output the model is asked to fill."""

    language: str = Field(description='Main language(s) spoken, e.g. "Bangla, English".')
    segments: list[Segment]


class TranscriptResponse(BaseModel):
    audio_url: str
    file_name: str
    mime_type: str
    size_bytes: int
    model: str
    language: str
    segments: list[Segment]
    transcript: str = Field(description='Plain-text transcript, one "[MM:SS] Speaker: text" line per segment.')
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
    except httpx.ConnectTimeout as e:
        host = urlparse(url).hostname
        raise DownloadError(
            f"Could not connect to {host} within 10 seconds. The server running this API cannot reach "
            f"{host}: check its network, firewall / IP whitelist on the recording server, or VPN."
        ) from e
    except httpx.HTTPError as e:
        raise DownloadError(f"Could not download the audio: {e.__class__.__name__}: {e or 'no details'}") from e

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
    return DownloadedAudio(file_name=file_name, mime_type=mime_type, data=data)


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


async def transcribe_audio(audio: DownloadedAudio) -> TranscriptOutput:
    structured_llm = get_llm().with_structured_output(TranscriptOutput, include_raw=True)
    messages = build_messages(audio)

    error = None
    for _ in range(LLM_RETRIES + 1):
        result = await structured_llm.ainvoke(messages)
        # LangChain auto-closes unfinished JSON, so a reply cut off at the output
        # limit would otherwise come back as a shorter "valid" transcript.
        finish_reason = str(result["raw"].response_metadata.get("finish_reason", "")).upper()
        if finish_reason.endswith(TRUNCATED_FINISH_REASONS):
            raise RuntimeError(
                "The transcript was cut off because it reached the model's output limit. "
                "The recording is too long for one request."
            )
        if result["parsed"] is not None:
            return result["parsed"]
        error = result.get("parsing_error")
    raise RuntimeError(f"The model did not return a valid transcript ({error!r}).")


def format_transcript(segments: list[Segment]) -> str:
    return "\n".join(f"[{s.start}] {s.speaker}: {s.text}" for s in segments)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Voice Transcript Generator",
    description="Downloads an audio file from a URL and returns a speaker-labelled transcript generated by an LLM via LangChain.",
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
            result = await transcribe_audio(audio)
        except Exception as e:
            raise TranscriptionError(f"Transcription failed: {str(e)[:300]}") from e

    return TranscriptResponse(
        audio_url=audio_url,
        file_name=audio.file_name,
        mime_type=audio.mime_type,
        size_bytes=len(audio.data),
        model=LLM_MODEL,
        language=result.language,
        segments=result.segments,
        transcript=format_transcript(result.segments),
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
        items=items,
        processing_seconds=round(time.perf_counter() - started, 2),
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
