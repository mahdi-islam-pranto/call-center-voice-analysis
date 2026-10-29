"""
Audio inputs: validating uploads, downloading URLs, and measuring duration.

The measured duration is used to catch hallucinated timestamps (evidence quoted
"at 07:30" in a 5-minute call) and to report handling time.
"""
import asyncio
import io
import mimetypes
import os
from dataclasses import dataclass
from typing import Optional
from urllib.parse import unquote, urlparse

import httpx

from config import MAX_UPLOAD_MB

try:
    import mutagen
except ImportError:  # duration is a nice-to-have, never a hard requirement
    mutagen = None

ALLOWED_EXTENSIONS = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
}


@dataclass
class AudioSource:
    file_name: str
    source: str  # original URL, or "upload"
    data: bytes
    mime_type: str
    agent_name: Optional[str] = None  # set by the uploader; overrides the detected name
    duration_seconds: Optional[float] = None


def _looks_like_audio(ext: str, data: bytes) -> bool:
    if ext == ".wav":
        return data[:4] == b"RIFF" and data[8:12] == b"WAVE"
    if ext == ".mp3":
        # ID3 tag, or a bare MPEG frame sync
        return data[:3] == b"ID3" or (len(data) > 1 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0)
    if ext == ".m4a":
        return data[4:8] == b"ftyp"
    if ext == ".ogg":
        return data[:4] == b"OggS"
    return False


def measure_duration(data: bytes) -> Optional[float]:
    if mutagen is None:
        return None
    try:
        audio = mutagen.File(io.BytesIO(data))
        return round(audio.info.length, 1) if audio is not None and audio.info else None
    except Exception:
        return None


def build_upload_source(file_name: str, data: bytes, agent_name: Optional[str] = None) -> AudioSource:
    """Validates an uploaded recording and wraps it as an AudioSource. Raises ValueError if invalid."""
    file_name = os.path.basename(file_name or "recording")
    ext = os.path.splitext(file_name)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"'{file_name}': only {', '.join(ALLOWED_EXTENSIONS)} files are supported.")
    if not data:
        raise ValueError(f"'{file_name}' is empty.")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"'{file_name}' is larger than the {MAX_UPLOAD_MB} MB limit.")
    if not _looks_like_audio(ext, data):
        raise ValueError(f"'{file_name}' does not look like a valid {ext[1:].upper()} file.")
    return AudioSource(
        file_name=file_name,
        source="upload",
        data=data,
        mime_type=ALLOWED_EXTENSIONS[ext],
        agent_name=(agent_name or "").strip() or None,
        duration_seconds=measure_duration(data),
    )


def _guess_mime_type(url: str, content_type_header: Optional[str]) -> str:
    if content_type_header and content_type_header.startswith("audio/"):
        return content_type_header.split(";")[0].strip()
    ext = os.path.splitext(urlparse(url).path)[1].lower()
    if ext in ALLOWED_EXTENSIONS:
        return ALLOWED_EXTENSIONS[ext]
    guessed, _ = mimetypes.guess_type(urlparse(url).path)
    return guessed if guessed and guessed.startswith("audio/") else "audio/mpeg"


async def _fetch(client: httpx.AsyncClient, url: str) -> tuple[str, Optional[AudioSource], Optional[str]]:
    try:
        resp = await client.get(url, timeout=60.0, follow_redirects=True)
        resp.raise_for_status()
        if len(resp.content) > MAX_UPLOAD_MB * 1024 * 1024:
            return url, None, f"Larger than the {MAX_UPLOAD_MB} MB limit."
        return url, AudioSource(
            file_name=unquote(os.path.basename(urlparse(url).path)) or url,
            source=url,
            data=resp.content,
            mime_type=_guess_mime_type(url, resp.headers.get("content-type")),
            duration_seconds=measure_duration(resp.content),
        ), None
    except Exception as e:
        return url, None, f"Download failed: {str(e)[:200]}"


async def fetch_all_audio(urls: list[str]) -> tuple[list[AudioSource], dict[str, str]]:
    """Downloads all URLs concurrently. Returns (sources, {url: failure reason})."""
    if not urls:
        return [], {}
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*(_fetch(client, url) for url in urls))
    return [a for _, a, _ in results if a is not None], {url: err for url, a, err in results if a is None}
