import os
import base64
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
from typing import Any
from qc_chain import run_quality_check as run_preissuance_quality_check
from qc_chain_tnps import run_quality_check as run_tnps_quality_check

# define fastapi
app = FastAPI(
    title="Agent Quality Check API",
    description="Analyze call center agent recordings and score them across 9 quality criteria.",
    version="1.0.0",
)

# pydantic output scoring class
class CriteriaScore(BaseModel):
    name: str
    score: int
    max_score: int = 5
    justification: str

# pydantic output result class - Pre-Issuance calls
class QualityCheckResult(BaseModel):
    agent_name: Optional[str] = None
    call_duration_note: Optional[str] = None
    criteria_scores: list[CriteriaScore]
    total_marks_obtained: int
    total_marks_possible: int
    percentage: float
    needs_counselling: bool
    counselling_reason: Optional[str] = None
    overall_summary: str
    usage_metadata: Any


# pydantic output result class - Outbound tNPS/cSAT Survey calls
class TnpsQualityCheckResult(BaseModel):
    agent_name: Optional[str] = None
    call_duration_note: Optional[str] = None
    tnps_score_given: Optional[int] = None
    criteria_scores: list[CriteriaScore]
    total_marks_obtained: int
    total_marks_possible: int
    percentage: float
    needs_counselling: bool
    counselling_reason: Optional[str] = None
    overall_summary: str
    usage_metadata: Any


ALLOWED_AUDIO_TYPES = {
    "audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav",
    "audio/ogg", "audio/mp4", "audio/m4a", "audio/x-m4a",
    "audio/flac", "audio/aac", "audio/webm",
}


@app.get("/")
async def root():
    return {
        "message": "Agent Quality Check API is running.",
        "endpoints": {
            "/analyze": "Evaluate a Pre-Issuance call recording.",
            "/analyze-tnps": "Evaluate an Outbound tNPS/cSAT survey call recording.",
        },
    }


async def _read_and_validate_audio(audio: UploadFile) -> tuple[bytes, str]:
    if audio.content_type and audio.content_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {audio.content_type}. Please upload an audio file.",
        )

    audio_bytes = await audio.read()
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

    mime_type = audio.content_type or "audio/mpeg"
    if mime_type in ("audio/mp3", "audio/mpeg", "audio/mpeg3"):
        mime_type = "audio/mpeg"

    return audio_bytes, mime_type


@app.post("/analyze", response_model=QualityCheckResult)
async def analyze_call(
    audio: UploadFile = File(..., description="Call recording audio file (mp3, wav, ogg, etc.)"),
):
    """
    Analyze a Pre-Issuance call center agent's call recording and return quality scores.

    - **audio**: The call recording file (MP3, WAV, OGG, M4A, FLAC, AAC, or WEBM).

    Returns scores (0-5) for each of the 9 quality criteria, total marks, percentage,
    and whether the agent requires counselling.
    """
    audio_bytes, mime_type = await _read_and_validate_audio(audio)

    try:
        result = await run_preissuance_quality_check(audio_bytes=audio_bytes, mime_type=mime_type)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quality check failed: {str(e)}")


@app.post("/analyze-tnps", response_model=TnpsQualityCheckResult)
async def analyze_tnps_call(
    audio: UploadFile = File(..., description="Call recording audio file (mp3, wav, ogg, etc.)"),
):
    """
    Analyze an Outbound tNPS/cSAT Survey call center agent's call recording and return quality scores.

    - **audio**: The call recording file (MP3, WAV, OGG, M4A, FLAC, AAC, or WEBM).

    Returns scores (0-5) for each of the 9 quality criteria, total marks, percentage,
    the tNPS score the customer gave, and whether the agent requires counselling.
    """
    audio_bytes, mime_type = await _read_and_validate_audio(audio)

    try:
        result = await run_tnps_quality_check(audio_bytes=audio_bytes, mime_type=mime_type)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quality check failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)