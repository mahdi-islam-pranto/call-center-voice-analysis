import os
import base64
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
from typing import Any
from qc_chain import run_quality_check

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

# pydantic output result class
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


@app.get("/")
async def root():
    return {"message": "Agent Quality Check API is running. POST /analyze to evaluate a call recording."}


@app.post("/analyze", response_model=QualityCheckResult)
async def analyze_call(
    audio: UploadFile = File(..., description="Call recording audio file (mp3, wav, ogg, etc.)"),
):
    """
    Analyze a call center agent's call recording and return quality scores.

    - **audio**: The call recording file (MP3, WAV, OGG, M4A, FLAC, AAC, or WEBM).

    Returns scores (0-5) for each of the 9 quality criteria, total marks, percentage,
    and whether the agent requires counselling.
    """
    allowed_types = {
        "audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav",
        "audio/ogg", "audio/mp4", "audio/m4a", "audio/x-m4a",
        "audio/flac", "audio/aac", "audio/webm",
    }
    if audio.content_type and audio.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {audio.content_type}. Please upload an audio file.",
        )

    audio_bytes = await audio.read()
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

    # Determine MIME type for Gemini
    mime_type = audio.content_type or "audio/mpeg"
    # Normalize common aliases
    if mime_type in ("audio/mp3", "audio/mpeg", "audio/mpeg3"):
        mime_type = "audio/mpeg"

    try:
        result = await run_quality_check(audio_bytes=audio_bytes, mime_type=mime_type)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quality check failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)