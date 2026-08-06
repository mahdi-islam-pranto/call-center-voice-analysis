import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, Any
import uvicorn
from qc_chain import run_quality_check
from qc_chain_inbound import run_inbound_quality_check

app = FastAPI(
    title="Agent Quality Check API",
    description="Analyze call center agent recordings (pre-issuance outbound and inbound) and "
    "score them against their respective quality criteria.",
    version="1.2.0",
)

ALLOWED_AUDIO_TYPES = {
    "audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav",
    "audio/ogg", "audio/mp4", "audio/m4a", "audio/x-m4a",
    "audio/flac", "audio/aac", "audio/webm",
}


def _validate_and_normalize(audio: UploadFile, audio_bytes: bytes) -> str:
    if audio.content_type and audio.content_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {audio.content_type}. Please upload an audio file.",
        )
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

    mime_type = audio.content_type or "audio/mpeg"
    if mime_type in ("audio/mp3", "audio/mpeg", "audio/mpeg3"):
        mime_type = "audio/mpeg"
    return mime_type


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class CriteriaScore(BaseModel):
    name: str
    score: int
    max_score: int = 5
    evidence: str
    justification: str


class PreIssuanceQualityCheckResult(BaseModel):
    agent_name: Optional[str] = None
    call_duration_note: Optional[str] = None
    call_language_note: Optional[str] = None
    criteria_scores: list[CriteriaScore]
    total_marks_obtained: int
    total_marks_possible: int
    percentage: float
    needs_counselling: bool
    counselling_reason: Optional[str] = None
    overall_summary: str
    low_confidence_flag: bool
    low_confidence_reason: Optional[str] = None
    score_variance_flag: bool = False
    score_variance_note: Optional[str] = None
    usage_metadata: Any


class InboundQualityCheckResult(BaseModel):
    agent_name: Optional[str] = None
    call_duration_note: Optional[str] = None
    call_language_note: Optional[str] = None
    customer_issue_summary: str
    criteria_scores: list[CriteriaScore]
    total_marks_obtained: int
    total_marks_possible: int
    percentage: float
    needs_counselling: bool
    counselling_reason: Optional[str] = None
    overall_summary: str
    low_confidence_flag: bool
    low_confidence_reason: Optional[str] = None
    score_variance_flag: bool = False
    score_variance_note: Optional[str] = None
    usage_metadata: Any


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {
        "message": "Agent Quality Check API is running.",
        "endpoints": {
            "pre_issuance_outbound": "POST /analyze",
            "inbound": "POST /analyze-inbound",
        },
    }


@app.post("/analyze", response_model=PreIssuanceQualityCheckResult)
async def analyze_call(
    audio: UploadFile = File(..., description="Pre-issuance outbound call recording (mp3, wav, ogg, etc.)"),
    double_check: bool = Query(
        default=False,
        description="Run the evaluation twice and flag any criterion where the two passes "
        "disagree by 2+ points. Roughly doubles API cost for this call.",
    ),
):
    """
    Analyze a **pre-issuance outbound** call recording against the approved script and the
    9 quality criteria (Greetings, Authentication, Etiquette, Pronunciation, Script Following,
    Handling Time, Complaint Handling, Attentiveness, Closing).
    """
    audio_bytes = await audio.read()
    mime_type = _validate_and_normalize(audio, audio_bytes)

    try:
        result = await run_quality_check(
            audio_bytes=audio_bytes, mime_type=mime_type, double_check=double_check
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quality check failed: {str(e)}")


@app.post("/analyze-inbound", response_model=InboundQualityCheckResult)
async def analyze_inbound_call(
    audio: UploadFile = File(..., description="Inbound call recording (mp3, wav, ogg, etc.)"),
    double_check: bool = Query(
        default=False,
        description="Run the evaluation twice and flag any criterion where the two passes "
        "disagree by 2+ points. Roughly doubles API cost for this call.",
    ),
):
    """
    Analyze an **inbound** call recording (no fixed script) against the 12 quality criteria
    (Greetings, Authentication, Etiquette, Pronunciation, Issue Identification, Information
    Accuracy, Issue Resolution, Handling Time, Complaint Handling, FCR, Attentiveness, Closing).
    Focused on how well the agent understood and resolved the customer's query, and how they
    behaved throughout the call.
    """
    audio_bytes = await audio.read()
    mime_type = _validate_and_normalize(audio, audio_bytes)

    try:
        result = await run_inbound_quality_check(
            audio_bytes=audio_bytes, mime_type=mime_type, double_check=double_check
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quality check failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)