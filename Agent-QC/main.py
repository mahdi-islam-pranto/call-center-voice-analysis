from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from qc_chain import run_quality_check, AgentPerformanceReport


app = FastAPI(
    title="Agent Performance QC API (Manual Criteria)",
    description=(
        "Evaluate one or more call recordings for the same agent using Gemini's native "
        "audio understanding (tone, pacing, emotion) against user-supplied evaluation criteria."
    ),
    version="2.0.0",
)


class AgentPerformanceRequest(BaseModel):
    paths: List[str]  # publicly accessible audio URLs, one agent's calls
    performance_types: Optional[List[str]] = None  # user-supplied criteria; falls back to defaults if empty


class AgentPerformanceResponse(BaseModel):
    agent_performance: AgentPerformanceReport
    audio_analysis_bearer: str
    failed_urls: List[str]


@app.get("/")
async def root():
    return {"message": "Agent Performance QC API is running. POST /agent-performance to evaluate call recordings."}


@app.post("/agent-performance", response_model=AgentPerformanceResponse)
async def api(request: AgentPerformanceRequest):
    if not request.paths:
        raise HTTPException(status_code=400, detail="At least one audio URL must be provided in 'paths'.")

    try:
        report = await run_quality_check(
            urls=request.paths,
            performance_types=request.performance_types,
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quality check failed: {str(e)}")

    return {
        "agent_performance": report,
        "audio_analysis_bearer": "Gemini (native audio - no separate transcription step)",
        "failed_urls": report.failed_urls,
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)