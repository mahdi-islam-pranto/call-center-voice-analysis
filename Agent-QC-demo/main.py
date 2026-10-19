from typing import List, Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from qc_chain import (
    MAX_UPLOAD_FILES,
    AgentPerformanceReport,
    AudioSource,
    EvaluationFailedError,
    build_upload_source,
    run_quality_check,
)


app = FastAPI(
    title="Agent Performance QC API (Manual Criteria)",
    description=(
        "Evaluate one or more call recordings for the same agent using Gemini's native "
        "audio understanding (tone, pacing, emotion) against user-supplied evaluation criteria. "
        "Recordings can be passed as public URLs or uploaded as .mp3 / .wav files."
    ),
    version="2.1.0",
)


_default_openapi = app.openapi


def _openapi_with_file_pickers():
    """
    FastAPI describes List[UploadFile] items with OpenAPI 3.1 `contentMediaType`, which Swagger UI
    renders as a text box. Rewrite them as `format: binary` so /docs shows a file picker per item.
    """
    if app.openapi_schema is None:
        schema = _default_openapi()
        for body in schema.get("components", {}).get("schemas", {}).values():
            for prop in body.get("properties", {}).values():
                items = prop.get("items", {})
                if items.get("contentMediaType") == "application/octet-stream":
                    items.pop("contentMediaType")
                    items["format"] = "binary"
    return app.openapi_schema


app.openapi = _openapi_with_file_pickers


class AgentPerformanceRequest(BaseModel):
    paths: List[str]  # publicly accessible audio URLs, one agent's calls
    performance_types: Optional[List[str]] = None  # user-supplied criteria; falls back to defaults if empty


class AgentPerformanceResponse(BaseModel):
    agent_performance: AgentPerformanceReport
    audio_analysis_bearer: str
    failed_urls: List[str]


async def _evaluate(
    paths: List[str],
    performance_types: Optional[List[str]],
    uploads: Optional[List[AudioSource]] = None,
) -> dict:
    try:
        report = await run_quality_check(
            urls=paths,
            performance_types=performance_types,
            uploads=uploads,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except EvaluationFailedError as e:
        # The recordings were fine but the AI service failed (quota, billing, outage...).
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quality check failed: {str(e)}")

    return {
        "agent_performance": report,
        "audio_analysis_bearer": "Gemini (native audio - no separate transcription step)",
        "failed_urls": report.failed_urls,
    }


@app.get("/")
async def root():
    return {
        "message": (
            "Agent Performance QC API is running. POST /agent-performance with audio URLs, "
            "or POST /agent-performance/upload with .mp3 / .wav files."
        )
    }


@app.post("/agent-performance", response_model=AgentPerformanceResponse)
async def api(request: AgentPerformanceRequest):
    if not request.paths:
        raise HTTPException(status_code=400, detail="At least one audio URL must be provided in 'paths'.")

    return await _evaluate(request.paths, request.performance_types)


@app.post("/agent-performance/upload", response_model=AgentPerformanceResponse)
async def upload_api(
    files: List[UploadFile] = File(..., description="Call recordings (.mp3 / .wav) handled by the same agent."),
    performance_types: List[str] = Form(
        [],
        description="Evaluation criteria (one per item); defaults are used if empty.",
        examples=[["Empathy/Tone", "Active Listening"]],
    ),
    paths: List[str] = Form([], description="Optional public audio URLs to evaluate alongside the files.", examples=[[]]),
):
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(status_code=400, detail=f"At most {MAX_UPLOAD_FILES} files can be uploaded per request.")

    uploads = []
    for upload in files:
        data = await upload.read()
        try:
            uploads.append(build_upload_source(upload.filename, data))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    paths = [p.strip() for p in (paths or []) if p and p.strip()]
    performance_types = [t.strip() for t in (performance_types or []) if t and t.strip()] or None
    return await _evaluate(paths, performance_types, uploads)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
