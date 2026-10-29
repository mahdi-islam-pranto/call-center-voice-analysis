import asyncio
import time
import uuid
from typing import List

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

import config
from audio import AudioSource, build_upload_source, fetch_all_audio
from pipeline import EvaluationFailedError, run_batch
from schemas import BatchReport

app = FastAPI(
    title=f"{config.BRAND_NAME} Call QC API",
    description=(
        f"Evaluates {config.BRAND_NAME} call-centre recordings against the 10-parameter QA scorecard "
        "(Good 1 / Average 0.5 / Bad 0, out of 10). Two-pass Gemini analysis: a listening pass "
        "(transcript, events, agent claims, voice delivery) and a QA scoring pass with verified evidence."
    ),
    version="1.0.0",
)

_default_openapi = app.openapi


def _openapi_with_file_pickers():
    """Swagger UI renders List[UploadFile] items as text boxes; rewrite them as binary so /docs shows file pickers."""
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

# In-memory job store: fine for a single uvicorn worker. Jobs are dropped after JOB_TTL_SECONDS.
JOBS: dict[str, dict] = {}
JOB_TTL_SECONDS = 6 * 3600


class JobCreated(BaseModel):
    job_id: str
    total: int


async def _collect_inputs(
    files: List[UploadFile], agent_names: List[str], urls: List[str]
) -> tuple[list[AudioSource], dict[str, str]]:
    files = [f for f in files or [] if f.filename]
    urls = [u.strip() for u in urls or [] if u and u.strip()]
    if not files and not urls:
        raise HTTPException(status_code=400, detail="Upload at least one recording or provide a URL.")
    if len(files) + len(urls) > config.MAX_UPLOAD_FILES:
        raise HTTPException(status_code=400, detail=f"At most {config.MAX_UPLOAD_FILES} recordings per batch.")

    audios = []
    for i, upload in enumerate(files):
        name = agent_names[i] if i < len(agent_names) else None
        try:
            audios.append(build_upload_source(upload.filename, await upload.read(), name))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    downloaded, failed = await fetch_all_audio(urls)
    audios.extend(downloaded)
    if not audios:
        raise HTTPException(status_code=400, detail=f"No audio could be loaded: {failed}")
    return audios, failed


def _purge_old_jobs():
    cutoff = time.time() - JOB_TTL_SECONDS
    for job_id in [j for j, job in JOBS.items() if job["created_at"] < cutoff]:
        JOBS.pop(job_id, None)


@app.get("/")
async def root():
    return {
        "message": f"{config.BRAND_NAME} Call QC API is running.",
        "endpoints": {
            "config": "GET /qc/config",
            "evaluate (sync)": "POST /qc/evaluate",
            "evaluate (background job)": "POST /qc/jobs, then GET /qc/jobs/{job_id}",
        },
    }


@app.get("/qc/config")
async def qc_config():
    """Scorecard definition: parameters, checklists, rating points, N/A policy, bands, critical errors."""
    return config.public_config()


@app.post("/qc/evaluate", response_model=BatchReport)
async def evaluate(
    files: List[UploadFile] = File([], description="Call recordings (.mp3 / .wav / .m4a / .ogg)."),
    agent_names: List[str] = Form([], description="Agent name per uploaded file, same order. Blank = detect from the call."),
    urls: List[str] = Form([], description="Optional public recording URLs."),
    double_check: bool = Form(False, description="Score every call twice and flag disagreements (about 1.6x cost)."),
):
    """Evaluates the batch and waits for the full report. Prefer /qc/jobs for large batches."""
    audios, failed = await _collect_inputs(files, agent_names, urls)
    try:
        return await run_batch(audios, failed, double_check)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except EvaluationFailedError as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/qc/jobs", response_model=JobCreated)
async def create_job(
    files: List[UploadFile] = File([], description="Call recordings (.mp3 / .wav / .m4a / .ogg)."),
    agent_names: List[str] = Form([], description="Agent name per uploaded file, same order. Blank = detect from the call."),
    urls: List[str] = Form([], description="Optional public recording URLs."),
    double_check: bool = Form(False, description="Score every call twice and flag disagreements (about 1.6x cost)."),
):
    """Starts the evaluation in the background and returns a job id to poll."""
    _purge_old_jobs()
    audios, failed = await _collect_inputs(files, agent_names, urls)
    job_id = uuid.uuid4().hex
    job = {"status": "running", "total": len(audios), "done": 0, "last_file": None,
           "result": None, "error": None, "created_at": time.time()}
    JOBS[job_id] = job

    async def progress(done: int, total: int, file_name: str):
        job["done"], job["last_file"] = done, file_name

    async def run():
        try:
            report = await run_batch(audios, failed, double_check, progress)
            job["result"] = report.model_dump(mode="json")
            job["status"] = "completed"
        except Exception as e:
            job["error"] = str(e)
            job["status"] = "failed"

    job["task"] = asyncio.create_task(run())  # keep a reference so the task is not garbage-collected
    return JobCreated(job_id=job_id, total=len(audios))


@app.get("/qc/jobs/{job_id}")
async def get_job(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found (it may have expired or the server restarted).")
    return {k: v for k, v in job.items() if k != "task"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
