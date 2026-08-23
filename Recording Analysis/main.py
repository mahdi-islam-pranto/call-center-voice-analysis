import logging
import os
import json
import mimetypes
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Header, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from call_analyzer import CallAnalyzerError, CallAnalyzerService
from schemas import CallAnalysisResponse, CallContext, CallDirection, OutputLanguage



app = FastAPI(title="Call Recording Analyzer", version="1.0.0")

# Built once at startup - reuses the same LangChain client/model config for every request.
_analyzer = CallAnalyzerService()



# Allow CORS (frontend to backend communication)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8666", "http://127.0.0.1:8666", "http://138.252.115.100"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.post("/crm/analyze-call", response_model=CallAnalysisResponse)
async def analyze_call(
    audio_file: UploadFile = File(..., description="The call recording (mp3/wav/ogg/m4a/etc)."),
    language: OutputLanguage = Form(..., description="Output language for the summary: 'bn' or 'en'."),
    agent_name: Optional[str] = Form(None),
    customer_name: Optional[str] = Form(None),
    call_direction: Optional[CallDirection] = Form(None),
    deal_or_lead_id: Optional[str] = Form(None),
    product_or_service: Optional[str] = Form(None),
):
    mime_type = audio_file.content_type or mimetypes.guess_type(audio_file.filename or "")[0]
    if not mime_type or not mime_type.startswith("audio"):
        raise HTTPException(status_code=400, detail="Uploaded file does not look like an audio file.")

    audio_bytes = await audio_file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

    context = CallContext(
        agent_name=agent_name,
        customer_name=customer_name,
        call_direction=call_direction,
        deal_or_lead_id=deal_or_lead_id,
        product_or_service=product_or_service,
    )

    try:
        result = _analyzer.analyze(
            audio_bytes=audio_bytes,
            filename=audio_file.filename or "call_recording",
            mime_type=mime_type,
            language=language,
            context=context,
        )
        
    except CallAnalyzerError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail="Unexpected error while analyzing the call.") from exc

    return result


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})
