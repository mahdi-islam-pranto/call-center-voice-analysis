from fastapi import FastAPI, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from fastapi import FastAPI, File, UploadFile
import base64
from typing import Literal, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(
    title="Voice transcription API",
    description="API to transcribe voice input into text.",
    version="1.0.0"
)

# Allow CORS (frontend to backend communication)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8666", "http://127.0.0.1:8666"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



SYSTEM_PROMPT = """
You are a professional transcription engine. Your only job is to convert spoken audio into written text with maximum accuracy.

Rules you must strictly follow:
- Make a transcript of user audio file. Do not change or make up the transcript by your own. Make the actual transcript from the audio.
- Do not add any commentary, explanation, or metadata. Output only the transcribed text.
- If there is any language specified, transcribe in that language only and put it in the appropriate language field. Put nothing in the other language fields.
- If no language is specified, transcribe in the original language of the audio, as default whether English or Bangla. The audio may contain a mix of both languages, so transcribe as it is without changing or making up any part of the transcript and put in in default field and put nothing in the other language fields.
- If language is specified as "all", transcribe in both English and Bangla and put in the both language fields appropriately.
"""

HUMAN_PROMPT = """
Please transcribe the following audio file accurately based on the specified language: {language}.
"""

class TranscriptionLanguage(BaseModel):
    english: str = Field(description="Transcripted text from the audio in English")
    bangla: str = Field(description="Transcripted text from the audio in Bangla")
    default: str = Field(description="Transcripted text from the audio in its original language, whether English or Bangla")

# pydantic class for Output
class TranscriptionOutput(BaseModel):
    ai_response: TranscriptionLanguage = Field(description="The transcribed text in English, Bangla, and default language")
    tokens_used: int = Field(description="Number of tokens used in the transcription process")


SUPPORTED_MIME_TYPES = {
    "audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav",
    "audio/ogg", "audio/webm", "audio/mp4", "audio/flac",
    "audio/aac", "audio/x-m4a",
}

# API endpoints here
@app.post("/transcribe", response_model=TranscriptionOutput)
async def track_user_progress(
    audio_file: UploadFile = File(..., description="The audio file to be transcribed"),
    language: Optional[Literal["english", "bangla", "all"]] = Form(
        None, 
        description="The language of the audio file"
    )):
    try:
        
        # Read and encode the audio file
        audio_bytes = await audio_file.read()
        audio_b64 = base64.standard_b64encode(audio_bytes).decode("utf-8")

        # Determine MIME type
        mime_type = audio_file.content_type or "audio/mpeg"
        if mime_type not in SUPPORTED_MIME_TYPES:
            mime_type = "audio/mpeg"  # Fallback default

        # Build the multimodal message manually (audio + text)
        # LangChain passes inline_data dicts straight through to Gemini
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=[
                {
                    "type": "media",
                    "data": audio_b64,
                    "mime_type": mime_type,
                },
                {
                    "type": "text",
                    "text": HUMAN_PROMPT.format(language=language),
                },
            ]),
        ]

        # Use with_structured_output to get pydantic model + raw response
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0,
            model_kwargs={
                "generation_config": {
                "thinking_config": {
                "thinking_budget": 0
                    }
                }
            }
        )

        # with_structured_output(include_raw=True) returns:
        # { "raw": AIMessage, "parsed": TranscriptionOutput, "parsing_error": ... }
        llm_structured = llm.with_structured_output(TranscriptionOutput, include_raw=True)

        result = llm_structured.invoke(messages)

        parsed: TranscriptionOutput = result["parsed"]
        raw_message = result["raw"]

        # Extract token usage from the raw AIMessage metadata
        usage_metadata = getattr(raw_message, "usage_metadata", {}) or {}
        tokens_used = usage_metadata.get("total_tokens", 0)

        return TranscriptionOutput(
            ai_response=parsed.ai_response,
            tokens_used=tokens_used,
        )
        
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))