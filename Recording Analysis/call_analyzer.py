import logging
import os
import tempfile
import time
from typing import Optional

from google import genai
from google.genai import types as genai_types
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from config import get_settings
from prompts import build_prompt
from schemas import (
    CallAnalysis,
    CallAnalysisResponse,
    CallContext,
    OutputLanguage,
    TokenUsage,
)

logger = logging.getLogger(__name__)


class CallAnalyzerError(Exception):
    """Raised for any failure during call-recording analysis (bad audio, timeout, API error)."""


class CallAnalyzerService:
    """
    Wraps Gemini's Files API (for uploading the audio) + LangChain's
    ChatGoogleGenerativeAI with structured output (for the actual analysis).

    Note: this uses the current `google-genai` SDK (the `google-generativeai` package is
    deprecated and no longer receives updates/bugfixes, and its old discovery-based file
    upload flow does not work correctly with newer API key formats).

    Usage:
        service = CallAnalyzerService()
        result = service.analyze(audio_bytes, "call.mp3", "audio/mpeg", OutputLanguage.BANGLA)
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._settings = settings
        self._client = genai.Client(api_key=settings.google_api_key)

        self._llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=0.1,
        ).with_structured_output(CallAnalysis, include_raw=True)

    def analyze(
        self,
        audio_bytes: bytes,
        filename: str,
        mime_type: str,
        language: OutputLanguage,
        context: Optional[CallContext] = None,
    ) -> CallAnalysisResponse:
        max_bytes = self._settings.max_audio_size_mb * 1024 * 1024
        if len(audio_bytes) > max_bytes:
            raise CallAnalyzerError(
                f"Audio file exceeds the {self._settings.max_audio_size_mb}MB limit."
            )

        uploaded_file = self._upload_and_wait(audio_bytes, filename, mime_type)
        try:
            system_text, human_text = build_prompt(language, context)

            message = HumanMessage(
                content=[
                    {"type": "text", "text": human_text},
                    {
                        "type": "media",
                        "mime_type": uploaded_file.mime_type,
                        "file_uri": uploaded_file.uri,
                    },
                ]
            )

            # include_raw=True returns {"raw": AIMessage, "parsed": CallAnalysis | None,
            # "parsing_error": Exception | None} instead of just the parsed object, so we
            # can also read token usage off the raw AIMessage.
            result = self._llm.invoke([SystemMessage(content=system_text), message])

            if result.get("parsing_error") is not None or result.get("parsed") is None:
                raise CallAnalyzerError(
                    f"Model response could not be parsed into the expected schema: {result.get('parsing_error')}"
                )

            analysis: CallAnalysis = result["parsed"]
            token_usage = self._extract_token_usage(result["raw"])

            return CallAnalysisResponse(
                language=language,
                analysis=analysis,
                token_usage=token_usage,
            )
        finally:
            # Always clean up the uploaded file from Google's side - it's only
            # needed for the duration of this one request.
            self._safe_delete(uploaded_file.name)

    def _upload_and_wait(self, audio_bytes: bytes, filename: str, mime_type: str):
        suffix = os.path.splitext(filename)[1] or ""
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            uploaded = self._client.files.upload(
                file=tmp_path,
                config=genai_types.UploadFileConfig(
                    mime_type=mime_type, display_name=filename
                ),
            )
        finally:
            os.remove(tmp_path)

        deadline = time.monotonic() + self._settings.file_processing_timeout_sec
        while uploaded.state.name == "PROCESSING":
            if time.monotonic() > deadline:
                self._safe_delete(uploaded.name)
                raise CallAnalyzerError("Timed out waiting for Gemini to process the audio file.")
            time.sleep(self._settings.file_processing_poll_interval_sec)
            uploaded = self._client.files.get(name=uploaded.name)

        if uploaded.state.name != "ACTIVE":
            self._safe_delete(uploaded.name)
            raise CallAnalyzerError(
                f"Gemini could not process the audio file (state={uploaded.state.name})."
            )

        return uploaded

    @staticmethod
    def _extract_token_usage(raw_message) -> Optional[TokenUsage]:
        usage = getattr(raw_message, "usage_metadata", None)
        if not usage:
            return None

        input_details = usage.get("input_token_details") or {}

        return TokenUsage(
            input_tokens=usage.get("input_tokens", 0) or 0,
            output_tokens=usage.get("output_tokens", 0) or 0,
            total_tokens=usage.get("total_tokens", 0) or 0,
            audio_tokens=input_details.get("audio"),
        )

    def _safe_delete(self, file_name: str) -> None:
        try:
            self._client.files.delete(name=file_name)
        except Exception:
            logger.warning("Could not delete temporary Gemini file %s", file_name, exc_info=True)