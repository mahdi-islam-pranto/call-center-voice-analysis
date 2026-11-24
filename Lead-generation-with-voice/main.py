from langchain_google_genai import ChatGoogleGenerativeAI
from fastapi import FastAPI, HTTPException, Request
from starlette.datastructures import UploadFile as StarletteUploadFile
from typing import Literal, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
import asyncio
import base64
import mimetypes
import os
import re
import requests
import urllib3
from urllib.parse import urlparse
from dotenv import load_dotenv
load_dotenv()

# recordings are often served from internal IPs with self-signed certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# initialize the FastAPI app
app = FastAPI(
    title="Lead Finder API",
    description=(
        "Analyze one or many call recordings (audio URLs OR uploaded audio files) and find potential leads "
        "based on lead keywords and business context supplied by the user. If no keywords and no business "
        "context are given, the FedEx (Bangladesh) lead criteria are used. Each recording is analyzed "
        "separately by the AI."
    ),
    version="2.1.0"
)

# how many recordings are sent to Gemini at the same time (each one is still its own prompt).
# set MAX_CONCURRENT_ANALYSES=1 in .env to process strictly one after another.
MAX_CONCURRENT_ANALYSES = int(os.getenv("MAX_CONCURRENT_ANALYSES", "3"))
DOWNLOAD_TIMEOUT_SECONDS = int(os.getenv("DOWNLOAD_TIMEOUT_SECONDS", "120"))
MAX_FILES_PER_REQUEST = int(os.getenv("MAX_FILES_PER_REQUEST", "1000"))

# audio formats supported by Gemini
SUPPORTED_AUDIO_MIME_TYPES = {
    "audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav", "audio/aac", "audio/ogg",
    "audio/flac", "audio/aiff", "audio/x-aiff", "audio/webm", "audio/mp4", "audio/m4a", "audio/x-m4a",
}
DEFAULT_AUDIO_MIME_TYPE = "audio/mpeg"


# ---------------------------------------------------------------------------
# Default lead criteria: FedEx (Bangladesh call center).
# Used when the caller sends neither lead_keywords nor business_context.
# Keywords come from "Key Words.xlsx".
# ---------------------------------------------------------------------------
FEDEX_BUSINESS_CONTEXT = """
Company: FedEx (FedEx Express service in Bangladesh, handled through the local FedEx call center).
FedEx is a global courier, express transportation and logistics company. Customers in Bangladesh call the
FedEx call center in Bangla, English, or a mix of both.

Services FedEx offers (anything here that a customer newly wants is a business opportunity):
- International express shipping of parcels and packages from Bangladesh to destinations worldwide
  (USA, UK, Europe, Middle East, Asia, etc.) - e.g. FedEx International Priority (fast, time-definite) and
  FedEx International Economy (lower cost, longer transit).
- International document courier - urgent documents, contracts, legal/visa/academic papers, samples.
- Export shipping for businesses - especially garments / RMG (ready-made garments) exporters, buying houses,
  textile, leather, handicraft and other manufacturers sending samples or commercial shipments.
- Import shipping into Bangladesh, including import of commercial goods and samples.
- Heavy / bulk shipments and international air freight for larger or palletized cargo.
- Customs clearance support, export/import documentation (commercial invoice, proforma invoice, etc.).
- Door-to-door pickup and delivery, scheduled / regular pickups for businesses.
- Corporate / business accounts with credit facilities, negotiated or discounted rates for regular or high
  volume shippers, rate cards and rate sheets.
- E-commerce and small business shipping solutions, packaging support.

Who is a POTENTIAL LEAD:
A customer (individual or business) who shows NEW business intent towards FedEx services, for example:
- asks for shipping rates, prices, charges, a quotation, rate card, discount or special rate for a new shipment;
- wants to send something abroad or import something (parcel, documents, samples, goods);
- wants to open a new account, corporate/business account, or register/sign up;
- mentions regular, recurring, bulk or high-volume shipping needs;
- is an exporter/importer (e.g. garments/RMG) looking for a courier or logistics partner;
- compares FedEx with other couriers or asks for a cheaper option for an upcoming shipment;
- wants to discuss a contract, agreement or MOU, or asks for a sales person / follow-up.
Business customers, recurring needs, bulk volume, and customers asking for next steps are stronger leads
than a single small personal shipment inquiry.
""".strip()

FEDEX_KEYWORDS = [
    # English
    "open account", "new account", "start shipping", "want to ship", "corporate account", "business account",
    "regular shipment", "bulk shipment", "registration", "sign up", "rate", "price", "charge",
    "how much does it cost", "what is cost", "quotation", "quote", "proforma", "discount", "special rate",
    "offer", "rate card", "rate sheet", "compare rate", "cheaper option", "send abroad", "send to USA",
    "send to UK", "international shipment", "overseas", "export", "import", "garments export", "RMG shipment",
    "document courier", "urgent document", "contract", "agreement", "MOU",
    # Bangla
    "অ্যাকাউন্ট খুলতে চাই", "নতুন অ্যাকাউন্ট", "শিপিং শুরু করতে চাই", "পাঠাতে চাই", "কর্পোরেট অ্যাকাউন্ট",
    "ব্যবসায়িক অ্যাকাউন্ট", "নিয়মিত পার্সেল", "বাল্ক শিপমেন্ট", "রেজিস্ট্রেশন করতে চাই", "সাইন আপ",
    "রেট", "দাম", "চার্জ", "মূল্য", "কত টাকা লাগবে", "কত খরচ", "কোটেশন", "দরপত্র", "ডিসকাউন্ট", "বিশেষ রেট",
    "অফার আছে কি", "রেট কার্ড পাঠান", "রেট লিস্ট", "তুলনামূলক রেট", "সস্তা অপশন আছে কি",
    "বিদেশে পাঠাতে চাই", "আমেরিকায়", "UK তে", "আন্তর্জাতিক শিপমেন্ট", "বিদেশে ডেলিভারি", "এক্সপোর্ট",
    "ইমপোর্ট করতে চাই", "গার্মেন্টস পাঠাতে চাই", "পোশাক রপ্তানি", "ডকুমেন্ট পাঠাতে চাই", "জরুরি কাগজ",
    "চুক্তি করতে চাই", "এমওইউ", "এগ্রিমেন্ট",
]

FEDEX_NOT_A_LEAD_CRITERIA = """
Calls where the customer has NO new business intent, for example:
- tracking an existing shipment, asking about delivery status, delays, or where a parcel is;
- complaints about a past shipment, damaged/lost parcels, refunds, or agent/courier behaviour;
- customs duty / tax / clearance questions about an existing incoming import that is already shipped,
  unless they also want new shipping services;
- address change or delivery rescheduling for an existing shipment;
- job inquiries, wrong numbers, personal calls, spam, vendors trying to sell something to FedEx;
- silent, very short, or unclear calls.
Mentioning words like "price", "charge" or "import" only in the context of an existing shipment or a
complaint is NOT a lead signal.
""".strip()


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------
def build_system_prompt(business_context: Optional[str], keywords: list[str], not_a_lead_criteria: Optional[str]) -> str:
    """Build the system prompt from the lead criteria.
    Plain string concatenation is used (not str.format) so braces in user text are safe."""
    keyword_block = "\n".join(f"- {k}" for k in keywords) if keywords else "- (no keywords provided — rely on the business context)"
    context_block = business_context.strip() if business_context and business_context.strip() else (
        "No business description was given. Infer what the business sells from the keywords below and the "
        "conversation itself."
    )
    exclusion_block = not_a_lead_criteria.strip() if not_a_lead_criteria and not_a_lead_criteria.strip() else (
        "Calls where the customer only wants support for something they already have (an existing order, "
        "account, or service), is complaining, is tracking a status, is a wrong number, is a spam/sales call "
        "to the company, or is otherwise unrelated — with no new buying or business intent."
    )

    return (
        "You are an expert AI Lead Qualification Analyst. You listen to call center recordings between a company's "
        "agent and a customer and decide whether the customer is a potential business lead FOR THIS SPECIFIC COMPANY.\n\n"
        "=== ABOUT THE COMPANY / WHAT COUNTS AS A LEAD ===\n"
        + context_block + "\n\n"
        "=== LEAD SIGNAL KEYWORDS (provided by the company) ===\n"
        "These keywords describe the topics, needs and phrases that indicate lead potential for this company. "
        "They are GUIDANCE about intent, not a checklist:\n"
        + keyword_block + "\n\n"
        "=== NOT A LEAD ===\n"
        + exclusion_block + "\n\n"
        "=== HOW TO ANALYZE ===\n"
        "1. First understand the whole call: who called, why, what the customer actually wants, and how the call ended.\n"
        "2. Focus primarily on what the CUSTOMER says and wants. The agent's own words (scripts, offers, greetings) are "
        "not lead signals unless the customer responds to them with genuine interest.\n"
        "3. Match the customer's INTENT against the company's keywords and lead description. A keyword counts when the "
        "customer means the same thing, even if they use a synonym, a paraphrase, a different language (e.g. Bangla, "
        "English, or a mix), or describe the need without the exact word.\n"
        "4. A keyword appearing in the call does NOT automatically make it a lead. Example: a customer saying 'price' "
        "while complaining about an old bill is not buying intent. Judge the context.\n"
        "5. A call can be a lead even if no keyword is said word-for-word, if the customer clearly shows new buying or "
        "business intent that fits the company's description.\n"
        "6. More and stronger signals (asking for price/quote, wanting to start/sign up, volume or recurring needs, "
        "asking for next steps, sharing contact details, requesting a follow-up) mean higher confidence. Vague or "
        "one-off curiosity means lower confidence.\n"
        "7. If the audio is silent, unclear, a voicemail, or too short to judge, mark it 'Not a Lead' with low "
        "confidence and say why in the justification.\n\n"
        "=== LEAD QUALITY SCALE ===\n"
        "- Strong Lead: clear, specific buying/business intent with multiple signals (e.g. asks for rates AND wants to start).\n"
        "- Potential Lead: real interest in a relevant product/service, but not yet committed or details missing.\n"
        "- Weak Interest: some relevant mention or curiosity, but low intent or unclear.\n"
        "- Not a Lead: no new business intent, or falls under 'NOT A LEAD'.\n"
        "is_potential_lead must be true for 'Strong Lead' and 'Potential Lead', and false otherwise.\n\n"
        "=== OUTPUT RULES ===\n"
        "- keywords_matched: only items from the company's keyword list above (copy them exactly as written in the "
        "list) whose MEANING the customer expressed in a real lead context.\n"
        "- other_lead_signals: lead signals you noticed that are not in the keyword list.\n"
        "- Write call_summary, customer_intent, justification and business_requirements in English, even if the call "
        "is in another language.\n"
    )


def resolve_criteria(keywords: list[str], business_context: Optional[str],
                     not_a_lead_criteria: Optional[str]) -> tuple[str, Literal["default_fedex", "custom"]]:
    """Return (system_prompt, criteria_source). Falls back to FedEx when neither keywords
    nor business context are supplied."""
    has_context = bool(business_context and business_context.strip())
    if not keywords and not has_context:
        not_a_lead = not_a_lead_criteria if (not_a_lead_criteria and not_a_lead_criteria.strip()) else FEDEX_NOT_A_LEAD_CRITERIA
        return build_system_prompt(FEDEX_BUSINESS_CONTEXT, FEDEX_KEYWORDS, not_a_lead), "default_fedex"
    return build_system_prompt(business_context, keywords, not_a_lead_criteria), "custom"


USER_INSTRUCTION = (
    "Analyze the call recording above and decide whether the customer is a potential lead for this company, "
    "following the system instructions. Base your decision on your overall understanding of the call, using the "
    "keywords as guidance about intent rather than doing literal word matching."
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
LeadQuality = Literal["Strong Lead", "Potential Lead", "Weak Interest", "Not a Lead"]


# what the LLM returns for one call
class LeadAnalysis(BaseModel):
    call_summary: str = Field(description="2-3 sentence summary of what happened in the call")
    customer_intent: str = Field(description="What the customer actually wants, in one sentence")
    is_potential_lead: bool = Field(description="True only for 'Strong Lead' or 'Potential Lead'")
    lead_quality: LeadQuality = Field(description="Categorical assessment of the lead quality")
    confidence_score: int = Field(ge=0, le=100, description="Confidence in the lead decision, 0-100")
    keywords_matched: list[str] = Field(default_factory=list, description="Items from the company's keyword list whose meaning the customer expressed in a lead context")
    other_lead_signals: list[str] = Field(default_factory=list, description="Lead signals found that are not in the keyword list")
    business_requirements: list[str] = Field(default_factory=list, description="Concrete needs the customer mentioned (product, volume, destination, timeline, budget, etc.)")
    justification: str = Field(description="Short explanation of the decision based on the conversation")


# API result for one call
class CallLeadResult(BaseModel):
    index: int = Field(description="Position of this recording in the request")
    source: str = Field(description="The audio URL or uploaded file name")
    source_type: Literal["url", "file"]
    status: Literal["success", "error"]
    error: Optional[str] = None
    call_summary: Optional[str] = None
    customer_intent: Optional[str] = None
    is_potential_lead: bool = False
    lead_quality: LeadQuality = "Not a Lead"
    confidence_score: int = 0
    keywords_matched: list[str] = Field(default_factory=list)
    other_lead_signals: list[str] = Field(default_factory=list)
    business_requirements: list[str] = Field(default_factory=list)
    justification: Optional[str] = None
    tokens_used: int = 0


class BatchSummary(BaseModel):
    criteria_used: Literal["default_fedex", "custom"] = Field(description="Which lead criteria were applied")
    total_calls: int
    analyzed: int
    failed: int
    potential_leads: int
    strong_leads: int
    potential_lead_quality: int = Field(description="Number of calls rated 'Potential Lead'")
    weak_interest: int
    not_a_lead: int
    total_tokens_used: int


class BatchLeadResponse(BaseModel):
    summary: BatchSummary
    leads: list[CallLeadResult] = Field(description="Only the potential leads, best first (Strong Lead, then highest confidence)")
    results: list[CallLeadResult] = Field(description="Every call in the original input order")


class BatchUrlRequest(BaseModel):
    audio_urls: list[str] = Field(description="One or more call recording URLs")
    lead_keywords: Optional[list[str]] = Field(default=None, description="Keywords / phrases that indicate a lead (any language). Leave empty together with business_context to use FedEx defaults.")
    business_context: Optional[str] = Field(default=None, description="Describe your company and what kind of customer counts as a lead. Leave empty together with lead_keywords to use FedEx defaults.")
    not_a_lead_criteria: Optional[str] = Field(default=None, description="Describe calls that should NOT be treated as leads")


# legacy single-URL request (kept for the existing frontend)
class PotentialLeadRequest(BaseModel):
    audio_url: str = Field(description="URL of the call recording audio file to be analyzed for lead qualification")
    lead_keywords: Optional[list[str]] = Field(default=None, description="Optional. Defaults to FedEx keywords")
    business_context: Optional[str] = Field(default=None, description="Optional. Defaults to FedEx business context")
    not_a_lead_criteria: Optional[str] = Field(default=None)


# legacy single-URL response (kept for the existing frontend)
class PotentialLeadResponse(BaseModel):
    is_potential_lead: bool
    keywords_matched: Optional[list[str]]
    justification: str
    lead_quality: LeadQuality
    tokens_used: int


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
structured_llm = llm.with_structured_output(LeadAnalysis, include_raw=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def guess_mime_type(name: str, declared: Optional[str] = None) -> str:
    """Pick an audio mime type from the declared content type or the file extension."""
    declared = (declared or "").split(";")[0].strip().lower()
    if declared in SUPPORTED_AUDIO_MIME_TYPES:
        return "audio/mpeg" if declared == "audio/mp3" else declared
    guessed, _ = mimetypes.guess_type(name)
    if guessed and guessed.startswith("audio/"):
        return "audio/mpeg" if guessed == "audio/mp3" else guessed
    return DEFAULT_AUDIO_MIME_TYPE


def download_audio(url: str) -> tuple[bytes, str]:
    """Download an audio file into memory and return (bytes, mime_type).
    Nothing is written to disk, so parallel requests can't overwrite each other's files."""
    with requests.get(url, stream=True, verify=False, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
        response.raise_for_status()
        data = b"".join(chunk for chunk in response.iter_content(chunk_size=8192) if chunk)
        mime_type = guess_mime_type(urlparse(url).path, response.headers.get("Content-Type"))
    if not data:
        raise ValueError("Downloaded audio file is empty")
    return data, mime_type


def parse_list_field(values: Optional[list[str]]) -> list[str]:
    """Accept repeated fields and/or a single field with comma / semicolon / newline separated values."""
    items: list[str] = []
    for value in values or []:
        for part in re.split(r"[\n,;]+", value or ""):
            part = part.strip()
            if part:
                items.append(part)
    # remove duplicates, keep order
    return list(dict.fromkeys(items))


def form_text(form, name: str) -> Optional[str]:
    value = form.get(name)
    return value.strip() if isinstance(value, str) and value.strip() else None


# ---------------------------------------------------------------------------
# Core analysis — ONE recording per Gemini call
# ---------------------------------------------------------------------------
async def analyze_audio(audio_bytes: bytes, mime_type: str, system_prompt: str) -> tuple[LeadAnalysis, int]:
    audio_b64 = base64.standard_b64encode(audio_bytes).decode("utf-8")
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=[
            {"type": "media", "data": audio_b64, "mime_type": mime_type},
            {"type": "text", "text": USER_INSTRUCTION},
        ]),
    ]
    result = await structured_llm.ainvoke(messages)
    parsed: Optional[LeadAnalysis] = result.get("parsed")
    if parsed is None:
        raise ValueError(f"Model did not return a valid analysis: {result.get('parsing_error')}")

    usage_metadata = getattr(result["raw"], "usage_metadata", {}) or {}
    tokens_used = usage_metadata.get("total_tokens", 0)

    # keep the flag consistent with the quality label
    parsed.is_potential_lead = parsed.lead_quality in ("Strong Lead", "Potential Lead")
    return parsed, tokens_used


async def process_one(index: int, source: str, source_type: str, system_prompt: str,
                      semaphore: asyncio.Semaphore, file_bytes: Optional[bytes] = None,
                      file_mime: Optional[str] = None) -> CallLeadResult:
    """Download (if URL) and analyze a single recording. Errors are captured per call
    so one bad recording never fails the whole batch."""
    async with semaphore:
        try:
            if source_type == "url":
                audio_bytes, mime_type = await asyncio.to_thread(download_audio, source)
            else:
                if not file_bytes:
                    raise ValueError("Uploaded file is empty")
                audio_bytes, mime_type = file_bytes, file_mime or DEFAULT_AUDIO_MIME_TYPE

            analysis, tokens_used = await analyze_audio(audio_bytes, mime_type, system_prompt)
            print(f"[{index}] {source} -> {analysis.lead_quality} ({analysis.confidence_score}%)")
            return CallLeadResult(
                index=index, source=source, source_type=source_type, status="success",
                tokens_used=tokens_used, **analysis.model_dump(),
            )
        except Exception as e:
            print(f"[{index}] {source} -> ERROR: {e}")
            return CallLeadResult(index=index, source=source, source_type=source_type, status="error", error=str(e))


async def run_batch(audio_urls: list[str], uploads: list[tuple[str, bytes, str]], system_prompt: str,
                    criteria_used: Literal["default_fedex", "custom"]) -> BatchLeadResponse:
    semaphore = asyncio.Semaphore(max(1, MAX_CONCURRENT_ANALYSES))
    if audio_urls:
        tasks = [process_one(i, url, "url", system_prompt, semaphore) for i, url in enumerate(audio_urls)]
    else:
        tasks = [
            process_one(i, name, "file", system_prompt, semaphore, file_bytes=data, file_mime=mime)
            for i, (name, data, mime) in enumerate(uploads)
        ]
    results: list[CallLeadResult] = list(await asyncio.gather(*tasks))

    ok = [r for r in results if r.status == "success"]
    quality_rank = {"Strong Lead": 0, "Potential Lead": 1}
    leads = sorted(
        (r for r in ok if r.is_potential_lead),
        key=lambda r: (quality_rank.get(r.lead_quality, 9), -r.confidence_score),
    )
    summary = BatchSummary(
        criteria_used=criteria_used,
        total_calls=len(results),
        analyzed=len(ok),
        failed=len(results) - len(ok),
        potential_leads=len(leads),
        strong_leads=sum(r.lead_quality == "Strong Lead" for r in ok),
        potential_lead_quality=sum(r.lead_quality == "Potential Lead" for r in ok),
        weak_interest=sum(r.lead_quality == "Weak Interest" for r in ok),
        not_a_lead=sum(r.lead_quality == "Not a Lead" for r in ok),
        total_tokens_used=sum(r.tokens_used for r in results),
    )
    return BatchLeadResponse(summary=summary, leads=leads, results=results)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
# The multipart form is described manually so Swagger UI (/docs) shows a real multi-file picker.
# (FastAPI's auto-generated schema for an optional list of files uses `anyOf` + `contentMediaType`,
# which Swagger UI renders as plain text boxes, so files could not be uploaded from /docs.)
FIND_LEADS_FORM_SCHEMA = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "files": {
                            "type": "array",
                            "items": {"type": "string", "format": "binary"},
                            "description": "Audio files (mp3, wav, m4a, ogg, flac, aac...). Use EITHER files OR audio_urls.",
                        },
                        "audio_urls": {
                            "type": "string",
                            "description": "Recording URLs, one per line or comma separated. Use EITHER audio_urls OR files.",
                        },
                        "lead_keywords": {
                            "type": "string",
                            "description": "Lead keywords/phrases, comma or newline separated (any language). "
                                           "Leave empty together with business_context to use FedEx defaults.",
                        },
                        "business_context": {
                            "type": "string",
                            "description": "Describe your company and what kind of customer counts as a lead. "
                                           "Leave empty together with lead_keywords to use FedEx defaults.",
                        },
                        "not_a_lead_criteria": {
                            "type": "string",
                            "description": "Optional. Describe calls that should NOT count as leads.",
                        },
                    },
                },
                "encoding": {"files": {"contentType": "audio/*"}},
            }
        },
    }
}


@app.post("/find_leads", response_model=BatchLeadResponse, openapi_extra=FIND_LEADS_FORM_SCHEMA)
async def find_leads(request: Request):
    """Analyze many recordings at once — send EITHER audio files OR audio URLs (not both).
    Each recording is sent to the AI in its own separate prompt.
    If neither lead_keywords nor business_context is given, FedEx lead criteria are used."""
    form = await request.form(max_files=MAX_FILES_PER_REQUEST, max_fields=MAX_FILES_PER_REQUEST)

    # Swagger UI / browsers may send empty placeholder entries — ignore anything without a filename
    file_items = [f for f in form.getlist("files") if isinstance(f, StarletteUploadFile) and f.filename]
    urls = parse_list_field([v for v in form.getlist("audio_urls") if isinstance(v, str)])

    if file_items and urls:
        raise HTTPException(status_code=400, detail="Send either audio files or audio URLs, not both.")
    if not file_items and not urls:
        raise HTTPException(status_code=400, detail="Provide at least one audio file or audio URL.")

    uploads: list[tuple[str, bytes, str]] = []
    for f in file_items:
        data = await f.read()
        uploads.append((f.filename, data, guess_mime_type(f.filename, f.content_type)))

    keywords = parse_list_field([v for v in form.getlist("lead_keywords") if isinstance(v, str)])
    system_prompt, criteria_used = resolve_criteria(
        keywords, form_text(form, "business_context"), form_text(form, "not_a_lead_criteria")
    )
    return await run_batch(urls, uploads, system_prompt, criteria_used)


@app.post("/find_leads/urls", response_model=BatchLeadResponse)
async def find_leads_from_urls(request: BatchUrlRequest):
    """JSON version of /find_leads for URL-only batches (easier to call from other systems)."""
    urls = parse_list_field(request.audio_urls)
    if not urls:
        raise HTTPException(status_code=400, detail="Provide at least one audio URL.")
    system_prompt, criteria_used = resolve_criteria(
        parse_list_field(request.lead_keywords), request.business_context, request.not_a_lead_criteria
    )
    return await run_batch(urls, [], system_prompt, criteria_used)


@app.post("/find_lead", response_model=PotentialLeadResponse)
async def find_potential_lead(request: PotentialLeadRequest) -> PotentialLeadResponse:
    """Legacy single-URL endpoint (used by frontend.py). Uses FedEx criteria when no custom criteria are sent."""
    system_prompt, _ = resolve_criteria(
        parse_list_field(request.lead_keywords), request.business_context, request.not_a_lead_criteria
    )
    result = await process_one(0, request.audio_url, "url", system_prompt, asyncio.Semaphore(1))
    if result.status == "error":
        return PotentialLeadResponse(
            is_potential_lead=False, keywords_matched=[],
            justification=f"Error processing the audio: {result.error}",
            lead_quality="Not a Lead", tokens_used=0,
        )
    return PotentialLeadResponse(
        is_potential_lead=result.is_potential_lead,
        keywords_matched=result.keywords_matched,
        justification=result.justification or "",
        lead_quality=result.lead_quality,
        tokens_used=result.tokens_used,
    )


# test URLs
# (strong lead, price discussion)
# https://103.204.81.3/RECORDINGS/MP3/20260615-100751_FEDEX_TN_FEDEX_louis_8801862901607-all.mp3
# (not a lead, irrelevant query)
# https://103.204.81.3/RECORDINGS/MP3/20260615-091435_FEDEX_TN__louis_01712174109-all.mp3
# (not a lead, tracking delivery status)
# https://103.204.81.3/RECORDINGS/MP3/20260615-101255_FEDEX_TN_FEDEX_IMPORT_bijoy_8801923430572-all.mp3
