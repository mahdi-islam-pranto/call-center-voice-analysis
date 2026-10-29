"""
Pydantic models.

LLM output models (what Gemini is constrained to return) are kept separate from
the result models the API returns: totals, bands, review flags and evidence
verification are computed in code, never trusted from the model.

Field order matters - Gemini generates fields in schema order, so evidence and
checklists come BEFORE the rating they justify.
"""
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

Speaker = Literal["Agent", "Customer", "Other"]
Rating = Literal["Good", "Average", "Bad", "N/A"]
Sentiment = Literal["positive", "neutral", "confused", "anxious", "frustrated", "angry"]

CallCategory = Literal[
    "new_flight_booking",
    "flight_change_reissue",
    "cancellation_refund",
    "refund_status_followup",
    "payment_issue",
    "ticket_or_voucher_not_received",
    "hotel",
    "holiday_package",
    "visa",
    "complaint",
    "general_enquiry",
    "outbound_sales_followup",
    "outbound_payment_or_reminder",
    "other",
]

EventType = Literal[
    "hold",
    "dead_air",
    "agent_interrupts_customer",
    "colleague_consultation",
    "transfer",
    "customer_asks_for_supervisor",
    "escalation_by_agent",
    "customer_frustration",
    "empathy_or_apology",
    "customer_repeats_information",
    "agent_asks_customer_to_repeat",
    "promise_or_commitment",
    "sensitive_data_request",
    "callback_promised",
    "call_disconnected",
]


# ---------------------------------------------------------------------------
# Pass 1 - listening / call forensics (observation only, no judgement)
# ---------------------------------------------------------------------------


class TranscriptTurn(BaseModel):
    t: str = Field(description="Start time of this turn from the start of the recording, MM:SS.")
    speaker: Speaker = Field(description="Agent, Customer, or Other (IVR, colleague, supervisor, hold message).")
    text: str = Field(
        description="Verbatim words in the language actually spoken (Bangla in Bangla script, English words in Latin "
        "script). Mark unclear audio as [inaudible] and non-speech as [hold music], [silence 12s], [typing]."
    )


class CallEvent(BaseModel):
    t: str = Field(description="When it happened, MM:SS.")
    type: EventType
    duration_seconds: Optional[int] = Field(default=None, description="For hold / dead_air: how long it lasted.")
    detail: str = Field(description="One short factual sentence describing what happened.")


class AgentClaim(BaseModel):
    t: str = Field(description="When the agent said it, MM:SS.")
    topic: Literal[
        "fare_or_price", "fees_or_penalty", "refund", "date_change_reissue", "cancellation", "baggage",
        "visa", "hotel", "package", "payment", "timeline", "booking_status", "process", "other",
    ]
    claim: str = Field(description="The factual statement the agent made, paraphrased precisely in English with any numbers kept exact.")


class VoiceDelivery(BaseModel):
    tone_opening: str = Field(description="Agent's tone in the first minute, e.g. 'warm and energetic', 'flat', 'rushed'.")
    tone_middle: str = Field(description="Agent's tone during the main part of the call.")
    tone_closing: str = Field(description="Agent's tone at the end of the call.")
    pace: Literal["too slow", "appropriate", "too fast", "uneven"]
    clarity: str = Field(description="How clearly the agent speaks: volume, articulation, mispronounced terms (quote them).")
    accent_and_register: str = Field(
        description="Standard Bangla or regional dialect, consistency across the call, respectful register (আপনি / স্যার / ম্যাডাম)."
    )
    filler_words: Literal["none", "few", "frequent"]
    filler_examples: list[str] = Field(default_factory=list, description="Examples of the agent's filler words, if any.")


class CallForensics(BaseModel):
    agent_name: Optional[str] = Field(default=None, description="Agent's name if they say it, else null.")
    primary_language: str = Field(description="e.g. 'Bangla with English travel terms'.")
    call_direction: Literal["inbound", "outbound", "unclear"]
    call_category: CallCategory
    customer_intent: str = Field(description="1-2 sentences: what the customer (or the outbound call) needed, in English.")
    secondary_requests: list[str] = Field(default_factory=list, description="Any other questions or requests raised.")
    audio_quality: Literal["clear", "acceptable", "poor"]
    audio_quality_note: Optional[str] = Field(default=None, description="If not clear: what makes it hard to hear.")
    recording_complete: bool = Field(description="False if the recording starts mid-call or is cut off before the call ends.")
    transcript: list[TranscriptTurn] = Field(description="The COMPLETE call, turn by turn, from start to end.")
    events: list[CallEvent] = Field(default_factory=list, description="Every notable event, in time order.")
    agent_claims: list[AgentClaim] = Field(
        default_factory=list, description="Every factual statement the agent made about prices, fees, rules, timelines or processes."
    )
    delivery: VoiceDelivery
    customer_sentiment_start: Sentiment
    customer_sentiment_end: Sentiment
    resolution_status: Literal[
        "resolved", "actioned_pending_third_party", "partially_resolved", "unresolved", "escalated", "no_request"
    ]
    resolution_note: str = Field(description="One sentence: what the outcome of the call was.")


# ---------------------------------------------------------------------------
# Pass 2 - QA scoring
# ---------------------------------------------------------------------------


class ClaimCheck(BaseModel):
    claim_index: int = Field(description="0-based index into agent_claims.")
    verdict: Literal["consistent", "contradicts_knowledge_base", "self_contradiction", "not_covered"]
    note: str = Field(description="Which knowledge-base line it matches / contradicts, or why it cannot be checked.")


class EvidenceQuote(BaseModel):
    t: str = Field(description="Timestamp MM:SS of the transcript turn being quoted.")
    speaker: Speaker
    quote: str = Field(description="Words copied EXACTLY from that transcript turn (a short excerpt is fine).")


class ChecklistResult(BaseModel):
    item_id: str = Field(description="Checklist item id, e.g. 'GR1'.")
    status: Literal["Met", "Partial", "Not met", "N/A"]
    note: str = Field(description="Short reason with a timestamp, or what was absent.")


class ParameterEvaluation(BaseModel):
    parameter_key: str = Field(description="The parameter key exactly as given in the rubric, e.g. 'greeting'.")
    evidence: list[EvidenceQuote] = Field(
        description="1-4 verbatim quotes from the transcript that the rating is based on. Empty only when the "
        "rating is based on something being ABSENT (then say so in the justification)."
    )
    checklist: list[ChecklistResult] = Field(description="One entry per checklist item of this parameter, in order.")
    rating: Rating
    justification: str = Field(description="2-3 sentences: why the evidence and checklist lead to this rating.")
    coaching_tip: Optional[str] = Field(
        default=None,
        description="If not Good: the exact better phrase or action the agent should use next time (Bangla is fine). Null if Good.",
    )
    confidence: Literal["high", "medium", "low"]


class CriticalErrorFinding(BaseModel):
    error_key: str = Field(description="One of the critical error keys from the rubric.")
    t: str = Field(description="MM:SS")
    evidence: str = Field(description="What exactly was said or done.")


class ScoringLLMOutput(BaseModel):
    claim_checks: list[ClaimCheck] = Field(default_factory=list, description="One entry per agent claim, same order.")
    parameters: list[ParameterEvaluation] = Field(description="Exactly one entry per QA parameter, in rubric order.")
    critical_errors: list[CriticalErrorFinding] = Field(default_factory=list, description="Empty if none occurred.")
    call_summary: str = Field(description="3-4 sentences a QA lead can read instead of listening: what happened and how the agent did.")
    what_went_well: list[str] = Field(description="2-4 specific strengths with timestamps.")
    improvement_areas: list[str] = Field(description="2-4 specific, actionable improvements, most important first.")
    reviewer_attention: Optional[str] = Field(
        default=None, description="Anything a human QA should double-check (ambiguous moment, poor audio). Null if nothing."
    )


# ---------------------------------------------------------------------------
# Agent roll-up narrative (text-only, written from already-scored calls)
# ---------------------------------------------------------------------------


class AgentCoachingLLMOutput(BaseModel):
    performance_summary: str = Field(description="3-4 sentences on the agent's overall performance across these calls.")
    consistent_strengths: list[str] = Field(description="Strengths seen in more than one call, with file names.")
    recurring_issues: list[str] = Field(description="Problems seen in more than one call, with file names.")
    coaching_plan: list[str] = Field(
        description="3 prioritised coaching actions, each with the target behaviour and an example phrase or drill."
    )


# ---------------------------------------------------------------------------
# API result models
# ---------------------------------------------------------------------------


class VerifiedEvidence(EvidenceQuote):
    verified: bool = Field(description="True if the quote was found in the transcript.")


class ParameterResult(BaseModel):
    key: str
    name: str
    rating: Optional[Rating]
    points: Optional[float] = Field(description="Points this rating earns; None when N/A is excluded from the score.")
    justification: str
    coaching_tip: Optional[str] = None
    confidence: str
    checklist: list[ChecklistResult]
    evidence: list[VerifiedEvidence]
    second_opinion: Optional[Rating] = Field(default=None, description="Rating from the consistency pass, if run.")


class ReviewFlag(BaseModel):
    severity: Literal["high", "medium", "low"]
    reason: str


class CallResult(BaseModel):
    call_id: str
    file_name: str
    source: str
    agent_name: str
    agent_name_source: Literal["provided", "detected", "unknown"]
    duration_seconds: Optional[float] = None
    forensics: CallForensics
    claim_checks: list[ClaimCheck]
    parameters: list[ParameterResult]
    critical_errors: list[CriticalErrorFinding]
    score: float
    out_of: int
    percentage: float
    band: str
    passed: bool
    call_summary: str
    what_went_well: list[str]
    improvement_areas: list[str]
    needs_review: bool
    review_flags: list[ReviewFlag]
    usage: dict[str, int] = Field(default_factory=dict)


class ParameterAverage(BaseModel):
    key: str
    name: str
    average_points: Optional[float] = Field(description="0-1 average over applicable calls.")
    good: int
    average: int
    bad: int
    na: int


class AgentSummary(BaseModel):
    agent_name: str
    calls: int
    average_score: float
    pass_rate: float
    calls_needing_review: int
    critical_errors: int
    parameter_averages: list[ParameterAverage]
    coaching: Optional[AgentCoachingLLMOutput] = None


class BatchReport(BaseModel):
    generated_at: str
    brand_name: str
    calls: list[CallResult]
    agents: list[AgentSummary]
    team_average_score: Optional[float]
    team_pass_rate: Optional[float]
    failed: dict[str, str] = Field(default_factory=dict, description="file / URL -> reason it could not be evaluated.")
    models: dict[str, str]
    usage: dict[str, int] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict, description="Scoring scheme used, so the portal can re-score overrides.")
