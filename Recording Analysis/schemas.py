from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class OutputLanguage(str, Enum):
    """Language the SUMMARY/analysis should be written in (not the call audio's language)."""

    BANGLA = "bn"
    ENGLISH = "en"


class CallDirection(str, Enum):
    OUTBOUND = "outbound"
    INBOUND = "inbound"


class CallContext(BaseModel):
    """Optional metadata from the CRM. Purely used to ground the analysis - never required."""

    agent_name: Optional[str] = None
    customer_name: Optional[str] = None
    call_direction: Optional[CallDirection] = None
    deal_or_lead_id: Optional[str] = None
    product_or_service: Optional[str] = None


class CallAnalysis(BaseModel):
    """
    Structured output the model must produce.
    This exact shape is what langchain's `with_structured_output` will force
    Gemini to return, and it's what you'd store against the call record in the CRM.
    """

    summary: str = Field(
        ...,
        description="A concise but complete narrative summary of the call, roughly 4-8 sentences.",
    )
    keywords: List[str] = Field(
        ...,
        description="8-15 important keywords/short phrases from the call (products, topics, objections, competitors, etc).",
    )
    customer_sentiment: str = Field(
        ..., description="One of: positive, neutral, negative, mixed."
    )
    call_outcome: str = Field(
        ...,
        description="Short label for how the call ended, e.g. 'Demo scheduled', 'Not interested', 'Needs follow-up', 'Deal closed', 'No answer/voicemail'.",
    )
    customer_pain_points: List[str] = Field(
        default_factory=list, description="Problems, needs, or goals the customer expressed."
    )
    objections_raised: List[str] = Field(
        default_factory=list, description="Objections, hesitations, or concerns raised by the customer, if any."
    )
    products_services_discussed: List[str] = Field(
        default_factory=list, description="Products, services, plans, or prices discussed on the call."
    )
    action_items: List[str] = Field(
        default_factory=list, description="Concrete action items or commitments made by either side."
    )
    follow_up_required: bool = Field(
        ..., description="Whether a follow-up call or action is needed."
    )
    follow_up_notes: Optional[str] = Field(
        None, description="When/what the follow-up should be about, if mentioned or clearly implied."
    )
    important_notes: Optional[str] = Field(
        None,
        description="Anything else genuinely important for the sales rep that doesn't fit the fields above. Leave null if nothing extra matters.",
    )


class TokenUsage(BaseModel):
    """Token accounting for this request, as reported by the model provider."""

    input_tokens: int = Field(..., description="Tokens consumed by the prompt + audio input.")
    output_tokens: int = Field(..., description="Tokens generated in the response.")
    total_tokens: int = Field(..., description="input_tokens + output_tokens.")
    audio_tokens: Optional[int] = Field(
        None, description="Portion of input_tokens attributable to the audio itself, if the provider reports it separately."
    )


class CallAnalysisResponse(BaseModel):
    language: OutputLanguage
    analysis: CallAnalysis
    token_usage: Optional[TokenUsage] = Field(
        None, description="Token usage for this request. Null if the provider didn't return usage data."
    )