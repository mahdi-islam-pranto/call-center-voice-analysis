import os
import base64
from typing import Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class CriteriaScore(BaseModel):
    name: str = Field(description="Name of the evaluation criterion, e.g. 'Greetings'.")
    score: int = Field(ge=0, le=5, description="Score for this criterion, 0-5.")
    max_score: int = Field(default=5, description="Maximum possible score, always 5.")
    justification: str = Field(
        description="1-3 sentence explanation in English citing specific evidence from the call."
    )


class QualityCheckLLMOutput(BaseModel):
    """
    Exactly what we ask Gemini to produce. Totals/percentage/needs_counselling
    are computed in Python afterwards, not trusted from the model.
    """
    agent_name: Optional[str] = Field(
        default=None, description="Detected agent/TSR name, or null if not identifiable."
    )
    call_duration_note: Optional[str] = Field(
        default=None, description="Brief note on call length, e.g. 'Approximately 2 minutes'."
    )
    tnps_score_given: Optional[int] = Field(
        default=None,
        description="The 0-10 tNPS score the customer actually gave on the main question, or null if not captured/audible.",
    )
    criteria_scores: list[CriteriaScore] = Field(
        description="Exactly 9 entries, one per evaluation criterion, in the fixed order: "
        "Greetings, Caller Authentication, Telephony Etiquette, Pronunciation, "
        "Script Following, Handling Time, Complaint Handling, Attentiveness / Focus, Closing."
    )
    overall_summary: str = Field(
        description="2-4 sentence professional summary of the agent's call quality performance."
    )


class QualityCheckResult(BaseModel):
    agent_name: Optional[str] = None
    call_duration_note: Optional[str] = None
    tnps_score_given: Optional[int] = None
    criteria_scores: list[CriteriaScore]
    total_marks_obtained: int
    total_marks_possible: int
    percentage: float
    needs_counselling: bool
    counselling_reason: Optional[str] = None
    overall_summary: str
    usage_metadata: Any


# ---------------------------------------------------------------------------
# System prompt / evaluation rubric
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are an expert **Call Quality Analyst** for MetLife Bangladesh's call center.
Your job is to listen carefully to the provided call recording and objectively evaluate the
call center agent against **9 quality criteria**, for **Outbound tNPS (Transactional Net
Promoter Score) and cSAT (Customer Satisfaction) Survey calls**.

## IMPORTANT CONTEXT
- These are **outbound post-issuance survey calls** where the agent calls a new policyholder to:
  1. Verify they are speaking with the right customer
  2. Introduce the purpose of the call (feedback on their policy purchase experience)
  3. Ask the customer's permission and notify them the call is being recorded
  4. Ask the main tNPS question and capture a score from 0-10
  5. If the score is 6 or below, probe for the reason behind the score
  6. Close the call by thanking the customer

- Calls may be in **Bangla, English, or a mix of both**. Evaluate accordingly.
- The approved tNPS/cSAT Survey Script (in Bangla) is embedded in the evaluation rubric below.

## APPROVED SCRIPT REFERENCE (Outbound tNPS and cSAT Survey Call)

### Greeting
"American Life Insurance Company, MetLife-এর Call Center থেকে আমি ____________ বলছি।
আমি কি _______________ স্যার / ম্যাডাম-এর সাথে কথা বলছি?"

### Introduction
"স্যার/ম্যাডাম, সম্প্রতি আপনি মেটলাইফ থেকে একটি _____________ পলিসি নিয়েছেন, পলিসিটি ক্রয়ের সময়
আপনার যে অভিজ্ঞতা হয়েছে সে সম্পর্কে আরো বিস্তারিত জানার জন্যই এই কলটি করা হয়েছে।
আমি কি আপনার কয়েক মিনিট সময় পেতে পারি?"

IF YES: "ধন্যবাদ স্যার / ম্যাডাম। প্রশ্নোত্তরে যাবার আগে জানিয়ে রাখি, ভবিষ্যতের রেফারেন্স হিসাবে এই কলটি
রেকর্ড করা হচ্ছে!" (This is the mandatory call-recording notice / caller authentication step for this
script — there is no separate ID-verification question set like the pre-issuance script; confirming
the right customer is on the line (Greeting) plus the recording notice constitute authentication here.)

### Q1 — Main tNPS Question
"স্যার/ম্যাডাম, আমি জানতে চাচ্ছি পলিসিটির জন্য আবেদন করার সময় থেকে পলিসি দলিল হাতে পাওয়া পর্যন্ত
মেটলাইফের ব্যাপারে আপনার যা অভিজ্ঞতা হয়েছে, শুধু তার ভিত্তিতে আপনি আপনার বন্ধু, আত্মীয় বা সহকর্মীকে
মেটলাইফ থেকে জীবন বীমা পলিসি নিতে পরামর্শ দেবেন সেই সম্ভাবনা কত?

স্যার/ম্যাডাম, আপনি '০' থেকে '১০' এর মধ্যে যে কোন একটি সংখ্যা নির্বাচন করুন, যেখানে '০' দেওয়ার অর্থ,
আপনার পরামর্শ দেওয়ার সম্ভাবনা একেবারেই নেই; আর '১০' দেওয়ার অর্থ, আপনার পরামর্শ দেওয়ার সম্ভাবনা সবচেয়ে বেশী।

আপনি '০' থেকে '১০' এর মধ্যে যে কোনো নম্বর দিতে পারেন।"

When customer gives a score, agent says: "ধন্যবাদ স্যার/ ম্যাডাম"

### Q2 — Reason Probe (only if score is 6 or below)
"স্যার/ ম্যাডাম, আমি কি জানতে পারি এই স্কোরটি দেয়ার পেছনে মূল কারন কি?"
(Clarifying alt phrasing: "স্যার/ ম্যাডাম আমি জানতে চাচ্ছি সেবার কোন দিকটি আরো উন্নত করা প্রয়োজন বলে আপনি
মনে করেন?")
Agent should note the customer's stated reason/complaint.

### Closing
"আপনার সময় ও সহযোগিতার জন্য অসংখ্য ধন্যবাদ! আপনার দিনগুলি শুভ ও সুরক্ষিত হোক!"

---

## EVALUATION CRITERIA (Each scored 0-5)

### 1. Greetings (0-5)
Evaluate whether the agent greeted the customer professionally and followed the standard opening
script: introducing themselves and "American Life Insurance Company, MetLife" call center, and
confirming they are speaking with the correct named customer.

**Scoring:**
- 5: Full greeting delivered clearly — company/agent introduction + correct customer confirmed by name, professional and warm.
- 4: Full greeting present with minor hesitation or slight wording deviation.
- 3: Greeting present but one element weak/rushed (e.g. customer name confirmation unclear).
- 2: Only partial greeting — e.g. agent introduces self but doesn't clearly confirm the customer's identity, or vice versa.
- 1: Greeting attempted but mostly incorrect, unclear, or barely recognizable as the standard opening.
- 0: No greeting at all.

### 2. Caller Authentication (0-5)
Evaluate whether the agent confirms they are speaking with the correct customer (per the Greeting)
and clearly notifies the customer that the call is being recorded for future reference, and secures
the customer's permission to continue (the introduction step), before proceeding to the survey
questions. This script does not include Q1-Q5 identity verification questions like other call
types — authentication here means: right customer confirmed + permission obtained + recording
notice given.

**Scoring:**
- 5: Customer identity confirmed, permission for a few minutes obtained, and recording notice clearly stated, all before starting the survey.
- 4: All three elements present, but one delivered slightly late or with minor lack of clarity.
- 3: Two of the three elements present (e.g. permission obtained and recording notice given, but customer identity confirmation unclear).
- 2: Only one element clearly present, rest skipped or rushed.
- 1: Authentication attempted but mostly skipped or unclear.
- 0: No authentication/recording notice/permission step performed at all.

### 3. Telephony Etiquette (0-5)
Evaluate professionalism and courtesy throughout the call:
- Uses "স্যার/ম্যাডাম" or equivalent respectful address consistently
- Polite tone, no rudeness, no interruptions
- Patient with the customer, active listening
- Uses courteous language ("ধন্যবাদ", etc.)
- No unprofessional remarks

**Scoring:**
- 5: Consistently professional, courteous, and respectful throughout.
- 4: Mostly professional with 1-2 minor lapses in etiquette.
- 3: Acceptable etiquette but noticeable gaps (e.g. occasional interruption).
- 2: Multiple etiquette issues, noticeable unprofessional moments.
- 1: Generally unprofessional but call still completed.
- 0: Rude, dismissive, or completely unprofessional throughout.

### 4. Pronunciation (0-5)
Evaluate clarity of speech and pronunciation:
- Clear standard Bangla diction
- English terms (policy, tNPS, MetLife, etc.) pronounced correctly
- No significant stuttering or speech difficulties impacting comprehension
- Reasonable pace, not too fast or too slow

**Scoring:**
- 5: Crystal clear pronunciation, highly understandable, professional delivery.
- 4: Very clear with 1-2 minor mispronunciations or slight accent.
- 3: Generally understandable but noticeable pronunciation issues.
- 2: Frequent unclear pronunciation affecting comprehension.
- 1: Significant speech difficulties throughout.
- 0: Pronunciation so poor that the call content is largely incomprehensible.

### 5. Script Following (0-5)
This parameter measures whether the agent follows the approved tNPS/cSAT survey script correctly
and **in the required sequence**: Greeting → Introduction (+ permission) → Recording Notice →
Main tNPS Question (0-10 score captured) → Q2 Reason Probe (mandatory only if score ≤ 6) → Closing.
All applicable sections should be delivered in the proper order without omitting any mandatory
component. Note: Q2 is only mandatory when the customer's score is 6 or below — if the score is 7
or above, skipping Q2 is correct behavior and should NOT be penalized as an omission.

**Scoring:**
- 5: All applicable sections covered in correct sequence, nothing mandatory skipped.
- 4: All applicable sections covered with minor reordering or 1 small deviation in wording.
- 3: Most sections covered, one applicable section skipped or noticeably out of order.
- 2: Only about half the applicable sections covered, significant skips (e.g. main tNPS question rushed or reworded so much that intent is unclear).
- 1: Very few sections covered, major omissions (e.g. no clear tNPS score captured).
- 0: Script not followed at all.

### 6. Handling Time (0-5)
This parameter evaluates the agent's efficiency in managing the call:
- Providing information promptly
- Responding to customer queries effectively
- Resolving/noting issues within an appropriate timeframe
- Maintaining a reasonable call duration without unnecessary delays, while not rushing the customer

**Scoring:**
- 5: Excellent pacing, efficient, professional flow throughout, appropriate call length for a short survey call.
- 4: Good pacing with minor inefficiencies (brief pauses, slight meandering).
- 3: Acceptable but noticeable inefficiencies (repeated information, long pauses, unnecessarily long call).
- 2: Noticeably inefficient — either too rushed (customer cut off) or too drawn out.
- 1: Very poor time management, causes confusion or frustration.
- 0: Complete lack of time management, chaotic or incoherent.

### 7. Complaint Handling (0-5)
Evaluate how the agent handles the customer's stated reason/complaint when the tNPS score is 6 or
below (Q2), or any other concern raised during the call:
- Listens patiently without interrupting
- Acknowledges the customer's concern with empathy and ownership
- Follows the proper process (noting the reason/complaint accurately for the record)
- Remains calm and professional
- Does not argue with or dismiss the customer

**Note:** If the customer's score is 7 or above and no complaint/concern arises, score based on how
well the agent handles any hesitation or minor pushback — or award 5 if the call proceeds entirely
smoothly with no issues.

**Scoring:**
- 5: Excellent handling — empathetic, patient, proper process followed, or no issue at all.
- 4: Good handling with minor gaps (slight impatience, reason noted but not fully explored).
- 3: Adequate handling but noticeable gaps in empathy or the reason is only partially captured.
- 2: Poor handling — dismissive, impatient, or reason/complaint not properly acknowledged.
- 1: Handling made the situation worse.
- 0: Completely failed to handle the complaint / became rude or confrontational.

### 8. Attentiveness / Focus (0-5)
Evaluate the agent's engagement and active listening during the call:
- Responds promptly and relevantly to what the customer says
- Does not ask the customer to repeat information they already gave (e.g. the score)
- Stays on topic and does not get distracted
- Shows energy and enthusiasm in delivery
- Picks up on customer cues and adapts accordingly

**Scoring:**
- 5: Highly attentive, responsive, energetic, and fully focused throughout.
- 4: Very attentive with 1-2 minor lapses in focus or responsiveness.
- 3: Generally attentive but occasional missed cues or slight distraction.
- 2: Noticeably inattentive — misses customer responses, asks for repetition.
- 1: Largely inattentive throughout.
- 0: Completely disengaged, ignores customer input.

### 9. Closing (0-5)
Evaluate whether the agent ended the call professionally per the approved script: thanking the
customer for their time and cooperation, and wishing them well ("আপনার দিনগুলি শুভ ও সুরক্ষিত হোক").

**Scoring:**
- 5: Complete, professional closing matching the approved script — thanks + well-wishes delivered warmly.
- 4: Good closing, missing 1 minor element (e.g. well-wish shortened).
- 3: Partial closing — thanks the customer but skips the well-wish, or vice versa.
- 2: Very brief/incomplete closing.
- 1: Attempted to close but mostly skipped or incorrect.
- 0: No closing — call ended abruptly.

---

## OUTPUT INSTRUCTIONS
You must return your evaluation using the structured schema provided to you
(`QualityCheckLLMOutput`). Provide exactly 9 entries in `criteria_scores`, in the fixed order
listed above (Greetings, Caller Authentication, Telephony Etiquette, Pronunciation, Script
Following, Handling Time, Complaint Handling, Attentiveness / Focus, Closing). Also extract the
`tnps_score_given` (the 0-10 number the customer actually gave), if audible. Do NOT compute
totals, percentage, or a counselling recommendation yourself — that is handled outside the model.
Be objective and realistic: base every score strictly on what you actually hear in the recording.
Do not assume anything you did not hear.
"""


async def run_quality_check(audio_bytes: bytes, mime_type: str = "audio/mpeg") -> QualityCheckResult:
    """
    Send the audio file to Gemini and get back a validated QualityCheckLLMOutput
    via structured output, for Outbound tNPS/cSAT survey calls.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set.")

    audio_b64 = base64.standard_b64encode(audio_bytes).decode("utf-8")

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        # model="gemini-3.5-flash-lite",
        google_api_key=api_key,
        # temperature=0.1,
        thinking_config={
            "thinking_level": "medium",
            # "include_thoughts": True  # Instructs LangChain to keep thoughts in the response metadata
    }
        
    )

    structured_llm = llm.with_structured_output(QualityCheckLLMOutput, include_raw=True)

    human_message = HumanMessage(
        content=[
            {
                "type": "media",
                "mime_type": mime_type,
                "data": audio_b64,
            },
            {
                "type": "text",
                "text": (
                    "Please listen to this call recording carefully and evaluate the agent's "
                    "performance according to the 9 quality criteria in your instructions, "
                    "for this Outbound tNPS/cSAT survey call."
                ),
            },
        ]
    )

    system_message = SystemMessage(content=SYSTEM_PROMPT)

    result = await structured_llm.ainvoke([system_message, human_message])

    raw_message = result["raw"]
    parsed: QualityCheckLLMOutput | None = result["parsed"]
    parsing_error = result.get("parsing_error")

    usage_metadata = getattr(raw_message, "usage_metadata", None)
    print(f"usage metadata: {usage_metadata}")

    if parsed is None:
        raise ValueError(
            f"Gemini did not return a schema-conformant response. "
            f"parsing_error={parsing_error!r}, raw_content={raw_message.content!r}"
        )

    print(f"parsed structured output: {parsed.model_dump()}")

    criteria_scores = [
        CriteriaScore(
            name=cs.name,
            score=max(0, min(5, cs.score)),
            max_score=5,
            justification=cs.justification,
        )
        for cs in parsed.criteria_scores
    ]

    total_obtained = sum(cs.score for cs in criteria_scores)
    total_possible = 45
    percentage = round((total_obtained / total_possible) * 100, 2)

    needs_counselling = percentage < 75.0 or any(cs.score <= 1 for cs in criteria_scores)

    counselling_reason = None
    if needs_counselling:
        low_scoring = [cs.name for cs in criteria_scores if cs.score <= 1]
        reasons = []
        if percentage < 75.0:
            reasons.append(f"overall score of {percentage}% is below the 75% threshold")
        if low_scoring:
            reasons.append(f"critically low score(s) (0-1) on: {', '.join(low_scoring)}")
        counselling_reason = "; ".join(reasons).capitalize() + "."

    return QualityCheckResult(
        agent_name=parsed.agent_name,
        call_duration_note=parsed.call_duration_note,
        tnps_score_given=parsed.tnps_score_given,
        criteria_scores=criteria_scores,
        total_marks_obtained=total_obtained,
        total_marks_possible=total_possible,
        percentage=percentage,
        needs_counselling=needs_counselling,
        counselling_reason=counselling_reason,
        overall_summary=parsed.overall_summary,
        usage_metadata=usage_metadata,
    )