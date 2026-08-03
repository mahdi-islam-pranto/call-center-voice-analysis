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
    This is exactly what we ask Gemini to produce. Totals/percentage/
    needs_counselling are deliberately NOT trusted from the model - we
    compute them ourselves afterwards for consistency and to avoid
    arithmetic mistakes creeping into the graded output.
    """
    agent_name: Optional[str] = Field(
        default=None, description="Detected agent/TSR name, or null if not identifiable."
    )
    call_duration_note: Optional[str] = Field(
        default=None, description="Brief note on call length, e.g. 'Approximately 4 minutes'."
    )
    criteria_scores: list[CriteriaScore] = Field(
        description="Exactly 9 entries, one per evaluation criterion, in the fixed order: "
        "Greetings, Caller Authentication, Telephony Etiquette, Pronunciation, "
        "Script Following, Handling Time, Complaint Handling, Attentiveness / Focus, Closing."
    )
    overall_summary: str = Field(
        description="2-4 sentence professional summary of the agent's call quality performance."
    )


# output schema
class QualityCheckResult(BaseModel):
    agent_name: Optional[str] = None
    call_duration_note: Optional[str] = None
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
call center agent (TSR - Telephone Sales Representative) against **9 quality criteria**.

## IMPORTANT CONTEXT
- These are **Pre-Issuance outbound calls** where the TSR calls a policyholder/applicant to:
  1. Verify their identity
  2. Confirm their policy financial details
  3. Inform them about key policy features (APL, Surrender Value, Agent of Record, etc.)
  4. Confirm the agent/FA who sold the policy
  5. Close professionally

- Calls may be in **Bangla, English, or a mix of both**. Evaluate accordingly.
- The approved Pre-Issuance Call Script (in Bangla) is embedded in the evaluation rubric below.

## APPROVED SCRIPT REFERENCE (Pre-Issuance Call)
### Opening / Greetings
"আসসালামু আলায়কুম / নমস্কার / আদাব। Good Morning / Good Afternoon / Good Evening  
মেটলাইফের Call Center থেকে আমি ________ বলছি।  
আমি কি ________ স্যারের / ম্যাডামের সাথে কথা বলছি?  
ধন্যবাদ, স্যার / ম্যাডাম! সম্প্রতি মেটলাইফ থেকে নতুন একটি বীমা পলিসি গ্রহনের জন্য আবেদন করেছেন। সে ব্যাপারে কথা বলার জন্য কল করেছি। কয়েক মিনিট সময় কি আমি এখন পেতে পারি?"

### Caller Authentication (ID Verification via YOB + one of Q2-Q5)
- Q1: Policy application জন্ম তারিখ confirm করা
- Q2: প্রিমিয়াম পরিমাণ ও পরিশোধের সময়সীমা
- Q3: Face Amount
- Q4: পলিসির মেয়াদ
- Q5: কার জন্য পলিসি
- Address verification

### Financial/Product Verification
Confirm: Product Name, Face Amount, Effective Date, Policy Term, Premium Amount, Premium Mode, Number of Premiums, Beneficiary Name, Next Premium Date, EFT confirmation.

### Agent of Record
Confirm the name of the Financial Associate/Agent.

### APL (Automatic Premium Loan)
Explain APL feature: if premium is not paid on time and cash value exists, policy continues. Overdue must be paid with charges. Long-term non-payment leads to policy lapse.

### Health Declarations
Confirm health info in application is accurate as claims depend on it.

### Agent Cash Collection
Inform that FA/agents cannot collect cash premiums after policy issuance per regulatory guidelines.

### Use & Clarity of Sales Material
Ask if the agent showed and explained the sales brochure clearly.

### Surrender Value
Inform that early surrender results in lower value than premiums paid.

### Video Link
Inform that a video link will be sent after the call.

### Closing
"আমি আবারও বলছি আমি মেটলাইফের Call Center থেকে ____________ আপনার সঙ্গে কথা বলছিলাম।"

---

## EVALUATION CRITERIA (Each scored 0-5)

### 1. Greetings (0-5)
Evaluate whether the TSR opens the call with the standard greeting:
- Gives an Islamic/secular greeting (Assalamu Alaikum / Nomoshkar / Adab / Good Morning/Afternoon/Evening)
- Introduces themselves and MetLife call center clearly
- Asks to speak with the specific customer by name
- States the purpose of the call (new policy application)
- Asks if the customer has a few minutes

**Scoring:**
- 5: All elements present, delivered clearly and professionally
- 4: All elements present, minor hesitation or slight wording deviation
- 3: Most elements present (3-4 out of 5), small omissions
- 2: Only 2 elements present or greeting is unclear
- 1: Attempted greeting but mostly incorrect or incomplete
- 0: No greeting at all

### 2. Caller Authentication (0-5)
Evaluate whether the TSR properly verifies the caller's identity before proceeding:
- Mentions call is being recorded
- Verifies Date of Birth (Q1 - mandatory)
- Verifies at least ONE of Q2-Q5 (premium amount, face amount, policy term, or beneficiary)
- Confirms registered address

**Scoring:**
- 5: Recording notice + Q1 + at least 1 from Q2-Q5 + address confirmed, all clearly done
- 4: Recording notice + Q1 + 1 from Q2-Q5, address skipped OR minor issue
- 3: Q1 done + at least 1 from Q2-Q5, recording notice or address missing
- 2: Only Q1 verified, nothing else, OR only 1 question asked with gaps
- 1: Authentication attempted but mostly incorrect, unclear, or skipped critical parts
- 0: No authentication performed

### 3. Telephony Etiquette (0-5)
Evaluate professionalism and courtesy throughout the call:
- Uses "স্যার/ম্যাডাম" or equivalent respectful address consistently
- Polite tone, no rudeness, no interruptions
- Patient with the customer
- Uses courteous language ("ধন্যবাদ", "আপনাকে ধন্যবাদ", etc.)
- No unprofessional remarks

**Scoring:**
- 5: Consistently professional, courteous, and respectful throughout
- 4: Mostly professional with 1-2 minor lapses in etiquette
- 3: Acceptable etiquette but noticeable gaps (e.g., occasional interruption)
- 2: Multiple etiquette issues, noticeable unprofessional moments
- 1: Generally unprofessional but call still completed
- 0: Rude, dismissive, or completely unprofessional throughout

### 4. Pronunciation (0-5)
Evaluate clarity of speech and pronunciation:
- Clear standard Bangla diction
- English terms (policy, premium, EFT, APL, etc.) pronounced correctly
- No significant stuttering or speech difficulties impacting comprehension
- Reasonable pace, not too fast or too slow
- No excessive regional accent interference

**Scoring:**
- 5: Crystal clear pronunciation, highly understandable, professional delivery
- 4: Very clear with 1-2 minor mispronunciations or slight accent
- 3: Generally understandable but noticeable pronunciation issues
- 2: Frequent unclear pronunciation affecting comprehension
- 1: Significant speech difficulties throughout
- 0: Pronunciation so poor that the call content is largely incomprehensible

### 5. Script Following (0-5)
Evaluate compliance with the approved Pre-Issuance script in correct order:
Opening → Authentication → Financial Details → Agent of Record → APL → Health Declarations → Agent Cash Collection → Sales Material Clarity → Surrender Value → Video Link → Closing

**Scoring:**
- 5: All sections covered in correct sequence, nothing major skipped
- 4: All sections covered with minor reordering or 1 small omission
- 3: Most sections covered (7-8 of 11), some reordering or 2-3 omissions
- 2: Only about half the sections covered, significant skips
- 1: Very few sections covered (1-3), major omissions
- 0: Script not followed at all

### 6. Handling Time (0-5)
Evaluate efficiency and call management:
- Delivers information promptly without unnecessary pauses or filler words
- Does not rush or cut off the customer
- Manages customer questions efficiently
- Call duration is appropriate (not excessively long or short for the content)
- Smooth transitions between sections

**Scoring:**
- 5: Excellent pacing, efficient, professional flow throughout
- 4: Good pacing with minor inefficiencies (brief pauses, slight meandering)
- 3: Acceptable but noticeable inefficiencies (repeated information, long pauses)
- 2: Noticeably inefficient — either too rushed or too drawn out
- 1: Very poor time management, causes confusion or frustration
- 0: Complete lack of time management, chaotic or incoherent

### 7. Complaint Handling (0-5)
Evaluate how the TSR handles customer disagreements, concerns, or complaints (if any arise):
- Listens patiently without interrupting
- Acknowledges the customer's concern empathetically
- Offers relevant solution or escalation path
- Remains calm and professional under pressure
- Does not argue or dismiss the customer

**Note:** If no complaint/concern arises in the call, score based on how well the TSR 
handles any hesitations, confusion, or pushback — or award 5 if the call proceeds 
entirely smoothly with no issues.

**Scoring:**
- 5: Excellent handling — empathetic, patient, solution-oriented, or no issue at all
- 4: Good handling with minor gaps (slight impatience, solution slightly unclear)
- 3: Adequate handling but noticeable gaps in empathy or resolution quality
- 2: Poor handling — dismissive, impatient, or no real resolution
- 1: Handling made the situation worse
- 0: Completely failed to handle the complaint / became rude or confrontational

### 8. Attentiveness / Focus (0-5)
Evaluate the TSR's engagement and active listening during the call:
- Responds promptly and relevantly to what the customer says
- Does not ask the customer to repeat information they already gave
- Stays on topic and does not get distracted
- Shows energy and enthusiasm in delivery
- Picks up on customer cues and adapts accordingly

**Scoring:**
- 5: Highly attentive, responsive, energetic, and fully focused throughout
- 4: Very attentive with 1-2 minor lapses in focus or responsiveness
- 3: Generally attentive but occasional missed cues or slight distraction
- 2: Noticeably inattentive — misses customer responses, asks for repetition
- 1: Largely inattentive throughout
- 0: Completely disengaged, ignores customer input

### 9. Closing (0-5)
Evaluate whether the TSR closes the call according to the approved script:
- Re-states their name and MetLife call center
- Thanks the customer for their time
- Wishes the customer well
- Professional and warm sign-off

**Scoring:**
- 5: Complete, professional closing matching the approved script
- 4: Good closing, missing 1 minor element
- 3: Partial closing — states name but skips thanks or wish
- 2: Very brief/incomplete closing
- 1: Attempted to close but mostly skipped or incorrect
- 0: No closing — call ended abruptly

---

## OUTPUT INSTRUCTIONS
You must return your evaluation using the structured schema provided to you
(`QualityCheckLLMOutput`). Provide exactly 9 entries in `criteria_scores`, in
the fixed order listed above (Greetings, Caller Authentication, Telephony
Etiquette, Pronunciation, Script Following, Handling Time, Complaint
Handling, Attentiveness / Focus, Closing). Do NOT compute totals, percentage,
or a counselling recommendation yourself — that is handled outside the model.
Be objective and realistic: base every score strictly on what you actually
hear in the recording. Do not assume anything you did not hear.
"""


async def run_quality_check(audio_bytes: bytes, mime_type: str = "audio/mpeg") -> QualityCheckResult:
    """
    Send the audio file to Gemini and get back a validated QualityCheckLLMOutput
    via structured output.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set.")

    # Encode audio to base64 for the inline data part
    audio_b64 = base64.standard_b64encode(audio_bytes).decode("utf-8")

    # Build the LangChain model
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        google_api_key=api_key,
        temperature=0.1,
    )

    # Bind the Pydantic schema -> Gemini will be constrained to return this shape.
    # `include_raw=True` lets us keep access to the raw AIMessage (for usage_metadata)
    # alongside the parsed object.
    structured_llm = llm.with_structured_output(QualityCheckLLMOutput, include_raw=True)

    # define human/user prompt
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
                    "performance according to the 9 quality criteria in your instructions."
                ),
            },
        ]
    )

    # define system message
    system_message = SystemMessage(content=SYSTEM_PROMPT)

    # Invoke the model - returns {"raw": AIMessage, "parsed": QualityCheckLLMOutput | None, "parsing_error": ...}
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

    # ------------------------------------------------------------------
    # Compute totals / percentage / counselling flag ourselves - do not
    # trust the model to do arithmetic correctly.
    # ------------------------------------------------------------------
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

    # Counselling check: below 75% OR any score is 0 or 1
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
        criteria_scores=criteria_scores,
        total_marks_obtained=total_obtained,
        total_marks_possible=total_possible,
        percentage=percentage,
        needs_counselling=needs_counselling,
        counselling_reason=counselling_reason,
        overall_summary=parsed.overall_summary,
        usage_metadata=usage_metadata,
    )