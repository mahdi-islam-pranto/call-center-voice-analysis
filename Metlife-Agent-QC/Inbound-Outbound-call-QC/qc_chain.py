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
    evidence: str = Field(
        description=(
            "The specific thing the TSR said or did (or failed to say/do) that this score is "
            "based on, in your own words, with an approximate timestamp if you can tell "
            "(e.g. 'around 0:45'). This is what a human reviewer checks the score against, so "
            "be concrete, not generic."
        )
    )
    justification: str = Field(
        description="1-2 sentence explanation of why the evidence maps to this score under the rubric."
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
    call_language_note: Optional[str] = Field(
        default=None,
        description="Brief note on language mix, e.g. 'Primarily Bangla with English product terms'.",
    )
    criteria_scores: list[CriteriaScore] = Field(
        description="Exactly 9 entries, one per evaluation criterion, in the fixed order: "
        "Greetings, Caller Authentication, Telephony Etiquette, Pronunciation, "
        "Script Following, Handling Time, Complaint Handling, Attentiveness / Focus, Closing."
    )
    overall_summary: str = Field(
        description="2-4 sentence professional summary of the agent's call quality performance."
    )
    low_confidence_flag: bool = Field(
        default=False,
        description=(
            "Set true if audio quality, cross-talk, or language mix made it genuinely hard to "
            "judge one or more criteria confidently. This routes the call to a human reviewer "
            "instead of silently guessing."
        ),
    )
    low_confidence_reason: Optional[str] = Field(
        default=None, description="If low_confidence_flag is true, briefly say why."
    )


# output schema
class QualityCheckResult(BaseModel):
    agent_name: Optional[str] = None
    call_duration_note: Optional[str] = None
    call_language_note: Optional[str] = None
    criteria_scores: list[CriteriaScore]
    total_marks_obtained: int
    total_marks_possible: int
    percentage: float
    needs_counselling: bool
    counselling_reason: Optional[str] = None
    overall_summary: str
    low_confidence_flag: bool
    low_confidence_reason: Optional[str] = None
    score_variance_flag: bool = False
    score_variance_note: Optional[str] = None
    usage_metadata: Any


# ---------------------------------------------------------------------------
# System prompt / evaluation rubric
# ---------------------------------------------------------------------------
# Changes from the original version, aimed at closing the gap to a human QA
# reviewer:
#   1. Requires concrete evidence per criterion (not just a score + generic
#      sentence) so a human can audit/overrule any score quickly.
#   2. Adds the parts of your actual approved script that were missing from
#      the original rubric: the recording notice line, the signature/
#      claims-basis disclosure, the EFT negative-answer / mismatch escalation
#      branch, and the DPS Super Policy brochure variant.
#   3. Explicitly tells the model how to separate TSR performance from
#      customer behavior, so a difficult/quiet customer doesn't unfairly tank
#      "Attentiveness" or "Handling Time" scores.
#   4. Adds calibration anchors (concrete examples of what a 5 vs a 1 sounds
#      like) for the criteria that are most subjective - anchors are the
#      single biggest lever for making an LLM's grading consistent across
#      many different calls and agents.
#   5. Explicitly forbids clustering scores around a "safe" 3, which is a
#      well-known LLM-grading failure mode that makes output useless for
#      actually identifying agents who need counselling.
#   6. Adds a low_confidence escape hatch instead of forcing a guess when
#      audio quality or cross-talk makes something genuinely unclear.

SYSTEM_PROMPT = """
You are an expert **Call Quality Analyst** for MetLife Bangladesh's call center.
Your job is to listen carefully to the provided call recording and objectively evaluate the
call center agent (TSR - Telephone Sales Representative) against **9 quality criteria**,
the way a strict, fair, experienced human QA reviewer would.

## IMPORTANT CONTEXT
- These are **Pre-Issuance outbound calls** where the TSR calls a policyholder/applicant to:
  1. Verify their identity
  2. Confirm their policy financial details
  3. Inform them about key policy features (APL, Surrender Value, Agent of Record, etc.)
  4. Confirm the agent/FA who sold the policy
  5. Close professionally
- Calls may be in **Bangla, English, or a mix of both**. Evaluate accordingly - do not penalize
  code-switching between Bangla and English; that is normal and expected in this call center.
- The approved Pre-Issuance Call Script (in Bangla) is embedded in the evaluation rubric below.

## APPROVED SCRIPT REFERENCE (Pre-Issuance Call)

### 1. Opening / Greetings
"আসসালামু আলায়কুম / নমস্কার / আদাব। Good Morning / Good Afternoon / Good Evening
মেটলাইফের Call Center থেকে আমি ________ বলছি।
আমি কি ________ স্যারের / ম্যাডামের সাথে কথা বলছি?
ধন্যবাদ, স্যার / ম্যাডাম! সম্প্রতি মেটলাইফ থেকে নতুন একটি বীমা পলিসি গ্রহনের জন্য আবেদন করেছেন।
সে ব্যাপারে কথা বলার জন্য কল করেছি। কয়েক মিনিট সময় কি আমি এখন পেতে পারি?"

### 2. Caller Authentication (ID Verification)
- Mentions the call is being recorded ("এই কল টি রেকর্ড করা হচ্ছে")
- Q1 (mandatory): Date of birth as stated on the policy application
- At least ONE of Q2-Q5: premium amount, face amount, policy term, or beneficiary
- Confirms registered address
- Discloses that approval, claims, and maturity settlement are based on the customer's own
  signature on the application form - this is a compliance-critical line, treat it as part of
  authentication, not filler.

### 3. Financial / Product Verification
Confirm: Product Name, Face Amount, Effective Date, Policy Term, Premium Amount, Premium Mode,
Number of Premiums, Beneficiary Name, next premium due date, and whether the next premium will
be paid via EFT.
- If the customer says EFT "No": TSR should ask how they plan to pay, and that's still fine
  as long as the answer is a normal payment intention.
- If the customer gives a *negative* answer (e.g. "I won't continue the policy" / "I didn't even
  apply for this"): the correct handling is to calmly note it and flag it for escalation to the
  Financial Associate / Complaint Handling Unit, NOT to argue, pressure, or ignore it. Score
  "Complaint Handling" specifically on how this moment is handled if it occurs.

### 4. Agent of Record
Confirm the name of the Financial Associate/Agent who sold the policy (or the Unit/Branch
Manager if the customer doesn't recognize the FA name - this is an acceptable substitution,
not an error).

### 5. APL (Automatic Premium Loan)
Explain: if premium isn't paid on time and there's cash value, the policy continues automatically;
overdue amounts must later be repaid with charges; prolonged non-payment eventually lapses the
policy and coverage ends.

### 6. Health Declarations
Confirm the health and other information on the application is accurate, since future claims
depend on it.

### 7. Agent Cash Collection
Inform the customer that, per regulatory guidance, agents/FAs cannot collect cash premiums after
policy issuance.

### 8. Use & Clarity of Sales Material
Ask whether the FA showed and clearly explained the sales brochure. (For DPS Super policies
specifically, the script asks about the two-page DPS Super illustration *and* the brochure -
treat that as the same script beat, just with product-specific wording.)

### 9. Surrender Value
Inform the customer that surrendering early results in a value lower than premiums paid.

### 10. Video Link
Inform the customer that a follow-up video link with more information will be sent after the call.

### 11. Closing
"আমি আবারও বলছি আমি মেটলাইফের Call Center থেকে ____________ আপনার সঙ্গে কথা বলছিলাম।"
Followed by thanks and a warm, professional sign-off.

---

## HOW TO SEPARATE AGENT PERFORMANCE FROM CUSTOMER BEHAVIOR
Score the **TSR's** performance, not the customer's. A quiet, confused, hesitant, or slow
customer is not the TSR's fault. Only lower a score when the *TSR's own* handling of that
situation was weak (e.g. they got impatient, talked over the customer, or failed to re-explain
something clearly). If a section is short only because the customer answered "yes" quickly and
needed no elaboration, that is not a script omission.

## AVOID DEFAULT-MIDDLE SCORING
Do not default to a "safe" 3 out of habit. If the evidence clearly supports a 5 or a 1, give it.
Human QA reviewers distrust a report where every call scores 3-4 across the board - it usually
means the rubric wasn't actually applied. Use the full 0-5 range across a batch of calls: most
should still land 3-5 (agents are usually competent), but real 0-1s on real failures are what
make this tool useful for identifying who needs counselling.

## EVALUATION CRITERIA (Each scored 0-5)
For every criterion, first identify the concrete evidence (what was actually said/done, or
notably absent), then assign the score using the anchors below.

### 1. Greetings (0-5)
Standard greeting elements: greeting phrase, self + company intro, asks for the named customer,
states purpose of call, asks for a few minutes.
- 5: All elements present, clear and natural delivery.
- 4: All elements present, minor wording deviation or slight rush.
- 3: 3-4 of 5 elements present.
- 2: Only 2 elements present, or greeting is garbled/unclear.
- 1: Attempted but mostly incorrect or incomplete.
- 0: No greeting at all.

### 2. Caller Authentication (0-5)
Recording notice + DOB (Q1, mandatory) + at least one of Q2-Q5 + address + signature-basis
disclosure.
- 5: All five elements present and clearly done.
- 4: Four of five present (commonly: signature-basis or address skipped), rest solid.
- 3: DOB + one of Q2-Q5 done, but two other elements missing.
- 2: Only DOB verified, nothing else attempted.
- 1: Authentication attempted but mostly incorrect, unclear, or skipped critical parts.
- 0: No authentication performed at all.

### 3. Telephony Etiquette (0-5)
Anchors:
- 5 sounds like: consistent "স্যার/ম্যাডাম", never interrupts, thanks the customer naturally
  more than once, tone stays warm even if the customer is slow or confused.
- 3 sounds like: generally polite but interrupts the customer once or twice, or drops honorifics
  in a few places, otherwise fine.
- 1 sounds like: audibly curt or rushed tone, talks over the customer repeatedly, or drops
  honorifics for most of the call.
- 0 sounds like: rude, dismissive, or argues with the customer.
Score between these anchors based on where the call actually falls.

### 4. Pronunciation (0-5)
- 5: Crystal clear, natural pace, English terms (policy, premium, EFT, APL) pronounced correctly.
- 4: Very clear, 1-2 minor mispronunciations.
- 3: Understandable but noticeably accented or occasionally hard to follow.
- 2: Frequent unclear speech that would make a real customer ask them to repeat themselves.
- 1: Significant speech difficulty throughout.
- 0: Largely incomprehensible.

### 5. Script Following (0-5)
Compliance with the approved sequence: Opening -> Authentication -> Financial Details ->
Agent of Record -> APL -> Health Declarations -> Agent Cash Collection -> Sales Material
Clarity -> Surrender Value -> Video Link -> Closing.
- 5: All 11 sections covered, correct order, nothing major skipped.
- 4: All sections covered, 1 minor reorder or omission.
- 3: 7-8 of 11 sections covered.
- 2: About half covered, significant skips.
- 1: Only 1-3 sections covered.
- 0: Script not followed at all.
List which sections (if any) were skipped in the evidence field for this criterion.

### 6. Handling Time (0-5)
Efficient delivery, no unnecessary dead air or filler, doesn't rush past the customer's
questions, appropriate total length for the content actually covered.
- 5: Efficient, smooth transitions, no wasted time.
- 4: Minor inefficiency (brief pauses, small repetition).
- 3: Noticeable inefficiency (repeated info, longer pauses) but call still lands.
- 2: Clearly too rushed or too drawn out for the content.
- 1: Poor time management causing visible confusion.
- 0: Chaotic, no time management at all.

### 7. Complaint Handling (0-5)
Covers any pushback, hesitation, confusion, or explicit complaint from the customer (including
the "customer doesn't want the policy" scenario described above). If nothing of the sort comes
up at all, score 5 by default - do not penalize a smooth call for lacking a complaint to handle.
- 5: Patient, empathetic, correctly escalates/resolves, or no issue arose at all.
- 4: Handled well with a minor gap (slight impatience, unclear next step).
- 3: Adequate but noticeably thin empathy or resolution.
- 2: Dismissive or impatient, no real resolution offered.
- 1: Response made the situation worse.
- 0: Rude or confrontational in response to the customer.

### 8. Attentiveness / Focus (0-5)
Responds relevantly, doesn't ask the customer to repeat things already given, picks up on
customer cues (confusion, hesitation) and adapts.
- 5: Fully engaged and responsive throughout.
- 4: 1-2 minor lapses.
- 3: Generally attentive, occasional missed cue.
- 2: Noticeably inattentive - misses responses, asks for repeats.
- 1: Largely inattentive.
- 0: Completely disengaged.

### 9. Closing (0-5)
Re-states name and MetLife Call Center, thanks the customer, warm professional sign-off.
- 5: Complete and matches script.
- 4: Missing one minor element.
- 3: States name but skips thanks or well-wishes.
- 2: Very brief/incomplete.
- 1: Attempted but mostly skipped.
- 0: No closing - call ends abruptly.

---

## OUTPUT INSTRUCTIONS
Return your evaluation using the structured schema provided to you (`QualityCheckLLMOutput`).
Provide exactly 9 entries in `criteria_scores`, in the fixed order listed above. For each one,
give concrete `evidence` before the `justification` - the evidence is what a human reviewer will
check first if they disagree with your score. Do NOT compute totals, percentage, or a counselling
recommendation yourself - that is handled in code. If audio quality, overlapping speech, or
language mix genuinely prevents you from judging something confidently, set `low_confidence_flag`
and explain why, rather than guessing. Base every score strictly on what you actually hear. Do
not assume anything you did not hear, and do not let a slow or confused customer lower the TSR's
score unless the TSR's own handling of them was actually weak.
"""

HUMAN_PROMPT_TEXT = (
    "Please listen to this call recording carefully and evaluate the agent's performance "
    "according to the 9 quality criteria in your instructions."
)

# How many points apart on the SAME criterion across two independent passes
# counts as "the model isn't sure" and should be surfaced to a human, rather
# than silently averaged away. See run_quality_check() for how this is used.
VARIANCE_THRESHOLD = 2


def _build_model() -> ChatGoogleGenerativeAI:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set.")

    # Bumped from flash-lite to flash by default: flash-lite is the
    # cheapest/smallest tier and is a common source of shallow judgment on
    # subjective criteria (empathy, tone, complaint handling). This is a
    # much smaller cost increase than switching vendors, and worth testing
    # against your current results before anything else. Override via
    # GEMINI_MODEL if you want to go back to flash-lite for cost reasons.
    model_name = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0,  # deterministic grading, not creative writing
    )


async def _single_pass(llm: ChatGoogleGenerativeAI, audio_b64: str, mime_type: str) -> QualityCheckLLMOutput:
    structured_llm = llm.with_structured_output(QualityCheckLLMOutput, include_raw=True)

    human_message = HumanMessage(
        content=[
            {"type": "media", "mime_type": mime_type, "data": audio_b64},
            {"type": "text", "text": HUMAN_PROMPT_TEXT},
        ]
    )
    system_message = SystemMessage(content=SYSTEM_PROMPT)

    result = await structured_llm.ainvoke([system_message, human_message])
    raw_message = result["raw"]
    parsed: Optional[QualityCheckLLMOutput] = result["parsed"]
    parsing_error = result.get("parsing_error")

    usage = getattr(raw_message, "usage_metadata", None)
    if usage:
        print(f"usage metadata: {usage}")

    if parsed is None:
        raise ValueError(
            f"Gemini did not return a schema-conformant response. "
            f"parsing_error={parsing_error!r}, raw_content={raw_message.content!r}"
        )
    return parsed, usage


async def run_quality_check(
    audio_bytes: bytes,
    mime_type: str = "audio/mpeg",
    double_check: bool = False,
) -> QualityCheckResult:
    """
    Send the audio file to Gemini and get back a validated, scored
    QualityCheckResult.

    double_check: if True, runs the evaluation twice and flags any criterion
    where the two passes disagree by more than VARIANCE_THRESHOLD points, so
    a human reviewer knows exactly which scores to double check rather than
    trusting a single pass blindly. Doubles your API cost per call - use it
    for spot-checking agents near the counselling threshold, or periodically
    on a sample, rather than on every single call.
    """
    audio_b64 = base64.standard_b64encode(audio_bytes).decode("utf-8")
    llm = _build_model()

    parsed, usage_metadata = await _single_pass(llm, audio_b64, mime_type)

    score_variance_flag = False
    score_variance_note = None

    if double_check:
        parsed_2, usage_metadata_2 = await _single_pass(llm, audio_b64, mime_type)
        scores_1 = {cs.name: cs.score for cs in parsed.criteria_scores}
        scores_2 = {cs.name: cs.score for cs in parsed_2.criteria_scores}
        diffs = {
            name: (scores_1.get(name), scores_2.get(name))
            for name in scores_1
            if name in scores_2 and abs(scores_1[name] - scores_2[name]) >= VARIANCE_THRESHOLD
        }
        if diffs:
            score_variance_flag = True
            score_variance_note = "; ".join(
                f"{name}: pass1={s1}, pass2={s2}" for name, (s1, s2) in diffs.items()
            )
        # Keep the first pass as the reported result (it's already fully
        # formed with evidence text); the second pass is purely a
        # consistency check, not something to silently average in.

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
            evidence=cs.evidence,
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
        call_language_note=parsed.call_language_note,
        criteria_scores=criteria_scores,
        total_marks_obtained=total_obtained,
        total_marks_possible=total_possible,
        percentage=percentage,
        needs_counselling=needs_counselling,
        counselling_reason=counselling_reason,
        overall_summary=parsed.overall_summary,
        low_confidence_flag=parsed.low_confidence_flag,
        low_confidence_reason=parsed.low_confidence_reason,
        score_variance_flag=score_variance_flag,
        score_variance_note=score_variance_note,
        usage_metadata=usage_metadata,
    )