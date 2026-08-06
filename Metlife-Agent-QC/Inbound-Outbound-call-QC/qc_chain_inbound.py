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
    name: str = Field(description="Name of the evaluation criterion, e.g. 'Issue Resolution'.")
    score: int = Field(ge=0, le=5, description="Score for this criterion, 0-5.")
    max_score: int = Field(default=5, description="Maximum possible score, always 5.")
    evidence: str = Field(
        description=(
            "The specific thing the agent said or did (or failed to say/do) that this score is "
            "based on, in your own words, with an approximate timestamp if you can tell "
            "(e.g. 'around 1:20'). This is what a human reviewer checks the score against, so "
            "be concrete, not generic."
        )
    )
    justification: str = Field(
        description="1-2 sentence explanation of why the evidence maps to this score under the rubric."
    )


class InboundQCLLMOutput(BaseModel):
    """
    This is exactly what we ask Gemini to produce. Totals/percentage/
    needs_counselling are deliberately NOT trusted from the model - we
    compute them ourselves afterwards for consistency and to avoid
    arithmetic mistakes creeping into the graded output.
    """
    agent_name: Optional[str] = Field(
        default=None, description="Detected agent name, or null if not identifiable."
    )
    call_duration_note: Optional[str] = Field(
        default=None, description="Brief note on call length, e.g. 'Approximately 5 minutes'."
    )
    call_language_note: Optional[str] = Field(
        default=None,
        description="Brief note on language mix, e.g. 'Primarily Bangla with English product terms'.",
    )
    customer_issue_summary: str = Field(
        description=(
            "1-2 sentence plain-language summary of what the customer called about. This gives "
            "a human reviewer instant context before reading the scores."
        )
    )
    criteria_scores: list[CriteriaScore] = Field(
        description="Exactly 12 entries, one per evaluation criterion, in the fixed order: "
        "Greetings, Caller Authentication, Telephony Etiquette, Pronunciation, "
        "Issue Identification, Information Accuracy, Issue Resolution, Handling Time, "
        "Complaint Handling, FCR (First Call Resolution), Attentiveness / Focus, Closing."
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
class InboundQCResult(BaseModel):
    agent_name: Optional[str] = None
    call_duration_note: Optional[str] = None
    call_language_note: Optional[str] = None
    customer_issue_summary: str
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
# Inbound calls have no fixed script to check compliance against - the
# customer drives the topic. So the rubric below leans much more heavily on
# behavioral anchors (what good listening/empathy/resolution actually
# sounds like) than on "was step X said", which is the pre-issuance model.
# Same reliability patterns as the pre-issuance chain: evidence-first
# scoring, no default-middle scoring, explicit agent-vs-customer separation,
# and a low-confidence escape hatch.

SYSTEM_PROMPT = """
You are an expert **Call Quality Analyst** for MetLife Bangladesh's call center.
Your job is to listen carefully to the provided **inbound** call recording and objectively
evaluate the agent's performance against **12 quality criteria**, the way a strict, fair,
experienced human QA reviewer would.

## IMPORTANT CONTEXT
- This is an **inbound** call: the customer called in with a query, request, or complaint.
  There is no fixed script to follow, unlike outbound pre-issuance calls. The customer sets the
  topic, and you are evaluating HOW WELL the agent handled it - listening, diagnosing the issue,
  giving correct information, resolving it (or setting correct expectations), and behaving
  professionally throughout.
- Calls may be in **Bangla, English, or a mix of both**. Evaluate accordingly - do not penalize
  code-switching between Bangla and English; that is normal and expected in this call center.
- Before scoring, form a one- or two-sentence understanding of what the customer actually called
  about (this goes in `customer_issue_summary`) - you cannot judge Issue Identification or
  Issue Resolution without first being clear on what the actual issue was.

## HOW TO SEPARATE AGENT PERFORMANCE FROM CUSTOMER BEHAVIOR
Score the **agent's** performance, not the customer's. A confused, upset, vague, or
hard-to-understand customer is not the agent's fault. Only lower a score when the *agent's own*
handling of that situation was weak (e.g. they got impatient, failed to probe for clarity, gave
up too early, or didn't adapt their explanation when the customer clearly didn't understand).
An angry customer that the agent calmed down and helped is a GOOD outcome for the agent, not a
penalty.

## AVOID DEFAULT-MIDDLE SCORING
Do not default to a "safe" 3 out of habit. If the evidence clearly supports a 5 or a 1, give it.
Human QA reviewers distrust a report where every call scores 3-4 across the board - it usually
means the rubric wasn't actually applied. Use the full 0-5 range across a batch of calls: most
agents are competent and should land 3-5, but real 0-1 scores on real failures are what make this
tool useful for identifying who needs counselling.

## EVALUATION CRITERIA (Each scored 0-5)
For every criterion, first identify the concrete evidence (what was actually said/done, or
notably absent), then assign the score using the anchors below.

### 1. Greetings (0-5)
Agent greets the customer professionally and follows the standard courteous opening (identifies
self and MetLife, greets warmly, asks how they can help).
- 5: Warm, professional, complete opening with no prompting needed.
- 4: Professional opening, one minor element missing (e.g. doesn't give own name).
- 3: Adequate but rushed or generic ("Hello, MetLife" with no warmth).
- 2: Minimal or perfunctory greeting.
- 1: Greeting barely present or noticeably curt.
- 0: No greeting at all - jumps straight into the conversation.

### 2. Caller Authentication (0-5)
Before sharing any account-specific or policy-specific information, the agent verifies the
caller's identity (e.g. policy number, name, date of birth, or other standard verification
questions) rather than disclosing sensitive information to an unverified caller.
- 5: Verification done clearly and fully before any sensitive info is shared.
- 4: Verification done, but with a minor gap (e.g. only one identifier checked when policy
  calls for two).
- 3: Some verification attempted, but info was shared before it was fully completed.
- 2: Verification attempted only after already sharing some sensitive information.
- 1: Verification is token/perfunctory (e.g. asks name only, no real check).
- 0: No verification at all before discussing account details.
- If the call genuinely never touches sensitive/account-specific information (e.g. a purely
  general product question), score 5 and note this in evidence - there was nothing to gate.

### 3. Telephony Etiquette (0-5)
Anchors:
- 5 sounds like: consistently polite, warm tone, never interrupts, uses respectful address,
  stays calm and courteous even if the customer is frustrated.
- 3 sounds like: generally polite but interrupts once or twice, or tone flattens under pressure.
- 1 sounds like: audibly curt, talks over the customer repeatedly, or sounds annoyed.
- 0 sounds like: rude, dismissive, or argues with the customer.

### 4. Pronunciation (0-5)
- 5: Crystal clear, natural pace, technical/product terms pronounced correctly.
- 4: Very clear, 1-2 minor mispronunciations.
- 3: Understandable but noticeably accented or occasionally hard to follow.
- 2: Frequent unclear speech that would make a real customer ask them to repeat themselves.
- 1: Significant speech difficulty throughout.
- 0: Largely incomprehensible.

### 5. Issue Identification (0-5)
Agent asks relevant clarifying questions and correctly pins down what the customer actually
needs, rather than guessing or assuming.
- 5: Quickly and accurately identifies the real issue, asks smart clarifying questions if the
  initial description was vague.
- 4: Identifies the issue correctly, with slightly more back-and-forth than necessary.
- 3: Gets there eventually, but with some avoidable confusion or a missed clarifying question.
- 2: Misunderstands the issue at least once, requiring the customer to re-explain.
- 1: Largely fails to identify what the customer actually wants.
- 0: Never establishes what the issue is.

### 6. Information Accuracy (0-5)
Information given to the customer is correct, complete, and consistent with MetLife policy -
judge this only on information you can actually verify from what's said (e.g. internally
inconsistent statements, or a clear factual error the agent later self-corrects, count as
evidence; don't penalize for information you have no way to fact-check from the recording alone).
- 5: Everything stated is accurate, complete, and clearly explained.
- 4: Accurate but incomplete in a minor way (e.g. forgets to mention a related detail).
- 3: Mostly accurate, one noticeable gap or slightly unclear explanation.
- 2: Gives information that is vague, hedged, or partially incorrect.
- 1: Gives information that is clearly wrong or contradicts itself.
- 0: Provides misleading or false information with confidence.
- If you cannot verify accuracy either way from the recording (no factual claim actually made,
  or nothing to check it against), score 5 and say so in evidence - don't penalize for
  unverifiable content.

### 7. Issue Resolution (0-5)
Agent resolves the issue on the call, or - if it genuinely can't be resolved immediately - gives
correct, specific next steps (what will happen, by whom, by when).
- 5: Issue fully resolved, or a clear and correct escalation/next-step path given with a
  timeframe.
- 4: Resolved or handed off correctly, but next steps were slightly vague (e.g. no timeframe).
- 3: Partial resolution, or next steps given but generic/unclear.
- 2: Issue left unresolved with no real guidance on what happens next.
- 1: Agent gives up or deflects without attempting to help.
- 0: Issue actively mishandled (wrong action taken, or customer sent in a clearly wrong direction).

### 8. Handling Time (0-5)
Efficient management of the call - no unnecessary dead air, doesn't rush past the customer,
appropriate length for the complexity of the issue actually raised.
- 5: Efficient, smooth, no wasted time, but never feels rushed.
- 4: Minor inefficiency (brief pauses, small repetition, a bit of hold time).
- 3: Noticeable inefficiency (long holds, repeated info) but call still lands.
- 2: Clearly too rushed (customer cut off) or too drawn out for the issue.
- 1: Poor time management causing visible customer frustration.
- 0: Chaotic, no time management at all.

### 9. Complaint Handling (0-5)
If the customer expresses frustration, dissatisfaction, or an explicit complaint: agent listens
without interrupting, acknowledges the concern with empathy, takes ownership rather than
deflecting blame, and follows the correct escalation/resolution process.
If no complaint or frustration arises at all, score 5 by default - do not penalize a smooth call
for lacking a complaint to handle.
- 5: Empathetic, patient, ownership taken, correct process followed, or no issue arose at all.
- 4: Handled well with a minor gap (slight impatience, ownership not fully stated).
- 3: Adequate but noticeably thin empathy or follow-through.
- 2: Dismissive, defensive, or blames the customer/another department without help.
- 1: Response visibly escalates the customer's frustration.
- 0: Rude or confrontational in response to a complaint.

### 10. FCR - First Call Resolution (0-5)
Judge this ONLY from what is actually said in the call. Evidence you can use: the customer
states this is a repeat contact about the same issue ("I called about this before"), or the
agent explicitly says the customer will need to call back or wait for a callback for something
that could have been handled now, or the issue is clearly and completely resolved in this single
call.
- 5: Issue fully resolved in this call, or clear evidence this is genuinely a one-and-done
  contact.
- 4: Resolved in this call but a small follow-up step remains that doesn't require a new inbound
  contact (e.g. an email will be sent).
- 3: Cannot be determined from the recording either way - default here rather than guessing.
- 2: Agent's own handling creates an avoidable reason for the customer to call back again.
- 1: Customer states this is already a repeat call about the same unresolved issue.
- 0: Customer states they have called multiple times before and it is still unresolved.
- Do NOT infer a repeat-contact history that was never mentioned in the call - if there's no
  evidence either way, use 3 and say so in evidence, and consider setting `low_confidence_flag`
  if this materially affects the overall picture of the call.

### 11. Attentiveness / Focus (0-5)
Responds relevantly, doesn't ask the customer to repeat things already given, picks up on
customer cues (confusion, frustration, urgency) and adapts.
- 5: Fully engaged and responsive throughout.
- 4: 1-2 minor lapses.
- 3: Generally attentive, occasional missed cue.
- 2: Noticeably inattentive - misses responses, asks for repeats.
- 1: Largely inattentive.
- 0: Completely disengaged.

### 12. Closing (0-5)
Agent confirms the resolution (or next steps) with the customer, offers further assistance, and
thanks the customer before ending the call.
- 5: All three elements present, warm and professional.
- 4: Two of three elements present.
- 3: Only a brief thanks/goodbye, no confirmation of resolution or offer of further help.
- 2: Very abrupt closing.
- 1: Barely a closing at all.
- 0: Call ends abruptly with no closing.

---

## OUTPUT INSTRUCTIONS
Return your evaluation using the structured schema provided to you (`InboundQCLLMOutput`).
First fill in `customer_issue_summary` so your own scoring is grounded in a clear understanding
of the call. Provide exactly 12 entries in `criteria_scores`, in the fixed order listed above.
For each one, give concrete `evidence` before the `justification` - the evidence is what a human
reviewer will check first if they disagree with your score. Do NOT compute totals, percentage,
or a counselling recommendation yourself - that is handled in code. If audio quality, overlapping
speech, or language mix genuinely prevents you from judging something confidently, set
`low_confidence_flag` and explain why, rather than guessing. Base every score strictly on what
you actually hear. Do not assume anything you did not hear, and do not let a difficult customer
lower the agent's score unless the agent's own handling of them was actually weak.
"""

HUMAN_PROMPT_TEXT = (
    "Please listen to this inbound call recording carefully and evaluate the agent's performance "
    "according to the 12 quality criteria in your instructions."
)

# How many points apart on the SAME criterion across two independent passes
# counts as "the model isn't sure" and should be surfaced to a human, rather
# than silently averaged away.
VARIANCE_THRESHOLD = 2


def _build_model() -> ChatGoogleGenerativeAI:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set.")

    model_name = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0,  # deterministic grading, not creative writing
    )


async def _single_pass(llm: ChatGoogleGenerativeAI, audio_b64: str, mime_type: str):
    structured_llm = llm.with_structured_output(InboundQCLLMOutput, include_raw=True)

    human_message = HumanMessage(
        content=[
            {"type": "media", "mime_type": mime_type, "data": audio_b64},
            {"type": "text", "text": HUMAN_PROMPT_TEXT},
        ]
    )
    system_message = SystemMessage(content=SYSTEM_PROMPT)

    result = await structured_llm.ainvoke([system_message, human_message])
    raw_message = result["raw"]
    parsed: Optional[InboundQCLLMOutput] = result["parsed"]
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


async def run_inbound_quality_check(
    audio_bytes: bytes,
    mime_type: str = "audio/mpeg",
    double_check: bool = False,
) -> InboundQCResult:
    """
    Send an inbound call recording to Gemini and get back a validated, scored
    InboundQCResult.

    double_check: if True, runs the evaluation twice and flags any criterion
    where the two passes disagree by more than VARIANCE_THRESHOLD points.
    Doubles API cost for this call - use for spot-checking, not every call.
    """
    audio_b64 = base64.standard_b64encode(audio_bytes).decode("utf-8")
    llm = _build_model()

    parsed, usage_metadata = await _single_pass(llm, audio_b64, mime_type)

    score_variance_flag = False
    score_variance_note = None

    if double_check:
        parsed_2, _ = await _single_pass(llm, audio_b64, mime_type)
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
    total_possible = 60  # 12 criteria x 5
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

    return InboundQCResult(
        agent_name=parsed.agent_name,
        call_duration_note=parsed.call_duration_note,
        call_language_note=parsed.call_language_note,
        customer_issue_summary=parsed.customer_issue_summary,
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