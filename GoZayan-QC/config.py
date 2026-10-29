"""
Single source of truth for the GoZayaan QC rubric and runtime settings.

The prompt, the API (/qc/config), the Streamlit portal and the Excel export
all read the parameters from here, so a rubric change is made in one place.
"""
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Runtime settings (all overridable from .env)
# ---------------------------------------------------------------------------

BRAND_NAME = os.environ.get("QC_BRAND_NAME", "GoZayaan")

# Pass 1 (listening / transcript / events) and pass 2 (scoring) can use different
# models, e.g. flash-lite for pass 1 and a stronger model for scoring.
FORENSICS_MODEL = os.environ.get("QC_FORENSICS_MODEL", "gemini-3.1-flash-lite")
SCORING_MODEL = os.environ.get("QC_SCORING_MODEL", "gemini-3.1-flash-lite")
ROLLUP_MODEL = os.environ.get("QC_ROLLUP_MODEL", "gemini-3.1-flash-lite")

# Gemini 3 thinking levels: minimal | low | medium | high. Empty = model default.
FORENSICS_THINKING = os.environ.get("QC_FORENSICS_THINKING", "low")
SCORING_THINKING = os.environ.get("QC_SCORING_THINKING", "medium")

MAX_CONCURRENT_CALLS = int(os.environ.get("QC_MAX_CONCURRENCY", "4"))
MAX_UPLOAD_MB = int(os.environ.get("QC_MAX_UPLOAD_MB", "20"))
MAX_UPLOAD_FILES = int(os.environ.get("QC_MAX_UPLOAD_FILES", "30"))
LLM_RETRIES = int(os.environ.get("QC_LLM_RETRIES", "2"))

KNOWLEDGE_BASE_PATH = Path(os.environ.get("QC_KNOWLEDGE_BASE", BASE_DIR / "knowledge" / "gozayaan_knowledge.md"))

# ---------------------------------------------------------------------------
# Scoring scheme (GoZayaan: 10 parameters x Good 1 / Average 0.5 / Bad 0 = /10)
# ---------------------------------------------------------------------------

RATING_POINTS = {"Good": 1.0, "Average": 0.5, "Bad": 0.0}
RATINGS = ["Good", "Average", "Bad", "N/A"]

# How an N/A parameter (e.g. Escalation when nothing needed escalating) counts:
#   full_credit -> counts as Good (1 point), score stays out of 10   (default, common QA practice)
#   exclude     -> left out, score is rescaled to 10 over the applicable parameters
NA_POLICY = os.environ.get("QC_NA_POLICY", "full_credit")

# Critical errors are always shown and always send the call to review.
#   flag -> score is unchanged (default, matches GoZayaan's current scheme)
#   zero -> the call score becomes 0 (classic "fatal error" policy)
CRITICAL_ERROR_POLICY = os.environ.get("QC_CRITICAL_ERROR_POLICY", "flag")

PASS_MARK = float(os.environ.get("QC_PASS_MARK", "8"))

# (minimum score out of 10, band label) - checked top-down.
SCORE_BANDS = [
    (9.0, "Excellent"),
    (8.0, "Good"),
    (6.5, "Needs Improvement"),
    (0.0, "Coaching Required"),
]

# Event thresholds the listening pass uses.
DEAD_AIR_SECONDS = 8
LONG_HOLD_SECONDS = 60


def score_band(score: float) -> str:
    for minimum, label in SCORE_BANDS:
        if score >= minimum:
            return label
    return SCORE_BANDS[-1][1]


# ---------------------------------------------------------------------------
# QA parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QAParameter:
    key: str
    name: str
    question: str
    checklist: list[tuple[str, str]]  # (item_id, observable behaviour)
    good: str
    average: str
    bad: str
    na_allowed: bool = False
    na_rule: str = ""
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        data = asdict(self)
        data["checklist"] = [{"id": i, "text": t} for i, t in self.checklist]
        return data


PARAMETERS: list[QAParameter] = [
    QAParameter(
        key="greeting",
        name="Greeting (Opening & Ending)",
        question="Did the agent greet the customer using the company's standard greeting and close with the standard ending?",
        checklist=[
            ("GR1", "Opened promptly with a salutation (Assalamu Alaikum / Good morning etc.), no dead air after the call connected"),
            ("GR2", f"Introduced self by name AND the company ({BRAND_NAME}) as in the standard greeting"),
            ("GR3", "Offered help (inbound) or confirmed the customer's identity and stated the reason for calling (outbound)"),
            ("GR4", "Before ending, asked whether the customer needs any further help"),
            ("GR5", f"Closed with thanks and the standard {BRAND_NAME} closing; the agent did not cut the customer off"),
        ],
        good="All five elements present and delivered naturally. Wording may differ slightly from the script as long as every element is there.",
        average="Opening and closing both present but one or two elements missing (typically: no own name, no further-help offer, no brand in the closing) or delivered rushed / mechanically.",
        bad="Opening OR closing missing altogether, no company introduction at all, or the agent ended the call abruptly.",
        notes=[
            "If the recording clearly starts mid-call or is cut by the recording system (not by the agent), judge only the part that exists, say so in the justification and set confidence to low.",
        ],
    ),
    QAParameter(
        key="active_listening",
        name="Active Listening",
        question="Did the agent listen attentively, understand the customer's needs, and ask relevant questions for clarification?",
        checklist=[
            ("AL1", "Let the customer finish speaking - no talking over or cutting off the customer"),
            ("AL2", "Captured key details right the first time (names, route, dates, booking ID / PNR, phone, email) - the customer never had to repeat something because the agent missed it"),
            ("AL3", "Asked relevant probing / clarifying questions to understand the real need (dates, flexibility, passengers, booking reference, what exactly went wrong)"),
            ("AL4", "Confirmed / read back the need or key details before acting on them"),
            ("AL5", "Every customer question or concern was addressed - nothing ignored, no off-topic or scripted replies"),
        ],
        good="Understood the need quickly and accurately, no interruptions, key details confirmed, every question addressed.",
        average="Got there, but with one lapse: one interruption, one detail asked twice, or a missed clarifying question that caused avoidable back-and-forth.",
        bad="Misunderstood the need, the customer had to repeat or correct themselves two or more times, repeated interruptions, or a customer question was ignored.",
    ),
    QAParameter(
        key="product_knowledge",
        name="Product/Service Knowledge",
        question="Did the agent have a proper understanding of the products, services, rules and processes?",
        checklist=[
            ("PK1", "Every factual statement about fares, fees, refund / change rules, timelines, documents and services was accurate (no claim contradicts the knowledge base)"),
            ("PK2", "Answered standard questions confidently, without guessing or excessive hedging ('I'm not sure', 'maybe')"),
            ("PK3", "Used and explained travel terms correctly (PNR, e-ticket, reissue, fare difference, no-show, baggage allowance, transit, visa requirements)"),
            ("PK4", "Found routine information without long searching or silence"),
            ("PK5", "Proactively shared information the customer would need for this request (fare rules, baggage, documents, payment deadline)"),
        ],
        good="Accurate, confident and complete for the topics this call actually covered.",
        average="Generally correct but with noticeable hesitation, a minor omission, or an imprecise answer that did not mislead the customer.",
        bad="Any materially wrong information (fare, fee, penalty, refund amount or timeline, document requirement), or unable to answer routine questions.",
        notes=[
            "Use the claim checks: a claim is wrong only if it contradicts the knowledge base, contradicts itself, or the agent later corrected it. A claim the knowledge base does not cover is NOT wrong.",
            "If the knowledge base is empty or silent on the topic, judge confidence, consistency and plausibility only, and set confidence to medium or low.",
        ],
    ),
    QAParameter(
        key="solution_explanation",
        name="Solution Explanation",
        question="Did the agent clearly explain the solution to the customer?",
        checklist=[
            ("SE1", "Stated the solution / answer / next steps clearly and in a logical order"),
            ("SE2", "Gave the specifics the customer needs: amounts (fare difference, penalty, service charge, refund), timelines, required documents, payment method and deadline"),
            ("SE3", "Explained in plain language the customer could follow; jargon was explained"),
            ("SE4", "Checked the customer understood or summarised what will happen before closing"),
            ("SE5", "Set accurate expectations - no vague 'it will be done soon' without a timeline, no unrealistic promises"),
        ],
        good="The customer clearly knows what will happen, what it costs, when, and what they must do; understanding was checked.",
        average="Solution explained, but missing a specific (amount, timeline or next step) or understanding was not checked.",
        bad="Explanation unclear, incomplete or confusing; the customer is left not knowing what happens next.",
    ),
    QAParameter(
        key="no_assistance",
        name="No Assistance",
        question="Did the agent handle the call independently without requiring additional assistance?",
        checklist=[
            ("NA1", "Did not put the customer on hold or transfer to ask a colleague / supervisor about a routine matter"),
            ("NA2", "No audible consultation with colleagues during the call"),
            ("NA3", "Answered all of the customer's questions without needing someone else"),
        ],
        good="Handled the whole call independently.",
        average="One brief consultation for a genuinely non-routine detail, handled smoothly with proper hold etiquette.",
        bad="Needed help on routine matters, consulted more than once, or handed the call over because they could not handle it.",
        notes=[
            "A policy-required escalation (refund approval by the finance team, complaint to a supervisor, airline waiver request) is NOT assistance - judge it under Escalation.",
            "A hold to check the booking system, airline fare or availability is NOT assistance. Assistance means needing another person's help to do the agent's own job.",
        ],
    ),
    QAParameter(
        key="politeness_empathy",
        name="Politeness and Empathy",
        question="Was the agent courteous, respectful and empathetic throughout, especially when the customer was confused or upset?",
        checklist=[
            ("PE1", "Respectful address and register throughout (স্যার / ম্যাডাম, আপনি - never তুমি / তুই or casual slang)"),
            ("PE2", "Courteous phrases used naturally (ধন্যবাদ, দুঃখিত, অনুগ্রহ করে / please, thank you, sorry)"),
            ("PE3", "Acknowledged the customer's situation or feelings when warranted (flight cancelled, refund delayed, missed flight, money deducted) with an empathy statement or apology"),
            ("PE4", "Tone stayed warm and patient for the whole call, including with a slow, repetitive or upset customer - no irritation, sarcasm, sighing, arguing or blaming the customer"),
            ("PE5", "Hold etiquette: asked permission before a hold and thanked the customer after it (N/A if there was no hold)"),
        ],
        good="Consistently respectful, warm and patient; empathy shown wherever the situation called for it.",
        average="Polite overall but flat or mechanical, empathy missing where it was warranted, or one brief lapse in tone or hold etiquette.",
        bad="Rude, irritated, dismissive, argumentative, blamed the customer, or ignored a clearly distressed customer.",
        notes=[
            "Judge tone from the AUDIO (voice, pace, sighs), not only from the words in the transcript.",
            "If the customer was calm and the situation needed no empathy, PE3 is N/A - do not penalise its absence.",
        ],
    ),
    QAParameter(
        key="language_proficiency",
        name="Language Proficiency",
        question="Did the agent demonstrate clear pronunciation and maintain a consistent accent?",
        checklist=[
            ("LP1", "Clear pronunciation and audible voice - the customer never had to ask the agent to repeat because of clarity"),
            ("LP2", "Consistent accent and register - standard Bangla (প্রমিত) without slipping into heavy regional dialect"),
            ("LP3", "Appropriate pace - neither rushed nor dragging"),
            ("LP4", "Correct sentence construction and professional vocabulary; English travel terms pronounced correctly"),
            ("LP5", "Few filler words / verbal crutches (আ..., মানে, umm) and no slang"),
        ],
        good="Clear, steady, professional speech for the whole call.",
        average="Understandable but with noticeable issues: frequent fillers, uneven pace, some dialect slips or a few mispronounced terms.",
        bad="Hard to understand, the customer asked for repeats because of clarity, or a heavy / inconsistent accent affected the call.",
        notes=[
            "Natural Bangla-English code-switching is normal in Bangladesh and is NOT penalised.",
            "Judge from the AUDIO itself; the transcript cannot show pronunciation or accent.",
        ],
    ),
    QAParameter(
        key="first_contact_resolution",
        name="First Contact Resolution",
        question="Did the agent resolve the customer's issue during the first interaction?",
        checklist=[
            ("FC1", "The customer's primary need was resolved or fully actioned on this call (booking made, change / refund request submitted, information given)"),
            ("FC2", "Secondary questions were answered too"),
            ("FC3", "No avoidable follow-up: the agent did not ask the customer to call back, or promise a callback, for something that could have been done on this call"),
            ("FC4", f"Where the last step depends on an airline / hotel / embassy / bank, everything on {BRAND_NAME}'s side was completed and a clear timeline given"),
        ],
        good="Resolved, or fully actioned with only third-party / system processing remaining.",
        average="Partially resolved - something left open that is at least partly down to the agent (needed information not collected, a secondary question left unanswered).",
        bad="Primary need left unresolved without a valid reason; the customer will have to call again.",
        na_allowed=True,
        na_rule="Only when the call ended before the customer's need could be stated (dropped call, wrong number, customer busy).",
        notes=[
            "Outbound calls: judge whether the call's objective was achieved as far as the agent controls it (e.g. payment reminder delivered and confirmed).",
            "Third-party processing time (e.g. the airline still has to process a refund) is NOT a failure of FCR if the agent completed their part and explained the timeline.",
        ],
    ),
    QAParameter(
        key="escalation",
        name="Escalation",
        question="Did the agent follow proper procedures for escalating the issue if needed?",
        checklist=[
            ("ES1", "Recognised that the issue needed escalation (beyond the agent's authority: refund / penalty exception, airline waiver, complaint, repeated failure, customer asked for a supervisor)"),
            ("ES2", "Told the customer why it is being escalated, to whom, and what happens next with a timeline"),
            ("ES3", "Captured / confirmed the details the escalation needs (booking ID, contact, issue summary) and gave a reference or ticket number if the process provides one"),
            ("ES4", "Did not escalate prematurely something they could handle, and did not refuse or ignore a legitimate supervisor request"),
        ],
        good="Escalation was needed and handled by the procedure: explained, details captured, timeline given.",
        average="Escalated, but the explanation, details or timeline were incomplete.",
        bad="A needed escalation was not done, a supervisor request was refused or ignored, or a routine issue was escalated / transferred unnecessarily.",
        na_allowed=True,
        na_rule="Only when nothing in the call required escalation and the customer did not ask for it.",
    ),
    QAParameter(
        key="qa_satisfaction",
        name="QA Satisfaction",
        question="As the QA examiner, are you satisfied with this call overall?",
        checklist=[
            ("QS1", "The customer ended with their need met or clear next steps, and no worse mood than they started with (unless the news itself was legitimately bad)"),
            ("QS2", f"The call represented {BRAND_NAME} well - you would be comfortable if the customer described this call publicly"),
            ("QS3", "No compliance, financial or reputational risk was created (no critical error, misinformation or false promise)"),
            ("QS4", "The call was efficient for its complexity (no unnecessary length, holds or dead air)"),
        ],
        good="You would use this call as a positive example; no material issue.",
        average="Acceptable, but you would coach the agent on specific points.",
        bad="You are not satisfied: the customer was let down, there is a brand risk, or any critical error occurred.",
        notes=[
            "This is a holistic judgement, NOT an average of the other nine parameters. A call can tick individual boxes and still be unsatisfying (technically correct but cold, customer leaves angry), and vice versa.",
            "Any critical error means this parameter is Bad.",
        ],
    ),
]

PARAMETER_BY_KEY = {p.key: p for p in PARAMETERS}


# ---------------------------------------------------------------------------
# Critical (zero-tolerance) errors
# ---------------------------------------------------------------------------

CRITICAL_ERRORS = {
    "rude_or_abusive": "Rude, abusive, sarcastic or argumentative toward the customer.",
    "sensitive_data": "Asked for, or read back, a full card number, CVV, OTP, PIN or password.",
    "financial_misinformation": "Stated a wrong fare, fee, penalty, refund amount or timeline that could cause the customer financial loss.",
    "false_promise": f"Guaranteed something outside {BRAND_NAME}'s control (visa approval, airline waiver, a specific refund amount or date) without basis.",
    "unverified_change": "Made or promised a change, cancellation or refund without verifying the caller is the booking owner or authorised (only if the knowledge base requires verification).",
    "call_avoidance": "Disconnected the customer, deliberately left them on a long hold, or told them to call back to avoid handling the request.",
    "privacy_breach": "Shared another passenger's or booking's personal details with an unverified person.",
}


def load_knowledge_base() -> str:
    """
    Returns the company knowledge base with authoring comments (lines starting
    with '//') removed, and headings that have nothing under them dropped so the
    model never mistakes an unfilled section for "no policy exists".
    """
    if not KNOWLEDGE_BASE_PATH.exists():
        return ""
    lines = [
        line for line in KNOWLEDGE_BASE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("//")
    ]
    kept = []
    for i, line in enumerate(lines):
        is_heading = line.lstrip().startswith("#")
        next_is_content = i + 1 < len(lines) and not lines[i + 1].lstrip().startswith("#")
        if not is_heading or next_is_content:
            kept.append(("\n" if is_heading and kept else "") + line)
    return "\n".join(kept).strip()


def public_config() -> dict:
    """What the portal needs to render and re-score results (served at GET /qc/config)."""
    return {
        "brand_name": BRAND_NAME,
        "parameters": [p.as_dict() for p in PARAMETERS],
        "rating_points": RATING_POINTS,
        "ratings": RATINGS,
        "na_policy": NA_POLICY,
        "critical_error_policy": CRITICAL_ERROR_POLICY,
        "critical_errors": CRITICAL_ERRORS,
        "pass_mark": PASS_MARK,
        "score_bands": [{"min": m, "label": label} for m, label in SCORE_BANDS],
        "models": {"forensics": FORENSICS_MODEL, "scoring": SCORING_MODEL, "rollup": ROLLUP_MODEL},
    }
