// =============================================================================
// GoZayaan QC knowledge base
//
// This file is injected into the QC prompt. The AI uses it to check whether
// what the agent said was CORRECT (Product/Service Knowledge, Solution
// Explanation) and whether the greeting / escalation followed GoZayaan's
// procedure. Without it the AI can only judge confidence and plausibility.
//
// RULES FOR EDITING
// - Lines starting with // are comments and are NOT sent to the AI.
// - Only write things that are actually GoZayaan policy. Anything written
//   here is treated as the truth when grading agents.
// - Keep it short and factual: bullet points, numbers, timelines.
// - Example lines below are commented out on purpose - replace them with the
//   real policy and remove the // to activate them.
// =============================================================================

# Standard greeting and closing
// Confirm the exact wording with the GoZayaan QA team.
- Inbound opening: "আসসালামু আলাইকুম, GoZayaan থেকে আমি <agent name> বলছি। আপনাকে কীভাবে সাহায্য করতে পারি?"
- Outbound opening: salutation, "GoZayaan থেকে আমি <agent name> বলছি", confirm the customer's name, state why we are calling, ask if this is a good time.
- Closing: ask "আর কোনোভাবে আপনাকে সাহায্য করতে পারি?", then thank the customer for choosing GoZayaan and say goodbye.

# Services
// - Domestic and international flight booking
// - Hotel booking
// - Holiday / tour packages
// - Visa processing assistance
// - Payment methods accepted: <bKash / Nagad / cards / bank transfer / EMI banks ...>

# Flight changes, cancellations and refunds
// - Date change / reissue: airline penalty + fare difference + GoZayaan service charge of <amount>.
// - Refund: airline cancellation fee + GoZayaan service charge of <amount>; refunded to the original payment method.
// - Refund timeline: <N> working days after the airline releases the refund.
// - No-show rules: <...>

# Hotels and packages
// - Hotel cancellation follows the hotel's policy shown at booking.
// - Package payment schedule: <...>

# Visa
// - GoZayaan assists with documents and submission; approval is decided by the embassy and must never be guaranteed.
// - Standard processing times by country: <...>

# Caller verification
// - Before any change, cancellation or refund on a booking, verify: <booking ID + registered phone / email ...>

# Escalation matrix
// - Refund / penalty exceptions -> <team>, response within <time>
// - Complaints -> <supervisor / complaint team>, ticket number given to customer
// - Customer asks for a supervisor -> <procedure>

# QA calibration rulings
// Decisions the QA team has made on borderline cases. Add one line each time
// a human reviewer disagrees with the AI for a reason that will repeat.
// - Saying "GoZayaan" without the agent's own name in the opening = Greeting Average, not Bad.
// - A hold of under 60 seconds to check airline availability is not assistance.
