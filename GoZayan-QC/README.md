# GoZayaan Call QC

AI quality assurance for GoZayaan call-centre recordings, scored on GoZayaan's 10-parameter scorecard
(Good = 1, Average = 0.5, Bad = 0, total out of 10).

## How it works

```
recording ─► Pass 1: listening analyst (Gemini, audio)
               transcript with timestamps · events (holds, dead air, interruptions, colleague
               consultations, escalations...) · every factual claim the agent made · voice delivery
          ─► Pass 2: QA auditor (Gemini, audio + pass-1 notes + GoZayaan knowledge base)
               claim check against policy · per parameter: verbatim quotes → checklist → rating → coaching tip
          ─► Code (no AI)
               quote verification against the transcript · consistency rules (e.g. "No Assistance = Good
               but a colleague consultation was heard") · score, band, pass/fail · review flags
batch    ─► per-agent averages + one coaching brief per agent
```

The model never does the arithmetic and never has the final say on a doubtful call. Anything the code
can't verify sends the call to the **review queue**, where a QA reviewer confirms or overrides it in
the portal.

## Files

| File | Purpose |
|---|---|
| `config.py` | **The scorecard**: parameters, checklists, Good/Average/Bad anchors, N/A rules, critical errors, bands, settings |
| `knowledge/gozayaan_knowledge.md` | **GoZayaan policies, scripts and QA calibration rulings.** Fill this in (see below) |
| `prompts.py` | Pass-1, pass-2 and coaching prompts, generated from `config.py` |
| `pipeline.py` | Evaluation pipeline, verification and consistency rules |
| `schemas.py` | Structured output and API models |
| `scoring.py` | Score calculation and QA overrides (shared by the API and the portal) |
| `main.py` | FastAPI: `GET /qc/config`, `POST /qc/jobs` + `GET /qc/jobs/{id}`, `POST /qc/evaluate` |
| `frontend.py` | Streamlit portal |
| `excel_report.py` | Excel export |
| `evaluate_against_human.py` | Measures AI vs human QA agreement |

## Run

```bash
pip install -r requirements.txt
cp .env.example .env          # add GOOGLE_API_KEY
uvicorn main:app --port 8000
streamlit run frontend.py
```

Run a single uvicorn worker: background jobs are held in memory.

## Before going live

1. **Fill the knowledge base.** Without it, the AI can only judge whether the agent *sounded* right.
   With it, the AI checks every fare, fee, refund rule and timeline the agent states. Only write
   real GoZayaan policy there.
2. **Calibrate against your human QA.** Have the QA team score 30-50 calls normally. Run the same
   calls through the portal, review them, then download the QA labels CSV and the JSON:
   ```bash
   python evaluate_against_human.py --report gozayaan_qc_*.json --human gozayaan_qa_labels_*.csv --out disagreements.csv
   ```
   Aim for at least 80% exact agreement per parameter. For each parameter below target, read the
   disagreements, then add a ruling to the *QA calibration rulings* section of the knowledge base.
   Re-run.
3. Repeat step 2 after any change to the prompt, model or knowledge base.

## Policies (`.env`)

- `QC_NA_POLICY`: `full_credit` (N/A counts as Good, the default) or `exclude` (score rescaled over the applicable parameters).
- `QC_CRITICAL_ERROR_POLICY`: `flag` (score unchanged, call always reviewed) or `zero` (call scores 0).
- `QC_PASS_MARK`: default 8/10. A call with a critical error never passes.
- `QC_SCORING_MODEL`: use a stronger model for pass 2 only, if calibration shows flash-lite is too lenient or inconsistent.
