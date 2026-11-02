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

| File                                | Purpose                                                                                                                |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `config.py`                       | **The scorecard**: parameters, checklists, Good/Average/Bad anchors, N/A rules, critical errors, bands, settings |
| `knowledge/gozayaan_knowledge.md` | **GoZayaan policies, scripts and QA calibration rulings.** Fill this in (see below)                              |
| `prompts.py`                      | Pass-1, pass-2 and coaching prompts, generated from `config.py`                                                      |
| `pipeline.py`                     | Evaluation pipeline, verification and consistency rules                                                                |
| `schemas.py`                      | Structured output and API models                                                                                       |
| `scoring.py`                      | Score calculation and QA overrides (shared by the API and the portal)                                                  |
| `main.py`                         | FastAPI: `GET /qc/config`, `POST /qc/jobs` + `GET /qc/jobs/{id}`, `POST /qc/evaluate`                          |
| `frontend.py`                     | Streamlit portal                                                                                                       |
| `excel_report.py`                 | Excel export                                                                                                           |
| `evaluate_against_human.py`       | Measures AI vs human QA agreement                                                                                      |

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




I've built GoZayan-QC in [GoZayan-QC/](GoZayan-QC/). It passes offline tests on mock calls and one small live Gemini check on synthetic data, but it hasn't been run on a real GoZayaan recording yet. Nothing is committed.

## Why it's more than "upload to Gemini and ask for a report"

1. **Two passes.** The first pass listens and records what happened: a timestamped Bangla transcript, holds, dead air, interruptions, colleague consultations, escalations, every fact the agent stated, and tone and pace. The second pass is the QA auditor. It gets the audio, those notes and GoZayaan's policies. For each parameter it quotes the transcript, works through 3–5 checklist items, then gives Good/Average/Bad and a coaching tip.
2. **The code checks the model.** It confirms every quote actually appears in the transcript. It also flags contradictions, for example "No Assistance = Good" when a colleague consultation was heard, or "Product Knowledge = Good" when a claim contradicts policy. The code calculates the score, not the model. Doubtful calls go to a **review queue**.
3. **Company knowledge** ([knowledge/gozayaan_knowledge.md](GoZayan-QC/knowledge/gozayaan_knowledge.md)). Fares, fees, refund timelines and greeting scripts are checked against GoZayaan's own policy. This is the main thing a general chatbot can't do. It's mostly empty until you fill it in.
4. **Portal.** It works like the demo: multiple uploads and an Excel download. It adds:
   - an agent name per file
   - a progress bar
   - a team dashboard with an agent × parameter heatmap
   - a call review screen with audio, verified quotes, the transcript and QA overrides
   - a coaching plan per agent
   - an Excel report with 7 sheets

**Testing:** the offline run caught every mistake I planted in a mock call. The live check used your key from `Agent-QC-demo/.env` and cost under one cent. Gemini accepted the schemas, and every quote it cited was verified.

## Problems in the criteria you pasted

- **Politeness and Empathy** had Escalation's description pasted in by mistake. I wrote a proper definition.
- Your example mentions **"Tonality"**, which isn't one of the 10 parameters. I assumed it falls under Politeness and Empathy.
- The official brand spelling is **GoZayaan**. That's what the prompts use; you can change it with `QC_BRAND_NAME`.

## Defaults I chose; please confirm with GoZayaan

- **N/A on Escalation or FCR** counts as a full point, so every call stays out of 10.
- **Pass mark** is 8/10.
- **Critical errors**, such as asking for an OTP or rudeness, flag the call and force "QA Satisfaction" to Bad. They don't zero the score. You can change this in `.env`.

## What I need from you

1. GoZayaan's real greeting and closing script, refund and date-change charges and timelines, escalation rules, and caller verification rules. All of these go in the knowledge file.
2. **30–50 calls already scored by their human QA.** This is the most important item.
3. A few real recordings, plus the dialler's filename format if agent names can be read from it.

## Model, budget and evaluation

- **Stay on 3.1 Flash-Lite for now.** I estimate about 2–3 US cents per 5-minute call, roughly $25 per 1,000 calls. That's my rough estimate from the published rates ($0.50/M audio input, $1.50/M output), not a measured figure. Gemini 3 Flash costs about twice as much. Only switch the scoring pass (`QC_SCORING_MODEL`) if calibration shows Flash-Lite is too lenient. Base that decision on measured agreement, not guesswork.
- **Yes, you need evaluation.** Without it you can't tell a business how accurate the system is. [evaluate_against_human.py](GoZayan-QC/evaluate_against_human.py) compares AI and human ratings for each parameter. The portal exports the human labels CSV for you once QA has reviewed calls. Aim for at least 80% exact agreement per parameter. Each time a disagreement will keep recurring, add a ruling to the knowledge file, then run the script again.
- **Your MetLife chains set temperature to 0 or 0.1.** Google recommends the default for Gemini 3 models ([guide](https://ai.google.dev/gemini-api/docs/gemini-3)), and lower values risk repeated lines in transcripts. I left it at the default here.
- **Batch API:** if QC can run overnight, Gemini's Batch API halves the cost. I haven't built that yet.

To run it: `pip install -r requirements.txt`, copy `.env.example` to `.env`, then `uvicorn main:app --port 8000` and `streamlit run frontend.py`.

Sources: [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing), [Gemini 3 developer guide](https://ai.google.dev/gemini-api/docs/gemini-3), [Gemini 3.1 Flash-Lite](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-lite)
