"""
Measures how closely the AI agrees with GoZayaan's human QA team.

    python evaluate_against_human.py --report qc_report.json --human human_scores.csv [--out disagreements.csv]

--report : the JSON downloaded from the portal ("Download JSON") or returned by POST /qc/evaluate.
--human  : CSV with one row per call: a `file_name` column, then one column per parameter (the parameter
           name or key, e.g. "Active Listening" or "active_listening") holding Good / Average / Bad / N/A.
           The portal's "Download QA labels (CSV)" button produces exactly this format from reviewed calls.

Reports per parameter: exact agreement, agreement within one step, quadratic-weighted Cohen's kappa,
and the direction of disagreement (is the AI stricter or more lenient?), plus total-score error and
pass/fail agreement. Run it on 30-50 calls a human has scored before trusting the AI's scores,
and again after every prompt / model / knowledge-base change.
"""
import argparse
import csv
import json
import sys
from collections import Counter

from scoring import compute_score

ORDER = {"Bad": 0, "Average": 1, "Good": 2}
TARGET_EXACT = 0.80
TARGET_KAPPA = 0.60


def _normalise_rating(value: str) -> str:
    v = (value or "").strip().lower()
    return {"good": "Good", "average": "Average", "avg": "Average", "bad": "Bad", "n/a": "N/A", "na": "N/A",
            "1": "Good", "0.5": "Average", "0": "Bad"}.get(v, "")


def weighted_kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Quadratic-weighted Cohen's kappa on the ordinal Bad < Average < Good scale (N/A pairs excluded)."""
    pairs = [(a, h) for a, h in pairs if a in ORDER and h in ORDER]
    n = len(pairs)
    if n == 0:
        return None
    k = len(ORDER)
    observed = [[0] * k for _ in range(k)]
    for a, h in pairs:
        observed[ORDER[a]][ORDER[h]] += 1
    rows = [sum(observed[i]) for i in range(k)]
    cols = [sum(observed[i][j] for i in range(k)) for j in range(k)]
    weight = lambda i, j: (i - j) ** 2 / (k - 1) ** 2
    num = sum(weight(i, j) * observed[i][j] for i in range(k) for j in range(k))
    den = sum(weight(i, j) * rows[i] * cols[j] / n for i in range(k) for j in range(k))
    return None if den == 0 else round(1 - num / den, 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", required=True)
    parser.add_argument("--human", required=True)
    parser.add_argument("--out", help="Optional CSV of every disagreement, for calibration meetings.")
    args = parser.parse_args()

    with open(args.report, encoding="utf-8") as f:
        report = json.load(f)
    cfg = report.get("config") or {}
    params = cfg.get("parameters") or []
    points = cfg.get("rating_points") or {"Good": 1.0, "Average": 0.5, "Bad": 0.0}
    na_policy = cfg.get("na_policy", "full_credit")
    pass_mark = cfg.get("pass_mark", 8)

    # AI ratings: always the ORIGINAL AI rating, even if the report contains QA overrides.
    ai = {c["file_name"]: {p["key"]: p.get("ai_rating", p["rating"]) for p in c["parameters"]} for c in report["calls"]}

    with open(args.human, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = {}
        for header in reader.fieldnames or []:
            h = header.strip().lower()
            for p in params:
                if h in (p["key"].lower(), p["name"].lower()):
                    columns[p["key"]] = header
        human = {row["file_name"].strip(): {k: _normalise_rating(row.get(col, "")) for k, col in columns.items()} for row in reader}

    missing_cols = [p["name"] for p in params if p["key"] not in columns]
    if missing_cols:
        print(f"Warning: human CSV has no column for: {', '.join(missing_cols)}")

    matched = sorted(set(ai) & set(human))
    if not matched:
        sys.exit("No file_name appears in both the report and the human CSV.")
    print(f"\nCalls compared: {len(matched)}  (AI only: {len(set(ai) - set(human))}, human only: {len(set(human) - set(ai))})\n")

    disagreements = []
    print(f"{'Parameter':<30}{'n':>4}{'Exact':>8}{'±1 step':>9}{'Kappa':>8}{'AI stricter':>13}{'AI lenient':>12}")
    print("-" * 84)
    for p in params:
        key = p["key"]
        if key not in columns:
            continue
        pairs = [(ai[f][key], human[f][key]) for f in matched if human[f].get(key)]
        if not pairs:
            continue
        exact = sum(a == h for a, h in pairs) / len(pairs)
        within = sum(a == h or (a in ORDER and h in ORDER and abs(ORDER[a] - ORDER[h]) <= 1) for a, h in pairs) / len(pairs)
        stricter = sum(a in ORDER and h in ORDER and ORDER[a] < ORDER[h] for a, h in pairs)
        lenient = sum(a in ORDER and h in ORDER and ORDER[a] > ORDER[h] for a, h in pairs)
        kappa = weighted_kappa(pairs)
        warn = "  <- below target" if exact < TARGET_EXACT or (kappa is not None and kappa < TARGET_KAPPA) else ""
        print(f"{p['name']:<30}{len(pairs):>4}{exact:>8.0%}{within:>9.0%}{kappa if kappa is not None else '-':>8}{stricter:>13}{lenient:>12}{warn}")
        for f in matched:
            if human[f].get(key) and ai[f][key] != human[f][key]:
                disagreements.append({"file_name": f, "parameter": p["name"], "ai": ai[f][key], "human": human[f][key]})

    keys = [p["key"] for p in params]
    errors, pass_agree = [], 0
    for f in matched:
        if not all(human[f].get(k) for k in keys):
            continue
        ai_score = compute_score([ai[f][k] for k in keys], points, na_policy)["score"]
        human_score = compute_score([human[f][k] for k in keys], points, na_policy)["score"]
        errors.append(ai_score - human_score)
        pass_agree += (ai_score >= pass_mark) == (human_score >= pass_mark)

    if errors:
        print(f"\nTotal score (calls with all {len(keys)} parameters rated: {len(errors)})")
        print(f"  Mean absolute error : {sum(abs(e) for e in errors) / len(errors):.2f} points out of 10")
        print(f"  Mean bias (AI-human): {sum(errors) / len(errors):+.2f}  (negative = AI scores lower)")
        print(f"  Pass/fail agreement : {pass_agree / len(errors):.0%}")

    if disagreements:
        common = Counter((d["parameter"], d["ai"], d["human"]) for d in disagreements).most_common(5)
        print("\nMost common disagreements (parameter: AI -> human):")
        for (name, a, h), count in common:
            print(f"  {count:>3}x  {name}: {a} -> {h}")
    print(f"\nTargets: exact agreement >= {TARGET_EXACT:.0%}, weighted kappa >= {TARGET_KAPPA}. "
          "Fix low parameters with calibration rulings in the knowledge base, then re-run.")

    if args.out and disagreements:
        with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["file_name", "parameter", "ai", "human"])
            writer.writeheader()
            writer.writerows(disagreements)
        print(f"Disagreements written to {args.out}")


if __name__ == "__main__":
    main()
