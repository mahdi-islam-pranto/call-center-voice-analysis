"""
Builds the downloadable Excel QC report from a BatchReport dict that has been
passed through scoring.apply_overrides (so it carries AI and final ratings).

Sheets:
  Summary            - batch KPIs and agent leaderboard
  Scorecard          - one row per call: the 10 final ratings, score /10, band, flags
  Parameter Detail   - one row per (call, parameter): AI vs final rating, evidence, checklist, coaching tip
  Agent Coaching     - coaching brief per agent
  Review Flags       - why each call was sent to human review
  Agent Claims       - every factual claim the agent made and its knowledge-base check
  Transcripts        - timestamped transcript of every call
"""
from datetime import datetime
from io import BytesIO
from typing import Any, Optional

from openpyxl import Workbook
from openpyxl.formatting.rule import CellIsRule, ColorScaleRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

HEADER_FILL = PatternFill("solid", start_color="0B3D5C")
HEADER_FONT = Font(bold=True, color="FFFFFF")
LABEL_FONT = Font(bold=True)
TITLE_FONT = Font(bold=True, size=16, color="0B3D5C")
SECTION_FONT = Font(bold=True, size=12, color="0B3D5C")
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
WRAP = Alignment(wrap_text=True, vertical="top")
CENTER = Alignment(horizontal="center", vertical="top", wrap_text=True)

RATING_FILLS = {
    "Good": PatternFill("solid", start_color="C6EFCE"),
    "Average": PatternFill("solid", start_color="FFEB9C"),
    "Bad": PatternFill("solid", start_color="FFC7CE"),
    "N/A": PatternFill("solid", start_color="E7E6E6"),
}
SEVERITY_FILLS = {
    "high": PatternFill("solid", start_color="FFC7CE"),
    "medium": PatternFill("solid", start_color="FFEB9C"),
    "low": PatternFill("solid", start_color="DDEBF7"),
}


def _set(ws, row: int, col: int, value: Any, **style):
    """Writes a cell; text starting with '=' is kept as plain text so model output / file names never run as formulas."""
    cell = ws.cell(row=row, column=col, value=value)
    if isinstance(value, str) and value.startswith("="):
        cell.data_type = "s"
    for key, val in style.items():
        setattr(cell, key, val)
    return cell


def _table(ws, header_row: int, headers: list[str], rows: list[list[Any]], widths: list[int], freeze: Optional[str] = None):
    for col, text in enumerate(headers, start=1):
        _set(ws, header_row, col, text, font=HEADER_FONT, fill=HEADER_FILL, alignment=CENTER, border=BORDER)
    for r, values in enumerate(rows, start=header_row + 1):
        for c, value in enumerate(values, start=1):
            cell = _set(ws, r, c, value, alignment=WRAP, border=BORDER)
            if isinstance(value, str) and value in RATING_FILLS:
                cell.fill = RATING_FILLS[value]
                cell.alignment = CENTER
    for col, width in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(col)].width = width
    ws.row_dimensions[header_row].height = 32
    last_row = header_row + max(len(rows), 1)
    ws.auto_filter.ref = f"A{header_row}:{get_column_letter(len(headers))}{last_row}"
    if freeze:
        ws.freeze_panes = freeze


def _score_scale(ws, cell_range: str, maximum: float):
    ws.conditional_formatting.add(cell_range, ColorScaleRule(
        start_type="num", start_value=0, start_color="F8696B",
        mid_type="num", mid_value=maximum * 0.7, mid_color="FFEB84",
        end_type="num", end_value=maximum, end_color="63BE7B",
    ))


def _bullets(items: Optional[list[str]]) -> str:
    return "\n".join(f"• {i}" for i in items or []) or "-"


def _duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"
    seconds = int(round(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def _summary_sheet(ws, report: dict, generated_at: datetime):
    ws.title = "Summary"
    calls = report.get("calls", [])
    params = (report.get("config") or {}).get("parameters", [])
    _set(ws, 1, 1, f"{report.get('brand_name', '')} Call QC Report", font=TITLE_FONT)

    info = [
        ("Generated at", generated_at.strftime("%Y-%m-%d %H:%M")),
        ("Calls evaluated", len(calls)),
        ("Team average score", f"{report.get('team_average_score')} / 10"),
        ("Pass rate", f"{report.get('team_pass_rate')}%  (pass mark {(report.get('config') or {}).get('pass_mark')} / 10, no critical error)"),
        ("Calls needing review", sum(c.get("needs_review", False) for c in calls)),
        ("Calls reviewed by QA", sum(c.get("reviewed", False) for c in calls)),
        ("Critical errors", sum(len(c.get("critical_errors") or []) for c in calls)),
        ("Could not evaluate", "\n".join(f"{k}: {v}" for k, v in (report.get("failed") or {}).items()) or "None"),
        ("Models", ", ".join(f"{k}: {v}" for k, v in (report.get("models") or {}).items())),
    ]
    for i, (label, value) in enumerate(info, start=3):
        _set(ws, i, 1, label, font=LABEL_FONT)
        _set(ws, i, 2, value, alignment=Alignment(horizontal="left", vertical="top", wrap_text=True))
        ws.merge_cells(start_row=i, start_column=2, end_row=i, end_column=10)
        ws.row_dimensions[i].height = max(15, 15 * (str(value).count("\n") + 1))

    start = len(info) + 5
    _set(ws, start - 1, 1, "Agent leaderboard (parameter columns = average points, 1.0 = always Good)", font=SECTION_FONT)
    headers = ["Agent", "Calls", "Avg Score /10", "Pass Rate %", "Needs Review", "Critical Errors", *[p["name"] for p in params]]
    rows = []
    for a in report.get("agents", []):
        rows.append([
            a["agent_name"], a["calls"], a["average_score"], a["pass_rate"], a["calls_needing_review"], a["critical_errors"],
            *[pa.get("average_points") if pa.get("average_points") is not None else "N/A" for pa in a["parameter_averages"]],
        ])
    _table(ws, start, headers, rows, [26, 8, 12, 11, 11, 11] + [14] * len(params))
    if rows:
        end = start + len(rows)
        _score_scale(ws, f"C{start + 1}:C{end}", 10)
        if params:
            _score_scale(ws, f"G{start + 1}:{get_column_letter(6 + len(params))}{end}", 1)


def _scorecard_sheet(ws, report: dict):
    params = (report.get("config") or {}).get("parameters", [])
    headers = ["#", "Call ID", "File", "Agent", "Direction", "Category", "Duration",
               *[p["name"] for p in params],
               "Score /10", "AI Score /10", "Band", "Passed", "Critical Errors", "Needs Review",
               "QA Overrides", "Reviewer Comment", "Call Summary"]
    rows = []
    for i, c in enumerate(report.get("calls", []), start=1):
        f = c["forensics"]
        critical = "; ".join(e["error_key"] for e in c.get("critical_errors") or [])
        if c.get("critical_dismissed"):
            critical += " (dismissed by QA)"
        rows.append([
            i, c["call_id"], c["file_name"], c["agent_name"], f["call_direction"], f["call_category"].replace("_", " "),
            _duration(c.get("duration_seconds")),
            *[p.get("final_rating", p["rating"]) or "-" for p in c["parameters"]],
            c["score"], c.get("ai_score", c["score"]), c["band"], "Yes" if c["passed"] else "No",
            critical or "-", "Yes" if c["needs_review"] else "No", c.get("overrides", 0),
            c.get("reviewer_comment") or "", c["call_summary"],
        ])
    widths = [5, 8, 30, 18, 10, 18, 9] + [13] * len(params) + [10, 10, 16, 8, 20, 9, 9, 30, 70]
    _table(ws, 1, headers, rows, widths, freeze="E2")
    if rows:
        score_col = get_column_letter(8 + len(params))
        _score_scale(ws, f"{score_col}2:{score_col}{len(rows) + 1}", 10)
        passed_col = get_column_letter(11 + len(params))
        ws.conditional_formatting.add(f"{passed_col}2:{passed_col}{len(rows) + 1}",
                                      CellIsRule(operator="equal", formula=['"No"'], fill=RATING_FILLS["Bad"]))


def _detail_sheet(ws, report: dict):
    headers = ["Call ID", "File", "Agent", "Parameter", "AI Rating", "Final Rating", "Confidence",
               "Justification", "Evidence (✓ = found in transcript)", "Checklist", "Coaching Tip"]
    rows = []
    for c in report.get("calls", []):
        for p in c["parameters"]:
            evidence = "\n".join(f"{'✓' if e['verified'] else '✗'} [{e['t']}] {e['speaker']}: {e['quote']}" for e in p["evidence"])
            checklist = "\n".join(f"{i['item_id']} {i['status']}: {i['note']}" for i in p["checklist"])
            rows.append([
                c["call_id"], c["file_name"], c["agent_name"], p["name"], p.get("ai_rating", p["rating"]) or "-",
                p.get("final_rating", p["rating"]) or "-", p["confidence"], p["justification"], evidence or "-",
                checklist, p.get("coaching_tip") or "",
            ])
    _table(ws, 1, headers, rows, [8, 26, 16, 22, 10, 10, 10, 55, 60, 60, 45], freeze="E2")


def _coaching_sheet(ws, report: dict):
    headers = ["Agent", "Calls", "Avg Score /10", "Weakest Parameters", "Performance Summary",
               "Consistent Strengths", "Recurring Issues", "Coaching Plan"]
    rows = []
    calls = report.get("calls", [])
    for a in report.get("agents", []):
        weakest = sorted((pa for pa in a["parameter_averages"] if pa.get("average_points") is not None),
                         key=lambda pa: pa["average_points"])[:3]
        weakest_text = "\n".join(f"{pa['name']} ({pa['average_points']})" for pa in weakest)
        coaching = a.get("coaching")
        if coaching:
            rows.append([a["agent_name"], a["calls"], a["average_score"], weakest_text, coaching["performance_summary"],
                         _bullets(coaching["consistent_strengths"]), _bullets(coaching["recurring_issues"]),
                         "\n".join(f"{i}. {s}" for i, s in enumerate(coaching["coaching_plan"], 1))])
        else:
            agent_calls = [c for c in calls if c["agent_name"] == a["agent_name"]]
            rows.append([a["agent_name"], a["calls"], a["average_score"], weakest_text,
                         " ".join(c["call_summary"] for c in agent_calls),
                         _bullets([s for c in agent_calls for s in c["what_went_well"]]),
                         _bullets([s for c in agent_calls for s in c["improvement_areas"]]),
                         "See the coaching tips in Parameter Detail (one call only, no cross-call brief)."])
    _table(ws, 1, headers, rows, [22, 7, 10, 28, 60, 50, 50, 70], freeze="B2")


def _flags_sheet(ws, report: dict):
    rows = [
        [c["call_id"], c["file_name"], c["agent_name"], f["severity"], f["reason"], "Yes" if c.get("reviewed") else "No"]
        for c in report.get("calls", []) for f in c["review_flags"]
    ]
    _table(ws, 1, ["Call ID", "File", "Agent", "Severity", "Reason", "Reviewed by QA"], rows, [8, 30, 18, 10, 100, 10], freeze="A2")
    for r, row in enumerate(rows, start=2):
        ws.cell(row=r, column=4).fill = SEVERITY_FILLS.get(row[3], PatternFill())


def _claims_sheet(ws, report: dict):
    rows = []
    for c in report.get("calls", []):
        checks = {cc["claim_index"]: cc for cc in c.get("claim_checks") or []}
        for i, claim in enumerate(c["forensics"].get("agent_claims") or []):
            check = checks.get(i, {})
            rows.append([c["call_id"], c["file_name"], c["agent_name"], claim["t"], claim["topic"].replace("_", " "),
                         claim["claim"], (check.get("verdict") or "not checked").replace("_", " "), check.get("note", "")])
    _table(ws, 1, ["Call ID", "File", "Agent", "Time", "Topic", "Claim", "Knowledge-Base Check", "Note"],
           rows, [8, 26, 16, 7, 16, 60, 20, 55], freeze="A2")
    for r, row in enumerate(rows, start=2):
        if row[6] in ("contradicts knowledge base", "self contradiction"):
            ws.cell(row=r, column=7).fill = RATING_FILLS["Bad"]


def _transcripts_sheet(ws, report: dict):
    rows = [
        [c["call_id"], c["file_name"], t["t"], t["speaker"], t["text"]]
        for c in report.get("calls", []) for t in c["forensics"]["transcript"]
    ]
    _table(ws, 1, ["Call ID", "File", "Time", "Speaker", "Text"], rows, [8, 26, 7, 10, 110], freeze="A2")


def build_excel_report(report: dict, generated_at: Optional[datetime] = None) -> bytes:
    """Returns .xlsx bytes for a report dict produced by scoring.apply_overrides()."""
    wb = Workbook()
    _summary_sheet(wb.active, report, generated_at or datetime.now())
    _scorecard_sheet(wb.create_sheet("Scorecard"), report)
    _detail_sheet(wb.create_sheet("Parameter Detail"), report)
    _coaching_sheet(wb.create_sheet("Agent Coaching"), report)
    _flags_sheet(wb.create_sheet("Review Flags"), report)
    _claims_sheet(wb.create_sheet("Agent Claims"), report)
    _transcripts_sheet(wb.create_sheet("Transcripts"), report)

    buffer = BytesIO()
    wb.save(buffer)
    return buffer.getvalue()
