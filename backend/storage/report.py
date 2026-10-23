"""
Export a ``SessionRecord`` as CSV or as a self-contained HTML report.

The HTML report has print styles, so "Print > Save as PDF" in a browser
gives a PDF without adding a PDF library. All text is HTML-escaped.
"""

from __future__ import annotations

import csv
import io
from html import escape
from typing import Optional

from backend.core.results import round_half_up
from backend.storage.records import SessionRecord

DISCLAIMER = (
    "Prototype practice aid. Scores are heuristic and have not been validated "
    "against human interviewers; they must not be used to make hiring decisions. "
    "Facial-expression estimates are not scored."
)


def _fmt(value: Optional[float]) -> str:
    return "" if value is None else str(round_half_up(value))


def to_csv(record: SessionRecord) -> str:
    """One row per question; blank cells mean 'not measured'."""
    criteria_keys: list[tuple[str, str]] = []
    for a in record.answers:
        for c in a.criteria:
            if (c.key, c.label) not in criteria_keys:
                criteria_keys.append((c.key, c.label))
    component_labels: list[str] = []
    for a in record.answers:
        for label in a.components:
            if label not in component_labels:
                component_labels.append(label)

    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(
        ["session_id", "created_at", "candidate", "role", "position", "question_id", "question",
         "status", "practice_score", "partial", "complete"]
        + component_labels
        + [f"rubric: {label}" for _key, label in criteria_keys]
        + ["words_per_minute", "fillers", "strengths", "improvements", "transcript"]
    )
    for a in record.answers:
        crit = {c.key: c.score for c in a.criteria}
        writer.writerow(
            [record.session_id, record.created_at, record.candidate or "", record.role,
             a.position, a.question_id, a.question_text, a.state, _fmt(a.overall),
             "yes" if a.is_partial and a.overall is not None else "",
             "yes" if a.complete else "no"]
            + [_fmt(a.components.get(label)) for label in component_labels]
            + [_fmt(crit.get(key)) for key, _label in criteria_keys]
            + [_fmt(a.wpm), "" if a.filler_count is None else a.filler_count,
               " | ".join(a.strengths), " | ".join(a.improvements), a.transcript or ""]
        )
    return out.getvalue()


_CSS = """
body{font-family:"Segoe UI",system-ui,sans-serif;color:#0d101a;margin:32px;max-width:900px}
h1{font-size:22px;margin:0}h2{font-size:17px;margin:28px 0 8px;border-bottom:2px solid #36c5ff;padding-bottom:4px}
.brand{letter-spacing:.2em;font-size:11px;color:#8b5cf6;font-weight:700}
.meta{color:#4a5568;font-size:13px;margin:6px 0 16px}.note{font-size:12px;color:#4a5568;background:#f3f4f8;padding:8px 10px;border-radius:6px}
table{border-collapse:collapse;width:100%;font-size:13px;margin:6px 0}td,th{border:1px solid #d6d9e3;padding:5px 8px;text-align:left;vertical-align:top}
th{background:#f3f4f8}.score{font-size:28px;font-weight:700;color:#1d4ed8}.na{color:#9aa6bd;font-style:italic}
ul{margin:4px 0 8px 18px;padding:0}li{margin:2px 0}
@media print{body{margin:12mm}h2{break-after:avoid}table{break-inside:avoid}}
"""


def to_html(record: SessionRecord) -> str:
    e = escape
    avg = record.average
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        f"<title>HUMAN NEXT interview report {e(record.session_id)}</title>",
        f"<style>{_CSS}</style></head><body>",
        "<div class='brand'>HUMAN NEXT · INTERVIEW REPORT</div>",
        f"<h1>{e(record.role)} practice interview</h1>",
        f"<div class='meta'>Session {e(record.session_id)} · {e(record.created_at)}"
        + (f" · {e(record.candidate)}" if record.candidate else "") + "</div>",
        f"<p class='note'>{e(DISCLAIMER)}</p>",
        "<h2>Summary</h2>",
    ]
    if avg is None:
        parts.append("<p class='na'>No question produced an overall practice score.</p>")
    else:
        parts.append(f"<div class='score'>{round_half_up(avg)}/100</div>"
                     f"<div class='meta'>Average practice score over {len(record.scored)} of "
                     f"{len(record.answers)} questions.</div>")

    for a in record.answers:
        parts.append(f"<h2>Question {a.position}: {e(a.question_text)}</h2>")
        parts.append(f"<div class='meta'>Status: {e(a.state)} — {e(a.message)}</div>")
        if a.overall is None:
            parts.append("<p class='na'>No overall score for this question.</p>")
        else:
            label = " (partial)" if a.is_partial else ""
            parts.append(f"<p><b>Practice score{label}: {round_half_up(a.overall)}/100</b></p>")
        for line in a.explanation:
            parts.append(f"<div class='meta'>{e(line)}</div>")
        if a.components:
            parts.append("<table><tr><th>Component</th><th>Score</th><th>Note</th></tr>")
            for label, value in a.components.items():
                cell = _fmt(value) or "<span class='na'>not measured</span>"
                parts.append(f"<tr><td>{e(label)}</td><td>{cell}</td>"
                             f"<td>{e(a.component_notes.get(label, ''))}</td></tr>")
            parts.append("</table>")
        if a.criteria:
            parts.append("<table><tr><th>Answer criterion</th><th>Score</th><th>Weight</th>"
                         "<th>Why</th></tr>")
            for c in a.criteria:
                parts.append(f"<tr><td>{e(c.label)}</td><td>{_fmt(c.score)}</td>"
                             f"<td>{c.weight:.0%}</td><td>{e(c.explanation)}</td></tr>")
            parts.append("</table>")
        for title, items in (("Strengths", a.strengths), ("To improve", a.improvements),
                             ("Suggestions", a.suggestions)):
            if items:
                parts.append(f"<b>{title}</b><ul>" + "".join(f"<li>{e(i)}</li>" for i in items) + "</ul>")
        if a.transcript:
            parts.append(f"<b>Transcript</b><p>{e(a.transcript)}</p>")
    parts.append("</body></html>")
    return "".join(parts)
