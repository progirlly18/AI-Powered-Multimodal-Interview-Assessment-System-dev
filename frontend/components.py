"""
Reusable HUMAN NEXT UI pieces as small HTML fragments.

Every piece of text that reaches these functions is HTML-escaped here, so
names, transcripts and model labels can never inject markup. Numbers are
formatted here too. Functions return HTML strings; ``show`` renders one.
"""

from __future__ import annotations

from html import escape
from typing import Iterable, Optional, Sequence

import streamlit as st

from backend.core.results import round_half_up
from frontend.theme import BRAND


def show(html: str, container=None) -> None:
    (container or st).markdown(html, unsafe_allow_html=True)


def chip(text: str, kind: str = "", dot: bool = False) -> str:
    """kind: live, rec, ok, warn, err, measured, model, na (or '')."""
    d = "<span class='dot'></span>" if dot else ""
    return f"<span class='hn-chip {escape(kind)}'>{d}{escape(text)}</span>"


def brand_bar(chips: Sequence[str] = ()) -> str:
    return (
        "<div class='hn-bar'><div class='hn-logo'><div class='hn-logo-mark'></div>"
        f"{escape(BRAND)}</div><div class='hn-bar-right'>{''.join(chips)}</div></div>"
    )


def card(body_html: str, label: Optional[str] = None, title: Optional[str] = None,
         glow: bool = False, cls: str = "") -> str:
    """Glass card. ``cls`` adds CSS classes, e.g. 'eq' for equal-height rows."""
    head = f"<div class='hn-label'>{escape(label)}</div>" if label else ""
    head += f"<h4>{escape(title)}</h4>" if title else ""
    classes = " ".join(c for c in ("hn-card", "glow" if glow else "", escape(cls)) if c)
    return f"<div class='{classes}'>{head}{body_html}</div>"


def para(text: str) -> str:
    return f"<p>{escape(text)}</p>"


def bullet_list(items: Iterable[str]) -> str:
    items = list(items)
    if not items:
        return ""
    return "<ul class='hn-list'>" + "".join(f"<li>{escape(i)}</li>" for i in items) + "</ul>"


def stepper(labels: Sequence[str], current: int, compact: bool = False) -> str:
    """Progress steps; ``current`` -1 means none started. ``compact`` = one tight row."""
    cells = []
    for i, label in enumerate(labels):
        state = "done" if i < current else "now" if i == current else ""
        prefix = f"{i + 1}" if compact else f"STEP {i + 1}"
        cells.append(f"<div class='hn-step {state}'><b>{prefix}</b>{escape(label)}</div>")
    return f"<div class='hn-steps{' compact' if compact else ''}'>{''.join(cells)}</div>"


def bar(label: str, value: Optional[float], note: str = "", right: Optional[str] = None,
        style: str = "") -> str:
    """Labelled 0-100 bar. ``value`` None renders a hatched 'not measured' track."""
    if value is None:
        right_text = right or "Not measured"
        track = "<div class='hn-track na'></div>"
    else:
        v = max(0.0, min(100.0, float(value)))
        right_text = right or f"{round_half_up(v)}"
        track = (f"<div class='hn-track'><div class='hn-fill {escape(style)}' "
                 f"style='width:{v:.1f}%'></div></div>")
    note_html = f"<div class='hn-row-note'>{escape(note)}</div>" if note else ""
    return (f"<div class='hn-row'><div class='hn-row-head'><span>{escape(label)}</span>"
            f"<span>{escape(right_text)}</span></div>{track}{note_html}</div>")


def score_ring(value: Optional[float], caption: str = "PRACTICE SCORE") -> str:
    if value is None:
        return ("<div class='hn-ring na'><div><b>—</b><small>WITHHELD</small></div></div>")
    v = max(0.0, min(100.0, float(value)))
    return (f"<div class='hn-ring' style='--v:{v:.1f}'><div><b>{round_half_up(v)}</b>"
            f"<small>{escape(caption)}</small></div></div>")


def legend() -> str:
    return ("<div class='hn-legend'>"
            + chip("Measured", "measured") + chip("Model estimate", "model")
            + chip("Not available", "na") + "</div>")


def stage_placeholder(title: str, text: str) -> str:
    return (f"<div class='hn-stage'><div><div class='ring'></div>"
            f"<h4>{escape(title)}</h4><p>{escape(text)}</p></div></div>")
