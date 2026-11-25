"""Deterministic quality scoring for a preprocessing attempt.

GLM-OCR exposes no confidence score — it either fills the JSON skeleton or
it returns garbage. "Good enough" therefore has to come from what came back,
not from how the model felt while doing it.

The score is in [0, 1]:

    0.30 * payload_ok                          # zero "ERROR" markers
    0.30 * meaningful_fields_present / 6       # required headline fields
    0.20 * min(1, line_count / 8)              # line items extracted
    0.20 * (1 - min(1, error_count / 3))       # marker penalty

The line-items sum vs. invoice total is deliberately NOT in the score —
invoices legitimately differ (rounding, discounts, tax on top of the
subtotal). It is surfaced as an informational flag only.

This module returns score + metrics; the *status* (SUCCESS / RETRYABLE /
MANUAL_REVIEW) is decided by the tool layer that owns the attempt policy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

#: Attempt outcomes. Threshold: score >= SUCCESS_THRESHOLD is a pass.
#: OUTCOME_ERROR is the tool layer's mark for a hard failure (no model
#: round-trip at all) — payload is None, score is the 0.0 floor.
OUTCOME_SUCCESS = "SUCCESS"
OUTCOME_RETRYABLE = "RETRYABLE"
OUTCOME_MANUAL_REVIEW = "MANUAL_REVIEW"
OUTCOME_ERROR = "ERROR"
SUCCESS_THRESHOLD = 0.5

#: The six headline fields a usable invoice needs.
_MEANINGFUL_FIELDS = (
    "invoice_number",
    "invoice_date",
    "bill_from",
    "bill_to",
    "line_items",
    "total",
)


@dataclass(frozen=True)
class QualityReport:
    """Score + the metrics behind it, for one attempt.

    ``to_metrics_dict()`` is what gets persisted — the score rides along in
    its own column, so the JSON blob stays the explainable part."""

    score: float
    payload_ok: bool
    present_fields: tuple[str, ...]
    missing_fields: tuple[str, ...]
    line_count: int
    error_count: int
    sum_of_lines: float | None
    invoice_total: float | None
    sum_mismatches_total: bool

    def to_metrics_dict(self) -> dict:
        d = asdict(self)
        d.pop("score")
        return d


def _payload_present(payload: dict | None, name: str) -> bool:
    """A field counts when it exists, is not an error marker, and (for lists)
    is non-empty. ``line_items: []`` is as bad as no line items."""
    if payload is None:
        return False
    value = payload.get(name)
    if value is None or value == "ERROR":
        return False
    if name == "line_items":
        return isinstance(value, list) and len(value) > 0
    return True


def _num(value) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def assess(payload: dict | None, failed: list[str]) -> QualityReport:
    """Score one extraction result. Pure — never raises on any payload shape.

    ``payload is None`` (model unreachable / hard failure) is the zero-score
    floor: nothing was extracted at all."""
    if payload is None:
        return QualityReport(
            score=0.0,
            payload_ok=False,
            present_fields=(),
            missing_fields=_MEANINGFUL_FIELDS,
            line_count=0,
            error_count=len(failed),
            sum_of_lines=None,
            invoice_total=None,
            sum_mismatches_total=False,
        )

    present = tuple(name for name in _MEANINGFUL_FIELDS if _payload_present(payload, name))
    missing = tuple(name for name in _MEANINGFUL_FIELDS if name not in present)

    line_items = payload.get("line_items")
    line_count = len(line_items) if isinstance(line_items, list) else 0
    error_count = len(failed)

    line_amounts = [
        _num(item.get("amount"))
        for item in (line_items if isinstance(line_items, list) else [])
        if isinstance(item, dict)
    ]
    sum_of_lines = round(sum(a for a in line_amounts if a is not None), 2) if any(a is not None for a in line_amounts) else None
    invoice_total = _num(payload.get("total"))
    # Rounding tolerance: 1% of the total or 2 cents — whichever is larger.
    sum_mismatch = (
        sum_of_lines is not None
        and invoice_total is not None
        and abs(sum_of_lines - invoice_total) > max(0.02, 0.01 * abs(invoice_total))
    )

    payload_ok = error_count == 0
    fields_part = len(present) / len(_MEANINGFUL_FIELDS)
    lines_part = min(1.0, line_count / 8.0)
    error_part = 1.0 - min(1.0, error_count / 3.0)
    score = round(0.30 * (1.0 if payload_ok else 0.0) + 0.30 * fields_part + 0.20 * lines_part + 0.20 * error_part, 4)

    return QualityReport(
        score=score,
        payload_ok=payload_ok,
        present_fields=present,
        missing_fields=missing,
        line_count=line_count,
        error_count=error_count,
        sum_of_lines=sum_of_lines,
        invoice_total=invoice_total,
        sum_mismatches_total=sum_mismatch,
    )
