"""Structured invoice extraction with GLM-OCR's information-extraction mode.

GLM-OCR is prompt-limited: for information extraction the prompt must be the
fixed instruction from the model card followed by a JSON skeleton that the
model fills in. The skeleton is generated from the Pydantic schema below, so
the prompt shape and the validation shape can never drift apart.

This module is the extraction core for the MCP server (tools/main.py); the
`__main__` block at the bottom is a standalone CLI for humans.
"""

import base64
import json
import re
import sys
from pathlib import Path
from typing import Any, Union, get_args, get_origin

import requests
from pydantic import BaseModel, ValidationError, field_validator
from types import UnionType

try:  # Package mode: imported as ``invoice_extractor.tools.invoice_json``.
    from .invoice_html import generate
except ImportError:  # Script mode: ``python invoice_json.py`` from the tools dir.
    from invoice_html import generate

# Exact instruction from the zai-org/GLM-OCR model card ("Prompt Limited"
# section). The model only recognizes this verbatim.
EXTRACTION_INSTRUCTION = "请按下列JSON格式输出图中信息:"


# --- schema ---------------------------------------------------------------

class Party(BaseModel):
    name: str
    address: str | None = None
    phone: str | None = None
    email: str | None = None
    contact: str | None = None


class LineItem(BaseModel):
    description: str
    quantity: float
    unit: str | None = None
    unit_price: float
    amount: float

    @field_validator("quantity", "unit_price", "amount", mode="before")
    @classmethod
    def _coerce_amount(cls, v: Any) -> Any:
        # Be tolerant of "$150.00" / "1,234.56" coming back as strings.
        if isinstance(v, str):
            v = v.strip().replace("$", "").replace(",", "")
            try:
                return float(v)
            except ValueError:
                return v
        return v


class Invoice(BaseModel):
    invoice_number: str
    invoice_date: str | None = None
    due_date: str | None = None
    currency: str | None = None
    bill_from: Party | None = None
    bill_to: Party | None = None
    line_items: list[LineItem]
    subtotal: float | None = None
    tax_rate: float | None = None  # percent, e.g. 8.5
    tax_amount: float | None = None
    total: float | None = None
    payment_terms: str | None = None
    payment_methods: list[str] | None = None
    notes: str | None = None

    @field_validator("subtotal", "tax_rate", "tax_amount", "total", mode="before")
    @classmethod
    def _coerce_amount(cls, v: Any) -> Any:
        if isinstance(v, str):
            v = v.strip().replace("$", "").replace(",", "")
            try:
                return float(v)
            except ValueError:
                return v
        return v


# --- skeleton + prompt ------------------------------------------------------

def build_skeleton(model: type[BaseModel]) -> dict:
    """JSON skeleton in the shape of `model`, every leaf an empty string,
    matching the model card's example format."""

    def placeholder(t: Any) -> Any:
        args = get_args(t)
        origin = get_origin(t)
        if origin in (Union, UnionType):  # Optional[X] -> X
            non_none = [a for a in args if a is not type(None)]
            return placeholder(non_none[0]) if non_none else ""
        if origin is list:
            return [placeholder(args[0]) if args else ""]
        if isinstance(t, type) and issubclass(t, BaseModel):
            return {n: placeholder(f.annotation) for n, f in t.model_fields.items()}
        return ""

    return placeholder(model)


def extraction_prompt(schema_model: type[BaseModel] = Invoice) -> str:
    skeleton = json.dumps(
        build_skeleton(schema_model), indent=4, ensure_ascii=False
    )
    return f"{EXTRACTION_INSTRUCTION}\n{skeleton}"


# --- extraction + validation -----------------------------------------------

def parse_payload(raw: str) -> dict:
    """Extract the first JSON object from the model's raw output.

    Tolerates markdown fences and surrounding chatter. Raises ValueError when
    there is no JSON object at all (model degenerated to prose)."""
    text = (raw or "").strip()
    # Tolerate the model wrapping the JSON in markdown fences.
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("model output contains no JSON object")
    data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("model output is not a JSON object")
    return data


def extract_json(raw: str, schema_model: type[BaseModel] = Invoice) -> BaseModel:
    """Parse the model's raw output and validate it strictly."""
    return schema_model.model_validate(parse_payload(raw))


def extract_invoice_json(image_path) -> BaseModel:
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    image_base64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    raw = generate(extraction_prompt(), image_base64)
    return extract_json(raw, Invoice)


def extract_invoice_lenient(image_path) -> tuple[dict, list[str]]:
    """Same model round-trip as extract_invoice_json, never raises on
    schema mismatch — failing values come back as "ERROR" markers and the
    failing paths in the second return value."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    image_base64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    raw = generate(extraction_prompt(), image_base64)
    return extract_lenient(raw, Invoice)


# --- lenient gate (MCP server) ----------------------------------------------
# The server must always produce a .json file, even when validation fails:
# failing values get replaced with "ERROR" and the caller is told exactly
# which fields failed. That is what extract_lenient below provides.

def apply_error_markers(data: dict, errors: list[dict]) -> dict:
    """In-place: replace each field the validator rejected with "ERROR".

    `errors` is a pydantic ValidationError.errors() list. Each error's `loc`
    points at the failing field (possibly nested, e.g. ("line_items", 0,
    "amount")). If the payload is missing the failing container entirely,
    nothing can be marked there — the caller still sees the field in the
    failure list the server builds from the same errors."""
    for err in errors:
        loc = err["loc"]
        if not loc:
            data.clear()
            data["ERROR"] = "ERROR"
            continue
        node: Any = data
        for key in loc[:-1]:
            if isinstance(node, dict):
                node = node.get(key)
            elif isinstance(node, list) and isinstance(key, int) and 0 <= key < len(node):
                node = node[key]
            else:
                node = None
                break
        last = loc[-1]
        if isinstance(node, dict):
            node[last] = "ERROR"
        elif isinstance(node, list) and isinstance(last, int) and 0 <= last < len(node):
            node[last] = "ERROR"
    return data


def failed_field_paths(errors: list[dict]) -> list[str]:
    """Unique, stable 'a.b.0.c' paths of every field the validator rejected."""
    paths = {".".join(str(part) for part in err["loc"]) for err in errors}
    return sorted(paths)


def extract_lenient(
    raw: str, schema_model: type[BaseModel] = Invoice
) -> tuple[dict, list[str]]:
    """Validate against the schema, but never refuse to hand back a payload.

    Returns ``(payload, failed)``:
      - success: payload is the *validated* model dump (normalized types),
        ``failed`` is empty.
      - validation failure: payload is the raw data with each failing field
        replaced by ``"ERROR"``; ``failed`` lists the failing paths.
      - no JSON object at all: every top-level field is a failure, so payload
        is the all-``"ERROR"`` top-level shape and ``failed`` is every field.

    Only raises on malformed JSON that a dict-based payload cannot represent
    (json.JSONDecodeError on a truncated object) — the server maps that onto
    the all-ERROR skeleton as well."""
    try:
        data = parse_payload(raw)
    except (ValueError, json.JSONDecodeError):
        return (
            {name: "ERROR" for name in schema_model.model_fields},
            list(schema_model.model_fields),
        )
    try:
        model = schema_model.model_validate(data)
    except ValidationError as exc:
        apply_error_markers(data, exc.errors())
        return data, failed_field_paths(exc.errors())
    return model.model_dump(), []


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python invoice_json.py assets/invoice.jpg")
        sys.exit(1)

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        print(f"Error: file not found: {image_path}")
        sys.exit(1)

    try:
        invoice = extract_invoice_json(image_path)
    except requests.RequestException as e:
        print(f"Ollama API error: {e}", file=sys.stderr)
        sys.exit(2)
    except (ValueError, json.JSONDecodeError, ValidationError) as e:
        # Print the failure, keep stdout clean: a pipeline can catch exit 1
        # and fall back to invoice_html.py (HTML table mode).
        print(f"JSON extraction failed: {e}", file=sys.stderr)
        if image_path.exists():
            print(
                "Falling back is up to you: python invoice_html.py "
                f"{image_path}",
                file=sys.stderr,
            )
        sys.exit(1)

    output = invoice.model_dump_json(indent=2)
    saved = image_path.with_suffix(".json")
    saved.write_text(output + "\n", encoding="utf-8")

    # stdout: pure JSON (pipeable); bookkeeping on stderr.
    print(output)
    print(f"\nSaved to: {saved}", file=sys.stderr)
