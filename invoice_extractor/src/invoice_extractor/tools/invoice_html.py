import base64
import re
import sys
from pathlib import Path

import requests
from markdownify import markdownify as md

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "glm-ocr:latest"
# MODEL = "glm-ocr:q8_0"

# GLM-OCR is prompt-limited (see the zai-org/GLM-OCR model card): it only
# reliably understands these official task prompts. Custom prose prompts
# silently degrade to unstructured plain text.
#   "Text Recognition:"  -> whole document as plain text
#   "Table Recognition:" -> tables in the document as HTML <table> markup
TEXT_PROMPT = "Text Recognition:"
TABLE_PROMPT = "Table Recognition:"


def generate(prompt: str, image_base64: str) -> str:
    """One /api/generate call. Returns the model's raw text.
    Shared with invoice_json.py, which reuses it for the JSON mode."""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "images": [image_base64],
        "stream": False,
        "options": {"temperature": 0},
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=300)
    response.raise_for_status()
    return response.json()["response"]


def _header_cells(table_html: str) -> list[str]:
    m = re.search(r"<thead[^>]*>(.*?)</thead>", table_html, flags=re.S)
    if not m:
        return []
    return [
        re.sub(r"<[^>]+>", "", c).strip()
        for c in re.findall(r"<th[^>]*>(.*?)</th>", m.group(1), flags=re.S)
    ]


def _merge_table(text: str, table_html: str) -> str:
    """Replace the plain-text version of the table with the HTML table."""
    lines = text.splitlines()
    cells = _header_cells(table_html)

    # Anchor on the line carrying the table header (e.g. "Description Quantity Rate Amount").
    start = None
    if cells:
        for i, line in enumerate(lines):
            if all(c and c in line for c in cells):
                start = i
                break

    if start is None:
        # Couldn't locate the table in the text (rare); keep both, no data loss.
        return "\n\n".join([text.rstrip(), table_html.strip()])

    ncols = max(len(cells), 2)
    end = start
    # Consume the contiguous body rows ("cell cell cell ..." lines).
    while (
        end + 1 < len(lines)
        and lines[end + 1].strip()
        and len(lines[end + 1].split()) >= ncols - 1
    ):
        end += 1

    after = lines[end + 1 :]
    # The model also renders totals into <tfoot>; drop plain-text lines that
    # are already inside the HTML table (keep blank lines intact).
    after = [ln for ln in after if not ln.strip() or ln.strip() not in table_html]

    out = "\n".join(lines[:start] + [table_html.strip()] + after)
    # Collapse the triple newlines left behind by dedup'd lines.
    return re.sub(r"\n{3,}", "\n\n", out)


def _html_table_to_md(match) -> str:
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", match.group(0), flags=re.S)
    grid = [
        [
            re.sub(r"<[^>]+>", "", c).strip()
            for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row, flags=re.S)
        ]
        for row in rows
    ]
    if not grid:
        return match.group(0)
    width = max(len(r) for r in grid)
    grid = [r + [""] * (width - len(r)) for r in grid]
    out = ["| " + " | ".join(grid[0]) + " |", "|" + "---|" * width]
    out += ["| " + " | ".join(r) + " |" for r in grid[1:]]
    return "\n".join(out)


def render_markdown(html: str) -> str:
    """markdownify doesn't understand <table>; convert HTML tables to
    Markdown pipe tables first, then markdownify the rest."""
    tables = re.sub(r"<table.*?</table>", _html_table_to_md, html, flags=re.S)
    return md(tables)


def extract_invoice(image_path) -> str:
    """Extract the full invoice: plain text + HTML tables."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    image_base64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")

    text = generate(TEXT_PROMPT, image_base64)
    table_html = generate(TABLE_PROMPT, image_base64)

    if "<table" not in table_html.lower():
        # Document has no (recognized) tables.
        return text

    return _merge_table(text, table_html)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python invoice_html.py invoice.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        html = extract_invoice(image_path)
        markdown = render_markdown(html)
        print("\n--- OCR RESULT ---\n")
        print(markdown)

        # Save extracted markdown
        output_path = Path(image_path).with_suffix(".md")
        output_path.write_text(markdown, encoding="utf-8")

        # Save extracted html (plain text + real HTML tables)
        output_path = Path(image_path).with_suffix(".html")
        output_path.write_text(html, encoding="utf-8")

        print(f"\nSaved to: {output_path}")

    except requests.RequestException as e:
        print(f"Ollama API error: {e}")

    except Exception as e:
        print(f"Error: {e}")
