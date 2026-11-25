# Preprocessing retry loop — implementation contract

OpenCV preprocessing runs **before every extraction attempt**, and the loop is
agent-driven: extract, score, retry with different knobs (at most 4 attempts).
Everything in this file is what the code actually does — the original wish
list it grew out of lives in `preprocessing-original-spec.md`.

Status: implemented and green — `tests/test_offline.py` 182/182, plus a live
smoke run over three photographed invoices (score 0.9, first-attempt finalize).

## What happens on each attempt

1. Preprocess the source image with the current knob set (pipeline below).
2. Run the extraction passes (structured JSON + HTML/Markdown) on the
   enhanced bytes.
3. Score the result **deterministically** (below) — no model confidence.
4. Record the attempt and outcome in `preprocessing_attempts`.
5. Finalize the cycle when the score ≥ `SUCCESS_THRESHOLD` (0.5), when this
   was attempt 4, or when an earlier cycle already kept an equal-or-better
   best. Otherwise keep the cycle open and hand the agent a `hint`.

## MCP tools

| Tool | Signature | Notes |
|---|---|---|
| `list_invoices` | `() -> {"images": [full paths]}` | Recursive; includes per-day date dirs. |
| `extract_invoice_from_image` | `(image_path: str, changes: dict \| None = None)` | Starts a **fresh cycle** (attempt 1). `changes` is an optional pre-tuned start; `None`/`{}` = defaults. |
| `retry_preprocessing` | `(image_path: str, changes: dict)` | Attempts 2–4 of an **open** cycle. `changes` is a non-empty cumulative delta over the *last* attempt's knobs — knobs you omit keep their last value. |
| `get_attempt_history` | `(image_path: str, max_attempts: int = 20)` | Latest cycle, ordered by attempt. |

All tools are sync. Refusal order in `retry_preprocessing`
(cycle-state checks first, **before** file existence — a finalized source was
already moved out, and the refusal must say the budget is gone, not "file
missing"):

- empty `changes` → `ValueError("retry_changes must set at least one knob")`
- 4 attempts recorded in latest cycle → `RuntimeError` ("the budget is
  exhausted…") and `cycle_is_closed` (a `best = 1` row) — start fresh with
  `extract_invoice_from_image`
- image missing → `FileNotFoundError`
- undecodable image → `ValueError` from `engine.load_image`

## Pipeline (fixed order, every attempt)

1. **Downscale** — long edge capped at 2048 px.
2. **Shadow removal** — divide by a heavy Gaussian (σ=50) estimate of the
   illumination field; dark corners on phone shots stop eating local contrast.
3. **CLAHE** — on the LAB lightness channel (color-safe); the agent's two main
   knobs.
4. **Unsharp-mask** — optional; amplifies sensor noise, off by default, capped
   at 0.8.
5. **Deskew** — rotate the dominant sheet back to axis-aligned, small angles
   only (≤ 10°, floor 0.25° below which rotation adds artifacts; sideways
   photos are left to the model).

Perspective correction (the `order_points` four-point warp from the original
spec) is deliberately **not** implemented — contour-based corner finding is
the most failure-prone part of that list, and modern OCR/VLM models handle
perspective themselves fine. Revisit only if you need flat scans for
downstream non-OCR work.

The engine is pure (ndarray in/out, no disk). `encode_image` re-encodes per
the source extension (JPEG q92 / PNG low) for both the model payload and the
attempt-history BLOB, so every recorded attempt is a self-contained re-runnable
image.

## Knobs

| Knob | Range | Default | Meaning |
|---|---|---|---|
| `clahe_clip` | 1.0–4.0 | 2.0 | CLAHE clip limit — higher = more aggressive local contrast |
| `clahe_tile` | 4–16 int | 8 | CLAHE tile grid — small = local, large = more global contrast |
| `sharpen_strength` | 0.0–0.8 | 0.0 | unsharp-mask amount; 0 = off |

Every construction path (`PreprocessConfig`, `from_dict`, `with_overrides`)
validates at the boundary, and bad values raise `ValueError` naming knob,
value, and the allowed range:

```
clahe_clip=9.0 outside allowed range [1.0, 4.0]
clahe_tile must be an integer grid size, got '8'
unknown preprocessing knob(s) ['bogus']; allowed: ['clahe_clip', 'clahe_tile', 'sharpen_strength']
```

Integral floats are coerced (`8.0` → `8`); strings and bools are rejected.
LLMs send `8.0` and `"8"` for tiles all the time — this gate keeps it boring.

## Scoring

Deterministic function of the JSON payload and its failed fields
(`preprocessing/quality.assess`):

```
score = 0.30 · (no ERROR fields)
      + 0.30 · (present / 6)          # vendor, currency, amount,
                                      # lines>0, line-sum, total
      + 0.20 · min(1, line_count / 8)
      + 0.20 · (1 − min(1, error_count / 3))
```

Bounded [0, 1], rounded to 4 decimals. Pass threshold: `SUCCESS_THRESHOLD = 0.5`.

## Cycle model

- A **cycle** is one extract+retries episode per **resolved source path**;
  `image_key` is that path as passed.
- `MAX_ATTEMPTS = 4`. Finalize when the score ≥ 0.5, or on attempt 4, or when a
  previous cycle kept an equal-or-better winner:
  `kept >= best_score` → `kept-previous`.
- Outcomes: `SUCCESS` / `RETRYABLE` / `MANUAL_REVIEW` (4th attempt, below
  threshold) / `ERROR` (a model pass hard-failed before any payload).
- Winner = highest score; a tie goes to the **lower** attempt number.

## `preprocessing_attempts` schema

Same SQLite file as `invoices` (`dbs/invoices.sqlite`, gitignored).

| Column | Meaning |
|---|---|
| `image_key` | resolved source path as passed |
| `cycle` / `attempt` | 1-based episode counter / position within it |
| `config` | JSON knob set the attempt ran with |
| `score` / `metrics` | score + present/missing fields, line counts, sum-vs-total |
| `outcome` | SUCCESS / RETRYABLE / MANUAL_REVIEW / ERROR |
| `payload` / `html` | JSON pass payload / HTML output (NULL on hard failure) |
| `enhanced_image` | BLOB of the enhanced bytes fed to the model (NULL if payload was None) |
| `best` / `best_processed_path` | winner flag + promoted image path, set at finalize |
| `created_at` | ISO timestamp |

`record_attempt` commits inline — an uncommitted INSERT once held the write
transaction open and deadlocked the next `InvoiceStore.ensure_schema` DDL.
Regressions live in the D/E groups of the offline suite.

## Finalize side effects (for the winner)

- Enhanced image promoted to `<date-dir>/<stem>_processed.<ext>`.
- Source disposition: `moved` (source was in the work dir) / `in-place`
  (already in the date dir) / `kept-source` (path collision) / `kept-previous`
  (an earlier cycle's winner stands; source untouched).
- JSON / Markdown / HTML outputs written next to the winner image.
- `InvoiceStore` row upsert (`row_id`, outputs, embedding/chunks).

## Tests

```
uv run python tests/test_offline.py
```

Plain-Python driver (no pytest in the uv lock), faked model passes, isolated
DB dir. 182 checks covering: config boundary + error strings, engine
determinism (same knobs → byte-identical output), store semantics, finalize
dispositions, the 4-attempt budget, `kept-previous`, history lookups, and a
wire-level round-trip through `fastmcp.client.Client` (test F).
