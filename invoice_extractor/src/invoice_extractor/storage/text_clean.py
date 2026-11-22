"""Tokenize + clean invoice text for BM25 indexing.

Deliberately dependency-free: a word tokenizer plus a fixed English
stopword list. No NLTK, no network, no model calls — the corpus this runs
over is small (a handful of invoices) and the goal is rankable, readable
tokens, not linguistic fidelity.
"""

import re

_WORD_RE = re.compile(r"[a-z0-9]+")

# Curated English stopwords. Short, open-vocabulary list: anything not
# listed here is kept. Numbers and model/currency tokens (e.g. "usd",
# "net", "wire") are intentional content for invoice search.
STOPWORDS = frozenset(
    """
    a an the and or but nor so yet
    of to in on at by for with as from into over under after before about
    against between out off up down again further then once here there when
    where why how all any both each few more most other some such only own
    same this that these those it its itself he she they them their his her
    i we you they me us them what which who whom whose
    is are was were be been being am
    do does did done having have has had will would can could should may
    might must shall not no nor
    per via than too very s t don should've
    """.split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase word/number tokens, stopwords removed, order preserved."""
    return [w for w in _WORD_RE.findall((text or "").lower()) if w not in STOPWORDS]


def clean_text(text: str) -> str:
    """Space-joined clean tokens — what gets stored in ``clean_text`` and
    what the BM25 corpus is built from (via ``.split()``)."""
    return " ".join(tokenize(text))
