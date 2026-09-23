"""Style-transferred text augmentation used by the StyleGuard framework.

These lightweight, model-free transforms simulate the kind of surface
rewriting that LLM style-transfer produces (e.g. dropped agency datelines,
paraphrased function words, expanded abbreviations, altered punctuation and
leading markers) while preserving the underlying semantics, so that the class
label of each training example is unchanged.
"""

from __future__ import annotations

import random
import re

from .config import RANDOM_STATE

DATELINE_PATTERN = re.compile(
    r"(?:^|\.\s)(?:[A-Z][A-Z/.,\s]+?)\s*\(reuters\)\s*[–-]\s*",
    flags=re.IGNORECASE,
)

_DATE_MONTH = re.compile(
    r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s*"
    r"\d{1,2},?\s*(?:19|20)\d{2}\b",
    flags=re.IGNORECASE,
)

_ABBREVIATIONS = {
    "u.s.": "united states",
    "u.s": "united states",
    "u.s.a.": "united states of america",
    "govt": "government",
    "gov't": "government",
    "dept": "department",
    "approx": "approximately",
    "st.": "saint",
    "mt.": "mount",
    "rep.": "representative",
    "sen.": "senator",
    "pres.": "president",
}

_FUNCTION_WORDS = {
    "the", "a", "an", "and", "of", "to", "in", "on", "for", "with",
    "at", "by", "from", "as", "that", "its", "it", "this", "these",
    "those", "their", "has", "have", "had", "he", "she", "they",
    "was", "were", "is", "are", "be", "been", "about", "over", "under",
}

def strip_datelines(text: str) -> str:
    """Remove agency dateline markers that real news carries and LLM rewrites drop."""
    text = DATELINE_PATTERN.sub("", text)
    text = re.sub(r"\bWASHINGTON\s*\(Reuters\)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\(Reuters\)\b", "", text, flags=re.IGNORECASE)
    return text


def scrub_dates(text: str) -> str:
    """Replace literal calendar dates with neutral placeholders."""
    return _DATE_MONTH.sub("on a recent day", text)


def expand_abbreviations(text: str) -> str:
    for short, long in _ABBREVIATIONS.items():
        text = re.sub(rf"\b{re.escape(short)}\b", long, text)
    return text


def _synonym_of(word: str) -> str | None:
    try:
        from nltk.corpus import wordnet
    except Exception:
        return None
    try:
        synsets = wordnet.synsets(word, pos=wordnet.NOUN)
        synsets += wordnet.synsets(word, pos=wordnet.VERB)
        synsets += wordnet.synsets(word, pos=wordnet.ADJ)
    except Exception:
        return None
    for syn in synsets:
        for lemma in syn.lemmas():
            candidate = lemma.name().replace("_", " ")
            if candidate.lower() != word and " " not in candidate and candidate.isalpha():
                return candidate.lower()
    return None


def synonym_swap(text: str, prob: float = 0.15, rng: random.Random | None = None) -> str:
    rng = rng or random.Random(RANDOM_STATE)
    tokens = text.split()
    out = []
    for token in tokens:
        alpha = re.fullmatch(r"[a-z]+", token)
        if alpha and rng.random() < prob:
            alias = _synonym_of(token)
            if alias:
                out.append(alias)
                continue
        out.append(token)
    return " ".join(out)


def function_word_drop(text: str, prob: float = 0.10, rng: random.Random | None = None) -> str:
    rng = rng or random.Random(RANDOM_STATE)
    tokens = text.split()
    kept = [t for t in tokens if not (t in _FUNCTION_WORDS and rng.random() < prob)]
    return " ".join(kept)


def phrase_permute(text: str, swaps: int = 2, rng: random.Random | None = None) -> str:
    rng = rng or random.Random(RANDOM_STATE)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) < 3:
        return text
    for _ in range(swaps):
        i, j = rng.sample(range(len(sentences)), 2)
        sentences[i], sentences[j] = sentences[j], sentences[i]
    return " ".join(sentences)


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def canonicalize(text: str) -> str:
    """Deterministic style normalization applied identically to every input.

    Strips wire datelines and dates and expands abbreviations -- the exact
    surface artifacts that LLM style-transfer removes -- so that the
    detector cannot rely on a fingerprint that is absent out-of-distribution.
    Unlike ``variants()``, no synonym/permutation randomness is introduced.
    """
    out = strip_datelines(text)
    out = scrub_dates(out)
    out = expand_abbreviations(out)
    return _collapse_whitespace(out)


class StyleAugmentor:
    """Samples style-varied transcriptions of a news article.

    Each variant applies a random subset of the interpretable surface
    rewrites that LLM style-transfer pipelines typically introduce:
    dateline stripping, date scrubbing, abbreviation expansion, synonym
    substitution, function-word deletion/reordering and clause permutation.
    """

    def __init__(self, seed: int = RANDOM_STATE, synonym_prob: float = 0.15):
        self.seed = seed
        self.synonym_prob = synonym_prob

    def variants(self, text: str, n: int = 1) -> list[str]:
        rng = random.Random(self.seed)
        results: list[str] = []
        for _ in range(n):
            out = strip_datelines(text)
            out = scrub_dates(out)
            out = expand_abbreviations(out)
            out = synonym_swap(out, prob=self.synonym_prob, rng=rng)
            out = function_word_drop(out, rng=rng)
            out = phrase_permute(out, rng=rng)
            out = _collapse_whitespace(out)
            if out:
                results.append(out)
        return results or [text]

    def augment_frame(self, frame, text_col: str = "text", per_sample: int = 2):
        """Expand a DataFrame with style-transferred copies, labels preserved."""
        rows = []
        for _, row in frame.iterrows():
            text = str(row[text_col])
            rows.append(row)
            for variant in self.variants(text, n=per_sample):
                copy = row.copy()
                copy[text_col] = variant
                rows.append(copy)
        import pandas as pd  # local import to keep module import light

        return pd.DataFrame(rows).reset_index(drop=True)