"""Output guardrails: a deterministic screen of the final memo, and the disclaimer (SPEC 11.4).

No model or network call: the same memo and verifier flags always give the same memo and events.
"""

import re
from collections.abc import Callable

from app.verification.grounding import _strip_citations

DISCLAIMER = (
    "Disclaimer: This memo was generated automatically from public sources for information only. "
    "It is not investment advice or a recommendation to buy or sell any security, and it may "
    "contain errors. Do your own research or consult a licensed financial adviser."
)

_FILINGS_LINE = "SEC Filings: Not Available."

_PRICE_TARGET_RE = re.compile(r"\b(?:price\s+target|target\s+price|price\s+objective)", re.I)
_GUARANTEED_RETURN_RE = re.compile(
    r"\b(?:guaranteed\s+(?:returns?|profits?)|risk-free\s+returns?|cannot\s+lose)\b", re.I
)
_MNPI_RE = re.compile(
    r"\b(?:based\s+on\s+insider\s+information|material\s+non-?public\s+information"
    r"|has\s+not\s+yet\s+been\s+publicly\s+announced)\b",
    re.I,
)
# A currency amount, a percentage, a multiple (23.3x), a decimal, or a figure with a scale word.
_NUMBER_RE = re.compile(
    r"\$\s?\d|\d\s?%|\b\d+(?:\.\d+)?x\b|(?<![\d.])\d+\.\d+(?!\d|\.\d)|\d\s*(?:million|billion|trillion)\b", re.I
)
_RECOMMENDATION_RE = re.compile(
    r"\brecommend|\b(?:buy|hold|sell|neutral)\W+(?:rating|stance|position|recommendation)\b", re.I
)
_PII_RE = re.compile(
    r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"  # email
    r"|(?:\(\d{3}\)\s?|(?<!\d)\d{3}[-.])\d{3}[-.]\d{4}(?!\d)"  # US phone
    r"|(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"  # US SSN
)


def redact_pii(text: str) -> tuple[str, int]:
    """`text` with each email, US phone number and US SSN replaced by `[redacted]`, and how many."""
    return _PII_RE.subn("[redacted]", text)


def without_sentences(memo: str, drop: Callable[[str], bool]) -> str:
    """The memo minus the sentences `drop` accepts; lines left empty go too."""
    lines = []
    for line in memo.split("\n"):
        parts = re.split(r"((?<=[.!?])\s+)", line)
        pieces = [(parts[i], "".join(parts[i + 1 : i + 2])) for i in range(0, len(parts), 2)]
        kept = [text + gap for text, gap in pieces if not drop(text)]
        if kept:
            lines.append("".join(kept).rstrip() if len(kept) < len(pieces) else line)
    return "\n".join(lines)


def _removal_code(sentence: str, flagged: bool, uncited: bool) -> str | None:
    """The event code of the rule that removes `sentence`, or None; `flagged` means the verifier
    did not mark it supported."""
    if sentence.startswith(_FILINGS_LINE):
        return None
    if flagged and _PRICE_TARGET_RE.search(sentence) and re.search(r"\d", sentence):
        return "price_target_removed"
    if uncited and _NUMBER_RE.search(sentence) and not _RECOMMENDATION_RE.search(sentence):
        return "uncited_numeric_removed"
    if _GUARANTEED_RETURN_RE.search(sentence):
        return "guaranteed_return_removed"
    if _MNPI_RE.search(sentence):
        return "mnpi_removed"
    return None


def screen_memo(
    memo: str, unsupported_sentences: set[str], uncited_sentences: set[str]
) -> tuple[str, list[dict]]:
    """Remove the sentences the output policy forbids and redact PII; one event per change.

    The two sets hold the sentences the verifier did not mark supported and the ones missing a
    citation, in the verifier's form. Events never carry the removed or redacted text.
    """
    events: list[dict] = []

    def drop(text: str) -> bool:
        sentence = _strip_citations(text)  # the verifier's form of the sentence
        code = _removal_code(
            sentence,
            sentence in unsupported_sentences or sentence in uncited_sentences,
            sentence in uncited_sentences,
        )
        if code:
            events.append({"code": code, "detail": None})
        return code is not None

    screened = without_sentences(memo, drop)

    redacted, count = redact_pii(screened)
    events.extend({"code": "pii_redacted", "detail": None} for _ in range(count))
    return redacted, events


def with_disclaimer(memo: str) -> str:
    """The memo ending with the disclaimer exactly once; `memo` is kept as is in front of it."""
    return memo if memo.endswith(DISCLAIMER) else f"{memo}\n\n{DISCLAIMER}"
