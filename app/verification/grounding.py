from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

# Bumped when claim-extraction or number-gate semantics change, so pre-fix and
# post-fix baseline artifacts can never be silently compared.
GROUNDING_EVAL_VERSION = 4

_SENTENCE_SPILT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_CITATION_RE = re.compile(r"\[(\d+)\]")
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9&'\-/]*")
_NUMBER_RE = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?")
_EMPHASIS_RE = re.compile(r"[*_`]+")
# A scale follows a figure as a word ("7 million") or as an abbreviation attached
# to the digits ("$7M", "3.2bn"); the abbreviation must end the word, so "10-K"
# (hyphen) and "7Mbps" (letters follow) carry no scale.
_SCALE_RE = re.compile(
    r"\s*(thousand|million|billion|trillion)\b|(MM|BN|K|M|B|T)(?![A-Za-z])", re.IGNORECASE
)
_SCALES = {
    "thousand": 1e3,
    "k": 1e3,
    "million": 1e6,
    "m": 1e6,
    "mm": 1e6,
    "billion": 1e9,
    "b": 1e9,
    "bn": 1e9,
    "trillion": 1e12,
    "t": 1e12,
}
# SEC filing tables state figures in thousands or millions.
_FILING_UNITS = (1e3, 1e6)

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "with",
}

_CLAIM_HINTS = {
    "business",
    "cash",
    "competition",
    "customer",
    "demand",
    "decline",
    "exposure",
    "financial",
    "flow",
    "generated",
    "grew",
    "growth",
    "guidance",
    "income",
    "increase",
    "inventory",
    "margin",
    "market",
    "product",
    "products",
    "profit",
    "regulatory",
    "revenue",
    "risk",
    "risks",
    "sales",
    "segment",
    "services",
    "share",
    "supply",
}

@dataclass(frozen=True, slots=True)
class ClaimAssessment:
    sentence: str
    citations: list[int]
    supported: bool
    overlap_score: float
    missing_citation: bool = False
    reason: str | None = None
    numbers_ok: bool = True

@dataclass(frozen=True, slots=True)
class GroundingCheckResult:
    passed: bool
    total_claims: int
    cited_claims: int
    grounded_claims: int
    citation_coverage_rate: float
    grounded_claim_rate: float
    claims: list[ClaimAssessment] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "total_claims": self.total_claims,
            "cited_claims": self.cited_claims,
            "grounded_claims": self.grounded_claims,
            "citation_coverage_rate": self.citation_coverage_rate,
            "grounded_claim_rate": self.grounded_claim_rate,
            "claims": [
                {
                    "sentence": claim.sentence,
                    "citations": claim.citations,
                    "supported": claim.supported,
                    "overlap_score": claim.overlap_score,
                    "missing_citation": claim.missing_citation,
                    "reason": claim.reason,
                    "numbers_ok": claim.numbers_ok,
                }
                for claim in self.claims
            ],
        }

@dataclass(frozen=True, slots=True)
class _EvidenceRef:
    index: int
    text: str
    source_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    filing_text: str | None = None

def evaluate_memo_grounding(
    memo: str,
    evidence: Sequence[object],
    *,
    min_citation_coverage: float = 0.80,
    min_grounded_claim_rate: float = 0.75,
    min_overlap_score: float = 0.20,
) -> GroundingCheckResult:
    claims = _extract_claim_sentences(memo)
    evidence_map = _build_evidence_map(evidence)

    assessments: list[ClaimAssessment] = []
    cited_claims = 0
    grounded_claims = 0

    for sentence in claims:
        citations = _extract_citations(sentence)
        cleaned_sentence = _strip_citations(sentence)

        if not citations:
            assessments.append(
                ClaimAssessment(
                    sentence=cleaned_sentence,
                    citations=[],
                    supported=False,
                    overlap_score=0.0,
                    missing_citation=True,
                    reason="claim is missing a citation",
                )
            )
            continue

        cited_claims += 1
        best_score = 0.0
        cited_evidence_texts: list[str] = []
        cited_filing_texts: list[str] = []

        for citation_idx in citations:
            evidence_ref = evidence_map.get(citation_idx)
            if evidence_ref is None:
                continue

            cited_evidence_texts.append(evidence_ref.text)
            if evidence_ref.filing_text is not None:
                cited_filing_texts.append(evidence_ref.filing_text)
            overlap_score = _token_overlap_score(cleaned_sentence, evidence_ref.text)
            if overlap_score > best_score:
                best_score = overlap_score

        numbers_ok, unmatched_numbers = _numbers_supported_any(
            cleaned_sentence, cited_evidence_texts, cited_filing_texts
        )
        supported = best_score >= min_overlap_score and numbers_ok
        if supported:
            grounded_claims += 1
            assessments.append(
                ClaimAssessment(
                    sentence=cleaned_sentence,
                    citations=citations,
                    supported=True,
                    overlap_score=best_score,
                    numbers_ok=numbers_ok,
                )
            )
        else:
            reason = "claim does not align with cited evidence"
            if not numbers_ok:
                reason = f"claim numbers not found in cited evidence: {', '.join(unmatched_numbers)}"
            assessments.append(
                ClaimAssessment(
                    sentence=cleaned_sentence,
                    citations=citations,
                    supported=False,
                    overlap_score=best_score,
                    reason=reason,
                    numbers_ok=numbers_ok,
                )
            )

    total_claims = len(assessments)
    citation_coverage_rate = cited_claims / total_claims if total_claims > 0 else 0.0
    grounded_claim_rate = grounded_claims / cited_claims if cited_claims > 0 else 0.0
    passed = (
        total_claims > 0
        and citation_coverage_rate >= min_citation_coverage
        and grounded_claim_rate >= min_grounded_claim_rate
    )

    return GroundingCheckResult(
        passed=passed,
        total_claims=total_claims,
        cited_claims=cited_claims,
        grounded_claims=grounded_claims,
        citation_coverage_rate=citation_coverage_rate,
        grounded_claim_rate=grounded_claim_rate,
        claims=assessments,
    )

def _build_evidence_map(evidence: Sequence[object]) -> dict[int, _EvidenceRef]:
    refs: dict[int, _EvidenceRef] = {}

    for index, item in enumerate(evidence, start=1):
        if isinstance(item, str):
            refs[index] = _EvidenceRef(index=index, text=item)
            continue

        text = getattr(item, "text", str(item))
        filing_text = None
        if isinstance(item, Mapping) and item.get("source_type") == "sec_filing":
            body = item.get("text")
            filing_text = body if isinstance(body, str) else None
        source_id = getattr(item, "chunk_id", None) or getattr(item, "id", None)
        metadata = getattr(item, "metadata", {}) or {}
        refs[index] = _EvidenceRef(
            index=index,
            text=text,
            source_id=source_id,
            metadata=metadata,
            filing_text=filing_text,
        )

    return refs

def _extract_claim_sentences(memo: str) -> list[str]:
    pieces = [piece.strip() for piece in _SENTENCE_SPILT_RE.split(memo) if piece.strip()]
    return [piece for piece in pieces if _looks_like_claim(piece)]

def _looks_like_claim(sentence: str) -> bool:
    stripped = sentence.strip()
    if not stripped:
        return False
    # ATX headings and blockquotes are never claims.
    if stripped.startswith(("#", ">")):
        return False

    has_citation = bool(_CITATION_RE.search(stripped))
    # Strip citations + markdown emphasis so heading markers and inline bold do
    # not leak into the heading test below or into token / number extraction.
    normalized = _EMPHASIS_RE.sub("", _CITATION_RE.sub("", stripped)).strip()
    if not normalized:
        return False

    # A scorable claim is a terminated sentence OR carries a citation. Section
    # headings written as bold fragments -- "**Recent News & Market Sentiment**",
    # "**Key Risk Factors (from SEC Filings)**" -- have neither, yet trip claim
    # hints ("market", "risk"). Without this guard they count as phantom uncited
    # claims and silently depress citation coverage on every memo.
    if not has_citation and normalized[-1] not in ".!?":
        return False

    tokens = _content_tokens(normalized)
    if len(tokens) < 3:
        return False

    if _extract_numbers(normalized):
        return True

    return any(token in _CLAIM_HINTS for token in tokens)

def _extract_citations(sentence: str) -> list[int]:
    return [int(match) for match in _CITATION_RE.findall(sentence)]

def _strip_citations(sentence: str) -> str:
    cleaned = _CITATION_RE.sub("", sentence)
    return re.sub(r"\s+", " ", cleaned).strip()

def _extract_numbers(text: str) -> set[str]:
    return {match.group(0).replace(",", "") for match in _NUMBER_RE.finditer(text)}

def _extract_number_tokens(text: str) -> list[tuple[float, int]]:
    """Return (value, stated_decimal_places) for each numeric token.

    Precision is preserved so support can be judged at the precision the claim
    actually states, rather than by exact string identity.
    """
    tokens: list[tuple[float, int]] = []
    for match in _NUMBER_RE.finditer(text):
        cleaned = match.group(0).replace("$", "").replace(",", "").replace("%", "")
        try:
            value = float(cleaned)
        except ValueError:
            continue
        decimals = len(cleaned.split(".")[1]) if "." in cleaned else 0
        tokens.append((value, decimals))
    return tokens

def _scale_after(text: str, end: int) -> float:
    """Scale (1.0 when none) stated right after the number token ending at `end`."""
    scale = _SCALE_RE.match(text, end)
    return _SCALES[(scale.group(1) or scale.group(2)).lower()] if scale else 1.0

def _absolute_amounts(texts: Sequence[str], filing_texts: Sequence[str]) -> list[float]:
    """Every figure in the cited evidence at its own stated scale. Comma-grouped
    filing-body figures without a scale also count in thousands and millions
    (SEC tables state them so); years and other plain numbers never scale."""
    amounts = [
        value * _scale_after(text, match.end())
        for text in texts
        for match, (value, _) in zip(_NUMBER_RE.finditer(text), _extract_number_tokens(text))
    ]
    amounts += [
        value * unit
        for text in filing_texts
        for match, (value, _) in zip(_NUMBER_RE.finditer(text), _extract_number_tokens(text))
        if "," in match.group(0) and _scale_after(text, match.end()) == 1.0
        for unit in _FILING_UNITS
    ]
    return amounts

def _numbers_supported_any(
    claim_text: str,
    evidence_texts: Sequence[str],
    filing_texts: Sequence[str] = (),
) -> tuple[bool, list[str]]:
    """A claim's numbers are supported if, for EACH number token in the claim, AT
    LEAST ONE cited evidence text contains a value that rounds (at the claim's
    stated precision) to it.

    This decouples the number gate from best-similarity evidence selection: a
    claim citing [1][2] must not be flagged unsupported just because [1] (not
    [2], which actually has the number) happens to score the higher similarity.

    Returns (all_supported, unmatched) where `unmatched` lists the offending
    number substrings verbatim as they appear in the claim, so a human can
    hand-audit exactly which figures were not found in any cited evidence.

    A claim figure that states a scale ("$356.7 million", "$3.2B") is part of the
    figure: it is supported only by an evidence figure of the same absolute
    amount, rounded in the claim's unit at the claim's stated precision. A bare
    "7" never supports "$7.0 million".
    """
    claim_tokens = _extract_number_tokens(claim_text)
    if not claim_tokens:
        return True, []

    evidence_values = [
        value for text in evidence_texts for value, _ in _extract_number_tokens(text)
    ]
    amounts = _absolute_amounts(evidence_texts, filing_texts)

    unmatched: list[str] = []
    for (value, decimals), match in zip(claim_tokens, _NUMBER_RE.finditer(claim_text)):
        scale = _scale_after(claim_text, match.end())
        if scale == 1.0:
            supported = any(round(ev, decimals) == value for ev in evidence_values)
        else:
            supported = any(round(amount / scale, decimals) == value for amount in amounts)
        if not supported:
            unmatched.append(match.group(0))

    return not unmatched, unmatched

def _token_overlap_score(claim_text: str, evidence_text: str) -> float:
    claim_tokens = set(_content_tokens(claim_text))
    if not claim_tokens:
        return 0.0

    evidence_tokens = set(_content_tokens(evidence_text))
    if not evidence_tokens:
        return 0.0

    return len(claim_tokens & evidence_tokens) / len(claim_tokens)

def _content_tokens(text: str) -> list[str]:
    tokens = [token.lower() for token in _WORD_RE.findall(text)]
    return [token for token in tokens if token not in _STOPWORDS and len(token) > 1]
