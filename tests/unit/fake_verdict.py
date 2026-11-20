"""The verdict a fake judge computes from a judge prompt, shared by the Foundry and Bedrock fakes.

The wire contract (docs/test-plan.md §14 "Judge verdicts (S6.3)"): the prompt's user content holds
exactly one `<claim>…</claim>` block and one `<evidence citation="N">…</evidence>` block per cited
entry, both HTML-escaped. The verdict is computed, never supplied by a test: a claim is supported
iff it carries at least one number and every number in it appears in some evidence block. The reply
also lists one assertion per distinct number of the claim, in order of appearance: its `quote` is
that number when it appears in some evidence block, else null.

A prompt that does not hold exactly one claim block, or whose claim holds the garble marker, gets a
reply with no parseable verdict. A claim holding one of the letters-only `MISBEHAVIORS` markers
gets a reply (always carrying a valid verdict) whose `assertions` or text is malformed, oversized or
at a length boundary; markers live in the claim text so both provider fakes misbehave alike.
"""

from __future__ import annotations

import html
import json
import re
from typing import Any

_CLAIM = re.compile(r"<claim>(.*?)</claim>", re.DOTALL)
_EVIDENCE = re.compile(r'<evidence citation="(\d+)">(.*?)</evidence>', re.DOTALL)
_NUMBER = re.compile(r"\d+(?:\.\d+)?")

UNABLE = "I am unable to reach a verdict on this claim."
ASSERTION_TEXT = "The claim states {number}."
LEAK_TEXT = "zqleak-assertion-text-sentinel"
LEAK_QUOTE = "zqleak-assertion-quote-sentinel"


def long_string(length: int) -> str:
    """A string whose every character depends on its position, so a head cut differs from a tail cut."""
    return "".join(chr(ord("a") + (i * 7 + i // 26) % 26) for i in range(length))


def _items(count: int) -> list[Any]:
    return [{"text": "a", "quote": None} for _ in range(count)]


# marker in the claim text -> (what to do with the computed assertions, replacement reply or None)
MISBEHAVIORS: dict[str, Any] = {
    "ASSERTMISSING": lambda base: ("drop", None),
    "ASSERTSTRING": lambda base: ("set", "the claim holds"),
    "ASSERTOBJECT": lambda base: ("set", {"text": "a", "quote": None}),
    "ASSERTNULL": lambda base: ("set", None),
    "ITEMNOTOBJECT": lambda base: ("set", [*base, "a bare string"]),
    "TEXTNOTSTRING": lambda base: ("set", [{"text": 7, "quote": None}]),
    "QUOTENOTSTRING": lambda base: ("set", [{"text": "a", "quote": 7}]),
    "EXACTLYHUNDRED": lambda base: ("set", _items(100)),
    "OVERHUNDRED": lambda base: ("set", _items(101)),
    "TEXTEXACT": lambda base: ("set", [{"text": long_string(1000), "quote": None}]),
    "TEXTOVER": lambda base: ("set", [{"text": long_string(1001), "quote": None}]),
    "QUOTEEXACT": lambda base: ("set", [{"text": "a", "quote": long_string(1000)}]),
    "QUOTEOVER": lambda base: ("set", [{"text": "a", "quote": long_string(1001)}]),
    "LEAKPROBE": lambda base: ("set", [{"text": LEAK_TEXT, "quote": LEAK_QUOTE}]),
    "REPLYEXACT": lambda base: ("reply", long_string(4000)),
    "REPLYOVER": lambda base: ("reply", long_string(4001)),
}


def verdict_text(prompt: str, garble_marker: str | None = None) -> str:
    claims = _CLAIM.findall(prompt)
    if len(claims) != 1 or (garble_marker and garble_marker in html.unescape(claims[0])):
        return UNABLE
    claim = html.unescape(claims[0])
    ordered = list(dict.fromkeys(_NUMBER.findall(claim)))
    evidence_numbers: set[str] = set()
    for _, text in _EVIDENCE.findall(prompt):
        evidence_numbers |= set(_NUMBER.findall(html.unescape(text)))
    supported = bool(ordered) and set(ordered) <= evidence_numbers
    assertions: Any = [
        {
            "text": ASSERTION_TEXT.format(number=number),
            "quote": number if number in evidence_numbers else None,
        }
        for number in ordered
    ]
    reply: dict[str, Any] = {
        "verdict": "supported" if supported else "unsupported",
        "reason": f"{len(set(ordered) & evidence_numbers)} of {len(ordered)} figures found",
    }
    for marker, misbehave in MISBEHAVIORS.items():
        if marker in claim:
            action, value = misbehave(assertions)
            if action == "reply":
                return str(value)
            assertions = value
            if action == "drop":
                return json.dumps(reply)
            break
    return json.dumps({"assertions": assertions, **reply})
