"""The verdict a fake judge computes from a judge prompt, shared by the Foundry and Bedrock fakes.

The wire contract (docs/test-plan.md §14 "Judge verdicts (S6.3)"): the prompt's user content holds
exactly one `<claim>…</claim>` block and one `<evidence citation="N">…</evidence>` block per cited
entry, both HTML-escaped. The verdict is computed, never supplied by a test: a claim is supported
iff it carries at least one number and every number in it appears in some evidence block. A prompt
that does not hold exactly one claim block, or whose claim holds the garble marker, gets a reply
with no parseable verdict.
"""

from __future__ import annotations

import html
import json
import re

_CLAIM = re.compile(r"<claim>(.*?)</claim>", re.DOTALL)
_EVIDENCE = re.compile(r'<evidence citation="(\d+)">(.*?)</evidence>', re.DOTALL)
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def verdict_text(prompt: str, garble_marker: str | None = None) -> str:
    claims = _CLAIM.findall(prompt)
    if len(claims) != 1 or (garble_marker and garble_marker in html.unescape(claims[0])):
        return "I am unable to reach a verdict on this claim."
    claim_numbers = set(_NUMBER.findall(html.unescape(claims[0])))
    evidence_numbers: set[str] = set()
    for _, text in _EVIDENCE.findall(prompt):
        evidence_numbers |= set(_NUMBER.findall(html.unescape(text)))
    supported = bool(claim_numbers) and claim_numbers <= evidence_numbers
    return json.dumps(
        {
            "verdict": "supported" if supported else "unsupported",
            "reason": f"{len(claim_numbers & evidence_numbers)} of {len(claim_numbers)} figures found",
        }
    )
