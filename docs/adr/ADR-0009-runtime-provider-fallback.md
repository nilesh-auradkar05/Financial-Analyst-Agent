# ADR-0009 — Bounded runtime provider fallback

Status: Accepted by explicit user request, 2026-09-29.
Trace: SPEC §§3/4/8.5; test-plan §16; authorized S7 slice.

Use the existing model factory and native runnable fallback support for Bedrock primary, then configured personal Anthropic and OpenAI. Bound attempts, include construction/authentication failures, and retain native model tracing. No new service or circuit-breaker framework. AWS_BEDROCK_MODEL overrides legacy LLM_MODEL. Canonical AWS_BEARER_TOKEN_BEDROCK takes precedence over AWS_BEARER_TOKEN. Ollama is explicitly selected for development, never automatic cloud fallback.

Azure is the intended first fallback after Bedrock; invocation is deferred to S7-AZURE-FALLBACK. Preserve supplied configuration without claiming it is active. Personal credentials enable fallback; successful primary execution must not call them.

Absent filings follow existing evidence-completeness semantics: available evidence can support a memo, missing required filings produce evidence_missing, and memo states SEC Filings: Not Available. A missing 10-K is distinct from infrastructure failure. Verification and citation rules stay in force.

Amendment 2026-09-30 (user decision): one variable per provider tier. AWS_BEDROCK_MODEL is the only Bedrock model variable; CLAUDE_LLM_MODEL and OPENAI_LLM_MODEL select the personal Anthropic and OpenAI fallback models (replacing ANTHROPIC_MODEL/OPENAI_MODEL); LLM_MODEL, LLM_PROVIDER and Ollama chat are removed; AZURE_FOUNDRY_MODEL is reserved for S7-AZURE-FALLBACK. This supersedes the LLM_MODEL alias and the explicit Ollama development selection above.

Consequences: fallback may consume personal API quota and change the generating model; native traces are authoritative. Offline tests establish behavior, not live credential validity or model entitlement.

## Amendment 2026-10-08 — production model ladder (owner decision D09, with ADR-0006)

Applies to the deployed (S10) configuration. Local development keeps the ladder above.

Decision:

- Draft model: DeepSeek V3.2 on Bedrock, called through the US cross-region inference profile (`AWS_BEDROCK_MODEL`). Thinking mode stays off for drafts unless the frozen-evidence eval shows a gain, because thinking tokens bill as output.
- Judge/verifier model: gpt-oss-120b on Bedrock. It is the judge model in the SPEC §11.2 measurement (heuristic verifier against judge, κ ≥ 0.85 on ≥ 50 claims) and the model for any LLM claim-check step S6 adopts. The runtime gate at HEAD remains the heuristic grounding verifier. Before gpt-oss-120b is trusted as the judge, S6 checks its verdicts against ≥ 50 owner-reviewed claims using the same κ ≥ 0.85 threshold; this adoption check is an S6 task, not a SPEC change.
- Fallback: the same two models on Microsoft Foundry (`AZURE_FOUNDRY_MODEL` plus a verifier variable added in S7-AZURE-FALLBACK). Ladder: Bedrock through the US inference profile (which already spreads requests across US regions) with bounded retries → per-provider breaker opens → Foundry → typed `failed` with `provider_unavailable`. Failover after partial output starts a new attempt; partial drafts are never spliced.
- Authentication to Foundry: AWS STS outbound identity federation issues a signed JWT that an Entra federated credential trusts. No Azure key is stored in AWS.
- Personal Anthropic and OpenAI keys are development and test only; production never configures them.

Why:

- Cost: about $0.014 per uncached memo (DeepSeek V3.2 $0.62/$1.85 per million input/output tokens; gpt-oss-120b $0.15/$0.60) against about $0.075 for a Claude Sonnet/Haiku pair. Owner priority: prefer cheaper models throughout a portfolio deployment.
- Same models on both hosts: qualifying the fallback becomes a parity check on the frozen eval set rather than a re-validation of a different model family, which is what makes a fallback trustworthy (production review finding 9).
- Continuity: the existing retrieval baselines in `evaluation/results/` were produced with `deepseek.v3.2` at temperature 0.3.
- A verifier from a different family than the drafter is less likely to share its mistakes.
- Both clouds host these open-weight models on their own infrastructure; prompts are not sent to DeepSeek's API.

Alternatives: Claude Sonnet + Haiku (rejected, about 5× the token cost); DeepSeek V3.2 verifying itself (rejected, shared blind spots); Amazon Nova Lite/Micro (rejected, no Azure twin); OpenAI models on Azure only (rejected, cross-family qualification).

Consequences and checks:

- The same weights can still differ by host (serving precision, defaults, token limits). Parity on the frozen eval set is required before Foundry is enabled, and repeated when either host changes the model version.
- Quality against frontier models is unmeasured; the eval gate decides, not the price.
- Verify before enabling: Foundry serverless price and quota for DeepSeek V3.2 and gpt-oss-120b; the Bedrock US profile identifiers; Entra acceptance of the STS token (spike).
