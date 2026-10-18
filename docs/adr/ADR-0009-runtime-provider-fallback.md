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

## Amendment 2026-10-17 — Azure-first draft ladder, GPT-5.6 Luna judge (owner decision)

Supersedes the model choices of the 2026-10-08 amendment and, for the model plane, ADR-0006's "Foundry serves fallbacks only". Applies to local development and the deployed configuration alike.

Decision:

- **Draft ladder:** Microsoft Foundry DeepSeek-V4-Flash (Global Standard deployment) → Amazon Bedrock DeepSeek V3.2 → personal Anthropic → personal OpenAI. Each attempt is bounded as before. The Azure tier is used only when the endpoint, key and draft deployment name are all set; otherwise it is skipped, exactly as a personal fallback without its key is skipped. Personal keys stay development-only; production never configures them.
- **Bedrock V3.2 identifier (correction):** `deepseek.v3.2`, on-demand. `us.deepseek.v3.2` does not exist (verified 2026-10-17). The 2026-10-08 wording "US cross-region inference profile" is wrong for V3.2. The default `AWS_BEDROCK_MODEL` becomes `deepseek.v3.2`.
- **A rate-limit (429) response on Azure is an attempt failure** and falls back at once; there is no retry inside an attempt. The draft deployment starts at 20K tokens/min. A memo request is about 3.3K–5.1K input tokens (recorded live runs) plus up to 4,096 requested output tokens, which Azure counts against the limit, so 20K tokens/min carries about two memos a minute before the expensive fallback starts.
- **Judge ladder (evaluation only):** GPT-5.6 Luna on Bedrock → GPT-5.6-luna (short context, Global Standard) on Foundry. Both are called through one OpenAI-compatible client with two base URLs. On Bedrock the call goes to the `bedrock-mantle` endpoint with the bare id `openai.gpt-5.6-luna` and the Bedrock API key, because this account's `bedrock-runtime` cross-region quota for Luna is 0 (403 `not available for this account`, 2026-10-17). The judge lives with the evaluation code; the runtime gate stays the heuristic grounding verifier, and `get_llm` gains no second ladder.
- **Judge trust:** the S6 adoption check applies to the judge role, not to a named model: Luna's verdicts must reach κ ≥ 0.85 against ≥ 50 owner-reviewed claims before any Luna score is cited.
- **Configuration:** `AZURE_OPENAI_ENDPOINT` is the Foundry resource endpoint; a trailing `/openai/v1` is stripped and a project endpoint (`/api/projects/`) is rejected when settings load. `AZURE_API_KEY` (local development only), `AZURE_FOUNDRY_MODEL` (draft deployment name), `AZURE_FOUNDRY_JUDGE_MODEL` (judge deployment name). Calls use the `/openai/v1/` route with the deployment name as the model. Production authentication stays AWS STS outbound identity federation trusted by an Entra federated credential (spike).

Why:

- Budget: drafting moves to Azure, where V4-Flash costs about $0.19/$0.51 per million input/output tokens against $0.62/$1.85 for V3.2 on Bedrock, and spend is split across the two clouds.
- DeepSeek family on both hosts keeps the drafting style close to the existing V3.2 baselines.
- V4-Flash over V4.1-Flash: generally available rather than preview, and cheaper ($0.30/$1.20).
- The judge is the same model on both hosts, so judge parity is a like-for-like check.

Consequences (accepted by the owner 2026-10-17):

- **The draft fallback is a different model.** V4-Flash and V3.2 differ in generation and size, so V3.2 needs its own quality check on the frozen eval set before it is trusted as a fallback, and V4-Flash needs a fresh memo baseline. The results in `evaluation/results/` (V3.2 at temperature 0.3) describe the fallback; they are never compared with V4-Flash while anything else changes.
- **Azure becomes a hot-path prerequisite.** The Azure tier (previously S10-T03) and the STS-to-Entra spike are now needed for every deployed draft, not only for fallback.
- **Two ladders run in opposite order** (draft Azure → Bedrock, judge Bedrock → Azure); an outage on either cloud moves one role to the other.
- **An Azure outage or sustained 429s raise cost** about 3 to 3.6 times per token. Budget alerts are set on both clouds.
- Open: whether V4-Flash reasons by default (billed as output); whether Luna accepts `temperature=0`; the `bedrock-mantle` Luna call itself (pending a live check).

Alternatives: Gemma 4 26B-A4B on Bedrock (withdrawn by the owner; served only on `bedrock-mantle`, and on Foundry only as managed compute); gpt-oss-120b as the Azure drafter (cross-family with V3.2, superseded by V4-Flash); V3.2 primary on Bedrock with Foundry as fallback only (the 2026-10-08 ladder; higher draft cost).
