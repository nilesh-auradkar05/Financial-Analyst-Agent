# ADR-0019 — Hosted sentiment: Claude Sonnet 5.5 on Microsoft Foundry

**Status:** Proposed (2026-11-07) by owner decision; accepted when the CM-6b label-parity run is recorded below. Amends ADR-0015 §1 (ONNX FinBERT in the worker) and ADR-0013 D13 (sentiment placement); implements ADR-0016 decision 6.
**Date:** 2026-11-07
**Related:** SPEC §§5, 7, 8.2; ADR-0009 (provider policy); ADR-0013; ADR-0015; ADR-0016; test-plan §§2, 10, 13, 24; sprint-plan CM-6b.

## Context

News sentiment runs FinBERT (`ProsusAI/finbert`) in process on torch, on CUDA when present and on CPU otherwise. The owner set the project to a cloud-only model plane (2026-11-07): no local CUDA, torch or Ollama. FinBERT is not offered as a managed model on AWS or Azure, so hosting it would mean running our own endpoint (SageMaker Serverless Inference), which the owner did not want. ADR-0016 decision 6 required this choice to be made after a spike and recorded here.

## Decision

1. **Model and host.** Sentiment is a classification call to Claude Sonnet 5.5 on Microsoft Foundry, deployment `sonnet-5-5-sentiment`, through the Anthropic SDK's `AnthropicFoundry` client. It reuses `AZURE_OPENAI_ENDPOINT` (the Foundry resource; the client takes its resource name) and `AZURE_API_KEY`. The deployment name is its own setting, `AZURE_FOUNDRY_SENTIMENT_MODEL`. There is no FinBERT fallback and no silent default: an unconfigured sentiment model fails closed with a typed error.
2. **Contract unchanged.** `analyze_sentiment_batch(texts) -> list[SentimentResult]` keeps its shape: one result per input text, in input order, each with `label` ∈ {positive, negative, neutral}, `scores` for all three labels, and `confidence` = the score of the label. The model returns this through structured output (JSON schema); any response that does not validate (wrong count, unknown label, scores outside [0, 1], refusal, truncation) raises a typed error. The graph node already turns a sentiment failure into an error entry and a sentiment gap (test-plan §2); no sentiment is fabricated.
3. **Request shape.** Effort `low`; the `thinking` parameter is omitted (Sonnet 5.5 rejects `disabled`). A fixed system prompt; each article is passed as delimited data and treated as untrusted (SPEC §5 security; `app/guardrails/content.py`), so instructions inside an article are classified, not followed. Several texts go in one bounded call to amortize the system prompt. Every call has a timeout and bounded retries (SDK `max_retries`); the call runs off the event loop as today (`asyncio.to_thread` in the graph, or the async client).
4. **Identity and lineage.** The evidence snapshot's `model_name` and the trace metadata record the deployment and the served model (`claude-sonnet-5-5`), so sentiment from FinBERT and from Claude are never confused in a release or a cache key (SPEC §11.3 C6 keys on `model_version`).
5. **Removal.** After the parity run, FinBERT code, `FINBERT_*` settings, the startup preload and the GPU Compose overlay's purpose go away. torch and transformers leave the dependency set with sentence-transformers in CM-6d.

## Spike evidence (2026-11-07)

One bounded call, owner-approved spend: `AnthropicFoundry(resource=<resource>, timeout=30, max_retries=1)`, `messages.create(model="sonnet-5-5-sentiment", max_tokens=1024, output_config={"effort": "low", "format": {"type": "json_schema", ...}})` on a negative Apple headline returned `{"label": "negative", "score": 0.9}`, `stop_reason` `end_turn`, served model `claude-sonnet-5-5`, 1.92 s, 302 input and 16 output tokens.

## Cost

At $2 / $10 per million input / output tokens (Foundry bills Claude at first-party API rates), one article costs about $0.0008 as a single call; a memo with 10 articles batched in one call costs under $0.01. FinBERT's marginal cost was zero but needed a 3 GB worker and a model warm-up (ADR-0015 §1).

## Label parity (ADR-0016 decision 6)

Before FinBERT is deleted, both models score the same frozen news sample (the news snapshots of `alpha-evidence:0.1.0`). Recorded here: sample size, label agreement rate, the confusion table and every disagreement with its text preview. Parity is a record of how labels change, not a gate: Claude reads context FinBERT cannot, and the owner chose it for that. A disagreement rate the owner judges too high reopens this ADR.

Result: pending (CM-6b).

## Alternatives considered

| Option | Verdict | Reason |
| --- | --- | --- |
| FinBERT on SageMaker Serverless Inference | Rejected | Self-managed endpoint and image; cold starts; the owner wants managed models only |
| FinBERT in ONNX in the Lambda worker (ADR-0015 §1) | Superseded | Keeps a local model and a 3 GB worker |
| Bedrock Claude for sentiment | Not available | Sonnet 5.5 is gated for this AWS account (verified 2026-09-30) |
| Haiku-tier classifier | Alternative | Cheaper; the owner chose Sonnet 5.5 for context understanding |

## Consequences

- Sentiment now needs network and spends money per article; replay mode still calls no sentiment code (test-plan §13).
- Sentiment labels change meaning (contextual vs. lexical); memo text that cites sentiment changes accordingly, and no FinBERT-era sentiment snapshot is reused for a Claude run.
- Tests: test-plan §24.
