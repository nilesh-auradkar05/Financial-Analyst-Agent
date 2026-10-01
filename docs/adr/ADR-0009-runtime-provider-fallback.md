# ADR-0009 — Bounded runtime provider fallback

Status: Accepted by explicit user request, 2026-09-29.
Trace: SPEC §§3/4/8.5; test-plan §16; authorized S7 slice.

Use the existing model factory and native runnable fallback support for Bedrock primary, then configured personal Anthropic and OpenAI. Bound attempts, include construction/authentication failures, and retain native model tracing. No new service or circuit-breaker framework. AWS_BEDROCK_MODEL overrides legacy LLM_MODEL. Canonical AWS_BEARER_TOKEN_BEDROCK takes precedence over AWS_BEARER_TOKEN. Ollama is explicitly selected for development, never automatic cloud fallback.

Azure is the intended first fallback after Bedrock; invocation is deferred to S7-AZURE-FALLBACK. Preserve supplied configuration without claiming it is active. Personal credentials enable fallback; successful primary execution must not call them.

Absent filings follow existing evidence-completeness semantics: available evidence can support a memo, missing required filings produce evidence_missing, and memo states SEC Filings: Not Available. A missing 10-K is distinct from infrastructure failure. Verification and citation rules stay in force.

Amendment 2026-09-30 (user decision): one variable per provider tier. AWS_BEDROCK_MODEL is the only Bedrock model variable; CLAUDE_LLM_MODEL and OPENAI_LLM_MODEL select the personal Anthropic and OpenAI fallback models (replacing ANTHROPIC_MODEL/OPENAI_MODEL); LLM_MODEL, LLM_PROVIDER and Ollama chat are removed; AZURE_FOUNDRY_MODEL is reserved for S7-AZURE-FALLBACK. This supersedes the LLM_MODEL alias and the explicit Ollama development selection above.

Consequences: fallback may consume personal API quota and change the generating model; native traces are authoritative. Offline tests establish behavior, not live credential validity or model entitlement.
