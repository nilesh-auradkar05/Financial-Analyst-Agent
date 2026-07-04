# Alpha Financial Analyst Agent — Runtime Analysis Sequence

**Companion to:** `docs/design-html/alpha-analyst-runtime-sequence.html`
**Purpose:** AI-readable sequence summary for the online path.

## Invariant

Evidence is built before memo generation. Verification runs after generation. Disabled controls are marked disabled; the system does not fake sentiment, SEC citations, or retrieval outputs, because apparently that has to be written down.

## Mermaid sequence

```mermaid
sequenceDiagram
    participant Client
    participant API as FastAPI
    participant State as Persistent Run State
    participant Graph as LangGraph Workflow
    participant Tools as Market/News Tools
    participant Retrieval as Retrieval Service
    participant LLM as LLM / Verifier

    Client->>API: POST /analyze {ticker, controls}
    API->>State: create run_id (queued → running)
    API->>Graph: invoke(ticker, controls)
    Graph->>Graph: normalize ticker/company identity

    opt include_news_sentiment=true
        Graph->>Tools: fetch market/news (≤ max_news_articles)
        Tools-->>Graph: market snapshot + articles/sentiment
    end

    opt include_filing_analysis=true
        Graph->>Retrieval: search(ticker, sections, query)
        Retrieval-->>Graph: EvidencePackets + diagnostics
    end

    Graph->>Graph: build structured evidence context
    Graph->>LLM: draft memo from context
    LLM-->>Graph: memo + citations
    Graph->>LLM: verify grounding / citation coverage
    LLM-->>Graph: verification payload
    Graph->>State: persist result (succeeded / failed / degraded)
    API-->>Client: result payload or run polling response
```

## Required behavior

- `include_filing_analysis=false` means no SEC retrieval and no SEC citations.
- `include_news_sentiment=false` means no news/sentiment step and no fabricated sentiment.
- `max_news_articles` bounds news retrieval.
- Empty retrieval produces an explicit limitation statement.
- Tool or LLM failure persists a safe failed/degraded run.
- Every memo citation resolves to an `evidence_id` in the returned evidence.

## Test-plan links

- API endpoint behavior: `test-plan.md §1`.
- Workflow controls and failure behavior: `test-plan.md §2`.
- Memo/verification grounding: `test-plan.md §7`.
