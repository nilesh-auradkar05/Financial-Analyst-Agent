"""The reranked-retrieval module under test, resolved at run time.

`search_reranked` and `RERANK_CANDIDATE_POOL` arrive with S5.3 (test-plan §25). Looking them up on
the module inside each test makes a missing one fail that test, not the collection of a file whose
other tests cover rows that already hold.
"""

from __future__ import annotations

import importlib
from typing import Any

rerank_api: Any = importlib.import_module("app.components.retrieval.reranked_hybrid_retrieve")
