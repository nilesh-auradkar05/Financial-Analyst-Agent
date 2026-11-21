"""The JSON a fake structured-output provider computes from a request's JSON schema.

Shared by the Foundry (Anthropic `output_config.format`) and Bedrock (Converse
`outputConfig.textFormat`) fakes: a request that asks for JSON-schema output gets a text reply whose
JSON conforms to the schema, every string field holding the requested model and every other field
its type's zero (the same convention as their tool replies). The schema is read from the request,
never supplied by a test.
"""

from __future__ import annotations

import json
from typing import Any

_ZEROS: dict[str, Any] = {"integer": 0, "number": 0.0, "boolean": False, "array": [], "null": None}


def _resolve(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    ref = schema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/"):
        node: Any = root
        for part in ref[2:].split("/"):
            node = node[part]
        return _resolve(node, root)
    return schema


def value_for(schema: dict[str, Any], text: str, root: dict[str, Any] | None = None) -> Any:
    root = root if root is not None else schema
    schema = _resolve(schema, root)
    for key in ("anyOf", "oneOf"):
        if key in schema:
            return value_for(schema[key][0], text, root)
    kind = schema.get("type")
    if isinstance(kind, list):
        kind = kind[0]
    if kind == "object":
        return {
            name: value_for(spec, text, root) for name, spec in schema.get("properties", {}).items()
        }
    if kind == "string" or kind is None:
        return text
    return _ZEROS.get(str(kind), text)


def json_reply(schema: dict[str, Any] | str, text: str) -> str:
    """The JSON text of a value conforming to `schema` (a dict, or its JSON text)."""
    parsed = json.loads(schema) if isinstance(schema, str) else schema
    return json.dumps(value_for(parsed, text))
