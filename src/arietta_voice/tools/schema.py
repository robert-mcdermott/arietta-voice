"""The small part of JSON Schema that tool arguments use.

Objects with typed properties (string, integer, number, boolean, enum),
``required`` and ``additionalProperties: false``. Anything outside this subset is
rejected when a tool loads, so a schema never silently means less than it says.
"""

from __future__ import annotations

from typing import Any

TYPES = {"string", "integer", "number", "boolean"}
PROPERTY_KEYS = {"type", "description", "enum", "minimum", "maximum", "maxLength"}
OBJECT_KEYS = {"type", "properties", "required", "additionalProperties"}
MAX_STRING_CHARS = 2000


def check_schema(schema: Any) -> list[str]:
    """Problems with a tool's parameter schema; empty when it is usable."""
    if not isinstance(schema, dict) or schema.get("type") != "object":
        return ["parameters must be a JSON schema with type 'object'"]
    problems = [f"unsupported schema key '{key}'" for key in schema if key not in OBJECT_KEYS]
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return [*problems, "properties must be an object"]
    for name, spec in properties.items():
        if not isinstance(spec, dict) or spec.get("type") not in TYPES:
            problems.append(f"property '{name}' needs a type: {', '.join(sorted(TYPES))}")
            continue
        problems += [
            f"property '{name}' has unsupported key '{key}'"
            for key in spec
            if key not in PROPERTY_KEYS
        ]
        if "enum" in spec and not (isinstance(spec["enum"], list) and spec["enum"]):
            problems.append(f"property '{name}' enum must be a non-empty list")
    required = schema.get("required", [])
    if not isinstance(required, list) or any(name not in properties for name in required):
        problems.append("required must list declared properties")
    if schema.get("additionalProperties", False) not in (False, True):
        problems.append("additionalProperties must be true or false")
    return problems


def validate_arguments(schema: dict[str, Any], arguments: Any) -> list[str]:
    """Problems with arguments for a schema that passed ``check_schema``."""
    if not isinstance(arguments, dict):
        return ["arguments must be an object"]
    properties = schema.get("properties", {})
    problems = [f"missing '{name}'" for name in schema.get("required", []) if name not in arguments]
    for name, value in arguments.items():
        spec = properties.get(name)
        if spec is None:
            if not schema.get("additionalProperties", False):
                problems.append(f"unexpected argument '{name}'")
            continue
        problems += [f"'{name}' {problem}" for problem in _check_value(spec, value)]
    return problems


def coerce_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    """Convert trigger captures, which are always text, to the declared types.

    A number where text is expected becomes text too ({"duration": 10}), since
    models often send one.
    """
    properties = schema.get("properties", {})
    converted = {}
    for name, value in arguments.items():
        kind = properties.get(name, {}).get("type")
        try:
            if kind == "integer" and isinstance(value, str):
                value = int(value)
            elif kind == "number" and isinstance(value, str):
                value = float(value)
            elif (
                kind == "string" and isinstance(value, (int, float)) and not isinstance(value, bool)
            ):
                value = f"{value:g}"
        except ValueError:
            pass  # Left as text; validation reports it.
        converted[name] = value
    return converted


def _check_value(spec: dict[str, Any], value: Any) -> list[str]:
    kind = spec["type"]
    if kind == "string":
        if not isinstance(value, str):
            return ["must be text"]
        if len(value) > spec.get("maxLength", MAX_STRING_CHARS):
            return ["is too long"]
    elif kind == "boolean":
        if not isinstance(value, bool):
            return ["must be true or false"]
    elif kind == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            return ["must be a whole number"]
    elif kind == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return ["must be a number"]
    if "enum" in spec and value not in spec["enum"]:
        return [f"must be one of {', '.join(map(str, spec['enum']))}"]
    if "minimum" in spec and value < spec["minimum"]:
        return [f"must be at least {spec['minimum']}"]
    if "maximum" in spec and value > spec["maximum"]:
        return [f"must be at most {spec['maximum']}"]
    return []
