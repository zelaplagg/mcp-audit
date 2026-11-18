"""A small JSON Schema validator for tool arguments (stdlib only).

It covers what MCP tool schemas actually use: type, enum, const, required, properties,
additionalProperties, items, numeric and length bounds, pattern, anyOf/oneOf/allOf.
Unknown keywords are ignored, so it can only under-report, never reject a valid call.
"""
from __future__ import annotations

import re
from typing import Any

MAX_ERRORS = 20
_PATTERN_LIMIT = 200  # server-supplied regexes are untrusted; refuse long ones
_PATTERN_INPUT_LIMIT = 2000


def _kind(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _is_type(value: Any, expected: str) -> bool:
    kind = _kind(value)
    if expected == "number":
        return kind in ("integer", "number")
    if expected == "integer":
        return kind == "integer" or (kind == "number" and float(value).is_integer())
    return kind == expected


def validate(value: Any, schema: Any, path: str = "$") -> list[str]:
    """Return human-readable errors; an empty list means the value is valid."""
    errors: list[str] = []
    _validate(value, schema, path, errors)
    return errors[:MAX_ERRORS]


def _validate(value: Any, schema: Any, path: str, errors: list[str]) -> None:
    if not isinstance(schema, dict) or len(errors) >= MAX_ERRORS:
        return
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: expected {schema['const']!r}, got {value!r}")
    if isinstance(schema.get("enum"), list) and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} is not one of {schema['enum']!r}")

    declared = schema.get("type")
    types = declared if isinstance(declared, list) else [declared] if isinstance(declared, str) else []
    if types and not any(_is_type(value, t) for t in types):
        errors.append(f"{path}: expected {'/'.join(types)}, got {_kind(value)}")
        return

    for key in ("anyOf", "oneOf"):
        options = schema.get(key)
        if isinstance(options, list) and options and not any(not validate(value, o, path) for o in options):
            errors.append(f"{path}: matches none of the {key} alternatives")
    if isinstance(schema.get("allOf"), list):
        for option in schema["allOf"]:
            _validate(value, option, path, errors)

    kind = _kind(value)
    if kind in ("integer", "number"):
        _bounds(value, schema, path, errors)
    elif kind == "string":
        _string(value, schema, path, errors)
    elif kind == "array":
        _array(value, schema, path, errors)
    elif kind == "object":
        _object(value, schema, path, errors)


def _bounds(value: float, schema: dict[str, Any], path: str, errors: list[str]) -> None:
    low, high = schema.get("minimum"), schema.get("maximum")
    if isinstance(low, (int, float)) and value < low:
        errors.append(f"{path}: {value} is below the minimum {low}")
    if isinstance(high, (int, float)) and value > high:
        errors.append(f"{path}: {value} is above the maximum {high}")
    low, high = schema.get("exclusiveMinimum"), schema.get("exclusiveMaximum")
    if isinstance(low, (int, float)) and not isinstance(low, bool) and value <= low:
        errors.append(f"{path}: {value} must be greater than {low}")
    if isinstance(high, (int, float)) and not isinstance(high, bool) and value >= high:
        errors.append(f"{path}: {value} must be less than {high}")


def _string(value: str, schema: dict[str, Any], path: str, errors: list[str]) -> None:
    low, high = schema.get("minLength"), schema.get("maxLength")
    if isinstance(low, int) and len(value) < low:
        errors.append(f"{path}: shorter than minLength {low}")
    if isinstance(high, int) and len(value) > high:
        errors.append(f"{path}: longer than maxLength {high}")
    pattern = schema.get("pattern")
    if isinstance(pattern, str) and len(pattern) <= _PATTERN_LIMIT:
        try:
            if not re.search(pattern, value[:_PATTERN_INPUT_LIMIT]):
                errors.append(f"{path}: does not match pattern {pattern!r}")
        except re.error:
            pass


def _array(value: list[Any], schema: dict[str, Any], path: str, errors: list[str]) -> None:
    low, high = schema.get("minItems"), schema.get("maxItems")
    if isinstance(low, int) and len(value) < low:
        errors.append(f"{path}: fewer than minItems {low}")
    if isinstance(high, int) and len(value) > high:
        errors.append(f"{path}: more than maxItems {high}")
    if isinstance(schema.get("items"), dict):
        for index, item in enumerate(value):
            _validate(item, schema["items"], f"{path}[{index}]", errors)


def _object(value: dict[str, Any], schema: dict[str, Any], path: str, errors: list[str]) -> None:
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    required = schema.get("required")
    for name in required if isinstance(required, list) else []:
        if name not in value:
            errors.append(f"{path}: missing required property '{name}'")
    for name, item in value.items():
        if name in properties:
            _validate(item, properties[name], f"{path}.{name}", errors)
        elif schema.get("additionalProperties") is False:
            errors.append(f"{path}: unexpected property '{name}'")
        elif isinstance(schema.get("additionalProperties"), dict):
            _validate(item, schema["additionalProperties"], f"{path}.{name}", errors)
