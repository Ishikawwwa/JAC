"""Explicit inputs and parameterized, serializable execution plans."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping


def snapshot(value: Any) -> Any:
    """Copy JSON data and reject ambiguous/non-finite keys and values."""

    def validate(item: Any) -> None:
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise TypeError("JSON object keys must be strings")
            for child in item.values():
                validate(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                validate(child)
        elif isinstance(item, float) and not math.isfinite(item):
            raise ValueError("Non-finite numbers cannot be cached")
        elif item is not None and not isinstance(item, (str, bool, int, float)):
            raise TypeError(f"Not JSON data: {type(item).__name__}")

    validate(value)
    return json.loads(json.dumps(value, allow_nan=False))


def fingerprint(value: Any) -> str:
    encoded = json.dumps(snapshot(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Context:
    scope: str
    dependencies: Mapping[str, str] = field(default_factory=dict)
    policy_version: str = "1"

    def to_dict(self) -> dict[str, Any]:
        if not isinstance(self.scope, str) or not isinstance(self.policy_version, str):
            raise TypeError("Scope and policy version must be strings")
        if not self.scope or not self.policy_version:
            raise ValueError("Scope and policy version must be explicit and nonempty")
        if any(
            not isinstance(k, str) or not isinstance(v, str) for k, v in self.dependencies.items()
        ):
            raise TypeError("Dependency names and revisions must be strings")
        return {
            "scope": self.scope,
            "dependencies": dict(self.dependencies),
            "policy_version": self.policy_version,
        }


@dataclass(frozen=True)
class Request:
    task: str
    params: Mapping[str, Any]
    context: Context
    text: str = ""
    output_contract: str = "json-v1"

    def to_dict(self) -> dict[str, Any]:
        if not isinstance(self.task, str) or not isinstance(self.output_contract, str):
            raise TypeError("Task and output contract must be strings")
        if not self.task or not self.output_contract:
            raise ValueError("Task and output contract must be nonempty")
        if not isinstance(self.text, str):
            raise TypeError("Request text must be a string")
        return snapshot(
            {
                "task": self.task,
                "params": dict(self.params),
                "text": self.text,
                "context": self.context.to_dict(),
                "output_contract": self.output_contract,
            }
        )


@dataclass(frozen=True)
class Ref:
    """A reference to a request parameter or a prior node's JSON output.

    Ref.input("portfolio_id") binds structured parameters. Ref("positions", ("items",))
    reads a node result. No eval, generated expressions, or arbitrary imports are allowed.
    """

    source: str
    path: tuple[str | int, ...] = ()

    @classmethod
    def input(cls, name: str) -> Ref:
        return cls("$input", (name,))

    @classmethod
    def text(cls) -> Ref:
        return cls("$request", ("text",))


def _encode(value: Any) -> Any:
    if isinstance(value, Ref):
        return {"type": "ref", "source": value.source, "path": list(value.path)}
    if isinstance(value, dict):
        return {"type": "object", "items": {k: _encode(v) for k, v in value.items()}}
    if isinstance(value, (tuple, list)):
        return {"type": "array", "items": [_encode(v) for v in value]}
    return {"type": "literal", "value": snapshot(value)}


def _decode(value: dict[str, Any]) -> Any:
    kind = value["type"]
    if kind == "ref":
        return Ref(value["source"], tuple(value["path"]))
    if kind == "object":
        return {k: _decode(v) for k, v in value["items"].items()}
    if kind == "array":
        return [_decode(v) for v in value["items"]]
    if kind == "literal":
        return snapshot(value["value"])
    raise ValueError(f"Unknown binding type: {kind}")


def references(value: Any) -> list[Ref]:
    if isinstance(value, Ref):
        return [value]
    if isinstance(value, dict):
        return [ref for child in value.values() for ref in references(child)]
    if isinstance(value, (tuple, list)):
        return [ref for child in value for ref in references(child)]
    return []


def resolve(value: Any, params: Mapping[str, Any], results: Mapping[str, Any]) -> Any:
    if isinstance(value, Ref):
        result: Any = params if value.source == "$input" else results[value.source]
        for part in value.path:
            result = result[part]
        return snapshot(result)
    if isinstance(value, dict):
        return {key: resolve(child, params, results) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [resolve(child, params, results) for child in value]
    return snapshot(value)


@dataclass(frozen=True)
class Node:
    id: str
    operation: str
    arguments: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "operation": self.operation,
            "arguments": _encode(dict(self.arguments)),
        }


@dataclass(frozen=True)
class Plan:
    task: str
    version: str
    parameters: Mapping[str, str]
    nodes: tuple[Node, ...]
    output: Any
    output_contract: str = "json-v1"
    text_independent: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "version": self.version,
            "parameters": dict(self.parameters),
            "nodes": [n.to_dict() for n in self.nodes],
            "output": _encode(self.output),
            "output_contract": self.output_contract,
            "text_independent": self.text_independent,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Plan:
        return cls(
            task=value["task"],
            version=value["version"],
            parameters=dict(value["parameters"]),
            nodes=tuple(
                Node(n["id"], n["operation"], _decode(n["arguments"])) for n in value["nodes"]
            ),
            output=_decode(value["output"]),
            output_contract=value["output_contract"],
            text_independent=value.get("text_independent", False),
        )

    def validate(self, request: Request, operations: set[str]) -> None:
        if not isinstance(self.text_independent, bool):
            raise TypeError("text_independent must be an explicit boolean")
        if not isinstance(self.task, str) or not isinstance(self.output_contract, str):
            raise TypeError("Plan task and output contract must be strings")
        if not isinstance(self.version, str):
            raise TypeError("Plan version must be a string")
        if self.task != request.task or self.output_contract != request.output_contract:
            raise ValueError("Plan task or output contract does not match request")
        if not self.version or not self.nodes:
            raise ValueError("Plan requires a version and at least one node")
        params = snapshot(dict(request.params))
        if set(params) != set(self.parameters):
            raise ValueError("Structured inputs must exactly match the plan parameter schema")
        types = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "object": dict,
            "array": list,
        }
        for name, kind in self.parameters.items():
            if kind not in types:
                raise ValueError(f"Unknown parameter type: {kind}")
            if not isinstance(params[name], types[kind]) or (
                kind in {"integer", "number"} and isinstance(params[name], bool)
            ):
                raise TypeError(f"Invalid type for parameter {name}: expected {kind}")
        ids = [n.id for n in self.nodes]
        if any(
            not isinstance(node.id, str) or not isinstance(node.operation, str)
            for node in self.nodes
        ):
            raise TypeError("Node ids and operation names must be strings")
        if (
            len(ids) != len(set(ids))
            or any(n.startswith("$") for n in ids)
            or any(not n for n in ids)
        ):
            raise ValueError("Node ids must be unique, nonempty, and not start with $")
        edges: dict[str, set[str]] = {}
        all_refs = references(self.output)
        for node in self.nodes:
            if node.operation not in operations:
                raise ValueError(f"Unregistered operation: {node.operation}")
            refs = references(dict(node.arguments))
            all_refs.extend(refs)
            edges[node.id] = {r.source for r in refs if not r.source.startswith("$")}
        used_inputs = set()
        for ref in all_refs:
            if any(not isinstance(p, (str, int)) or isinstance(p, bool) for p in ref.path):
                raise ValueError("Reference paths must contain strings or integers")
            if ref.source == "$input":
                if not ref.path or ref.path[0] not in self.parameters:
                    raise ValueError("Input references must name a declared parameter")
                used_inputs.add(ref.path[0])
            elif ref.source == "$request":
                if ref.path != ("text",):
                    raise ValueError("Request references may bind only the request text")
            elif ref.source not in edges:
                raise ValueError(f"Reference to missing node: {ref.source}")
        if used_inputs != set(self.parameters):
            raise ValueError("Every declared parameter must be bound through an input reference")
        remaining = dict(edges)
        completed: set[str] = set()
        while remaining:
            ready = {key for key, deps in remaining.items() if deps <= completed}
            if not ready:
                raise ValueError("Execution plan contains a cycle")
            completed.update(ready)
            remaining = {key: deps for key, deps in remaining.items() if key not in ready}
