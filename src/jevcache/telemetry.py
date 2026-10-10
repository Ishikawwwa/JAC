"""Per-request measurements; payloads are deliberately absent from trace events."""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Event:
    name: str
    kind: str
    status: str
    elapsed_ms: float
    avoided_ms: float = 0.0


@dataclass
class Trace:
    request_id: str
    task: str
    path: str = "miss"
    elapsed_ms: float = 0.0
    llm_calls: int = 0
    tool_calls: int = 0
    jev_calls: int = 0
    plan_hit: bool = False
    cached_nodes: int = 0
    coalesced_nodes: int = 0
    executed_nodes: int = 0
    decision_status: str = "not_needed"
    events: list[Event] = field(default_factory=list)
    mode: str = "cache_first"
    race_winner: str | None = None
    speculative_llm_calls: int = 0
    llm_cancellation_requested: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunResult:
    value: Any
    trace: Trace

    @property
    def path(self) -> str:
        return self.trace.path
