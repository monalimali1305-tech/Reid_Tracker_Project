from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Presence:
    total_seconds: float = 0.0
    entries: int = 0
    exits: int = 0


@dataclass(frozen=True)
class PresenceEvent:
    person_name: str
    event_type: str
    occurred_at: datetime


@dataclass
class PresenceTracker:
    absence_timeout: float = 1.0
    records: dict[str, Presence] = field(default_factory=dict)
    active_names: set[str] = field(default_factory=set)
    missing_seconds: dict[str, float] = field(default_factory=dict)
    events: list[PresenceEvent] = field(default_factory=list)

    def update(self, visible_names: set[str], elapsed: float, occurred_at: datetime) -> None:
        elapsed = max(0.0, elapsed)
        for name in self.active_names:
            self.records[name].total_seconds += elapsed
        for name in visible_names:
            self.missing_seconds.pop(name, None)
        for name in visible_names - self.active_names:
            self.records.setdefault(name, Presence()).entries += 1
            self.active_names.add(name)
            self.events.append(PresenceEvent(name, 'IN', occurred_at))
        for name in self.active_names - visible_names:
            missing = self.missing_seconds.get(name, 0.0) + elapsed
            if missing >= self.absence_timeout:
                self.records[name].exits += 1
                self.active_names.remove(name)
                self.missing_seconds.pop(name, None)
                self.events.append(PresenceEvent(name, 'OUT', occurred_at))
            else:
                self.missing_seconds[name] = missing