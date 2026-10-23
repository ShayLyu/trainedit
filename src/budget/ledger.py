"""Counts environment steps by purpose. The unit is one control step."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Ledger:
    entries: list[tuple[str, int, int]] = field(default_factory=list)
    total: int = 0

    def add(self, kind: str, steps: int) -> int:
        steps = int(steps)
        if steps < 0:
            raise ValueError("steps must be non-negative")
        self.total += steps
        self.entries.append((kind, steps, self.total))
        return self.total

    def by_kind(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for kind, steps, _ in self.entries:
            out[kind] = out.get(kind, 0) + steps
        return out

    def to_dict(self) -> dict:
        return {"total": self.total, "by_kind": self.by_kind()}
