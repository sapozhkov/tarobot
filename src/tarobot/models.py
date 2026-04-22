from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class TarotCard:
    name: str
    arcana: str
    upright_meaning: str
    reversed_meaning: str
    suit: Optional[str] = None
    rank: Optional[str] = None


@dataclass(frozen=True)
class DrawnCard:
    position: int
    position_label: str
    card: TarotCard
    orientation: str

    @property
    def meaning(self) -> str:
        if self.orientation == "upright":
            return self.card.upright_meaning
        return self.card.reversed_meaning

    @property
    def display_name(self) -> str:
        suffix = "прямо" if self.orientation == "upright" else "перевернутая"
        return f"{self.card.name} ({suffix})"


@dataclass(frozen=True)
class ReadingRequest:
    question: str
    cards_count: int = 5
    seed: Optional[int] = None


@dataclass(frozen=True)
class ReadingArtifact:
    kind: str
    path: Path


@dataclass(frozen=True)
class ReadingNarrative:
    title: str
    summary: str
    card_sections: List[str]
    advice: str
    spoken_text: str


@dataclass(frozen=True)
class ReadingResult:
    request: ReadingRequest
    cards: List[DrawnCard]
    narrative: ReadingNarrative
    artifacts: List[ReadingArtifact]
    run_id: str
    run_dir: Path
    metadata: Dict[str, str] = field(default_factory=dict)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
