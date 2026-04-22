from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, List, Optional
from uuid import uuid4

from .models import DrawnCard, ReadingArtifact, ReadingNarrative, ReadingRequest, utc_now_iso


def slugify(text: str, max_length: int = 36) -> str:
    slug = re.sub(r"[^a-zA-Z0-9а-яА-Я]+", "-", text.strip()).strip("-").lower()
    return slug[:max_length] or "reading"


class ArtifactStore:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir

    def create_run_dir(self, request: ReadingRequest) -> tuple[str, Path]:
        run_id = f"{utc_now_iso().replace(':', '').replace('+00:00', 'z')}-{uuid4().hex[:8]}"
        run_dir = self.base_dir / f"{run_id}-{slugify(request.question)}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_id, run_dir

    def save_request(self, run_dir: Path, request: ReadingRequest) -> ReadingArtifact:
        path = run_dir / "request.json"
        payload = {
            "question": request.question,
            "cards_count": request.cards_count,
            "seed": request.seed,
            "created_at": utc_now_iso(),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return ReadingArtifact(kind="request", path=path)

    def save_cards(self, run_dir: Path, cards: Iterable[DrawnCard]) -> ReadingArtifact:
        path = run_dir / "cards.json"
        payload = [
            {
                "position": card.position,
                "position_label": card.position_label,
                "name": card.card.name,
                "arcana": card.card.arcana,
                "suit": card.card.suit,
                "rank": card.card.rank,
                "orientation": card.orientation,
                "meaning": card.meaning,
            }
            for card in cards
        ]
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return ReadingArtifact(kind="cards", path=path)

    def save_narrative(self, run_dir: Path, narrative: ReadingNarrative) -> List[ReadingArtifact]:
        text_path = run_dir / "reading.txt"
        spoken_path = run_dir / "spoken_text.txt"
        json_path = run_dir / "reading.json"

        reading_text = "\n\n".join(
            [
                narrative.title,
                narrative.summary,
                *narrative.card_sections,
                narrative.advice,
            ]
        )
        text_path.write_text(reading_text, encoding="utf-8")
        spoken_path.write_text(narrative.spoken_text, encoding="utf-8")
        json_path.write_text(
            json.dumps(
                {
                    "title": narrative.title,
                    "summary": narrative.summary,
                    "card_sections": narrative.card_sections,
                    "advice": narrative.advice,
                    "spoken_text": narrative.spoken_text,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        return [
            ReadingArtifact(kind="reading_text", path=text_path),
            ReadingArtifact(kind="spoken_text", path=spoken_path),
            ReadingArtifact(kind="reading_json", path=json_path),
        ]

    def save_audio(self, run_dir: Path, audio_path: Optional[Path]) -> Optional[ReadingArtifact]:
        if audio_path is None:
            return None
        return ReadingArtifact(kind="audio", path=audio_path)

    def save_manifest(
        self,
        run_id: str,
        run_dir: Path,
        artifacts: Iterable[ReadingArtifact],
        metadata: dict,
    ) -> ReadingArtifact:
        path = run_dir / "manifest.json"
        payload = {
            "run_id": run_id,
            "created_at": utc_now_iso(),
            "run_dir": str(run_dir),
            "metadata": metadata,
            "artifacts": [
                {
                    "kind": artifact.kind,
                    "path": str(artifact.path),
                }
                for artifact in artifacts
            ],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return ReadingArtifact(kind="manifest", path=path)
