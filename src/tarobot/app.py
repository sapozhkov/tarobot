from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .artifacts import ArtifactStore
from .cards import draw_cards
from .llm import MockLLMService
from .models import ReadingArtifact, ReadingRequest, ReadingResult
from .tts import TTSProvider


class TarobotApp:
    def __init__(
        self,
        artifact_store: ArtifactStore,
        llm_service: Optional[MockLLMService] = None,
        tts_provider: Optional[TTSProvider] = None,
    ) -> None:
        self.artifact_store = artifact_store
        self.llm_service = llm_service or MockLLMService()
        self.tts_provider = tts_provider

    def run(self, request: ReadingRequest) -> ReadingResult:
        run_id, run_dir = self.artifact_store.create_run_dir(request)
        cards = draw_cards(request.cards_count, seed=request.seed)
        narrative = self.llm_service.generate_reading(request, cards)

        artifacts: List[ReadingArtifact] = []
        artifacts.append(self.artifact_store.save_request(run_dir, request))
        artifacts.append(self.artifact_store.save_cards(run_dir, cards))
        artifacts.extend(self.artifact_store.save_narrative(run_dir, narrative))

        audio_artifact = None
        if self.tts_provider:
            audio_path = self.tts_provider.synthesize(narrative.spoken_text, run_dir, "reading_audio")
            audio_artifact = self.artifact_store.save_audio(run_dir, audio_path)
            if audio_artifact:
                artifacts.append(audio_artifact)

        metadata = {
            "llm_backend": self.llm_service.__class__.__name__,
            "tts_backend": getattr(self.tts_provider, "backend_name", "none"),
        }
        artifacts.append(self.artifact_store.save_manifest(run_id, run_dir, artifacts, metadata))

        return ReadingResult(
            request=request,
            cards=cards,
            narrative=narrative,
            artifacts=artifacts,
            run_id=run_id,
            run_dir=run_dir,
            metadata=metadata,
        )


def build_default_app(base_dir: Path, enable_tts: bool = True) -> TarobotApp:
    from .tts import MacOSTTSProvider, SilentTTSProvider

    tts_provider = MacOSTTSProvider() if enable_tts else SilentTTSProvider()
    return TarobotApp(
        artifact_store=ArtifactStore(base_dir=base_dir),
        tts_provider=tts_provider,
    )
