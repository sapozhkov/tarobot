from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .artifacts import ArtifactStore
from .cards import draw_cards
from .config import TarobotSettings, load_settings
from .llm import MockLLMService, ReadingGenerator, YandexLLMService
from .models import ReadingArtifact, ReadingRequest, ReadingResult
from .tts import MacOSTTSProvider, SilentTTSProvider, SpeechSynthesizer, YandexSpeechKitTTSProvider


class TarobotApp:
    def __init__(
        self,
        artifact_store: ArtifactStore,
        llm_service: Optional[ReadingGenerator] = None,
        tts_provider: Optional[SpeechSynthesizer] = None,
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
            audio_path = self.tts_provider.synthesize(narrative.speech_plan, run_dir, "reading_audio")
            audio_artifact = self.artifact_store.save_audio(run_dir, audio_path)
            if audio_artifact:
                artifacts.append(audio_artifact)

        metadata = {"llm_backend": getattr(self.llm_service, "backend_name", self.llm_service.__class__.__name__)}
        metadata.update(self.llm_service.metadata())
        if self.tts_provider:
            metadata.update(self.tts_provider.metadata())
        else:
            metadata["tts_backend"] = "none"
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


def build_default_app(
    base_dir: Path,
    enable_tts: bool = True,
    tts_voice: str = "Milena",
    tts_rate: int = 165,
    llm_provider_override: Optional[str] = None,
    tts_provider_override: Optional[str] = None,
) -> TarobotApp:
    settings = load_settings()
    llm_service = _build_llm_service(settings, llm_provider_override)
    tts_provider = _build_tts_provider(settings, enable_tts, tts_voice, tts_rate, tts_provider_override)
    return TarobotApp(
        artifact_store=ArtifactStore(base_dir=base_dir),
        llm_service=llm_service,
        tts_provider=tts_provider,
    )


def _build_llm_service(settings: TarobotSettings, override: Optional[str]) -> ReadingGenerator:
    provider = (override or settings.llm_provider or "mock").lower()
    if provider in {"auto", "mock"}:
        return MockLLMService()
    if provider == "yandex":
        if not settings.yandex_api_key:
            raise RuntimeError("Для TAROBOT_LLM_PROVIDER=yandex нужен заполненный YANDEX_API_KEY")
        model_uri = settings.resolve_yandex_model_uri()
        if not model_uri:
            raise RuntimeError(
                "Чтобы включить Yandex LLM, укажи YANDEX_FOLDER_ID или готовый YANDEX_LLM_MODEL_URI"
            )
        return YandexLLMService(
            api_key=settings.yandex_api_key,
            model_uri=model_uri,
            temperature=settings.yandex_llm_temperature,
            max_tokens=settings.yandex_llm_max_tokens,
            timeout_seconds=settings.yandex_llm_timeout_seconds,
            tts_voice=settings.yandex_tts_voice,
            tts_role=settings.yandex_tts_role,
            tts_speed=settings.yandex_tts_speed,
            tts_pitch_shift=settings.yandex_tts_pitch_shift,
        )
    raise RuntimeError(f"Неподдерживаемый LLM provider: {provider}")


def _build_tts_provider(
    settings: TarobotSettings,
    enable_tts: bool,
    macos_voice: str,
    macos_rate: int,
    override: Optional[str],
) -> SpeechSynthesizer:
    if not enable_tts:
        return SilentTTSProvider()

    provider = (override or settings.tts_provider or "macos").lower()
    if provider in {"macos", "say", "auto"}:
        return MacOSTTSProvider(voice=macos_voice, rate=macos_rate)
    if provider == "yandex":
        if not settings.yandex_api_key:
            raise RuntimeError("Для TAROBOT_TTS_PROVIDER=yandex нужен заполненный YANDEX_API_KEY")
        return YandexSpeechKitTTSProvider(
            api_key=settings.yandex_api_key,
            voice=settings.yandex_tts_voice,
            role=settings.yandex_tts_role,
            speed=settings.yandex_tts_speed,
            pitch_shift=settings.yandex_tts_pitch_shift,
            loudness_normalization_type=settings.yandex_tts_loudness_normalization_type,
            volume=settings.yandex_tts_volume,
            unsafe_mode=settings.yandex_tts_unsafe_mode,
            timeout_seconds=settings.yandex_tts_timeout_seconds,
        )
    raise RuntimeError(f"Неподдерживаемый TTS provider: {provider}")
