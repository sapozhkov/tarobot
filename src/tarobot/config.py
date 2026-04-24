from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


def _read_env_file(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}

    values: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
            value = value[1:-1]
        values[key] = value
    return values


def _get_bool(values: Dict[str, str], key: str, default: bool) -> bool:
    raw = values.get(key)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_float(values: Dict[str, str], key: str, default: float) -> float:
    raw = values.get(key)
    if raw is None or raw == "":
        return default
    return float(raw)


def _get_int(values: Dict[str, str], key: str, default: int) -> int:
    raw = values.get(key)
    if raw is None or raw == "":
        return default
    return int(raw)


def _get_str(values: Dict[str, str], key: str, default: Optional[str] = None) -> Optional[str]:
    raw = values.get(key)
    if raw is None:
        return default
    return raw or default


@dataclass(frozen=True)
class TarobotSettings:
    llm_provider: str
    tts_provider: str
    yandex_api_key: Optional[str]
    yandex_folder_id: Optional[str]
    yandex_llm_model: str
    yandex_llm_model_uri: Optional[str]
    yandex_llm_temperature: float
    yandex_llm_max_tokens: int
    yandex_llm_timeout_seconds: int
    yandex_tts_voice: str
    yandex_tts_role: Optional[str]
    yandex_tts_speed: float
    yandex_tts_pitch_shift: float
    yandex_tts_loudness_normalization_type: str
    yandex_tts_volume: Optional[float]
    yandex_tts_unsafe_mode: bool
    yandex_tts_timeout_seconds: int

    def resolve_yandex_model_uri(self) -> Optional[str]:
        if self.yandex_llm_model_uri:
            return self.yandex_llm_model_uri
        if not self.yandex_folder_id:
            return None
        model_name = self.yandex_llm_model.strip()
        return f"gpt://{self.yandex_folder_id}/{model_name}/latest"


def load_settings(env_path: Optional[Path] = None) -> TarobotSettings:
    values: Dict[str, str] = {}
    values.update(_read_env_file(env_path or Path.cwd() / ".env.local"))

    for key, value in os.environ.items():
        if key.startswith("TAROBOT_") or key.startswith("YANDEX_"):
            values[key] = value

    raw_volume = _get_str(values, "YANDEX_TTS_VOLUME")
    yandex_tts_volume = float(raw_volume) if raw_volume not in {None, ""} else None

    return TarobotSettings(
        llm_provider=_get_str(values, "TAROBOT_LLM_PROVIDER", "mock") or "mock",
        tts_provider=_get_str(values, "TAROBOT_TTS_PROVIDER", "macos") or "macos",
        yandex_api_key=_get_str(values, "YANDEX_API_KEY"),
        yandex_folder_id=_get_str(values, "YANDEX_FOLDER_ID"),
        yandex_llm_model=_get_str(values, "YANDEX_LLM_MODEL", "yandexgpt") or "yandexgpt",
        yandex_llm_model_uri=_get_str(values, "YANDEX_LLM_MODEL_URI"),
        yandex_llm_temperature=_get_float(values, "YANDEX_LLM_TEMPERATURE", 0.75),
        yandex_llm_max_tokens=_get_int(values, "YANDEX_LLM_MAX_TOKENS", 1800),
        yandex_llm_timeout_seconds=_get_int(values, "YANDEX_LLM_TIMEOUT_SECONDS", 60),
        yandex_tts_voice=_get_str(values, "YANDEX_TTS_VOICE", "marina") or "marina",
        yandex_tts_role=_get_str(values, "YANDEX_TTS_ROLE"),
        yandex_tts_speed=_get_float(values, "YANDEX_TTS_SPEED", 1.0),
        yandex_tts_pitch_shift=_get_float(values, "YANDEX_TTS_PITCH_SHIFT", 0.0),
        yandex_tts_loudness_normalization_type=(
            _get_str(values, "YANDEX_TTS_LOUDNESS_NORMALIZATION_TYPE", "LUFS") or "LUFS"
        ),
        yandex_tts_volume=yandex_tts_volume,
        yandex_tts_unsafe_mode=_get_bool(values, "YANDEX_TTS_UNSAFE_MODE", False),
        yandex_tts_timeout_seconds=_get_int(values, "YANDEX_TTS_TIMEOUT_SECONDS", 30),
    )
