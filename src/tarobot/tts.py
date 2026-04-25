from __future__ import annotations

import base64
import io
import json
import re
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .models import SpeechPlan, SpeechSegment


class SpeechSynthesizer:
    backend_name = "unknown"

    def synthesize(self, plan: SpeechPlan, output_dir: Path, stem: str) -> Optional[Path]:
        raise NotImplementedError

    def metadata(self) -> Dict[str, str]:
        return {"tts_backend": self.backend_name}


class MacOSTTSProvider(SpeechSynthesizer):
    backend_name = "macos-say"

    def __init__(self, voice: str = "Milena", rate: int = 165) -> None:
        self.voice = voice
        self.rate = rate

    def metadata(self) -> Dict[str, str]:
        return {
            "tts_backend": self.backend_name,
            "tts_voice": self.voice,
            "tts_rate": str(self.rate),
            "tts_mode": "segmented",
        }

    def synthesize(self, plan: SpeechPlan, output_dir: Path, stem: str) -> Optional[Path]:
        say_bin = shutil.which("say")
        if not say_bin:
            return None

        output_dir.mkdir(parents=True, exist_ok=True)
        afconvert_bin = shutil.which("afconvert")
        if not afconvert_bin:
            aiff_path = output_dir / f"{stem}.aiff"
            self._synthesize_segment(say_bin, plan.full_text, self.rate, aiff_path)
            return aiff_path

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            segment_paths = []
            for index, segment in enumerate(plan.segments, start=1):
                raw_segment_path = tmp_path / f"{stem}_segment_{index:02d}.aiff"
                wav_segment_path = tmp_path / f"{stem}_segment_{index:02d}.wav"
                self._synthesize_segment(
                    say_bin,
                    segment.text,
                    self._rate_for_segment(segment),
                    raw_segment_path,
                )
                self._convert_to_wav(afconvert_bin, raw_segment_path, wav_segment_path)
                segment_paths.append((segment.pause_ms, wav_segment_path))

            wav_path = output_dir / f"{stem}.wav"
            _combine_wave_files(segment_paths, wav_path)

        return self._convert_to_m4a(afconvert_bin, wav_path)

    def _synthesize_segment(self, say_bin: str, text: str, rate: int, output_path: Path) -> None:
        subprocess.run(
            [
                say_bin,
                "-v",
                self.voice,
                "-r",
                str(rate),
                "-o",
                str(output_path),
                text,
            ],
            check=True,
        )

    def _rate_for_segment(self, segment: SpeechSegment) -> int:
        if segment.section == "intro":
            return max(120, self.rate - 10)
        if segment.section == "outro":
            return max(120, self.rate - 15)
        return self.rate

    def _convert_to_wav(self, afconvert_bin: str, input_path: Path, output_path: Path) -> None:
        subprocess.run(
            [
                afconvert_bin,
                "-f",
                "WAVE",
                "-d",
                "LEI16",
                str(input_path),
                str(output_path),
            ],
            check=True,
        )

    def _convert_to_m4a(self, afconvert_bin: str, wav_path: Path) -> Path:
        m4a_path = wav_path.with_suffix(".m4a")
        try:
            subprocess.run(
                [
                    afconvert_bin,
                    "-f",
                    "m4af",
                    "-d",
                    "aac",
                    str(wav_path),
                    str(m4a_path),
                ],
                check=True,
            )
            wav_path.unlink(missing_ok=True)
            return m4a_path
        except subprocess.CalledProcessError:
            return wav_path


class YandexSpeechKitTTSProvider(SpeechSynthesizer):
    backend_name = "yandex-speechkit-v3"
    endpoint = "https://tts.api.cloud.yandex.net/tts/v3/utteranceSynthesis"
    max_request_chars = 240

    def __init__(
        self,
        *,
        api_key: str,
        voice: str,
        role: Optional[str] = None,
        speed: float = 1.0,
        pitch_shift: float = 0.0,
        loudness_normalization_type: str = "LUFS",
        volume: Optional[float] = None,
        unsafe_mode: bool = False,
        timeout_seconds: int = 30,
    ) -> None:
        self.api_key = api_key
        self.voice = voice
        self.role = role
        self.speed = speed
        self.pitch_shift = pitch_shift
        self.loudness_normalization_type = loudness_normalization_type
        self.volume = volume
        self.unsafe_mode = unsafe_mode
        self.timeout_seconds = timeout_seconds

    def metadata(self) -> Dict[str, str]:
        metadata = {
            "tts_backend": self.backend_name,
            "tts_voice": self.voice,
            "tts_speed": str(self.speed),
            "tts_pitch_shift": str(self.pitch_shift),
            "tts_loudness_normalization_type": self.loudness_normalization_type,
            "tts_mode": "tts_markup",
        }
        if self.role:
            metadata["tts_role"] = self.role
        if self.volume is not None:
            metadata["tts_volume"] = str(self.volume)
        return metadata

    def synthesize(self, plan: SpeechPlan, output_dir: Path, stem: str) -> Optional[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        wav_segments: List[Tuple[int, bytes]] = []

        for segment in plan.segments:
            chunks = _split_text_for_yandex_tts(_prepare_text_for_yandex_tts(segment.text), self.max_request_chars)
            for index, chunk_text in enumerate(chunks):
                pause_ms = segment.pause_ms if index == len(chunks) - 1 else 180
                wav_segments.append((pause_ms, self._synthesize_chunk(chunk_text)))

        if not wav_segments:
            return None

        output_path = output_dir / f"{stem}.wav"
        _combine_wave_bytes(wav_segments, output_path)
        return output_path

    def _synthesize_chunk(self, text: str) -> bytes:
        payload: Dict[str, object] = {
            "text": text,
            "hints": self._hints(),
            "outputAudioSpec": {"containerAudio": {"containerAudioType": "WAV"}},
            "loudnessNormalizationType": self.loudness_normalization_type,
            "unsafeMode": self.unsafe_mode,
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = Request(
            self.endpoint,
            data=data,
            headers={
                "Authorization": f"Api-Key {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(req, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Запрос к Yandex SpeechKit завершился ошибкой HTTP {exc.code}: {details}") from exc
        except URLError as exc:
            raise RuntimeError(f"Не удалось выполнить запрос к Yandex SpeechKit: {exc.reason}") from exc

        result = body.get("result", body)
        audio_chunk = result.get("audioChunk", {})
        encoded = audio_chunk.get("data")
        if not encoded:
            raise RuntimeError("Ответ Yandex SpeechKit не содержит audioChunk.data")
        return base64.b64decode(encoded)

    def _hints(self) -> List[Dict[str, object]]:
        hints: List[Dict[str, object]] = [{"voice": self.voice}]
        if self.role:
            hints.append({"role": self.role})
        if abs(self.speed - 1.0) > 1e-9:
            hints.append({"speed": str(self.speed)})
        if abs(self.pitch_shift) > 1e-9:
            hints.append({"pitchShift": str(self.pitch_shift)})
        if self.volume is not None:
            hints.append({"volume": str(self.volume)})
        return hints


class SilentTTSProvider(SpeechSynthesizer):
    backend_name = "silent"

    def synthesize(self, plan: SpeechPlan, output_dir: Path, stem: str) -> Optional[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        transcript_path = output_dir / f"{stem}.txt"
        transcript_path.write_text(plan.full_text, encoding="utf-8")
        return transcript_path


def _split_text_for_yandex_tts(text: str, max_chars: int) -> List[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= max_chars:
        return [normalized]

    parts = re.split(r"(?<=[.!?…])\s+", normalized)
    chunks: List[str] = []
    current = ""

    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(part) > max_chars:
            comma_parts = re.split(r"(?<=[,;:])\s+", part)
            for comma_part in comma_parts:
                comma_part = comma_part.strip()
                if not comma_part:
                    continue
                chunks.extend(_split_hard(comma_part, max_chars))
            continue

        tentative = f"{current} {part}".strip()
        if current and len(tentative) > max_chars:
            chunks.append(current)
            current = part
        else:
            current = tentative

    if current:
        chunks.append(current)
    return chunks or [normalized[:max_chars]]


def _split_hard(text: str, max_chars: int) -> List[str]:
    words = text.split()
    chunks: List[str] = []
    current = ""

    for word in words:
        tentative = f"{current} {word}".strip()
        if current and len(tentative) > max_chars:
            chunks.append(current)
            current = word
        else:
            current = tentative

    if current:
        chunks.append(current)
    return chunks


def _prepare_text_for_yandex_tts(text: str) -> str:
    replacements = {
        "для проекта Tarobot": "для проекта",
        "для проекта tarobot": "для проекта",
        "для Tarobot": "для проекта",
        "для tarobot": "для проекта",
        "проекта Tarobot": "проекта",
        "проекта tarobot": "проекта",
        "проект Tarobot": "проект",
        "проект tarobot": "проект",
        "Tarobot'а": "проекта",
        "tarobot'а": "проекта",
        "Tarobot": "оракул",
        "tarobot": "оракул",
        "Таробот": "оракул",
        "теработ": "оракул",
        "LLM": "эл-эл-эм",
        "MVP": "эм-ви-пи",
        "POC": "пи-о-си",
    }
    prepared = text
    for source, target in replacements.items():
        prepared = prepared.replace(source, target)
    return prepared


def _combine_wave_files(segment_paths: Iterable[Tuple[int, Path]], output_path: Path) -> None:
    with wave.open(str(output_path), "wb") as writer:
        params_set = False
        frame_rate = 0
        frame_width = 0

        for pause_ms, segment_path in segment_paths:
            with wave.open(str(segment_path), "rb") as reader:
                if not params_set:
                    writer.setparams(reader.getparams())
                    frame_rate = reader.getframerate()
                    frame_width = reader.getnchannels() * reader.getsampwidth()
                    params_set = True

                writer.writeframes(reader.readframes(reader.getnframes()))

            if pause_ms > 0:
                writer.writeframes(b"\x00" * int(frame_rate * pause_ms / 1000) * frame_width)


def _combine_wave_bytes(segments: Iterable[Tuple[int, bytes]], output_path: Path) -> None:
    with wave.open(str(output_path), "wb") as writer:
        params_set = False
        frame_rate = 0
        frame_width = 0

        for pause_ms, wav_bytes in segments:
            with wave.open(io.BytesIO(wav_bytes), "rb") as reader:
                if not params_set:
                    writer.setparams(reader.getparams())
                    frame_rate = reader.getframerate()
                    frame_width = reader.getnchannels() * reader.getsampwidth()
                    params_set = True

                writer.writeframes(reader.readframes(reader.getnframes()))

            if pause_ms > 0:
                writer.writeframes(b"\x00" * int(frame_rate * pause_ms / 1000) * frame_width)


TTSProvider = SpeechSynthesizer
