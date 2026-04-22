from __future__ import annotations

import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Dict, Optional

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
                segment_paths.append((segment, wav_segment_path))

            wav_path = output_dir / f"{stem}.wav"
            self._combine_segments(segment_paths, wav_path)

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

    def _combine_segments(self, segment_paths: list[tuple[SpeechSegment, Path]], output_path: Path) -> None:
        if not segment_paths:
            raise ValueError("speech plan must contain at least one segment")

        with wave.open(str(output_path), "wb") as writer:
            params_set = False
            frame_rate = 0
            frame_width = 0

            for segment, segment_path in segment_paths:
                with wave.open(str(segment_path), "rb") as reader:
                    if not params_set:
                        writer.setparams(reader.getparams())
                        frame_rate = reader.getframerate()
                        frame_width = reader.getnchannels() * reader.getsampwidth()
                        params_set = True

                    writer.writeframes(reader.readframes(reader.getnframes()))

                if segment.pause_ms > 0:
                    silence_frames = int(frame_rate * segment.pause_ms / 1000)
                    writer.writeframes(b"\x00" * silence_frames * frame_width)

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


class SilentTTSProvider(SpeechSynthesizer):
    backend_name = "silent"

    def synthesize(self, plan: SpeechPlan, output_dir: Path, stem: str) -> Optional[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        transcript_path = output_dir / f"{stem}.txt"
        transcript_path.write_text(plan.full_text, encoding="utf-8")
        return transcript_path


TTSProvider = SpeechSynthesizer
