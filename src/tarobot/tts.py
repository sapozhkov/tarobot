from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional


class TTSProvider:
    backend_name = "unknown"

    def synthesize(self, text: str, output_dir: Path, stem: str) -> Optional[Path]:
        raise NotImplementedError


class MacOSTTSProvider(TTSProvider):
    backend_name = "macos-say"

    def __init__(self, voice: str = "Milena") -> None:
        self.voice = voice

    def synthesize(self, text: str, output_dir: Path, stem: str) -> Optional[Path]:
        say_bin = shutil.which("say")
        if not say_bin:
            return None

        output_dir.mkdir(parents=True, exist_ok=True)
        aiff_path = output_dir / f"{stem}.aiff"

        subprocess.run(
            [say_bin, "-v", self.voice, "-o", str(aiff_path), text],
            check=True,
        )

        afconvert_bin = shutil.which("afconvert")
        if not afconvert_bin:
            return aiff_path

        m4a_path = output_dir / f"{stem}.m4a"
        try:
            subprocess.run(
                [
                    afconvert_bin,
                    "-f",
                    "m4af",
                    "-d",
                    "aac",
                    str(aiff_path),
                    str(m4a_path),
                ],
                check=True,
            )
            aiff_path.unlink(missing_ok=True)
            return m4a_path
        except subprocess.CalledProcessError:
            return aiff_path


class SilentTTSProvider(TTSProvider):
    backend_name = "silent"

    def synthesize(self, text: str, output_dir: Path, stem: str) -> Optional[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        transcript_path = output_dir / f"{stem}.txt"
        transcript_path.write_text(text, encoding="utf-8")
        return transcript_path
