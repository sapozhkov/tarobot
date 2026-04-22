from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tarobot.app import TarobotApp
from tarobot.artifacts import ArtifactStore
from tarobot.cards import draw_cards
from tarobot.llm import MockLLMService
from tarobot.models import ReadingRequest, SpeechPlan
from tarobot.tts import TTSProvider


class FakeTTSProvider(TTSProvider):
    backend_name = "fake"

    def synthesize(self, plan: SpeechPlan, output_dir: Path, stem: str) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{stem}.txt"
        path.write_text(plan.full_text, encoding="utf-8")
        return path


class TarobotFlowTests(unittest.TestCase):
    def test_end_to_end_flow_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            app = TarobotApp(
                artifact_store=ArtifactStore(Path(tmp_dir)),
                tts_provider=FakeTTSProvider(),
            )

            result = app.run(
                ReadingRequest(
                    question="Стоит ли мне менять работу этим летом?",
                    cards_count=3,
                    seed=7,
                )
            )

            self.assertEqual(len(result.cards), 3)
            self.assertTrue((result.run_dir / "request.json").exists())
            self.assertTrue((result.run_dir / "cards.json").exists())
            self.assertTrue((result.run_dir / "reading.txt").exists())
            self.assertTrue((result.run_dir / "speech_plan.json").exists())
            self.assertTrue((result.run_dir / "reading_audio.txt").exists())
            self.assertTrue((result.run_dir / "manifest.json").exists())
            self.assertIn("Стоит ли мне менять работу", result.narrative.spoken_text)
            self.assertGreaterEqual(len(result.narrative.speech_plan.segments), 5)

    def test_mock_llm_builds_segmented_speech_plan(self) -> None:
        llm = MockLLMService()
        narrative = llm.generate_reading(
            ReadingRequest(question="Стоит ли мне делать новый шаг?", cards_count=3, seed=42),
            draw_cards(cards_count=3, seed=42),
        )

        sections = [segment.section for segment in narrative.speech_plan.segments]
        self.assertEqual(sections[0], "intro")
        self.assertIn("cards", sections)
        self.assertEqual(sections[-1], "outro")
        self.assertIn("3 карты", narrative.spoken_text)
        self.assertNotIn("3 карт.", narrative.spoken_text)


if __name__ == "__main__":
    unittest.main()
