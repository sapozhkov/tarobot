from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tarobot.vision.playing_cards import analyze_many_images, build_reference_library, load_card_manifest


class PlayingCardsVisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest_path = ROOT / "tests" / "examples" / "card_set" / "manifest.json"
        cls.manifest = load_card_manifest(cls.manifest_path)
        cls.library = build_reference_library(cls.manifest, cls.manifest_path.parent)

    def test_good_and_missing_images_are_recognized(self) -> None:
        image_paths = [
            ROOT / "tests" / "examples" / "card_set" / "IMG_3568.JPG",
            ROOT / "tests" / "examples" / "card_set" / "IMG_3569.JPG",
            ROOT / "tests" / "examples" / "card_set" / "IMG_3570.JPG",
            ROOT / "tests" / "examples" / "card_set" / "IMG_3571.JPG",
        ]
        results = analyze_many_images(image_paths, library=self.library, manifest=self.manifest)

        expected_labels = {
            "IMG_3568.JPG": ["QH", "8C", "10S", "KH", "6C"],
            "IMG_3569.JPG": ["7D", "10D", "8D", "JH", "7S"],
            "IMG_3570.JPG": ["QD", "AH", "QC", "9C", "7C"],
            "IMG_3571.JPG": ["JD", "10H", "7H", "9S"],
        }
        expected_detected = {
            "IMG_3568.JPG": 5,
            "IMG_3569.JPG": 5,
            "IMG_3570.JPG": 5,
            "IMG_3571.JPG": 4,
        }

        for result in results:
            self.assertEqual(result.detected_count, expected_detected[result.image_path.name])
            labels = [card.label for card in result.cards if card.label is not None]
            self.assertEqual(labels, expected_labels[result.image_path.name])

        missing_result = next(result for result in results if result.image_path.name == "IMG_3571.JPG")
        self.assertIn("cards_count_mismatch", missing_result.reason_codes)

    def test_overlap_case_is_flagged_and_partial_results_remain(self) -> None:
        image_path = ROOT / "tests" / "examples" / "card_set" / "IMG_3572.JPG"
        result = analyze_many_images([image_path], library=self.library, manifest=self.manifest)[0]

        self.assertEqual(result.detected_count, 4)
        self.assertIn("cards_count_mismatch", result.reason_codes)
        self.assertIn("cards_overlap_suspected", result.reason_codes)
        self.assertIn("card_low_confidence", result.reason_codes)
        self.assertEqual(result.cards[-1].label, "6S")

    def test_debug_artifacts_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = ROOT / "tests" / "examples" / "card_set" / "IMG_3568.JPG"
            result = analyze_many_images(
                [image_path],
                library=self.library,
                manifest=self.manifest,
                output_root=Path(tmp_dir),
            )[0]

            self.assertIsNotNone(result.debug_dir)
            self.assertTrue((result.debug_dir / "mask.png").exists())
            self.assertTrue((result.debug_dir / "overlay.png").exists())
            self.assertTrue((result.debug_dir / "card_01.png").exists())
            self.assertTrue((result.debug_dir / "debug.json").exists())


if __name__ == "__main__":
    unittest.main()
