from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tarobot.vision.tarot_cards import (
    analyze_many_tarot_images,
    build_tarot_reference_library,
    load_tarot_manifest,
)


class TarotCardsVisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest_path = ROOT / "tests" / "examples" / "taro_cards" / "manifest.json"
        cls.manifest = load_tarot_manifest(cls.manifest_path)
        cls.library = build_tarot_reference_library(cls.manifest, cls.manifest_path.parent)

    def test_reference_library_contains_full_tarot_deck(self) -> None:
        self.assertEqual(len(self.library.templates), 78)
        self.assertEqual(set(self.manifest.cards), {template.label for template in self.library.templates})

    def test_layout_examples_are_recognized_with_orientation(self) -> None:
        image_paths = [
            ROOT / "tests" / "examples" / "taro_cards" / "set" / "IMG_3581.jpg",
            ROOT / "tests" / "examples" / "taro_cards" / "set" / "IMG_3582.JPG",
            ROOT / "tests" / "examples" / "taro_cards" / "set" / "IMG_3583.JPG",
            ROOT / "tests" / "examples" / "taro_cards" / "set" / "IMG_3584.JPG",
        ]
        results = analyze_many_tarot_images(image_paths, library=self.library, manifest=self.manifest)

        for result in results:
            manifest_entry = self.manifest.layout_images[result.image_path.name]
            expected_cards = [(card.id, card.orientation) for card in manifest_entry.visible_cards_left_to_right]
            actual_cards = [(card.label, card.orientation) for card in result.cards]

            self.assertEqual(result.reason_codes, [])
            self.assertEqual(result.detected_count, manifest_entry.expected_total_count)
            self.assertEqual(actual_cards, expected_cards)
            self.assertTrue(all(card.accepted for card in result.cards))
            self.assertTrue(all(card.confidence >= 0.45 for card in result.cards))

    def test_debug_artifacts_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = ROOT / "tests" / "examples" / "taro_cards" / "set" / "IMG_3583.JPG"
            result = analyze_many_tarot_images(
                [image_path],
                library=self.library,
                manifest=self.manifest,
                output_root=Path(tmp_dir),
            )[0]

            self.assertIsNotNone(result.debug_dir)
            self.assertTrue((result.debug_dir / "overlay.png").exists())
            self.assertTrue((result.debug_dir / "card_01.png").exists())
            self.assertTrue((result.debug_dir / "debug.json").exists())


if __name__ == "__main__":
    unittest.main()
