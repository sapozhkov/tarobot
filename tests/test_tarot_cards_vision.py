from __future__ import annotations

import json
import subprocess
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
    recognize_tarot_photo,
    recognize_tarot_photos,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


class TarotCardsVisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest_path = ROOT / "tests" / "examples" / "taro_cards" / "manifest.json"
        cls.layout_dir = ROOT / "tests" / "examples" / "taro_cards" / "set"
        cls.manifest = load_tarot_manifest(cls.manifest_path)
        cls.library = build_tarot_reference_library(cls.manifest, cls.manifest_path.parent)

    @classmethod
    def layout_image_paths(cls) -> list[Path]:
        return sorted(
            (
                path
                for path in cls.layout_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            ),
            key=lambda path: path.name.lower(),
        )

    def test_reference_library_contains_full_tarot_deck(self) -> None:
        self.assertEqual(len(self.library.templates), 78)
        self.assertEqual(set(self.manifest.cards), {template.label for template in self.library.templates})

    def test_layout_examples_are_recognized_with_orientation(self) -> None:
        image_paths = self.layout_image_paths()
        self.assertGreater(len(image_paths), 0)
        self.assertEqual(
            {path.name for path in image_paths},
            set(self.manifest.layout_images),
            "Каждое layout-фото должно быть описано в manifest, и наоборот",
        )
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

    def test_public_recognition_function_returns_json_ready_payload(self) -> None:
        image_path = self.layout_dir / "IMG_3583.JPG"
        payload = recognize_tarot_photo(image_path, manifest_path=self.manifest_path)

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["detected_count"], 3)
        self.assertEqual(payload["reason_codes"], [])
        self.assertEqual(
            [(card["card_id"], card["orientation"]) for card in payload["cards"]],
            [("lovers", "upright"), ("chariot", "upright"), ("justice", "upright")],
        )
        json.dumps(payload, ensure_ascii=False)

    def test_public_batch_recognition_matches_manifest_for_all_layout_images(self) -> None:
        image_paths = self.layout_image_paths()
        payloads = recognize_tarot_photos(image_paths, manifest_path=self.manifest_path)

        self.assertEqual(len(payloads), len(image_paths))
        for payload in payloads:
            manifest_entry = self.manifest.layout_images[Path(str(payload["image_path"])).name]
            expected_cards = [(card.id, card.orientation) for card in manifest_entry.visible_cards_left_to_right]
            actual_cards = [(card["card_id"], card["orientation"]) for card in payload["cards"]]

            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["reason_codes"], [])
            self.assertEqual(payload["detected_count"], manifest_entry.expected_total_count)
            self.assertEqual(actual_cards, expected_cards)

    def test_cli_can_print_json_payload(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "recognize_taro.py"),
                str(self.layout_dir / "IMG_3583.JPG"),
                "--manifest",
                str(self.manifest_path),
                "--no-debug",
                "--json",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)

        self.assertEqual(len(payload["results"]), 1)
        self.assertEqual(payload["results"][0]["status"], "ok")
        self.assertEqual(
            [card["card_id"] for card in payload["results"][0]["cards"]],
            ["lovers", "chariot", "justice"],
        )

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
