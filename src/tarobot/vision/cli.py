from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence

from .playing_cards import analyze_many_images, build_reference_library, load_card_manifest


def expand_inputs(inputs: Sequence[str]) -> List[Path]:
    expanded: List[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            expanded.extend(sorted(path.glob("*.JPG")))
            expanded.extend(sorted(path.glob("*.jpg")))
            expanded.extend(sorted(path.glob("*.png")))
        else:
            expanded.append(path)
    unique_paths = []
    seen = set()
    for path in expanded:
        resolved = path.resolve()
        if resolved not in seen:
            unique_paths.append(resolved)
            seen.add(resolved)
    return unique_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recognize ordinary playing cards on photos")
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Image file(s) or directory with images",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("tests/examples/card_set/manifest.json"),
        help="Path to image manifest with reference labels",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/playing_cards"),
        help="Directory for debug artifacts and summary.json",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=None,
        help="Override expected number of cards for all images",
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Do not save overlays, masks and warped cards",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    image_paths = expand_inputs(args.inputs)
    if not image_paths:
        parser.error("No images were found")

    manifest = load_card_manifest(args.manifest)
    reference_library = build_reference_library(manifest, args.manifest.parent)
    output_dir = None if args.no_debug else args.output_dir

    results = analyze_many_images(
        image_paths=image_paths,
        library=reference_library,
        manifest=manifest,
        output_root=output_dir,
        expected_total_count_override=args.expected_count,
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "summary.json"
        summary_path.write_text(
            json.dumps([result.to_dict() for result in results], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    for result in results:
        print(
            f"{result.image_path.name}: detected={result.detected_count}, "
            f"expected={result.expected_total_count}, reasons={result.reason_codes or ['ok']}"
        )
        for card in result.cards:
            label = card.label or f"? {card.best_guess_label}"
            print(
                f"  {card.index}. {label}, confidence={card.confidence:.3f}, "
                f"classifier={card.classifier}, overlap={card.overlap_suspected}"
            )
        if result.debug_dir is not None:
            print(f"  debug: {result.debug_dir}")

    return 0
