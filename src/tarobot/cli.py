from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .app import build_default_app
from .models import ReadingRequest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tarobot MVP CLI")
    parser.add_argument("question", help="Question for the tarot reading")
    parser.add_argument(
        "--cards",
        type=int,
        default=5,
        help="Number of cards in spread. Recommended values: 3 or 5",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for reproducible simulated draws",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs"),
        help="Directory where artifacts will be saved",
    )
    parser.add_argument(
        "--silent-tts",
        action="store_true",
        help="Do not synthesize audio; save spoken text instead",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    request = ReadingRequest(
        question=args.question,
        cards_count=args.cards,
        seed=args.seed,
    )
    app = build_default_app(args.output_dir, enable_tts=not args.silent_tts)
    result = app.run(request)

    print(f"Run ID: {result.run_id}")
    print(f"Artifacts: {result.run_dir}")
    print("Cards:")
    for card in result.cards:
        print(f"  {card.position}. {card.position_label}: {card.display_name}")
    print()
    print(result.narrative.summary)
    print(result.narrative.advice)

    return 0
