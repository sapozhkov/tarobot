from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


DEFAULT_RESIZED_WIDTH = 1400
MULTISCALE_RESIZED_WIDTHS = (1400, 1800, 2200)
WARP_WIDTH = 240
WARP_HEIGHT = 360
CARD_CONFIDENCE_THRESHOLD = 0.50
OVERLAP_IMAGE_CONFIDENCE_THRESHOLD = 0.65
EXACT_TEMPLATE_THRESHOLD = 0.90
MIN_CARD_BBOX_AREA = 25000
MIN_CARD_ASPECT_RATIO = 1.05
MAX_CARD_ASPECT_RATIO = 2.10
MIN_REASONABLE_WIDTH_RATIO = 0.55
MIN_REASONABLE_HEIGHT_RATIO = 0.55
MIN_REASONABLE_AREA_RATIO = 0.40

RANK_TO_RU = {
    "6": "шестерка",
    "7": "семерка",
    "8": "восьмерка",
    "9": "девятка",
    "10": "десятка",
    "J": "валет",
    "Q": "дама",
    "K": "король",
    "A": "туз",
}

SUIT_TO_RU = {
    "C": "треф",
    "D": "бубен",
    "H": "червей",
    "S": "пик",
}


@dataclass(frozen=True)
class CardImageManifestEntry:
    filename: str
    role: str
    kind: str
    expected_total_count: int
    visible_labels_left_to_right: List[str]
    notes: str = ""


@dataclass(frozen=True)
class PlayingCardManifest:
    expected_total_count_default: int
    images: Dict[str, CardImageManifestEntry]


@dataclass(frozen=True)
class CardCandidate:
    index: int
    bbox: Tuple[int, int, int, int]
    bbox_area: int
    aspect_ratio: float
    component_area: int
    center_x: float
    box_points: List[List[int]]
    overlap_suspected: bool


@dataclass(frozen=True)
class PlayingCardRecognition:
    index: int
    bbox: Tuple[int, int, int, int]
    overlap_suspected: bool
    label: Optional[str]
    best_guess_label: str
    rank: Optional[str]
    suit: Optional[str]
    human_label_ru: Optional[str]
    confidence: float
    classifier: str
    accepted: bool
    top_rank_candidates: List[Dict[str, float]] = field(default_factory=list)
    top_suit_candidates: List[Dict[str, float]] = field(default_factory=list)
    top_face_candidates: List[Dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["bbox"] = list(self.bbox)
        return payload


@dataclass(frozen=True)
class PlayingCardImageResult:
    image_path: Path
    expected_total_count: int
    detected_count: int
    reason_codes: List[str]
    cards: List[PlayingCardRecognition]
    debug_dir: Optional[Path] = None

    def to_dict(self) -> Dict[str, object]:
        payload = {
            "image_path": str(self.image_path),
            "expected_total_count": self.expected_total_count,
            "detected_count": self.detected_count,
            "reason_codes": self.reason_codes,
            "cards": [card.to_dict() for card in self.cards],
        }
        if self.debug_dir is not None:
            payload["debug_dir"] = str(self.debug_dir)
        return payload


@dataclass(frozen=True)
class ReferenceLibrary:
    rank_templates: Dict[str, np.ndarray]
    suit_templates: Dict[str, np.ndarray]
    card_templates: Dict[str, np.ndarray]
    rank_face_templates: Dict[str, np.ndarray]


@dataclass(frozen=True)
class DetectionPass:
    resized_image: np.ndarray
    mask: np.ndarray
    candidates: List[CardCandidate]
    target_width: int


def load_card_manifest(manifest_path: Path) -> PlayingCardManifest:
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    images = {
        filename: CardImageManifestEntry(
            filename=filename,
            role=payload["role"],
            kind=payload["kind"],
            expected_total_count=payload["expected_total_count"],
            visible_labels_left_to_right=payload["visible_labels_left_to_right"],
            notes=payload.get("notes", ""),
        )
        for filename, payload in raw["images"].items()
    }
    return PlayingCardManifest(
        expected_total_count_default=raw["expected_total_count_default"],
        images=images,
    )


def resize_for_detection(image: np.ndarray, target_width: int = DEFAULT_RESIZED_WIDTH) -> np.ndarray:
    scale = target_width / image.shape[1]
    return cv2.resize(image, None, fx=scale, fy=scale)


def build_card_mask(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    otsu_threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = max(175, int(otsu_threshold) + 25)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return mask


def order_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    return np.array(
        [
            points[np.argmin(sums)],
            points[np.argmin(diffs)],
            points[np.argmax(sums)],
            points[np.argmax(diffs)],
        ],
        dtype=np.float32,
    )


def rectify_card(image: np.ndarray, box_points: Iterable[Iterable[float]]) -> np.ndarray:
    source = order_points(np.asarray(list(box_points), dtype=np.float32))
    target = np.array(
        [[0, 0], [WARP_WIDTH - 1, 0], [WARP_WIDTH - 1, WARP_HEIGHT - 1], [0, WARP_HEIGHT - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source, target)
    return cv2.warpPerspective(image, matrix, (WARP_WIDTH, WARP_HEIGHT))


def detect_card_candidates(image: np.ndarray, expected_total_count: Optional[int] = None) -> List[CardCandidate]:
    mask = build_card_mask(image)
    components_count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)

    raw_candidates: List[CardCandidate] = []
    for component_id in range(1, components_count):
        x, y, width, height, area = stats[component_id]
        bbox_area = int(width * height)
        aspect_ratio = max(width, height) / max(1, min(width, height))
        if not (bbox_area > MIN_CARD_BBOX_AREA and MIN_CARD_ASPECT_RATIO < aspect_ratio < MAX_CARD_ASPECT_RATIO):
            continue

        component_mask = np.where(labels[y : y + height, x : x + width] == component_id, 255, 0).astype("uint8")
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        rotated_rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rotated_rect)
        box[:, 0] += x
        box[:, 1] += y

        raw_candidates.append(
            CardCandidate(
                index=0,
                bbox=(int(x), int(y), int(width), int(height)),
                bbox_area=bbox_area,
                aspect_ratio=float(aspect_ratio),
                component_area=int(area),
                center_x=float(centers[component_id][0]),
                box_points=box.astype(int).tolist(),
                overlap_suspected=False,
            )
        )

    raw_candidates.sort(key=lambda candidate: candidate.center_x)
    if not raw_candidates:
        return []

    raw_candidates = filter_reasonable_candidates(raw_candidates)
    if not raw_candidates:
        return []

    widths = [candidate.bbox[2] for candidate in raw_candidates]
    heights = [candidate.bbox[3] for candidate in raw_candidates]
    ratios = [candidate.aspect_ratio for candidate in raw_candidates]
    median_width = median(widths)
    median_height = median(heights)
    median_ratio = median(ratios)

    cards_count_mismatch = expected_total_count is not None and len(raw_candidates) != expected_total_count
    normalized_candidates: List[CardCandidate] = []
    for index, candidate in enumerate(raw_candidates, start=1):
        width = candidate.bbox[2]
        height = candidate.bbox[3]
        overlap_suspected = False
        if cards_count_mismatch:
            if width > median_width * 1.20 or height > median_height * 1.20:
                overlap_suspected = True
            if candidate.aspect_ratio > median_ratio * 1.20 or candidate.aspect_ratio < median_ratio * 0.85:
                overlap_suspected = True

        normalized_candidates.append(
            CardCandidate(
                index=index,
                bbox=candidate.bbox,
                bbox_area=candidate.bbox_area,
                aspect_ratio=candidate.aspect_ratio,
                component_area=candidate.component_area,
                center_x=candidate.center_x,
                box_points=candidate.box_points,
                overlap_suspected=overlap_suspected,
            )
        )

    return normalized_candidates


def filter_reasonable_candidates(candidates: List[CardCandidate]) -> List[CardCandidate]:
    if len(candidates) < 3:
        return candidates

    widths = [candidate.bbox[2] for candidate in candidates]
    heights = [candidate.bbox[3] for candidate in candidates]
    areas = [candidate.bbox_area for candidate in candidates]
    median_width = median(widths)
    median_height = median(heights)
    median_area = median(areas)

    filtered = [
        candidate
        for candidate in candidates
        if candidate.bbox[2] >= median_width * MIN_REASONABLE_WIDTH_RATIO
        and candidate.bbox[3] >= median_height * MIN_REASONABLE_HEIGHT_RATIO
        and candidate.bbox_area >= median_area * MIN_REASONABLE_AREA_RATIO
    ]
    return filtered or candidates


def choose_detection_pass(image: np.ndarray, expected_total_count: Optional[int]) -> DetectionPass:
    widths = MULTISCALE_RESIZED_WIDTHS if expected_total_count is not None else (DEFAULT_RESIZED_WIDTH,)
    detection_passes: List[DetectionPass] = []
    for target_width in widths:
        resized_image = resize_for_detection(image, target_width=target_width)
        mask = build_card_mask(resized_image)
        candidates = detect_card_candidates(resized_image, expected_total_count=expected_total_count)
        detection_passes.append(
            DetectionPass(
                resized_image=resized_image,
                mask=mask,
                candidates=candidates,
                target_width=target_width,
            )
        )

    if expected_total_count is None:
        return detection_passes[0]

    return min(
        detection_passes,
        key=lambda detection_pass: (
            abs(len(detection_pass.candidates) - expected_total_count),
            detection_pass.target_width,
        ),
    )


def symbol_mask(image: np.ndarray) -> np.ndarray:
    blue, green, red = cv2.split(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    red_pixels = (red > 120) & ((red.astype(int) - green.astype(int)) > 35) & ((red.astype(int) - blue.astype(int)) > 35)
    black_pixels = gray < 105
    return ((red_pixels | black_pixels).astype("uint8") * 255)


def normalize_binary_mask(mask: np.ndarray, size: Tuple[int, int], largest_only: bool, min_area: int = 15) -> np.ndarray:
    components_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    components = []
    for component_id in range(1, components_count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area >= min_area:
            components.append((area, component_id))

    if not components:
        return np.zeros((size[1], size[0]), dtype="uint8")

    if largest_only:
        keep_ids = {max(components)[1]}
    else:
        keep_ids = {component_id for _, component_id in components}

    ys, xs = np.where(np.isin(labels, list(keep_ids)))
    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1
    cropped = np.where(np.isin(labels[y1:y2, x1:x2], list(keep_ids)), 255, 0).astype("uint8")
    return cv2.resize(cropped, size, interpolation=cv2.INTER_AREA)


def extract_corner_rois_from_patch(corner_patch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rank_raw = symbol_mask(corner_patch[5:82, 4:60])
    suit_raw = symbol_mask(corner_patch[55:112, 4:50])
    rank_mask = normalize_binary_mask(rank_raw, size=(52, 74), largest_only=False)
    suit_mask = normalize_binary_mask(suit_raw, size=(38, 42), largest_only=True)
    return rank_mask, suit_mask


def extract_corner_rois_variants(card_image: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    top_left_corner = card_image[:120, :90]
    bottom_right_corner = cv2.rotate(card_image[-120:, -90:], cv2.ROTATE_180)
    return [
        extract_corner_rois_from_patch(top_left_corner),
        extract_corner_rois_from_patch(bottom_right_corner),
    ]


def extract_corner_rois(card_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return extract_corner_rois_variants(card_image)[1]


def extract_suit_patch(card_image: np.ndarray) -> np.ndarray:
    corner_patch = cv2.rotate(card_image[-120:, -90:], cv2.ROTATE_180)
    return corner_patch[55:112, 4:50]


def infer_suit_family(card_image: np.ndarray) -> Optional[str]:
    suit_patch = extract_suit_patch(card_image)
    blue, green, red = cv2.split(suit_patch)
    gray = cv2.cvtColor(suit_patch, cv2.COLOR_BGR2GRAY)
    red_pixels = int(
        (
            (red > 120)
            & ((red.astype(int) - green.astype(int)) > 35)
            & ((red.astype(int) - blue.astype(int)) > 35)
        ).sum()
    )
    black_pixels = int((gray < 105).sum())

    if red_pixels > max(20, int(black_pixels * 1.2)):
        return "red"
    if black_pixels > max(20, int(red_pixels * 1.2)):
        return "black"
    return None


def normalize_face_card(card_image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (120, 180), interpolation=cv2.INTER_AREA)


def template_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left = left.astype("float32") / 255.0
    right = right.astype("float32") / 255.0
    left = left - left.mean()
    right = right - right.mean()
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if left_norm < 1e-6 or right_norm < 1e-6:
        return -1.0
    return float((left * right).sum() / (left_norm * right_norm))


def parse_label(label: str) -> Tuple[str, str]:
    if label.startswith("10"):
        return "10", label[2:]
    return label[0], label[1:]


def human_label(label: str) -> str:
    rank, suit = parse_label(label)
    return f"{RANK_TO_RU[rank]} {SUIT_TO_RU[suit]}"


def build_reference_library(manifest: PlayingCardManifest, base_dir: Path) -> ReferenceLibrary:
    rank_templates: Dict[str, List[np.ndarray]] = {}
    suit_templates: Dict[str, List[np.ndarray]] = {}
    card_templates: Dict[str, np.ndarray] = {}
    rank_face_templates: Dict[str, List[np.ndarray]] = {}

    for entry in manifest.images.values():
        if entry.role != "reference":
            continue

        image = cv2.imread(str(base_dir / entry.filename))
        if image is None:
            raise FileNotFoundError(base_dir / entry.filename)
        image = resize_for_detection(image)
        candidates = detect_card_candidates(image, expected_total_count=None)
        if len(candidates) != len(entry.visible_labels_left_to_right):
            raise ValueError(
                f"Reference image {entry.filename} produced {len(candidates)} cards, "
                f"expected {len(entry.visible_labels_left_to_right)}"
            )

        for candidate, label in zip(candidates, entry.visible_labels_left_to_right):
            warped = rectify_card(image, candidate.box_points)
            rank, suit = parse_label(label)
            for rank_mask, suit_mask in extract_corner_rois_variants(warped):
                rank_templates.setdefault(rank, []).append(rank_mask)
                suit_templates.setdefault(suit, []).append(suit_mask)
            normalized_face = normalize_face_card(warped)
            card_templates[label] = normalized_face
            rank_face_templates.setdefault(rank, []).append(normalized_face)

    return ReferenceLibrary(
        rank_templates={rank: np.mean(samples, axis=0).astype("uint8") for rank, samples in rank_templates.items()},
        suit_templates={suit: np.mean(samples, axis=0).astype("uint8") for suit, samples in suit_templates.items()},
        card_templates=card_templates,
        rank_face_templates={
            rank: np.mean(samples, axis=0).astype("uint8") for rank, samples in rank_face_templates.items()
        },
    )


def sorted_top_scores(scores: Dict[str, float], limit: int = 3) -> List[Dict[str, float]]:
    return [
        {"label": label, "score": round(score, 4)}
        for label, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
    ]


def recognize_card(
    card_image: np.ndarray,
    library: ReferenceLibrary,
    candidate: CardCandidate,
    image_overlap_suspected: bool = False,
) -> PlayingCardRecognition:
    normalized_face = normalize_face_card(card_image)
    exact_scores = {
        label: template_similarity(normalized_face, template)
        for label, template in library.card_templates.items()
    }
    best_exact_label = max(exact_scores, key=exact_scores.get) if exact_scores else None
    best_exact_score = exact_scores[best_exact_label] if best_exact_label else -1.0

    corner_variants = extract_corner_rois_variants(card_image)
    corner_rank_scores = {
        rank: max(template_similarity(rank_mask, template) for rank_mask, _ in corner_variants)
        for rank, template in library.rank_templates.items()
    }
    face_rank_scores = {
        rank: template_similarity(normalized_face, template)
        for rank, template in library.rank_face_templates.items()
    }
    rank_scores = {
        rank: (corner_rank_scores[rank] * 0.55) + (face_rank_scores[rank] * 0.45)
        for rank in library.rank_templates.keys()
    }
    suit_family = infer_suit_family(card_image)
    if suit_family == "red":
        allowed_suits = {"D", "H"}
    elif suit_family == "black":
        allowed_suits = {"C", "S"}
    else:
        allowed_suits = set(library.suit_templates.keys())
    suit_scores = {
        suit: max(template_similarity(suit_mask, template) for _, suit_mask in corner_variants)
        for suit, template in library.suit_templates.items()
        if suit in allowed_suits
    }
    best_rank = max(rank_scores, key=rank_scores.get)
    best_suit = max(suit_scores, key=suit_scores.get)
    best_guess_label = f"{best_rank}{best_suit}"
    corner_confidence = (rank_scores[best_rank] * 0.80) + (suit_scores[best_suit] * 0.20)
    threshold = OVERLAP_IMAGE_CONFIDENCE_THRESHOLD if image_overlap_suspected else CARD_CONFIDENCE_THRESHOLD

    if best_exact_score >= EXACT_TEMPLATE_THRESHOLD:
        adjusted_exact_confidence = best_exact_score if not candidate.overlap_suspected else best_exact_score - 0.10
        accepted = adjusted_exact_confidence >= threshold
        final_label = best_exact_label if accepted else None
        return PlayingCardRecognition(
            index=candidate.index,
            bbox=candidate.bbox,
            overlap_suspected=candidate.overlap_suspected,
            label=final_label,
            best_guess_label=best_exact_label,
            rank=parse_label(best_exact_label)[0] if final_label else None,
            suit=parse_label(best_exact_label)[1] if final_label else None,
            human_label_ru=human_label(best_exact_label) if final_label else None,
            confidence=round(adjusted_exact_confidence, 4),
            classifier="exact_template",
            accepted=accepted,
            top_rank_candidates=sorted_top_scores(rank_scores),
            top_suit_candidates=sorted_top_scores(suit_scores),
            top_face_candidates=sorted_top_scores(exact_scores),
        )

    adjusted_confidence = corner_confidence if not candidate.overlap_suspected else corner_confidence - 0.12
    accepted = adjusted_confidence >= threshold
    final_label = best_guess_label if accepted else None
    return PlayingCardRecognition(
        index=candidate.index,
        bbox=candidate.bbox,
        overlap_suspected=candidate.overlap_suspected,
        label=final_label,
        best_guess_label=best_guess_label,
        rank=best_rank if accepted else None,
        suit=best_suit if accepted else None,
        human_label_ru=human_label(best_guess_label) if accepted else None,
        confidence=round(adjusted_confidence, 4),
        classifier="corner_template",
        accepted=accepted,
        top_rank_candidates=sorted_top_scores(rank_scores),
        top_suit_candidates=sorted_top_scores(suit_scores),
        top_face_candidates=sorted_top_scores(exact_scores),
    )


def save_debug_artifacts(
    image_path: Path,
    image: np.ndarray,
    mask: np.ndarray,
    candidates: List[CardCandidate],
    recognitions: List[PlayingCardRecognition],
    debug_dir: Path,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_dir / "mask.png"), mask)
    overlay = image.copy()
    for candidate, recognition in zip(candidates, recognitions):
        points = np.asarray(candidate.box_points, dtype=np.int32)
        color = (0, 0, 255) if candidate.overlap_suspected else (0, 255, 0)
        cv2.drawContours(overlay, [points], 0, color, 2)
        text = recognition.label or f"? {recognition.best_guess_label}"
        x, y, _, _ = candidate.bbox
        cv2.putText(
            overlay,
            text,
            (int(x), max(20, int(y) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
        warped = rectify_card(image, candidate.box_points)
        cv2.imwrite(str(debug_dir / f"card_{candidate.index:02d}.png"), warped)
    cv2.imwrite(str(debug_dir / "overlay.png"), overlay)
    payload = {
        "image": str(image_path),
        "candidates": [
            {
                "index": candidate.index,
                "bbox": list(candidate.bbox),
                "bbox_area": candidate.bbox_area,
                "aspect_ratio": round(candidate.aspect_ratio, 4),
                "overlap_suspected": candidate.overlap_suspected,
                "recognition": recognitions[index].to_dict(),
            }
            for index, candidate in enumerate(candidates)
        ],
    }
    (debug_dir / "debug.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def analyze_image(
    image_path: Path,
    library: ReferenceLibrary,
    expected_total_count: Optional[int] = None,
    output_root: Optional[Path] = None,
) -> PlayingCardImageResult:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)

    detection_pass = choose_detection_pass(image, expected_total_count=expected_total_count)
    image = detection_pass.resized_image
    mask = detection_pass.mask
    candidates = detection_pass.candidates

    reason_codes: List[str] = []
    if not candidates:
        reason_codes.append("capture_invalid")

    if expected_total_count is not None and len(candidates) != expected_total_count:
        reason_codes.append("cards_count_mismatch")

    if any(candidate.overlap_suspected for candidate in candidates):
        reason_codes.append("cards_overlap_suspected")

    image_overlap_suspected = any(candidate.overlap_suspected for candidate in candidates)
    recognitions = [
        recognize_card(
            rectify_card(image, candidate.box_points),
            library,
            candidate,
            image_overlap_suspected=image_overlap_suspected,
        )
        for candidate in candidates
    ]
    if any(not recognition.accepted for recognition in recognitions):
        reason_codes.append("card_low_confidence")

    debug_dir = None
    if output_root is not None:
        debug_dir = output_root / image_path.stem
        save_debug_artifacts(image_path, image, mask, candidates, recognitions, debug_dir)

    return PlayingCardImageResult(
        image_path=image_path,
        expected_total_count=expected_total_count or len(candidates),
        detected_count=len(candidates),
        reason_codes=reason_codes,
        cards=recognitions,
        debug_dir=debug_dir,
    )


def analyze_many_images(
    image_paths: Iterable[Path],
    library: ReferenceLibrary,
    manifest: Optional[PlayingCardManifest] = None,
    output_root: Optional[Path] = None,
    expected_total_count_override: Optional[int] = None,
) -> List[PlayingCardImageResult]:
    results = []
    for image_path in image_paths:
        manifest_entry = manifest.images.get(image_path.name) if manifest else None
        expected_total_count = expected_total_count_override
        if expected_total_count is None:
            expected_total_count = manifest_entry.expected_total_count if manifest_entry else None
        results.append(
            analyze_image(
                image_path=image_path,
                library=library,
                expected_total_count=expected_total_count,
                output_root=output_root,
            )
        )
    return results
