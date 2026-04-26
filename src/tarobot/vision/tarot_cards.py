from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


TAROT_SCENE_WIDTH = 1800
TAROT_TEMPLATE_HEIGHT = 500
TAROT_WARP_WIDTH = 240
TAROT_WARP_HEIGHT = 420
MIN_GOOD_MATCHES = 8
MIN_INLIERS = 8
MIN_ACCEPTED_CONFIDENCE = 0.45
NMS_IOU_THRESHOLD = 0.30
DEFAULT_TAROT_MANIFEST_PATH = Path("tests/examples/taro_cards/manifest.json")


@dataclass(frozen=True)
class TarotCardInfo:
    id: str
    name_ru: str
    arcana: str
    suit: Optional[str] = None


@dataclass(frozen=True)
class TarotExpectedCard:
    id: str
    orientation: str


@dataclass(frozen=True)
class TarotLayoutManifestEntry:
    filename: str
    kind: str
    expected_total_count: int
    visible_cards_left_to_right: List[TarotExpectedCard]
    notes: str = ""


@dataclass(frozen=True)
class TarotManifest:
    expected_total_count_default: int
    reference_images: Dict[str, List[List[str]]]
    layout_images: Dict[str, TarotLayoutManifestEntry]
    cards: Dict[str, TarotCardInfo]


@dataclass(frozen=True)
class TarotReferenceTemplate:
    label: str
    name_ru: str
    source_image: Path
    source_bbox: Tuple[int, int, int, int]
    image: np.ndarray
    keypoints: Tuple[cv2.KeyPoint, ...]
    descriptors: np.ndarray


@dataclass(frozen=True)
class TarotReferenceLibrary:
    templates: Tuple[TarotReferenceTemplate, ...]
    cards: Dict[str, TarotCardInfo]


@dataclass(frozen=True)
class TarotMatchCandidate:
    label: str
    name_ru: str
    bbox: Tuple[int, int, int, int]
    polygon: List[List[int]]
    orientation: str
    confidence: float
    inliers: int
    good_matches: int
    source_image: Path


@dataclass(frozen=True)
class TarotCardRecognition:
    index: int
    bbox: Tuple[int, int, int, int]
    polygon: List[List[int]]
    label: Optional[str]
    best_guess_label: str
    name_ru: Optional[str]
    orientation: Optional[str]
    confidence: float
    accepted: bool
    inliers: int
    good_matches: int
    top_candidates: List[Dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["bbox"] = list(self.bbox)
        return payload

    def to_public_dict(self) -> Dict[str, object]:
        return {
            "position": self.index,
            "card_id": self.label,
            "best_guess_card_id": self.best_guess_label,
            "name_ru": self.name_ru,
            "orientation": self.orientation,
            "confidence": self.confidence,
            "accepted": self.accepted,
            "bbox": list(self.bbox),
            "polygon": self.polygon,
            "match": {
                "inliers": self.inliers,
                "good_matches": self.good_matches,
            },
            "top_candidates": self.top_candidates,
        }


@dataclass(frozen=True)
class TarotImageResult:
    image_path: Path
    expected_total_count: int
    detected_count: int
    reason_codes: List[str]
    cards: List[TarotCardRecognition]
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

    def to_public_dict(self) -> Dict[str, object]:
        payload = {
            "image_path": str(self.image_path),
            "status": "ok" if not self.reason_codes else "failed",
            "expected_total_count": self.expected_total_count,
            "detected_count": self.detected_count,
            "reason_codes": self.reason_codes,
            "cards": [card.to_public_dict() for card in self.cards],
        }
        if self.debug_dir is not None:
            payload["debug_dir"] = str(self.debug_dir)
        return payload


def load_tarot_manifest(manifest_path: Path) -> TarotManifest:
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    cards = {
        card_id: TarotCardInfo(
            id=card_id,
            name_ru=payload["name_ru"],
            arcana=payload["arcana"],
            suit=payload.get("suit"),
        )
        for card_id, payload in raw["cards"].items()
    }
    layout_images = {
        filename: TarotLayoutManifestEntry(
            filename=filename,
            kind=payload["kind"],
            expected_total_count=payload["expected_total_count"],
            visible_cards_left_to_right=[
                TarotExpectedCard(id=card["id"], orientation=card["orientation"])
                for card in payload["visible_cards_left_to_right"]
            ],
            notes=payload.get("notes", ""),
        )
        for filename, payload in raw.get("layout_images", {}).items()
    }
    return TarotManifest(
        expected_total_count_default=raw["expected_total_count_default"],
        reference_images=raw["reference_images"],
        layout_images=layout_images,
        cards=cards,
    )


def resize_to_width(image: np.ndarray, target_width: int = TAROT_SCENE_WIDTH) -> np.ndarray:
    scale = target_width / image.shape[1]
    return cv2.resize(image, None, fx=scale, fy=scale)


def crop_reference_grid(image: np.ndarray, rows: Sequence[Sequence[str]]) -> List[Tuple[str, Tuple[int, int, int, int], np.ndarray]]:
    height, width = image.shape[:2]
    row_edges = np.linspace(0.02 * height, 0.98 * height, len(rows) + 1)
    crops: List[Tuple[str, Tuple[int, int, int, int], np.ndarray]] = []

    for row_index, row in enumerate(rows):
        y1 = max(0, int(row_edges[row_index] - 0.01 * height))
        y2 = min(height, int(row_edges[row_index + 1] + 0.01 * height))
        if len(row) == 1:
            column_edges = [0.02 * width, 0.18 * width]
        else:
            column_edges = np.linspace(0.02 * width, 0.98 * width, len(row) + 1)

        for column_index, label in enumerate(row):
            x1 = max(0, int(column_edges[column_index] - 0.005 * width))
            x2 = min(width, int(column_edges[column_index + 1] + 0.005 * width))
            crops.append((label, (x1, y1, x2 - x1, y2 - y1), image[y1:y2, x1:x2].copy()))

    return crops


def build_tarot_reference_library(manifest: TarotManifest, base_dir: Path) -> TarotReferenceLibrary:
    sift = cv2.SIFT_create(nfeatures=800)
    templates: List[TarotReferenceTemplate] = []
    seen_labels = set()

    for relative_path, rows in manifest.reference_images.items():
        source_image = base_dir / relative_path
        image = cv2.imread(str(source_image))
        if image is None:
            raise FileNotFoundError(source_image)

        for label, bbox, crop in crop_reference_grid(image, rows):
            if label not in manifest.cards:
                raise ValueError(f"Unknown tarot card id in manifest: {label}")
            if label in seen_labels:
                raise ValueError(f"Duplicate tarot reference label in manifest: {label}")
            seen_labels.add(label)

            scale = TAROT_TEMPLATE_HEIGHT / crop.shape[0]
            template_image = cv2.resize(crop, None, fx=scale, fy=scale)
            gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = sift.detectAndCompute(gray, None)
            if descriptors is None or len(keypoints) < MIN_GOOD_MATCHES:
                raise ValueError(f"Reference crop for {label} has too few visual features")

            templates.append(
                TarotReferenceTemplate(
                    label=label,
                    name_ru=manifest.cards[label].name_ru,
                    source_image=source_image,
                    source_bbox=bbox,
                    image=template_image,
                    keypoints=tuple(keypoints),
                    descriptors=descriptors,
                )
            )

    missing = set(manifest.cards) - seen_labels
    if missing:
        raise ValueError(f"Tarot reference library is missing cards: {sorted(missing)}")

    return TarotReferenceLibrary(templates=tuple(templates), cards=manifest.cards)


def polygon_area(points: np.ndarray) -> float:
    return float(abs(cv2.contourArea(points.astype("float32"))))


def compute_orientation(projected_corners: np.ndarray) -> str:
    top_y = float((projected_corners[0][1] + projected_corners[1][1]) / 2)
    bottom_y = float((projected_corners[2][1] + projected_corners[3][1]) / 2)
    return "upright" if top_y < bottom_y else "reversed"


def compute_confidence(inliers: int, good_matches: int, template_keypoints_count: int) -> float:
    inlier_part = min(1.0, inliers / 20.0) * 0.65
    match_part = min(1.0, good_matches / 35.0) * 0.20
    density_part = min(1.0, inliers / max(1.0, template_keypoints_count ** 0.5) / 2.0) * 0.15
    return round(inlier_part + match_part + density_part, 4)


def is_reasonable_card_projection(
    bbox: Tuple[int, int, int, int],
    projected_corners: np.ndarray,
    scene_shape: Tuple[int, int, int],
) -> bool:
    scene_height, scene_width = scene_shape[:2]
    x, y, width, height = bbox
    if width < 120 or height < 120:
        return False
    if max(width, height) < 300:
        return False
    if x < -scene_width * 0.05 or y < -scene_height * 0.05:
        return False
    if x + width > scene_width * 1.05 or y + height > scene_height * 1.05:
        return False

    ratio = max(width, height) / max(1, min(width, height))
    if not (1.15 <= ratio <= 2.80):
        return False

    area_ratio = (width * height) / float(scene_width * scene_height)
    if not (0.012 <= area_ratio <= 0.35):
        return False

    if polygon_area(projected_corners) < scene_width * scene_height * 0.008:
        return False

    return True


def candidate_iou(left: TarotMatchCandidate, right: TarotMatchCandidate) -> float:
    ax, ay, aw, ah = left.bbox
    bx, by, bw, bh = right.bbox
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


def find_tarot_matches(image: np.ndarray, library: TarotReferenceLibrary) -> List[TarotMatchCandidate]:
    scene = resize_to_width(image)
    gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=900)
    scene_keypoints, scene_descriptors = sift.detectAndCompute(gray, None)
    if scene_descriptors is None or len(scene_keypoints) < MIN_GOOD_MATCHES:
        return []

    matcher = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5),
        dict(checks=50),
    )
    candidates: List[TarotMatchCandidate] = []

    for template in library.templates:
        raw_matches = matcher.knnMatch(template.descriptors, scene_descriptors, k=2)
        good_matches = [left for left, right in raw_matches if left.distance < 0.70 * right.distance]
        if len(good_matches) < MIN_GOOD_MATCHES:
            continue

        source_points = np.float32([template.keypoints[match.queryIdx].pt for match in good_matches]).reshape(-1, 1, 2)
        scene_points = np.float32([scene_keypoints[match.trainIdx].pt for match in good_matches]).reshape(-1, 1, 2)
        homography, inlier_mask = cv2.findHomography(source_points, scene_points, cv2.RANSAC, 5.0)
        if homography is None or inlier_mask is None:
            continue

        inliers = int(inlier_mask.sum())
        if inliers < MIN_INLIERS:
            continue

        template_height, template_width = template.image.shape[:2]
        corners = np.float32(
            [[0, 0], [template_width, 0], [template_width, template_height], [0, template_height]]
        ).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(corners, homography).reshape(4, 2)
        if not np.isfinite(projected).all():
            continue

        x, y, width, height = cv2.boundingRect(projected.astype("float32"))
        bbox = (int(x), int(y), int(width), int(height))
        if not is_reasonable_card_projection(bbox, projected, scene.shape):
            continue

        confidence = compute_confidence(inliers, len(good_matches), len(template.keypoints))
        candidates.append(
            TarotMatchCandidate(
                label=template.label,
                name_ru=template.name_ru,
                bbox=bbox,
                polygon=projected.astype(int).tolist(),
                orientation=compute_orientation(projected),
                confidence=confidence,
                inliers=inliers,
                good_matches=len(good_matches),
                source_image=template.source_image,
            )
        )

    candidates.sort(key=lambda candidate: (candidate.confidence, candidate.inliers), reverse=True)
    selected: List[TarotMatchCandidate] = []
    used_labels = set()
    for candidate in candidates:
        if candidate.label in used_labels:
            continue
        if any(candidate_iou(candidate, selected_candidate) >= NMS_IOU_THRESHOLD for selected_candidate in selected):
            continue
        selected.append(candidate)
        used_labels.add(candidate.label)

    return sorted(selected, key=lambda candidate: candidate.bbox[0])


def top_candidates_for_match(
    match: TarotMatchCandidate,
    all_matches: Sequence[TarotMatchCandidate],
    limit: int = 3,
) -> List[Dict[str, object]]:
    overlapping = [
        other
        for other in all_matches
        if other.label == match.label or candidate_iou(match, other) >= NMS_IOU_THRESHOLD
    ]
    overlapping.sort(key=lambda candidate: (candidate.confidence, candidate.inliers), reverse=True)
    return [
        {
            "label": candidate.label,
            "name_ru": candidate.name_ru,
            "orientation": candidate.orientation,
            "confidence": candidate.confidence,
            "inliers": candidate.inliers,
        }
        for candidate in overlapping[:limit]
    ]


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


def warp_scene_card(image: np.ndarray, polygon: Sequence[Sequence[int]]) -> np.ndarray:
    source = order_points(np.asarray(polygon, dtype=np.float32))
    target = np.array(
        [[0, 0], [TAROT_WARP_WIDTH - 1, 0], [TAROT_WARP_WIDTH - 1, TAROT_WARP_HEIGHT - 1], [0, TAROT_WARP_HEIGHT - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source, target)
    return cv2.warpPerspective(image, matrix, (TAROT_WARP_WIDTH, TAROT_WARP_HEIGHT))


def scale_match_to_original(match: TarotMatchCandidate, scale: float) -> TarotMatchCandidate:
    x, y, width, height = match.bbox
    scaled_polygon = [[int(px / scale), int(py / scale)] for px, py in match.polygon]
    return TarotMatchCandidate(
        label=match.label,
        name_ru=match.name_ru,
        bbox=(int(x / scale), int(y / scale), int(width / scale), int(height / scale)),
        polygon=scaled_polygon,
        orientation=match.orientation,
        confidence=match.confidence,
        inliers=match.inliers,
        good_matches=match.good_matches,
        source_image=match.source_image,
    )


def prune_matches_to_expected_count(
    matches: Sequence[TarotMatchCandidate],
    expected_total_count: Optional[int],
) -> List[TarotMatchCandidate]:
    if expected_total_count is None or len(matches) <= expected_total_count:
        return list(matches)

    strongest = sorted(matches, key=lambda match: (match.confidence, match.inliers), reverse=True)[:expected_total_count]
    return sorted(strongest, key=lambda match: match.bbox[0])


def save_tarot_debug_artifacts(
    image_path: Path,
    image: np.ndarray,
    matches: Sequence[TarotMatchCandidate],
    recognitions: Sequence[TarotCardRecognition],
    debug_dir: Path,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    overlay = image.copy()
    for match, recognition in zip(matches, recognitions):
        points = np.asarray(match.polygon, dtype=np.int32)
        color = (0, 180, 0) if recognition.accepted else (0, 0, 255)
        cv2.drawContours(overlay, [points], 0, color, 4)
        x, y, _, _ = match.bbox
        label = recognition.label or f"? {recognition.best_guess_label}"
        text = f"{label} {recognition.orientation or '?'} {recognition.confidence:.2f}"
        cv2.putText(
            overlay,
            text,
            (int(x), max(32, int(y) - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            3,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(debug_dir / f"card_{recognition.index:02d}.png"), warp_scene_card(image, match.polygon))

    cv2.imwrite(str(debug_dir / "overlay.png"), overlay)
    payload = {
        "image": str(image_path),
        "cards": [recognition.to_dict() for recognition in recognitions],
    }
    (debug_dir / "debug.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def analyze_tarot_image(
    image_path: Path,
    library: TarotReferenceLibrary,
    expected_total_count: Optional[int] = None,
    output_root: Optional[Path] = None,
) -> TarotImageResult:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)

    scale = TAROT_SCENE_WIDTH / image.shape[1]
    matches_small = find_tarot_matches(image, library)
    matches = [scale_match_to_original(match, scale) for match in matches_small]
    matches = prune_matches_to_expected_count(matches, expected_total_count)

    reason_codes: List[str] = []
    if not matches:
        reason_codes.append("capture_invalid")
        reason_codes.append("card_not_found")

    if expected_total_count is not None and len(matches) != expected_total_count:
        reason_codes.append("cards_count_mismatch")

    recognitions: List[TarotCardRecognition] = []
    for index, match in enumerate(matches, start=1):
        accepted = match.confidence >= MIN_ACCEPTED_CONFIDENCE
        if not accepted and "card_low_confidence" not in reason_codes:
            reason_codes.append("card_low_confidence")
        recognitions.append(
            TarotCardRecognition(
                index=index,
                bbox=match.bbox,
                polygon=match.polygon,
                label=match.label if accepted else None,
                best_guess_label=match.label,
                name_ru=match.name_ru if accepted else None,
                orientation=match.orientation if accepted else None,
                confidence=match.confidence,
                accepted=accepted,
                inliers=match.inliers,
                good_matches=match.good_matches,
                top_candidates=top_candidates_for_match(match, matches),
            )
        )

    debug_dir = None
    if output_root is not None:
        debug_dir = output_root / image_path.stem
        save_tarot_debug_artifacts(image_path, image, matches, recognitions, debug_dir)

    return TarotImageResult(
        image_path=image_path,
        expected_total_count=expected_total_count or len(matches),
        detected_count=len(matches),
        reason_codes=reason_codes,
        cards=recognitions,
        debug_dir=debug_dir,
    )


def analyze_many_tarot_images(
    image_paths: Iterable[Path],
    library: TarotReferenceLibrary,
    manifest: Optional[TarotManifest] = None,
    output_root: Optional[Path] = None,
    expected_total_count_override: Optional[int] = None,
) -> List[TarotImageResult]:
    results = []
    for image_path in image_paths:
        manifest_entry = manifest.layout_images.get(image_path.name) if manifest else None
        expected_total_count = expected_total_count_override
        if expected_total_count is None:
            expected_total_count = manifest_entry.expected_total_count if manifest_entry else None
        results.append(
            analyze_tarot_image(
                image_path=image_path,
                library=library,
                expected_total_count=expected_total_count,
                output_root=output_root,
            )
        )
    return results


def recognize_tarot_photo(
    image_path: Path | str,
    manifest_path: Path | str = DEFAULT_TAROT_MANIFEST_PATH,
    expected_total_count: Optional[int] = None,
    output_root: Optional[Path | str] = None,
) -> Dict[str, object]:
    """Recognize Tarot cards on one photo and return a JSON-serializable payload."""
    image_path = Path(image_path)
    manifest_path = Path(manifest_path)
    manifest = load_tarot_manifest(manifest_path)
    manifest_entry = manifest.layout_images.get(image_path.name)
    resolved_expected_total_count = expected_total_count
    if resolved_expected_total_count is None and manifest_entry is not None:
        resolved_expected_total_count = manifest_entry.expected_total_count

    library = build_tarot_reference_library(manifest, manifest_path.parent)
    result = analyze_tarot_image(
        image_path=image_path,
        library=library,
        expected_total_count=resolved_expected_total_count,
        output_root=Path(output_root) if output_root is not None else None,
    )
    return result.to_public_dict()


def recognize_tarot_photos(
    image_paths: Iterable[Path | str],
    manifest_path: Path | str = DEFAULT_TAROT_MANIFEST_PATH,
    expected_total_count: Optional[int] = None,
    output_root: Optional[Path | str] = None,
) -> List[Dict[str, object]]:
    """Recognize Tarot cards on many photos and return JSON-serializable payloads."""
    manifest_path = Path(manifest_path)
    manifest = load_tarot_manifest(manifest_path)
    library = build_tarot_reference_library(manifest, manifest_path.parent)
    results = analyze_many_tarot_images(
        image_paths=[Path(image_path) for image_path in image_paths],
        library=library,
        manifest=manifest,
        output_root=Path(output_root) if output_root is not None else None,
        expected_total_count_override=expected_total_count,
    )
    return [result.to_public_dict() for result in results]
