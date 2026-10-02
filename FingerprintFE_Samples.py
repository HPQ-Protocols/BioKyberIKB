#!/usr/bin/env python3
"""Measure a set-based fingerprint fuzzy-vault prototype on FVC2002 B sets.

This program creates ``fingerprint_fe_samples.csv`` for the Bio-Kyber IKB
experiment.  It reads FVC2002 DB1_B--DB4_B ZIP archives directly, extracts
minutiae with pyfing/LEADER, protects a random key with a polynomial fuzzy
vault, and measures enrollment (Gen) and reconstruction (Rep).

The FVC2002 ``source2002.zip`` archive contains competition API skeletons,
not a fingerprint extractor.  It is therefore not used as an algorithm here.

Install the runtime dependencies on the experiment laptop:

    python -m pip install pyfing tensorflow keras opencv-python pillow numpy pandas

Typical run:

    python FingerprintFE_Samples.py \
        --dataset-zips DB1_B.zip DB2_B.zip DB3_B.zip DB4_B.zip \
        --output fingerprint_fe_samples.csv

By default, subjects 101 and 102 from every database are used only for
automatic parameter selection; subjects 103--110 form a subject-disjoint
evaluation set.  The default impostor policy tests the first impression of
every non-mated evaluation subject.

Quick archive validation, without TensorFlow/pyfing:

    python FingerprintFE_Samples.py --dataset-zips DB?_B.zip --dry-run

Important research limitations
------------------------------
This is a reproducible research prototype, not a standardized or production
fingerprint fuzzy extractor.  Its security and reliability depend on the
minutiae extractor, canonicalization, quantization, vault parameters, and the
population being evaluated.  Report FTE/FRR/FAR and helper-data size together
with timing.  Do not interpret successful reconstruction as 256 bits of
biometric entropy.  Bio-Kyber IKB obtains cryptographic entropy from its
independent non-exportable device secret.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import secrets
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, NamedTuple, Sequence
from zipfile import ZipFile

import numpy as np
import pandas as pd
from PIL import Image


SCRIPT_VERSION = "2.0.0"
PRIME = 2_147_483_647  # 2^31 - 1, a Mersenne prime.
IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".bmp", ".jpg", ".jpeg"}
DATASET_DPI = {
    "DB1_B": 500,
    "DB2_B": 569,
    "DB3_B": 500,
    "DB4_B": 500,
}
TUNING_GRID_SIZES = (8, 12, 16)
TUNING_ANGLE_BINS = (1, 2, 4)
TUNING_QUALITY_THRESHOLDS = (0.40, 0.55, 0.70)
TUNING_STABLE_VOTES = (1, 2)


@dataclass(frozen=True)
class Config:
    dataset_zips: tuple[Path, ...]
    output: Path
    summary_output: Path
    enrollment_count: int
    grid_size: int
    angle_bins: int
    stable_votes: int
    polynomial_points: int
    chaff_points: int
    max_unlock_combinations: int
    minutia_quality: float
    max_minutiae: int
    impostor_mode: str
    development_datasets: tuple[str, ...]
    development_subjects: tuple[str, ...]
    auto_tune: bool
    alignment_distance: float
    alignment_angle_degrees: float
    alignment_min_matches: int
    alignment_max_candidates: int
    random_seed: int
    dry_run: bool
    self_test: bool


class ImageRecord(NamedTuple):
    dataset: str
    subject_id: str
    impression: int
    member_name: str
    archive_path: Path


@dataclass(frozen=True)
class Vault:
    points: tuple[tuple[int, int], ...]
    verifier: str
    polynomial_points: int
    universe_size: int


@dataclass(frozen=True)
class Enrollment:
    key: bytes
    vault: Vault
    stable_tokens: tuple[int, ...]
    gen_ms: float
    minutiae_count: int
    reference_minutiae: tuple["MinutiaFeature", ...]
    reference_shape: tuple[int, int]
    enrollment_alignment_matches: tuple[int, ...]


@dataclass(frozen=True)
class Reproduction:
    key: bytes | None
    rep_ms: float
    probe_tokens: int
    matched_vault_points: int
    combinations_tried: int
    failure_reason: str
    alignment_matches: int


@dataclass(frozen=True)
class MinutiaFeature:
    x: float
    y: float
    direction: float
    kind: int
    quality: float


@dataclass(frozen=True)
class AlignmentResult:
    features: tuple[MinutiaFeature, ...]
    matches: int
    score: float
    angle: float
    translation_x: float
    translation_y: float


@dataclass(frozen=True)
class TuningResult:
    selected: dict[str, Any]
    candidates: tuple[dict[str, Any], ...]
    development_datasets: tuple[str, ...]
    development_subjects: tuple[str, ...]


@dataclass(frozen=True)
class PreparedImage:
    record: ImageRecord
    shape: tuple[int, int]
    minutiae: tuple[MinutiaFeature, ...]


def now_ns() -> int:
    return time.perf_counter_ns()


def elapsed_ms(start_ns: int) -> float:
    return (time.perf_counter_ns() - start_ns) / 1_000_000.0


def parse_image_name(name: str) -> tuple[str, int] | None:
    stem = Path(name).stem
    if "_" not in stem:
        return None
    subject, impression_text = stem.rsplit("_", 1)
    try:
        impression = int(impression_text)
    except ValueError:
        return None
    return subject, impression


def discover_records(archives: Sequence[Path]) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for archive in archives:
        if not archive.exists():
            raise SystemExit(f"Dataset archive does not exist: {archive}")
        dataset = archive.stem
        with ZipFile(archive) as handle:
            for member in handle.namelist():
                if Path(member).suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                parsed = parse_image_name(member)
                if parsed is None:
                    continue
                subject, impression = parsed
                records.append(
                    ImageRecord(dataset, subject, impression, member, archive)
                )
    records.sort(key=lambda x: (x.dataset, x.subject_id, x.impression))
    if not records:
        raise SystemExit("No fingerprint images were found in the supplied archives")
    return records


def validate_dataset(records: Sequence[ImageRecord], enrollment_count: int) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[ImageRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.dataset, record.subject_id)].append(record)

    datasets: dict[str, dict[str, Any]] = {}
    for (dataset, subject), images in grouped.items():
        item = datasets.setdefault(dataset, {"subjects": 0, "images": 0, "impressions": {}})
        item["subjects"] += 1
        item["images"] += len(images)
        item["impressions"][subject] = sorted(x.impression for x in images)
        if len(images) <= enrollment_count:
            raise SystemExit(
                f"{dataset}/{subject} has {len(images)} images; more than "
                f"--enrollment-count={enrollment_count} are required"
            )
    return datasets


def load_grayscale(record: ImageRecord) -> np.ndarray:
    with ZipFile(record.archive_path) as handle:
        payload = handle.read(record.member_name)
    with Image.open(BytesIO(payload)) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


def record_key(record: ImageRecord) -> str:
    return f"{record.dataset}:{record.member_name}"


def load_pyfing() -> Any:
    try:
        import pyfing
    except ImportError as exc:
        raise SystemExit(
            "pyfing is required for minutiae extraction. Install the experiment "
            "dependencies with: python -m pip install pyfing tensorflow keras "
            "opencv-python pillow numpy pandas"
        ) from exc
    return pyfing


def dataset_dpi(dataset: str) -> int:
    return DATASET_DPI.get(dataset.upper(), 500)


def extract_minutiae(
    pyfing: Any,
    image: np.ndarray,
    dpi: int,
    quality: float,
    maximum: int,
) -> list[MinutiaFeature]:
    minutiae = pyfing.minutiae_extraction(image, dpi=dpi, method="LEADER")
    selected = [
        MinutiaFeature(
            x=float(minutia.x),
            y=float(minutia.y),
            direction=float(minutia.direction) % (2.0 * math.pi),
            kind=minutia_kind(minutia.type),
            quality=float(minutia.quality),
        )
        for minutia in minutiae
        if float(minutia.quality) >= quality
    ]
    selected.sort(key=lambda item: item.quality, reverse=True)
    return selected[:maximum]


def minutia_kind(value: Any) -> int:
    """Return 1 for bifurcation and 0 for termination.

    pyfing exposes an enum, whose string representation can be
    ``MinutiaType.BIFURCATION``.  Inspecting ``.name`` avoids classifying every
    enum value as a termination.
    """
    name = str(getattr(value, "name", value)).upper()
    return int(name == "BIFURCATION" or name.endswith(".BIFURCATION"))


def circular_distance(a: float, b: float) -> float:
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def transform_features(
    features: Sequence[MinutiaFeature],
    angle: float,
    translation_x: float,
    translation_y: float,
) -> tuple[MinutiaFeature, ...]:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return tuple(
        MinutiaFeature(
            x=cosine * item.x - sine * item.y + translation_x,
            y=sine * item.x + cosine * item.y + translation_y,
            direction=(item.direction + angle) % (2.0 * math.pi),
            kind=item.kind,
            quality=item.quality,
        )
        for item in features
    )


def count_alignment_matches(
    reference: Sequence[MinutiaFeature],
    candidate: Sequence[MinutiaFeature],
    distance_threshold: float,
    angle_threshold: float,
) -> tuple[int, float]:
    if not reference or not candidate:
        return 0, 0.0
    candidate_x = np.asarray([item.x for item in candidate])[:, None]
    candidate_y = np.asarray([item.y for item in candidate])[:, None]
    candidate_direction = np.asarray([item.direction for item in candidate])[:, None]
    candidate_kind = np.asarray([item.kind for item in candidate])[:, None]
    reference_x = np.asarray([item.x for item in reference])[None, :]
    reference_y = np.asarray([item.y for item in reference])[None, :]
    reference_direction = np.asarray([item.direction for item in reference])[None, :]
    reference_kind = np.asarray([item.kind for item in reference])[None, :]

    distance = np.hypot(candidate_x - reference_x, candidate_y - reference_y)
    direction_delta = candidate_direction - reference_direction
    direction_error = np.abs(np.arctan2(np.sin(direction_delta), np.cos(direction_delta)))
    valid = (
        (candidate_kind == reference_kind)
        & (distance <= distance_threshold)
        & (direction_error <= angle_threshold)
    )
    candidate_indices, reference_indices = np.nonzero(valid)
    normalized_values = (
        distance[candidate_indices, reference_indices] / distance_threshold
        + direction_error[candidate_indices, reference_indices] / angle_threshold
    )
    edges = sorted(
        zip(normalized_values.tolist(), candidate_indices.tolist(), reference_indices.tolist())
    )

    matched_candidates: set[int] = set()
    matched_references: set[int] = set()
    score = 0.0
    for normalized, candidate_index, reference_index in sorted(edges):
        if candidate_index in matched_candidates or reference_index in matched_references:
            continue
        matched_candidates.add(candidate_index)
        matched_references.add(reference_index)
        score += 2.0 - normalized
    return len(matched_candidates), score


def align_features(
    reference: Sequence[MinutiaFeature],
    probe: Sequence[MinutiaFeature],
    distance_threshold: float,
    angle_threshold_degrees: float,
    max_candidates: int,
) -> AlignmentResult:
    if not reference or not probe:
        return AlignmentResult(tuple(probe), 0, 0.0, 0.0, 0.0, 0.0)

    angle_threshold = math.radians(angle_threshold_degrees)
    reference_candidates = sorted(reference, key=lambda item: item.quality, reverse=True)[
        :max_candidates
    ]
    probe_candidates = sorted(probe, key=lambda item: item.quality, reverse=True)[
        :max_candidates
    ]
    best = AlignmentResult(tuple(probe), 0, 0.0, 0.0, 0.0, 0.0)

    for target in reference_candidates:
        for source in probe_candidates:
            if target.kind != source.kind:
                continue
            angle = math.atan2(
                math.sin(target.direction - source.direction),
                math.cos(target.direction - source.direction),
            )
            cosine = math.cos(angle)
            sine = math.sin(angle)
            translated_x = target.x - (cosine * source.x - sine * source.y)
            translated_y = target.y - (sine * source.x + cosine * source.y)
            transformed = transform_features(probe, angle, translated_x, translated_y)
            matches, score = count_alignment_matches(
                reference, transformed, distance_threshold, angle_threshold
            )
            if (matches, score) > (best.matches, best.score):
                best = AlignmentResult(
                    transformed,
                    matches,
                    score,
                    angle,
                    translated_x,
                    translated_y,
                )
    return best


def canonical_tokens(
    minutiae: Sequence[MinutiaFeature],
    reference_shape: tuple[int, int],
    grid_size: int,
    angle_bins: int,
) -> set[int]:
    """Quantize aligned minutiae in the fixed enrollment image frame."""
    if not minutiae:
        return set()

    height, width = reference_shape
    tokens: set[int] = set()
    for minutia in minutiae:
        if not (0.0 <= minutia.x < width and 0.0 <= minutia.y < height):
            continue
        nx = min(1.0 - 1e-12, max(0.0, minutia.x / width))
        ny = min(1.0 - 1e-12, max(0.0, minutia.y / height))
        gx = int(nx * grid_size)
        gy = int(ny * grid_size)
        ga = int(minutia.direction / (2.0 * math.pi) * angle_bins) % angle_bins
        token = (((gx * grid_size) + gy) * angle_bins + ga) * 2 + minutia.kind + 1
        tokens.add(token)
    return tokens


def stable_enrollment_tokens(token_sets: Sequence[set[int]], votes: int) -> tuple[int, ...]:
    counter: Counter[int] = Counter()
    for tokens in token_sets:
        counter.update(tokens)
    stable = sorted(token for token, count in counter.items() if count >= votes)
    return tuple(stable)


def polynomial_evaluate(coefficients: Sequence[int], x: int) -> int:
    value = 0
    for coefficient in reversed(coefficients):
        value = (value * x + coefficient) % PRIME
    return value


def modular_inverse(value: int) -> int:
    return pow(value % PRIME, PRIME - 2, PRIME)


def interpolate_coefficients(points: Sequence[tuple[int, int]]) -> tuple[int, ...]:
    """Return polynomial coefficients using Lagrange interpolation over PRIME."""
    count = len(points)
    result = [0] * count
    for i, (x_i, y_i) in enumerate(points):
        basis = [1]
        denominator = 1
        for j, (x_j, _) in enumerate(points):
            if i == j:
                continue
            next_basis = [0] * (len(basis) + 1)
            for degree, coefficient in enumerate(basis):
                next_basis[degree] = (next_basis[degree] - coefficient * x_j) % PRIME
                next_basis[degree + 1] = (next_basis[degree + 1] + coefficient) % PRIME
            basis = next_basis
            denominator = denominator * (x_i - x_j) % PRIME
        scale = y_i * modular_inverse(denominator) % PRIME
        for degree, coefficient in enumerate(basis):
            result[degree] = (result[degree] + scale * coefficient) % PRIME
    return tuple(result)


def coefficients_to_bytes(coefficients: Sequence[int]) -> bytes:
    return b"".join(int(value).to_bytes(4, "big") for value in coefficients)


def key_from_coefficients(coefficients: Sequence[int]) -> bytes:
    return hashlib.sha256(b"FVC2002-fuzzy-vault-key\x00" + coefficients_to_bytes(coefficients)).digest()


def verifier_for_coefficients(coefficients: Sequence[int]) -> str:
    return hashlib.sha256(b"FVC2002-fuzzy-vault-check\x00" + coefficients_to_bytes(coefficients)).hexdigest()


def build_vault(
    stable_tokens: Sequence[int],
    polynomial_points: int,
    chaff_points: int,
    universe_size: int,
    rng: random.Random,
) -> tuple[bytes, Vault]:
    if len(stable_tokens) < polynomial_points:
        raise ValueError(
            f"Only {len(stable_tokens)} stable tokens; {polynomial_points} are required"
        )
    # A fuzzy vault evaluates one degree-(k-1) polynomial at every stable
    # enrollment feature.  Reconstruction therefore needs any k overlapping
    # genuine points, not one particular randomly selected k-subset.
    genuine_x = sorted(set(stable_tokens))
    coefficients = tuple(secrets.randbelow(PRIME) for _ in range(polynomial_points))
    genuine = [(x, polynomial_evaluate(coefficients, x)) for x in genuine_x]

    # Chaff must not occupy any enrollment token coordinate.  Otherwise an
    # honest probe can select a chaff point merely because that stable minutia
    # was not one of the polynomial-bearing enrollment points.
    available = list(set(range(1, universe_size + 1)) - set(stable_tokens))
    if chaff_points > len(available):
        raise ValueError(
            f"Requested {chaff_points} chaff points, but the token universe has "
            f"only {len(available)} available x coordinates"
        )
    chaff_x = rng.sample(available, chaff_points)
    chaff: list[tuple[int, int]] = []
    for x in chaff_x:
        true_y = polynomial_evaluate(coefficients, x)
        y = secrets.randbelow(PRIME)
        while y == true_y:
            y = secrets.randbelow(PRIME)
        chaff.append((x, y))

    points = genuine + chaff
    rng.shuffle(points)
    vault = Vault(
        points=tuple(points),
        verifier=verifier_for_coefficients(coefficients),
        polynomial_points=polynomial_points,
        universe_size=universe_size,
    )
    return key_from_coefficients(coefficients), vault


def candidate_combinations(
    points: Sequence[tuple[int, int]],
    choose: int,
    maximum: int,
    rng: random.Random,
) -> Iterable[tuple[tuple[int, int], ...]]:
    total = math.comb(len(points), choose)
    if total <= maximum:
        yield from itertools.combinations(points, choose)
        return

    seen: set[tuple[int, ...]] = set()
    while len(seen) < maximum:
        indices = tuple(sorted(rng.sample(range(len(points)), choose)))
        if indices in seen:
            continue
        seen.add(indices)
        yield tuple(points[index] for index in indices)


def unlock_vault(
    vault: Vault,
    probe_tokens: set[int],
    max_combinations: int,
    rng: random.Random,
) -> tuple[bytes | None, int, int]:
    candidates = [point for point in vault.points if point[0] in probe_tokens]
    if len(candidates) < vault.polynomial_points:
        return None, len(candidates), 0

    tried = 0
    for subset in candidate_combinations(
        candidates, vault.polynomial_points, max_combinations, rng
    ):
        tried += 1
        coefficients = interpolate_coefficients(subset)
        if verifier_for_coefficients(coefficients) == vault.verifier:
            return key_from_coefficients(coefficients), len(candidates), tried
    return None, len(candidates), tried


def estimate_helper_bytes(vault: Vault, reference_minutiae_count: int) -> int:
    # Two 32-bit integers per vault point, SHA-256 verifier, compact parameters,
    # and an approximate 20-byte public alignment record per reference minutia.
    return len(vault.points) * 8 + 32 + 12 + reference_minutiae_count * 20


def enroll_subject(
    pyfing: Any,
    records: Sequence[ImageRecord],
    config: Config,
    rng: random.Random,
) -> Enrollment:
    start = now_ns()
    token_sets: list[set[int]] = []
    minutiae_total = 0
    reference_minutiae: tuple[MinutiaFeature, ...] = ()
    reference_shape = (0, 0)
    alignment_matches: list[int] = []
    for index, record in enumerate(records):
        image = load_grayscale(record)
        minutiae = extract_minutiae(
            pyfing,
            image,
            dataset_dpi(record.dataset),
            config.minutia_quality,
            config.max_minutiae,
        )
        minutiae_total += len(minutiae)
        if index == 0:
            reference_minutiae = tuple(minutiae)
            reference_shape = image.shape
            aligned = reference_minutiae
            alignment_matches.append(len(reference_minutiae))
        else:
            alignment = align_features(
                reference_minutiae,
                minutiae,
                config.alignment_distance,
                config.alignment_angle_degrees,
                config.alignment_max_candidates,
            )
            if alignment.matches < config.alignment_min_matches:
                alignment_matches.append(alignment.matches)
                continue
            aligned = alignment.features
            alignment_matches.append(alignment.matches)
        token_sets.append(
            canonical_tokens(
                aligned, reference_shape, config.grid_size, config.angle_bins
            )
        )
    if len(token_sets) < config.stable_votes:
        raise ValueError(
            f"Only {len(token_sets)} aligned enrollment impressions; "
            f"{config.stable_votes} are required"
        )
    stable_tokens = stable_enrollment_tokens(token_sets, config.stable_votes)
    universe_size = config.grid_size * config.grid_size * config.angle_bins * 2
    key, vault = build_vault(
        stable_tokens,
        config.polynomial_points,
        config.chaff_points,
        universe_size,
        rng,
    )
    return Enrollment(
        key=key,
        vault=vault,
        stable_tokens=stable_tokens,
        gen_ms=elapsed_ms(start),
        minutiae_count=minutiae_total,
        reference_minutiae=reference_minutiae,
        reference_shape=reference_shape,
        enrollment_alignment_matches=tuple(alignment_matches),
    )


def reproduce_subject(
    pyfing: Any,
    record: ImageRecord,
    enrollment: Enrollment,
    config: Config,
    rng: random.Random,
) -> Reproduction:
    start = now_ns()
    image = load_grayscale(record)
    minutiae = extract_minutiae(
        pyfing,
        image,
        dataset_dpi(record.dataset),
        config.minutia_quality,
        config.max_minutiae,
    )
    alignment = align_features(
        enrollment.reference_minutiae,
        minutiae,
        config.alignment_distance,
        config.alignment_angle_degrees,
        config.alignment_max_candidates,
    )
    if alignment.matches < config.alignment_min_matches:
        return Reproduction(
            key=None,
            rep_ms=elapsed_ms(start),
            probe_tokens=0,
            matched_vault_points=0,
            combinations_tried=0,
            failure_reason="alignment_failed",
            alignment_matches=alignment.matches,
        )
    tokens = canonical_tokens(
        alignment.features,
        enrollment.reference_shape,
        config.grid_size,
        config.angle_bins,
    )
    key, matched, tried = unlock_vault(
        enrollment.vault, tokens, config.max_unlock_combinations, rng
    )
    reason = ""
    if key is None:
        reason = "insufficient_overlap" if matched < config.polynomial_points else "vault_unlock_failed"
    return Reproduction(
        key=key,
        rep_ms=elapsed_ms(start),
        probe_tokens=len(tokens),
        matched_vault_points=matched,
        combinations_tried=tried,
        failure_reason=reason,
        alignment_matches=alignment.matches,
    )


def group_records(records: Sequence[ImageRecord]) -> dict[str, dict[str, list[ImageRecord]]]:
    grouped: dict[str, dict[str, list[ImageRecord]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        grouped[record.dataset][record.subject_id].append(record)
    for subjects in grouped.values():
        for images in subjects.values():
            images.sort(key=lambda x: x.impression)
    return grouped


def prepare_development_images(
    pyfing: Any,
    records: Sequence[ImageRecord],
    config: Config,
) -> dict[str, PreparedImage]:
    prepared: dict[str, PreparedImage] = {}
    minimum_quality = min(TUNING_QUALITY_THRESHOLDS)
    print(
        f"Preparing {len(records)} development images for quantization tuning",
        flush=True,
    )
    for record in records:
        image = load_grayscale(record)
        minutiae = extract_minutiae(
            pyfing,
            image,
            dataset_dpi(record.dataset),
            minimum_quality,
            config.max_minutiae,
        )
        prepared[record_key(record)] = PreparedImage(
            record=record,
            shape=image.shape,
            minutiae=tuple(minutiae),
        )
    return prepared


def quality_filter(
    minutiae: Sequence[MinutiaFeature], quality: float
) -> tuple[MinutiaFeature, ...]:
    return tuple(item for item in minutiae if item.quality >= quality)


def cached_alignment(
    reference_image: PreparedImage,
    probe_image: PreparedImage,
    config: Config,
    cache: dict[tuple[str, str, float], AlignmentResult],
) -> AlignmentResult:
    key = (
        record_key(reference_image.record),
        record_key(probe_image.record),
        config.minutia_quality,
    )
    if key not in cache:
        cache[key] = align_features(
            quality_filter(reference_image.minutiae, config.minutia_quality),
            quality_filter(probe_image.minutiae, config.minutia_quality),
            config.alignment_distance,
            config.alignment_angle_degrees,
            config.alignment_max_candidates,
        )
    return cache[key]


def proxy_enrollment(
    images: Sequence[PreparedImage],
    config: Config,
    alignment_cache: dict[tuple[str, str, float], AlignmentResult],
) -> tuple[tuple[int, ...], PreparedImage] | None:
    enrollment_images = images[: config.enrollment_count]
    reference_image = enrollment_images[0]
    reference = quality_filter(reference_image.minutiae, config.minutia_quality)
    if len(reference) < config.alignment_min_matches:
        return None
    reference_shape = reference_image.shape
    token_sets = [
        canonical_tokens(reference, reference_shape, config.grid_size, config.angle_bins)
    ]
    for image in enrollment_images[1:]:
        alignment = cached_alignment(reference_image, image, config, alignment_cache)
        if alignment.matches >= config.alignment_min_matches:
            token_sets.append(
                canonical_tokens(
                    alignment.features,
                    reference_shape,
                    config.grid_size,
                    config.angle_bins,
                )
            )
    if len(token_sets) < config.stable_votes:
        return None
    stable = stable_enrollment_tokens(token_sets, config.stable_votes)
    if len(stable) < config.polynomial_points:
        return None
    return stable, reference_image


def proxy_probe_tokens(
    image: PreparedImage,
    reference_image: PreparedImage,
    config: Config,
    alignment_cache: dict[tuple[str, str, float], AlignmentResult],
) -> set[int] | None:
    alignment = cached_alignment(reference_image, image, config, alignment_cache)
    if alignment.matches < config.alignment_min_matches:
        return None
    return canonical_tokens(
        alignment.features, reference_image.shape, config.grid_size, config.angle_bins
    )


def evaluate_tuning_candidate(
    prepared: dict[str, PreparedImage],
    records: Sequence[ImageRecord],
    config: Config,
    alignment_cache: dict[tuple[str, str, float], AlignmentResult],
) -> dict[str, Any]:
    grouped = group_records(records)
    subjects_total = 0
    failures_to_enroll = 0
    genuine_total = 0
    genuine_accepts = 0
    impostor_total = 0
    impostor_accepts = 0

    for _, subjects in sorted(grouped.items()):
        for subject_id, subject_records in sorted(subjects.items()):
            subjects_total += 1
            subject_images = [prepared[record_key(item)] for item in subject_records]
            enrollment = proxy_enrollment(subject_images, config, alignment_cache)
            if enrollment is None:
                failures_to_enroll += 1
                continue
            stable, reference_image = enrollment
            stable_set = set(stable)

            for image in subject_images[config.enrollment_count :]:
                genuine_total += 1
                tokens = proxy_probe_tokens(
                    image, reference_image, config, alignment_cache
                )
                if tokens is not None and len(stable_set & tokens) >= config.polynomial_points:
                    genuine_accepts += 1

            for other_id, other_records in sorted(subjects.items()):
                if other_id == subject_id:
                    continue
                for other_record in other_records:
                    impostor_total += 1
                    tokens = proxy_probe_tokens(
                        prepared[record_key(other_record)],
                        reference_image,
                        config,
                        alignment_cache,
                    )
                    if (
                        tokens is not None
                        and len(stable_set & tokens) >= config.polynomial_points
                    ):
                        impostor_accepts += 1

    fte = failures_to_enroll / subjects_total if subjects_total else 1.0
    gar = genuine_accepts / genuine_total if genuine_total else 0.0
    far = impostor_accepts / impostor_total if impostor_total else 1.0
    score = gar - 5.0 * far - 0.5 * fte
    return {
        "grid_size": config.grid_size,
        "angle_bins": config.angle_bins,
        "minutia_quality": config.minutia_quality,
        "stable_votes": config.stable_votes,
        "subjects": subjects_total,
        "failure_to_enroll_rate": fte,
        "genuine_attempts": genuine_total,
        "genuine_accept_rate_proxy": gar,
        "impostor_attempts": impostor_total,
        "false_accept_rate_proxy": far,
        "selection_score": score,
    }


def tune_quantization(
    pyfing: Any,
    development_records: Sequence[ImageRecord],
    config: Config,
) -> tuple[Config, TuningResult]:
    prepared = prepare_development_images(pyfing, development_records, config)
    candidates: list[dict[str, Any]] = []
    alignment_cache: dict[tuple[str, str, float], AlignmentResult] = {}
    best_config: Config | None = None
    best_key: tuple[float, float, float, float] | None = None

    for grid_size in TUNING_GRID_SIZES:
        for angle_bins in TUNING_ANGLE_BINS:
            for quality in TUNING_QUALITY_THRESHOLDS:
                for stable_votes in TUNING_STABLE_VOTES:
                    if stable_votes > config.enrollment_count:
                        continue
                    candidate_config = replace(
                        config,
                        grid_size=grid_size,
                        angle_bins=angle_bins,
                        minutia_quality=quality,
                        stable_votes=stable_votes,
                    )
                    metrics = evaluate_tuning_candidate(
                        prepared, development_records, candidate_config, alignment_cache
                    )
                    candidates.append(metrics)
                    key = (
                        metrics["selection_score"],
                        -metrics["false_accept_rate_proxy"],
                        metrics["genuine_accept_rate_proxy"],
                        -metrics["failure_to_enroll_rate"],
                    )
                    if best_key is None or key > best_key:
                        best_key = key
                        best_config = candidate_config

    if best_config is None:
        raise RuntimeError("Quantization tuning produced no candidate configuration")
    selected = evaluate_tuning_candidate(
        prepared, development_records, best_config, alignment_cache
    )
    print(f"Selected development configuration: {json.dumps(selected, sort_keys=True)}")
    return best_config, TuningResult(
        selected=selected,
        candidates=tuple(candidates),
        development_datasets=config.development_datasets,
        development_subjects=config.development_subjects,
    )


def impostor_records(
    subjects: dict[str, list[ImageRecord]],
    subject_id: str,
    mode: str,
) -> list[ImageRecord]:
    if mode == "none":
        return []
    other_ids = [value for value in sorted(subjects) if value != subject_id]
    if mode == "one":
        ordered = sorted(subjects)
        index = ordered.index(subject_id)
        other_ids = [ordered[(index + 1) % len(ordered)]]
    if mode == "all-images":
        return [record for value in other_ids for record in subjects[value]]
    return [subjects[value][0] for value in other_ids]


def benchmark(config: Config, records: Sequence[ImageRecord]) -> pd.DataFrame:
    pyfing = load_pyfing()
    grouped = group_records(records)
    rows: list[dict[str, Any]] = []
    master_rng = random.Random(config.random_seed)

    for dataset, subjects in sorted(grouped.items()):
        for subject_id, images in sorted(subjects.items()):
            print(f"[{dataset}] enrollment subject {subject_id}", flush=True)
            enrollment_images = images[: config.enrollment_count]
            subject_seed = master_rng.randrange(0, 2**63)
            subject_rng = random.Random(subject_seed)
            try:
                enrollment = enroll_subject(
                    pyfing, enrollment_images, config, subject_rng
                )
            except ValueError as exc:
                rows.append(
                    {
                        "dataset": dataset,
                        "dpi": dataset_dpi(dataset),
                        "subject_id": subject_id,
                        "probe_id": "",
                        "attempt_type": "enrollment",
                        "gen_ms": np.nan,
                        "rep_ms": np.nan,
                        "success": False,
                        "failure_reason": f"failure_to_enroll: {exc}",
                        "key_match": False,
                        "enrollment_images": ";".join(x.member_name for x in enrollment_images),
                        "enrollment_minutiae": 0,
                        "stable_tokens": 0,
                        "probe_tokens": 0,
                        "matched_vault_points": 0,
                        "alignment_matches": 0,
                        "combinations_tried": 0,
                        "vault_points": 0,
                        "helper_bytes": 0,
                        "grid_size": config.grid_size,
                        "angle_bins": config.angle_bins,
                        "minutia_quality": config.minutia_quality,
                        "stable_votes": config.stable_votes,
                        "backend": "pyfing-leader-aligned-polynomial-fuzzy-vault",
                    }
                )
                continue

            probes: list[tuple[str, ImageRecord]] = [
                ("genuine", record) for record in images[config.enrollment_count :]
            ]
            probes.extend(
                ("impostor", record)
                for record in impostor_records(subjects, subject_id, config.impostor_mode)
            )
            for attempt_type, record in probes:
                probe_seed_material = (
                    f"{subject_seed}|{record.member_name}|{attempt_type}".encode("utf-8")
                )
                probe_seed = int.from_bytes(
                    hashlib.sha256(probe_seed_material).digest()[:8], "big"
                )
                reproduction = reproduce_subject(
                    pyfing,
                    record,
                    enrollment,
                    config,
                    random.Random(probe_seed),
                )
                key_match = reproduction.key == enrollment.key
                # Genuine success means correct reconstruction.  An impostor
                # success is intentionally recorded as a false accept.
                success = bool(key_match)
                rows.append(
                    {
                        "dataset": dataset,
                        "dpi": dataset_dpi(dataset),
                        "subject_id": subject_id,
                        "probe_id": record.member_name,
                        "probe_subject_id": record.subject_id,
                        "attempt_type": attempt_type,
                        "gen_ms": enrollment.gen_ms,
                        "rep_ms": reproduction.rep_ms,
                        "success": success,
                        "failure_reason": reproduction.failure_reason,
                        "key_match": key_match,
                        "enrollment_images": ";".join(x.member_name for x in enrollment_images),
                        "enrollment_minutiae": enrollment.minutiae_count,
                        "stable_tokens": len(enrollment.stable_tokens),
                        "probe_tokens": reproduction.probe_tokens,
                        "matched_vault_points": reproduction.matched_vault_points,
                        "alignment_matches": reproduction.alignment_matches,
                        "combinations_tried": reproduction.combinations_tried,
                        "vault_points": len(enrollment.vault.points),
                        "helper_bytes": estimate_helper_bytes(
                            enrollment.vault, len(enrollment.reference_minutiae)
                        ),
                        "grid_size": config.grid_size,
                        "angle_bins": config.angle_bins,
                        "minutia_quality": config.minutia_quality,
                        "stable_votes": config.stable_votes,
                        "backend": "pyfing-leader-aligned-polynomial-fuzzy-vault",
                    }
                )
    return pd.DataFrame.from_records(rows)


def summarize_results(
    dataframe: pd.DataFrame,
    config: Config,
    tuning: TuningResult | None,
) -> dict[str, Any]:
    enrollment_rows = dataframe[dataframe["attempt_type"] == "enrollment"]
    attempts = dataframe[dataframe["attempt_type"].isin(["genuine", "impostor"])]
    genuine = attempts[attempts["attempt_type"] == "genuine"]
    impostor = attempts[attempts["attempt_type"] == "impostor"]

    def rate(frame: pd.DataFrame, success_value: bool) -> float | None:
        if frame.empty:
            return None
        return float((frame["success"].astype(bool) == success_value).mean())

    successful_genuine = genuine[genuine["success"].astype(bool)]
    successful_enrollments = attempts[["dataset", "subject_id"]].drop_duplicates()
    enrollment_subjects = len(enrollment_rows) + len(successful_enrollments)
    summary: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": SCRIPT_VERSION,
        "configuration": {
            **asdict(config),
            "dataset_zips": [str(x) for x in config.dataset_zips],
            "output": str(config.output),
            "summary_output": str(config.summary_output),
        },
        "development_tuning": asdict(tuning) if tuning is not None else None,
        "counts": {
            "rows": int(len(dataframe)),
            "failure_to_enroll": int(len(enrollment_rows)),
            "genuine_attempts": int(len(genuine)),
            "impostor_attempts": int(len(impostor)),
            "successful_genuine": int(genuine["success"].astype(bool).sum()) if not genuine.empty else 0,
            "false_accepts": int(impostor["success"].astype(bool).sum()) if not impostor.empty else 0,
        },
        "rates": {
            "failure_to_enroll_rate": (
                float(len(enrollment_rows) / enrollment_subjects)
                if enrollment_subjects
                else None
            ),
            "genuine_accept_rate": rate(genuine, True),
            "false_reject_rate": rate(genuine, False),
            "false_accept_rate": rate(impostor, True),
        },
        "timing_ms_successful_genuine": {},
    }
    if not successful_genuine.empty:
        for column in ("gen_ms", "rep_ms"):
            values = successful_genuine[column].astype(float).to_numpy()
            summary["timing_ms_successful_genuine"][column] = {
                "n": int(len(values)),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "p95": float(np.percentile(values, 95)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
    return summary


def run_self_test() -> None:
    rng = random.Random(7)
    stable = tuple(range(1, 31))
    key, vault = build_vault(stable, 5, 40, 512, rng)
    recovered, matched, tried = unlock_vault(
        vault, set(stable), 20_000, random.Random(8)
    )
    if recovered != key or matched < 5 or tried < 1:
        raise SystemExit("Fuzzy-vault self-test failed for a genuine token set")
    rejected, _, _ = unlock_vault(
        vault, set(range(400, 430)), 20_000, random.Random(9)
    )
    if rejected is not None:
        raise SystemExit("Fuzzy-vault self-test accepted an unrelated token set")

    reference = tuple(
        MinutiaFeature(
            x=30.0 + index * 17.0,
            y=40.0 + (index % 3) * 23.0,
            direction=(0.25 * index) % (2.0 * math.pi),
            kind=index % 2,
            quality=0.9,
        )
        for index in range(8)
    )
    displaced = transform_features(reference, 0.22, 14.0, -9.0)
    alignment = align_features(reference, displaced, 3.0, 8.0, 8)
    if alignment.matches != len(reference):
        raise SystemExit(
            f"Alignment self-test recovered {alignment.matches}/{len(reference)} matches"
        )
    reference_tokens = canonical_tokens(reference, (256, 256), 16, 4)
    aligned_tokens = canonical_tokens(alignment.features, (256, 256), 16, 4)
    if reference_tokens != aligned_tokens:
        raise SystemExit("Alignment self-test did not reproduce enrollment tokens")
    print("Self-test passed")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-zips", nargs="+", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("fingerprint_fe_samples.csv"))
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--enrollment-count", type=int, default=3)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--angle-bins", type=int, default=4)
    parser.add_argument("--stable-votes", type=int, default=2)
    parser.add_argument("--polynomial-points", type=int, default=5)
    parser.add_argument("--chaff-points", type=int, default=160)
    parser.add_argument("--max-unlock-combinations", type=int, default=50_000)
    parser.add_argument("--minutia-quality", type=float, default=0.60)
    parser.add_argument("--max-minutiae", type=int, default=80)
    parser.add_argument(
        "--impostor-mode",
        choices=("none", "one", "all", "all-images"),
        default="all",
    )
    parser.add_argument(
        "--development-datasets",
        nargs="*",
        default=[],
        help="Optional whole datasets used only for parameter selection",
    )
    parser.add_argument(
        "--development-subjects",
        nargs="+",
        default=["101", "102"],
        help="Subject identifiers used only for parameter selection in each dataset",
    )
    parser.add_argument("--no-auto-tune", action="store_true")
    parser.add_argument("--alignment-distance", type=float, default=18.0)
    parser.add_argument("--alignment-angle-degrees", type=float, default=30.0)
    parser.add_argument("--alignment-min-matches", type=int, default=4)
    parser.add_argument("--alignment-max-candidates", type=int, default=16)
    parser.add_argument("--random-seed", type=int, default=20260929)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    archives = args.dataset_zips
    if archives is None:
        archives = sorted(Path.cwd().glob("DB?_B.zip"))
    summary_output = args.summary_output or args.output.with_suffix(".summary.json")

    if not archives and not args.self_test:
        parser.error("No dataset ZIPs supplied and no DB?_B.zip files found")
    if args.enrollment_count < 1:
        parser.error("--enrollment-count must be positive")
    if not 1 <= args.stable_votes <= args.enrollment_count:
        parser.error("--stable-votes must be between 1 and --enrollment-count")
    if args.grid_size < 2 or args.angle_bins < 1:
        parser.error("--grid-size must be >= 2 and --angle-bins must be positive")
    if args.polynomial_points < 2 or args.chaff_points < 0:
        parser.error("invalid fuzzy-vault parameters")
    if not 0.0 <= args.minutia_quality <= 1.0:
        parser.error("--minutia-quality must be between 0 and 1")
    if args.alignment_distance <= 0 or args.alignment_angle_degrees <= 0:
        parser.error("alignment thresholds must be positive")
    if args.alignment_min_matches < 2 or args.alignment_max_candidates < 2:
        parser.error("alignment match and candidate counts must be at least 2")

    return Config(
        dataset_zips=tuple(archives),
        output=args.output,
        summary_output=summary_output,
        enrollment_count=args.enrollment_count,
        grid_size=args.grid_size,
        angle_bins=args.angle_bins,
        stable_votes=args.stable_votes,
        polynomial_points=args.polynomial_points,
        chaff_points=args.chaff_points,
        max_unlock_combinations=args.max_unlock_combinations,
        minutia_quality=args.minutia_quality,
        max_minutiae=args.max_minutiae,
        impostor_mode=args.impostor_mode,
        development_datasets=tuple(value.upper() for value in args.development_datasets),
        development_subjects=tuple(str(value) for value in args.development_subjects),
        auto_tune=not args.no_auto_tune,
        alignment_distance=args.alignment_distance,
        alignment_angle_degrees=args.alignment_angle_degrees,
        alignment_min_matches=args.alignment_min_matches,
        alignment_max_candidates=args.alignment_max_candidates,
        random_seed=args.random_seed,
        dry_run=args.dry_run,
        self_test=args.self_test,
    )


def main() -> int:
    config = parse_args()
    if config.self_test:
        run_self_test()
        return 0

    records = discover_records(config.dataset_zips)
    dataset_info = validate_dataset(records, config.enrollment_count)
    print(json.dumps(dataset_info, indent=2, sort_keys=True))
    if config.dry_run:
        print(f"Dry run passed: {len(records)} images across {len(dataset_info)} datasets")
        return 0

    development_records = [
        record
        for record in records
        if record.dataset.upper() in config.development_datasets
        or record.subject_id in config.development_subjects
    ]
    evaluation_records = [
        record
        for record in records
        if record.dataset.upper() not in config.development_datasets
        and record.subject_id not in config.development_subjects
    ]
    tuning: TuningResult | None = None
    effective_config = config
    if config.auto_tune:
        if not development_records:
            raise SystemExit(
                "Automatic tuning requested, but no records match "
                f"development datasets/subjects={config.development_datasets}/"
                f"{config.development_subjects}"
            )
        if not evaluation_records:
            raise SystemExit("Development datasets leave no independent evaluation data")
        effective_config, tuning = tune_quantization(
            load_pyfing(), development_records, config
        )
    else:
        evaluation_records = records

    dataframe = benchmark(effective_config, evaluation_records)
    config.output.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(config.output, index=False)
    summary = summarize_results(dataframe, effective_config, tuning)
    config.summary_output.parent.mkdir(parents=True, exist_ok=True)
    config.summary_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + os.linesep,
        encoding="utf-8",
    )

    print("\nExperiment summary")
    print(json.dumps(summary["counts"], indent=2))
    print(json.dumps(summary["rates"], indent=2))
    print(f"CSV: {config.output.resolve()}")
    print(f"Summary: {config.summary_output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())