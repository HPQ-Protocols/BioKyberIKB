#!/usr/bin/env python3
"""Evaluate matcher-gated BioSeed release on the optical MOLF subsets.

The evaluator uses NBIS MINDTCT for minutiae extraction and BOZORTH3 for
matching.  A successful match releases a software-emulated sealed random
BioSeed; the biometric does not generate cryptographic entropy.  Subjects are
split before threshold selection, so evaluation identities never influence
the selected matcher threshold.

Expected extracted MOLF directories under --molf-root:

    DB1_Lumidgm/
    DB2_Secugen/
    DB3_A_CrossMatchCropped/

DB4_Latent, DB5_SimLatent, and the uncropped DB3 slap images are ignored.
MOLF filenames are interpreted as:

    DB1/DB2: subject_capture_finger.wsq
    DB3_A:   subject_capture_slap_finger.wsq

Example:

    python MOLF_MatcherGate_Samples.py \
        --molf-root MOLF \
        --output molf_biometric_gate_samples.csv

NBIS executables ``mindtct`` and ``bozorth3`` must be on PATH.  This script is
an experimental software emulation; it does not claim to implement a hardware
secure element or presentation-attack detector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import secrets
import shutil
import subprocess
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, NamedTuple, Sequence

import numpy as np
import pandas as pd


SCRIPT_VERSION = "1.0.0"
SENSOR_DIRECTORIES = (
    "DB1_Lumidgm",
    "DB2_Secugen",
    "DB3_A_CrossMatchCropped",
)
IMAGE_SUFFIXES = {".wsq", ".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"}


@dataclass(frozen=True)
class Config:
    molf_root: Path
    output: Path
    summary_output: Path
    cache_dir: Path
    sensors: tuple[str, ...]
    development_subjects: int
    enrollment_capture: int
    impostors_per_identity: int
    target_far: float
    maximum_frr: float
    minimum_impostor_attempts: int
    mindtct: str
    bozorth3: str
    command_timeout: float
    random_seed: int
    reuse_cache: bool
    dry_run: bool
    self_test: bool


class ImageRecord(NamedTuple):
    sensor: str
    subject_id: str
    capture: int
    finger_id: str
    path: Path

    @property
    def identity_id(self) -> str:
        return f"{self.sensor}:{self.subject_id}:{self.finger_id}"


@dataclass(frozen=True)
class ExtractedTemplate:
    record: ImageRecord
    xyt_path: Path | None
    elapsed_ms: float
    error: str


@dataclass(frozen=True)
class ThresholdResult:
    threshold: int
    genuine_accept_rate: float
    false_accept_rate: float
    genuine_attempts: int
    impostor_attempts: int
    meets_target_far: bool


def now_ns() -> int:
    return time.perf_counter_ns()


def elapsed_ms(start_ns: int) -> float:
    return (time.perf_counter_ns() - start_ns) / 1_000_000.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric_key(value: str) -> tuple[int, str]:
    try:
        return int(value), value
    except ValueError:
        return math.inf, value


def parse_molf_path(path: Path, sensor: str) -> ImageRecord | None:
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        return None
    parts = path.stem.split("_")
    try:
        if sensor == "DB3_A_CrossMatchCropped":
            if len(parts) != 4:
                return None
            subject, capture_text, _slap, finger = parts
        else:
            if len(parts) != 3:
                return None
            subject, capture_text, finger = parts
        capture = int(capture_text)
    except ValueError:
        return None
    return ImageRecord(sensor, subject, capture, finger, path)


def discover_records(root: Path, sensors: Sequence[str]) -> list[ImageRecord]:
    if not root.is_dir():
        raise SystemExit(
            f"MOLF root does not exist: {root}. Extract MOLF.7z before running."
        )
    records: list[ImageRecord] = []
    for sensor in sensors:
        directory = root / sensor
        if not directory.is_dir():
            raise SystemExit(f"Required MOLF sensor directory does not exist: {directory}")
        for path in directory.rglob("*"):
            record = parse_molf_path(path, sensor)
            if record is not None:
                records.append(record)
    records.sort(
        key=lambda item: (
            item.sensor,
            numeric_key(item.subject_id),
            numeric_key(item.finger_id),
            item.capture,
        )
    )
    if not records:
        raise SystemExit("No supported MOLF images were found")
    return records


def group_identities(
    records: Iterable[ImageRecord],
) -> dict[str, dict[str, list[ImageRecord]]]:
    grouped: dict[str, dict[str, list[ImageRecord]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        grouped[record.sensor][record.identity_id].append(record)
    for identities in grouped.values():
        for images in identities.values():
            images.sort(key=lambda item: item.capture)
    return {sensor: dict(identities) for sensor, identities in grouped.items()}


def validate_records(
    grouped: dict[str, dict[str, list[ImageRecord]]], enrollment_capture: int
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for sensor, identities in sorted(grouped.items()):
        subjects = {images[0].subject_id for images in identities.values()}
        captures = sorted({item.capture for images in identities.values() for item in images})
        missing_enrollment = sum(
            not any(item.capture == enrollment_capture for item in images)
            for images in identities.values()
        )
        result[sensor] = {
            "subjects": len(subjects),
            "identities": len(identities),
            "images": sum(len(images) for images in identities.values()),
            "captures": captures,
            "missing_enrollment_capture": missing_enrollment,
        }
        if missing_enrollment:
            raise SystemExit(
                f"{sensor} has {missing_enrollment} identities without capture "
                f"{enrollment_capture}"
            )
    return result


def require_executable(value: str) -> str:
    resolved = shutil.which(value)
    if resolved is None:
        raise SystemExit(
            f"Required NBIS executable is not on PATH: {value}. "
            "Install NBIS before running the measured experiment."
        )
    return resolved


def cache_prefix(config: Config, record: ImageRecord) -> Path:
    relative = record.path.relative_to(config.molf_root)
    return config.cache_dir / relative.parent / relative.stem


def extract_template(
    config: Config, mindtct: str, record: ImageRecord
) -> ExtractedTemplate:
    prefix = cache_prefix(config, record)
    xyt_path = prefix.with_suffix(".xyt")
    timing_path = prefix.with_suffix(".timing.json")
    if config.reuse_cache and xyt_path.exists() and timing_path.exists():
        try:
            metadata = json.loads(timing_path.read_text(encoding="utf-8"))
            return ExtractedTemplate(
                record, xyt_path, float(metadata["elapsed_ms"]), ""
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass

    prefix.parent.mkdir(parents=True, exist_ok=True)
    start = now_ns()
    process = subprocess.run(
        [mindtct, str(record.path), str(prefix)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=config.command_timeout,
        check=False,
    )
    duration = elapsed_ms(start)
    if process.returncode != 0 or not xyt_path.exists():
        detail = (process.stderr or process.stdout).strip().replace("\n", " ")
        return ExtractedTemplate(
            record,
            None,
            duration,
            f"mindtct_failed:{process.returncode}:{detail[:240]}",
        )
    timing_path.write_text(
        json.dumps({"elapsed_ms": duration, "source": str(record.path)}) + os.linesep,
        encoding="utf-8",
    )
    return ExtractedTemplate(record, xyt_path, duration, "")


def match_templates(
    bozorth3: str,
    probe: ExtractedTemplate,
    gallery: ExtractedTemplate,
    timeout: float,
) -> tuple[int, float, str]:
    if probe.xyt_path is None:
        return 0, 0.0, probe.error or "probe_extraction_failed"
    if gallery.xyt_path is None:
        return 0, 0.0, gallery.error or "gallery_extraction_failed"
    start = now_ns()
    process = subprocess.run(
        [bozorth3, str(probe.xyt_path), str(gallery.xyt_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
    )
    duration = elapsed_ms(start)
    if process.returncode != 0:
        detail = (process.stderr or process.stdout).strip().replace("\n", " ")
        return 0, duration, f"bozorth3_failed:{process.returncode}:{detail[:240]}"
    integers: list[int] = []
    for token in process.stdout.replace("\n", " ").split():
        try:
            integers.append(int(token))
        except ValueError:
            continue
    if not integers:
        return 0, duration, "bozorth3_no_score"
    return integers[-1], duration, ""


def sampled_impostors(
    identity_ids: Sequence[str],
    claimed_identity: str,
    count: int,
    rng: random.Random,
) -> list[str]:
    candidates = [value for value in identity_ids if value != claimed_identity]
    if count >= len(candidates):
        return candidates
    return rng.sample(candidates, count)


def choose_threshold(
    genuine_scores: Sequence[int], impostor_scores: Sequence[int], target_far: float
) -> ThresholdResult:
    if not genuine_scores or not impostor_scores:
        raise RuntimeError("Threshold selection requires genuine and impostor scores")
    candidates = sorted(set([0, *genuine_scores, *impostor_scores, max(impostor_scores) + 1]))
    evaluations: list[ThresholdResult] = []
    for threshold in candidates:
        gar = sum(score >= threshold for score in genuine_scores) / len(genuine_scores)
        far = sum(score >= threshold for score in impostor_scores) / len(impostor_scores)
        evaluations.append(
            ThresholdResult(
                threshold=threshold,
                genuine_accept_rate=gar,
                false_accept_rate=far,
                genuine_attempts=len(genuine_scores),
                impostor_attempts=len(impostor_scores),
                meets_target_far=far <= target_far,
            )
        )
    acceptable = [item for item in evaluations if item.meets_target_far]
    pool = acceptable if acceptable else evaluations
    return max(
        pool,
        key=lambda item: (
            item.genuine_accept_rate,
            -item.false_accept_rate,
            -item.threshold,
        ),
    )


def identity_split(
    identities: dict[str, list[ImageRecord]], development_subjects: int
) -> tuple[list[str], list[str]]:
    subjects = sorted(
        {images[0].subject_id for images in identities.values()}, key=numeric_key
    )
    if not 1 <= development_subjects < len(subjects):
        raise SystemExit(
            f"--development-subjects must be between 1 and {len(subjects) - 1}"
        )
    development_set = set(subjects[:development_subjects])
    development = [
        identity
        for identity, images in identities.items()
        if images[0].subject_id in development_set
    ]
    evaluation = [identity for identity in identities if identity not in development]
    return sorted(development), sorted(evaluation)


def find_enrollment(
    images: Sequence[ImageRecord], enrollment_capture: int
) -> ImageRecord:
    return next(item for item in images if item.capture == enrollment_capture)


def prepare_templates(
    config: Config,
    mindtct: str,
    identities: dict[str, list[ImageRecord]],
) -> dict[Path, ExtractedTemplate]:
    templates: dict[Path, ExtractedTemplate] = {}
    all_images = [item for images in identities.values() for item in images]
    for index, record in enumerate(all_images, start=1):
        if index == 1 or index % 100 == 0:
            print(
                f"[mindtct] {record.sensor}: {index}/{len(all_images)}",
                flush=True,
            )
        templates[record.path] = extract_template(config, mindtct, record)
    return templates


def development_scores(
    config: Config,
    bozorth3: str,
    identities: dict[str, list[ImageRecord]],
    identity_ids: Sequence[str],
    templates: dict[Path, ExtractedTemplate],
    rng: random.Random,
) -> tuple[list[int], list[int]]:
    genuine: list[int] = []
    impostor: list[int] = []
    for identity_id in identity_ids:
        images = identities[identity_id]
        enrollment = find_enrollment(images, config.enrollment_capture)
        gallery = templates[enrollment.path]
        if gallery.xyt_path is None:
            continue
        for probe_record in images:
            if probe_record.capture == config.enrollment_capture:
                continue
            score, _, error = match_templates(
                bozorth3, templates[probe_record.path], gallery, config.command_timeout
            )
            if not error:
                genuine.append(score)
        for impostor_id in sampled_impostors(
            identity_ids,
            identity_id,
            config.impostors_per_identity,
            rng,
        ):
            impostor_record = find_enrollment(
                identities[impostor_id], config.enrollment_capture
            )
            score, _, error = match_templates(
                bozorth3,
                templates[impostor_record.path],
                gallery,
                config.command_timeout,
            )
            if not error:
                impostor.append(score)
    return genuine, impostor


def evaluate_sensor(
    config: Config,
    bozorth3: str,
    sensor: str,
    identities: dict[str, list[ImageRecord]],
    templates: dict[Path, ExtractedTemplate],
    development_ids: Sequence[str],
    evaluation_ids: Sequence[str],
    rng: random.Random,
) -> tuple[list[dict[str, Any]], ThresholdResult, dict[str, int]]:
    genuine_scores, impostor_scores = development_scores(
        config,
        bozorth3,
        identities,
        development_ids,
        templates,
        rng,
    )
    threshold = choose_threshold(genuine_scores, impostor_scores, config.target_far)
    print(
        f"[{sensor}] selected threshold {threshold.threshold}: "
        f"development GAR={threshold.genuine_accept_rate:.6f}, "
        f"FAR={threshold.false_accept_rate:.6f}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    failed_enrollments = 0
    for index, identity_id in enumerate(evaluation_ids, start=1):
        if index == 1 or index % 100 == 0:
            print(f"[{sensor}] evaluation identity {index}/{len(evaluation_ids)}", flush=True)
        images = identities[identity_id]
        enrollment_record = find_enrollment(images, config.enrollment_capture)
        gallery = templates[enrollment_record.path]
        sealed_seed_start = now_ns()
        sealed_seed = secrets.token_bytes(32)
        seed_verifier = hashlib.sha256(b"BIKB-sealed-seed\x00" + sealed_seed).digest()
        seed_operation_ms = elapsed_ms(sealed_seed_start)
        gen_ms = gallery.elapsed_ms + seed_operation_ms
        if gallery.xyt_path is None:
            failed_enrollments += 1
            continue

        for probe_record in images:
            if probe_record.capture == config.enrollment_capture:
                continue
            probe = templates[probe_record.path]
            score, match_ms, error = match_templates(
                bozorth3, probe, gallery, config.command_timeout
            )
            accepted = not error and score >= threshold.threshold
            release_start = now_ns()
            key_match = accepted and hashlib.sha256(
                b"BIKB-sealed-seed\x00" + sealed_seed
            ).digest() == seed_verifier
            release_ms = elapsed_ms(release_start)
            rows.append(
                {
                    "dataset": "MOLF",
                    "sensor": sensor,
                    "identity_id": identity_id,
                    "subject_id": enrollment_record.subject_id,
                    "finger_id": enrollment_record.finger_id,
                    "probe_id": str(probe_record.path.relative_to(config.molf_root)),
                    "probe_identity_id": identity_id,
                    "attempt_type": "genuine",
                    "gen_ms": gen_ms,
                    "feature_ms": probe.elapsed_ms,
                    "match_ms": match_ms,
                    "release_ms": release_ms,
                    "rep_ms": probe.elapsed_ms + match_ms + release_ms,
                    "matcher_score": score,
                    "matcher_threshold": threshold.threshold,
                    "success": bool(key_match),
                    "failure_reason": error if error else ("" if accepted else "score_below_threshold"),
                    "key_match": bool(key_match),
                    "backend": "nbis-mindtct-bozorth3-matcher-gated-sealed-seed",
                }
            )

        for impostor_id in sampled_impostors(
            evaluation_ids,
            identity_id,
            config.impostors_per_identity,
            rng,
        ):
            impostor_record = find_enrollment(
                identities[impostor_id], config.enrollment_capture
            )
            probe = templates[impostor_record.path]
            score, match_ms, error = match_templates(
                bozorth3, probe, gallery, config.command_timeout
            )
            accepted = not error and score >= threshold.threshold
            release_start = now_ns()
            released = accepted and hashlib.sha256(
                b"BIKB-sealed-seed\x00" + sealed_seed
            ).digest() == seed_verifier
            release_ms = elapsed_ms(release_start)
            rows.append(
                {
                    "dataset": "MOLF",
                    "sensor": sensor,
                    "identity_id": identity_id,
                    "subject_id": enrollment_record.subject_id,
                    "finger_id": enrollment_record.finger_id,
                    "probe_id": str(impostor_record.path.relative_to(config.molf_root)),
                    "probe_identity_id": impostor_id,
                    "attempt_type": "impostor",
                    "gen_ms": gen_ms,
                    "feature_ms": probe.elapsed_ms,
                    "match_ms": match_ms,
                    "release_ms": release_ms,
                    "rep_ms": probe.elapsed_ms + match_ms + release_ms,
                    "matcher_score": score,
                    "matcher_threshold": threshold.threshold,
                    "success": bool(released),
                    "failure_reason": error if error else ("" if accepted else "score_below_threshold"),
                    "key_match": bool(released),
                    "backend": "nbis-mindtct-bozorth3-matcher-gated-sealed-seed",
                }
            )
    counts = {
        "evaluation_identities": len(evaluation_ids),
        "failed_enrollments": failed_enrollments,
    }
    return rows, threshold, counts


def rate(frame: pd.DataFrame, success_value: bool) -> float | None:
    if frame.empty:
        return None
    return float((frame["success"].astype(bool) == success_value).mean())


def summarize(
    dataframe: pd.DataFrame,
    config: Config,
    thresholds: dict[str, ThresholdResult],
    sensor_counts: dict[str, dict[str, int]],
) -> dict[str, Any]:
    genuine = dataframe[dataframe["attempt_type"] == "genuine"]
    impostor = dataframe[dataframe["attempt_type"] == "impostor"]
    total_identities = sum(item["evaluation_identities"] for item in sensor_counts.values())
    failed_enrollments = sum(item["failed_enrollments"] for item in sensor_counts.values())
    fte = failed_enrollments / total_identities if total_identities else 1.0
    frr = rate(genuine, False)
    far = rate(impostor, True)
    by_sensor: dict[str, Any] = {}
    every_sensor_meets_targets = True
    for sensor, frame in dataframe.groupby("sensor"):
        sensor_genuine = frame[frame["attempt_type"] == "genuine"]
        sensor_impostor = frame[frame["attempt_type"] == "impostor"]
        sensor_identities = sensor_counts[str(sensor)]["evaluation_identities"]
        sensor_fte = (
            sensor_counts[str(sensor)]["failed_enrollments"] / sensor_identities
            if sensor_identities
            else 1.0
        )
        sensor_frr = rate(sensor_genuine, False)
        sensor_far = rate(sensor_impostor, True)
        sensor_meets = bool(
            sensor_fte <= 0.05
            and sensor_frr is not None
            and sensor_frr <= config.maximum_frr
            and sensor_far is not None
            and sensor_far <= config.target_far
        )
        every_sensor_meets_targets = every_sensor_meets_targets and sensor_meets
        by_sensor[str(sensor)] = {
            **sensor_counts[str(sensor)],
            "threshold": thresholds[str(sensor)].threshold,
            "genuine_attempts": int(len(sensor_genuine)),
            "failure_to_enroll_rate": sensor_fte,
            "false_reject_rate": sensor_frr,
            "impostor_attempts": int(len(sensor_impostor)),
            "false_accept_rate": sensor_far,
            "meets_targets": sensor_meets,
        }

    meets = bool(
        fte <= 0.05
        and frr is not None
        and frr <= config.maximum_frr
        and far is not None
        and far <= config.target_far
        and len(impostor) >= config.minimum_impostor_attempts
        and every_sensor_meets_targets
    )

    successful_genuine = genuine[genuine["success"].astype(bool)]
    timing: dict[str, Any] = {}
    for column in ("gen_ms", "feature_ms", "match_ms", "release_ms", "rep_ms"):
        values = successful_genuine[column].astype(float).to_numpy()
        if len(values):
            timing[column] = {
                "n": int(len(values)),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "p95": float(np.percentile(values, 95)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": SCRIPT_VERSION,
        "biometric_model": "matcher-gated release of a sealed random BioSeed",
        "secure_element": "software emulation only",
        "configuration": {
            **asdict(config),
            "molf_root": str(config.molf_root),
            "output": str(config.output),
            "summary_output": str(config.summary_output),
            "cache_dir": str(config.cache_dir),
        },
        "development_thresholds": {
            sensor: asdict(value) for sensor, value in thresholds.items()
        },
        "counts": {
            "rows": int(len(dataframe)),
            "evaluation_identities": total_identities,
            "failure_to_enroll": failed_enrollments,
            "genuine_attempts": int(len(genuine)),
            "impostor_attempts": int(len(impostor)),
            "successful_genuine": int(genuine["success"].astype(bool).sum()),
            "false_accepts": int(impostor["success"].astype(bool).sum()),
        },
        "rates": {
            "failure_to_enroll_rate": fte,
            "genuine_accept_rate": rate(genuine, True),
            "false_reject_rate": frr,
            "false_accept_rate": far,
        },
        "by_sensor": by_sensor,
        "paper_readiness": {
            "targets": {
                "maximum_failure_to_enroll_rate": 0.05,
                "maximum_false_reject_rate": config.maximum_frr,
                "maximum_false_accept_rate": config.target_far,
                "minimum_impostor_attempts": config.minimum_impostor_attempts,
            },
            "meets_all_targets": meets,
            "every_sensor_meets_targets": every_sensor_meets_targets,
            "status": "READY" if meets else "NOT_READY",
        },
        "timing_ms_successful_genuine": timing,
    }


def run_self_test() -> None:
    db1 = parse_molf_path(Path("DB1_Lumidgm/100_4_10.wsq"), "DB1_Lumidgm")
    db3 = parse_molf_path(
        Path("DB3_A_CrossMatchCropped/100_2_3_12.wsq"),
        "DB3_A_CrossMatchCropped",
    )
    if db1 is None or (db1.subject_id, db1.capture, db1.finger_id) != ("100", 4, "10"):
        raise SystemExit("DB1 filename parser self-test failed")
    if db3 is None or (db3.subject_id, db3.capture, db3.finger_id) != ("100", 2, "12"):
        raise SystemExit("DB3_A filename parser self-test failed")
    selected = choose_threshold([80, 70, 60], [5, 10, 20, 30], 0.25)
    if selected.false_accept_rate > 0.25:
        raise SystemExit("Threshold-selection self-test failed")
    print("Self-test passed")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molf-root", type=Path, default=Path("MOLF"))
    parser.add_argument(
        "--output", type=Path, default=Path("molf_biometric_gate_samples.csv")
    )
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=Path("molf_nbis_cache"))
    parser.add_argument("--sensors", nargs="+", choices=SENSOR_DIRECTORIES, default=list(SENSOR_DIRECTORIES))
    parser.add_argument("--development-subjects", type=int, default=20)
    parser.add_argument("--enrollment-capture", type=int, default=1)
    parser.add_argument("--impostors-per-identity", type=int, default=20)
    parser.add_argument("--target-far", type=float, default=0.01)
    parser.add_argument("--maximum-frr", type=float, default=0.10)
    parser.add_argument("--minimum-impostor-attempts", type=int, default=1000)
    parser.add_argument("--mindtct", default="mindtct")
    parser.add_argument("--bozorth3", default="bozorth3")
    parser.add_argument("--command-timeout", type=float, default=120.0)
    parser.add_argument("--random-seed", type=int, default=20260930)
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if not 0.0 <= args.target_far < 1.0:
        parser.error("--target-far must be in [0, 1)")
    if not 0.0 <= args.maximum_frr < 1.0:
        parser.error("--maximum-frr must be in [0, 1)")
    if args.impostors_per_identity < 1 or args.minimum_impostor_attempts < 1:
        parser.error("impostor counts must be positive")
    if args.command_timeout <= 0:
        parser.error("--command-timeout must be positive")

    output = args.output
    return Config(
        molf_root=args.molf_root,
        output=output,
        summary_output=args.summary_output or output.with_suffix(".summary.json"),
        cache_dir=args.cache_dir,
        sensors=tuple(args.sensors),
        development_subjects=args.development_subjects,
        enrollment_capture=args.enrollment_capture,
        impostors_per_identity=args.impostors_per_identity,
        target_far=args.target_far,
        maximum_frr=args.maximum_frr,
        minimum_impostor_attempts=args.minimum_impostor_attempts,
        mindtct=args.mindtct,
        bozorth3=args.bozorth3,
        command_timeout=args.command_timeout,
        random_seed=args.random_seed,
        reuse_cache=args.reuse_cache,
        dry_run=args.dry_run,
        self_test=args.self_test,
    )


def main() -> int:
    config = parse_args()
    if config.self_test:
        run_self_test()
        return 0

    records = discover_records(config.molf_root, config.sensors)
    grouped = group_identities(records)
    dataset_info = validate_records(grouped, config.enrollment_capture)
    print(json.dumps(dataset_info, indent=2, sort_keys=True))
    if config.dry_run:
        print(f"Dry run passed: {len(records)} optical MOLF images")
        return 0

    mindtct = require_executable(config.mindtct)
    bozorth3 = require_executable(config.bozorth3)
    all_rows: list[dict[str, Any]] = []
    thresholds: dict[str, ThresholdResult] = {}
    sensor_counts: dict[str, dict[str, int]] = {}
    master_rng = random.Random(config.random_seed)

    for sensor in config.sensors:
        identities = grouped[sensor]
        development_ids, evaluation_ids = identity_split(
            identities, config.development_subjects
        )
        print(
            f"[{sensor}] development identities={len(development_ids)}, "
            f"evaluation identities={len(evaluation_ids)}",
            flush=True,
        )
        templates = prepare_templates(config, mindtct, identities)
        rows, threshold, counts = evaluate_sensor(
            config,
            bozorth3,
            sensor,
            identities,
            templates,
            development_ids,
            evaluation_ids,
            random.Random(master_rng.randrange(0, 2**63)),
        )
        all_rows.extend(rows)
        thresholds[sensor] = threshold
        sensor_counts[sensor] = counts

    dataframe = pd.DataFrame.from_records(all_rows)
    config.output.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(config.output, index=False)
    summary = summarize(dataframe, config, thresholds, sensor_counts)
    summary["artifacts"] = {
        "csv_rows": int(len(dataframe)),
        "csv_sha256": sha256_file(config.output),
    }
    config.summary_output.parent.mkdir(parents=True, exist_ok=True)
    config.summary_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + os.linesep,
        encoding="utf-8",
    )

    print("\nExperiment summary")
    print(json.dumps(summary["counts"], indent=2))
    print(json.dumps(summary["rates"], indent=2))
    print(json.dumps(summary["paper_readiness"], indent=2))
    print(f"CSV: {config.output.resolve()}")
    print(f"Summary: {config.summary_output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())