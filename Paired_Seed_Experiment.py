"""Measure equally gated regeneration/stored-seed sessions on original MOLF images.

The timer surrounds fresh NBIS extraction/matching and all cryptography. It
does not surround live capture, hardware sealing, PAD, or network I/O.
Historical CSV timings are never used as new measured durations.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess
import tempfile
import time

import pandas as pd


def alternating_order(index):
    return ("derive", "stored") if index % 2 == 0 else ("stored", "derive")


def summarize(frame):
    results = {}
    for name, group in frame.groupby("mode"):
        success = group[group.accepted]
        results[name] = {
            "attempts": len(group), "accepted": len(success),
            "all_attempt_mean_ms": float(group.elapsed_ms.mean()),
            "accepted_mean_ms": float(success.elapsed_ms.mean()) if len(success) else None,
            "accepted_median_ms": float(success.elapsed_ms.median()) if len(success) else None,
            "accepted_p95_ms": float(success.elapsed_ms.quantile(.95)) if len(success) else None,
            "by_attempt_type": {
                kind: {"attempts": len(part), "accepted": int(part.accepted.sum())}
                for kind, part in group.groupby("attempt_type")
            } if "attempt_type" in group else {},
        }
    wide = frame.pivot(index="pair_id", columns="mode", values="elapsed_ms")
    outcomes = frame.pivot(index="pair_id", columns="mode", values="accepted")
    matched = outcomes["derive"] & outcomes["stored"]
    delta = (wide["derive"] - wide["stored"])[matched]
    results["paired_successful_delta_derive_minus_stored"] = {
        "n": len(delta),
        "mean_ms": float(delta.mean()) if len(delta) else None,
        "median_ms": float(delta.median()) if len(delta) else None,
        "scope": "Descriptive paired difference; repeated attempts are not independent subjects",
    }
    return results


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molf-root", type=Path, required=True)
    parser.add_argument("--samples-csv", type=Path, required=True)
    parser.add_argument("--samples-summary", type=Path, required=True)
    parser.add_argument("--mindtct", default="mindtct")
    parser.add_argument("--bozorth3", default="bozorth3")
    parser.add_argument("--runs", type=int, default=5000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.runs < 1 or args.warmup < 0 or args.timeout <= 0:
        parser.error("runs/timeout must be positive and warmup non-negative")
    return args


def main():
    args = parse_args()
    # Defer optional cryptographic imports so analysis/tests work without them.
    import BioKyberIKB_Experiment as core
    import MOLF_MatcherGate_Samples as molf
    meta = json.loads(args.samples_summary.read_text())
    digest = hashlib.sha256(args.samples_csv.read_bytes()).hexdigest()
    if digest != meta["artifacts"]["csv_sha256"]:
        raise ValueError("CSV digest mismatch")
    frame = pd.read_csv(args.samples_csv)
    if frame.sensor.nunique() != 1 or frame.matcher_threshold.nunique() != 1:
        raise ValueError("Use one sensor and one development-frozen threshold")
    sensor = str(frame.sensor.iloc[0])
    threshold = int(frame.matcher_threshold.iloc[0])
    if threshold != meta["development_thresholds"][sensor]["threshold"]:
        raise ValueError("Threshold was not frozen on the recorded development set")
    mindtct = molf.require_executable(args.mindtct)
    bozorth3 = molf.require_executable(args.bozorth3)
    records = molf.discover_records(args.molf_root, [sensor])
    by_path = {r.path.relative_to(args.molf_root).as_posix(): r for r in records}
    enrollment = {r.identity_id: r for r in records if r.capture == 1}
    if any(str(p) not in by_path for p in frame.probe_id):
        raise ValueError("Original probe images are missing")
    if any(str(i) not in enrollment for i in frame.identity_id):
        raise ValueError("Original enrollment images are missing")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = core.Config(args.runs, args.warmup, 100, "ML-KEM-768", "ML-DSA-65",
                         args.output_dir, None, False, args.seed, True)
    core.validate_mechanisms(config)
    handles, operations, rows = [], {}, []
    state = {"probe": None, "score": None, "error": "", "gallery": None}
    rng = random.Random(args.seed)
    order = list(range(len(frame)))
    rng.shuffle(order)
    try:
        with tempfile.TemporaryDirectory(prefix="paired-molf-") as temp:
            gate_config = molf.Config(args.molf_root, args.output_dir / "unused.csv",
                args.output_dir / "unused.json", Path(temp), (sensor,), 20, 1,
                20, .01, .1, 1000, mindtct, bozorth3, args.timeout, args.seed,
                False, False, False)
            gallery = {}
            for identity in sorted(set(frame.identity_id)):
                template = molf.extract_template(gate_config, mindtct, enrollment[identity])
                if template.error:
                    raise RuntimeError(f"Enrollment failed: {identity}: {template.error}")
                gallery[identity] = template

            def gate():
                try:
                    probe = molf.extract_template(gate_config, mindtct, state["probe"])
                    score, _, error = molf.match_templates(
                        bozorth3, probe, state["gallery"], args.timeout)
                except (OSError, subprocess.TimeoutExpired) as exc:
                    score, error = 0, f"gate_error:{exc}"
                state["score"], state["error"] = score, error
                return not error and score >= threshold

            for identity in gallery:
                for mode in ["derive", "stored"]:
                    operation, resources, _ = core.bio_kyber_operation(
                        config, seed_mode=mode, biometric_gate=gate,
                        uid_a=str(identity).encode())
                    operations[identity, mode] = operation
                    handles.extend(resources)
            for pair in range(args.warmup + args.runs):
                item = frame.iloc[order[pair % len(order)]]
                state["probe"] = by_path[str(item.probe_id)]
                state["gallery"] = gallery[item.identity_id]
                for mode in alternating_order(pair):
                    start = time.perf_counter_ns()
                    result = operations[item.identity_id, mode]()
                    elapsed = (time.perf_counter_ns() - start) / 1e6
                    if pair >= args.warmup:
                        rows.append({
                            "pair_id": pair - args.warmup, "mode": mode,
                            "subject_id": item.subject_id, "identity_id": item.identity_id,
                            "probe_id": item.probe_id, "attempt_type": item.attempt_type,
                            "matcher_score": state["score"], "threshold": threshold,
                            "recorded_matcher_score": int(item.matcher_score),
                            "gate_error": state["error"], "accepted": result.success,
                            "elapsed_ms": elapsed, "payload_bytes": result.payload_bytes,
                        })
                if pair % 100 == 0:
                    print(f"Pair {pair}/{args.warmup + args.runs}", flush=True)
    finally:
        for handle in handles:
            handle.free()
    measured = pd.DataFrame(rows)
    measured.to_csv(args.output_dir / "paired_sessions.csv", index=False)
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": digest,
        "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "environment": core.environment_metadata(config),
        "scope": "Integrated software NBIS gate plus both cryptographic endpoints; no live capture, PAD, hardware, channel acceptance notification, or sockets",
        "models": {
            "derive": "Fresh identity/context seed HKDF inside every accepted session",
            "stored": "Enrollment-derived seed retained; identical gate, key regeneration, public-key check, transcript, confirmation, and traffic KDF",
        },
        "summary": summarize(measured),
    }
    (args.output_dir / "paired_sessions.summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()