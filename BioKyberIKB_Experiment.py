#!/usr/bin/env python3
"""Unified Bio-Kyber IKB laptop experiment.

This script benchmarks the cryptographic core of four protocols:

1. Bio-Kyber IKB (the proposal in the paper).
2. Ephemeral ML-KEM authenticated with a cached ML-DSA credential.
3. A KEMTLS component model with a cached static server KEM credential.
4. Classical TLS 1.3 components using X25519 and ECDSA-P256.

It intentionally does not implement a fake biometric subsystem. If measured
biometric-gate timings are supplied with --biometric-samples-csv, successful
genuine verification timings are added to the Bio-Kyber cryptographic-core
samples and are clearly marked as a component sum. Without that file,
Bio-Kyber results exclude biometric cost.

The protocol measurements include both endpoint computations. Credential
provisioning and certificate validation are outside the timed region, and
public credentials are assumed to be cached/authenticated for all protocols.
Communication results therefore represent cryptographic handshake payloads,
not complete TLS records, certificates, TCP behavior, or application headers.

Required packages:
    pip install liboqs-python cryptography numpy pandas matplotlib

Example:
    python BioKyberIKB_Experiment.py --runs 1000 --warmup 100

Final-paper run with measured biometric-gate timings:
    python BioKyberIKB_Experiment.py --runs 5000 --warmup 200 \
        --biometric-samples-csv molf_biometric_gate_samples.csv --require-fe

The optional biometric CSV must contain a non-negative ``rep_ms``
column. If ``attempt_type`` and ``success`` are present, only successful
genuine reconstructions are added to successful protocol timings. An optional
``gen_ms`` column is imported once per dataset subject into primitive results.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import hmac
import importlib.metadata
import json
import math
import os
import platform
import secrets
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

# Avoid failures on managed systems whose home directory is read-only.
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.getenv("TMPDIR", "/tmp")) / "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec, x25519
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
except ImportError as exc:
    raise SystemExit(
        "The 'cryptography' package is required. Install dependencies with: "
        "pip install liboqs-python cryptography numpy pandas matplotlib"
    ) from exc

try:
    import oqs
except ImportError as exc:
    raise SystemExit(
        "liboqs-python is required. Install the package named "
        "'liboqs-python' (not the unrelated 'oqs' package)."
    ) from exc


SCRIPT_VERSION = "1.5.1"
DEFAULT_KEM = "ML-KEM-768"
DEFAULT_SIGNATURE = "ML-DSA-65"
PROTOCOL_VERSION = b"BIKBv2"
HASH_LENGTH = 32
NONCE_LENGTH = 32
HMAC_TAG_LENGTH = 32
TCP_IP_HEADER_BYTES = 40
MTU_PAYLOAD_BYTES = 1460

# These are explicit analytical assumptions, not measured network results.
NETWORK_SCENARIOS = (
    ("LAN model", 1000.0, 1.0),
    ("WAN model", 25.0, 40.0),
    ("Constrained-link model", 0.05, 600.0),
)

ROUND_TRIPS = {
    "Bio-Kyber IKB": 1.0,
    "ML-KEM + ML-DSA": 1.0,
    "KEMTLS component model": 1.5,
    "TLS 1.3 classical": 1.0,
}


@dataclass(frozen=True)
class Config:
    runs: int
    warmup: int
    correctness_runs: int
    kem_name: str
    signature_name: str
    output_dir: Path
    fe_samples_csv: Path | None
    require_fe: bool
    random_seed: int
    no_plots: bool


@dataclass(frozen=True)
class ProtocolOutcome:
    success: bool
    payload_bytes: int


class ChallengeVerifier:
    """Track outstanding one-time challenges and verify confirmation tags."""

    def __init__(self) -> None:
        self._outstanding: set[bytes] = set()

    def issue(self, nonce: bytes) -> None:
        if nonce in self._outstanding:
            raise ValueError("Challenge nonce is already outstanding")
        self._outstanding.add(nonce)

    def verify(self, nonce: bytes, received_tag: bytes, expected_tag: bytes) -> bool:
        if nonce not in self._outstanding:
            return False
        # A challenge is single use even when tag verification fails.
        self._outstanding.remove(nonce)
        return hmac.compare_digest(received_tag, expected_tag)


def now_ns() -> int:
    return time.perf_counter_ns()


def elapsed_ms(start_ns: int) -> float:
    return (time.perf_counter_ns() - start_ns) / 1_000_000.0


def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def encode_fields(*fields: bytes) -> bytes:
    """Return an injective, length-prefixed encoding of byte strings."""
    encoded = bytearray()
    for field in fields:
        if not isinstance(field, bytes):
            raise TypeError(f"encode_fields accepts bytes, got {type(field)!r}")
        encoded.extend(len(field).to_bytes(4, "big"))
        encoded.extend(field)
    return bytes(encoded)


def hkdf_sha256(ikm: bytes, salt: bytes | None, info: bytes, length: int) -> bytes:
    return HKDF(
        algorithm=hashes.SHA256(),
        length=length,
        salt=salt,
        info=info,
    ).derive(ikm)


def derive_bio_seed(
    kappa: bytes,
    biometric_secret: bytes,
    salt: bytes,
    info: bytes,
    seed_length: int,
) -> bytes:
    return hkdf_sha256(kappa + biometric_secret, salt, info, seed_length)


def derive_labeled_key(shared_secret: bytes, transcript_hash: bytes, label: bytes) -> bytes:
    return hkdf_sha256(shared_secret, transcript_hash, label + transcript_hash, 32)


def hmac_sha256(key: bytes, message: bytes) -> bytes:
    return hmac.new(key, message, hashlib.sha256).digest()


def package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def safe_oqs_version(function_name: str) -> str:
    function = getattr(oqs, function_name, None)
    if not callable(function):
        return "unavailable"
    try:
        return str(function())
    except Exception as exc:  # Metadata collection must not abort experiments.
        return f"unavailable: {exc}"


def cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def environment_metadata(config: Config) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": SCRIPT_VERSION,
        "command": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": cpu_model(),
        "logical_cpu_count": os.cpu_count(),
        "liboqs_python_version": package_version("liboqs-python"),
        "cryptography_version": package_version("cryptography"),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "oqs_python_runtime_version": safe_oqs_version("oqs_python_version"),
        "liboqs_runtime_version": safe_oqs_version("oqs_version"),
        "config": {
            **asdict(config),
            "output_dir": str(config.output_dir),
            "fe_samples_csv": str(config.fe_samples_csv) if config.fe_samples_csv else None,
        },
        "measurement_scope": (
            "Both endpoint cryptographic computations; cached authenticated "
            "credentials; no certificate parsing, socket I/O, or real secure element."
        ),
        "network_scenarios": [
            {"name": name, "bandwidth_mbps": bw, "rtt_ms": rtt}
            for name, bw, rtt in NETWORK_SCENARIOS
        ],
    }


def mechanism_details(kem_name: str, signature_name: str) -> dict[str, Any]:
    with oqs.KeyEncapsulation(kem_name) as kem:
        kem_details = dict(kem.details)
    with oqs.Signature(signature_name) as signature:
        signature_details = dict(signature.details)
    return {"kem": kem_details, "signature": signature_details}


def validate_mechanisms(config: Config) -> None:
    enabled_kems = set(oqs.get_enabled_kem_mechanisms())
    enabled_signatures = set(oqs.get_enabled_sig_mechanisms())
    if config.kem_name not in enabled_kems:
        raise SystemExit(
            f"KEM {config.kem_name!r} is not enabled. Available ML-KEM-like "
            f"mechanisms: {[x for x in sorted(enabled_kems) if 'KEM' in x or 'Kyber' in x]}"
        )
    if config.signature_name not in enabled_signatures:
        raise SystemExit(
            f"Signature {config.signature_name!r} is not enabled. Available "
            f"ML-DSA-like mechanisms: {[x for x in sorted(enabled_signatures) if 'DSA' in x or 'Dilithium' in x]}"
        )


def seeded_keypair(kem: Any, seed: bytes) -> bytes:
    generator = getattr(kem, "generate_keypair_seed", None)
    if not callable(generator):
        raise RuntimeError(
            "This liboqs-python build does not expose generate_keypair_seed(). "
            "Use a current liboqs-python/liboqs release with deterministic ML-KEM support."
        )
    return generator(seed)


def keypair_seed_length(kem: Any) -> int:
    value = kem.details.get("length_keypair_seed")
    if not isinstance(value, int) or value <= 0:
        raise RuntimeError(
            f"{kem.details.get('name', 'Selected KEM')} does not report a valid "
            "length_keypair_seed and cannot be used for Bio-Kyber regeneration."
        )
    return value


def summarize(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty sample set")
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    return {
        "n": int(array.size),
        "mean_ms": float(np.mean(array)),
        "std_ms": std,
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.percentile(array, 95)),
        "min_ms": float(np.min(array)),
        "max_ms": float(np.max(array)),
        "ci95_half_ms": float(1.96 * std / math.sqrt(array.size)),
    }


def measure_many(
    category: str,
    operation: str,
    function: Callable[[], Any],
    runs: int,
    warmup: int,
    records: list[dict[str, Any]],
) -> None:
    for _ in range(warmup):
        function()

    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for iteration in range(runs):
            start = now_ns()
            function()
            records.append(
                {
                    "category": category,
                    "operation": operation,
                    "iteration": iteration,
                    "elapsed_ms": elapsed_ms(start),
                }
            )
    finally:
        if gc_was_enabled:
            gc.enable()


def benchmark_primitives(config: Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    message = b"Bio-Kyber IKB primitive benchmark transcript"

    with (
        oqs.KeyEncapsulation(config.kem_name) as owner_kem,
        oqs.KeyEncapsulation(config.kem_name) as peer_kem,
    ):
        seed_length = keypair_seed_length(owner_kem)
        fixed_seed = secrets.token_bytes(seed_length)

        measure_many(
            "mlkem",
            "deterministic_keygen",
            lambda: seeded_keypair(owner_kem, fixed_seed),
            config.runs,
            config.warmup,
            records,
        )
        measure_many(
            "mlkem",
            "random_keygen",
            owner_kem.generate_keypair,
            config.runs,
            config.warmup,
            records,
        )

        public_key = owner_kem.generate_keypair()
        ciphertext, shared_peer = peer_kem.encap_secret(public_key)

        measure_many(
            "mlkem",
            "encapsulation",
            lambda: peer_kem.encap_secret(public_key),
            config.runs,
            config.warmup,
            records,
        )
        measure_many(
            "mlkem",
            "decapsulation",
            lambda: owner_kem.decap_secret(ciphertext),
            config.runs,
            config.warmup,
            records,
        )
        if owner_kem.decap_secret(ciphertext) != shared_peer:
            raise RuntimeError("ML-KEM primitive correctness check failed")

    with oqs.Signature(config.signature_name) as signer:
        signature_public_key = signer.generate_keypair()
        signature = signer.sign(message)

        measure_many(
            "mldsa",
            "sign",
            lambda: signer.sign(message),
            config.runs,
            config.warmup,
            records,
        )

        def verify_signature() -> None:
            if not signer.verify(message, signature, signature_public_key):
                raise RuntimeError("ML-DSA verification failed")

        measure_many(
            "mldsa",
            "verify",
            verify_signature,
            config.runs,
            config.warmup,
            records,
        )

    kappa = secrets.token_bytes(32)
    bio_secret = secrets.token_bytes(32)
    salt = secrets.token_bytes(32)
    info = encode_fields(PROTOCOL_VERSION, b"alice", b"application", config.kem_name.encode())
    measure_many(
        "kdf",
        "bio_seed_hkdf",
        lambda: derive_bio_seed(kappa, bio_secret, salt, info, 64),
        config.runs,
        config.warmup,
        records,
    )

    mac_key = secrets.token_bytes(32)
    tag = hmac_sha256(mac_key, message)
    measure_many(
        "mac",
        "hmac_generate",
        lambda: hmac_sha256(mac_key, message),
        config.runs,
        config.warmup,
        records,
    )

    def verify_hmac() -> None:
        if not hmac.compare_digest(tag, hmac_sha256(mac_key, message)):
            raise RuntimeError("HMAC verification failed")

    measure_many(
        "mac",
        "hmac_verify",
        verify_hmac,
        config.runs,
        config.warmup,
        records,
    )

    x_peer = x25519.X25519PrivateKey.generate()
    x_owner = x25519.X25519PrivateKey.generate()
    measure_many(
        "classical",
        "x25519_keygen",
        x25519.X25519PrivateKey.generate,
        config.runs,
        config.warmup,
        records,
    )
    measure_many(
        "classical",
        "x25519_exchange",
        lambda: x_owner.exchange(x_peer.public_key()),
        config.runs,
        config.warmup,
        records,
    )

    ecdsa_private = ec.generate_private_key(ec.SECP256R1())
    ecdsa_public = ecdsa_private.public_key()
    ecdsa_signature = ecdsa_private.sign(message, ec.ECDSA(hashes.SHA256()))
    measure_many(
        "classical",
        "ecdsa_p256_sign",
        lambda: ecdsa_private.sign(message, ec.ECDSA(hashes.SHA256())),
        config.runs,
        config.warmup,
        records,
    )
    measure_many(
        "classical",
        "ecdsa_p256_verify",
        lambda: ecdsa_public.verify(ecdsa_signature, message, ec.ECDSA(hashes.SHA256())),
        config.runs,
        config.warmup,
        records,
    )

    dataframe = pd.DataFrame.from_records(records)
    details = mechanism_details(config.kem_name, config.signature_name)
    return dataframe, details


def load_fe_samples(path: Path | None, require_fe: bool) -> pd.DataFrame | None:
    if path is None:
        if require_fe:
            raise SystemExit(
                "--require-fe was set but --biometric-samples-csv was not provided"
            )
        return None
    if not path.exists():
        raise SystemExit(f"Biometric-sample CSV does not exist: {path}")

    dataframe = pd.read_csv(path)
    if "rep_ms" not in dataframe.columns:
        raise SystemExit("Biometric-sample CSV must contain a 'rep_ms' column")

    if require_fe:
        required_columns = {"attempt_type", "success", "rep_ms"}
        missing_columns = sorted(required_columns - set(dataframe.columns))
        if missing_columns:
            raise SystemExit(
                "Final-paper biometric CSV is missing required evaluation "
                f"columns: {missing_columns}"
            )

        summary_path = path.with_suffix(".summary.json")
        if not summary_path.exists():
            raise SystemExit(
                "Final-paper biometric benchmarking requires the matching "
                f"summary file: {summary_path}"
            )
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(
                f"Cannot read biometric summary {summary_path}: {exc}"
            ) from exc

        readiness = summary.get("paper_readiness", {})
        if readiness.get("status") != "READY" or not readiness.get(
            "meets_all_targets", False
        ):
            raise SystemExit(
                "Biometric evaluation is NOT_READY for final-paper "
                f"benchmarking according to {summary_path}."
            )

        artifacts = summary.get("artifacts", {})
        expected_hash = artifacts.get("csv_sha256")
        expected_rows = artifacts.get("csv_rows")
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if not expected_hash or expected_hash != actual_hash:
            raise SystemExit(
                "Biometric CSV does not match its summary SHA-256. "
                "Regenerate both files with the biometric evaluation script."
            )
        if expected_rows != len(dataframe):
            raise SystemExit(
                "Biometric CSV row count does not match its summary. "
                "Regenerate both files with the biometric evaluation script."
            )

        attempt_type = dataframe["attempt_type"].astype(str).str.lower()
        success_text = dataframe["success"].astype(str).str.strip().str.lower()
        success = success_text.isin({"1", "true", "yes", "y"})
        genuine = attempt_type == "genuine"
        impostor = attempt_type == "impostor"
        genuine_count = int(genuine.sum())
        impostor_count = int(impostor.sum())
        false_reject_rate = (
            float((genuine & ~success).sum() / genuine_count)
            if genuine_count
            else 1.0
        )
        false_accept_rate = (
            float((impostor & success).sum() / impostor_count)
            if impostor_count
            else 1.0
        )
        readiness_failures = []
        if false_reject_rate > 0.10:
            readiness_failures.append(f"FRR={false_reject_rate:.6f} > 0.10")
        if false_accept_rate > 0.01:
            readiness_failures.append(f"FAR={false_accept_rate:.6f} > 0.01")
        if impostor_count < 1000:
            readiness_failures.append(
                f"impostor_attempts={impostor_count} < 1000"
            )
        if readiness_failures:
            raise SystemExit(
                "Biometric evaluation is NOT_READY for final-paper "
                "benchmarking: " + "; ".join(readiness_failures) + ". "
                "Omit --require-fe only for explicitly exploratory runs."
            )

    dataframe["rep_ms"] = pd.to_numeric(dataframe["rep_ms"], errors="coerce")
    dataframe = dataframe[np.isfinite(dataframe["rep_ms"]) & (dataframe["rep_ms"] >= 0)].copy()

    if "attempt_type" in dataframe.columns:
        dataframe = dataframe[
            dataframe["attempt_type"].astype(str).str.lower() == "genuine"
        ].copy()
    if "success" in dataframe.columns:
        success_text = dataframe["success"].astype(str).str.strip().str.lower()
        success = success_text.isin({"1", "true", "yes", "y"})
        dataframe = dataframe[success].copy()

    if dataframe.empty:
        raise SystemExit(
            "Biometric CSV has no successful genuine attempts with valid "
            "non-negative rep_ms values"
        )
    if "gen_ms" in dataframe.columns:
        dataframe["gen_ms"] = pd.to_numeric(dataframe["gen_ms"], errors="coerce")
    return dataframe


def append_external_fe_primitives(primitive_df: pd.DataFrame, fe_df: pd.DataFrame | None) -> pd.DataFrame:
    if fe_df is None:
        return primitive_df
    records = []
    for index, value in enumerate(fe_df["rep_ms"].dropna().astype(float)):
        records.append(
            {
                "category": "biometric_external",
                "operation": "biometric_verify_and_release",
                "iteration": index,
                "elapsed_ms": value,
            }
        )
    if "gen_ms" in fe_df.columns:
        gen_source = fe_df
        identity_columns = (
            ["identity_id"]
            if "identity_id" in gen_source.columns
            else [
                column
                for column in ("dataset", "sensor", "subject_id", "finger_id")
                if column in gen_source.columns
            ]
        )
        if identity_columns:
            gen_source = gen_source.drop_duplicates(identity_columns)
        for index, value in enumerate(gen_source["gen_ms"].dropna().astype(float)):
            if value >= 0:
                records.append(
                    {
                        "category": "biometric_external",
                        "operation": "biometric_enroll_and_seal",
                        "iteration": index,
                        "elapsed_ms": value,
                    }
                )
    return pd.concat([primitive_df, pd.DataFrame.from_records(records)], ignore_index=True)


def run_protocol_samples(
    name: str,
    operation: Callable[[], ProtocolOutcome],
    config: Config,
    records: list[dict[str, Any]],
    fe_rep_values: np.ndarray | None = None,
) -> None:
    for _ in range(config.warmup):
        outcome = operation()
        if not outcome.success:
            raise RuntimeError(f"Warm-up failed for {name}")

    paired_fe_values: np.ndarray | None = None
    if fe_rep_values is not None:
        rng = np.random.default_rng(config.random_seed)
        paired_count = min(config.runs, len(fe_rep_values))
        paired_fe_values = rng.permutation(fe_rep_values)[:paired_count]
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for iteration in range(config.runs):
            start = now_ns()
            outcome = operation()
            crypto_ms = elapsed_ms(start)
            if not outcome.success:
                raise RuntimeError(f"Protocol correctness failed for {name}, iteration {iteration}")

            fe_included = paired_fe_values is not None and iteration < len(paired_fe_values)
            fe_ms = float(paired_fe_values[iteration]) if fe_included else np.nan
            reported_ms = crypto_ms + fe_ms if fe_included else (
                crypto_ms if fe_rep_values is None else np.nan
            )

            records.append(
                {
                    "protocol": name,
                    "iteration": iteration,
                    "crypto_core_ms": crypto_ms,
                    "fe_rep_ms": fe_ms if fe_included else np.nan,
                    "biometric_ms": fe_ms if fe_included else np.nan,
                    "reported_ms": reported_ms,
                    "fe_included": fe_included,
                    "biometric_included": fe_included,
                    "payload_bytes": outcome.payload_bytes,
                    "success": True,
                }
            )
    finally:
        if gc_was_enabled:
            gc.enable()


def bio_kyber_operation(
    config: Config, *, seed_mode: str = "cached",
    biometric_gate: Callable[[], bool] | None = None,
    uid_a: bytes = b"alice@example",
) -> tuple[Callable[[], ProtocolOutcome], list[Any], dict[str, Any]]:
    if seed_mode not in {"cached", "derive", "stored"}:
        raise ValueError("seed_mode must be cached, derive, or stored")
    alice_kem = oqs.KeyEncapsulation(config.kem_name)
    bob_kem = oqs.KeyEncapsulation(config.kem_name)
    seed_length = keypair_seed_length(alice_kem)

    uid_b = b"relying-party@example"
    context = b"Bio-Kyber-IKB-authentication"
    record_version = b"1"
    kappa = secrets.token_bytes(32)
    biometric_secret = secrets.token_bytes(32)
    salt = secrets.token_bytes(32)
    info = encode_fields(PROTOCOL_VERSION, uid_a, context, config.kem_name.encode())
    deterministic_seed = derive_bio_seed(kappa, biometric_secret, salt, info, seed_length)
    enrolled_public_key = seeded_keypair(alice_kem, deterministic_seed)
    verifier = ChallengeVerifier()
    if seed_mode == "derive":
        seed_provider = lambda: derive_bio_seed(kappa, biometric_secret, salt, info, seed_length)
    else:
        seed_provider = lambda: deterministic_seed

    def operation() -> ProtocolOutcome:
        if biometric_gate is not None and not biometric_gate():
            return ProtocolOutcome(False, 0)
        # 'cached' preserves the historical 1.5.0 timing scope. The paired
        # experiment uses 'derive' versus an equally gated stored seed.
        session_seed = seed_provider()
        regenerated_public_key = seeded_keypair(alice_kem, session_seed)
        if regenerated_public_key != enrolled_public_key:
            return ProtocolOutcome(False, 0)

        nonce_b = secrets.token_bytes(NONCE_LENGTH)
        verifier.issue(nonce_b)
        ciphertext, key_b = bob_kem.encap_secret(enrolled_public_key)
        transcript = encode_fields(
            PROTOCOL_VERSION,
            uid_a,
            uid_b,
            sha256(enrolled_public_key),
            ciphertext,
            context,
            nonce_b,
            config.kem_name.encode(),
            record_version,
        )
        transcript_hash = sha256(transcript)
        confirmation_b = derive_labeled_key(key_b, transcript_hash, b"BIKB-confirm-A")
        _traffic_ab_b = derive_labeled_key(key_b, transcript_hash, b"BIKB-A2B")
        _traffic_ba_b = derive_labeled_key(key_b, transcript_hash, b"BIKB-B2A")

        message_1 = encode_fields(uid_a, uid_b, record_version, ciphertext, nonce_b)
        key_a = alice_kem.decap_secret(ciphertext)
        confirmation_a = derive_labeled_key(key_a, transcript_hash, b"BIKB-confirm-A")
        _traffic_ab_a = derive_labeled_key(key_a, transcript_hash, b"BIKB-A2B")
        _traffic_ba_a = derive_labeled_key(key_a, transcript_hash, b"BIKB-B2A")
        finished_input = encode_fields(b"BIKB-A-finished", transcript_hash)
        tag_a = hmac_sha256(confirmation_a, finished_input)
        expected_tag = hmac_sha256(confirmation_b, finished_input)
        success = key_a == key_b and verifier.verify(nonce_b, tag_a, expected_tag)
        return ProtocolOutcome(success, len(message_1) + len(tag_a))

    metadata = {
        "seed_length": seed_length,
        "persistent_device_secret_bytes": len(kappa),
        "public_key_bytes": len(enrolled_public_key),
        "seed_mode": seed_mode,
        "seed_hkdf_inside_timer": seed_mode == "derive",
        "software_gate_inside_timer": biometric_gate is not None,
        "persistent_secret_design_bytes": 64,
        "note": "Secure element is software-emulated; biometric verification is external unless CSV supplied.",
    }
    return operation, [alice_kem, bob_kem], metadata


def signed_mlkem_operation(config: Config) -> tuple[Callable[[], ProtocolOutcome], list[Any], dict[str, Any]]:
    server_kem = oqs.KeyEncapsulation(config.kem_name)
    client_kem = oqs.KeyEncapsulation(config.kem_name)
    signer = oqs.Signature(config.signature_name)
    signature_public_key = signer.generate_keypair()
    context = b"ephemeral-ML-KEM-with-ML-DSA"

    def operation() -> ProtocolOutcome:
        nonce = secrets.token_bytes(NONCE_LENGTH)
        ephemeral_public_key = server_kem.generate_keypair()
        signed_transcript = encode_fields(context, ephemeral_public_key, nonce)
        signature = signer.sign(signed_transcript)
        if not signer.verify(signed_transcript, signature, signature_public_key):
            return ProtocolOutcome(False, 0)

        ciphertext, key_client = client_kem.encap_secret(ephemeral_public_key)
        key_server = server_kem.decap_secret(ciphertext)
        transcript_hash = sha256(encode_fields(signed_transcript, signature, ciphertext))
        finish_client = hmac_sha256(
            derive_labeled_key(key_client, transcript_hash, b"signed-mlkem-finished"),
            transcript_hash,
        )
        finish_server = hmac_sha256(
            derive_labeled_key(key_server, transcript_hash, b"signed-mlkem-finished"),
            transcript_hash,
        )
        message_server = encode_fields(ephemeral_public_key, nonce, signature)
        message_client = encode_fields(ciphertext, finish_client)
        success = key_client == key_server and hmac.compare_digest(finish_client, finish_server)
        return ProtocolOutcome(success, len(message_server) + len(message_client))

    metadata = {
        "cached_signature_public_key_bytes": len(signature_public_key),
        "note": "Static ML-DSA credential provisioning/certificate bytes are excluded.",
    }
    return operation, [server_kem, client_kem, signer], metadata


def kemtls_operation(config: Config) -> tuple[Callable[[], ProtocolOutcome], list[Any], dict[str, Any]]:
    client_ephemeral_kem = oqs.KeyEncapsulation(config.kem_name)
    server_static_kem = oqs.KeyEncapsulation(config.kem_name)
    encapsulator = oqs.KeyEncapsulation(config.kem_name)
    server_static_public_key = server_static_kem.generate_keypair()
    context = b"KEMTLS-component-model"

    def operation() -> ProtocolOutcome:
        nonce = secrets.token_bytes(NONCE_LENGTH)
        client_ephemeral_public_key = client_ephemeral_kem.generate_keypair()

        ciphertext_s2c, secret_server_ephemeral = encapsulator.encap_secret(
            client_ephemeral_public_key
        )
        secret_client_ephemeral = client_ephemeral_kem.decap_secret(ciphertext_s2c)

        ciphertext_c2s, secret_client_auth = encapsulator.encap_secret(
            server_static_public_key
        )
        secret_server_auth = server_static_kem.decap_secret(ciphertext_c2s)

        transcript = encode_fields(
            context,
            client_ephemeral_public_key,
            sha256(server_static_public_key),
            ciphertext_s2c,
            ciphertext_c2s,
            nonce,
        )
        transcript_hash = sha256(transcript)
        client_master = hkdf_sha256(
            secret_client_ephemeral + secret_client_auth,
            transcript_hash,
            b"KEMTLS-master",
            32,
        )
        server_master = hkdf_sha256(
            secret_server_ephemeral + secret_server_auth,
            transcript_hash,
            b"KEMTLS-master",
            32,
        )
        client_finished = hmac_sha256(client_master, encode_fields(b"client", transcript_hash))
        server_finished = hmac_sha256(server_master, encode_fields(b"server", transcript_hash))

        message_1 = encode_fields(client_ephemeral_public_key, nonce)
        message_2 = encode_fields(ciphertext_s2c, ciphertext_c2s, server_finished)
        message_3 = encode_fields(client_finished)
        success = (
            secret_client_ephemeral == secret_server_ephemeral
            and secret_client_auth == secret_server_auth
            and client_master == server_master
        )
        return ProtocolOutcome(success, len(message_1) + len(message_2) + len(message_3))

    metadata = {
        "cached_server_kem_public_key_bytes": len(server_static_public_key),
        "note": (
            "Measured component model, not a wire-compatible KEMTLS implementation; "
            "cached static KEM credential and certificates are excluded."
        ),
    }
    return operation, [client_ephemeral_kem, server_static_kem, encapsulator], metadata


def tls13_classical_operation() -> tuple[Callable[[], ProtocolOutcome], list[Any], dict[str, Any]]:
    server_signing_key = ec.generate_private_key(ec.SECP256R1())
    server_verification_key = server_signing_key.public_key()
    context = b"TLS-1.3-classical-component-model"

    def operation() -> ProtocolOutcome:
        nonce = secrets.token_bytes(NONCE_LENGTH)
        client_private = x25519.X25519PrivateKey.generate()
        server_private = x25519.X25519PrivateKey.generate()
        client_public = client_private.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
        server_public = server_private.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
        transcript = encode_fields(context, client_public, server_public, nonce)
        signature = server_signing_key.sign(transcript, ec.ECDSA(hashes.SHA256()))
        server_verification_key.verify(signature, transcript, ec.ECDSA(hashes.SHA256()))

        secret_client = client_private.exchange(server_private.public_key())
        secret_server = server_private.exchange(client_private.public_key())
        transcript_hash = sha256(encode_fields(transcript, signature))
        client_master = hkdf_sha256(secret_client, transcript_hash, b"TLS13-master", 32)
        server_master = hkdf_sha256(secret_server, transcript_hash, b"TLS13-master", 32)
        client_finished = hmac_sha256(client_master, encode_fields(b"client", transcript_hash))
        server_finished = hmac_sha256(server_master, encode_fields(b"server", transcript_hash))

        message_1 = encode_fields(client_public, nonce)
        message_2 = encode_fields(server_public, signature, server_finished)
        message_3 = encode_fields(client_finished)
        return ProtocolOutcome(
            secret_client == secret_server and client_master == server_master,
            len(message_1) + len(message_2) + len(message_3),
        )

    public_der = server_verification_key.public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    metadata = {
        "cached_ecdsa_public_key_bytes": len(public_der),
        "note": (
            "TLS 1.3 cryptographic component model; certificate chain, TLS record "
            "framing, extensions, parsing, and network I/O are excluded."
        ),
    }
    return operation, [], metadata


def close_resources(resources: Iterable[Any]) -> None:
    for resource in resources:
        close = getattr(resource, "free", None)
        if callable(close):
            close()


def benchmark_protocols(
    config: Config,
    fe_df: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    protocol_metadata: dict[str, Any] = {}
    fe_values = fe_df["rep_ms"].to_numpy(dtype=float) if fe_df is not None else None

    factories: list[tuple[str, Callable[[], tuple[Callable[[], ProtocolOutcome], list[Any], dict[str, Any]]]]] = [
        ("Bio-Kyber IKB", lambda: bio_kyber_operation(config)),
        ("ML-KEM + ML-DSA", lambda: signed_mlkem_operation(config)),
        ("KEMTLS component model", lambda: kemtls_operation(config)),
        ("TLS 1.3 classical", tls13_classical_operation),
    ]

    for name, factory in factories:
        print(f"[protocol] {name}")
        operation, resources, metadata = factory()
        try:
            run_protocol_samples(
                name,
                operation,
                config,
                records,
                fe_rep_values=fe_values if name == "Bio-Kyber IKB" else None,
            )
        finally:
            close_resources(resources)
        protocol_metadata[name] = metadata

    return pd.DataFrame.from_records(records), protocol_metadata


def run_correctness_tests(config: Config) -> dict[str, Any]:
    with (
        oqs.KeyEncapsulation(config.kem_name) as alice_kem,
        oqs.KeyEncapsulation(config.kem_name) as bob_kem,
    ):
        seed_length = keypair_seed_length(alice_kem)
        seed = secrets.token_bytes(seed_length)
        enrolled_public_key = seeded_keypair(alice_kem, seed)
        identical_public_keys = 0
        successful_confirmations = 0

        for _ in range(config.correctness_runs):
            regenerated_public_key = seeded_keypair(alice_kem, seed)
            identical_public_keys += int(regenerated_public_key == enrolled_public_key)
            ciphertext, key_b = bob_kem.encap_secret(enrolled_public_key)
            key_a = alice_kem.decap_secret(ciphertext)
            transcript_hash = sha256(encode_fields(ciphertext, secrets.token_bytes(32)))
            confirmation_a = derive_labeled_key(key_a, transcript_hash, b"BIKB-confirm-A")
            confirmation_b = derive_labeled_key(key_b, transcript_hash, b"BIKB-confirm-A")
            tag_a = hmac_sha256(confirmation_a, transcript_hash)
            expected = hmac_sha256(confirmation_b, transcript_hash)
            successful_confirmations += int(hmac.compare_digest(tag_a, expected))

        seeded_keypair(alice_kem, seed)
        ciphertext, key_b = bob_kem.encap_secret(enrolled_public_key)
        transcript_hash = sha256(encode_fields(ciphertext, b"negative-test"))
        key_a = alice_kem.decap_secret(ciphertext)
        confirmation_a = derive_labeled_key(key_a, transcript_hash, b"BIKB-confirm-A")
        confirmation_b = derive_labeled_key(key_b, transcript_hash, b"BIKB-confirm-A")
        valid_tag = hmac_sha256(confirmation_a, transcript_hash)
        expected = hmac_sha256(confirmation_b, transcript_hash)

        modified_tag = bytearray(valid_tag)
        modified_tag[0] ^= 0x01
        modified_tag_rejected = not hmac.compare_digest(bytes(modified_tag), expected)

        modified_ciphertext = bytearray(ciphertext)
        modified_ciphertext[-1] ^= 0x01
        try:
            modified_key = alice_kem.decap_secret(bytes(modified_ciphertext))
            modified_confirmation = derive_labeled_key(
                modified_key, transcript_hash, b"BIKB-confirm-A"
            )
            modified_ct_tag = hmac_sha256(modified_confirmation, transcript_hash)
            modified_ciphertext_rejected = not hmac.compare_digest(modified_ct_tag, expected)
        except Exception:
            modified_ciphertext_rejected = True

        replay_nonce = secrets.token_bytes(32)
        replay_verifier = ChallengeVerifier()
        replay_verifier.issue(replay_nonce)
        first_nonce_accept = replay_verifier.verify(replay_nonce, valid_tag, expected)
        replay_nonce_rejected = not replay_verifier.verify(
            replay_nonce, valid_tag, expected
        )

    return {
        "runs": config.correctness_runs,
        "identical_public_keys": identical_public_keys,
        "successful_confirmations": successful_confirmations,
        "modified_tag_rejected": modified_tag_rejected,
        "modified_ciphertext_rejected": modified_ciphertext_rejected,
        "first_nonce_accept": first_nonce_accept,
        "replay_nonce_rejected": replay_nonce_rejected,
        "all_passed": (
            identical_public_keys == config.correctness_runs
            and successful_confirmations == config.correctness_runs
            and modified_tag_rejected
            and modified_ciphertext_rejected
            and first_nonce_accept
            and replay_nonce_rejected
        ),
    }


def primitive_summary(primitive_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (category, operation), group in primitive_df.groupby(["category", "operation"]):
        rows.append({"category": category, "operation": operation, **summarize(group["elapsed_ms"])})
    return pd.DataFrame(rows).sort_values(["category", "operation"]).reset_index(drop=True)


def protocol_summary(protocol_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for protocol, group in protocol_df.groupby("protocol", sort=False):
        reported = group["reported_ms"].dropna()
        if reported.empty:
            raise ValueError(f"No reportable samples for protocol {protocol}")
        stats = summarize(reported)
        rows.append(
            {
                "protocol": protocol,
                **stats,
                "crypto_core_n": int(len(group)),
                "crypto_core_median_ms": float(np.median(group["crypto_core_ms"])),
                "fe_included": bool(group["fe_included"].any()),
                "fe_source_n": int(group["fe_rep_ms"].notna().sum()),
                "biometric_source_n": int(group["biometric_ms"].notna().sum()),
                "biometric_included": bool(group["biometric_included"].any()),
                "payload_bytes_median": float(np.median(group["payload_bytes"])),
                "round_trips_model": ROUND_TRIPS[protocol],
            }
        )
    return pd.DataFrame(rows)


def wire_bytes(payload_bytes: float) -> float:
    packets = math.ceil(payload_bytes / MTU_PAYLOAD_BYTES)
    return payload_bytes + packets * TCP_IP_HEADER_BYTES


def modeled_network_latency(protocol_summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, protocol in protocol_summary_df.iterrows():
        for scenario_name, bandwidth_mbps, rtt_ms in NETWORK_SCENARIOS:
            payload_wire_bytes = wire_bytes(float(protocol["payload_bytes_median"]))
            transmission_ms = payload_wire_bytes * 8.0 / (bandwidth_mbps * 1_000_000.0) * 1000.0
            total_ms = (
                float(protocol["median_ms"])
                + transmission_ms
                + float(protocol["round_trips_model"]) * rtt_ms
            )
            rows.append(
                {
                    "protocol": protocol["protocol"],
                    "scenario": scenario_name,
                    "bandwidth_mbps": bandwidth_mbps,
                    "rtt_ms": rtt_ms,
                    "round_trips": protocol["round_trips_model"],
                    "payload_wire_bytes": payload_wire_bytes,
                    "cpu_median_ms": protocol["median_ms"],
                    "transmission_ms": transmission_ms,
                    "modeled_total_ms": total_ms,
                }
            )
    return pd.DataFrame(rows)


def plot_cpu(summary_df: pd.DataFrame, output_dir: Path, fe_supplied: bool) -> None:
    labels = summary_df["protocol"].tolist()
    medians = summary_df["median_ms"].to_numpy()
    p95 = summary_df["p95_ms"].to_numpy()
    upper = np.maximum(p95 - medians, 0)
    colors = ["#2a9d8f", "#457b9d", "#7b2cbf", "#e76f51"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, medians, color=colors, alpha=0.88)
    ax.errorbar(x, medians, yerr=[np.zeros_like(upper), upper], fmt="none", color="black", capsize=5)
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_ylabel("CPU time (ms), median with P95 upper marker")
    suffix = "including measured biometric gate" if fe_supplied else "biometric cost excluded"
    ax.set_title(f"Measured Cryptographic Protocol Cost ({suffix})")
    positive = medians[medians > 0]
    if len(positive) and float(np.max(positive) / np.min(positive)) > 100.0:
        ax.set_yscale("symlog", linthresh=1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for bar, value in zip(bars, medians):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_dir / "protocol_cpu_cost.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_communication(summary_df: pd.DataFrame, output_dir: Path) -> None:
    labels = summary_df["protocol"].tolist()
    sizes = summary_df["payload_bytes_median"].to_numpy()
    colors = ["#2a9d8f", "#457b9d", "#7b2cbf", "#e76f51"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, sizes, color=colors, alpha=0.88)
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_ylabel("Cryptographic payload (bytes)")
    ax.set_title("Measured Serialized Cryptographic Payload (Cached Credentials)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for bar, value in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{int(value)}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_dir / "protocol_communication.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff(summary_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2a9d8f", "#457b9d", "#7b2cbf", "#e76f51"]
    for color, (_, row) in zip(colors, summary_df.iterrows()):
        ax.scatter(row["payload_bytes_median"], row["median_ms"], s=100, color=color)
        ax.annotate(
            row["protocol"],
            (row["payload_bytes_median"], row["median_ms"]),
            xytext=(7, 7),
            textcoords="offset points",
        )
    ax.set_xlabel("Cryptographic payload (bytes)")
    ax.set_ylabel("Measured CPU time (ms), median")
    ax.set_title("Computation and Communication Trade-off")
    positive = summary_df["median_ms"][summary_df["median_ms"] > 0]
    if len(positive) and float(positive.max() / positive.min()) > 100.0:
        ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "protocol_tradeoff.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_network_model(network_df: pd.DataFrame, output_dir: Path) -> None:
    protocols = list(dict.fromkeys(network_df["protocol"].tolist()))
    scenarios = [name for name, _, _ in NETWORK_SCENARIOS]
    x = np.arange(len(scenarios))
    width = 0.18
    colors = ["#2a9d8f", "#457b9d", "#7b2cbf", "#e76f51"]

    fig, ax = plt.subplots(figsize=(11, 5.8))
    for index, (protocol, color) in enumerate(zip(protocols, colors)):
        values = [
            float(
                network_df[
                    (network_df["protocol"] == protocol) & (network_df["scenario"] == scenario)
                ]["modeled_total_ms"].iloc[0]
            )
            for scenario in scenarios
        ]
        ax.bar(x + (index - 1.5) * width, values, width, label=protocol, color=color)
    ax.set_xticks(x, scenarios)
    ax.set_ylabel("Modeled latency (ms)")
    ax.set_yscale("log")
    ax.set_title("Analytical Network Model Using Measured CPU and Payload")
    ax.grid(axis="y", linestyle="--", alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "modeled_network_latency.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=1000, help="Measured iterations per operation/protocol")
    parser.add_argument("--warmup", type=int, default=100, help="Warm-up iterations")
    parser.add_argument("--correctness-runs", type=int, default=100, help="Deterministic regeneration tests")
    parser.add_argument("--kem", default=DEFAULT_KEM, help="liboqs KEM mechanism")
    parser.add_argument("--signature", default=DEFAULT_SIGNATURE, help="liboqs signature mechanism")
    parser.add_argument("--output-dir", type=Path, default=None, help="Result directory")
    parser.add_argument(
        "--fe-samples-csv",
        "--biometric-samples-csv",
        dest="fe_samples_csv",
        type=Path,
        default=None,
        help="CSV with measured successful biometric verification rep_ms values",
    )
    parser.add_argument(
        "--require-fe",
        action="store_true",
        help="Abort unless measured biometric data pass the readiness checks",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=20260929,
        help="Seed for biometric sample pairing only",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG generation")
    args = parser.parse_args()

    if args.runs < 2:
        parser.error("--runs must be at least 2")
    if args.warmup < 0 or args.correctness_runs < 1:
        parser.error("--warmup must be non-negative and --correctness-runs positive")

    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = Path("results") / f"bio-kyber-{timestamp}"

    return Config(
        runs=args.runs,
        warmup=args.warmup,
        correctness_runs=args.correctness_runs,
        kem_name=args.kem,
        signature_name=args.signature,
        output_dir=output_dir,
        fe_samples_csv=args.fe_samples_csv,
        require_fe=args.require_fe,
        random_seed=args.random_seed,
        no_plots=args.no_plots,
    )


def main() -> int:
    config = parse_args()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    validate_mechanisms(config)

    fe_df = load_fe_samples(config.fe_samples_csv, config.require_fe)
    if fe_df is None:
        print(
            "WARNING: No measured biometric CSV was supplied. Bio-Kyber "
            "timings are cryptographic-core results and exclude biometric-gate cost.",
            file=sys.stderr,
        )

    metadata = environment_metadata(config)
    metadata["mechanisms"] = mechanism_details(config.kem_name, config.signature_name)
    metadata["biometric_gate"] = {
        "supplied": fe_df is not None,
        "source": str(config.fe_samples_csv) if config.fe_samples_csv else None,
        "rows": len(fe_df) if fe_df is not None else 0,
        "integration": (
            "Measured successful biometric-gate samples are added to cryptographic-core samples as a component sum."
            if fe_df is not None
            else "Not included."
        ),
    }
    write_json(config.output_dir / "environment.json", metadata)

    print(f"Output directory: {config.output_dir}")
    print(f"KEM: {config.kem_name}; signature: {config.signature_name}")
    print(f"Runs: {config.runs}; warm-up: {config.warmup}")

    print("[1/4] Primitive benchmarks")
    primitive_df, mechanism_info = benchmark_primitives(config)
    primitive_df = append_external_fe_primitives(primitive_df, fe_df)
    primitive_summary_df = primitive_summary(primitive_df)
    primitive_df.to_csv(config.output_dir / "primitive_samples.csv", index=False)
    primitive_summary_df.to_csv(config.output_dir / "primitive_summary.csv", index=False)
    write_json(config.output_dir / "mechanism_details.json", mechanism_info)

    print("[2/4] Protocol benchmarks")
    protocol_df, protocol_metadata = benchmark_protocols(config, fe_df)
    protocol_summary_df = protocol_summary(protocol_df)
    protocol_df.to_csv(config.output_dir / "protocol_samples.csv", index=False)
    protocol_summary_df.to_csv(config.output_dir / "protocol_summary.csv", index=False)
    write_json(config.output_dir / "protocol_models.json", protocol_metadata)

    print("[3/4] Correctness and negative tests")
    correctness = run_correctness_tests(config)
    write_json(config.output_dir / "correctness.json", correctness)
    if not correctness["all_passed"]:
        raise RuntimeError(f"Correctness tests failed: {correctness}")

    print("[4/4] Analytical network model and charts")
    network_df = modeled_network_latency(protocol_summary_df)
    network_df.to_csv(config.output_dir / "modeled_network_latency.csv", index=False)
    if not config.no_plots:
        plot_cpu(protocol_summary_df, config.output_dir, fe_df is not None)
        plot_communication(protocol_summary_df, config.output_dir)
        plot_tradeoff(protocol_summary_df, config.output_dir)
        plot_network_model(network_df, config.output_dir)

    print("\nProtocol summary")
    columns = [
        "protocol",
        "n",
        "crypto_core_n",
        "biometric_source_n",
        "mean_ms",
        "median_ms",
        "p95_ms",
        "ci95_half_ms",
        "payload_bytes_median",
        "biometric_included",
    ]
    print(protocol_summary_df[columns].to_string(index=False))
    print(f"\nCorrectness: {correctness}")
    print(f"Results written to: {config.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())