"""Audit the supplied October 2 artifacts and prepare manuscript statistics.

Run from the project root. This analyzes existing measurements, not new trials.
"""

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(".")
INPUT = ROOT / "prism-uploads"
OUTPUT = ROOT / "IEEE-conference-template-062824" / "figures"
BOOTSTRAP_SEED = 20261002
BOOTSTRAP_RUNS = 10000


def stats(values):
    values = np.asarray(values, dtype=float)
    assert np.isfinite(values).all() and (values >= 0).all()
    return {
        "n": len(values),
        "mean_ms": float(values.mean()),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.quantile(values, 0.95)),
    }


def main():
    OUTPUT.mkdir(exist_ok=True)
    csv = INPUT / "molf_db1_gate_samples.csv"
    gate = pd.read_csv(csv)
    summary = json.loads((INPUT / "molf_db1_gate_samples.summary.json").read_text())
    assert hashlib.sha256(csv.read_bytes()).hexdigest() == summary["artifacts"]["csv_sha256"]
    assert len(gate) == 18400 and gate.subject_id.nunique() == 80
    assert gate.subject_id.between(21, 100).all()
    assert not gate.duplicated(["identity_id", "probe_id", "attempt_type"]).any()
    assert (gate.success == (gate.matcher_score >= 19)).all()
    assert np.allclose(gate.rep_ms, gate.feature_ms + gate.match_ms + gate.release_ms)
    enrolled = gate.drop_duplicates("identity_id")
    genuine = gate[gate.attempt_type == "genuine"]
    impostor = gate[gate.attempt_type == "impostor"]
    accepted = genuine[genuine.success]
    assert len(enrolled) == 800 and len(genuine) == 2400 and len(impostor) == 16000
    assert len(accepted) == 2202 and impostor.success.sum() == 128
    protocols = pd.read_csv(INPUT / "protocol_samples_3.csv")
    bio = protocols[(protocols.protocol == "Bio-Kyber IKB") & protocols.reported_ms.notna()]
    assert len(bio) == 2202 and not accepted.rep_ms.duplicated().any()
    assert np.allclose(bio.reported_ms, bio.crypto_core_ms + bio.biometric_ms)
    # These timings are unique in this artifact; use them only to recover clusters.
    joined = bio.merge(
        accepted[["rep_ms", "subject_id"]],
        left_on="biometric_ms", right_on="rep_ms", validate="one_to_one",
    )
    assert len(joined) == len(bio)
    subjects = np.sort(gate.subject_id.unique())
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draw = rng.integers(0, len(subjects), (BOOTSTRAP_RUNS, len(subjects)))

    def cluster_interval(frame, column):
        grouped = frame.groupby("subject_id")[column].agg(["sum", "count"]).reindex(subjects)
        assert grouped.notna().all().all()
        sums = grouped["sum"].to_numpy(dtype=float)
        counts = grouped["count"].to_numpy(dtype=float)
        means = sums[draw].sum(axis=1) / counts[draw].sum(axis=1)
        return [float(v) for v in np.quantile(means, [0.025, 0.975])]

    gar_interval = cluster_interval(genuine, "success")
    report = {
        "source_sha256": summary["artifacts"]["csv_sha256"],
        "bootstrap": {
            "replicates": BOOTSTRAP_RUNS, "seed": BOOTSTRAP_SEED,
            "method": "Percentile bootstrap of 80 claimed-subject clusters; fixed threshold and probe library.",
            "far_scope": "Conditional on the sampled impostor probe library, not a two-population interval.",
        },
        "genuine_accept_rate_ci95": gar_interval,
        "false_reject_rate_ci95": [1 - gar_interval[1], 1 - gar_interval[0]],
        "false_accept_rate_ci95_conditional": cluster_interval(impostor, "success"),
        "gate_mean_ci95_ms": cluster_interval(accepted, "rep_ms"),
        "component_sum_mean_ci95_ms": cluster_interval(joined, "reported_ms"),
        "biometric_timing": {
            "enrollment": stats(enrolled.gen_ms),
            **{column: stats(accepted[column]) for column in ["feature_ms", "match_ms", "release_ms", "rep_ms"]},
        },
        "protocol_timing": {},
        "primitive_timing": {},
    }
    primitive = pd.read_csv(INPUT / "primitive_samples_3.csv")
    primitive_summary = pd.read_csv(INPUT / "primitive_summary_3.csv")
    for (category, operation), frame in primitive.groupby(["category", "operation"]):
        recomputed = stats(frame.elapsed_ms)
        recorded = primitive_summary[
            (primitive_summary.category == category) & (primitive_summary.operation == operation)
        ].iloc[0]
        for field in ["n", "mean_ms", "median_ms", "p95_ms"]:
            assert np.isclose(recomputed[field], recorded[field])
        report["primitive_timing"][f"{category}/{operation}"] = recomputed
    for name, frame in protocols.groupby("protocol", sort=False):
        report["protocol_timing"][name] = {
            "core": stats(frame.crypto_core_ms),
            "reported": stats(frame.reported_ms.dropna()),
            "payload_bytes": float(frame.payload_bytes.median()),
        }
    (OUTPUT / "manuscript_statistics.json").write_text(json.dumps(report, indent=2) + "\n")

    labels = ["Bio-Kyber IKB", "ML-KEM + ML-DSA", "KEMTLS model", "Classical TLS model"]
    names = protocols.protocol.drop_duplicates().tolist()
    colors = ["#238b70", "#b64759", "#4b71a5", "#8c6a31"]
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65), layout="constrained")
    medians = [report["protocol_timing"][name]["core"]["median_ms"] for name in names]
    payloads = [report["protocol_timing"][name]["payload_bytes"] for name in names]
    for ax, values, xlabel in zip(axes, [medians, payloads], ["Core median (ms)", "Encoded payload (bytes)"]):
        ax.barh(labels, values, color=colors, height=0.6)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
    axes[0].set_title("Cryptographic core")
    axes[1].set_title("Cached-key payload")
    fig.savefig(OUTPUT / "component_comparison.pdf")
    fig.savefig(OUTPUT / "component_comparison.png", dpi=200)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()