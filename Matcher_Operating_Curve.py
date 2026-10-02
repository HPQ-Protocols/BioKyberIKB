"""Describe held-out matcher scores; never tune a deployment threshold on them."""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "../../tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm


def operating_curve(genuine, impostor):
    genuine = np.sort(np.asarray(genuine, dtype=float))
    impostor = np.sort(np.asarray(impostor, dtype=float))
    if not len(genuine) or not len(impostor):
        raise ValueError("Both genuine and impostor scores are required")
    scores = np.r_[genuine, impostor]
    if not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("Scores must be finite and non-negative")
    thresholds = np.r_[0, np.unique(scores), np.nextafter(scores.max(), np.inf)]
    thresholds = np.unique(thresholds)
    rejects = np.searchsorted(genuine, thresholds, side="left")
    accepts = len(impostor) - np.searchsorted(impostor, thresholds, side="left")
    return pd.DataFrame({
        "threshold": thresholds,
        "genuine_rejects": rejects,
        "genuine_attempts": len(genuine),
        "impostor_accepts": accepts,
        "impostor_attempts": len(impostor),
        "frr": rejects / len(genuine),
        "far": accepts / len(impostor),
    })


def at_threshold(genuine, impostor, threshold):
    rejected = int(np.count_nonzero(np.asarray(genuine) < threshold))
    accepted = int(np.count_nonzero(np.asarray(impostor) >= threshold))
    return {
        "threshold": threshold,
        "genuine_rejects": rejected, "genuine_attempts": len(genuine),
        "impostor_accepts": accepted, "impostor_attempts": len(impostor),
        "frr": rejected / len(genuine), "far": accepted / len(impostor),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.csv.read_bytes()
    recorded = json.loads(args.summary.read_text())
    digest = hashlib.sha256(source).hexdigest()
    if digest != recorded["artifacts"]["csv_sha256"]:
        raise ValueError("CSV does not match the recorded digest")
    frame = pd.read_csv(args.csv)
    if set(frame.attempt_type.unique()) != {"genuine", "impostor"}:
        raise ValueError("Only genuine and impostor attempts may enter this curve")
    if frame.sensor.nunique() != 1:
        raise ValueError("Analyze each sensor separately")
    reasons = set(frame.failure_reason.fillna(""))
    if reasons - {"", "score_below_threshold"}:
        raise ValueError("Extraction/quality failures need separate accounting")
    if frame.matcher_threshold.nunique() != 1:
        raise ValueError("Expected one development-frozen threshold")
    genuine = frame.loc[frame.attempt_type == "genuine", "matcher_score"].to_numpy()
    impostor = frame.loc[frame.attempt_type == "impostor", "matcher_score"].to_numpy()
    curve = operating_curve(genuine, impostor)
    frozen = float(frame.matcher_threshold.iloc[0])
    point = at_threshold(genuine, impostor, frozen)
    operating = [at_threshold(genuine, impostor, t) for t in [frozen, 30, 40, 50]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    curve.to_csv(args.output_dir / "matcher_curve.csv", index=False)
    report = {
        "source_sha256": digest,
        "scope": "Descriptive held-out recognition curve, not PAD or new threshold selection",
        "acceptance_rule": "score >= threshold",
        "frozen_development_point": point,
        "illustrative_points_not_selected_for_deployment": operating,
        "zero_accept_caution": "Zero observed accepts is not zero population FAR",
    }
    (args.output_dir / "matcher_curve.json").write_text(json.dumps(report, indent=2) + "\n")
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.5), layout="constrained")
    axes[0].plot(curve.far, 1 - curve.frr, color="#21836a")
    axes[0].scatter([point["far"]], [1 - point["frr"]], color="#b44958", zorder=3)
    axes[0].set(xlabel="False-accept rate", ylabel="Genuine-accept rate", title="Held-out ROC", xlim=(0, .05), ylim=(.5, 1))
    # Clip only for plotting the probit endpoints, not for reported rates.
    eps_far, eps_frr = .5 / len(impostor), .5 / len(genuine)
    axes[1].plot(norm.ppf(np.clip(curve.far, eps_far, 1-eps_far)),
                 norm.ppf(np.clip(curve.frr, eps_frr, 1-eps_frr)), color="#21836a")
    axes[1].scatter([norm.ppf(np.clip(point["far"], eps_far, 1-eps_far))],
                    [norm.ppf(np.clip(point["frr"], eps_frr, 1-eps_frr))],
                    color="#b44958", zorder=3)
    ticks = np.array([.0001, .001, .01, .1, .5])
    labels = ["0.01", "0.1", "1", "10", "50"]
    for axis in [axes[1].xaxis, axes[1].yaxis]:
        axis.set_ticks(norm.ppf(ticks), labels=labels)
    axes[1].tick_params(axis="x", labelrotation=35)
    axes[1].set(xlabel="False-accept rate (%)", ylabel="False-reject rate (%)", title="Held-out DET")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.2)
    fig.savefig(args.output_dir / "matcher_roc_det.png", dpi=220)
    fig.savefig(args.output_dir / "matcher_roc_det.pdf")
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()