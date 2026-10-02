# Bio-Kyber IKB

**Biometric-Gated Identity-Key Binding for ML-KEM**

Research manuscript, software experiments, and reproducibility materials for
Bio-Kyber IKB by Khiem Pham-Tuan, Minh Quang Le, and Khuong Nguyen-An.

## Overview

Bio-Kyber IKB specifies a device-bound composition of biometric authorization,
deterministic ML-KEM key regeneration, identity/context binding, and
transcript-bound key confirmation.

In the deployment design, fingerprint matching authorizes internal access to
a sealed random 256-bit seed. HKDF-SHA-256 combines that seed with an
independent device secret and encoded identity/context metadata to regenerate
an ML-KEM key pair. The expanded decapsulation key is transient rather than
persistently stored.

The fingerprint is an **authorization factor, not a source of cryptographic
entropy**. Confirmation establishes registered-key possession to an already
authenticated relying party; it does not provide independently verifiable
biometric user verification or device attestation.

This repository supports a software feasibility study. It introduces no new
cryptographic primitive and is **not a production-ready authenticator**.

## Reported Results

The recorded experiments were conducted on October 2, 2026. Reanalysis of
existing artifacts is distinct from collecting new measurements.

### Fingerprint Matcher Evaluation

MOLF DB1 contains 100 subjects, ten fingers per subject, and four captures
per finger. Subjects 1--20 are used for development; subjects 21--100 are
held out for evaluation. Capture 1 is the enrollment image and captures 2--4
are genuine probes. All fingers of each subject stay in the same partition.

The development rule maximizes genuine acceptance subject to empirical FAR
at most 1%, breaking ties by lower FAR and then lower threshold. The selected
BOZORTH3 threshold, **19**, is frozen for evaluation.

| Held-out metric | Count | Rate |
| --- | ---: | ---: |
| Evaluation subjects / finger identities | 80 / 800 | -- |
| Genuine accepts | 2,202 / 2,400 | 91.75% |
| Genuine rejects | 198 / 2,400 | 8.25% |
| Sampled impostor accepts | 128 / 16,000 | 0.80% |

ROC/DET curves describe matcher scores, not presentation-attack detection
(PAD). Operating-point uncertainty resamples claimed-subject clusters with
the sampled probe library fixed. It is not a worst-case adversarial bound.

### Paired Software Sessions

After 200 warm-up pairs, 5,000 distinct claim/probe attempts run both methods
in alternating order. The comparison is between per-session seed derivation
and an equally gated control retaining the same enrollment-derived seed.

Both methods accept **601/648 genuine** and **38/4,352 impostor** attempts.
The latency table below includes only the 601 accepted genuine pairs.

| Method | Mean (ms) | Median (ms) | P95 (ms) |
| --- | ---: | ---: | ---: |
| Per-session seed derivation | 47.58 | 44.34 | 68.25 |
| Stored-seed control | 47.82 | 44.46 | 68.21 |

The mean paired difference, derivation minus storage, is approximately
-0.24 ms, with a subject-cluster 95% bootstrap interval of [-1.05, 0.40] ms.
This single-run comparison establishes **neither a speed advantage nor
statistical equivalence**.

The timer includes fresh NBIS extraction/matching, seed derivation in the
derivation method, and both cryptographic endpoints. It excludes live capture,
PAD, hardware protection, authenticated-channel acceptance notification, and
network I/O. The 38 false accepts also complete confirmation: cryptographic
key possession does not correct a mistaken biometric gate decision.

## Repository Layout

The commands below assume this source-bundle layout and run from the
repository root:

```text
README.md
IEEE-conference-template-062824/
  IEEE-conference-template-062824.tex
  ref.bib
  figures/
    manuscript_statistics.json
    paired_statistics.json
    matcher_curve.csv
    matcher_curve.json
    matcher_roc_det.png
    matcher_roc_det.pdf
  experiments/
    MOLF_MatcherGate_Samples.py
    BioKyberIKB_Experiment.py
    Paired_Seed_Experiment.py
    Analyze_Paired_Results.py
    Matcher_Operating_Curve.py
    Prepare_Manuscript_Results.py
    Build_Reproducibility_Manifest.py
    test_research_upgrade.py
    reproducibility_manifest.json
    RESEARCH_UPGRADE.md
    REVIEW_RESPONSE.md
prism-uploads/
  molf_db1_gate_samples.csv
  molf_db1_gate_samples.summary.json
  paired_sessions.csv
  paired_sessions.summary.json
  primitive_samples_3.csv
  primitive_summary_3.csv
  protocol_samples_3.csv
  environment_3.json
  ...
```

The `prism-uploads` directory retains the supplied result artifacts. Files
ending in `_3` are used by the manuscript preparation script; earlier files
are historical artifacts and should not be silently substituted.

## Requirements

**Analysis of existing artifacts:** Python with NumPy, pandas, SciPy, and
Matplotlib. This path does not need the original fingerprint images, NBIS,
liboqs, or a GPU.

**New measurements:** the original MOLF DB1 images, NBIS MINDTCT/BOZORTH3,
`cryptography`, and liboqs/liboqs-python with seeded ML-KEM-768 key generation.
The core benchmark also requires ML-DSA-65 for its archived comparison models.

The recorded laptop environment used Python 3.12.3, NBIS 5.0.0,
liboqs/liboqs-python 0.15.0, and cryptography 48.0.0 on an Intel Core
i5-1135G7 under WSL2. See
[environment_3.json](prism-uploads/environment_3.json) for the recorded
configuration. These are provenance details, not a guarantee that other
versions reproduce identical timings.

Use an existing environment satisfying these requirements. No GPU or
PyTorch-based PAD pipeline is required by the reported experiments.

## Reproduce the Analysis

### 1. Audit Component Results

```bash
python IEEE-conference-template-062824/experiments/Prepare_Manuscript_Results.py
```

This script validates the supplied matcher CSV checksum, recomputes component
statistics, deduplicates enrollment timings, and bootstraps subject clusters.
It writes `manuscript_statistics.json` and archived component-comparison
figures under `IEEE-conference-template-062824/figures/`.

### 2. Generate Matcher ROC/DET Curves

```bash
python IEEE-conference-template-062824/experiments/Matcher_Operating_Curve.py \
  --csv prism-uploads/molf_db1_gate_samples.csv \
  --summary prism-uploads/molf_db1_gate_samples.summary.json \
  --output-dir IEEE-conference-template-062824/figures
```

Outputs include the threshold sweep CSV/JSON and ROC/DET PNG/PDF. The figure
shows marginal 95% intervals at the frozen operating point, not simultaneous
confidence bands for the entire curve. Evaluation sweeps do not select a new
deployment threshold. DET endpoint clipping affects display only.

### 3. Validate Paired Results

```bash
python IEEE-conference-template-062824/experiments/Analyze_Paired_Results.py \
  --csv prism-uploads/paired_sessions.csv \
  --summary prism-uploads/paired_sessions.summary.json \
  --source-csv prism-uploads/molf_db1_gate_samples.csv \
  --source-summary prism-uploads/molf_db1_gate_samples.summary.json \
  --output IEEE-conference-template-062824/figures/paired_statistics.json
```

Validation checks the source digest, configured sampling/order, paired rows,
scores, threshold decisions, payload lengths, and recorded summary. Genuine
accepts and false accepts are analyzed separately. The benchmark's combined
639-accept summary must not be mistaken for genuine-session latency.

### 4. Run Analysis and Control-Flow Tests

```bash
python -m unittest discover \
  -s IEEE-conference-template-062824/experiments \
  -p 'test_*.py' -v
```

Tests cover score curves, operating-point intervals, paired validation,
summary consistency, and selected control flow. Fake KEMs in these tests are
not evidence of ML-KEM security, hardware isolation, or AKE correctness.

## Optional Benchmark Reproduction

The study's scope is fixed to the recorded measurements. The commands below
document how to reproduce measurements; they are not additional results.
Write reruns to a new output directory rather than replacing supplied CSVs.

With MOLF images in `MOLF/` and NBIS executables available on `PATH`:

```bash
python IEEE-conference-template-062824/experiments/Paired_Seed_Experiment.py \
  --molf-root MOLF \
  --samples-csv prism-uploads/molf_db1_gate_samples.csv \
  --samples-summary prism-uploads/molf_db1_gate_samples.summary.json \
  --mindtct mindtct --bozorth3 bozorth3 \
  --runs 5000 --warmup 200 --seed 20261002 \
  --output-dir results/paired-seed-rerun
```

For cryptographic components, synthetic-seed correctness checks, and
archived analytical comparison models:

```bash
python IEEE-conference-template-062824/experiments/BioKyberIKB_Experiment.py \
  --runs 5000 --warmup 200 --correctness-runs 100 \
  --biometric-samples-csv prism-uploads/molf_db1_gate_samples.csv \
  --require-fe --output-dir results/core-rerun
```

The core benchmark's biometric/core total is a **component accounting sum**,
not a fresh integrated session. Its cached-seed core omits the initial seed
HKDF. The paired script measures that derivation inside its session timer.
Current core source version 1.5.1 is not an archived snapshot of the exact
historically executed version 1.5.0.

## Build the Manuscript

With a LaTeX distribution containing the packages used by the source and
`latexmk` available:

```bash
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error \
  IEEE-conference-template-062824/IEEE-conference-template-062824.tex
```

The manuscript source is
[IEEE-conference-template-062824.tex](IEEE-conference-template-062824/IEEE-conference-template-062824.tex).
The bibliography is [ref.bib](IEEE-conference-template-062824/ref.bib).

## Artifact Integrity and Provenance

The checksum manifest is
[reproducibility_manifest.json](IEEE-conference-template-062824/experiments/reproducibility_manifest.json).
Its full SHA-256 hashes identify the current source bundle and supplied input
artifacts. Relevant input hashes are:

```text
molf_db1_gate_samples.csv
e6ca579c6f2d798f2129bac2753eb988e344456dfabcae00c9c69ae0fbf6cc96

paired_sessions.csv
f42f89e665e6a171a0332dc7c175701449f9f920fe9ca716b6e51c60d2654599
```

After intentionally updating the source bundle, regenerate its manifest:

```bash
python IEEE-conference-template-062824/experiments/Build_Reproducibility_Manifest.py
```

Do not replace a recorded expected digest merely to make a validation failure
disappear. Keep the original artifacts and explain any intentional changes.
The execution-source Git commit and liboqs build commit were not recorded.
Current hashes do not retroactively prove which source revision was executed.
Use a versioned release to identify the bundle used for future replications.

## Security Scope and Limitations

- Security reasoning is conditional, not a quantified game-based AKE proof.
- Bob and registry records must already be authenticated. End-to-end
  quantum resistance also depends on those authentication mechanisms.
- The prototype uses ordinary process memory; non-exportability, hardware
  sealing, protected seed release, and reliable zeroization are design
  requirements, not demonstrated implementation properties.
- Key confirmation does not carry signed UV/PAD evidence or attestation.
- Forward secrecy is not provided. Seed or decapsulation-key compromise can
  expose recorded sessions.
- Both paired methods use an enrollment-derived 64-byte seed whose entropy
  is bounded by the 256-bit HKDF-SHA-256 extraction output. Neither output
  length nor use of an internal ML-KEM interface establishes FIPS validation.
- One sensor, one enrollment capture per finger, and a single timing run
  limit generalization. Cross-session acquisition separation is not established.
- One hundred synthetic fixed-seed correctness trials do not establish
  independent biometric enrollment correctness or complete protocol security.
- Analytical communication models are not measured network handshakes or
  equivalent-security comparisons with deployed TLS/PQ-AKE systems.

COLFISPOOF-based PAD, protected hardware, live capture/network timing,
additional sensors, independent timing replications, and formal AKE analysis
are future work. **No COLFISPOOF or PAD results are reported.**

## Data and Licensing

Raw biometric dataset images are not included in this source bundle. Obtain
MOLF and any future PAD datasets separately from their providers and check
the applicable access, usage, and redistribution terms. Dataset inventories
are not substitutes for images or experimental results.

A code license has not yet been specified in the supplied bundle. Do not
assume a particular license from repository visibility. Third-party software
and datasets retain their own licensing terms.

## Citation and Contact

The following is a provisional repository citation, not an assertion of
journal publication or a DOI. Use the published article details when available
and identify the relevant release/commit when citing experimental artifacts.

```bibtex
@misc{BioKyberIKB2026,
  author = {Pham-Tuan, Khiem and Le, Minh Quang and Nguyen-An, Khuong},
  title = {Bio-Kyber IKB: Biometric-Gated Identity-Key Binding for ML-KEM},
  year = {2026},
  howpublished = {Manuscript and experimental materials},
  url = {https://github.com/HPQ-Protocols/BioKyberIKB}
}
```

Corresponding authors: Minh Quang Le (`minh.lq@ou.edu.vn`) and
Khuong Nguyen-An (`nakhuong@hcmut.edu.vn`). Implementation contact:
Khiem Pham-Tuan (`khiempt.25ai@ou.edu.vn`; `khiempt@huit.edu.vn`).

Additional scope and reproduction notes:
[RESEARCH_UPGRADE.md](IEEE-conference-template-062824/experiments/RESEARCH_UPGRADE.md).