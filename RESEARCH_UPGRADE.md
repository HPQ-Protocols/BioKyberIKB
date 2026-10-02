# Research Upgrade Status

This work extends the supplied October 2, 2026 measurements. It does not
fabricate new PAD results, hardware guarantees, or cryptographic benchmarks.

## Finalized Experimental Scope

At the authors' decision on October 2, 2026, no further experiments are
being added to this manuscript. The final scope is the existing MOLF matcher
evaluation, ROC/DET analysis, cryptographic component/correctness measurements,
and the completed paired software-session experiment. Instructions below
document reproducibility, not an outstanding request to run new benchmarks.
COLFISPOOF/PAD, protected hardware, live capture/network measurement,
independent timing replications, and a full AKE proof are future work.

## Completed Here

- `Matcher_Operating_Curve.py` verifies the supplied CSV digest and recomputes
  descriptive held-out ROC/DET curves. Threshold 19 remains the operating
  point frozen on development subjects. Evaluation sweeps must not be used
  as a replacement for development-only threshold selection.
- The manuscript explains the equally gated stored-seed control, narrows the
  security propositions, defines unpartnered acceptance, and requires an
  authenticated application-channel acceptance notification for Alice.
- `Paired_Seed_Experiment.py` measures original-image NBIS extraction and
  matching inside the same timer as cryptography. It compares fresh seed
  derivation with retention of an enrollment-derived seed. Both methods use
  identical matching, key generation, key checks, transcript, confirmation,
  and traffic KDF. Model order alternates, and unsuccessful trials are kept.
- Historical version 1.5.0 data remain unchanged. The updated core script's
  default `cached` mode preserves their measurement scope. Only the paired
  script uses the new `derive` and `stored` modes.

## Reproduce the Available Score Analysis

Run from the project root:

```bash
python IEEE-conference-template-062824/experiments/Matcher_Operating_Curve.py \
  --csv prism-uploads/molf_db1_gate_samples.csv \
  --summary prism-uploads/molf_db1_gate_samples.summary.json \
  --output-dir IEEE-conference-template-062824/figures
```

The figure, complete threshold CSV, and JSON report are generated together.
No curve point is relabeled as a presentation-attack result. DET clipping
affects display only; zero empirical FAR remains zero in the CSV and is not
interpreted as a zero population probability.

## Paired Results Received and Verified

The uploaded `paired_sessions.csv` and `paired_sessions.summary.json` contain
5,000 pairs (10,000 method observations), after 200 warm-up pairs. The new
`Analyze_Paired_Results.py` checks the original CSV checksum, exact shuffled
sample and alternating execution order, scores, outcomes, payload accounting,
and all printed summary statistics. All checks pass for the supplied run.

Both methods accept 601/648 genuine and 38/4352 impostor attempts. The latter
are false accepts, not successful genuine authentication. Accepted genuine
latency is 47.580/47.818 ms mean, 44.336/44.456 ms median, and 68.249/68.206 ms
P95 for derivation/storage. Mean paired difference is -0.238 ms; its 95%
claimed-subject cluster bootstrap interval is [-1.046, 0.405] ms (10,000
replicates, seed 20261002). This conditional, single-run interval establishes
neither a speed advantage nor equivalence. The 639-accept printed summary
mixes 601 genuine accepts with 38 false accepts, so it is not the paper's
genuine-session latency table. The full MOLF recognition results stay separate.

Actual MOLF identity strings give 1210--1211 payload bytes; 1207 belongs to
the older component benchmark's shorter identifier. Hardware isolation,
live capture, PAD, channel notification, and sockets remain unmeasured.

Reproduce validation and disaggregated analysis from the project root:

```bash
python IEEE-conference-template-062824/experiments/Analyze_Paired_Results.py \
  --csv prism-uploads/paired_sessions.csv \
  --summary prism-uploads/paired_sessions.summary.json \
  --source-csv prism-uploads/molf_db1_gate_samples.csv \
  --source-summary prism-uploads/molf_db1_gate_samples.summary.json \
  --output IEEE-conference-template-062824/figures/paired_statistics.json
```

## Reproduce the Integrated Baseline Experiment

Prerequisites: the existing `pqc_env` with liboqs-python and cryptography,
ML-KEM-768 seeded key generation, NBIS MINDTCT/BOZORTH3, and the original MOLF
DB1 images. No environment creation or dependency installation is performed
by these scripts. Example from the project root, with datasets under `MOLF`:

```bash
python IEEE-conference-template-062824/experiments/Paired_Seed_Experiment.py \
  --molf-root MOLF \
  --samples-csv prism-uploads/molf_db1_gate_samples.csv \
  --samples-summary prism-uploads/molf_db1_gate_samples.summary.json \
  --mindtct "$HOME/nbis-install/bin/mindtct" \
  --bozorth3 "$HOME/nbis-install/bin/bozorth3" \
  --runs 5000 --warmup 200 --output-dir results/paired-seed
```

Both endpoints run in ordinary software, not a secure element. Enrollment is
outside the session timer; fresh extraction, matching, seed access/derivation,
and cryptographic confirmation are inside. Live capture, PAD, application
channel setup/acceptance notification, and socket I/O remain excluded.
NBIS intermediate files are temporary and are not reused as measured probes.
The OS file cache is not flushed. The timer includes NBIS subprocess and
intermediate-file bookkeeping overhead, equally in both methods.

The shuffled trial list cycles when runs exceed its size. Report actual
genuine/impostor denominators; 5000 is a timing budget, not the complete
18,400-row recognition evaluation. Successful paired differences are
descriptive: do not treat repeated probes as independent population samples.
The supplied run establishes no timing advantage of either method. Multiple
independent runs and a prespecified equivalence margin would be needed for
a defensible timing-equivalence claim.

## What COLFISPOOF Can and Cannot Resolve

COLFISPOOF is a contactless presentation-attack dataset, not a protected-seed
baseline, a cryptographic implementation, or a security proof. The official
authors' repository supplies partitions/preprocessing; its database page
describes 7,200 samples from 72 attack-instrument species. References:

- Kolberg et al., COLFISPOOF, WACVW 2023, pp. 653-661,
  DOI `10.1109/WACVW58289.2023.00072`.
- Priesnitz et al., COLFIPAD, IJCB 2023,
  DOI `10.1109/IJCB57857.2023.10448552`, combines COLFISPOOF attacks with
  contactless bona fide databases and evaluates unseen-attack generalization.

The uploaded `colfispoof_structure.txt` is an inventory, not image pixels.
No COLFISPOOF image archive, trained PAD model, or compatible bona fide
contactless control population is present here. PAD cannot be measured from
that inventory or from BOZORTH3 recognition scores.

Before a defensible PAD experiment:

1. Supply the actual COLFISPOOF images and licensed bona fide contactless
   images with acquisition-device metadata. Do not silently treat MOLF
   contact-based WSQ images as equivalent controls: acquisition differences
   would be confounded with the attack label.
2. Use a reproducible existing PAD implementation/model, and document its
   training sources and preprocessing. Never train or tune on the test set.
3. Preserve the authors' partitions and group repeated instrument/source
   captures together. Evaluate seen attacks and held-out attack materials
   separately; avoid frame-level random splits that leak shared instruments.
4. Freeze the decision threshold on development data. Record APCER per attack
   species and BPCER on bona fide presentations, with raw counts and explicit
   handling of acquisition/processing failures.
5. If reporting the combined matcher/PAD gate, collect their joint decisions
   on compatible presentations. Do not multiply separately measured error
   rates or add their marginal quantiles as if independence were established.

These requirements remain outstanding. No PAD acceptance/rejection metric
has been inserted into the manuscript.

## Archived Analytical Network Model

The older cryptographic-component comparison is also archived here to make
room for the matched paired-session control in the eight-page manuscript.
These are 5,000 trials per model, with cached credentials, not equivalent-
security or wire-compatible handshakes. Bio-Kyber omits the initial seed HKDF.

| Model | Mean ms | Median ms | P95 ms | Encoded bytes |
| --- | ---: | ---: | ---: | ---: |
| Bio-Kyber core | 0.07575 | 0.07248 | 0.09419 | 1207 |
| ML-KEM + ML-DSA | 0.17555 | 0.15819 | 0.28284 | 5665 |
| KEMTLS components | 0.08139 | 0.07804 | 0.09998 | 3480 |
| Classical TLS components | 0.23953 | 0.23368 | 0.26455 | 255 |

The old counted 1207-byte payload is 78.69% smaller than the signed-KEM
model and 65.32% smaller than KEMTLS components; it exceeds the classical
model. These percentages exclude certificates, record overhead, credential
provisioning, channel authentication, and the acceptance notification.

This table was moved out of the main paper to prioritize actual matcher
measurements within eight pages. It is illustrative, not measured traffic.
The original raw files remain `prism-uploads/modeled_network_latency_3.csv`
and `prism-uploads/protocol_models_3.json`.

For payload S bytes, bandwidth b bits/s, RTT d ms, and assumed RTT count r:

```text
t_model_ms = t_CPU_ms + r*d + 8*(S + 40*ceil(S/1460))/b * 1000
```

The per-segment 40-byte allowance and 1460-byte payload are modeling choices.
Channel setup, application acceptance notification, certificate traffic,
and retransmissions are excluded. r is 1 for Bio-Kyber, signed ML-KEM, and
classical TLS components, and 1.5 for KEMTLS components. Bio-Kyber uses its
successful-gate component-sum median; others use core medians.

| Model | LAN (ms) | WAN (ms) | Constrained (ms) |
| --- | ---: | ---: | ---: |
| Bio-Kyber + gate components | 42.35 | 81.74 | 840.86 |
| ML-KEM + ML-DSA | 1.20 | 42.02 | 1532.16 |
| KEMTLS components | 1.61 | 61.23 | 1476.08 |
| Classical TLS components | 1.24 | 40.33 | 647.43 |

LAN: 1 Gb/s, 1 ms RTT; WAN: 25 Mb/s, 40 ms RTT; constrained: 50 kb/s,
600 ms RTT. These are not production handshake times. The smaller counted
payload may offset the biometric gate in the constrained model only under
these assumptions; this is not an unconditional latency advantage.

## Verification

```bash
python -m unittest discover \
  -s IEEE-conference-template-062824/experiments \
  -p test_research_upgrade.py -v
```

Tests exercise threshold ties/endpoints, paired summaries, denied gates,
per-session derivation, and equal counted payloads. The test KEM is explicitly
fake and tests control flow only. It is never used to generate paper results.