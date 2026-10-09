# FoG cue controller on a smartphone

Pre-specified analysis plan, portable implementation and results for running a lightweight LSTM cue controller for
freezing of gait (FoG) in Parkinson's disease on an unmodified Android smartphone (single lower-back accelerometer).

The study examines **implementation fidelity and computational cost** of the deployed controller, not its clinical
efficacy. The model and its training/evaluation are described in the companion repository
[FoG-lower-back-cue-controller-transfer](https://github.com/Gameel2026/FoG-lower-back-cue-controller-transfer)
(DOI [10.5281/zenodo.23184573](https://doi.org/10.5281/zenodo.23184573)).

## Pre-registration
| File | Content | Written |
|---|---|---|
| `CONF3_PLAN.md` | Primary plan: criteria E1 (portable fp32 = reference), E2 (int8), E3 (phone latency and fidelity) | before any part of the study was run |
| `CONF3_AMENDMENT_1.md` | Home-trained model, operating point by maximum F1 on out-of-fold probabilities | after E1–E3, before it was run |
| `CONF3_AMENDMENT_2.md` | P1 numerical precision (fp16, int8/int6/int4, W8A8), P2 stage timing, P3 radio load, P4 microcontroller budget | after Amendment 1, before it was run |

The commit history of this repository records the order in which plans, code and results were added.

## Code
| File | Runs on | Purpose |
|---|---|---|
| `fog_edge.py` | phone and PC (numpy only) | Streaming causal filter (second-order sections, state carried across windows), 72 features, LSTM + head, hysteresis |
| `conf3_step1_equivalence.py` | PC | E1/E2: exports the laboratory model (fp32, int8), compares the portable pipeline with the reference pipeline, writes the replay file |
| `conf3_step1b_home_model.py` | PC | Amendment 1: home model, operating point, cross-validated performance, export |
| `conf3_step1c_precision.py` | PC | Amendment 2: P1, P3, P4 |
| `conf3_phone_benchmark.py` | phone (Termux) | E3: fidelity and latency on the phone, 60-minute real-time run, P2 stage timing |

The PC scripts import modules of the companion repository (`bmel_dev`, `bmel_transfer`, `bmel_posthoc`,
`fog_pipeline`, `fog_study`, `fog_deep`, `fog_lstm_controller`, `fog_stats`, ...) and must be run from its folder,
after its own pipeline has produced the cached probabilities and models.

### Run order
```
# PC (companion repository folder)
python conf3_step1_equivalence.py
python conf3_step1b_home_model.py
python conf3_step1c_precision.py
# phone: copy fog_edge.py, conf3_phone_benchmark.py, edge_*.npz and replay_*.npz to one folder
pkg install python python-numpy
python conf3_phone_benchmark.py quick lab
python conf3_phone_benchmark.py quick home
python conf3_phone_benchmark.py full lab
python conf3_phone_benchmark.py stages lab
```

## Data
Kaggle *Parkinson's Freezing of Gait Prediction* (defog, home recordings) and FoG-STAR. The datasets are not
redistributed here; `replay_*.npz` (excerpts of defog) are produced by the PC scripts.

## Device
Android 16 smartphone, aarch64, 8 CPU cores; Termux (Google Play), Python 3.13, numpy 2.4.

## Results (summary)
- **E1–E3 and Amendment 1:** all pre-specified criteria met (`results_conference3/Q1`–`Q5`).
  On the phone, the maximum difference from the PC probabilities was 1.8 × 10⁻⁷; processing time per 0.5-s decision
  had a median of 3.6 ms and a 99th percentile below 7 ms (fp32 and int8); peak memory about 30 MB;
  model files 280 KB (fp32) and 83 KB (int8).
- **Home model (cross-validated, defog):** 26.2 false alarms/h, 57.1 % episode coverage.
- **P1 precision:** fp16 and W8A8 met their criteria for both models and both datasets (decision agreement ≥ 99.3 %);
  int4 weights reduced agreement to about 98 %.
- **P2 stages (phone, median):** filter 1.73 ms, features 1.03 ms, LSTM 0.78 ms, hysteresis < 0.001 ms.
- **P3 radio load:** one byte per cue state change reduces the transmitted data by more than 99.9 % relative to raw
  acceleration.
- **P4 microcontroller:** 0.54 M multiply–accumulate operations per decision; 74 KB of int8 weights.

### Note on the real-time runs
`perf_counter` (monotonic clock) pauses while Android suspends the device, so the 60-minute monotonic run took longer
in wall-clock time. Run 1 was interrupted by use of the phone; run 2 (phone not used) ran from 09:03 to 11:30 wall
time (screen on from 10:08), with a CPU duty cycle of 3.9 % and a battery drop of 6 percentage points
(46 % → 40 %), which is an upper bound for the controller. Delayed decisions were caused by operating-system power
management (app freezing and device suspend), not by computation.

## Citation
See `CITATION.cff`.

## License
MIT (see `LICENSE`).
