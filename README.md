# FoG cue controller on a smartphone

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23262426.svg)](https://doi.org/10.5281/zenodo.23262426)

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
| `CONF3_AMENDMENT_3.md` | A1 second device, A2 60-min run with the device kept awake (boot-time clock logged), A3 complete per-decision operation count | after Amendment 2, before it was run |

The commit history of this repository records the order in which plans, code and results were added.

## Code
| File | Runs on | Purpose |
|---|---|---|
| `fog_edge.py` | phone and PC (numpy only) | Streaming causal filter (second-order sections, state carried across windows), 72 features, LSTM + head, hysteresis |
| `conf3_step1_equivalence.py` | PC | E1/E2: exports the laboratory model (fp32, int8), compares the portable pipeline with the reference pipeline, writes the replay file |
| `conf3_step1b_home_model.py` | PC | Amendment 1: home model, operating point, cross-validated performance, export |
| `conf3_step1c_precision.py` | PC | Amendment 2: P1, P3, P4 |
| `conf3_step1d_opcount.py` | PC | Amendment 3: A3 |
| `conf3_phone_benchmark.py` | phone (Termux) | E3: fidelity and latency on the phone, 60-minute real-time run, P2 stage timing, Amendment 3 (A1, A2) |

The PC scripts import modules of the companion repository (`bmel_dev`, `bmel_transfer`, `bmel_posthoc`,
`fog_pipeline`, `fog_study`, `fog_deep`, `fog_lstm_controller`, `fog_stats`, ...) and must be run from its folder,
after its own pipeline has produced the cached probabilities and models.

### Run order
```
# PC (companion repository folder)
python conf3_step1_equivalence.py
python conf3_step1b_home_model.py
python conf3_step1c_precision.py
python conf3_step1d_opcount.py
# phone: copy fog_edge.py, conf3_phone_benchmark.py, edge_*.npz and replay_*.npz to one folder
pkg install python python-numpy
python conf3_phone_benchmark.py quick lab
python conf3_phone_benchmark.py quick home
python conf3_phone_benchmark.py full lab
python conf3_phone_benchmark.py stages lab
python conf3_phone_benchmark.py screen lab              # Amendment 3, A2
# second phone (Amendment 3, A1)
python conf3_phone_benchmark.py quick lab device2
python conf3_phone_benchmark.py stages lab device2
```

## Data
Kaggle *Parkinson's Freezing of Gait Prediction* (defog, home recordings) and FoG-STAR. The datasets are not
redistributed here; `replay_*.npz` (excerpts of defog) are produced by the PC scripts.

## Devices
- Main phone: OPPO smartphone, Android 16 (ColorOS), aarch64, 8 CPU cores.
- Second phone (Amendment 3, A1): Google Pixel 7 (Google Tensor G2), Android 17, 8 CPU cores.

Both: Termux (Google Play), Python 3.13, numpy 2.4.

## Result files
`results_conference3/`: `Q1`–`Q9` (PC analyses) and the exported models `edge_*_fp32.npz`, `edge_*_int8.npz`.

`phone_results/` (`*_device2` = Pixel 7, otherwise main phone):
| File | Run |
|---|---|
| `phone_results_lab_quick.json` | first quick test, laboratory model |
| `phone_results_home_quick.json` | quick test, home model |
| `phone_results_lab_full.json`, `phone_realtime_lab_full.csv` | E3 60-min run, screen off |
| `phone_results_lab_full_run1.json` | earlier 60-min attempt during which the phone was used (not a protocol run) |
| `phone_stages_lab.json` | P2 stage timing |
| `phone_results_lab_screen.json`, `phone_realtime_lab_screen.csv` | A2 60-min run, device kept awake |
| `phone_results_lab_quick_device2.json`, `phone_realtime_lab_quick_device2.csv`, `phone_stages_lab_device2.json` | A1 second device |
| `phone_results_lab_screen_device2.json`, `phone_realtime_lab_screen_device2.csv` | additional 60-min run with the device kept awake, second device |

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
- **A1 second device (Pixel 7):** maximum difference from the PC 1.8 × 10⁻⁷, 99th percentile of processing time 3.1 ms; criteria met.
- **A2 device kept awake (main phone, 60 min):** 0 deadline misses, 99.99 % of decisions within 100 ms, 0 s suspended
  (screen-off run: 394 decisions delayed by more than 500 ms). Pixel 7, 60 min awake: 0 deadline misses, all within 100 ms.
- **A3 operation count:** about 1.12 M floating-point operations per decision, of which 96.9 % in the LSTM and head;
  roughly 3.5 % (nRF52840) and 0.9 % (ESP32-S3) processor load at two decisions per second (estimate).

### Note on the real-time runs
`perf_counter` (monotonic clock) pauses while Android suspends the device, so the 60-minute monotonic run took longer
in wall-clock time; from Amendment 3 on, the boot-time clock is logged as well. The first 60-minute attempt was disturbed
by use of the phone; the protocol run (phone not used) ran from 09:03 to 11:30 wall
time (screen on from 10:08), with a CPU duty cycle of 3.9 % and a battery drop of 6 percentage points
(46 % → 40 %), which is an upper bound for the controller. Delayed decisions were caused by operating-system power
management (app freezing and device suspend), not by computation.

## Citation
See `CITATION.cff`.

## License
MIT (see `LICENSE`).
