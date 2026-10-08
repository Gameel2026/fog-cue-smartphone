# Analysis plan: real-time smartphone deployment of a lower-back FoG cue controller (conference paper 3)

**Written before any part of this study was run** (no export, equivalence test or phone measurement existed).
Commit this file before running `conf3_step1_equivalence.py` or `conf3_phone_benchmark.py`.

**Prior work disclosed:** the controller is the frozen laboratory LSTM cue controller of our BMEL study (trained on
tdcsfog; operating point θon/θoff from its frozen configuration), used unchanged. Its clinical performance is not
re-tested here; this study tests whether it can be **implemented** on a smartphone without changing its decisions,
and at what computational cost. Files of the BMEL study are only read, never modified.

## Implementations
- **Reference:** the original PC pipeline (scipy filter, pandas/scipy features, PyTorch LSTM), whose outputs were saved
  by the BMEL transfer study.
- **Portable fp32:** `fog_edge.py`, numpy only, streaming: sample-by-sample causal Butterworth filter (second-order
  sections, state carried across windows), the same 72 features re-implemented in numpy, LSTM forward pass in numpy,
  hysteresis; one decision every 0.5 s.
- **Portable int8:** the same, with LSTM and head weights stored as int8 with one float32 scale per output row
  (symmetric weight-only quantisation), dequantised at load time.

## Data
defog (primary; 45 patients, 148,871 windows, home recordings) and FoG-STAR (22 patients, 10,071 windows), lower back,
streamed recording by recording exactly as the reference pipeline cuts them.

## Device
OPPO Reno10 Pro+ 5G (CPH2521), Qualcomm Snapdragon 8+ Gen 1, 12 GB RAM, 4,700 mAh battery, Android 16;
Python and numpy in Termux; CPU only.

## Outcomes and criteria fixed in advance
- **E1 fidelity (fp32 vs reference), each dataset:** cue decision agreement ≥ 99.9 % of windows **and** pooled false
  alarms/h, episode coverage and timely activation each within ±0.5 (per hour / percentage points) of the reference.
- **E2 quantisation (int8 vs reference), each dataset:** decision agreement ≥ 99.0 % **and** false alarms/h within ±5 %
  of the reference value **and** coverage and timely activation within ±2 percentage points.
- **E3 real time (phone):** 99th percentile of the end-to-end processing time per window (filter + features + LSTM +
  hysteresis) ≤ 50 ms (10 % of the 500 ms window) for fp32 and int8, **and** phone fp32 probabilities equal to the
  PC fp32 probabilities on the replay recording (max absolute difference ≤ 1e-4).
- **Secondary (descriptive):** maximum feature and probability differences; per-patient agreement (minimum);
  model size (parameters, bytes fp32 vs int8); peak memory; median/max latency; deadline misses, CPU duty cycle and
  battery drop during a 60-minute real-time run (one decision every 0.5 s, screen off, unplugged); radio data rate
  needed if the decision is computed on the phone instead of streaming raw data (raw 3-axis 16-bit at 64 Hz =
  384 B/s vs one decision byte per 0.5 s = 2 B/s), computed analytically.
- All results are reported whatever their direction.
