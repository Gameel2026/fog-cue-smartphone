# Amendment 2 to CONF3_PLAN.md: numerical precision, processing stages, radio load, microcontroller budget

**Written after E1–E3 of CONF3_PLAN.md and Amendment 1 were obtained (all met) and before anything below was run.**
Earlier plans and results are unchanged. All analyses below are additional; P1 has pre-specified criteria, P2–P4 are
descriptive.

## P1 Numerical precision (PC; both models)
The LSTM and head of each deployed model (laboratory model on defog and FoG-STAR; home model on defog) are evaluated
with: fp32 (reference = portable fp32 implementation, shown in E1 to equal the PyTorch outputs), fp16 weights,
int8 / int6 / int4 weights (symmetric, one scale per output row), and **W8A8**: int8 weights and int8 inputs of every
matrix multiplication (layer inputs and hidden states, symmetric per-tensor scales fixed in advance from their known
ranges: ±10 for the clipped standardised features, ±1 for LSTM hidden states), int32-equivalent accumulation, gate
non-linearities and cell state in float32. Features are those of the reference pipeline (identical to the portable
features, E1). Hysteresis uses each model's frozen operating point.
- **Criteria:** fp16 — decision agreement ≥ 99.9 % and false alarms/h, coverage and timely activation within ±0.5;
  W8A8 — agreement ≥ 99.0 %, false alarms/h within ±5 %, coverage and timely activation within ±2 percentage points
  (same thresholds as E1 and E2). int6 and int4 are descriptive (precision–fidelity trade-off).

## P2 Processing stages (phone; quick)
On the replay recording, the time of each stage (causal filter, feature extraction, standardisation + LSTM, hysteresis)
is measured per window for the fp32 and int8 laboratory models; median and 99th percentile are reported.

## P3 Radio load (analytic, from the recordings)
Bytes per second that a wearable sensor would transmit for (a) raw lower-back acceleration (3 axes × 16 bit × 64 Hz),
(b) the 72 features every 0.5 s (fp32 and fp16), (c) one decision byte every 0.5 s, and (d) one byte only when the cue
changes state (counted from each model's output on defog), and the number of 20-byte BLE notification payloads per
second (default ATT MTU of 23 bytes).

## P4 Microcontroller budget (analytic)
Multiply–accumulate operations per decision of the LSTM and head (eight time steps recomputed every window), weight
memory in int8 and fp32, and working memory of one decision, compared with the published memory of two common
low-power microcontrollers (nRF52840: 64 MHz Cortex-M4 with FPU, 1 MB flash, 256 KB RAM; ESP32-S3: dual-core 240 MHz,
512 KB SRAM). This is an estimate; no microcontroller is run.
