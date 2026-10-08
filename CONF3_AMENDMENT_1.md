# Amendment 1 to CONF3_PLAN.md: a second deployed model trained on home data

**Written after E1 and E2 of CONF3_PLAN.md were obtained for the laboratory model (all met) and before anything
described below was run.** CONF3_PLAN.md and its results are unchanged.

**Reason (disclosed):** the laboratory model, used unchanged at home, produced about 150 false alarms per hour on
defog (as reported in the BMEL study). A model of the same architecture trained on home recordings is the more
relevant model to deploy. This amendment adds it; it does not replace the laboratory model.

## Home model
- **Architecture and training:** the same 2-layer LSTM, features and settings (training length from
  frozen_config.json, which is only read), trained once on **all** 45 defog patients.
- **Operating point:** the (θon, θoff) pair of the usual grid that maximises window-level F1 of the cue-needed target
  on the **out-of-fold** defog probabilities (subject-grouped 10-fold CV, `results_bmel_posthoc/ref_lstm_defog.npy`:
  each patient's probabilities come from a model not trained on that patient), all windows pooled.
- **Expected performance on new patients:** the out-of-fold probabilities with that pair (false alarms/h, coverage,
  timely activation, specificity; circular-shift surrogate, 2,000 shifts). Reported as the cross-validated estimate,
  not as a new test.

## Outcomes and criteria (same thresholds as CONF3_PLAN.md)
- **E1-home:** portable fp32 streaming implementation vs the PyTorch home model on defog: decision agreement ≥ 99.9 %
  and false alarms/h, coverage and timely activation within ±0.5.
- **E2-home:** portable int8 vs the PyTorch home model: agreement ≥ 99.0 %, false alarms/h within ±5 %, coverage and
  timely activation within ±2 percentage points.
- **E3-home:** the phone benchmark (quick mode) with the home model: 99th percentile per-window time ≤ 50 ms and phone
  fp32 probabilities equal to the PC fp32 probabilities (max |Δp| ≤ 1e-4). The 60-minute run is done once, with the
  laboratory model as planned (same architecture, so the same computational cost).
- Fidelity of the home model is evaluated on its own training patients (defog); this tests the implementation, not
  generalisation, which is given by the cross-validated estimate above.
- All results are reported whatever their direction.
