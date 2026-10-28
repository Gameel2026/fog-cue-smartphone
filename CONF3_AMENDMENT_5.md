# Amendment 5 to CONF3_PLAN.md: compiled implementation of the controller

**Written after the plan, Amendments 1–4 and all their results were obtained, and before anything below was run.**
Earlier plans and results are unchanged. The analyses below are additional.
No patient takes part. The data are the recorded defog and FoG-STAR recordings and the replay files of E3.

## Rationale
All phone results so far were obtained with the interpreted Python module `fog_edge.py` in Termux. During live
streaming its processing took a median of 43 ms per window instead of 3.6 ms in the benchmark. This amendment
tests whether a compiled implementation of the same controller (i) reproduces the decisions of the validated
model and (ii) removes the processing share of the decision delay. The question is fixed here before any compiled
result exists on a phone. A development check of the C code against `fog_edge.py` on the PC replay recording was
made while the code was written (maximum probability difference 1.2 × 10⁻⁷, all 240 decisions equal); it is not
an outcome of this amendment.

## Implementation (version of this commit)
`fog_edge_c.c`: C99, one window at a time, same processing as `fog_edge.py`. Filter and features in double
precision, LSTM and head in single precision, compiled on the phone with Termux clang (`-O2`). Model weights are
read from binary files written by `conf5_export.py` from the unchanged `.npz` model files; int8 weights are
dequantised exactly as in `fog_edge.py`. Models: laboratory model (frozen operating point 0.65/0.40) and home model
(0.85/0.10), each fp32 and int8. Summaries: `conf5_summarize.py`. Phone commands: `conf5_phone.sh`. PC input for F1:
`conf5_pc_streams.py`. Corrections that are needed for the code to run on a device are committed and documented
before the corresponding run.

## Outcomes and pre-specified criteria
- **F1 Full-dataset fidelity (main phone).** The compiled controller runs on every recording stretch of defog
  (laboratory and home models) and FoG-STAR (laboratory model), cut exactly as in E1/E2. It is compared with the
  portable Python implementation of the same model file on the same windows. Criterion for each model, precision
  and dataset: decision agreement ≥ 99.9 % and maximum absolute probability difference ≤ 10⁻⁴.
  Descriptive: agreement and probability difference with the reference PyTorch pipeline (laboratory model).
- **C1 Replay fidelity (each phone).** Same comparison on the replay recording for the four model files.
  Criterion as F1.
- **C2 Processing time (each phone).** Replay as fast as possible, 10 passes. Criterion: 99th percentile of the
  processing time per decision ≤ 50 ms (as E3). Descriptive: median, maximum, stage times (filter, features,
  LSTM with head and hysteresis), peak memory.
- **R1 Real-time run, screen on (main phone: laboratory and home int8 models; Pixel 7: laboratory int8).** 60 min,
  one decision every 0.5 s on an absolute schedule, replay looped, device kept awake, unplugged, airplane mode, not
  used. Descriptive (as Amendment 3): decision delay median, 99th percentile and maximum on the monotonic and
  boot-time clocks, decisions within 100 ms, deadline misses over 500 ms, CPU duty cycle, suspended time.
- **R2 Real-time run, screen off with a partial wake lock (main phone, laboratory int8).** As R1 with the screen
  switched off and `termux-wake-lock` held, the configuration a cueing application would use. Descriptive, compared
  with the screen-off run of the original plan.
- **L1 Live streaming (main phone; Pixel 7 attempted).** Amendment 4 repeated with the compiled receiver
  (`fog_edge_c stream`), unchanged sender `conf3_stream_sender.py`, same replay, same 60 min, same network
  conditions. Criteria as Amendment 4: S1 100 % of streamed decisions equal the offline compiled decisions on the
  received samples with maximum probability difference ≤ 10⁻⁶; S2 ≥ 99 % of decisions returned within 500 ms
  (laptop clock); S3 99th percentile of the on-phone delay ≤ 50 ms. If the Pixel 7 still cannot reach the local
  network from Termux (Amendment 4), this is reported.

## Reporting
All results are reported whatever their direction, next to the Python results of E3 and Amendments 3–4, which they
do not replace. Further amendments may be added; each is committed before its own analysis.

**Limitations of this amendment:** the compiled code runs as a Termux command-line program, not as an Android
application; the data are replayed, not acquired from a worn sensor; the link is Wi-Fi/TCP, not Bluetooth Low
Energy; no cue is delivered to a person; energy is not measured.
