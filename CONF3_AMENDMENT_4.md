# Amendment 4 to CONF3_PLAN.md: live wireless streaming of recorded data to the phone

**Written after E1–E3 and Amendments 1–3 were obtained and before anything below was run.**
Earlier plans and results are unchanged. The analysis below is additional. No patient takes part: the data are the
recorded defog replay used in E3 (`replay_defog.npz`).

## L1 Live streaming (laptop → phone over Wi-Fi; 60 min)
A laptop acts as the sensor. It sends the lower-back acceleration of the replay recording in real time at 64 Hz,
four samples per packet (one packet every 62.5 ms), over TCP (TCP_NODELAY) on a local Wi-Fi network, looping the
recording for 60 min (7,200 windows). Samples are sent as stored (float64) so that decisions can be compared exactly.
The phone (main phone of E3, unchanged code and `fog_edge.py`, laptop-to-phone connection only) receives the
stream, runs the deployed controller (int8 laboratory model, frozen operating point) once each 32-sample window is
complete, and sends the decision (cue on/off and probability) back to the laptop. The controller runs continuously over
the whole stream (no reset when the recording loops). Conditions: phone unplugged, kept awake (screen "Stay awake"
and `termux-wake-lock`), not used, Wi-Fi on and mobile data off; laptop on mains power; both on the same network.
The run is then repeated, without any change of code, model, network or conditions, on the second phone of
Amendment 3 (A1, Pixel 7). Criteria apply to each phone separately.
Scripts: `conf3_stream_sender.py` (laptop) and `conf3_stream_receiver.py` (phone). A connection check of up to
2 min may precede each run; it is reported but not analysed.

**Outcomes and pre-specified criteria**
- **S1 Streaming fidelity:** after the stream ends, the phone re-runs the same controller offline on exactly the
  samples it received. Criterion: 100 % of streamed decisions equal the offline decisions and the maximum absolute
  difference in probability ≤ 1e-6.
- **S2 End-to-end timing (laptop clock):** for each window, the time from the scheduled end of its last sample to the
  arrival of the decision back at the laptop. This round trip includes the return link, which a cue delivered on the
  phone would not need, and is therefore an upper bound. Criterion: ≥ 99 % of decisions within 500 ms (one window
  step). Median, 99th percentile, maximum and the proportion within 100 ms are reported.
- **S3 On-phone delay (phone clock):** time from the arrival of the last packet of a window to the decision.
  Criterion: 99th percentile ≤ 50 ms (as E3).
- **Descriptive (no criterion):** decisions missing, sequence errors, packet inter-arrival gaps (median, 99th
  percentile, maximum, number > 100 ms and > 500 ms), sender scheduling lateness, CPU duty cycle and suspended time on
  the phone, and, for the first pass of the recording, decision agreement and maximum probability difference with the
  PC reference probabilities (`p_pc`).

**Limitations stated in advance:** the link is Wi-Fi/TCP, not Bluetooth Low Energy; the data are replayed, not
acquired from a worn sensor; no cue is delivered to a person.

All results are reported whatever their direction.
