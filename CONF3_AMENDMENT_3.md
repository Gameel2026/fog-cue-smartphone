# Amendment 3 to CONF3_PLAN.md: second device, awake-device real-time run, complete per-decision operation count

**Written after E1–E3, Amendment 1 and Amendment 2 were obtained and before anything below was run.**
Earlier plans and results are unchanged. All analyses below are additional.

## A1 Second device (phone; quick and stages)
The fidelity/latency test and the 2-minute real-time run (`quick lab`) and the stage timing (`stages lab`) are repeated
without any change of code or model on a second Android smartphone of a different model. The device is reported
(platform string, CPU cores, Python and numpy versions).
- **Criteria (as E3):** 99th percentile of processing time ≤ 50 ms for the fp32 and the int8 model, and maximum
  absolute difference between phone and PC probabilities ≤ 1e-4.

## A2 Real-time run with the device kept awake (main phone; 60 min)
The 60-minute real-time run of E3 is repeated on the main phone with the screen kept on for the whole run (screen
timeout set to its maximum or "Stay awake"), the phone not used, unplugged, airplane mode on. In addition to the
monotonic clock, the boot-time clock (CLOCK_BOOTTIME, which includes device suspend) is logged at every decision,
so that the time during which the device was suspended is measured instead of noted by hand.
- **Descriptive (no criterion):** deadline misses, proportion of decisions delivered within 100 ms and within 500 ms of
  the end of their window, median and 99th percentile of decision delay, CPU duty cycle, suspended time
  (boot-time minus monotonic elapsed time). Battery change is recorded but not interpreted as the cost of the
  controller, because the screen is on. These are compared descriptively with the screen-off run of E3.

## A3 Complete per-decision operation count (analytic)
The operation count of P4 (LSTM and head only) is extended to the whole decision: causal filter (second-order
sections, three axes), statistical features, spectral features (three 256-point real FFTs and the band/moment
computations), standardisation, LSTM and head, and hysteresis. Counts are approximate floating-point operations
from the implementation in `fog_edge.py` (FFT counted as 2.5·N·log2 N for a real N-point transform; sorting for
median/mode as 2·n·log2 n comparisons) and are reported per stage together with the time this would take on the two
microcontrollers of P4 at one operation per cycle. This is an estimate; no microcontroller is run.
