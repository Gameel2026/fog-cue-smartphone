# Amendment 4 results: live wireless streaming

## Main phone (OPPO Reno10 Pro+, Android 16), 60 min, int8 laboratory model
| Outcome | Result | Criterion |
|---|---|---|
| S1 streamed vs offline decisions | 100 % identical (7,200 windows), max abs prob. diff. 0 | met |
| S2 decisions returned within 500 ms (round trip, laptop clock) | 99.79 % (15 later, after 4 link interruptions of 2.0–3.7 s); median 62.6 ms, p99 125 ms, max 3,250 ms; within 100 ms 97.8 % | met |
| S3 on-phone delay p99 | 51.7 ms (median 43.8, max 66.8) | not met (≤ 50 ms) |
| Samples / packets / sequence errors / missing decisions | 230,400 / 57,600 / 0 / 0 | descriptive |
| Agreement with PC reference, first pass (547 windows) | 99.63 % | descriptive |
| CPU duty cycle / suspended time | 10.6 % / 0 s | descriptive |

Connection check (2 min) before the run: S1 met, S2 met (100 % within 500 ms), S3 not met (p99 53.5 ms).
Files: `stream_phone_int8*.json/.csv`, `stream_laptop*.json/.csv` (`_check` = 2-min connection check).

## Second phone (Pixel 7, Android 17)
Not run. Android 17 blocked the Termux app from the local network (ICMP "Operation not permitted", TCP timeouts in
both connection directions) while internet access from the same app worked; no change of code, model or network
could be made without departing from the registered conditions.
