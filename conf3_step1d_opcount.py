import os
import json
import numpy as np

OUT = "results_conference3"
FS, W, NFFT, AX, T = 64, 32, 256, 3, 8
NBIN = NFFT // 2 + 1


def main():
    d = np.load(os.path.join(OUT, "edge_model_fp32.npz"))
    S = d["sos"].shape[0]; H = d["weight_hh_l0"].shape[1]; n_in = d["weight_ih_l0"].shape[1]
    lg = lambda n: np.log2(n)
    ops = {}

    ops["filter"] = W * AX * S * 9

    ops["statistical features"] = AX * (25 * W + 2 * 2 * W * lg(W))

    ops["spectral features"] = AX * (2 * W + 2.5 * NFFT * lg(NFFT) + 3 * NBIN + 25 * NBIN)
    ops["standardisation"] = 3 * n_in

    mac = T * 4 * H * (n_in + H) + T * 4 * H * (H + H) + 2 * H
    ops["LSTM and head"] = 2 * mac + 2 * T * 5 * H + 2
    ops["hysteresis"] = 2
    total = sum(ops.values())
    R = {"operations_per_decision": {k: int(round(v)) for k, v in ops.items()},
         "total_operations_per_decision": int(round(total)),
         "share_%": {k: round(100 * v / total, 2) for k, v in ops.items()},
         "operations_per_second_at_2_decisions": int(round(2 * total)),
         "time_ms_at_1_operation_per_cycle": {"nRF52840 (64 MHz)": round(total / 64e6 * 1e3, 1),
                                              "ESP32-S3 (240 MHz, one core)": round(total / 240e6 * 1e3, 1)},
         "processor_load_%_at_2_decisions": {"nRF52840 (64 MHz)": round(100 * 2 * total / 64e6, 1),
                                             "ESP32-S3 (240 MHz, one core)": round(100 * 2 * total / 240e6, 2)},
         "assumptions": "approximate floating-point operations of fog_edge.py; FFT 2.5 N log2 N (real N-point); "
                        "sort 2 n log2 n; one operation per cycle; memory access and transcendental-function cost "
                        "beyond the counted operations are not modelled"}
    os.makedirs(OUT, exist_ok=True)
    json.dump(R, open(f"{OUT}/Q9_opcount.json", "w"), indent=2)
    print(json.dumps(R, indent=2))


if __name__ == "__main__":
    main()