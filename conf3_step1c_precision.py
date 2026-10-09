import os
import json
import pickle
import numpy as np
import pandas as pd

import bmel_dev as bd
import bmel_posthoc as bp
from fog_pipeline import Config, contiguous_runs
from fog_study import apply_hysteresis, sequences, CUE_ON
from fog_deep import past_index, gather

OUT = "results_conference3"
MODELS = {"lab": ("edge_model_fp32.npz", ["defog", "FoG-STAR"]), "home": ("edge_home_fp32.npz", ["defog"])}
VARIANTS = ["fp32", "fp16 weights", "int8 weights", "int6 weights", "int4 weights", "W8A8"]
CHUNK = 20000


def load(path):
    d = dict(np.load(path)); return d


def qrow(W, bits):
    lv = 2 ** (bits - 1) - 1
    s = np.maximum(np.abs(W).max(1), 1e-12) / lv
    return np.clip(np.round(W / s[:, None]), -lv, lv) * s[:, None], W.astype(np.float32).nbytes * bits / 32 + s.astype(np.float32).nbytes


def weights(d, variant):
    keys = [k for k in d if k.startswith("weight_") or k == "head_weight"]
    W = {k: d[k].astype(np.float64) for k in keys}; nbytes = 0.0
    for k in keys:
        if variant == "fp16 weights":
            W[k] = d[k].astype(np.float16).astype(np.float64); nbytes += d[k].size * 2
        elif variant in ("int8 weights", "W8A8"):
            W[k], b = qrow(d[k].astype(np.float64), 8); nbytes += b
        elif variant == "int6 weights":
            W[k], b = qrow(d[k].astype(np.float64), 6); nbytes += b
        elif variant == "int4 weights":
            W[k], b = qrow(d[k].astype(np.float64), 4); nbytes += b
        else:
            nbytes += d[k].size * 4
    for k in d:
        if k.startswith("bias") or k == "head_bias": nbytes += d[k].size * 4
    return W, nbytes


def qact(x, rng):                                   
    s = rng / 127.0
    return np.clip(np.round(x / s), -127, 127) * s


def lstm_batch(Z, d, W, act8):
    sig = lambda v: 1 / (1 + np.exp(-v))
    x = Z.astype(np.float64); H = d["weight_hh_l0"].shape[1]
    for l in range(2):
        Wi, Wh = W[f"weight_ih_l{l}"], W[f"weight_hh_l{l}"]; b = (d[f"bias_ih_l{l}"] + d[f"bias_hh_l{l}"]).astype(np.float64)
        xin = qact(x, 10.0 if l == 0 else 1.0) if act8 else x
        h = np.zeros((len(x), H)); c = np.zeros((len(x), H)); hs = []
        for t in range(x.shape[1]):
            hin = qact(h, 1.0) if act8 else h
            g = xin[:, t] @ Wi.T + hin @ Wh.T + b
            i, f, gg, o = np.split(g, 4, axis=1)
            c = sig(f) * c + sig(i) * np.tanh(gg); h = sig(o) * np.tanh(c); hs.append(h)
        x = np.stack(hs, 1)
    hl = qact(x[:, -1], 1.0) if act8 else x[:, -1]
    z = hl @ W["head_weight"].T + d["head_bias"]; z = z - z.max(1, keepdims=True); e = np.exp(z)
    return e[:, 1] / e.sum(1)


def probs(Z, d, variant):
    W, nbytes = weights(d, variant); act8 = variant == "W8A8"
    p = np.concatenate([lstm_batch(Z[k:k + CHUNK], d, W, act8) for k in range(0, len(Z), CHUNK)])
    return p, nbytes


def main():
    os.makedirs(OUT, exist_ok=True); pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
    cfg = Config(causal_filter=True, sensors=("tr",)); rows, radio = [], []
    for mname, (fname, datasets) in MODELS.items():
        d = load(os.path.join(OUT, fname)); th = (float(d["theta_on"]), float(d["theta_off"]))
        for ds in datasets:
            X, meta, y4, pr = pickle.load(open(os.path.join(bp.TR, f"T_probs_{ds}.pkl"), "rb"))
            F = np.nan_to_num(X.to_numpy(np.float64)); allm = np.ones(len(y4), bool)
            Z = gather(np.clip((F - d["mu"]) / d["sd"], -10, 10).astype(np.float32), past_index(meta, 8))
            onsets = bd.eligible_onsets(meta["y2"].to_numpy(), meta, cfg)
            ref_p, _ = probs(Z, d, "fp32"); ref_on = apply_hysteresis(ref_p, meta, allm, *th)
            if mname == "lab":
                print(f"  sanity ({ds}): max |numpy fp32 - PyTorch| = {np.max(np.abs(ref_p - pr['LSTM'])):.2e}")
            rs = bd.summarise(y4, ref_on, meta, cfg, onsets)
            for v in VARIANTS:
                p, nb = (ref_p, None) if v == "fp32" else probs(Z, d, v)
                if v == "fp32": _, nb = weights(d, "fp32")
                on = ref_on if v == "fp32" else apply_hysteresis(p, meta, allm, *th)
                s = bd.summarise(y4, on, meta, cfg, onsets)
                r = {"model": mname, "dataset": ds, "variant": v, "parameter_bytes": int(nb),
                     "max_abs_prob_diff": float(np.max(np.abs(p - ref_p))), "decision_agreement_%": 100 * float(np.mean(on == ref_on)),
                     "false_alarms_per_hour": s["false_alarms_per_hour"], "coverage_%": s["detected_%"],
                     "timely_%": s["timely_eligible_%"], "specificity": s["cue_specificity"],
                     "d_fa": s["false_alarms_per_hour"] - rs["false_alarms_per_hour"], "d_cov": s["detected_%"] - rs["detected_%"],
                     "d_timely": s["timely_eligible_%"] - rs["timely_eligible_%"]}
                if v == "fp16 weights":
                    r["criterion"] = "P1 fp16"; r["met"] = bool(r["decision_agreement_%"] >= 99.9 and abs(r["d_fa"]) <= 0.5
                                                               and abs(r["d_cov"]) <= 0.5 and abs(r["d_timely"]) <= 0.5)
                elif v == "W8A8":
                    r["criterion"] = "P1 W8A8"; r["met"] = bool(r["decision_agreement_%"] >= 99.0 and abs(r["d_fa"]) <= 0.05 * rs["false_alarms_per_hour"]
                                                               and abs(r["d_cov"]) <= 2 and abs(r["d_timely"]) <= 2)
                rows.append(r)
            print(f"{mname} / {ds}: done")
            if ds == "defog":                                          
                hours = len(meta) * cfg.win_sec / 3600; changes = 0
                for _, idx in sequences(meta):
                    o = ref_on[idx]; changes += int(np.sum(o[1:] != o[:-1])) + int(o[0])
                radio.append({"model": mname, "state_changes_per_hour": changes / hours})
    P = pd.DataFrame(rows); P.round(6).to_csv(f"{OUT}/Q6_precision.csv", index=False)

    ch = {r["model"]: r["state_changes_per_hour"] for r in radio}
    R = [{"stream": "raw acceleration (3 x 16 bit x 64 Hz), sent every 0.5 s", "bytes_per_message": 3 * 2 * 32, "messages_per_s": 2.0},
         {"stream": "72 features (fp32), every 0.5 s", "bytes_per_message": 72 * 4, "messages_per_s": 2.0},
         {"stream": "72 features (fp16), every 0.5 s", "bytes_per_message": 72 * 2, "messages_per_s": 2.0},
         {"stream": "1 decision byte, every 0.5 s", "bytes_per_message": 1, "messages_per_s": 2.0}]
    for m, c in ch.items():
        R.append({"stream": f"1 byte per cue state change ({m} model, defog)", "bytes_per_message": 1, "messages_per_s": c / 3600})
    R = pd.DataFrame(R); R["bytes_per_s"] = R.bytes_per_message * R.messages_per_s
    R["ble_20B_payloads_per_s"] = np.ceil(R.bytes_per_message / 20) * R.messages_per_s
    R["reduction_vs_raw_%"] = 100 * (1 - R.bytes_per_s / R.bytes_per_s.iloc[0])
    R.round(4).to_csv(f"{OUT}/Q7_radio.csv", index=False)

    d = load(os.path.join(OUT, "edge_model_fp32.npz")); H = d["weight_hh_l0"].shape[1]; n_in = d["weight_ih_l0"].shape[1]; T = 8
    macs = T * 4 * H * (n_in + H) + T * 4 * H * (H + H) + 2 * H
    n_par = sum(d[k].size for k in d if k.startswith(("weight_", "bias_", "head_")))
    work = (T * n_in + 2 * T * H + 2 * H + 4 * H) * 4
    M = {"lstm_head_MACs_per_decision": int(macs), "MACs_per_second_at_2_decisions": int(2 * macs), "parameters": int(n_par),
         "weights_int8_KB": round(float(P[(P.model == "lab") & (P.variant == "int8 weights")].parameter_bytes.iloc[0]) / 1024, 1),
         "weights_fp32_KB": round(n_par * 4 / 1024, 1), "working_memory_KB_fp32": round(work / 1024, 1),
         "time_ms_at_1_MAC_per_cycle": {"nRF52840 (64 MHz)": round(macs / 64e6 * 1e3, 1), "ESP32-S3 (240 MHz, one core)": round(macs / 240e6 * 1e3, 1)},
         "memory_reference": {"nRF52840": "1 MB flash, 256 KB RAM", "ESP32-S3": "512 KB SRAM"},
         "note": "analytic estimate; feature extraction (three 256-point FFTs and statistics per window) not included"}
    json.dump(M, open(f"{OUT}/Q8_mcu_budget.json", "w"), indent=2)

    print("\nQ6 precision (agreement with the fp32 decisions)\n",
          P[["model", "dataset", "variant", "parameter_bytes", "max_abs_prob_diff", "decision_agreement_%", "false_alarms_per_hour",
             "coverage_%", "timely_%", "d_fa", "d_cov", "d_timely"]].round(4).to_string(index=False))
    for _, r in P[P.criterion.notna()].iterrows():
        print(f"PRE-SPECIFIED {r.criterion} ({r.model}, {r.dataset}): {'MET' if r.met else 'NOT met'}")
    print("\nQ7 radio load\n", R.round(3).to_string(index=False))
    print("\nQ8 microcontroller budget\n", json.dumps(M, indent=2))


if __name__ == "__main__":
    main()
