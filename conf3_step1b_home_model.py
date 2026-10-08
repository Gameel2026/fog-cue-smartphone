import os
import json
import pickle
import numpy as np
import pandas as pd
from scipy import signal

import bmel_dev as bd
import bmel_posthoc as bp
import bmel_sensitivity as bs
import fog_lstm_controller as flc
import fog_edge as fe
import conf3_step1_equivalence as c3
from fog_pipeline import Config
from fog_study import apply_hysteresis, CUE_ON
from fog_deep import past_index

OUT = c3.OUT
N_SHIFTS = 2000


def f1(o, t):
    tp = np.sum(o & t); return 2 * tp / max(2 * tp + np.sum(o & ~t) + np.sum(~o & t), 1)


def export_home(model, mu, sd, theta, cfg):
    st = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    sos = signal.butter(cfg.filt_order, cfg.band, btype="bandpass", fs=cfg.fs, output="sos")
    base = {"mu": mu, "sd": sd, "sos": sos, "zi_unit": signal.sosfilt_zi(sos), "theta_on": theta[0], "theta_off": theta[1]}
    w = {f"{k}_l{l}": st[f"lstm.{k}_l{l}"].astype(np.float32) for l in range(2) for k in ["weight_ih", "weight_hh", "bias_ih", "bias_hh"]}
    head = [k for k in st if k.startswith("head") and k.endswith("weight")][0]
    w["head_weight"] = st[head].astype(np.float32); w["head_bias"] = st[head.replace("weight", "bias")].astype(np.float32)
    np.savez(os.path.join(OUT, "edge_home_fp32.npz"), **base, **w)
    q = dict(w)
    for k in list(w):
        if "weight" in k:
            s = np.maximum(np.abs(w[k]).max(1), 1e-12) / 127.0
            q[k] = np.clip(np.round(w[k] / s[:, None]), -127, 127).astype(np.int8); q[k + "__scale"] = s.astype(np.float32)
    np.savez(os.path.join(OUT, "edge_home_int8.npz"), **base, **q)


def main():
    os.makedirs(OUT, exist_ok=True); pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
    cfg = Config(causal_filter=True, sensors=("tr",)); epochs = json.load(open("frozen_config.json"))["lstm_final_epochs"]
    X, meta, y4, _ = pickle.load(open(os.path.join(bp.TR, "T_probs_defog.pkl"), "rb"))
    y_on = np.isin(y4, CUE_ON).astype(int); t = y_on.astype(bool); N = len(y4); allm = np.ones(N, bool)
    onsets = bd.eligible_onsets(meta["y2"].to_numpy(), meta, cfg)
    # 1-2 operating point and cross-validated expected performance
    oof = np.load(os.path.join(bp.OUT, "ref_lstm_defog.npy")); best = (None, -1.0)
    for a, b in bd.GRID:
        th = (float(a), float(b)); v = f1(apply_hysteresis(oof, meta, allm, *th), t)
        if v > best[1]: best = (th, v)
    theta = best[0]; on_cv = apply_hysteresis(oof, meta, allm, *theta)
    s = bd.summarise(y4, on_cv, meta, cfg, onsets)
    sg = bs.surrogate_segments(on_cv, meta, cfg, onsets, bs.recording_segments(meta), N_SHIFTS)
    cv = {"theta": theta, "window_f1": best[1], "false_alarms_per_hour": s["false_alarms_per_hour"], "coverage_%": s["detected_%"],
          "coverage_surrogate": sg["detection_surrogate_mean"], "p_coverage": sg["detection_p"],
          "timely_%": s["timely_eligible_%"], "timely_surrogate": sg["timely_surrogate_mean"], "p_timely": sg["timely_p"],
          "cue_specificity": s["cue_specificity"], "cue_sensitivity": s["cue_sensitivity"]}
    print("Home model operating point and cross-validated expected performance:\n", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in cv.items()})
    # 3 final model on all defog patients
    F = np.nan_to_num(X.to_numpy(np.float64)); P8 = past_index(meta, 8); idx = np.arange(N)
    Z = flc.standardise(F, idx, P8); model, _, _ = flc.train(Z, y_on, idx, epochs, 42); p_ref = flc.prob(model, Z)
    mu = F.mean(0); sd = F.std(0); sd = np.where(sd > 0, sd, 1.0)
    export_home(model, mu, sd, theta, cfg)
    on_ref = apply_hysteresis(p_ref, meta, allm, *theta); ref_m = c3.metrics(on_ref, y4, meta, cfg, onsets)
    # 4 streaming portable implementation vs PyTorch home model
    m32 = fe.EdgeLSTM(os.path.join(OUT, "edge_home_fp32.npz")); m8 = fe.EdgeLSTM(os.path.join(OUT, "edge_home_int8.npz"))
    R, replay = c3.stream("defog", cfg, m32, m8); assert len(R["p32"]) == N
    rows = []
    for name, p, o in [("fp32 (portable)", R["p32"], R["on32"].astype(bool)), ("int8 (portable)", R["p8"], R["on8"].astype(bool))]:
        mm = c3.metrics(o, y4, meta, cfg, onsets); agree = 100 * float(np.mean(o == on_ref))
        per = [100 * float(np.mean(o[meta.subject.eq(sj).to_numpy()] == on_ref[meta.subject.eq(sj).to_numpy()])) for sj in sorted(meta.subject.unique())]
        if name.startswith("fp32"):
            ok = agree >= 99.9 and abs(mm["false_alarms_per_hour"] - ref_m["false_alarms_per_hour"]) <= 0.5 \
                and abs(mm["detected_%"] - ref_m["detected_%"]) <= 0.5 and abs(mm["timely_eligible_%"] - ref_m["timely_eligible_%"]) <= 0.5
        else:
            ok = agree >= 99.0 and abs(mm["false_alarms_per_hour"] - ref_m["false_alarms_per_hour"]) <= 0.05 * ref_m["false_alarms_per_hour"] \
                and abs(mm["detected_%"] - ref_m["detected_%"]) <= 2 and abs(mm["timely_eligible_%"] - ref_m["timely_eligible_%"]) <= 2
        rows.append({"implementation": name, "criterion": "E1-home" if name.startswith("fp32") else "E2-home",
                     "max_abs_prob_diff": float(np.max(np.abs(p - p_ref))), "mean_abs_prob_diff": float(np.mean(np.abs(p - p_ref))),
                     "decision_agreement_%": agree, "min_patient_agreement_%": min(per),
                     **{f"ref_{k}": v for k, v in ref_m.items()}, **mm, "met": ok})
    E = pd.DataFrame(rows); E.round(6).to_csv(f"{OUT}/Q4_home_equivalence.csv", index=False)
    json.dump(cv, open(f"{OUT}/Q5_home_cv_performance.json", "w"), indent=2, default=str)
    # 5 replay for the phone with home-model reference probabilities
    rp = np.load(os.path.join(OUT, "replay_defog.npz"))["raw"]; c = fe.CueController(m32)
    np.savez(os.path.join(OUT, "replay_home.npz"), raw=rp, p_pc=np.asarray([c.step(rp[k * 32:(k + 1) * 32])[0] for k in range(len(rp) // 32)]))
    print("\nQ4 home model: portable implementation vs PyTorch\n", E.round(4).to_string(index=False))
    for _, r in E.iterrows():
        print(f"PRE-SPECIFIED {r.criterion}: {'MET' if r.met else 'NOT met'}")
    print(f"\nCopy to the phone (same folder as before): {OUT}/edge_home_fp32.npz, {OUT}/edge_home_int8.npz, {OUT}/replay_home.npz")


if __name__ == "__main__":
    main()
