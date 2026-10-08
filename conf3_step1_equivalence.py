import os
import json
import time
import pickle
import numpy as np
import pandas as pd
from scipy import signal

import bmel_dev as bd
import bmel_transfer as bt
import bmel_posthoc as bp
import kaggle_convert as kc
import fog_edge as fe
from fog_pipeline import Config, load_daphnet, contiguous_runs
from fog_study import apply_hysteresis, CUE_ON

OUT = "results_conference3"
DATASETS = ["defog", "FoG-STAR"]
QUICK = False                                 


def export(cfg):
    M = pickle.load(open(os.path.join(bt.OUT, "T_models_lab.pkl"), "rb"))
    th = json.load(open("frozen_config.json"))["operating_points"]["LSTM"]
    sd = {k: v.detach().cpu().numpy() for k, v in M["lstm"].state_dict().items()}
    sos = signal.butter(cfg.filt_order, cfg.band, btype="bandpass", fs=cfg.fs, output="sos"); zi = signal.sosfilt_zi(sos)
    base = {"mu": np.asarray(M["mu"], np.float64), "sd": np.asarray(M["sd"], np.float64), "sos": sos, "zi_unit": zi,
            "theta_on": th["theta_on"], "theta_off": th["theta_off"]}
    w = {}
    for l in range(2):
        for k in ["weight_ih", "weight_hh", "bias_ih", "bias_hh"]:
            w[f"{k}_l{l}"] = sd[f"lstm.{k}_l{l}"].astype(np.float32)
    head = [k for k in sd if k.startswith("head") and k.endswith("weight")][0]
    w["head_weight"] = sd[head].astype(np.float32); w["head_bias"] = sd[head.replace("weight", "bias")].astype(np.float32)
    np.savez(os.path.join(OUT, "edge_model_fp32.npz"), **base, **w)
    q = dict(w)
    for k in list(w):
        if "weight" in k:                                            
            s = np.maximum(np.abs(w[k]).max(1), 1e-12) / 127.0
            q[k] = np.clip(np.round(w[k] / s[:, None]), -127, 127).astype(np.int8); q[k + "__scale"] = s.astype(np.float32)
    np.savez(os.path.join(OUT, "edge_model_int8.npz"), **base, **q)
    n_par = sum(v.size for k, v in w.items())
    sizes = {"parameters": int(n_par), "weights_fp32_bytes": int(sum(v.nbytes for v in w.values())),
             "weights_int8_bytes": int(sum(v.nbytes for k, v in q.items())),
             "file_fp32_bytes": os.path.getsize(os.path.join(OUT, "edge_model_fp32.npz")),
             "file_int8_bytes": os.path.getsize(os.path.join(OUT, "edge_model_int8.npz")), "theta": (th["theta_on"], th["theta_off"])}
    return M, sizes


def stream(ds, cfg, m32, m8):
    path, merge = bt.TESTS[ds]; raw = load_daphnet(path)
    if merge:
        for _, idx in raw.groupby(["subject", "run"]).groups.items():
            raw.loc[idx, "label"] = kc.merge_gaps(raw.loc[idx, "label"].to_numpy().astype(int))
    cols = [f"tr_{a}" for a in "xyz"]; W = cfg.win; res = {"p32": [], "p8": [], "on32": [], "on8": [], "F": [], "ms": []}
    subjects = sorted(raw.subject.unique())[: (3 if QUICK else None)]
    replay = None
    for (subj, run), g in raw.groupby(["subject", "run"], sort=True):
        if subj not in subjects: continue
        acc = g[cols].to_numpy(float); lab = g["label"].to_numpy()
        for seg, (a, b) in enumerate(contiguous_runs(lab != 0)):
            if b - a < 3 * W: continue
            c32, c8 = fe.CueController(m32), fe.CueController(m8)
            for k in range((b - a) // W):
                x = acc[a + k * W: a + (k + 1) * W]
                t0 = time.perf_counter(); p, o, f = c32.step(x); res["ms"].append(1000 * (time.perf_counter() - t0))
                p8, o8, _ = c8.step(x)
                res["p32"].append(p); res["on32"].append(o); res["p8"].append(p8); res["on8"].append(o8); res["F"].append(f)
            if ds == "defog" and (replay is None or (b - a) > len(replay)):
                replay = acc[a:b - ((b - a) % W)]
    return {k: np.asarray(v) for k, v in res.items()}, replay


def metrics(on, y4, meta, cfg, onsets):
    s = bd.summarise(y4, on, meta, cfg, onsets)
    return {k: s[k] for k in ["false_alarms_per_hour", "detected_%", "timely_eligible_%", "cue_specificity"]}


def main():
    os.makedirs(OUT, exist_ok=True); pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
    cfg = Config(causal_filter=True, sensors=("tr",))
    M, sizes = export(cfg)
    m32 = fe.EdgeLSTM(os.path.join(OUT, "edge_model_fp32.npz")); m8 = fe.EdgeLSTM(os.path.join(OUT, "edge_model_int8.npz"))
    print("Model:", sizes)
    rows, per_rows = [], []
    for ds in DATASETS:
        X, meta, y4, probs = pickle.load(open(os.path.join(bp.TR, f"T_probs_{ds}.pkl"), "rb"))
        t0 = time.time(); R, replay = stream(ds, cfg, m32, m8); n = len(R["p32"])
        print(f"\n{ds}: streamed {n} windows in {(time.time() - t0) / 60:.1f} min (reference: {len(meta)} windows)")
        if QUICK:
            keep = meta.subject.isin(sorted(meta.subject.unique())[:3]).to_numpy()
            X, meta, y4, probs = X[keep], meta[keep].reset_index(drop=True), y4[keep], {k: v[keep] for k, v in probs.items()}
        assert n == len(meta), "window count differs from the reference pipeline"
        Fref = np.nan_to_num(X.to_numpy(np.float64)); p_ref = probs["LSTM"]
        on_ref = apply_hysteresis(p_ref, meta, np.ones(n, bool), *sizes["theta"])
        onsets = bd.eligible_onsets(meta["y2"].to_numpy(), meta, cfg)
        feat_rel = np.max(np.abs(R["F"] - Fref) / (np.abs(Fref) + 1e-9))
        ref_m = metrics(on_ref, y4, meta, cfg, onsets)
        for name, p, o in [("fp32 (portable)", R["p32"], R["on32"].astype(bool)), ("int8 (portable)", R["p8"], R["on8"].astype(bool))]:
            mm = metrics(o, y4, meta, cfg, onsets)
            rows.append({"dataset": ds, "implementation": name, "windows": n, "max_feature_rel_diff": feat_rel,
                         "max_abs_prob_diff": float(np.max(np.abs(p - p_ref))), "mean_abs_prob_diff": float(np.mean(np.abs(p - p_ref))),
                         "decision_agreement_%": 100 * float(np.mean(o == on_ref)),
                         **{f"ref_{k}": v for k, v in ref_m.items()}, **{k: v for k, v in mm.items()},
                         **{f"diff_{k}": mm[k] - ref_m[k] for k in mm}})
            for s in sorted(meta.subject.unique()):
                msk = meta.subject.eq(s).to_numpy()
                per_rows.append({"dataset": ds, "implementation": name, "subject": s, "agreement_%": 100 * float(np.mean(o[msk] == on_ref[msk]))})
        if ds == "defog":
            ms = R["ms"]
            pc = {"pc_ms_median": float(np.median(ms)), "pc_ms_p99": float(np.percentile(ms, 99)), "pc_ms_max": float(ms.max())}
            print("PC latency per window (fp32, filter+features+LSTM+hysteresis):", {k: round(v, 3) for k, v in pc.items()})
            c = fe.CueController(m32); ref_out = [c.step(replay[k * 32:(k + 1) * 32])[0] for k in range(len(replay) // 32)]
            np.savez(os.path.join(OUT, "replay_defog.npz"), raw=replay.astype(np.float64), p_pc=np.asarray(ref_out))
            print(f"replay file: {len(replay) / 64 / 60:.1f} min of defog lower-back signal")
    E = pd.DataFrame(rows); P = pd.DataFrame(per_rows)
    crit = []
    for _, r in E.iterrows():
        if r.implementation.startswith("fp32"):
            ok = r["decision_agreement_%"] >= 99.9 and abs(r.diff_false_alarms_per_hour) <= 0.5 and abs(r["diff_detected_%"]) <= 0.5 \
                and abs(r["diff_timely_eligible_%"]) <= 0.5
            crit.append(("E1", r.dataset, ok))
        else:
            ok = r["decision_agreement_%"] >= 99.0 and abs(r.diff_false_alarms_per_hour) <= 0.05 * r.ref_false_alarms_per_hour \
                and abs(r["diff_detected_%"]) <= 2 and abs(r["diff_timely_eligible_%"]) <= 2
            crit.append(("E2", r.dataset, ok))
    E["criterion"] = [c[0] for c in crit]; E["met"] = [c[2] for c in crit]
    E.round(6).to_csv(f"{OUT}/Q1_equivalence.csv", index=False); P.round(4).to_csv(f"{OUT}/Q2_agreement_per_patient.csv", index=False)
    json.dump({**sizes, **pc}, open(f"{OUT}/Q3_model_and_pc_latency.json", "w"), indent=2, default=str)
    print("\nQ1 equivalence with the reference pipeline\n", E.round(4).to_string(index=False))
    print("\nlowest per-patient agreement:", P.groupby(["dataset", "implementation"])["agreement_%"].min().round(3).to_dict())
    for c, ds, ok in crit:
        print(f"PRE-SPECIFIED {c} ({ds}): {'MET' if ok else 'NOT met'}")
    print(f"\nCopy to the phone: fog_edge.py, conf3_phone_benchmark.py, {OUT}/edge_model_fp32.npz, {OUT}/edge_model_int8.npz, {OUT}/replay_defog.npz")


if __name__ == "__main__":
    main()