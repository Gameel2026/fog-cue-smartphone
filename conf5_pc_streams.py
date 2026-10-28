import os
import pickle
import struct
import numpy as np

import bmel_transfer as bt
import bmel_posthoc as bp
import kaggle_convert as kc
import fog_edge as fe
from fog_pipeline import Config, load_daphnet, contiguous_runs

SRC = "results_conference3"
OUT = "results_conference5"
DATASETS = ["defog", "FoG-STAR"]


def stretches(ds, cfg):
    path, merge = bt.TESTS[ds]
    raw = load_daphnet(path)
    if merge:
        for _, idx in raw.groupby(["subject", "run"]).groups.items():
            raw.loc[idx, "label"] = kc.merge_gaps(raw.loc[idx, "label"].to_numpy().astype(int))
    cols = [f"tr_{a}" for a in "xyz"]
    W = cfg.win
    for (subj, run), g in raw.groupby(["subject", "run"], sort=True):
        acc = g[cols].to_numpy(float)
        lab = g["label"].to_numpy()
        for a, b in contiguous_runs(lab != 0):
            if b - a < 3 * W:
                continue
            n = (b - a) // W
            yield acc[a:a + n * W]


def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = Config(causal_filter=True, sensors=("tr",))
    models = {"lab": (os.path.join(SRC, "edge_model_fp32.npz"), os.path.join(SRC, "edge_model_int8.npz")),
              "home": (os.path.join(SRC, "edge_home_fp32.npz"), os.path.join(SRC, "edge_home_int8.npz"))}
    for ds in DATASETS:
        use = ["lab", "home"] if ds == "defog" else ["lab"]
        M = {f"{k}_{v}": fe.EdgeLSTM(models[k][i]) for k in use for i, v in enumerate(["fp32", "int8"])}
        P = {k: [] for k in M}
        S = list(stretches(ds, cfg))
        with open(os.path.join(OUT, f"c_streams_{ds}.bin"), "wb") as f:
            f.write(b"FOGS" + struct.pack("<i", len(S)))
            for x in S:
                f.write(struct.pack("<i", len(x) // 32) + np.ascontiguousarray(x, np.float64).tobytes())
                C = {k: fe.CueController(m) for k, m in M.items()}
                for j in range(len(x) // 32):
                    w = x[j * 32:(j + 1) * 32]
                    for k in M:
                        P[k].append(C[k].step(w)[0])
        n = len(P["lab_fp32"])
        X, meta, y4, probs = pickle.load(open(os.path.join(bp.TR, f"T_probs_{ds}.pkl"), "rb"))
        assert n == len(meta), "window count differs from the reference pipeline"
        refs = {f"p_{k}": np.asarray(v, np.float64) for k, v in P.items()}
        refs["p_torch"] = np.asarray(probs["LSTM"], np.float64)
        refs["theta_lab"] = np.asarray(M["lab_fp32"].theta)
        refs["theta_torch"] = refs["theta_lab"]
        if "home_fp32" in M:
            refs["theta_home"] = np.asarray(M["home_fp32"].theta)
        np.savez(os.path.join(OUT, f"c_refs_{ds}.npz"), **refs)
        print(f"{ds}: {len(S)} stretches, {n} windows -> {OUT}/c_streams_{ds}.bin and {OUT}/c_refs_{ds}.npz")
    print(f"\nCopy to the phone: {OUT}/c_streams_*.bin and {OUT}/c_refs_*.npz")


if __name__ == "__main__":
    main()
