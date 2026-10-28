import os
import sys
import json
import struct
import platform
import subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)


def csv(path):
    d = np.genfromtxt(path, delimiter=",", names=True)
    return {k: np.atleast_1d(d[k]) for k in d.dtype.names}


def pct(x, q):
    return float(np.percentile(x, q))


def hyst(p, on_th, off_th, groups=None):
    out, st, g0 = [], False, None
    for i, v in enumerate(p):
        if groups is not None and groups[i] != g0:
            st, g0 = False, groups[i]
        st = (v >= off_th) if st else (v >= on_th)
        out.append(st)
    return np.array(out)


def device():
    return {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
            "numpy": np.__version__}


def save(name, R):
    json.dump(R, open(name, "w"), indent=2)
    print(json.dumps(R, indent=2))
    print("saved", name)


def bench(prefix, model_npz):
    import fog_edge as fe
    d = csv(prefix + "_bench.csv")
    p0 = d["pass"] == 0
    m = fe.EdgeLSTM(model_npz)
    raw = None
    for rp in ("replay_defog.npz", "replay_home.npz"):
        if os.path.exists(rp) and len(np.load(rp)["raw"]) // 32 == int(p0.sum()):
            r = np.load(rp)
            if np.allclose(r["p_pc"][: 5], d["p_ref"][p0][: 5]):
                raw = r["raw"]
    assert raw is not None, "replay npz matching this bench file not found"
    c = fe.CueController(m)
    py = np.array([c.step(raw[k * 32:(k + 1) * 32])[0] for k in range(int(p0.sum()))])
    py_on = hyst(py, *m.theta)
    ref = d["p_ref"][p0]
    R = {"device": device(), "model": model_npz, "windows_per_pass": int(p0.sum()), "passes": int(d["pass"].max() + 1),
         "C1_max_abs_prob_diff_vs_python_same_model": float(np.max(np.abs(d["p"][p0] - py))),
         "C1_decision_agreement_vs_python_same_model_%": 100 * float(np.mean(d["cue_on"][p0] == py_on)),
         "max_abs_prob_diff_vs_pc_fp32_reference": float(np.max(np.abs(d["p"][p0] - ref))),
         "decision_agreement_vs_pc_fp32_reference_%": 100 * float(np.mean(d["cue_on"][p0] == hyst(ref, *m.theta))),
         "time_per_decision_median_ms": float(np.median(d["total_ms"])), "time_per_decision_p99_ms": pct(d["total_ms"], 99),
         "time_per_decision_max_ms": float(d["total_ms"].max()),
         "stage_median_ms": {"filter": float(np.median(d["filter_ms"])), "features": float(np.median(d["features_ms"])),
                             "lstm_head_hysteresis": float(np.median(d["lstm_ms"]))}}
    R["C1_fidelity_met"] = bool(R["C1_max_abs_prob_diff_vs_python_same_model"] <= 1e-4
                                and R["C1_decision_agreement_vs_python_same_model_%"] >= 99.9)
    R["C2_time_p99_le_50ms_met"] = bool(R["time_per_decision_p99_ms"] <= 50)
    save(prefix + "_bench_summary.json", R)


def realtime(prefix):
    d = csv(prefix + "_realtime.csv")
    run = csv(prefix + "_realtime_run.csv")
    dm, db = d["delay_monotonic_ms"], d["delay_boottime_ms"]
    R = {"device": device(), "decisions": int(len(dm)),
         "processing_median_ms": float(np.median(d["processing_ms"])), "processing_p99_ms": pct(d["processing_ms"], 99),
         "delay_median_ms": float(np.median(dm)), "delay_p99_ms": pct(dm, 99), "delay_max_ms": float(dm.max()),
         "decisions_within_100ms_%": 100 * float(np.mean(dm <= 100)),
         "deadline_misses_over_500ms_monotonic": int(np.sum(dm > 500)), "deadline_misses_over_500ms_boottime": int(np.sum(db > 500)),
         "cpu_duty_cycle_%": 100 * float(run["cpu_s"][0] / run["wall_s"][0]),
         "suspended_s": float(run["boottime_s"][0] - run["wall_s"][0]), "peak_memory_mb": float(run["peak_memory_mb"][0])}
    save(prefix + "_realtime_summary.json", R)


def stream(prefix, model_npz, c_model, replay_npz):
    import fog_edge as fe
    d = csv(prefix + "_stream.csv")
    run = csv(prefix + "_stream_run.csv")
    raw = np.fromfile(prefix + "_stream_samples.bin", dtype="<f8").reshape(-1, 3)
    n = len(d["p"])
    sfile = prefix + "_offline_streams.bin"
    open(sfile, "wb").write(b"FOGS" + struct.pack("<i", 1) + struct.pack("<i", n) + raw[: n * 32].tobytes())
    subprocess.run([os.path.join(HERE, "fog_edge_c"), "dataset", c_model, sfile, prefix + "_offline.csv"], check=True)
    off = csv(prefix + "_offline.csv")
    m = fe.EdgeLSTM(model_npz)
    p_pc = np.load(replay_npz)["p_pc"]
    n1 = min(len(p_pc), n)
    local = d["on_phone_delay_ms"]
    gaps = np.diff(d["arrival_s"]) * 1000 if n > 1 else np.zeros(1)
    R = {"device": device(), "windows": int(n), "sequence_errors": int(run["sequence_errors"][0]),
         "S1_decision_agreement_with_offline_%": 100 * float(np.mean(d["cue_on"] == off["cue_on"][:n])),
         "S1_max_abs_prob_diff_offline": float(np.max(np.abs(d["p"] - off["p"][:n]))),
         "first_pass_windows": int(n1),
         "first_pass_decision_agreement_with_pc_%": 100 * float(np.mean(d["cue_on"][:n1] == hyst(p_pc[:n1], *m.theta))),
         "first_pass_max_abs_prob_diff_pc": float(np.max(np.abs(d["p"][:n1] - p_pc[:n1]))),
         "processing_median_ms": float(np.median(d["processing_ms"])), "processing_p99_ms": pct(d["processing_ms"], 99),
         "on_phone_delay_median_ms": float(np.median(local)), "on_phone_delay_p99_ms": pct(local, 99),
         "on_phone_delay_max_ms": float(local.max()),
         "window_arrival_gap_median_ms": float(np.median(gaps)), "window_arrival_gap_max_ms": float(gaps.max()),
         "window_arrival_gaps_over_1000ms": int(np.sum(gaps > 1000)),
         "cpu_duty_cycle_%": 100 * float(run["cpu_s"][0] / run["wall_s"][0]),
         "suspended_s": float(run["boottime_s"][0] - run["wall_s"][0])}
    R["S1_met"] = bool(R["S1_decision_agreement_with_offline_%"] == 100.0 and R["S1_max_abs_prob_diff_offline"] <= 1e-6)
    R["S3_met"] = bool(R["on_phone_delay_p99_ms"] <= 50)
    save(prefix + "_stream_summary.json", R)


def dataset(refs_npz, c_csv, key):
    z = np.load(refs_npz)
    c = csv(c_csv)
    ref, th = z["p_" + key], z["theta_" + key.split("_")[0]]
    assert len(ref) == len(c["p"]), "C output and reference differ in length"
    R = {"device": device(), "reference": refs_npz, "variant": key, "windows": int(len(ref)),
         "decision_agreement_%": 100 * float(np.mean(c["cue_on"] == hyst(ref, *th, groups=c["stretch"]))),
         "max_abs_prob_diff": float(np.max(np.abs(c["p"] - ref)))}
    R["C1_fidelity_met"] = bool(R["decision_agreement_%"] >= 99.9 and R["max_abs_prob_diff"] <= 1e-4)
    save(os.path.splitext(c_csv)[0] + ("_vs_torch" if key == "torch" else "") + "_summary.json", R)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "bench":
        bench(sys.argv[2], sys.argv[3])
    elif mode == "realtime":
        realtime(sys.argv[2])
    elif mode == "stream":
        stream(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    elif mode == "dataset":
        dataset(sys.argv[2], sys.argv[3], sys.argv[4])
