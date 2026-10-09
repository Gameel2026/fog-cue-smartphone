import os
import sys
import json
import time
import platform
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE); sys.path.insert(0, HERE)                 
import numpy as np
import fog_edge as fe

MODE = "quick"          
MODEL = "lab"           

FILES = {"lab": ("edge_model_fp32.npz", "edge_model_int8.npz", "replay_defog.npz"),
         "home": ("edge_home_fp32.npz", "edge_home_int8.npz", "replay_home.npz")}
if len(sys.argv) > 1: MODE = sys.argv[1]
if len(sys.argv) > 2: MODEL = sys.argv[2]          
REALTIME_MIN = 2 if MODE == "quick" else 60


def peak_memory_mb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        try:
            for line in open("/proc/self/status"):
                if line.startswith("VmHWM"): return int(line.split()[1]) / 1024.0
        except Exception:
            return None


def battery():
    try:
        out = subprocess.run(["termux-battery-status"], capture_output=True, text=True, timeout=20).stdout
        return json.loads(out)
    except Exception:
        return None


def run_replay(model_path, raw):
    m = fe.EdgeLSTM(model_path); c = fe.CueController(m); p, ms = [], []
    for k in range(len(raw) // 32):
        t0 = time.perf_counter(); pk, _, _ = c.step(raw[k * 32:(k + 1) * 32]); ms.append(1000 * (time.perf_counter() - t0)); p.append(pk)
    return np.asarray(p), np.asarray(ms)


def stats(ms):
    return {"median_ms": float(np.median(ms)), "p99_ms": float(np.percentile(ms, 99)), "max_ms": float(ms.max()), "windows": int(len(ms))}


def stages(model_path, raw):
    m = fe.EdgeLSTM(model_path); filt = fe.SOSFilter(m.sos, m.zi); buf = []; on = False; T = {k: [] for k in ["filter", "features", "lstm", "hysteresis"]}
    pc = time.perf_counter
    for k in range(len(raw) // 32):
        t0 = pc(); w = filt(raw[k * 32:(k + 1) * 32]); t1 = pc(); f = fe.window_features(w); t2 = pc()
        z = np.clip((f - m.mu) / m.sd, -10, 10).astype(np.float32); buf.append(z); buf = buf[-8:]
        Z = np.zeros((8, len(z)), np.float32); Z[8 - len(buf):] = np.stack(buf); p = m.prob(Z); t3 = pc()
        a, b = m.theta; on = (p >= b) if on else (p >= a); t4 = pc()
        for key, dt in zip(T, [t1 - t0, t2 - t1, t3 - t2, t4 - t3]): T[key].append(1000 * dt)
    return {k: {"median_ms": float(np.median(v)), "p99_ms": float(np.percentile(v, 99))} for k, v in T.items()}


def main_stages():
    raw = np.load(FILES[MODEL][2])["raw"]; F32, F8 = FILES[MODEL][0], FILES[MODEL][1]
    stages(F32, raw[:32 * 20])                                          
    R = {"model": MODEL, "fp32": stages(F32, raw), "int8": stages(F8, raw)}
    json.dump(R, open(f"phone_stages_{MODEL}.json", "w"), indent=2)
    for v in ["fp32", "int8"]:
        print(v, {k: round(x["median_ms"], 3) for k, x in R[v].items()}, "(median ms)")
        print(v, {k: round(x["p99_ms"], 3) for k, x in R[v].items()}, "(p99 ms)")
    print(f"Saved phone_stages_{MODEL}.json in {HERE}")


def main():
    if MODE == "stages":
        return main_stages()
    R = {"device": platform.platform(), "machine": platform.machine(), "python": platform.python_version(), "numpy": np.__version__,
         "cpu_count": os.cpu_count()}
    F32, F8, REPLAY = FILES[MODEL]; R["model"] = MODEL
    d = np.load(REPLAY); raw, p_pc = d["raw"], d["p_pc"]
    print(f"replay: {len(raw) / 64 / 60:.1f} min, {len(raw) // 32} windows")
    run_replay(F32, raw[:32 * 20])
    p32, ms32 = run_replay(F32, raw)
    p8, ms8 = run_replay(F8, raw)
    R["fidelity_fp32_vs_pc_max_abs_prob_diff"] = float(np.max(np.abs(p32 - p_pc)))
    R["latency_fp32"] = stats(ms32); R["latency_int8"] = stats(ms8)
    R["model_file_bytes"] = {f: os.path.getsize(f) for f in [F32, F8]}
    R["peak_memory_MB"] = peak_memory_mb(); R["mode"] = MODE
    print("fidelity (max |p_phone - p_pc|):", R["fidelity_fp32_vs_pc_max_abs_prob_diff"])
    print("latency fp32:", R["latency_fp32"]); print("latency int8:", R["latency_int8"])
    b0 = battery(); m = fe.EdgeLSTM(F8); c = fe.CueController(m)
    if b0 is None and MODE == "full":
        b0 = ask_battery("Battery % NOW (before the run), then press Enter: ")
    n_win = int(REALTIME_MIN * 60 / 0.5); nrep = len(raw) // 32; late = 0; ms = []; lag = []
    cpu0, t_start = time.process_time(), time.perf_counter()
    print(f"real-time run: {REALTIME_MIN} min - screen off, phone unplugged, do not use the phone ...")
    for k in range(n_win):
        if k % nrep == 0: c = fe.CueController(m)                       
        target = t_start + 0.5 * (k + 1)                             
        t0 = time.perf_counter(); c.step(raw[(k % nrep) * 32:((k % nrep) + 1) * 32]); t1 = time.perf_counter()
        ms.append(1000 * (t1 - t0)); lag.append(1000 * (t1 - (target - 0.5)))   
        wait = target - time.perf_counter()
        if wait > 0: time.sleep(wait)
        else: late += 1
    wall = time.perf_counter() - t_start; cpu = time.process_time() - cpu0
    lag = np.asarray(lag)
    R["realtime"] = {"minutes": REALTIME_MIN, **stats(np.asarray(ms)), "deadline_misses": late,
                     "decision_delay_median_ms": float(np.median(lag)), "decision_delay_p99_ms": float(np.percentile(lag, 99)),
                     "decision_delay_max_ms": float(lag.max()), "decisions_delayed_over_100ms": int((lag > 100).sum()),
                     "decisions_delayed_over_500ms": int((lag > 500).sum()),
                     "cpu_seconds": cpu, "wall_seconds": wall, "cpu_duty_cycle_%": 100 * cpu / wall, "battery_start": b0}
    np.savetxt(f"phone_realtime_{MODEL}_{MODE}.csv", np.c_[np.arange(n_win), ms, lag], delimiter=",",
               header="window,processing_ms,decision_delay_ms", comments="", fmt="%.4f")
    R["E3_met"] = bool(R["latency_int8"]["p99_ms"] <= 50 and R["latency_fp32"]["p99_ms"] <= 50
                       and R["fidelity_fp32_vs_pc_max_abs_prob_diff"] <= 1e-4)
    out = f"phone_results_{MODEL}_{MODE}.json"
    json.dump(R, open(out, "w"), indent=2, default=str)                
    print("\nreal-time:", {k: v for k, v in R["realtime"].items() if not k.startswith("battery_")})
    print(f"\nPRE-SPECIFIED E3: {'MET' if R['E3_met'] else 'NOT met'}\nSaved {out} in {HERE}")
    b1 = battery()
    if b1 is None and MODE == "full":
        b1 = ask_battery("Battery % NOW (after the run), then press Enter: ")
    R["realtime"]["battery_end"] = b1
    p0 = b0.get("percentage") if isinstance(b0, dict) else b0; p1 = b1.get("percentage") if isinstance(b1, dict) else b1
    if p0 is not None and p1 is not None:
        R["realtime"]["battery_drop_%"] = p0 - p1
    json.dump(R, open(out, "w"), indent=2, default=str)
    print("battery drop (%):", R["realtime"].get("battery_drop_%", "not recorded"))


def ask_battery(msg):
    try:
        return float(input(msg).strip().replace("%", ""))
    except Exception:
        return None


if __name__ == "__main__":
    main()
