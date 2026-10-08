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


def main():
    R = {"device": platform.platform(), "machine": platform.machine(), "python": platform.python_version(), "numpy": np.__version__,
         "cpu_count": os.cpu_count()}
    F32, F8, REPLAY = FILES[MODEL]; R["model"] = MODEL
    d = np.load(REPLAY); raw, p_pc = d["raw"], d["p_pc"]
    print(f"replay: {len(raw) / 64 / 60:.1f} min, {len(raw) // 32} windows")
    # warm-up, then fidelity + latency (as fast as possible)
    run_replay(F32, raw[:32 * 20])
    p32, ms32 = run_replay(F32, raw)
    p8, ms8 = run_replay(F8, raw)
    R["fidelity_fp32_vs_pc_max_abs_prob_diff"] = float(np.max(np.abs(p32 - p_pc)))
    R["latency_fp32"] = stats(ms32); R["latency_int8"] = stats(ms8)
    R["model_file_bytes"] = {f: os.path.getsize(f) for f in [F32, F8]}
    R["peak_memory_MB"] = peak_memory_mb(); R["mode"] = MODE
    print("fidelity (max |p_phone - p_pc|):", R["fidelity_fp32_vs_pc_max_abs_prob_diff"])
    print("latency fp32:", R["latency_fp32"]); print("latency int8:", R["latency_int8"])
    # real-time run: one window every 0.5 s, replay looped, int8 model
    b0 = battery(); m = fe.EdgeLSTM(F8); c = fe.CueController(m)
    n_win = int(REALTIME_MIN * 60 / 0.5); nrep = len(raw) // 32; late = 0; ms = []
    cpu0, t_start = time.process_time(), time.perf_counter()
    print(f"real-time run: {REALTIME_MIN} min (keep the screen off and the phone unplugged) ...")
    for k in range(n_win):
        if k % nrep == 0: c = fe.CueController(m)                      
        target = t_start + 0.5 * (k + 1)
        t0 = time.perf_counter(); c.step(raw[(k % nrep) * 32:((k % nrep) + 1) * 32]); ms.append(1000 * (time.perf_counter() - t0))
        wait = target - time.perf_counter()
        if wait > 0: time.sleep(wait)
        else: late += 1
    wall = time.perf_counter() - t_start; cpu = time.process_time() - cpu0; b1 = battery()
    R["realtime"] = {"minutes": REALTIME_MIN, **stats(np.asarray(ms)), "deadline_misses": late,
                     "cpu_seconds": cpu, "wall_seconds": wall, "cpu_duty_cycle_%": 100 * cpu / wall,
                     "battery_start": b0, "battery_end": b1}
    if b0 and b1:
        R["realtime"]["battery_drop_%"] = b0.get("percentage", np.nan) - b1.get("percentage", np.nan)
    R["E3_met"] = bool(R["latency_int8"]["p99_ms"] <= 50 and R["latency_fp32"]["p99_ms"] <= 50
                       and R["fidelity_fp32_vs_pc_max_abs_prob_diff"] <= 1e-4)
    json.dump(R, open(f"phone_results_{MODEL}_{MODE}.json", "w"), indent=2, default=str)
    print("\nreal-time:", {k: v for k, v in R["realtime"].items() if not k.startswith("battery_")})
    print("battery drop (%):", R["realtime"].get("battery_drop_%", "not available (Termux:API not installed)"))
    print(f"\nPRE-SPECIFIED E3: {'MET' if R['E3_met'] else 'NOT met'}\nSaved phone_results_{MODEL}_{MODE}.json in {HERE}")


if __name__ == "__main__":
    main()
