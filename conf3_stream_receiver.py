import os
import sys
import json
import time
import struct
import socket
import platform

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)
import numpy as np
import fog_edge as fe

PORT = 50007
WIN = 32
PER_PACKET = 4
PKT = struct.Struct("<I12d")
REP = struct.Struct("<IBd")
MODEL = sys.argv[1] if len(sys.argv) > 1 else "int8"
TAG = sys.argv[2] if len(sys.argv) > 2 else ""


def boottime():
    try:
        return time.clock_gettime(time.CLOCK_BOOTTIME)
    except Exception:
        return float("nan")


def local_ip():
    try:
        u = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        u.connect(("10.255.255.255", 1))
        ip = u.getsockname()[0]
        u.close()
        return ip
    except Exception:
        return "unknown"


def main():
    path = f"edge_model_{MODEL}.npz"
    m = fe.EdgeLSTM(path)
    rep = np.load("replay_defog.npz")
    p_pc, n_raw = rep["p_pc"], len(rep["raw"])
    warm = fe.CueController(m)
    for k in range(20):
        warm.step(rep["raw"][(k * WIN) % (n_raw - WIN):(k * WIN) % (n_raw - WIN) + WIN])
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", PORT))
    srv.listen(1)
    print(f"model {path}; waiting on {local_ip()}:{PORT} ... (start the sender on the laptop)")
    conn, addr = srv.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print("connected:", addr[0])
    c = fe.CueController(m)
    buf = bytearray()
    win = []
    samples = []
    P, ON, proc, local, arr_last, boot = [], [], [], [], [], []
    gaps = []
    seq_err = 0
    expect = 0
    prev = None
    cpu0, w0, b0 = time.process_time(), time.perf_counter(), boottime()
    while True:
        chunk = conn.recv(65536)
        if not chunk:
            break
        t_arr = time.perf_counter()
        buf.extend(chunk)
        while len(buf) >= PKT.size:
            v = PKT.unpack_from(buf, 0)
            del buf[:PKT.size]
            if v[0] != expect:
                seq_err += 1
            expect = v[0] + 1
            if prev is not None:
                gaps.append(1000 * (t_arr - prev))
            prev = t_arr
            x = np.asarray(v[1:], np.float64).reshape(PER_PACKET, 3)
            win.append(x)
            samples.append(x)
            if len(win) * PER_PACKET == WIN:
                w = np.concatenate(win)
                win = []
                t1 = time.perf_counter()
                p, on, _ = c.step(w)
                t2 = time.perf_counter()
                k = len(P)
                conn.sendall(REP.pack(k, int(on), p))
                P.append(p)
                ON.append(bool(on))
                proc.append(1000 * (t2 - t1))
                local.append(1000 * (t2 - t_arr))
                arr_last.append(t_arr - w0)
                boot.append(boottime() - b0)
                if k % 600 == 0:
                    print(f"window {k}  p={p:.3f}  cue={'ON' if on else 'off'}  processing {1000 * (t2 - t1):.1f} ms")
    wall = time.perf_counter() - w0
    cpu = time.process_time() - cpu0
    boot_total = boottime() - b0
    conn.close()
    srv.close()
    print("stream ended; offline check ...")
    S = np.concatenate(samples)
    o = fe.CueController(m)
    Po, Oo = [], []
    for k in range(len(P)):
        p, on, _ = o.step(S[k * WIN:(k + 1) * WIN])
        Po.append(p)
        Oo.append(bool(on))
    P, Po, ON, Oo = np.array(P), np.array(Po), np.array(ON), np.array(Oo)
    proc, local, gaps = np.array(proc), np.array(local), np.array(gaps) if gaps else np.zeros(1)
    n1 = min(len(p_pc), len(P))
    a, b = m.theta
    ref_on, st = [], False
    for p in p_pc[:n1]:
        st = (p >= b) if st else (p >= a)
        ref_on.append(st)
    R = {"device": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
         "numpy": np.__version__, "cpu_count": os.cpu_count(), "model": path, "windows": int(len(P)),
         "sequence_errors": int(seq_err), "samples_received": int(len(S)),
         "S1_decision_agreement_with_offline_%": 100 * float(np.mean(ON == Oo)) if len(P) else None,
         "S1_max_abs_prob_diff_offline": float(np.max(np.abs(P - Po))) if len(P) else None,
         "first_pass_windows": int(n1),
         "first_pass_decision_agreement_with_pc_%": 100 * float(np.mean(ON[:n1] == np.array(ref_on))) if n1 else None,
         "first_pass_max_abs_prob_diff_pc": float(np.max(np.abs(P[:n1] - p_pc[:n1]))) if n1 else None,
         "processing_median_ms": float(np.median(proc)), "processing_p99_ms": float(np.percentile(proc, 99)),
         "processing_max_ms": float(proc.max()),
         "on_phone_delay_median_ms": float(np.median(local)), "on_phone_delay_p99_ms": float(np.percentile(local, 99)),
         "on_phone_delay_max_ms": float(local.max()),
         "packet_gap_median_ms": float(np.median(gaps)), "packet_gap_p99_ms": float(np.percentile(gaps, 99)),
         "packet_gap_max_ms": float(gaps.max()), "packet_gaps_over_100ms": int(np.sum(gaps > 100)),
         "packet_gaps_over_500ms": int(np.sum(gaps > 500)),
         "cue_on_windows": int(ON.sum()), "wall_seconds": wall, "cpu_seconds": cpu,
         "cpu_duty_cycle_%": 100 * cpu / wall, "boottime_seconds": boot_total, "suspended_seconds": boot_total - wall}
    R["S1_met"] = bool(len(P) and R["S1_decision_agreement_with_offline_%"] == 100.0 and R["S1_max_abs_prob_diff_offline"] <= 1e-6)
    R["S3_met"] = bool(R["on_phone_delay_p99_ms"] <= 50)
    out = f"stream_phone_{MODEL}{('_' + TAG) if TAG else ''}.json"
    json.dump(R, open(out, "w"), indent=2)
    np.savetxt(out.replace(".json", ".csv"), np.c_[np.arange(len(P)), P, ON, proc, local, arr_last, boot], delimiter=",",
               header="window,p,cue_on,processing_ms,on_phone_delay_ms,arrival_s,boottime_s", comments="",
               fmt=["%d", "%.9f", "%d", "%.4f", "%.4f", "%.4f", "%.4f"])
    print(json.dumps(R, indent=2))
    print(f"S1 (streamed = offline): {'MET' if R['S1_met'] else 'NOT met'}   S3 (on-phone delay p99 <= 50 ms): {'MET' if R['S3_met'] else 'NOT met'}")
    print(f"Saved {out} and {out.replace('.json', '.csv')} in {HERE}")


if __name__ == "__main__":
    main()
