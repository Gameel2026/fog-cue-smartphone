import os
import sys
import json
import time
import struct
import socket
import threading
import platform
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

PORT = 50007
FS = 64
PER_PACKET = 4
WIN = 32
PKT = struct.Struct("<I12d")
REP = struct.Struct("<IBd")


def ask(msg, default):
    try:
        v = input(msg).strip()
    except Exception:
        v = ""
    return v if v else default


def reader(sock, recv, done):
    buf = bytearray()
    while True:
        try:
            chunk = sock.recv(4096)
        except Exception:
            break
        if not chunk:
            break
        t = time.perf_counter()
        buf.extend(chunk)
        while len(buf) >= REP.size:
            k, on, p = REP.unpack_from(buf, 0)
            del buf[:REP.size]
            recv.append((k, t, on, p))
    done.set()


def main():
    host = sys.argv[1] if len(sys.argv) > 1 else ask("Phone IP address: ", "")
    minutes = float(sys.argv[2]) if len(sys.argv) > 2 else float(ask("Minutes [60]: ", "60"))
    tag = sys.argv[3] if len(sys.argv) > 3 else ask("Device tag (Enter = main phone, device2 = Pixel): ", "")
    raw = np.load("replay_defog.npz")["raw"].astype(np.float64)
    n_win = int(round(minutes * 60 * FS / WIN))
    n_pkt = n_win * WIN // PER_PACKET
    n_raw = len(raw)
    s = socket.create_connection((host, PORT), timeout=10)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s.settimeout(None)
    recv, done = [], threading.Event()
    th = threading.Thread(target=reader, args=(s, recv, done), daemon=True)
    th.start()
    print(f"connected to {host}:{PORT}; streaming {minutes:g} min ({n_win} windows) at {FS} Hz ...")
    sched = np.zeros(n_pkt)
    sent = np.zeros(n_pkt)
    t0 = time.perf_counter() + 0.5
    for i in range(n_pkt):
        target = t0 + (i + 1) * PER_PACKET / FS
        while True:
            rem = target - time.perf_counter()
            if rem <= 0:
                break
            if rem > 0.02:
                time.sleep(rem - 0.02)
        a = (i * PER_PACKET) % n_raw
        idx = [(a + j) % n_raw for j in range(PER_PACKET)]
        s.sendall(PKT.pack(i, *raw[idx].ravel()))
        sched[i] = target
        sent[i] = time.perf_counter()
        if i % (n_pkt // 20 or 1) == 0:
            print(f"{100 * i / n_pkt:5.1f} %  decisions received: {len(recv)}")
    try:
        s.shutdown(socket.SHUT_WR)
    except Exception:
        pass
    done.wait(30)
    s.close()
    ppw = WIN // PER_PACKET
    win_end = sched[ppw - 1::ppw][:n_win]
    got = {k: (t, on, p) for k, t, on, p in recv}
    delay = np.array([1000 * (got[k][0] - win_end[k]) if k in got else np.nan for k in range(n_win)])
    ok = delay[np.isfinite(delay)]
    late_send = 1000 * (sent - sched)
    R = {"laptop": platform.platform(), "python": platform.python_version(), "numpy": np.__version__, "phone": host,
         "minutes": minutes, "windows": n_win, "packets_sent": n_pkt, "samples_per_packet": PER_PACKET,
         "decisions_received": int(len(ok)), "decisions_missing": int(n_win - len(ok)),
         "roundtrip_delay_median_ms": float(np.median(ok)) if len(ok) else None,
         "roundtrip_delay_p99_ms": float(np.percentile(ok, 99)) if len(ok) else None,
         "roundtrip_delay_max_ms": float(ok.max()) if len(ok) else None,
         "decisions_within_100ms_%": 100 * float(np.mean(np.nan_to_num(delay, nan=1e9) <= 100)),
         "decisions_within_500ms_%": 100 * float(np.mean(np.nan_to_num(delay, nan=1e9) <= 500)),
         "decisions_over_500ms": int(np.sum(np.nan_to_num(delay, nan=1e9) > 500)),
         "sender_lateness_median_ms": float(np.median(late_send)), "sender_lateness_p99_ms": float(np.percentile(late_send, 99)),
         "sender_lateness_max_ms": float(late_send.max())}
    R["S2_met"] = bool(R["decisions_within_500ms_%"] >= 99.0)
    out = f"stream_laptop{('_' + tag) if tag else ''}.json"
    json.dump(R, open(out, "w"), indent=2)
    on_arr = np.array([got[k][1] if k in got else -1 for k in range(n_win)])
    p_arr = np.array([got[k][2] if k in got else np.nan for k in range(n_win)])
    np.savetxt(out.replace(".json", ".csv"), np.c_[np.arange(n_win), delay, on_arr, p_arr], delimiter=",",
               header="window,roundtrip_delay_ms,cue_on,p", comments="", fmt=["%d", "%.3f", "%d", "%.9f"])
    print(json.dumps(R, indent=2))
    print(f"S2 (>= 99% of decisions within 500 ms): {'MET' if R['S2_met'] else 'NOT met'}")
    print(f"Saved {out} and {out.replace('.json', '.csv')} in {HERE}")


if __name__ == "__main__":
    main()
