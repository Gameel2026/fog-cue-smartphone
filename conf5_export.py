import os
import struct
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)


def get(d, k):
    if k + "__scale" in d:
        return d[k].astype(np.float32) * d[k + "__scale"][:, None]
    return d[k].astype(np.float32)


def export_model(src, dst):
    d = np.load(src)
    f64 = lambda a: np.ascontiguousarray(np.asarray(a, np.float64)).tobytes()
    f32 = lambda a: np.ascontiguousarray(np.asarray(a, np.float32)).tobytes()
    assert d["weight_ih_l0"].shape == (256, 72) and d["sos"].shape == (4, 6)
    parts = [b"FOGC", f64(d["mu"]), f64(d["sd"]), f64(d["sos"]), f64(d["zi_unit"]),
             f64([float(d["theta_on"])]), f64([float(d["theta_off"])])]
    for l in range(2):
        parts += [f32(get(d, f"weight_ih_l{l}")), f32(get(d, f"weight_hh_l{l}")),
                  f32((d[f"bias_ih_l{l}"] + d[f"bias_hh_l{l}"]).astype(np.float32))]
    parts += [f32(get(d, "head_weight")), f32(d["head_bias"])]
    open(dst, "wb").write(b"".join(parts))
    print(f"{src} -> {dst} ({os.path.getsize(dst)} bytes; theta {float(d['theta_on'])}/{float(d['theta_off'])})")


def export_replay(src, dst):
    r = np.load(src)
    raw = np.ascontiguousarray(r["raw"].astype(np.float64))
    p = np.ascontiguousarray(r["p_pc"].astype(np.float64)) if "p_pc" in r.files else np.zeros(0)
    open(dst, "wb").write(b"FOGR" + struct.pack("<ii", len(raw), len(p)) + raw.tobytes() + p.tobytes())
    print(f"{src} -> {dst} ({len(raw)} samples, {len(p)} reference probabilities)")


if __name__ == "__main__":
    for src, dst in [("edge_model_fp32.npz", "c_model_lab_fp32.bin"), ("edge_model_int8.npz", "c_model_lab_int8.bin"),
                     ("edge_home_fp32.npz", "c_model_home_fp32.bin"), ("edge_home_int8.npz", "c_model_home_int8.bin")]:
        if os.path.exists(src):
            export_model(src, dst)
    for src, dst in [("replay_defog.npz", "c_replay_lab.bin"), ("replay_home.npz", "c_replay_home.bin")]:
        if os.path.exists(src):
            export_replay(src, dst)
