import numpy as np

FS, W, NFFT, MODE_RES, EPS = 64, 32, 256, 10.0, 1e-12
LOCO, FREEZE = (0.5, 3.0), (3.0, 8.0)
STAT = ["mean", "median", "mode", "min", "max", "range", "harmean", "StDev", "Variance", "Mean_Ab", "Med_Ab",
        "kurtosis", "skewness", "RMS"]
SPEC = ["Locomotorypower", "Freezepower", "Freezeindex", "Sumpower", "Meanf", "medf", "spectralcentroid",
        "spectralKurtosis", "spectralEntropy", "Spectralpeak"]
_F = np.arange(NFFT // 2 + 1) * FS / NFFT
_HANN = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(W) / W)            


def stat_feats(x):
    vals, cnt = np.unique(np.round(x / MODE_RES), return_counts=True)
    m = x.mean(); d = x - m; m2 = np.mean(d ** 2)
    a = np.abs(x) + EPS
    return [m, np.median(x), vals[cnt.argmax()] * MODE_RES, x.min(), x.max(), x.max() - x.min(),
            len(a) / np.sum(1.0 / a), x.std(ddof=1), x.var(ddof=1), np.mean(np.abs(d)),
            np.mean(np.abs(x - np.median(x))),
            np.mean(d ** 4) / m2 ** 2 if m2 > 0 else np.nan,
            np.mean(d ** 3) / m2 ** 1.5 if m2 > 0 else np.nan,
            np.sqrt(np.mean(x ** 2))]


def spec_feats(x):
    xw = (x - x.mean()) * _HANN
    P = np.abs(np.fft.rfft(xw, NFFT)) ** 2 / (FS * np.sum(_HANN ** 2))
    P[1:-1] *= 2                                                     
    f = _F; df_ = f[1] - f[0]
    bp = lambda lo, hi: P[(f >= lo) & (f < hi)].sum() * df_
    loco, frz = bp(*LOCO), bp(*FREEZE)
    p = P / (P.sum() + EPS); mean_f = np.sum(f * p)
    mag = np.sqrt(P); mag = mag / (mag.sum() + EPS); centroid = np.sum(f * mag)
    sd = np.sqrt(np.sum((f - mean_f) ** 2 * p)) + EPS
    return [loco, frz, frz / (loco + EPS), frz + loco, mean_f,
            f[min(np.searchsorted(np.cumsum(P), 0.5 * P.sum()), len(f) - 1)], centroid,
            np.sum((f - mean_f) ** 4 * p) / sd ** 4, -np.sum(p * np.log2(p + EPS)) / np.log2(len(p)), f[np.argmax(P)]]


def window_features(w):
    out = []
    for ax in range(w.shape[1]):
        out += stat_feats(w[:, ax]) + spec_feats(w[:, ax])
    v = np.asarray(out, float)
    v[~np.isfinite(v)] = 0.0                                         
    return v


class SOSFilter:
    def __init__(self, sos, zi_unit):
        self.sos = np.asarray(sos, float); self.zi_unit = np.asarray(zi_unit, float); self.z = None

    def __call__(self, x):
        x = np.asarray(x, float)
        if self.z is None:                                            
            self.z = self.zi_unit[:, :, None] * x[0][None, None, :]
        y = np.empty_like(x)
        for n in range(len(x)):
            v = x[n]
            for s, (b0, b1, b2, a0, a1, a2) in enumerate(self.sos):
                z = self.z[s]; out = b0 * v + z[0]
                z[0] = b1 * v - a1 * out + z[1]; z[1] = b2 * v - a2 * out
                v = out
            y[n] = v
        return y


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


class EdgeLSTM:
    def __init__(self, path):
        d = np.load(path)
        get = lambda k: (d[k].astype(np.float32) * d[k + "__scale"][:, None]) if (k + "__scale") in d else d[k].astype(np.float32)
        self.layers = [(get(f"weight_ih_l{l}"), get(f"weight_hh_l{l}"), (d[f"bias_ih_l{l}"] + d[f"bias_hh_l{l}"]).astype(np.float32))
                       for l in range(2)]
        self.Wh, self.bh = get("head_weight"), d["head_bias"].astype(np.float32)
        self.mu, self.sd = d["mu"].astype(np.float64), d["sd"].astype(np.float64)
        self.sos, self.zi = d["sos"], d["zi_unit"]
        self.theta = (float(d["theta_on"]), float(d["theta_off"]))
        self.H = self.layers[0][1].shape[1]

    def prob(self, Z):
        x = Z
        for Wi, Wh, b in self.layers:
            h = np.zeros(self.H, np.float32); c = np.zeros(self.H, np.float32); hs = []
            xi = x @ Wi.T + b
            for t in range(len(x)):
                g = xi[t] + Wh @ h
                i, f, gg, o = np.split(g, 4)
                c = _sig(f) * c + _sig(i) * np.tanh(gg); h = _sig(o) * np.tanh(c); hs.append(h)
            x = np.stack(hs)
        z = self.Wh @ x[-1] + self.bh; z = z - z.max(); e = np.exp(z)
        return float(e[1] / e.sum())


class CueController:
    def __init__(self, model, n_past=8):
        self.m = model; self.filt = SOSFilter(model.sos, model.zi); self.buf = []; self.n_past = n_past; self.on = False

    def step(self, raw_window):
        w = self.filt(raw_window)
        f = window_features(w)
        z = np.clip((f - self.m.mu) / self.m.sd, -10, 10).astype(np.float32)
        self.buf.append(z); self.buf = self.buf[-self.n_past:]
        Z = np.zeros((self.n_past, len(z)), np.float32); Z[self.n_past - len(self.buf):] = np.stack(self.buf)
        p = self.m.prob(Z)
        a, b = self.m.theta
        self.on = (p >= b) if self.on else (p >= a)
        return p, self.on, f