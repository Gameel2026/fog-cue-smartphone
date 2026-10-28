#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <errno.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/resource.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <signal.h>

#define FS 64
#define W 32
#define NAX 3
#define NF 72
#define NPAST 8
#define H 64
#define G (4 * H)
#define NFFT 256
#define NB (NFFT / 2 + 1)
#define EPS 1e-12
#define MODE_RES 10.0
#define NSOS 4

typedef struct {
    double mu[NF], sd[NF], sos[NSOS][6], zi[NSOS][2], th_on, th_off;
    float Wi0[G * NF], Wh0[G * H], b0[G], Wi1[G * H], Wh1[G * H], b1[G], Wo[2 * H], bo[2];
} Model;

static void die(const char *m) { fprintf(stderr, "error: %s\n", m); exit(1); }

static void rd(FILE *f, void *p, size_t sz, size_t n) { if (fread(p, sz, n, f) != n) die("model/replay file too short"); }

static Model *load_model(const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) die("cannot open model file");
    char magic[4]; rd(f, magic, 1, 4); if (memcmp(magic, "FOGC", 4)) die("bad model magic");
    Model *m = calloc(1, sizeof(Model));
    rd(f, m->mu, 8, NF); rd(f, m->sd, 8, NF); rd(f, m->sos, 8, NSOS * 6); rd(f, m->zi, 8, NSOS * 2);
    rd(f, &m->th_on, 8, 1); rd(f, &m->th_off, 8, 1);
    rd(f, m->Wi0, 4, G * NF); rd(f, m->Wh0, 4, G * H); rd(f, m->b0, 4, G);
    rd(f, m->Wi1, 4, G * H); rd(f, m->Wh1, 4, G * H); rd(f, m->b1, 4, G);
    rd(f, m->Wo, 4, 2 * H); rd(f, m->bo, 4, 2);
    fclose(f); return m;
}

static double HANN[W], HANN_SS, COSK[NB][W], SINK[NB][W];

static void init_tables(void) {
    HANN_SS = 0;
    for (int n = 0; n < W; n++) { HANN[n] = 0.5 - 0.5 * cos(2 * M_PI * n / W); HANN_SS += HANN[n] * HANN[n]; }
    for (int k = 0; k < NB; k++)
        for (int n = 0; n < W; n++) { double a = 2 * M_PI * (double)k * n / NFFT; COSK[k][n] = cos(a); SINK[k][n] = -sin(a); }
}

static int cmpd(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

static void stat_feats(const double *x, double *o) {
    double s[W], m = 0, m2 = 0, m3 = 0, m4 = 0, ha = 0, sq = 0, mad = 0, medad = 0;
    for (int i = 0; i < W; i++) m += x[i];
    m /= W;
    for (int i = 0; i < W; i++) {
        double d = x[i] - m; m2 += d * d; m3 += d * d * d; m4 += d * d * d * d;
        ha += 1.0 / (fabs(x[i]) + EPS); sq += x[i] * x[i]; mad += fabs(d);
    }
    memcpy(s, x, sizeof s); qsort(s, W, sizeof(double), cmpd);
    double med = 0.5 * (s[W / 2 - 1] + s[W / 2]);
    for (int i = 0; i < W; i++) medad += fabs(x[i] - med);

    double r[W]; for (int i = 0; i < W; i++) r[i] = rint(x[i] / MODE_RES);
    qsort(r, W, sizeof(double), cmpd);
    double best = r[0]; int bc = 0;
    for (int i = 0; i < W;) { int j = i; while (j < W && r[j] == r[i]) j++; if (j - i > bc) { bc = j - i; best = r[i]; } i = j; }
    double m2p = m2 / W, var1 = m2 / (W - 1);
    o[0] = m; o[1] = med; o[2] = best * MODE_RES; o[3] = s[0]; o[4] = s[W - 1]; o[5] = s[W - 1] - s[0];
    o[6] = W / ha; o[7] = sqrt(var1); o[8] = var1; o[9] = mad / W; o[10] = medad / W;
    o[11] = m2p > 0 ? (m4 / W) / (m2p * m2p) : NAN;
    o[12] = m2p > 0 ? (m3 / W) / pow(m2p, 1.5) : NAN;
    o[13] = sqrt(sq / W);
}

static void spec_feats(const double *x, double *o) {
    double xw[W], P[NB], m = 0;
    for (int i = 0; i < W; i++) m += x[i];
    m /= W;
    for (int i = 0; i < W; i++) xw[i] = (x[i] - m) * HANN[i];
    double norm = FS * HANN_SS, tot = 0, df = (double)FS / NFFT;
    for (int k = 0; k < NB; k++) {
        double re = 0, im = 0;
        for (int n = 0; n < W; n++) { re += xw[n] * COSK[k][n]; im += xw[n] * SINK[k][n]; }
        P[k] = (re * re + im * im) / norm;
        if (k > 0 && k < NB - 1) P[k] *= 2;
        tot += P[k];
    }
    double loco = 0, frz = 0, mean_f = 0, msum = 0, cen = 0, var = 0, kurt = 0, ent = 0, cs = 0;
    for (int k = 0; k < NB; k++) {
        double f = k * df;
        if (f >= 0.5 && f < 3.0) loco += P[k];
        if (f >= 3.0 && f < 8.0) frz += P[k];
        msum += sqrt(P[k]);
    }
    loco *= df; frz *= df;
    double den = tot + EPS, mden = msum + EPS;
    for (int k = 0; k < NB; k++) { double p = P[k] / den; mean_f += k * df * p; cen += k * df * (sqrt(P[k]) / mden); }
    for (int k = 0; k < NB; k++) { double p = P[k] / den, d = k * df - mean_f; var += d * d * p; kurt += d * d * d * d * p; ent += p * log2(p + EPS); }
    double sd = sqrt(var) + EPS;
    int med = NB - 1, ip = 0;
    for (int k = 0; k < NB; k++) { cs += P[k]; if (cs >= 0.5 * tot) { med = k; break; } }
    for (int k = 1; k < NB; k++) if (P[k] > P[ip]) ip = k;
    o[0] = loco; o[1] = frz; o[2] = frz / (loco + EPS); o[3] = frz + loco; o[4] = mean_f; o[5] = med * df;
    o[6] = cen; o[7] = kurt / (sd * sd * sd * sd); o[8] = -ent / log2((double)NB); o[9] = ip * df;
}

typedef struct {
    const Model *m; double z[NSOS][2][NAX]; int started, nbuf, on; float buf[NPAST][NF];
} Ctrl;

static void ctrl_init(Ctrl *c, const Model *m) { memset(c, 0, sizeof *c); c->m = m; }

static double now_s(clockid_t id) { struct timespec t; clock_gettime(id, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

static inline float sigf(float x) { return 1.0f / (1.0f + expf(-x)); }

static void lstm_layer(const float *X, int nin, const float *Wi, const float *Wh, const float *b, float *Y) {
    float h[H] = {0}, c[H] = {0}, g[G];
    for (int t = 0; t < NPAST; t++) {
        const float *x = X + t * nin;
        for (int r = 0; r < G; r++) {
            float s = 0; const float *wi = Wi + r * nin; for (int j = 0; j < nin; j++) s += wi[j] * x[j];
            s += b[r];
            float u = 0; const float *wh = Wh + r * H; for (int j = 0; j < H; j++) u += wh[j] * h[j];
            g[r] = s + u;
        }
        for (int j = 0; j < H; j++) {
            float i = sigf(g[j]), f = sigf(g[H + j]), gg = tanhf(g[2 * H + j]), o = sigf(g[3 * H + j]);
            c[j] = f * c[j] + i * gg; h[j] = o * tanhf(c[j]); Y[t * H + j] = h[j];
        }
    }
}

static double ctrl_step(Ctrl *c, const double *raw, double *st) {
    const Model *m = c->m; double t0 = st ? now_s(CLOCK_MONOTONIC) : 0;
    double y[W * NAX];
    if (!c->started) {
        for (int s = 0; s < NSOS; s++) for (int a = 0; a < NAX; a++) { c->z[s][0][a] = m->zi[s][0] * raw[a]; c->z[s][1][a] = m->zi[s][1] * raw[a]; }
        c->started = 1;
    }
    for (int n = 0; n < W; n++)
        for (int a = 0; a < NAX; a++) {
            double v = raw[n * NAX + a];
            for (int s = 0; s < NSOS; s++) {
                const double *q = m->sos[s]; double out = q[0] * v + c->z[s][0][a];
                c->z[s][0][a] = q[1] * v - q[4] * out + c->z[s][1][a]; c->z[s][1][a] = q[2] * v - q[5] * out; v = out;
            }
            y[n * NAX + a] = v;
        }
    double t1 = st ? now_s(CLOCK_MONOTONIC) : 0;
    double f[NF], col[W];
    for (int a = 0; a < NAX; a++) {
        for (int n = 0; n < W; n++) col[n] = y[n * NAX + a];
        stat_feats(col, f + a * 24); spec_feats(col, f + a * 24 + 14);
    }
    float z[NF];
    for (int k = 0; k < NF; k++) {
        double v = isfinite(f[k]) ? f[k] : 0.0;
        v = (v - m->mu[k]) / m->sd[k]; if (v > 10) v = 10; if (v < -10) v = -10; z[k] = (float)v;
    }
    double t2 = st ? now_s(CLOCK_MONOTONIC) : 0;
    if (c->nbuf == NPAST) { memmove(c->buf[0], c->buf[1], sizeof(float) * NF * (NPAST - 1)); c->nbuf--; }
    memcpy(c->buf[c->nbuf++], z, sizeof z);
    float X[NPAST * NF] = {0}, Y1[NPAST * H], Y2[NPAST * H];
    memcpy(X + (NPAST - c->nbuf) * NF, c->buf, sizeof(float) * NF * c->nbuf);
    lstm_layer(X, NF, m->Wi0, m->Wh0, m->b0, Y1);
    lstm_layer(Y1, H, m->Wi1, m->Wh1, m->b1, Y2);
    float zo[2];
    for (int r = 0; r < 2; r++) { float s = 0; for (int j = 0; j < H; j++) s += m->Wo[r * H + j] * Y2[(NPAST - 1) * H + j]; zo[r] = s + m->bo[r]; }
    float mx = zo[0] > zo[1] ? zo[0] : zo[1]; float e0 = expf(zo[0] - mx), e1 = expf(zo[1] - mx);
    double p = (double)(e1 / (e0 + e1));
    c->on = c->on ? (p >= m->th_off) : (p >= m->th_on);
    if (st) { double t3 = now_s(CLOCK_MONOTONIC); st[0] = t1 - t0; st[1] = t2 - t1; st[2] = t3 - t2; }
    return p;
}

typedef struct { int n; double *raw; int np; double *p_ref; } Replay;

static Replay load_replay(const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) die("cannot open replay file");
    char magic[4]; rd(f, magic, 1, 4); if (memcmp(magic, "FOGR", 4)) die("bad replay magic");
    Replay r; int32_t n, np; rd(f, &n, 4, 1); rd(f, &np, 4, 1); r.n = n; r.np = np;
    r.raw = malloc(sizeof(double) * n * NAX); rd(f, r.raw, 8, (size_t)n * NAX);
    r.p_ref = malloc(sizeof(double) * (np > 0 ? np : 1)); if (np > 0) rd(f, r.p_ref, 8, np);
    fclose(f); return r;
}

static double peak_rss_mb(void) { struct rusage u; getrusage(RUSAGE_SELF, &u); return u.ru_maxrss / 1024.0; }

static int mode_bench(Model *m, const char *rp, const char *pre, int repeats) {
    Replay r = load_replay(rp); int nw = r.n / W; char fn[512];
    snprintf(fn, sizeof fn, "%s_bench.csv", pre); FILE *o = fopen(fn, "w"); if (!o) die("cannot write output");
    fprintf(o, "pass,window,p,cue_on,p_ref,total_ms,filter_ms,features_ms,lstm_ms\n");
    double c0 = (double)clock() / CLOCKS_PER_SEC, w0 = now_s(CLOCK_MONOTONIC);
    for (int pass = 0; pass < repeats; pass++) {
        Ctrl c; ctrl_init(&c, m);
        for (int k = 0; k < nw; k++) {
            double st[3], t0 = now_s(CLOCK_MONOTONIC); double p = ctrl_step(&c, r.raw + (size_t)k * W * NAX, st); double t1 = now_s(CLOCK_MONOTONIC);
            fprintf(o, "%d,%d,%.12f,%d,%.12f,%.6f,%.6f,%.6f,%.6f\n", pass, k, p, c.on, k < r.np ? r.p_ref[k] : NAN,
                    1e3 * (t1 - t0), 1e3 * st[0], 1e3 * st[1], 1e3 * st[2]);
        }
    }
    fclose(o);
    printf("bench: %d passes x %d windows; wall %.2f s, cpu %.2f s, peak memory %.1f MB -> %s\n", repeats, nw,
           now_s(CLOCK_MONOTONIC) - w0, (double)clock() / CLOCKS_PER_SEC - c0, peak_rss_mb(), fn);
    return 0;
}

static int mode_realtime(Model *m, const char *rp, double minutes, const char *pre) {
    Replay r = load_replay(rp); int nw_rec = r.n / W; int nw = (int)lround(minutes * 60 * FS / W); char fn[512];
    snprintf(fn, sizeof fn, "%s_realtime.csv", pre); FILE *o = fopen(fn, "w"); if (!o) die("cannot write output");
    fprintf(o, "window,p,cue_on,processing_ms,delay_monotonic_ms,delay_boottime_ms\n");
    Ctrl c; ctrl_init(&c, m);
    struct timespec t0; clock_gettime(CLOCK_MONOTONIC, &t0);
    double mono0 = t0.tv_sec + 1e-9 * t0.tv_nsec + 1.0, boot0 = now_s(CLOCK_BOOTTIME) + 1.0, cpu0 = (double)clock() / CLOCKS_PER_SEC;
    for (int k = 0; k < nw; k++) {
        double target = mono0 + (k + 1) * (double)W / FS; struct timespec ts;
        ts.tv_sec = (time_t)target; ts.tv_nsec = (long)((target - ts.tv_sec) * 1e9);
        while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, NULL) == EINTR) {}
        double a = now_s(CLOCK_MONOTONIC);
        if (k % nw_rec == 0) ctrl_init(&c, m);
        double p = ctrl_step(&c, r.raw + (size_t)(k % nw_rec) * W * NAX, NULL);
        double b = now_s(CLOCK_MONOTONIC), bb = now_s(CLOCK_BOOTTIME);
        double btarget = boot0 + (k + 1) * (double)W / FS;
        fprintf(o, "%d,%.12f,%d,%.6f,%.6f,%.6f\n", k, p, c.on, 1e3 * (b - a), 1e3 * (b - target), 1e3 * (bb - btarget));
        if (k % 600 == 0) { printf("window %d/%d  p=%.3f  cue=%s\n", k, nw, p, c.on ? "ON" : "off"); fflush(stdout); }
    }
    fclose(o);
    double wall = now_s(CLOCK_MONOTONIC) - (mono0 - 1.0), cpu = (double)clock() / CLOCKS_PER_SEC - cpu0;
    printf("realtime: %d decisions; wall %.1f s, cpu %.2f s (duty cycle %.2f %%), peak memory %.1f MB -> %s\n",
           nw, wall, cpu, 100 * cpu / wall, peak_rss_mb(), fn);
    snprintf(fn, sizeof fn, "%s_realtime_run.csv", pre); o = fopen(fn, "w");
    fprintf(o, "wall_s,cpu_s,boottime_s,peak_memory_mb\n%.6f,%.6f,%.6f,%.3f\n", wall, cpu, now_s(CLOCK_BOOTTIME) - (boot0 - 1.0), peak_rss_mb());
    fclose(o);
    return 0;
}

static int mode_dataset(Model *m, const char *sp, const char *out) {
    FILE *f = fopen(sp, "rb"); if (!f) die("cannot open streams file");
    char magic[4]; rd(f, magic, 1, 4); if (memcmp(magic, "FOGS", 4)) die("bad streams magic");
    int32_t ns; rd(f, &ns, 4, 1);
    FILE *o = fopen(out, "w"); if (!o) die("cannot write output");
    fprintf(o, "stretch,window,p,cue_on\n");
    long tot = 0;
    for (int s = 0; s < ns; s++) {
        int32_t nw; rd(f, &nw, 4, 1);
        double *raw = malloc(sizeof(double) * (size_t)nw * W * NAX); rd(f, raw, 8, (size_t)nw * W * NAX);
        Ctrl c; ctrl_init(&c, m);
        for (int k = 0; k < nw; k++) { double p = ctrl_step(&c, raw + (size_t)k * W * NAX, NULL); fprintf(o, "%d,%d,%.12f,%d\n", s, k, p, c.on); }
        tot += nw; free(raw);
    }
    fclose(f); fclose(o); printf("dataset: %d stretches, %ld windows -> %s\n", ns, tot, out); return 0;
}

#define PKT_SZ 100
#define REP_SZ 13
static int mode_stream(Model *m, const char *peer, const char *pre) {
    int port = 50007, s;
    if (strcmp(peer, "listen") == 0) {
        int srv = socket(AF_INET, SOCK_STREAM, 0), one = 1; setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
        struct sockaddr_in a = {0}; a.sin_family = AF_INET; a.sin_port = htons(port); a.sin_addr.s_addr = INADDR_ANY;
        if (bind(srv, (struct sockaddr *)&a, sizeof a) < 0 || listen(srv, 1) < 0) die("cannot listen on port 50007");
        printf("waiting on port %d ... (start the sender on the laptop)\n", port); fflush(stdout);
        s = accept(srv, NULL, NULL); close(srv);
    } else {
        struct sockaddr_in a = {0}; a.sin_family = AF_INET; a.sin_port = htons(port);
        if (inet_pton(AF_INET, peer, &a.sin_addr) != 1) die("bad laptop IP address");
        printf("connecting to %s:%d ... (start the sender on the laptop first)\n", peer, port); fflush(stdout);
        for (;;) { s = socket(AF_INET, SOCK_STREAM, 0); if (connect(s, (struct sockaddr *)&a, sizeof a) == 0) break; close(s); sleep(1); }
    }
    int one = 1; setsockopt(s, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    printf("connected\n"); fflush(stdout);
    char fn[512]; snprintf(fn, sizeof fn, "%s_stream.csv", pre); FILE *o = fopen(fn, "w");
    snprintf(fn, sizeof fn, "%s_stream_samples.bin", pre); FILE *sm = fopen(fn, "wb");
    if (!o || !sm) die("cannot write output");
    fprintf(o, "window,p,cue_on,processing_ms,on_phone_delay_ms,arrival_s,boottime_s,seq_errors_so_far\n");
    static unsigned char buf[1 << 17]; size_t have = 0; double win[W * NAX]; int nin = 0, k = 0; uint32_t expect = 0; long seqerr = 0;
    Ctrl c; ctrl_init(&c, m);
    double w0 = now_s(CLOCK_MONOTONIC), b0 = now_s(CLOCK_BOOTTIME), cpu0 = (double)clock() / CLOCKS_PER_SEC;
    for (;;) {
        ssize_t got = recv(s, buf + have, sizeof buf - have, 0);
        if (got <= 0) break;
        double t_arr = now_s(CLOCK_MONOTONIC); have += (size_t)got; size_t off = 0;
        while (have - off >= PKT_SZ) {
            uint32_t seq; memcpy(&seq, buf + off, 4); if (seq != expect) seqerr++; expect = seq + 1;
            memcpy(win + nin * NAX, buf + off + 4, 96); nin += 4; off += PKT_SZ;
            if (nin == W) {
                fwrite(win, sizeof(double), W * NAX, sm);
                double t1 = now_s(CLOCK_MONOTONIC); double p = ctrl_step(&c, win, NULL); double t2 = now_s(CLOCK_MONOTONIC);
                unsigned char rep[REP_SZ]; uint32_t kk = (uint32_t)k; unsigned char on = (unsigned char)c.on;
                memcpy(rep, &kk, 4); rep[4] = on; memcpy(rep + 5, &p, 8);
                if (send(s, rep, REP_SZ, MSG_NOSIGNAL) != REP_SZ) { fprintf(stderr, "send failed at window %d\n", k); }
                fprintf(o, "%d,%.12f,%d,%.6f,%.6f,%.6f,%.6f,%ld\n", k, p, c.on, 1e3 * (t2 - t1), 1e3 * (t2 - t_arr), t_arr - w0,
                        now_s(CLOCK_BOOTTIME) - b0, seqerr);
                if (k % 600 == 0) { printf("window %d  p=%.3f  cue=%s  processing %.2f ms\n", k, p, c.on ? "ON" : "off", 1e3 * (t2 - t1)); fflush(stdout); }
                k++; nin = 0;
            }
        }
        memmove(buf, buf + off, have - off); have -= off;
    }
    double wall = now_s(CLOCK_MONOTONIC) - w0, cpu = (double)clock() / CLOCKS_PER_SEC - cpu0, boot = now_s(CLOCK_BOOTTIME) - b0;
    close(s); fclose(o); fclose(sm);
    snprintf(fn, sizeof fn, "%s_stream_run.csv", pre); o = fopen(fn, "w");
    fprintf(o, "windows,sequence_errors,wall_s,cpu_s,boottime_s,peak_memory_mb\n%d,%ld,%.6f,%.6f,%.6f,%.3f\n", k, seqerr, wall, cpu, boot, peak_rss_mb());
    fclose(o);
    printf("stream ended: %d decisions, %ld sequence errors; cpu duty cycle %.2f %%\n", k, seqerr, 100 * cpu / wall);
    return 0;
}

int main(int argc, char **argv) {
    signal(SIGPIPE, SIG_IGN);
    init_tables();
    if (argc < 2) { fprintf(stderr, "usage: fog_edge_c bench MODEL REPLAY PREFIX [repeats] | realtime MODEL REPLAY MINUTES PREFIX | dataset MODEL STREAMS OUT.csv | stream MODEL (listen|LAPTOP_IP) PREFIX\n"); return 2; }
    if (!strcmp(argv[1], "bench") && argc >= 5) return mode_bench(load_model(argv[2]), argv[3], argv[4], argc > 5 ? atoi(argv[5]) : 1);
    if (!strcmp(argv[1], "realtime") && argc >= 6) return mode_realtime(load_model(argv[2]), argv[3], atof(argv[4]), argv[5]);
    if (!strcmp(argv[1], "dataset") && argc >= 5) return mode_dataset(load_model(argv[2]), argv[3], argv[4]);
    if (!strcmp(argv[1], "stream") && argc >= 5) return mode_stream(load_model(argv[2]), argv[3], argv[4]);
    fprintf(stderr, "usage: fog_edge_c bench MODEL REPLAY PREFIX [repeats] | realtime MODEL REPLAY MINUTES PREFIX | dataset MODEL STREAMS OUT.csv | stream MODEL (listen|LAPTOP_IP) PREFIX\n"); return 2;
}
