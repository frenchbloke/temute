"""
Small, dependency-light DSP toolkit (numpy + scipy only).

Replaces librosa/numba so the app installs in seconds and can be frozen into a
single double-click program with no Python on the tester's machine.
"""
import numpy as np
from scipy.signal import fftconvolve, find_peaks


# ------------------------------------------------------------------- STFT --

def _hann(n):
    return np.hanning(n + 1)[:-1]          # periodic Hann


def stft(x, n_fft=2048, hop=512):
    pad = n_fft // 2
    xp = np.pad(np.asarray(x, dtype=np.float64), pad, mode="reflect")
    n = 1 + (len(xp) - n_fft) // hop
    idx = np.arange(n)[:, None] * hop + np.arange(n_fft)[None, :]
    return np.fft.rfft(xp[idx] * _hann(n_fft), axis=1).T      # (bins, frames)


def istft(D, n_fft=2048, hop=512, length=None):
    win = _hann(n_fft)
    frames = np.fft.irfft(D.T, n=n_fft, axis=1) * win
    n = frames.shape[0]
    total = n_fft + hop * (n - 1)
    y = np.zeros(total)
    wsum = np.zeros(total)
    for i in range(n):
        y[i * hop:i * hop + n_fft] += frames[i]
        wsum[i * hop:i * hop + n_fft] += win ** 2
    y = np.where(wsum > 1e-8, y / np.maximum(wsum, 1e-8), 0.0)
    y = y[n_fft // 2:]
    if length is not None:
        y = y[:length] if len(y) >= length else np.pad(y, (0, length - len(y)))
    return y


def time_stretch(x, rate, n_fft=2048, hop=512):
    """Phase-vocoder stretch. rate > 1 = faster/shorter, rate < 1 = slower/longer."""
    x = np.asarray(x, dtype=np.float64)
    out_len = int(round(len(x) / rate))
    D = stft(x, n_fft, hop)
    n_bins, n_frames = D.shape
    steps = np.arange(0, n_frames, rate)
    omega = 2 * np.pi * hop * np.arange(n_bins) / n_fft
    Dp = np.pad(D, ((0, 0), (0, 2)))
    phase = np.angle(Dp[:, 0])
    out = np.empty((n_bins, len(steps)), dtype=np.complex128)
    for i, step in enumerate(steps):
        k = int(step)
        a = step - k
        c0, c1 = Dp[:, k], Dp[:, k + 1]
        mag = (1 - a) * np.abs(c0) + a * np.abs(c1)
        out[:, i] = mag * np.exp(1j * phase)
        dphi = np.angle(c1) - np.angle(c0) - omega
        dphi -= 2 * np.pi * np.round(dphi / (2 * np.pi))
        phase += omega + dphi
    return istft(out, n_fft, hop, length=out_len)


# ------------------------------------------------------- onset analysis --

def _band_filterbank(freqs, n_bands, fmin, fmax):
    edges = np.geomspace(fmin, fmax, n_bands + 2)
    fb = np.zeros((n_bands, len(freqs)), dtype=np.float32)
    for i in range(n_bands):
        lo, c, hi = edges[i:i + 3]
        up = (freqs - lo) / (c - lo)
        dn = (hi - freqs) / (hi - c)
        fb[i] = np.clip(np.minimum(up, dn), 0, None)
        if fb[i].sum() == 0:
            fb[i, np.argmin(np.abs(freqs - c))] = 1.0
        fb[i] /= fb[i].sum()
    return fb


def analyse(x, sr, hop=64, n_fft=1024, n_bands=40, chunk=4096):
    """
    One streaming pass over the audio. Returns
      env : spectral-flux onset envelope at `hop` resolution
      low : positive flux of the <150 Hz band (kick detector)
    Chunked so long files do not need gigabytes of RAM.
    """
    x = np.asarray(x, dtype=np.float32)
    peak = float(np.abs(x).max()) if len(x) else 0.0
    if peak > 0:
        x = x / peak
    pad = n_fft // 2
    xp = np.pad(x, (pad, pad + hop), mode="reflect")
    n_frames = 1 + len(x) // hop
    win = _hann(n_fft).astype(np.float32)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    fb = _band_filterbank(freqs, n_bands, 40.0, min(16000.0, sr / 2 * 0.98))
    low_bins = freqs < 150

    env = np.zeros(n_frames, dtype=np.float32)
    low = np.zeros(n_frames, dtype=np.float32)
    prev = prev_low = None
    for s in range(0, n_frames, chunk):
        e = min(s + chunk, n_frames)
        idx = (np.arange(s, e) * hop)[:, None] + np.arange(n_fft)[None, :]
        mag = np.abs(np.fft.rfft(xp[idx] * win, axis=1))
        lm = 10.0 * np.log10((mag ** 2) @ fb.T + 1e-10)
        lm = np.maximum(lm, -80.0)
        lo_e = mag[:, low_bins].sum(axis=1)
        d = np.diff(lm, axis=0, prepend=(lm[:1] if prev is None else prev[None, :]))
        env[s:e] = np.maximum(d, 0).mean(axis=1)
        dl = np.diff(lo_e, prepend=(lo_e[0] if prev_low is None else prev_low))
        low[s:e] = np.maximum(dl, 0)
        prev, prev_low = lm[-1], lo_e[-1]
    return env, low


def pick_onsets(env, sr, hop, prominence=0.07, min_gap=0.03):
    """Onset times (backtracked to the start of the attack) and strengths 0..1."""
    e = np.asarray(env, dtype=np.float64)
    if e.max() <= 0:
        return np.array([]), np.array([])
    e = e / e.max()
    peaks, _ = find_peaks(e, prominence=prominence,
                          distance=max(1, int(min_gap * sr / hop)))
    if len(peaks) == 0:
        return np.array([]), np.array([])
    mins = np.flatnonzero((e[1:-1] <= e[:-2]) & (e[1:-1] < e[2:])) + 1
    back = np.empty(len(peaks), dtype=int)
    for n, p in enumerate(peaks):
        j = np.searchsorted(mins, p, side="right") - 1
        back[n] = mins[j] if j >= 0 else 0
    t = back * hop / sr
    s = e[peaks]
    order = np.argsort(t)
    return t[order], s[order]


# ----------------------------------------------------------- beat tracking --

def estimate_period(env, fps, bpm_lo=60.0, bpm_hi=200.0, prior=120.0):
    """Beat period in frames, from autocorrelation with a tempo prior."""
    e = np.asarray(env, dtype=np.float64)
    e = e - e.mean()
    n = len(e)
    f = np.fft.rfft(e, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    if ac[0] <= 0:
        raise RuntimeError("No rhythmic content found.")
    ac = ac / ac[0]
    lags = np.arange(int(fps * 60 / bpm_hi), min(int(fps * 60 / bpm_lo) + 1, n // 2))
    bpm = 60.0 * fps / lags
    prior_w = np.exp(-0.5 * (np.log2(bpm / prior)) ** 2)
    score = (ac[lags] + 0.5 * ac[np.minimum(2 * lags, n - 1)]) * prior_w
    best = int(lags[np.argmax(score)])
    # parabolic refinement of the autocorrelation peak
    if 0 < best < n - 1:
        a, b, c = ac[best - 1], ac[best], ac[best + 1]
        den = a - 2 * b + c
        if den != 0:
            return best + 0.5 * (a - c) / den
    return float(best)


def beat_track(env, fps, tightness=100.0):
    """Dynamic-programming beat tracker (Ellis 2007). Returns (frames, bpm)."""
    e = np.asarray(env, dtype=np.float64)
    period = estimate_period(e, fps)
    e = e / (e.std() + 1e-12)
    w = int(round(period))
    win = np.exp(-0.5 * (np.arange(-w, w + 1) * 32.0 / period) ** 2)
    local = np.convolve(e, win, mode="same")
    n = len(local)
    cum = local.copy()
    back = np.full(n, -1, dtype=int)
    lo_off, hi_off = int(round(2 * period)), int(round(period / 2))
    for t in range(n):
        hi = t - hi_off
        if hi < 0:
            continue
        lo = max(t - lo_off, 0)
        idx = np.arange(lo, hi + 1)
        pen = -tightness * np.log((t - idx) / period) ** 2
        cand = cum[idx] + pen
        k = int(np.argmax(cand))
        if cand[k] > 0:
            cum[t] = local[t] + cand[k]
            back[t] = idx[k]
    t = int(np.argmax(cum))
    beats = [t]
    while back[t] >= 0:
        t = back[t]
        beats.append(t)
    return np.array(beats[::-1]), 60.0 * fps / period


# ------------------------------------------------------------------ WSOLA --

def wsola(x, out_len, knots=None):
    """
    Time-domain stretch (waveform-similarity overlap-add).

    x        : (channels, n) float array
    out_len  : exact output length in samples
    knots    : optional (out_positions, in_positions) piecewise-linear time map,
               both ascending, from (0, 0) to (out_len, n)

    Nothing is resampled, so pitch AND formants stay exactly where they were.
    All channels share the same splice points, which keeps the stereo image
    stable. Frames are joined where the waveforms already agree, which avoids
    the 'phasey' sound and the pre-echo of a phase vocoder.
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    ch, n = x.shape
    out_len = int(out_len)
    if out_len <= 0:
        return np.zeros((ch, 0))
    if n < 64 or out_len < 64:
        idx = np.linspace(0, max(n - 1, 0), out_len)
        return np.vstack([np.interp(idx, np.arange(n), x[c]) for c in range(ch)]) if n else np.zeros((ch, out_len))
    if knots is None and abs(out_len - n) <= 1:
        return x[:, :out_len] if n >= out_len else np.pad(x, ((0, 0), (0, out_len - n)))

    m = min(n, out_len)
    N = 2048
    while N > 256 and N * 3 > m:
        N //= 2
    N = min(N, (m // 2) * 2)
    half = N // 2
    M = max(2, int(np.ceil((out_len - N) / half)) + 1)
    o = np.rint(np.linspace(0, out_len - N, M)).astype(int)
    if knots is None:
        ko, ki = np.array([0.0, out_len]), np.array([0.0, float(n)])
    else:
        ko, ki = np.asarray(knots[0], dtype=float), np.asarray(knots[1], dtype=float)
    p = np.rint(np.interp(o + half, ko, ki) - half).astype(int)
    p = np.clip(p, 0, n - N)
    p[0], p[-1] = 0, n - N
    D = max(96, N // 4)

    mono = x.mean(axis=0)
    cs2 = np.concatenate([[0.0], np.cumsum(mono ** 2)])
    win = _hann(N)
    first, last = win.copy(), win.copy()
    first[:half] = 1.0
    last[half:] = 1.0

    starts = np.zeros(M, dtype=int)
    for k in range(1, M):
        hop = o[k] - o[k - 1]
        T = N - hop
        if T < 16:
            starts[k] = p[k]
            continue
        t0 = starts[k - 1] + hop
        if t0 + T > n:
            t0 = n - T
        tpl = mono[t0:t0 + T]
        e_t = cs2[t0 + T] - cs2[t0]
        lo, hi = max(0, p[k] - D), min(n - N, p[k] + D)
        if hi < lo:
            starts[k] = int(np.clip(p[k], 0, n - N))
            continue
        corr = fftconvolve(mono[lo:hi + T], tpl[::-1], mode="valid")
        e_c = cs2[lo + T:hi + T + 1] - cs2[lo:hi + 1]
        score = corr / np.sqrt(e_c * e_t + 1e-12)
        starts[k] = lo + int(np.argmax(score))

    out = np.zeros((ch, out_len))
    wsum = np.zeros(out_len)
    for k in range(M):
        w = first if k == 0 else (last if k == M - 1 else win)
        out[:, o[k]:o[k] + N] += x[:, starts[k]:starts[k] + N] * w
        wsum[o[k]:o[k] + N] += w
    return out / np.maximum(wsum, 1e-3)


# ------------------------------------------- harmonic / percussive split --

def hpss(x, n_fft=2048, hop=512, k_time=31, k_freq=17):
    """
    Split into tonal ('harmonic') and hit ('percussive') parts that add back up
    to the original. Median filtering along time keeps steady tones; along
    frequency keeps clicks and hits (Fitzgerald 2010). x: 1-D.
    """
    from scipy.ndimage import median_filter
    x = np.asarray(x, dtype=np.float64)
    D = stft(x, n_fft, hop)
    S = np.abs(D)
    Hm = median_filter(S, size=(1, k_time), mode="reflect")
    Pm = median_filter(S, size=(k_freq, 1), mode="reflect")
    mask = Hm ** 2 / (Hm ** 2 + Pm ** 2 + 1e-12)
    return (istft(D * mask, n_fft, hop, length=len(x)),
            istft(D * (1.0 - mask), n_fft, hop, length=len(x)))


# --------------------------------------- phase vocoder, phase-locked ------

def pv_lock(x, out_len, knots=None, n_fft=4096):
    """
    Phase vocoder with identity phase locking (Laroche & Dolson 1999): each
    spectral peak keeps its own phase track and the bins around it keep their
    phase relationship to it. That removes the 'phasey' / reverberant sound of
    a plain phase vocoder on tonal material. x: 1-D, returns 1-D of out_len.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    out_len = int(out_len)
    hs = n_fft // 4
    D = stft(x, n_fft, hs)
    K, F = D.shape
    mag_all, ang_all = np.abs(D), np.angle(D)
    n_out = int(np.ceil(out_len / hs)) + 2
    if knots is None:
        ko, ki = np.array([0.0, out_len]), np.array([0.0, float(n)])
    else:
        ko, ki = np.asarray(knots[0], float), np.asarray(knots[1], float)
    pos = np.interp(np.arange(n_out) * hs, ko, ki) / hs
    pos = np.clip(pos, 0, F - 1.001)
    omega = 2 * np.pi * hs * np.arange(K) / n_fft
    ks = np.arange(K)
    out = np.empty((K, n_out), dtype=np.complex128)
    phase = ang_all[:, int(pos[0])].copy()
    for m in range(n_out):
        f = pos[m]
        i = int(f)
        a = f - i
        mag = (1 - a) * mag_all[:, i] + a * mag_all[:, i + 1]
        near = i + (1 if a >= 0.5 else 0)
        ang = ang_all[:, near]
        if m > 0:
            dphi = ang_all[:, i + 1] - ang_all[:, i] - omega
            dphi -= 2 * np.pi * np.round(dphi / (2 * np.pi))
            inst = omega + dphi                       # phase advance per hop at each bin
            pk = np.flatnonzero((mag[1:-1] > mag[:-2]) & (mag[1:-1] >= mag[2:])) + 1
            if len(pk) == 0:
                phase = phase + inst
            else:
                j = np.searchsorted(pk, ks)
                left = pk[np.clip(j - 1, 0, len(pk) - 1)]
                right = pk[np.clip(j, 0, len(pk) - 1)]
                owner = np.where(np.abs(ks - left) <= np.abs(ks - right), left, right)
                new_pk = phase[pk] + inst[pk]
                lut = np.zeros(K)
                lut[pk] = new_pk
                ang_pk = np.zeros(K)
                ang_pk[pk] = ang[pk]
                phase = lut[owner] + (ang - ang_pk[owner])
        else:
            phase = ang.copy()
        out[:, m] = mag * np.exp(1j * phase)
    return istft(out, n_fft, hs, length=out_len)


# ------------------------------------------------- continuous phase vocoder --

def pv_warp(x, out_len, ko, ki, zones=(), N=2048):
    """
    Continuous time-warp with a phase vocoder, built for beat-matching.

    x      : (channels, n) audio
    ko, ki : ascending knots (samples) mapping OUTPUT time -> INPUT time,
             piecewise linear, slope 1 beyond the ends
    zones  : (out_start, out_end) output-time intervals where the map has slope 1.
             Frames there are copied with their original phases, so a drum hit is
             reproduced exactly (no smearing, no pre-echo). After a zone the
             phases carry on from the original, so nothing is ever cut or
             crossfaded and sustained notes keep their phase.

    Improvements over a plain phase vocoder:
      * identity phase locking (partials keep their relationship to the peak)
      * all channels share ONE phase track (from the mid signal) and keep their
        original inter-channel phase differences, so the stereo image holds
      * no resampling: pitch and formants are untouched
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    ch, n = x.shape
    out_len = int(out_len)
    Hs = N // 4
    win = _hann(N)
    ko = np.asarray(ko, dtype=float)
    ki = np.asarray(ki, dtype=float)
    K = int(np.ceil((out_len + N) / Hs)) + 2
    c = -N // 2 + np.arange(K) * Hs
    a = np.interp(c, ko, ki)
    a = np.where(c < ko[0], c + (ki[0] - ko[0]), a)
    a = np.where(c > ko[-1], c + (ki[-1] - ko[-1]), a)
    a_int = np.rint(a).astype(np.int64)
    ident = np.zeros(K, dtype=bool)
    for z0, z1 in zones:
        ident |= (c >= z0) & (c <= z1)

    pad_in = N + 8
    xp = np.pad(x, ((0, 0), (pad_in, pad_in)))
    omega = 2 * np.pi * np.arange(N // 2 + 1) / N
    bins = np.arange(N // 2 + 1)
    w2 = win ** 2
    out = np.zeros((ch, out_len + 3 * N))
    wsum = np.zeros(out_len + 3 * N)
    base = N                                   # output index of c == 0 is `base`
    phi_prev = None
    ang_prev = None
    a_prev = None
    BLOCK = 256
    for b0 in range(0, K, BLOCK):
        b1 = min(b0 + BLOCK, K)
        idx = (a_int[b0:b1, None] - N // 2 + pad_in) + np.arange(N)[None, :]
        idx = np.clip(idx, 0, xp.shape[1] - 1)
        X = np.fft.rfft(xp[:, idx] * win, axis=2)              # (ch, blk, bins)
        M = X.mean(axis=0)
        for j in range(b1 - b0):
            k = b0 + j
            m = M[j]
            ang = np.angle(m)
            if phi_prev is None or ident[k]:
                phi = ang
            else:
                Ha = max(a[k] - a_prev, 1.0)
                d = ang - ang_prev - omega * Ha
                d -= 2 * np.pi * np.round(d / (2 * np.pi))
                w_inst = omega + d / Ha
                mag = np.abs(m)
                pk = np.flatnonzero((mag[1:-1] > mag[:-2]) & (mag[1:-1] >= mag[2:]) &
                                    (mag[1:-1] > 1e-6 * (mag.max() + 1e-12))) + 1
                if len(pk) == 0:
                    phi = phi_prev + w_inst * Hs
                else:
                    new_pk = phi_prev[pk] + w_inst[pk] * Hs
                    owner = np.searchsorted((pk[:-1] + pk[1:]) / 2.0, bins)
                    phi = new_pk[owner] + (ang - ang[pk][owner])
            phi_prev, ang_prev, a_prev = phi, ang, a[k]
            rot = np.exp(1j * (phi - ang))                       # (bins,)
            Y = X[:, j, :] * rot                                  # keeps channel phase offsets
            fr = np.fft.irfft(Y, n=N, axis=1) * win
            s = base + int(c[k]) - N // 2
            lo, hi = max(s, 0), min(s + N, out.shape[1])
            if hi > lo:
                out[:, lo:hi] += fr[:, lo - s:hi - s]
                wsum[lo:hi] += w2[lo - s:hi - s]
    res = out[:, base:base + out_len] / np.maximum(wsum[base:base + out_len], 1e-3)
    return res
