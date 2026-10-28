"""Granular cloud + paulx (transient-preserving) stretch engines."""
import numpy as np


def granular_cloud_channel(x, stretch, grain=512, jitter=0.0):
    """Granular cloud: dense Hann grains scattered from a scrub position."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    out_len = int(round(n * stretch)) if stretch > 1 else n
    if n < grain or out_len < grain:
        idx = np.linspace(0, max(n-1,0), out_len)
        return np.interp(idx, np.arange(n), x) if n else np.zeros(out_len)
    win = 0.5 * (1 - np.cos(2*np.pi*np.arange(grain)/grain))
    out = np.zeros(out_len + grain, dtype=np.float64)
    wsum = np.zeros(out_len + grain, dtype=np.float64)
    rng = np.random.default_rng(1)
    hop_out = max(1, grain // 4)
    scrub = 0.0
    scrub_rate = 1.0 / stretch  # source advances slowly
    pos_out = 0
    while pos_out + grain < len(out):
        # scrub position in source
        si = int(scrub) % n
        # jitter
        j = int(rng.uniform(-grain*0.1, grain*0.1)) if jitter else 0
        si = int(np.clip(si + j, 0, n - grain))
        grain_data = x[si:si+grain] * win
        out[pos_out:pos_out+grain] += grain_data
        wsum[pos_out:pos_out+grain] += win
        pos_out += hop_out
        scrub += scrub_rate * hop_out
    out = out[:out_len]
    wsum = wsum[:out_len]
    out = np.where(wsum > 1e-8, out / np.maximum(wsum, 1e-8), 0.0)
    return np.clip(out, -0.99, 0.99)


def granular_cloud(samples, stretch, grain=512):
    num_ch = samples.shape[0]
    out_len = int(round(samples.shape[1] * stretch))
    out = np.zeros((num_ch, out_len), dtype=np.float64)
    for c in range(num_ch):
        out[c] = granular_cloud_channel(samples[c], stretch, grain=grain)
    return out


def paulx_channel(x, stretch, n_fft=1024):
    """PaulStretch with transient preservation: keep original phase at onsets."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    if n < 64 or stretch <= 1.0:
        out_len = int(round(n * max(1.0, stretch)))
        if out_len == n:
            return x.copy()
        idx = np.linspace(0, max(n-1,0), out_len)
        return np.interp(idx, np.arange(n), x) if n else np.zeros(out_len)
    out_len = int(round(n * stretch))
    win = 0.5 * (1 - np.cos(2*np.pi*np.arange(n_fft)/n_fft))
    pad = n_fft // 2
    xp = np.pad(x, (pad, pad + n_fft), mode="reflect")
    hop_in = n_fft // 4
    hop_out = int(hop_in * stretch)
    out = np.zeros(out_len + n_fft, dtype=np.float64)
    wsum = np.zeros(out_len + n_fft, dtype=np.float64)
    rng = np.random.default_rng(0)
    # onset detection: high energy frames
    frame_energies = []
    pos = 0
    while pos + n_fft < len(xp):
        frame_energies.append(np.sum(xp[pos:pos+n_fft]**2))
        pos += hop_in
    fe = np.array(frame_energies)
    # onset if energy jumps
    onset_thresh = np.median(fe) * 1.5 if len(fe) else 0
    pos_in = 0
    pos_out = 0
    fi = 0
    while pos_out + n_fft < len(out) and pos_in + n_fft < len(xp):
        frame = xp[pos_in:pos_in+n_fft] * win
        spec = np.fft.rfft(frame)
        mag = np.abs(spec)
        is_onset = fi < len(fe) and fe[fi] > onset_thresh
        if is_onset:
            phase = np.angle(spec)  # preserve original phase
        else:
            phase = rng.uniform(0, 2*np.pi, size=mag.shape)
        spec2 = mag * np.exp(1j*phase)
        frame_out = np.fft.irfft(spec2, n=n_fft) * win
        out[pos_out:pos_out+n_fft] += frame_out
        wsum[pos_out:pos_out+n_fft] += win*win
        pos_in += hop_in
        pos_out += hop_out
        fi += 1
    out = out[:out_len]
    wsum = wsum[:out_len]
    out = np.where(wsum > 1e-8, out / np.maximum(wsum, 1e-8), 0.0)
    return np.clip(out, -0.99, 0.99)


def paulx(samples, stretch, n_fft=1024):
    num_ch = samples.shape[0]
    out_len = int(round(samples.shape[1] * stretch))
    out = np.zeros((num_ch, out_len), dtype=np.float64)
    for c in range(num_ch):
        out[c] = paulx_channel(samples[c], stretch, n_fft=n_fft)
    return out
