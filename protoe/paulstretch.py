"""PaulStretch-style extreme stretching (smooth, artifact-free)."""
import numpy as np


def paulstretch_channel(x, stretch, n_fft=4096):
    """Stretch a 1D signal by `stretch` (>=1). Returns 1D array."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    if n < 64 or stretch <= 1.0:
        # no stretch: pad/truncate
        out_len = int(round(n * stretch)) if stretch > 1 else n
        if out_len == n:
            return x.copy()
        # linear interp
        idx = np.linspace(0, max(n-1,0), out_len)
        return np.interp(idx, np.arange(n), x) if n else np.zeros(out_len)
    out_len = int(round(n * stretch))
    win = np.hanning(n_fft + 1)[:-1] if False else None
    # periodic hann
    win = 0.5 * (1 - np.cos(2*np.pi*np.arange(n_fft)/n_fft))
    # pad input
    pad = n_fft // 2
    xp = np.pad(x, (pad, pad + n_fft), mode="reflect")
    n_frames = 1 + (len(xp) - n_fft) // (n_fft // 4)
    hop_in = n_fft // 4
    hop_out = int(hop_in * stretch)
    out = np.zeros(out_len + n_fft, dtype=np.float64)
    wsum = np.zeros(out_len + n_fft, dtype=np.float64)
    # random phase per frame (deterministic seed for reproducibility)
    rng = np.random.default_rng(0)
    frame_idx = 0
    pos_in = 0
    pos_out = 0
    while pos_out + n_fft < len(out) and pos_in + n_fft < len(xp):
        frame = xp[pos_in:pos_in+n_fft] * win
        spec = np.fft.rfft(frame)
        mag = np.abs(spec)
        # random phase
        phase = rng.uniform(0, 2*np.pi, size=mag.shape)
        spec2 = mag * np.exp(1j*phase)
        frame_out = np.fft.irfft(spec2, n=n_fft) * win
        out[pos_out:pos_out+n_fft] += frame_out
        wsum[pos_out:pos_out+n_fft] += win*win
        pos_in += hop_in
        pos_out += hop_out
        frame_idx += 1
    # normalize
    out = out[:out_len+n_fft]
    wsum = wsum[:out_len+n_fft]
    out = np.where(wsum > 1e-8, out / np.maximum(wsum, 1e-8), 0.0)
    return out[:out_len]


def paulstretch(samples, stretch, n_fft=4096):
    """samples: (channels, n). Returns (channels, out_len)."""
    num_ch = samples.shape[0]
    n = samples.shape[1]
    out_len = int(round(n * stretch))
    out = np.zeros((num_ch, out_len), dtype=np.float64)
    for c in range(num_ch):
        out[c] = paulstretch_channel(samples[c], stretch, n_fft=n_fft)
    out = np.clip(out, -0.99, 0.99)
    return out
