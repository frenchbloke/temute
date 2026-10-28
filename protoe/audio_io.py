"""Audio file reading/writing. soundfile when available, scipy fallback for WAV."""
import numpy as np


def load_audio(path):
    """Returns (samples shaped (channels, n) float32, sample_rate)."""
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        return np.ascontiguousarray(data.T), int(sr)
    except ImportError:
        pass
    if not path.lower().endswith((".wav", ".wave")):
        raise RuntimeError("Reading this file type needs the 'soundfile' library.")
    from scipy.io import wavfile
    sr, data = wavfile.read(path)
    if data.dtype.kind == "i":
        data = data.astype(np.float32) / float(np.iinfo(data.dtype).max)
    elif data.dtype.kind == "u":
        data = (data.astype(np.float32) - 128.0) / 128.0
    else:
        data = data.astype(np.float32)
    if data.ndim == 1:
        data = data[:, None]
    return np.ascontiguousarray(data.T), int(sr)


def save_wav(path, audio, sr):
    """audio shaped (channels, n). Written as 24-bit PCM, which every player and DAW reads."""
    data = np.ascontiguousarray(np.clip(np.asarray(audio, dtype=np.float64), -1.0, 1.0).T)
    try:
        import soundfile as sf
        sf.write(path, data, sr, subtype="PCM_24")
    except ImportError:
        from scipy.io import wavfile
        wavfile.write(path, sr, np.rint(data * 2147483647.0).astype(np.int32))
