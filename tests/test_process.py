import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from protoe.process import process_audio


def make_click_track(bpm=120, bars=2, sr=22050):
    beat = 60.0 / bpm
    six = beat / 4
    total = bars * 16 * six
    n = int(total * sr)
    x = np.zeros((1, n), dtype=np.float64)
    for b in range(bars):
        for s in range(16):
            t0 = (b * 16 + s) * six
            i0 = int(t0 * sr)
            for j in range(int(0.01 * sr)):
                if i0 + j < n:
                    x[0, i0 + j] = 0.8 * np.exp(-j / (0.002 * sr))
    return x, sr


def test_fill_mode_runs():
    x, sr = make_click_track()
    out, sr2, bpm, off = process_audio(x, sr, bpm=120, offset=0.0, mode="fill")
    assert sr2 == sr
    assert out.shape[0] == 1
    assert out.shape[1] > 0


def test_anchor_mode_runs():
    x, sr = make_click_track()
    out, sr2, bpm, off = process_audio(x, sr, bpm=120, offset=0.0, mode="anchor")
    assert out.shape[1] > 0


if __name__ == "__main__":
    test_fill_mode_runs()
    test_anchor_mode_runs()
    print("OK")
