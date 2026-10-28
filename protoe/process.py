"""Core beat-slice + stretch pipeline."""
import numpy as np
from .formant_pv import stretch_block
from .paulstretch import paulstretch
from .audio_io import load_audio, save_wav
from .analyze import detect_bpm_and_offset


def _concat_crossfade(segments, fade):
    if not segments:
        return np.zeros(0, dtype=np.float64)
    if len(segments) == 1:
        return segments[0].astype(np.float64).copy()
    fade = min(fade, min(len(s) for s in segments) // 2, min(len(s) for s in segments[1:]) // 2)
    if fade < 1:
        return np.concatenate(segments).astype(np.float64)
    total = sum(len(s) for s in segments) - fade * (len(segments) - 1)
    out = np.zeros(total, dtype=np.float64)
    pos = 0
    out[:len(segments[0])] = segments[0]
    pos = len(segments[0])
    for i in range(1, len(segments)):
        seg = segments[i]
        # crossfade overlap
        for f in range(fade):
            t = (f + 1) / (fade + 1)
            out[pos - fade + f] = (1 - t) * out[pos - fade + f] + t * seg[f]
        tail = seg[fade:]
        out[pos:pos + len(tail)] = tail
        pos += len(tail)
    return out


def process_audio(samples, sr, bpm=None, offset=None, mode="fill", perm=None, engine="formant"):
    if bpm is None or offset is None:
        bpm, offset = detect_bpm_and_offset(samples, sr)
    if perm is None:
        perm = (0, 1, 2, 3)

    num_ch = samples.shape[0]
    n = samples.shape[1]

    beat_sec = 60.0 / bpm
    sixteenth_sec = beat_sec / 4.0
    slice_samples = max(64, int(round(sixteenth_sec * sr)))
    start_offset = min(max(0, int(round(offset * sr))), n - slice_samples)
    usable = n - start_offset
    total_slices = usable // slice_samples

    fade = 8  # small crossfade to avoid clicks without tempo change

    out_ch = [[] for _ in range(num_ch)]  # each: list of 1D segment arrays

    slices_per_bar = 16
    num_bars = total_slices // slices_per_bar
    for bar in range(num_bars):
        base = bar * slices_per_bar
        for g in range(4):
            group_start = base + g * 4
            anchor_global = group_start

            if mode == "anchor":
                src_idx = anchor_global
                ratio = 4
                src_start = start_offset + src_idx * slice_samples
                src_len = slice_samples
                target_len = src_len * ratio
                if engine == "paul":
                    src_slice_arr = np.zeros((num_ch, src_len), dtype=np.float64)
                    for c in range(num_ch):
                        src_slice_arr[c] = samples[c, src_start:src_start + src_len].astype(np.float64)
                    stretched = paulstretch(src_slice_arr, target_len / src_len, n_fft=1024)
                elif engine == "paulx":
                    src_slice_arr = np.zeros((num_ch, src_len), dtype=np.float64)
                    for c in range(num_ch):
                        src_slice_arr[c] = samples[c, src_start:src_start + src_len].astype(np.float64)
                    stretched = paulx(src_slice_arr, target_len / src_len, n_fft=1024)
                elif engine == "granular":
                    src_slice_arr = np.zeros((num_ch, src_len), dtype=np.float64)
                    for c in range(num_ch):
                        src_slice_arr[c] = samples[c, src_start:src_start + src_len].astype(np.float64)
                    stretched = granular_cloud(src_slice_arr, target_len / src_len, grain=256)
                else:
                    stretched_list = stretch_block([samples[c] for c in range(num_ch)], src_start, src_len, target_len, 0.0)
                    stretched = np.asarray(stretched_list, dtype=np.float64)
                if stretched.ndim == 1:
                    stretched = stretched[None, :]
                for c in range(num_ch):
                    seg = stretched[c].astype(np.float64).copy()
                    out_ch[c].append(seg)

            elif mode in ("fill2a", "fill2b"):
                # 2x stretch over two slices, keep one original slice
                if mode == "fill2a":
                    # stretch slices 2,3 (0-based 1,2); keep slice 4 (index 3)
                    src_idx = group_start + 1
                    keep_idx = group_start + 3
                else:
                    # stretch slices 3,4 (0-based 2,3); keep slice 2 (index 1)
                    src_idx = group_start + 2
                    keep_idx = group_start + 1
                ratio = 2
                src_start = start_offset + src_idx * slice_samples
                src_len = slice_samples
                target_len = src_len * ratio
                if engine == "paul":
                    src_slice_arr = np.zeros((num_ch, src_len), dtype=np.float64)
                    for c in range(num_ch):
                        src_slice_arr[c] = samples[c, src_start:src_start + src_len].astype(np.float64)
                    stretched = paulstretch(src_slice_arr, target_len / src_len, n_fft=1024)
                elif engine == "paulx":
                    src_slice_arr = np.zeros((num_ch, src_len), dtype=np.float64)
                    for c in range(num_ch):
                        src_slice_arr[c] = samples[c, src_start:src_start + src_len].astype(np.float64)
                    stretched = paulx(src_slice_arr, target_len / src_len, n_fft=1024)
                elif engine == "granular":
                    src_slice_arr = np.zeros((num_ch, src_len), dtype=np.float64)
                    for c in range(num_ch):
                        src_slice_arr[c] = samples[c, src_start:src_start + src_len].astype(np.float64)
                    stretched = granular_cloud(src_slice_arr, target_len / src_len, grain=256)
                else:
                    stretched_list = stretch_block([samples[c] for c in range(num_ch)], src_start, src_len, target_len, 0.0)
                    stretched = np.asarray(stretched_list, dtype=np.float64)
                if stretched.ndim == 1:
                    stretched = stretched[None, :]
                # anchor (beat 1) kept
                anc_start = start_offset + anchor_global * slice_samples
                if mode == "fill2a":
                    for c in range(num_ch):
                        seg = samples[c, anc_start:anc_start + slice_samples].astype(np.float64).copy()
                        out_ch[c].append(seg)
                    for c in range(num_ch):
                        seg = stretched[c].astype(np.float64).copy()
                        out_ch[c].append(seg)
                    # kept slice 4
                    ks = start_offset + keep_idx * slice_samples
                    for c in range(num_ch):
                        seg = samples[c, ks:ks + slice_samples].astype(np.float64).copy()
                        out_ch[c].append(seg)
                else:
                    # keep slice 2 original, then stretched
                    ks = start_offset + keep_idx * slice_samples
                    for c in range(num_ch):
                        seg = samples[c, anc_start:anc_start + slice_samples].astype(np.float64).copy()
                        out_ch[c].append(seg)
                    for c in range(num_ch):
                        seg = samples[c, ks:ks + slice_samples].astype(np.float64).copy()
                        out_ch[c].append(seg)
                    for c in range(num_ch):
                        seg = stretched[c].astype(np.float64).copy()
                        out_ch[c].append(seg)

            elif mode == "shuffle":
                for pi in perm:
                    si = group_start + pi
                    ss = start_offset + si * slice_samples
                    for c in range(num_ch):
                        seg = samples[c, ss:ss + slice_samples].astype(np.float64).copy()
                        out_ch[c].append(seg)

            elif mode == "reverse4":
                for pi in (0, 1, 2):
                    si = group_start + pi
                    ss = start_offset + si * slice_samples
                    for c in range(num_ch):
                        seg = samples[c, ss:ss + slice_samples].astype(np.float64).copy()
                        out_ch[c].append(seg)
                # reverse slice 3 (beat 4,8,12,16)
                si = group_start + 3
                ss = start_offset + si * slice_samples
                for c in range(num_ch):
                    seg = samples[c, ss:ss + slice_samples].astype(np.float64).copy()
                    seg = seg[::-1].copy()
                    out_ch[c].append(seg)

            else:
                # fill (3x): anchor + stretched 3
                src_idx = group_start + 1
                ratio = 3
                src_start = start_offset + src_idx * slice_samples
                src_len = slice_samples
                target_len = src_len * ratio
                if engine == "paul":
                    src_slice_arr = np.zeros((num_ch, src_len), dtype=np.float64)
                    for c in range(num_ch):
                        src_slice_arr[c] = samples[c, src_start:src_start + src_len].astype(np.float64)
                    stretched = paulstretch(src_slice_arr, target_len / src_len, n_fft=1024)
                elif engine == "paulx":
                    src_slice_arr = np.zeros((num_ch, src_len), dtype=np.float64)
                    for c in range(num_ch):
                        src_slice_arr[c] = samples[c, src_start:src_start + src_len].astype(np.float64)
                    stretched = paulx(src_slice_arr, target_len / src_len, n_fft=1024)
                elif engine == "granular":
                    src_slice_arr = np.zeros((num_ch, src_len), dtype=np.float64)
                    for c in range(num_ch):
                        src_slice_arr[c] = samples[c, src_start:src_start + src_len].astype(np.float64)
                    stretched = granular_cloud(src_slice_arr, target_len / src_len, grain=256)
                else:
                    stretched_list = stretch_block([samples[c] for c in range(num_ch)], src_start, src_len, target_len, 0.0)
                    stretched = np.asarray(stretched_list, dtype=np.float64)
                if stretched.ndim == 1:
                    stretched = stretched[None, :]
                anc_start = start_offset + anchor_global * slice_samples
                for c in range(num_ch):
                    seg = samples[c, anc_start:anc_start + slice_samples].astype(np.float64).copy()
                    out_ch[c].append(seg)
                for c in range(num_ch):
                    seg = stretched[c].astype(np.float64).copy()
                    out_ch[c].append(seg)

    # remainder slices
    used = num_bars * slices_per_bar
    rem = total_slices - used
    for s in range(rem):
        gi = used + s
        st = start_offset + gi * slice_samples
        for c in range(num_ch):
            seg = samples[c, st:st + slice_samples].astype(np.float64).copy()
            out_ch[c].append(seg)

    # concatenate with crossfades
    out = np.zeros((num_ch, 0), dtype=np.float64)
    out_list = []
    for c in range(num_ch):
        out_list.append(_concat_crossfade(out_ch[c], fade))
    maxl = max(len(a) for a in out_list) if out_list else 0
    out = np.zeros((num_ch, maxl), dtype=np.float64)
    for c in range(num_ch):
        out[c, :len(out_list[c])] = out_list[c]

    out = np.clip(out, -0.99, 0.99)
    return out, sr, bpm, offset


def run(input_path, output_path, mode="fill", bpm=None, offset=None, perm=None, engine="formant"):
    samples, sr = load_audio(input_path)
    out, sr, bpm, offset = process_audio(samples, sr, bpm=bpm, offset=offset, mode=mode, perm=perm, engine=engine)
    save_wav(output_path, out, sr)
    return bpm, offset, out.shape[1] / sr
