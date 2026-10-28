"""Formant-preserving phase vocoder (ported from Disappointment)."""
import numpy as np

FFT_SIZE = 2048
HALF_FFT = FFT_SIZE // 2
NUM_BINS = HALF_FFT + 1

# bit-reversal table
bitRev = np.zeros(FFT_SIZE, dtype=np.int64)
bits = 11
for i in range(FFT_SIZE):
    rev = 0
    v = i
    for b in range(bits):
        rev = (rev << 1) | (v & 1)
        v >>= 1
    bitRev[i] = rev

cosTable = np.zeros(HALF_FFT, dtype=np.float64)
sinTable = np.zeros(HALF_FFT, dtype=np.float64)
for i in range(HALF_FFT):
    angle = (-2 * np.pi * i) / FFT_SIZE
    cosTable[i] = np.cos(angle)
    sinTable[i] = np.sin(angle)

hannWindow = np.zeros(FFT_SIZE, dtype=np.float64)
for i in range(FFT_SIZE):
    hannWindow[i] = 0.5 * (1 - np.cos((2 * np.pi * i) / FFT_SIZE))


def fft2048(real, imag, invert):
    real = np.asarray(real, dtype=np.float64).copy()
    imag = np.asarray(imag, dtype=np.float64).copy()
    n = FFT_SIZE
    # bit reversal
    for i in range(n):
        j = int(bitRev[i])
        if i < j:
            real[i], real[j] = real[j], real[i]
            imag[i], imag[j] = imag[j], imag[i]
    length = 2
    while length <= n:
        halfLen = length // 2
        step = n // length
        for i in range(0, n, length):
            for k in range(halfLen):
                twiddleIdx = k * step
                wReal = cosTable[twiddleIdx]
                wImag = (sinTable[twiddleIdx] if not invert else -sinTable[twiddleIdx])
                evenIdx = i + k
                oddIdx = evenIdx + halfLen
                uR = real[evenIdx]
                uI = imag[evenIdx]
                vR = real[oddIdx] * wReal - imag[oddIdx] * wImag
                vI = real[oddIdx] * wImag + imag[oddIdx] * wReal
                real[evenIdx] = uR + vR
                imag[evenIdx] = uI + vI
                real[oddIdx] = uR - vR
                imag[oddIdx] = uI - vI
        length <<= 1
    if invert:
        real /= n
        imag /= n
    return real, imag


def wrapPhase(angle):
    return angle - 2 * np.pi * np.floor((angle + np.pi) / (2 * np.pi))


def extractFormantSpectralEnvelope(mag, envelopeOut, scratchA, scratchB, radiusBins=16):
    mag = np.asarray(mag, dtype=np.float64)
    nb = len(mag)
    for k in range(nb):
        scratchA[k] = np.log(mag[k] + 1e-7)
    # pass 1 box
    runningSum = 0.0
    count = 0
    for k in range(radiusBins + 1):
        runningSum += scratchA[k]
        count += 1
    for k in range(nb):
        scratchB[k] = runningSum / count
        addIdx = k + radiusBins + 1
        remIdx = k - radiusBins
        if addIdx < nb:
            runningSum += scratchA[addIdx]
            count += 1
        if remIdx >= 0:
            runningSum -= scratchA[remIdx]
            count -= 1
    # pass 2
    runningSum = 0.0
    count = 0
    for k in range(radiusBins + 1):
        runningSum += scratchB[k]
        count += 1
    for k in range(nb):
        envelopeOut[k] = np.exp(runningSum / count)
        addIdx = k + radiusBins + 1
        remIdx = k - radiusBins
        if addIdx < nb:
            runningSum += scratchB[addIdx]
            count += 1
        if remIdx >= 0:
            runningSum -= scratchB[remIdx]
            count -= 1


def stretch_block(source_channels, src_start, in_block_samples, out_block_samples, formant_shift_semitones=0.0):
    """source_channels: list of 1D np arrays (full source). Returns list of 1D arrays length out_block_samples."""
    num_channels = len(source_channels)
    total_source = source_channels[0].shape[0]
    is_stereo = num_channels > 1
    ch0 = np.asarray(source_channels[0], dtype=np.float64)
    ch1 = np.asarray(source_channels[1], dtype=np.float64) if is_stereo else ch0

    if abs(out_block_samples - in_block_samples) <= 4:
        copy_len = min(in_block_samples, out_block_samples)
        res = []
        for c in range(num_channels):
            arr = np.zeros(out_block_samples, dtype=np.float64)
            src = np.asarray(source_channels[c], dtype=np.float64)
            for i in range(copy_len):
                si = src_start + i
                di = i
                if 0 <= si < total_source and di < out_block_samples:
                    arr[di] = src[si]
            res.append(arr)
        return res

    stretch_ratio = out_block_samples / max(1, in_block_samples)
    synthesis_hop = FFT_SIZE // 4  # 512
    analysis_hop = synthesis_hop / stretch_ratio
    formant_ratio = 2.0 ** (formant_shift_semitones / 12.0)

    work_len = out_block_samples + FFT_SIZE
    accumL = np.zeros(work_len, dtype=np.float64)
    accumR = np.zeros(work_len, dtype=np.float64) if is_stereo else accumL
    cola_weight = np.zeros(work_len, dtype=np.float64)

    pre_pad = HALF_FFT
    out_pos = 0
    in_offset_float = float(-pre_pad)
    frame_idx = 0
    prev_flux_energy = 1e-5

    # scratch buffers
    realL = np.zeros(FFT_SIZE, dtype=np.float64)
    imagL = np.zeros(FFT_SIZE, dtype=np.float64)
    realR = np.zeros(FFT_SIZE, dtype=np.float64)
    imagR = np.zeros(FFT_SIZE, dtype=np.float64)
    magL = np.zeros(NUM_BINS, dtype=np.float64)
    phaseL = np.zeros(NUM_BINS, dtype=np.float64)
    magR = np.zeros(NUM_BINS, dtype=np.float64)
    phaseR = np.zeros(NUM_BINS, dtype=np.float64)
    magMid = np.zeros(NUM_BINS, dtype=np.float64)
    phaseMid = np.zeros(NUM_BINS, dtype=np.float64)
    prevAnalysisPhaseMid = np.zeros(NUM_BINS, dtype=np.float64)
    synthPhaseMid = np.zeros(NUM_BINS, dtype=np.float64)
    prevSynthPhaseMid = np.zeros(NUM_BINS, dtype=np.float64)
    phaseRot = np.zeros(NUM_BINS, dtype=np.float64)
    origFormantL = np.zeros(NUM_BINS, dtype=np.float64)
    origFormantR = np.zeros(NUM_BINS, dtype=np.float64)
    modFormantL = np.zeros(NUM_BINS, dtype=np.float64)
    modFormantR = np.zeros(NUM_BINS, dtype=np.float64)
    adjMagL = np.zeros(NUM_BINS, dtype=np.float64)
    adjMagR = np.zeros(NUM_BINS, dtype=np.float64)
    scratchA = np.zeros(NUM_BINS, dtype=np.float64)
    scratchB = np.zeros(NUM_BINS, dtype=np.float64)
    peakIndices = np.zeros(NUM_BINS, dtype=np.int64)
    omegaExpected = np.array([(2 * np.pi * k) / FFT_SIZE for k in range(NUM_BINS)], dtype=np.float64)

    while out_pos < out_block_samples + pre_pad:
        in_offset = int(round(in_offset_float))
        # extract windowed frame
        for n in range(FFT_SIZE):
            w = hannWindow[n]
            gIdx = src_start + in_offset + n
            clampedIdx = max(0, min(total_source - 1, gIdx))
            realL[n] = ch0[clampedIdx] * w
            imagL[n] = 0.0
            if is_stereo:
                realR[n] = ch1[clampedIdx] * w
                imagR[n] = 0.0
        realL, imagL = fft2048(realL, imagL, False)
        if is_stereo:
            realR, imagR = fft2048(realR, imagR, False)

        highEnergy = 0.0
        for k in range(NUM_BINS):
            rL = realL[k]; iL = imagL[k]
            magL[k] = np.hypot(rL, iL)
            phaseL[k] = np.arctan2(iL, rL)
            if is_stereo:
                rR = realR[k]; iR = imagR[k]
                magR[k] = np.hypot(rR, iR)
                phaseR[k] = np.arctan2(iR, rR)
                rM = 0.5 * (rL + rR)
                iM = 0.5 * (iL + iR)
                magMid[k] = np.hypot(rM, iM)
                phaseMid[k] = np.arctan2(iM, rM)
            else:
                magMid[k] = magL[k]
                phaseMid[k] = phaseL[k]
            if 16 <= k <= 380:
                highEnergy += magMid[k]

        extractFormantSpectralEnvelope(magL, origFormantL, scratchA, scratchB, 16)
        if is_stereo:
            extractFormantSpectralEnvelope(magR, origFormantR, scratchA, scratchB, 16)

        fluxRatio = highEnergy / (prev_flux_energy + 1e-5)
        prev_flux_energy = highEnergy * 0.6 + prev_flux_energy * 0.4
        is_transient = (frame_idx <= 1) or (fluxRatio > 1.52)

        if is_transient:
            for k in range(NUM_BINS):
                synthPhaseMid[k] = phaseMid[k]
                prevSynthPhaseMid[k] = phaseMid[k]
                prevAnalysisPhaseMid[k] = phaseMid[k]
                phaseRot[k] = 0.0
        else:
            # peaks
            numPeaks = 0
            for k in range(1, NUM_BINS - 1):
                if magMid[k] > magMid[k-1] and magMid[k] >= magMid[k+1]:
                    peakIndices[numPeaks] = k
                    numPeaks += 1
            if numPeaks == 0:
                peakIndices[0] = 0
                numPeaks = 1
            # Laroche & Dolson identity phase locking
            for p in range(numPeaks):
                kPeak = int(peakIndices[p])
                dPhi = phaseMid[kPeak] - prevAnalysisPhaseMid[kPeak] - omegaExpected[kPeak] * analysis_hop
                # wrap
                dPhi = dPhi - 2*np.pi*np.floor((dPhi + np.pi)/(2*np.pi))
                trueOmega = omegaExpected[kPeak] + dPhi / analysis_hop
                synthPhaseMid[kPeak] = wrapPhase(prevSynthPhaseMid[kPeak] + trueOmega * synthesis_hop)
            for p in range(numPeaks):
                kPeak = int(peakIndices[p])
                rot = synthPhaseMid[kPeak] - phaseMid[kPeak]
                if p == 0:
                    startBin = 0
                else:
                    startBin = ((int(peakIndices[p-1]) + kPeak) // 2) + 1
                if p == numPeaks - 1:
                    endBin = NUM_BINS - 1
                else:
                    endBin = (kPeak + int(peakIndices[p+1])) // 2
                for k in range(startBin, endBin + 1):
                    phaseRot[k] = rot
                    if k != kPeak:
                        synthPhaseMid[k] = wrapPhase(phaseMid[k] + rot)
            for k in range(NUM_BINS):
                prevAnalysisPhaseMid[k] = phaseMid[k]

        # Formant compensation
        for k in range(NUM_BINS):
            synthAdvance = wrapPhase(synthPhaseMid[k] - prevSynthPhaseMid[k] - omegaExpected[k] * synthesis_hop)
            coherenceLoss = max(0.78, np.cos(synthAdvance * 0.35))
            adjMagL[k] = magL[k] * coherenceLoss
            if is_stereo:
                adjMagR[k] = magR[k] * coherenceLoss
            prevSynthPhaseMid[k] = synthPhaseMid[k]

        extractFormantSpectralEnvelope(adjMagL, modFormantL, scratchA, scratchB, 16)
        if is_stereo:
            extractFormantSpectralEnvelope(adjMagR, modFormantR, scratchA, scratchB, 16)

        for k in range(NUM_BINS):
            targetEnvBin = int(round(k / formant_ratio))
            targetEnvBin = min(NUM_BINS - 1, max(0, targetEnvBin))
            envRestoreL = min(1.6, max(0.65, origFormantL[targetEnvBin] / (modFormantL[k] + 1e-7)))
            mOutL = magL[k] * envRestoreL
            pOutL = phaseL[k] + phaseRot[k]
            realL[k] = mOutL * np.cos(pOutL)
            imagL[k] = mOutL * np.sin(pOutL)
            if is_stereo:
                envRestoreR = min(1.6, max(0.65, origFormantR[targetEnvBin] / (modFormantR[k] + 1e-7)))
                mOutR = magR[k] * envRestoreR
                pOutR = phaseR[k] + phaseRot[k]
                realR[k] = mOutR * np.cos(pOutR)
                imagR[k] = mOutR * np.sin(pOutR)

        for k in range(1, HALF_FFT):
            realL[FFT_SIZE - k] = realL[k]
            imagL[FFT_SIZE - k] = -imagL[k]
            if is_stereo:
                realR[FFT_SIZE - k] = realR[k]
                imagR[FFT_SIZE - k] = -imagR[k]
        imagL[0] = 0.0
        imagL[HALF_FFT] = 0.0
        if is_stereo:
            imagR[0] = 0.0
            imagR[HALF_FFT] = 0.0

        realL, imagL = fft2048(realL, imagL, True)
        if is_stereo:
            realR, imagR = fft2048(realR, imagR, True)

        for n in range(FFT_SIZE):
            idx = out_pos + n
            if idx < work_len:
                w = hannWindow[n]
                cola_weight[idx] += w * w
                accumL[idx] += realL[n] * w
                if is_stereo:
                    accumR[idx] += realR[n] * w

        out_pos += synthesis_hop
        in_offset_float += analysis_hop
        frame_idx += 1

    # copy normalized output with transient anchoring
    transient_anchor = min(600, int(out_block_samples * 0.2))
    dstL = np.zeros(out_block_samples, dtype=np.float64)
    dstR = np.zeros(out_block_samples, dtype=np.float64) if is_stereo else None
    for i in range(out_block_samples):
        workIdx = pre_pad + i
        norm = 1.0 / cola_weight[workIdx] if cola_weight[workIdx] > 1e-4 else 1.0
        valL = accumL[workIdx] * norm
        valR = accumR[workIdx] * norm if is_stereo else valL
        if i < transient_anchor and src_start + i < total_source:
            alpha = 0.5 * (1 - np.cos((np.pi * i) / transient_anchor))
            verbatimL = ch0[src_start + i]
            valL = verbatimL * (1 - alpha) + valL * alpha
            if is_stereo:
                verbatimR = ch1[src_start + i]
                valR = verbatimR * (1 - alpha) + valR * alpha
        dstL[i] = valL
        if is_stereo:
            dstR[i] = valR

    # clip
    dstL = np.clip(dstL, -0.99, 0.99)
    if is_stereo:
        dstR = np.clip(dstR, -0.99, 0.99)

    res = [dstL]
    if is_stereo:
        res.append(dstR)
    # pad channels if more than 2
    for c in range(2, num_channels):
        # fallback: copy left
        res.append(dstL.copy())
    return res
