"""Vowel-onset detector used for closed-loop timing (and for the timing report).

Vowel onset = start of *sustained periodicity*: normalised autocorrelation (15 ms centred window, best lag
in 70-900 Hz) above 0.6 for >= 20 ms, with low-band (150-1500 Hz) energy within 18 dB of the note's
sustain. This is the same definition as the independent audit's detector (audit/onset.py), computed at
16 kHz with a 1 ms hop, so what we correct in the engine is what is measured on the stems.
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly

FS = 16000
W = int(0.015 * FS)
LMIN, LMAX = int(FS / 900), int(FS / 70)
_SOS = butter(4, [150, 1500], "bandpass", fs=FS, output="sos")


def to16(x, sr):
    if sr == FS:
        return np.asarray(x, np.float64)
    from math import gcd
    g = gcd(int(sr), FS)
    return resample_poly(np.asarray(x, np.float64), FS // g, int(sr) // g)


def features(x, t0, t1, lo=None):
    """x: 16 kHz signal. Returns times (s), periodicity, low-band dB at 1 ms hop over [t0, t1)."""
    hop = FS // 1000
    i0, i1 = int(round(t0 * FS)), int(round(t1 * FS))
    cs = np.arange(i0, i1, hop)
    per = np.zeros(len(cs))
    edb = np.full(len(cs), -140.0)
    if lo is None:
        base = max(0, i0 - 2000)
        lo_seg = sosfiltfilt(_SOS, x[base:min(len(x), i1 + 2000 + LMAX + W)])
    else:
        base, lo_seg = 0, lo
    ok = (cs - W // 2 >= 0) & (cs - W // 2 + W + LMAX <= len(x))
    if not ok.any():
        return cs / FS, per, edb
    idx = np.nonzero(ok)[0]
    starts = cs[idx] - W // 2
    win = np.lib.stride_tricks.sliding_window_view(x, W)
    A = win[starts]                                     # (n, W)
    ea = np.einsum("ij,ij->i", A, A)
    best = np.full(len(idx), -1.0)
    for L in range(LMIN, LMAX):
        B = win[starts + L]
        eb = np.einsum("ij,ij->i", B, B)
        r = np.einsum("ij,ij->i", A, B) / np.sqrt(ea * eb + 1e-12)
        best = np.maximum(best, r)
    best[ea <= 1e-10] = 0.0
    per[idx] = best
    for j, c in zip(idx, cs[idx]):
        a = c - base - 80
        seg = lo_seg[max(0, a):max(0, a) + 160]
        if len(seg):
            edb[j] = 10 * np.log10(np.mean(seg ** 2) + 1e-14)
    return cs / FS, per, edb


def voiced_runs(ts, per, edb, ref_db, pthr=0.6, edrop=18.0, minlen=20):
    v = (per > pthr) & (edb > ref_db - edrop)
    runs, j, n = [], 0, len(v)
    while j < n:
        if v[j]:
            k = j
            while k < n and v[k]:
                k += 1
            if k - j >= minlen or k == n:
                runs.append((ts[j], ts[k - 1]))
            j = k
        else:
            j += 1
    return runs


def vowel_onset(x16, T, sustain, before=0.16, after=0.14):
    """Measured vowel onset near expected time T (s, in x16's timeline), or None if voicing is continuous
    into the window (no acoustic boundary) or nothing voiced is found. sustain: note length (s)."""
    ts, per, edb = features(x16, T - before, T + after)
    _, _, edbn = features(x16, T + 0.03, T + max(0.06, min(sustain, 0.3)))
    ref = np.percentile(edbn, 90)
    runs = voiced_runs(ts, per, edb, ref)
    if not runs:
        return None
    cand = [r for r in runs if r[0] <= T + 0.06 and r[1] >= T + 0.02] or runs
    r = min(cand, key=lambda r: abs(r[0] - T))
    if r[0] <= T - before + 0.005:
        return None
    return float(r[0])
