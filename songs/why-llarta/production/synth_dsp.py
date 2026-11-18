"""Small DSP toolkit shared by the instrumental and the mix (numpy/scipy/numba)."""
import math

import numba
import numpy as np
from scipy import signal

SR = 44100


def mtof(m):
    return 440.0 * 2.0 ** ((np.asarray(m, dtype=float) - 69.0) / 12.0)


def _sos(kind, f, order=2):
    return signal.butter(order, f, btype=kind, fs=SR, output="sos")


def lp(x, fc, order=2):
    return signal.sosfilt(_sos("low", min(fc, 0.45 * SR), order), x, axis=0)


def hp(x, fc, order=2):
    return signal.sosfilt(_sos("high", fc, order), x, axis=0)


def bp(x, lo, hi, order=2):
    return signal.sosfilt(_sos("band", [lo, min(hi, 0.45 * SR)], order), x, axis=0)


def peq(x, f0, gain_db, q=1.0):
    """RBJ peaking EQ."""
    A = 10 ** (gain_db / 40)
    w = 2 * np.pi * f0 / SR
    al = np.sin(w) / (2 * q)
    b = [1 + al * A, -2 * np.cos(w), 1 - al * A]
    a = [1 + al / A, -2 * np.cos(w), 1 - al / A]
    return signal.lfilter(np.array(b) / a[0], np.array(a) / a[0], x, axis=0)


def shelf(x, f0, gain_db, high=True):
    """RBJ shelving EQ (S=1)."""
    A = 10 ** (gain_db / 40)
    w = 2 * np.pi * f0 / SR
    al = np.sin(w) / 2 * np.sqrt(2)
    c = np.cos(w)
    sA = 2 * np.sqrt(A) * al
    if high:
        b = [A * ((A + 1) + (A - 1) * c + sA), -2 * A * ((A - 1) + (A + 1) * c), A * ((A + 1) + (A - 1) * c - sA)]
        a = [(A + 1) - (A - 1) * c + sA, 2 * ((A - 1) - (A + 1) * c), (A + 1) - (A - 1) * c - sA]
    else:
        b = [A * ((A + 1) - (A - 1) * c + sA), 2 * A * ((A - 1) - (A + 1) * c), A * ((A + 1) - (A - 1) * c - sA)]
        a = [(A + 1) + (A - 1) * c + sA, -2 * ((A - 1) + (A + 1) * c), (A + 1) + (A - 1) * c - sA]
    return signal.lfilter(np.array(b) / a[0], np.array(a) / a[0], x, axis=0)


def env_exp(n, tau, attack=0.002):
    tt = np.arange(n) / SR
    a = np.clip(tt / attack, 0, 1) if attack > 0 else 1.0
    return a * np.exp(-tt / tau)


def adsr(n, a, d, s, r, gate):
    """gate = note-on length (s); n total samples (should cover gate + r)."""
    tt = np.arange(n) / SR
    e = np.where(tt < a, tt / max(a, 1e-4), s + (1 - s) * np.exp(-(tt - a) / max(d, 1e-4)))
    e_gate = (s + (1 - s) * np.exp(-(gate - a) / max(d, 1e-4))) if gate > a else gate / max(a, 1e-4)
    rel = tt > gate
    e[rel] = e_gate * np.exp(-(tt[rel] - gate) / max(r, 1e-4))
    return e


def sat(x, drive):
    return np.tanh(drive * x) / np.tanh(drive)


@numba.njit(cache=True)
def _blep(t, dt):
    if t < dt:
        t = t / dt
        return t + t - t * t - 1.0
    elif t > 1.0 - dt:
        t = (t - 1.0) / dt
        return t * t + t + t + 1.0
    return 0.0


@numba.njit(cache=True)
def saw(freq, sr, phase0):
    n = freq.shape[0]
    out = np.empty(n)
    ph = phase0
    for i in range(n):
        dt = freq[i] / sr
        ph += dt
        if ph >= 1.0:
            ph -= 1.0
        out[i] = 2.0 * ph - 1.0 - _blep(ph, dt)
    return out


@numba.njit(cache=True)
def square(freq, sr, phase0, pw):
    n = freq.shape[0]
    out = np.empty(n)
    ph = phase0
    for i in range(n):
        dt = freq[i] / sr
        ph += dt
        if ph >= 1.0:
            ph -= 1.0
        v = 1.0 if ph < pw else -1.0
        v += _blep(ph, dt)
        t2 = ph - pw
        if t2 < 0.0:
            t2 += 1.0
        v -= _blep(t2, dt)
        out[i] = v
    return out


@numba.njit(cache=True)
def svf(x, fc, k, sr, mode):
    """TPT state-variable filter. fc: per-sample cutoff array. k = 1/Q. mode 0=LP 1=BP 2=HP."""
    n = x.shape[0]
    y = np.empty(n)
    ic1 = 0.0
    ic2 = 0.0
    for i in range(n):
        f = fc[i]
        if f > 0.45 * sr:
            f = 0.45 * sr
        if f < 10.0:
            f = 10.0
        g = math.tan(math.pi * f / sr)
        a1 = 1.0 / (1.0 + g * (g + k))
        a2 = g * a1
        a3 = g * a2
        v3 = x[i] - ic2
        v1 = a1 * ic1 + a2 * v3
        v2 = ic2 + a2 * ic1 + a3 * v3
        ic1 = 2.0 * v1 - ic1
        ic2 = 2.0 * v2 - ic2
        if mode == 0:
            y[i] = v2
        elif mode == 1:
            y[i] = v1
        else:
            y[i] = x[i] - k * v1 - v2
    return y


def sine(freq, n, phase=0.0):
    if np.isscalar(freq):
        return np.sin(2 * np.pi * freq * np.arange(n) / SR + phase)
    return np.sin(2 * np.pi * np.cumsum(freq) / SR + phase)


def pan_stereo(x, pan):
    """Equal-power pan, pan in [-1, 1]. x mono -> (n, 2)."""
    l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
    return np.stack([x * l * np.sqrt(2), x * r * np.sqrt(2)], axis=1)


class Bus:
    """A stereo stem buffer on the song timeline."""

    def __init__(self, name, n):
        self.name = name
        self.x = np.zeros((n, 2), dtype=np.float32)
        self.n = n

    def add(self, sig, t, gain=1.0, pan=0.0):
        if sig.ndim == 1:
            sig = pan_stereo(sig, pan)
        i = int(round(t * SR))
        if i >= self.n or i + len(sig) <= 0:
            return
        j0 = max(0, -i)
        sig = sig[j0:]
        i = max(i, 0)
        m = min(len(sig), self.n - i)
        self.x[i:i + m] += (sig[:m] * gain).astype(np.float32)


def reverb_ir(seconds, decay, lo=200, hi=7000, predelay=0.02, seed=0, density_fade=0.04):
    """Stereo synthetic hall/plate IR (decorrelated L/R exponentially-decaying noise)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    tt = np.arange(n) / SR
    out = []
    for c in range(2):
        x = rng.standard_normal(n) * np.exp(-tt / decay) * np.minimum(1, tt / density_fade)
        x = lp(hp(x, lo), hi)
        x = np.concatenate([np.zeros(int(predelay * SR)), x])
        out.append(x / np.sqrt(np.sum(x ** 2)))
    return out


def convolve_stereo(x, ir):
    """x (n,2) -> wet (n,2) using IR pair; mono-sum input feeds both (classic send)."""
    m = x.mean(axis=1)
    n = x.shape[0]
    return np.stack([signal.fftconvolve(m, ir[c])[:n] for c in range(2)], axis=1).astype(np.float32)


def pingpong(x, delay_s, feedback, n_taps=6, lp_fc=5000):
    """Stereo ping-pong delay (returns wet only)."""
    n = x.shape[0]
    m = x.mean(axis=1)
    m = lp(hp(m, 300), lp_fc)
    wet = np.zeros((n, 2), dtype=np.float32)
    d = int(round(delay_s * SR))
    g = 1.0
    for k in range(1, n_taps + 1):
        g *= feedback
        if k * d >= n:
            break
        ch = (k + 1) % 2
        wet[k * d:, ch] += (m[: n - k * d] * g).astype(np.float32)
    return wet
