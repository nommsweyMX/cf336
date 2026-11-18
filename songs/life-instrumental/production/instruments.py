"""Instruments for the life instrumental. Every sound is synthesized from oscillators, noise, filters and
simple physical models (no samples). Functions return mono float arrays at 44.1 kHz.

Pitched:   nylon, harp, bass_ac, marimba, vibes, trumpet, flute, strings, choir, rhodes, pad, musicbox
Percussion: conga, bongo, clave, maraca, cowbell, guiro, timbale, bombo, tambourine, triangle, rim, cabasa
"""
import numba
import numpy as np

import shared  # noqa: F401
import synth_dsp as D

SR = D.SR
rng = np.random.default_rng(1961)


def noise(n):
    return rng.standard_normal(n)


def _release(y, dur, tau):
    tt = np.arange(len(y)) / SR
    rel = tt > dur
    y[rel] *= np.exp(-(tt[rel] - dur) / tau)
    return y


# ------------------------------------------------------------------ plucked strings (tuned Karplus-Strong)
@numba.njit(cache=True)
def _ks(exc, n, period, S, rho):
    """Karplus-Strong loop with a two-point loop filter (S=1 bright ... 0.5 dark) and an allpass for the
    fractional part of the delay, so the pitch is exact."""
    d_filt = 1.0 - S
    N = int(np.floor(period - d_filt - 1e-3))
    frac = period - N - d_filt
    if frac < 0.1:
        N -= 1
        frac += 1.0
    C = (1.0 - frac) / (1.0 + frac)
    buf = np.zeros(N)
    for i in range(min(N, len(exc))):
        buf[i] = exc[i]
    out = np.zeros(n)
    x1 = 0.0
    ax1 = 0.0
    ay1 = 0.0
    idx = 0
    for i in range(n):
        v = buf[idx]
        out[i] = v
        f = S * v + (1.0 - S) * x1
        x1 = v
        a = C * f + ax1 - C * ay1
        ax1 = f
        ay1 = a
        buf[idx] = rho * a
        idx += 1
        if idx >= N:
            idx = 0
    return out


def _pluck(midi, n, t60, S, exc_cut, pick=0.18):
    f0 = D.mtof(midi)
    period = SR / f0
    L = int(period) + 2
    exc = D.lp(rng.uniform(-1, 1, L + 64), exc_cut)[32:32 + L]
    k = max(1, int(pick * period))                    # pick-position comb
    exc[k:] -= 0.7 * exc[:-k]
    exc -= exc.mean()
    rho = float(np.exp(-6.91 / (f0 * t60)))
    return _ks(exc, n, period, S, rho)


def nylon(midi, dur, vel=0.6):
    """Nylon-string guitar: warm pluck with a wooden body."""
    n = int((dur + 0.35) * SR)
    t60 = float(np.clip(3.2 * (196.0 / D.mtof(midi)) ** 0.5, 0.9, 5.0))
    y = _pluck(midi, n, t60, 0.62, 1800 + 3200 * vel, pick=0.14)
    y = D.peq(D.peq(y, 105, 4.0, 1.4), 215, 2.5, 1.6)
    thump = D.bp(noise(n), 80, 260) * D.env_exp(n, 0.012, 0.0005) * 0.25
    y = _release(D.lp(y, 5200) + thump, dur, 0.07)
    return vel * D.hp(y, 65) * 0.9


def harp(midi, dur, vel=0.6):
    """Folk harp (arpa): bright, long-ringing pluck."""
    n = int((dur + 0.6) * SR)
    t60 = float(np.clip(4.5 * (261.6 / D.mtof(midi)) ** 0.4, 1.2, 6.0))
    y = _pluck(midi, n, t60, 0.82, 3000 + 5000 * vel, pick=0.3)
    return vel * D.hp(_release(y, dur, 0.25), 90) * 0.8


def bass_ac(midi, dur, vel=0.7):
    """Plucked acoustic bass (tololoche / guitarrón): string plus a round fundamental."""
    n = int((dur + 0.25) * SR)
    f0 = D.mtof(midi)
    y = _pluck(midi, n, 2.2, 0.52, 900 + 900 * vel, pick=0.2)
    tt = np.arange(n) / SR
    sub = np.sin(2 * np.pi * f0 * tt) * np.exp(-tt / 0.55) * 0.5
    y = D.lp(y, 1800) + sub
    y *= np.minimum(1, tt / 0.003)
    return vel * D.hp(_release(y, dur, 0.06), 32)


# ------------------------------------------------------------------ bars and bells
def marimba(midi, vel=0.6, dur=None):
    """Marimba bar: fundamental with the tuned 4th and 10th partials, soft mallet, resonator bloom."""
    f0 = D.mtof(midi)
    tau = float(np.clip(0.55 * (261.6 / f0) ** 0.55, 0.12, 1.4))
    n = int(min(6 * tau + 0.1, 4.0) * SR)
    tt = np.arange(n) / SR
    y = np.sin(2 * np.pi * f0 * tt) * np.exp(-tt / tau) * (1 - 0.35 * np.exp(-tt / 0.03))
    y += 0.22 * (0.4 + vel) * np.sin(2 * np.pi * 3.98 * f0 * tt) * np.exp(-tt / (tau / 4.5))
    y += 0.07 * (0.3 + vel) * np.sin(2 * np.pi * 9.9 * f0 * tt) * np.exp(-tt / (tau / 12))
    y += D.lp(noise(n), 1200 + 2500 * vel) * D.env_exp(n, 0.006, 0.0004) * 0.22
    y *= np.minimum(1, tt / 0.0008)
    if dur is not None:
        y = _release(y, dur, 0.05)
    return vel * y


def vibes(midi, vel=0.5, dur=2.5, trem=4.5):
    """Vibraphone: long metal bar with the motor tremolo."""
    f0 = D.mtof(midi)
    n = int((dur + 0.8) * SR)
    tt = np.arange(n) / SR
    y = np.sin(2 * np.pi * f0 * tt) * np.exp(-tt / 2.4)
    y += 0.16 * np.sin(2 * np.pi * 4.0 * f0 * tt) * np.exp(-tt / 0.6)
    y += 0.05 * np.sin(2 * np.pi * 10.0 * f0 * tt) * np.exp(-tt / 0.12)
    y += D.bp(noise(n), 1500, 6000) * D.env_exp(n, 0.004, 0.0003) * 0.1
    y *= (1 + 0.28 * np.sin(2 * np.pi * trem * tt)) * np.minimum(1, tt / 0.001)
    return vel * _release(y, dur, 0.3)


def musicbox(midi, vel=0.5):
    """Music-box comb tooth: glassy, slightly detuned, with a tiny mechanical tick."""
    f0 = D.mtof(midi) * 2 ** (rng.uniform(-5, 5) / 1200)
    n = int(1.6 * SR)
    tt = np.arange(n) / SR
    y = np.sin(2 * np.pi * f0 * tt) * np.exp(-tt / 0.55)
    y += 0.3 * np.sin(2 * np.pi * 2.01 * f0 * tt) * np.exp(-tt / 0.25)
    y += 0.2 * np.sin(2 * np.pi * 6.3 * f0 * tt) * np.exp(-tt / 0.05)
    y += D.bp(noise(n), 2500, 7000) * D.env_exp(n, 0.003, 0.0002) * 0.18
    return vel * y * np.minimum(1, tt / 0.0005)


# ------------------------------------------------------------------ winds
def trumpet(midi, dur, vel=0.7, vib=1.0, mute=False):
    """Mariachi-style trumpet: bright saw through a filter that opens with the breath, attack scoop,
    delayed vibrato. mute=True gives a thin harmon-mute colour."""
    n = int((dur + 0.12) * SR)
    tt = np.arange(n) / SR
    f0 = D.mtof(midi)
    scoop = -35.0 * np.exp(-tt / 0.022)
    vdepth = 14.0 * vib * np.clip((tt - 0.22) / 0.3, 0, 1)
    cents = scoop + vdepth * np.sin(2 * np.pi * 5.6 * tt + rng.uniform(0, 6)) + 3.0 * D.lp(noise(n), 6)
    f = f0 * 2 ** (cents / 1200)
    env = D.adsr(n, 0.028, 0.09, 0.82, 0.07, dur)
    src = D.saw(f, SR, rng.uniform(0, 1)) * 0.8 + D.square(f, SR, rng.uniform(0, 1), 0.3) * 0.2
    cut = np.clip(f0 * (2.2 + (5.5 + 7.0 * vel) * env ** 1.4), 300, 9000)
    y = D.svf(src, cut, 0.8, SR, 0)
    y = D.peq(D.peq(y, 1250, 4.0, 1.3), 2500, 3.0, 1.5)
    spit = D.bp(noise(n), 1500, 5000) * D.env_exp(n, 0.012, 0.001) * 0.12
    y = (y + spit) * env
    if mute:
        y = D.peq(D.hp(y, 900), 1900, 8.0, 2.5) * 0.7
    return vel * D.hp(y, 180) * 0.6


def flute(midi, dur, vel=0.5, vib=1.0, breath=1.0):
    """Wooden flute / quena: sine-like tone with breath noise and a soft chiff."""
    n = int((dur + 0.15) * SR)
    tt = np.arange(n) / SR
    f0 = D.mtof(midi)
    vdepth = 10.0 * vib * np.clip((tt - 0.18) / 0.35, 0, 1)
    f = f0 * 2 ** ((vdepth * np.sin(2 * np.pi * 4.8 * tt) - 12.0 * np.exp(-tt / 0.03)) / 1200)
    ph = 2 * np.pi * np.cumsum(f) / SR
    env = D.adsr(n, 0.05, 0.12, 0.85, 0.11, dur)
    y = np.sin(ph) + 0.28 * np.sin(2 * ph + 0.4) + 0.09 * np.sin(3 * ph) + 0.03 * np.sin(4 * ph)
    air = D.bp(noise(n), max(400.0, f0 * 1.5), min(9000.0, f0 * 9)) * (0.10 * breath)
    chiff = D.bp(noise(n), f0 * 2, min(10000.0, f0 * 12)) * D.env_exp(n, 0.02, 0.002) * 0.25 * breath
    return vel * (y * (0.85 + 0.15 * env) + air + chiff) * env * 0.5


# ------------------------------------------------------------------ sustained ensembles
def strings(midis, dur, vel=0.4, attack=0.22, release=0.5, bright=3200.0):
    """String section: six detuned players per note, slow bow attack."""
    n = int((dur + release + 0.1) * SR)
    tt = np.arange(n) / SR
    y = np.zeros(n)
    for m in midis:
        f0 = D.mtof(m)
        for det in (-13.0, -7.0, -2.0, 3.0, 8.0, 12.0):
            drift = 2.5 * np.sin(2 * np.pi * rng.uniform(0.15, 0.5) * tt + rng.uniform(0, 6))
            vibr = 6.0 * np.sin(2 * np.pi * rng.uniform(4.8, 5.8) * tt + rng.uniform(0, 6))
            y += D.saw(f0 * 2 ** ((det + drift + vibr) / 1200), SR, rng.uniform(0, 1))
    y /= 6 * max(1, len(midis)) ** 0.7
    env = D.adsr(n, attack, 0.2, 0.9, release, dur)
    y = D.lp(D.hp(y, 110), bright * (0.6 + 0.8 * vel))
    return vel * D.peq(y, 2800, -2.0, 0.8) * env


def choir(midis, dur, vel=0.4, vowel="oo", attack=0.35, release=0.7):
    """Wordless choir pad: pulse sources through vowel formants ("oo" or "ah"). No words."""
    n = int((dur + release + 0.1) * SR)
    tt = np.arange(n) / SR
    src = np.zeros(n)
    for m in midis:
        f0 = D.mtof(m)
        for det in (-9.0, 0.0, 8.0):
            vibr = 9.0 * np.sin(2 * np.pi * rng.uniform(4.6, 5.6) * tt + rng.uniform(0, 6))
            src += D.square(f0 * 2 ** ((det + vibr) / 1200), SR, rng.uniform(0, 1), 0.28)
    src = src / (3 * max(1, len(midis)) ** 0.7) + noise(n) * 0.015
    forms = {"oo": ((330, 1.0, 5.0), (800, 0.35, 6.0), (2400, 0.08, 8.0)),
             "ah": ((720, 1.0, 4.0), (1150, 0.6, 5.0), (2700, 0.18, 8.0))}[vowel]
    y = np.zeros(n)
    for fc, a, q in forms:
        y += a * D.bp(src, fc * (1 - 0.5 / q), fc * (1 + 0.5 / q))
    env = D.adsr(n, attack, 0.3, 0.9, release, dur)
    y = D.lp(y, 4500)
    y *= 0.16 / (np.sqrt(np.mean(y ** 2)) + 1e-9)        # even loudness whatever the note (formants are narrow)
    return vel * y * env


def rhodes(midi, dur, vel=0.6):
    """Electric piano: FM tine with a bell on the attack and a slow tremolo."""
    n = int((dur + 0.5) * SR)
    tt = np.arange(n) / SR
    f0 = D.mtof(midi)
    idx = (0.4 + 2.2 * vel) * np.exp(-tt / 0.35) + 0.25
    y = np.sin(2 * np.pi * f0 * tt + idx * np.sin(2 * np.pi * f0 * tt))
    y += 0.25 * vel * np.sin(2 * np.pi * f0 * 7.0 * tt) * np.exp(-tt / 0.06)
    tau = float(np.clip(2.6 * (220.0 / f0) ** 0.4, 0.8, 4.0))
    y *= np.exp(-tt / tau) * np.minimum(1, tt / 0.002) * (1 + 0.08 * np.sin(2 * np.pi * 4.2 * tt))
    return vel * D.lp(_release(y, dur, 0.18), 4200) * 0.7


def pad(midis, dur, vel=0.4, cut=1800.0, attack=0.6, release=1.2):
    """Warm analogue pad: detuned saws, slow filter."""
    n = int((dur + release + 0.1) * SR)
    tt = np.arange(n) / SR
    y = np.zeros(n)
    for m in midis:
        f0 = D.mtof(m)
        for det in (-11.0, -4.0, 5.0, 10.0):
            y += D.saw(np.full(n, f0 * 2 ** (det / 1200)), SR, rng.uniform(0, 1))
    y /= 4 * max(1, len(midis)) ** 0.7
    env = D.adsr(n, attack, 0.4, 0.85, release, dur)
    c = cut * (0.55 + 0.45 * env) * (1 + 0.12 * np.sin(2 * np.pi * 0.13 * tt))
    return vel * D.svf(y, c, 1.2, SR, 0) * env


# ------------------------------------------------------------------ Latin percussion
def conga(kind="open", hi=True, amp=0.7):
    """Conga: 'open' tone, 'slap' or 'mute'; hi=True is the quinto, False the tumba."""
    n = int(0.5 * SR)
    tt = np.arange(n) / SR
    f0 = 235.0 if hi else 160.0
    tau = {"open": 0.16, "slap": 0.05, "mute": 0.035}[kind]
    f = f0 * (1 + 0.12 * np.exp(-tt / 0.012))
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt / tau)
    y += 0.3 * np.sin(2 * np.pi * np.cumsum(f * 1.59) / SR) * np.exp(-tt / (tau * 0.5))
    skin = D.bp(noise(n), 700, 3800 if kind == "slap" else 2200) * D.env_exp(n, 0.012 if kind == "slap" else 0.006, 0.0004)
    y += skin * (0.9 if kind == "slap" else 0.3)
    return amp * y * np.minimum(1, tt / 0.0006)


def bongo(hi=True, amp=0.6):
    n = int(0.25 * SR)
    tt = np.arange(n) / SR
    f0 = 440.0 if hi else 320.0
    f = f0 * (1 + 0.1 * np.exp(-tt / 0.008))
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt / 0.06)
    y += D.bp(noise(n), 1200, 5000) * D.env_exp(n, 0.006, 0.0003) * 0.4
    return amp * y * np.minimum(1, tt / 0.0005)


def clave(amp=0.6):
    """Two hardwood sticks."""
    n = int(0.18 * SR)
    tt = np.arange(n) / SR
    y = np.sin(2 * np.pi * 2480 * tt) * np.exp(-tt / 0.028) + 0.35 * np.sin(2 * np.pi * 5350 * tt) * np.exp(-tt / 0.012)
    y += D.hp(noise(n), 3000) * D.env_exp(n, 0.0015, 0.0001) * 0.3
    return amp * y * np.minimum(1, tt / 0.0003)


def maraca(amp=0.3, long_=False):
    n = int((0.16 if long_ else 0.08) * SR)
    env = D.env_exp(n, 0.05 if long_ else 0.022, 0.012 if long_ else 0.004)
    return amp * D.hp(noise(n), 5200) * env


def cabasa(amp=0.25):
    n = int(0.12 * SR)
    return amp * D.bp(noise(n), 6000, 12000) * D.env_exp(n, 0.03, 0.01)


def cowbell(amp=0.4):
    n = int(0.35 * SR)
    tt = np.arange(n) / SR
    y = D.square(np.full(n, 545.0), SR, 0.0, 0.5) + D.square(np.full(n, 812.0), SR, 0.3, 0.5)
    y = D.bp(y, 400, 5000) * (0.75 * np.exp(-tt / 0.09) + 0.25 * np.exp(-tt / 0.012))
    return amp * y * 0.5 * np.minimum(1, tt / 0.0004)


def guiro(amp=0.35, seconds=0.22, up=True):
    """Scraped gourd: a ratchet of short ticks."""
    n = int(seconds * SR)
    tt = np.arange(n) / SR
    rate = np.linspace(55, 95, n) if up else np.linspace(95, 55, n)
    ticks = (np.sin(2 * np.pi * np.cumsum(rate) / SR) > 0.55).astype(float)
    y = D.bp(noise(n) * ticks, 1800, 6500) * np.sin(np.pi * tt / seconds) ** 0.5
    return amp * y


def timbale(amp=0.6, hi=True):
    n = int(0.45 * SR)
    tt = np.arange(n) / SR
    f0 = 310.0 if hi else 215.0
    y = np.sin(2 * np.pi * f0 * tt) * np.exp(-tt / 0.14) + 0.5 * np.sin(2 * np.pi * f0 * 1.52 * tt) * np.exp(-tt / 0.09)
    y += 0.3 * np.sin(2 * np.pi * f0 * 2.48 * tt) * np.exp(-tt / 0.05)
    y += D.bp(noise(n), 1500, 7000) * D.env_exp(n, 0.02, 0.0003) * 0.5
    return amp * y * np.minimum(1, tt / 0.0004)


def bombo(amp=0.9, f_lo=62.0):
    """Deep hand drum / tambora."""
    n = int(0.6 * SR)
    tt = np.arange(n) / SR
    f = f_lo * (1 + 0.7 * np.exp(-tt / 0.02))
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt / 0.2)
    y += D.bp(noise(n), 120, 900) * D.env_exp(n, 0.015, 0.0006) * 0.5
    return amp * D.lp(y, 3000) * np.minimum(1, tt / 0.001)


def tambourine(amp=0.3):
    n = int(0.22 * SR)
    tt = np.arange(n) / SR
    y = D.hp(noise(n), 6500) * np.exp(-tt / 0.05)
    for f in (7300, 8900, 10400, 11800):
        y += 0.12 * np.sin(2 * np.pi * f * tt + rng.uniform(0, 6)) * np.exp(-tt / 0.07)
    return amp * y * np.minimum(1, tt / 0.001)


def triangle(amp=0.3, open_=True):
    n = int((1.6 if open_ else 0.25) * SR)
    tt = np.arange(n) / SR
    tau = 0.55 if open_ else 0.05
    y = sum(a * np.sin(2 * np.pi * f * tt) for f, a in ((4180, 1.0), (5730, 0.6), (8120, 0.45), (10650, 0.3)))
    return amp * 0.4 * y * np.exp(-tt / tau) * np.minimum(1, tt / 0.0003)


def rim(amp=0.5):
    n = int(0.08 * SR)
    tt = np.arange(n) / SR
    y = np.sin(2 * np.pi * 1720 * tt) * np.exp(-tt / 0.012) + D.bp(noise(n), 900, 4000) * D.env_exp(n, 0.004, 0.0002) * 0.6
    return amp * y


def organ(midis, dur, vel=0.4, bright=1.0):
    """Small church / parlour organ: drawbar sines with a slow chorus."""
    n = int((dur + 0.25) * SR)
    tt = np.arange(n) / SR
    y = np.zeros(n)
    for m in midis:
        f0 = D.mtof(m)
        for mult, a in ((0.5, 0.5), (1.0, 1.0), (2.0, 0.6), (3.0, 0.3 * bright), (4.0, 0.25 * bright), (6.0, 0.1 * bright)):
            y += a * np.sin(2 * np.pi * f0 * mult * tt * (1 + 0.0007 * np.sin(2 * np.pi * 0.7 * tt)) + rng.uniform(0, 6))
    y /= 2.2 * max(1, len(midis)) ** 0.75
    y *= 1 + 0.05 * np.sin(2 * np.pi * 5.9 * tt)
    return vel * y * D.adsr(n, 0.03, 0.05, 0.95, 0.2, dur)
