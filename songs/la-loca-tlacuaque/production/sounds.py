"""Sound design for "La Loca Tlacuaque de la Calle Cerrada": every sound is synthesized here from
oscillators, noise and filters (no samples). Functions return mono float arrays at 44.1 kHz."""
import numpy as np

import shared  # noqa: F401  (adds the shared DSP toolkit to the path)
import synth_dsp as D

SR = D.SR
rng = np.random.default_rng(1707)


def noise(n):
    return rng.standard_normal(n)


# ------------------------------------------------------------------ drums
def kick(amp=1.0, f_hi=125.0, f_lo=47.0, tau=0.24):
    """Dusty boom-bap kick: sine sweep with a knock and a short click."""
    n = int(0.55 * SR)
    tt = np.arange(n) / SR
    f = f_lo + (f_hi - f_lo) * np.exp(-tt / 0.035)
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(ph) * np.exp(-tt / tau)
    knock = D.bp(noise(n), 90, 420) * D.env_exp(n, 0.02, 0.0005) * 0.5
    click = D.bp(noise(n), 1800, 5200) * D.env_exp(n, 0.0035, 0.0002) * 0.35
    y = D.sat(body * 1.5 + knock, 1.6) + click
    y *= np.minimum(1, tt / 0.0008)
    return amp * D.lp(y, 7000)


def snare(amp=1.0, ghost=False):
    """Crack snare: two body tones, a noise rattle and a clap-like smear."""
    n = int(0.42 * SR)
    tt = np.arange(n) / SR
    tone = (np.sin(2 * np.pi * 186 * tt) + 0.6 * np.sin(2 * np.pi * 331 * tt + 0.5)) * np.exp(-tt / 0.05)
    rattle = D.bp(noise(n), 1300, 8500) * D.env_exp(n, 0.045 if ghost else 0.105, 0.0008)
    smear = np.zeros(n)
    for off in (0.0, 0.011, 0.021):
        i = int(off * SR)
        smear[i:] += D.env_exp(n - i, 0.012, 0.0004)
    crack = D.bp(noise(n) * smear, 900, 4800) * 0.55
    y = D.sat(0.55 * tone + 0.9 * rattle + (0.0 if ghost else crack), 1.3)
    return amp * (0.45 if ghost else 1.0) * D.hp(y, 130)


def clap(amp=0.7):
    n = int(0.45 * SR)
    out = np.zeros(n)
    for k, off in enumerate((0.0, 0.010, 0.021, 0.030)):
        i = int(off * SR)
        out[i:] += D.env_exp(n - i, 0.006 if k < 3 else 0.15, 0.0005) * (1.0 if k < 3 else 0.9)
    x = D.bp(noise(n) * out, 900, 5200)
    return amp * D.peq(x, 1300, 4, 1.2)


def hat(amp=0.5, open_=False, tau=None):
    """Metallic hat: six detuned squares (808 style) plus noise, high-passed."""
    n = int((0.45 if open_ else 0.14) * SR)
    tt = np.arange(n) / SR
    met = np.zeros(n)
    for f in (2140.0, 3170.0, 4290.0, 5230.0, 6680.0, 8150.0):
        met += np.sign(np.sin(2 * np.pi * f * tt + rng.uniform(0, 6)))
    x = D.lp(D.hp(met / 6 * 0.6 + noise(n) * 0.5, 5500, order=4), 11500)
    t = tau or (0.16 if open_ else 0.028)
    return amp * x * D.env_exp(n, t, 0.0006)


def guira(amp=0.3, long_=False):
    """Cumbia güira scrape."""
    n = int((0.16 if long_ else 0.07) * SR)
    tt = np.arange(n) / SR
    scr = D.hp(noise(n), 4200, order=4) * (0.55 + 0.45 * np.sign(np.sin(2 * np.pi * (95 if long_ else 140) * tt)))
    return amp * scr * D.env_exp(n, 0.07 if long_ else 0.028, 0.004)


def sleigh(amp=0.35):
    """One shake of sleigh bells: a cluster of tiny high bells with scattered onsets."""
    n = int(0.30 * SR)
    tt = np.arange(n) / SR
    y = np.zeros(n)
    for f in rng.uniform(2600, 9200, 14):
        off = int(rng.uniform(0, 0.028) * SR)
        e = np.zeros(n)
        e[off:] = D.env_exp(n - off, rng.uniform(0.03, 0.09), 0.0005)
        y += np.sin(2 * np.pi * f * tt + rng.uniform(0, 6)) * e * rng.uniform(0.4, 1.0)
    y += D.hp(noise(n), 7000) * D.env_exp(n, 0.02, 0.001) * 1.5
    return amp * y / 6


def sub808(midi, dur, amp=1.0, glide_from=None, drive=1.8):
    """808 sub: sine with a pitch-drop attack, long decay and soft saturation."""
    n = int((dur + 0.25) * SR)
    tt = np.arange(n) / SR
    f1 = D.mtof(midi)
    f = f1 * (1 + 0.9 * np.exp(-tt / 0.018))
    if glide_from is not None:
        f0 = D.mtof(glide_from)
        f = f1 + (f0 - f1) * np.exp(-tt / 0.045)
    ph = 2 * np.pi * np.cumsum(f) / SR
    env = np.minimum(1, tt / 0.004) * np.exp(-tt / max(0.35, dur * 1.1))
    rel = tt > dur
    env[rel] *= np.exp(-(tt[rel] - dur) / 0.05)
    y = D.sat(np.sin(ph) * env * 1.2, drive)
    return amp * D.lp(y, 900)


# ------------------------------------------------------------------ pitched instruments
def piano(midi, dur, vel=0.6, dark=True):
    """Additive inharmonic piano; dark=True closes the top for a dusty sampled-loop feel."""
    f0 = D.mtof(midi)
    n = int((dur + 1.0) * SR)
    tt = np.arange(n) / SR
    B = 0.00035
    tau0 = float(np.clip(3.0 * (261.6 / f0) ** 0.45, 0.7, 5.0))
    y = np.zeros(n)
    for k in range(1, 13):
        fk = k * f0 * np.sqrt(1 + B * k * k)
        if fk > 15000:
            break
        a = (1 / k ** 1.15) * (0.6 + 0.6 * vel if k > 3 else 1)
        tk = tau0 / (1 + 0.32 * k)
        for det in (-0.9, 0.9):
            y += a * np.sin(2 * np.pi * fk * 2 ** (det / 1200) * tt + rng.uniform(0, 6)) * np.exp(-tt / tk)
    ham = D.lp(noise(n) * D.env_exp(n, 0.004, 0.0003), 2500) * 0.3
    y = (y / 8 + ham) * np.minimum(1, tt / 0.002)
    rel = tt > dur
    y[rel] *= np.exp(-(tt[rel] - dur) / 0.28)
    return vel * D.lp(y, (1700 + 2600 * vel) if dark else (3000 + 5000 * vel))


def accordion(midis, dur, vel=0.5, bright=3200.0):
    """Norteño button accordion: three reeds per note (musette detune), bellows swell, slight tremolo."""
    n = int((dur + 0.12) * SR)
    tt = np.arange(n) / SR
    y = np.zeros(n)
    for m in midis:
        f0 = D.mtof(m)
        for det, a in ((-9.0, 0.8), (0.0, 1.0), (10.0, 0.8)):
            fr = np.full(n, f0 * 2 ** (det / 1200))
            y += a * (0.6 * D.saw(fr, SR, rng.uniform(0, 1)) + 0.4 * D.square(fr, SR, rng.uniform(0, 1), 0.38))
    y /= max(1, len(midis)) * 2.4
    bell = D.adsr(n, 0.018, 0.09, 0.82, 0.05, dur)
    trem = 1 + 0.06 * np.sin(2 * np.pi * 5.6 * tt)
    y = D.lp(D.hp(y, 180), bright) * bell * trem
    return vel * D.peq(y, 1100, 3, 1.0)


def pluck(midi, dur=0.5, vel=0.5, damp=0.45):
    """Karplus-Strong plucked string (bajo sexto / muted guitar)."""
    f0 = D.mtof(midi)
    n = int((dur + 0.3) * SR)
    L = max(2, int(round(SR / f0)))
    buf = D.lp(rng.uniform(-1, 1, L), 4500)
    y = np.zeros(n)
    g = 0.996 - 0.02 * damp
    prev = 0.0
    for i in range(n):
        j = i % L
        v = buf[j]
        y[i] = v
        buf[j] = g * (0.5 * v + 0.5 * prev)
        prev = v
    tt = np.arange(n) / SR
    rel = tt > dur
    y[rel] *= np.exp(-(tt[rel] - dur) / 0.04)
    return vel * D.hp(y, 70)


def glock(midi, vel=0.5):
    """Glockenspiel bar: fundamental plus the bar's inharmonic overtones."""
    f0 = D.mtof(midi)
    n = int(1.4 * SR)
    tt = np.arange(n) / SR
    y = np.sin(2 * np.pi * f0 * tt) * np.exp(-tt / 0.45)
    y += 0.42 * np.sin(2 * np.pi * 2.756 * f0 * tt) * np.exp(-tt / 0.16)
    y += 0.22 * np.sin(2 * np.pi * 5.404 * f0 * tt) * np.exp(-tt / 0.06)
    y += D.hp(noise(n), 5000) * D.env_exp(n, 0.004, 0.0002) * 0.25
    return vel * y * np.minimum(1, tt / 0.0005)


def toypiano(midi, vel=0.5):
    """Toy piano: plinky metal rod, slightly out of tune, short."""
    f0 = D.mtof(midi) * 2 ** (rng.uniform(-9, 9) / 1200)
    n = int(0.8 * SR)
    tt = np.arange(n) / SR
    mod = np.sin(2 * np.pi * f0 * 3.01 * tt) * 1.6 * np.exp(-tt / 0.05)
    y = np.sin(2 * np.pi * f0 * tt + mod) * np.exp(-tt / 0.22)
    y += 0.3 * np.sin(2 * np.pi * f0 * 6.27 * tt) * np.exp(-tt / 0.03)
    y += D.bp(noise(n), 1500, 6000) * D.env_exp(n, 0.006, 0.0003) * 0.3
    return vel * D.hp(y, 250) * np.minimum(1, tt / 0.0006)


def tuba(midi, dur, vel=0.6):
    """Sousaphone-style oom for the hook bounce."""
    n = int((dur + 0.1) * SR)
    f = np.full(n, D.mtof(midi))
    y = D.saw(f, SR, 0.0) * 0.7 + D.square(f, SR, 0.25, 0.5) * 0.3
    cut = 180 + 520 * D.adsr(n, 0.02, 0.12, 0.35, 0.05, dur)
    y = D.svf(y, cut, 0.9, SR, 0)
    return vel * y * D.adsr(n, 0.02, 0.1, 0.8, 0.05, dur)


# ------------------------------------------------------------------ FX
def crackle(seconds, amp=0.05):
    """Vinyl crackle and hiss."""
    n = int(seconds * SR)
    imp = np.zeros(n)
    k = rng.random(n) < (9.0 / SR)
    imp[k] = rng.uniform(-1, 1, k.sum()) * rng.uniform(0.3, 1.0, k.sum())
    pops = D.bp(imp, 900, 7000) * 6
    hiss = D.bp(noise(n), 2500, 9000) * 0.12
    rumble = D.lp(noise(n), 60) * 0.5
    return amp * (pops + hiss + rumble)


def siren(seconds, amp=0.4, lo=620.0, hi=1350.0, rate=2.6):
    """Patrol whoop-whoop."""
    n = int(seconds * SR)
    tt = np.arange(n) / SR
    saw = (tt * rate) % 1.0
    f = lo + (hi - lo) * np.sin(np.pi * saw) ** 0.8
    y = D.square(f, SR, 0.0, 0.5) * 0.6 + D.saw(f * 1.004, SR, 0.3) * 0.4
    y = D.lp(D.sat(y, 2.0), 3800)
    e = np.minimum(1, tt / 0.03) * np.minimum(1, (seconds - tt) / 0.08)
    return amp * D.hp(y, 400) * e


def glass_crash(amp=0.7, seed=0):
    """A glass apple hitting tile: bright crack, ringing shards, tinkles."""
    r = np.random.default_rng(100 + seed)
    n = int(1.6 * SR)
    tt = np.arange(n) / SR
    y = D.hp(r.standard_normal(n), 3000, order=4) * D.env_exp(n, 0.035, 0.0003) * 1.2
    for f in r.uniform(1800, 9500, 22):
        y += np.sin(2 * np.pi * f * tt + r.uniform(0, 6)) * np.exp(-tt / r.uniform(0.03, 0.22)) * r.uniform(0.1, 0.5)
    for _ in range(18):  # tinkles as the pieces settle
        off = int(r.uniform(0.06, 1.0) * SR)
        m = min(n - off, int(0.12 * SR))
        f = r.uniform(3200, 11000)
        y[off:off + m] += np.sin(2 * np.pi * f * np.arange(m) / SR) * D.env_exp(m, r.uniform(0.008, 0.03), 0.0003) \
            * r.uniform(0.15, 0.5) * np.exp(-off / SR / 0.5)
    return amp * D.hp(y, 1200) / 4


def splash(amp=0.6):
    """Belly-flop into the pool."""
    n = int(1.3 * SR)
    tt = np.arange(n) / SR
    slap = D.bp(noise(n), 300, 2500) * D.env_exp(n, 0.05, 0.002) * 1.2
    spray = D.bp(noise(n), 1800, 9000) * np.minimum(1, tt / 0.04) * np.exp(-tt / 0.38)
    wob = 1 + 0.5 * np.sin(2 * np.pi * (14 - 8 * tt) * tt)
    thump = np.sin(2 * np.pi * 70 * tt) * np.exp(-tt / 0.07) * 0.8
    return amp * (slap + spray * wob * 0.8 + thump) / 2


def door_slam(amp=0.8):
    """Door slam followed by the lock clicking shut."""
    n = int(0.75 * SR)
    tt = np.arange(n) / SR
    y = np.sin(2 * np.pi * 88 * tt) * np.exp(-tt / 0.06) + D.bp(noise(n), 150, 1800) * D.env_exp(n, 0.035, 0.0005)
    for off, f in ((0.30, 2400.0), (0.36, 3300.0)):
        i = int(off * SR)
        m = n - i
        y[i:] += (np.sin(2 * np.pi * f * np.arange(m) / SR) * 0.5 + D.hp(noise(m), 2500)) * D.env_exp(m, 0.006, 0.0002) * 0.6
    return amp * y / 1.6


def hiss(seconds=0.7, amp=0.3):
    """Possum hiss."""
    n = int(seconds * SR)
    tt = np.arange(n) / SR
    y = D.bp(noise(n), 3500, 9500, order=4) * (1 + 0.35 * np.sin(2 * np.pi * 31 * tt))
    e = np.minimum(1, tt / 0.05) * np.minimum(1, (seconds - tt) / 0.18) ** 1.5
    return amp * y * e


def scratch(amp=0.4, n_moves=3):
    """Record scratch: band-passed noise whose band is dragged back and forth."""
    n = int(0.12 * n_moves * SR)
    tt = np.arange(n) / SR
    pos = np.abs(np.sin(np.pi * tt / 0.12))
    fc = 500 + 3500 * pos
    y = D.svf(noise(n) + 0.6 * D.saw(np.full(n, 220.0) * (0.5 + pos), SR, 0.0), fc, 0.35, SR, 1)
    return amp * y * (0.25 + pos) * np.minimum(1, (tt[-1] - tt) / 0.02)


def sub_drop(amp=0.9, seconds=1.4):
    n = int(seconds * SR)
    tt = np.arange(n) / SR
    f = 34 + 60 * np.exp(-tt / 0.22)
    return amp * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt / 0.5) * np.minimum(1, tt / 0.004)


def reverse_swell(seconds, amp=0.4):
    n = int(seconds * SR)
    tt = np.arange(n) / SR
    y = D.bp(noise(n), 1500, 9000) * (tt / seconds) ** 3
    return amp * y
