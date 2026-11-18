"""Sound effects, batch 1: the old life, travel, the bay, the cafe and the street. All synthesized (no samples).

Every function is deterministic for a given `seed`, returns a float array at 44.1 kHz, mono (n,) unless the
docstring says stereo (n, 2), and peaks at roughly 0.3-0.9 before `amp` is applied.
"""
import numpy as np

import shared  # noqa: F401
import synth_dsp as D

SR = D.SR


def _rng(seed):
    return np.random.default_rng(1000 + int(seed))


def _slow(r, n, fc):
    """Slow random curve: low-passed noise scaled to unit standard deviation."""
    y = D.lp(r.standard_normal(n), fc)
    return y / (y.std() + 1e-12)


def _click(r, n, lo, hi, tau, attack=0.0002):
    return D.bp(r.standard_normal(n), lo, hi) * D.env_exp(n, tau, attack)


def _place(out, sig, t, gain=1.0):
    i = int(round(t * SR))
    if i >= len(out) or i < 0:
        return
    m = min(len(sig), len(out) - i)
    out[i:i + m] += sig[:m] * gain


# ------------------------------------------------------------------ the internal clock, the office
def tick(hi=True, amp=0.6, seed=0):
    """One clock tick (hi=True) or tock (lower): escapement click with a small wooden-case ring."""
    r = _rng(seed)
    n = int(0.09 * SR)
    tt = np.arange(n) / SR
    f = 2350.0 if hi else 1650.0
    y = _click(r, n, 1500, 7000, 0.0016) * 0.9
    y += np.sin(2 * np.pi * f * tt) * np.exp(-tt / 0.006) * 0.5
    y += np.sin(2 * np.pi * (410.0 if hi else 330.0) * tt) * np.exp(-tt / 0.018) * 0.25
    return amp * y


def clock(seconds, bpm=120.0, amp=0.6, slow_to=None, seed=0):
    """Tick-tock for `seconds` at `bpm` ticks per minute; slow_to=<bpm> makes it wind down to that rate."""
    n = int(seconds * SR)
    out = np.zeros(n)
    t, k = 0.0, 0
    while t < seconds - 0.05:
        out_gain = 1.0 if slow_to is None else 1.0 - 0.5 * t / seconds
        _place(out, tick(k % 2 == 0, 1.0, seed + k % 7), t, out_gain)
        rate = bpm if slow_to is None else bpm + (slow_to - bpm) * (t / seconds) ** 1.5
        t += 60.0 / rate
        k += 1
    return amp * out


def typing(seconds, rate=9.0, amp=0.5, seed=0):
    """Office keyboard: uneven bursts of key clicks with the odd spacebar thunk and a return key."""
    r = _rng(seed)
    n = int(seconds * SR)
    out = np.zeros(n)
    t = 0.02
    while t < seconds - 0.08:
        kind = r.random()
        m = int(0.07 * SR)
        if kind < 0.12:                                   # spacebar: lower, longer
            k = _click(r, m, 250, 1800, 0.012) * 1.0 + _click(r, m, 1500, 4500, 0.003) * 0.4
        else:                                             # key: down-click, then the softer release
            k = _click(r, m, 1400, 6500, 0.0028) * (0.6 + 0.4 * r.random())
            up = _click(r, m, 2000, 7000, 0.002) * 0.3
            _place(k, up[: m - int(0.035 * SR)], 0.03 + 0.02 * r.random())
            k += np.sin(2 * np.pi * r.uniform(700, 1100) * np.arange(m) / SR) * np.exp(-np.arange(m) / SR / 0.008) * 0.12
        _place(out, k, t)
        gap = r.exponential(1.0 / rate)
        t += float(np.clip(gap, 0.045, 0.5)) + (0.35 * (r.random() < 0.07))
    return amp * out


def stamp(amp=0.8, seed=0):
    """Rubber stamp on a stack of forms: thump, paper slap, the desk answering."""
    r = _rng(seed)
    n = int(0.3 * SR)
    tt = np.arange(n) / SR
    y = np.sin(2 * np.pi * (95 + 60 * np.exp(-tt / 0.01)) * tt) * np.exp(-tt / 0.05)
    y += _click(r, n, 500, 3500, 0.012) * 1.6 + _click(r, n, 150, 600, 0.03) * 0.6
    return amp * 0.7 * y * np.minimum(1, tt / 0.0005)


def paper(seconds=0.5, amp=0.4, seed=0):
    """A page turned / a form pulled off a pile."""
    r = _rng(seed)
    n = int(seconds * SR)
    tt = np.arange(n) / SR
    env = np.sin(np.pi * tt / seconds) ** 2 * (0.7 + 0.35 * _slow(r, n, 30)).clip(0, 1.5)
    y = D.bp(r.standard_normal(n), 1800, 9000) * env
    crackle = (r.random(n) > 0.9985) * r.uniform(0.5, 1.0, n)
    return amp * (y + D.bp(crackle, 2500, 9000) * 3.0 * np.sin(np.pi * tt / seconds))


# ------------------------------------------------------------------ travel
def cabin_chime(amp=0.5, two=True):
    """Aircraft cabin chime: a soft high-low 'ding-dong' (two=True) or a single ding."""
    n = int(2.8 * SR)
    tt = np.arange(n) / SR
    out = np.zeros(n)
    for k, f in enumerate((587.33, 466.16) if two else (587.33,)):
        t0 = 0.55 * k
        e = np.where(tt >= t0, np.exp(-(tt - t0) / 0.42) * np.minimum(1, (tt - t0) / 0.004), 0.0)
        out += (np.sin(2 * np.pi * f * tt) + 0.22 * np.sin(2 * np.pi * 2 * f * tt) + 0.08 * np.sin(2 * np.pi * 3 * f * tt)) * e
    return amp * out * 0.7 * np.minimum(1, (2.8 - tt) / 0.4)


def jet(seconds, amp=0.5, landing=True, seed=0):
    """Jet airliner heard from inside and then outside: cabin roar, turbine whine; with landing=True the
    whine spools down and the roar thins at the end (arrival). Stereo (n, 2)."""
    r = _rng(seed)
    n = int(seconds * SR)
    tt = np.arange(n) / SR
    u = tt / seconds
    out = []
    for c in range(2):
        roar = D.lp(r.standard_normal(n), 520) * 1.6 + D.bp(r.standard_normal(n), 900, 3800) * 0.22
        wob = 1 + 0.12 * _slow(r, n, 0.8)
        out.append(roar * wob.clip(0.6, 1.5))
    y = np.stack(out, axis=1)
    fw = 2600.0 * (1 - (0.38 * u ** 2 if landing else 0.0))
    whine = np.zeros(n)
    for mult, a in ((1.0, 1.0), (1.503, 0.5), (2.01, 0.3)):
        whine += a * np.sin(2 * np.pi * np.cumsum(fw * mult) / SR + r.uniform(0, 6))
    whine *= 0.05 * (1 + 0.3 * np.sin(2 * np.pi * 0.31 * tt))
    env = np.minimum(1, tt / (0.25 * seconds)) * (1 - (0.75 * np.clip((u - 0.6) / 0.4, 0, 1) ** 1.5 if landing else 0.0))
    y = (y * 0.5 + whine[:, None]) * env[:, None] * np.minimum(1, (seconds - tt) / (0.12 * seconds))[:, None]
    return amp * y


def suitcase(seconds, amp=0.5, speed=1.0, seed=0):
    """Suitcase wheels dragged over cobblestones: a plastic rumble with a clack at every stone."""
    r = _rng(seed)
    n = int(seconds * SR)
    tt = np.arange(n) / SR
    rumble = D.bp(r.standard_normal(n), 90, 700) * (0.8 + 0.35 * _slow(r, n, 12)).clip(0.2, 1.5)
    out = rumble * 0.5
    t = 0.05
    while t < seconds - 0.05:
        m = int(0.06 * SR)
        k = _click(r, m, 500, 3200, 0.006) * r.uniform(0.5, 1.0) + _click(r, m, 120, 500, 0.014) * 0.6
        _place(out, k, t)
        if r.random() < 0.35:                              # second wheel a moment later
            _place(out, k * 0.6, t + r.uniform(0.02, 0.05))
        t += r.uniform(0.09, 0.2) / speed
    return amp * out * np.minimum(1, tt / 0.15) * np.minimum(1, (seconds - tt) / 0.2)


def zip_(seconds=0.45, amp=0.4, up=True, seed=0):
    """A zip pulled: a fast ratchet rising (or falling) in pitch."""
    r = _rng(seed)
    n = int(seconds * SR)
    u = np.arange(n) / n
    rate = (220 + 380 * u) if up else (600 - 380 * u)
    teeth = (np.sin(2 * np.pi * np.cumsum(rate) / SR) > 0.3).astype(float)
    y = D.bp(r.standard_normal(n) * teeth, 1800, 8000) * np.sin(np.pi * u) ** 0.4
    return amp * y * np.minimum(1, u / 0.03) * np.minimum(1, (1 - u) / 0.05)


# ------------------------------------------------------------------ the bay
def surf(seconds, amp=0.5, period=7.5, seed=0, calm=0.0):
    """Waves on the shore of the bay: each wave swells, breaks and hisses back down the sand.
    calm=0..1 softens the breaks. Stereo (n, 2), waves drifting across the image."""
    r = _rng(seed)
    n = int(seconds * SR)
    tt = np.arange(n) / SR
    out = np.zeros((n, 2))
    bed = np.stack([D.lp(r.standard_normal(n), 420) for _ in range(2)], axis=1) * 0.5
    t = -r.uniform(0, period * 0.5)
    while t < seconds:
        dur = period * r.uniform(0.85, 1.25)
        m = int(dur * SR)
        u = np.arange(m) / m
        rise, brk = 0.32, 0.4
        env = np.where(u < rise, (u / rise) ** 2.2 * 0.6,
                       np.where(u < brk, 0.6 + 0.4 * (u - rise) / (brk - rise), np.exp(-(u - brk) / 0.22)))
        cut = 350 + (3000 - 1600 * calm) * np.where(u < brk, (u / brk) ** 3, np.exp(-(u - brk) / 0.3))
        w = r.standard_normal(m)
        lo = D.lp(w, 900)
        hi_ = D.svf(w, cut, 1.4, SR, 0) - D.lp(w, 300)
        wave = (lo * 1.3 + hi_ * (0.8 - 0.4 * calm)) * env
        fizz = D.bp(r.standard_normal(m), 3500, 9000) * np.where(u > brk, np.exp(-(u - brk) / 0.35), 0.0) * (0.10 * (1 - calm))
        pan = r.uniform(-0.6, 0.6)
        drift = pan + 0.5 * (u - 0.4) * np.sign(r.uniform(-1, 1))
        l = np.cos((np.clip(drift, -1, 1) + 1) * np.pi / 4)
        rr = np.sin((np.clip(drift, -1, 1) + 1) * np.pi / 4)
        i0 = int(t * SR)
        a, b = max(0, i0), min(n, i0 + m)
        if b > a:
            s = (wave + fizz)[a - i0:b - i0]
            out[a:b, 0] += s * l[a - i0:b - i0]
            out[a:b, 1] += s * rr[a - i0:b - i0]
        t += dur * r.uniform(0.6, 0.85)
    out += bed
    fade = np.minimum(1, tt / 1.0) * np.minimum(1, (seconds - tt) / 1.5)
    return amp * out * fade[:, None] * 0.8


# ------------------------------------------------------------------ the street
def footsteps(steps=8, interval=0.55, amp=0.5, kind="sandal", seed=0, stop=False):
    """Unhurried footsteps. kind='sandal' (flip-flop: sole, then the slap) or 'shoe' (hard heel).
    stop=True makes the walker slow down and halt on the last steps."""
    r = _rng(seed)
    n = int((steps * interval * (1.6 if stop else 1.0) + 0.5) * SR)
    out = np.zeros(n)
    t = 0.02
    for k in range(steps):
        m = int(0.16 * SR)
        if kind == "sandal":
            s = _click(r, m, 90, 500, 0.02) * 0.8 + _click(r, m, 600, 2600, 0.008) * 0.35
            slap = _click(r, m, 900, 4500, 0.007) * 0.6
            _place(s, slap[: m - int(0.07 * SR)], 0.045 + 0.02 * r.random())
        else:
            s = _click(r, m, 120, 700, 0.012) * 0.9 + _click(r, m, 1200, 3800, 0.004) * 0.6
        _place(out, s, t, 1.6 * (0.75 + 0.25 * r.random()))
        step = interval * (1 + 0.04 * r.standard_normal())
        if stop and k >= steps - 4:
            step *= 1.0 + 0.35 * (k - (steps - 5))
        t += step
    return amp * out


def broom(sweeps=4, interval=0.85, amp=0.4, seed=0):
    """Early-morning broom on a sidewalk: long dry strokes."""
    r = _rng(seed)
    n = int((sweeps * interval + 0.6) * SR)
    out = np.zeros(n)
    for k in range(sweeps):
        d = interval * r.uniform(0.5, 0.62)
        m = int(d * SR)
        u = np.arange(m) / m
        w = r.standard_normal(m)
        cut = 6500 - 3500 * u
        s = (D.svf(w, cut, 1.0, SR, 0) - D.lp(w, 1400)) * np.sin(np.pi * u) ** 0.7 * (0.8 + 0.25 * _slow(r, m, 40)).clip(0.3, 1.4)
        _place(out, s, 0.02 + k * interval + 0.03 * r.standard_normal(), 0.8 + 0.2 * r.random())
    return amp * out * 0.5


def keys(amp=0.4, seconds=0.7, seed=0):
    """A bunch of keys taken out and one pushed into a lock."""
    r = _rng(seed)
    n = int((seconds + 0.25) * SR)
    tt = np.arange(n) / SR
    out = np.zeros(n)
    for _ in range(int(14 * seconds / 0.7)):
        f = r.uniform(3800, 9500)
        m = int(0.09 * SR)
        mt = np.arange(m) / SR
        j = (np.sin(2 * np.pi * f * mt) + 0.5 * np.sin(2 * np.pi * f * 1.37 * mt)) * np.exp(-mt / r.uniform(0.01, 0.03))
        _place(out, j, r.uniform(0, seconds * 0.8), r.uniform(0.3, 1.0))
    _place(out, _click(r, int(0.08 * SR), 700, 3500, 0.006) * 2.2, seconds)      # into the lock
    _place(out, _click(r, int(0.1 * SR), 300, 1800, 0.012) * 2.0, seconds + 0.12)  # the turn
    return amp * out * 0.35


def shutter(seconds=1.6, amp=0.5, seed=0):
    """A shop's metal roller shutter going up: rattle rising, then the clunk at the top."""
    r = _rng(seed)
    n = int((seconds + 0.3) * SR)
    tt = np.arange(n) / SR
    u = np.clip(tt / seconds, 0, 1)
    rate = 16 + 16 * u
    slats = (np.sin(2 * np.pi * np.cumsum(rate) / SR) > 0.2).astype(float)
    w = r.standard_normal(n)
    y = D.bp(w * slats, 350, 4200) * 0.7 + D.bp(w, 1500, 6000) * 0.15
    ring = np.sin(2 * np.pi * 730 * tt) * 0.06 + np.sin(2 * np.pi * 1190 * tt) * 0.04
    y = (y + ring * slats) * np.where(tt < seconds, np.minimum(1, tt / 0.15), 0.0)
    _place(y, (_click(r, int(0.2 * SR), 120, 900, 0.03) * 2.0 + _click(r, int(0.2 * SR), 900, 3500, 0.012)), seconds)
    return amp * y * 0.6


# ------------------------------------------------------------------ the cafe and the plaza
def espresso(seconds=2.2, amp=0.4, seed=0):
    """Espresso machine: the steam wand screams up, gurgles in the milk, cuts off."""
    r = _rng(seed)
    n = int(seconds * SR)
    tt = np.arange(n) / SR
    u = tt / seconds
    w = r.standard_normal(n)
    cut = 1800 + 2600 * np.minimum(1, u / 0.25)
    hiss_ = D.svf(w, cut, 1.2, SR, 2)
    gurgle = 1 + 0.5 * np.sin(2 * np.pi * (28 + 10 * np.sin(2 * np.pi * 1.3 * tt)) * tt) * np.clip((u - 0.25) / 0.2, 0, 1)
    tone = np.sin(2 * np.pi * (1500 + 500 * u) * tt) * 0.05
    env = np.minimum(1, tt / 0.12) * np.minimum(1, (seconds - tt) / 0.06)
    return amp * (D.lp(hiss_, 9000) * 0.5 * gurgle + tone) * env


def cup(amp=0.5, seed=0):
    """A cup set down on its saucer: ceramic clink."""
    r = _rng(seed)
    n = int(0.3 * SR)
    tt = np.arange(n) / SR
    base = r.uniform(1900, 2500)
    y = sum(a * np.sin(2 * np.pi * base * m * tt + r.uniform(0, 6)) * np.exp(-tt / tau)
            for m, a, tau in ((1.0, 1.0, 0.035), (1.61, 0.7, 0.025), (2.47, 0.5, 0.015), (3.9, 0.3, 0.008)))
    y += _click(r, n, 2500, 9000, 0.002) * 0.8
    y2 = np.zeros(n)
    _place(y2, y[: n - int(0.07 * SR)] * 0.45, 0.055 + 0.02 * r.random())           # the little second touch
    return amp * (y + y2) * 0.5 * np.minimum(1, tt / 0.0003)


def ice(amp=0.5, cubes=5, seed=0):
    """Ice cubes dropped and settling in a glass."""
    r = _rng(seed)
    n = int(0.9 * SR)
    out = np.zeros(n)
    t = 0.01
    for k in range(cubes):
        m = int(0.25 * SR)
        mt = np.arange(m) / SR
        f = r.uniform(2600, 5200)
        c = (np.sin(2 * np.pi * f * mt) + 0.6 * np.sin(2 * np.pi * f * 1.52 * mt) + 0.35 * np.sin(2 * np.pi * f * 2.3 * mt)) * np.exp(-mt / r.uniform(0.03, 0.09))
        c += _click(r, m, 1500, 8000, 0.003) * 0.7 + _click(r, m, 250, 900, 0.01) * 0.3
        _place(out, c, t, (0.9 - 0.12 * k) * r.uniform(0.7, 1.0))
        t += r.uniform(0.05, 0.16) * (1 + 0.3 * k)
    return amp * out * 0.45


def cocktail_shaker(shakes=8, interval=0.14, amp=0.4, seed=0):
    """Ice in a metal cocktail shaker: fast double-sided rattle."""
    r = _rng(seed)
    n = int((shakes * interval + 0.3) * SR)
    out = np.zeros(n)
    for k in range(shakes):
        m = int(0.11 * SR)
        mt = np.arange(m) / SR
        s = _click(r, m, 1800, 9000, 0.02, 0.004) + 0.3 * np.sin(2 * np.pi * (3100 if k % 2 else 3600) * mt) * np.exp(-mt / 0.02)
        s += _click(r, m, 400, 1400, 0.012) * 0.4
        _place(out, s, 0.01 + k * interval, 1.0 if k % 2 == 0 else 0.75)
    return amp * out * 0.6


def cutlery(amp=0.4, seed=0):
    """Fork and knife on a plate."""
    r = _rng(seed)
    n = int(0.5 * SR)
    out = np.zeros(n)
    for k in range(3):
        m = int(0.2 * SR)
        mt = np.arange(m) / SR
        f = r.uniform(3200, 6500)
        c = (np.sin(2 * np.pi * f * mt) + 0.5 * np.sin(2 * np.pi * f * 1.71 * mt)) * np.exp(-mt / r.uniform(0.02, 0.06)) + _click(r, m, 3000, 10000, 0.002)
        _place(out, c, r.uniform(0, 0.28), r.uniform(0.4, 1.0))
    return amp * out * 0.4


def murmur(seconds, amp=0.3, people=7, seed=0):
    """Open-air cafe chatter with no words: several 'voices' as buzzing sources through moving vowel bands,
    each gated in syllables and phrases. Stereo (n, 2)."""
    r = _rng(seed)
    n = int(seconds * SR)
    tt = np.arange(n) / SR
    out = np.zeros((n, 2))
    for p in range(people):
        f0 = r.uniform(95, 230)
        pitch = f0 * (1 + 0.10 * _slow(r, n, 2.5)).clip(0.75, 1.4)
        src = D.saw(pitch, SR, r.random()) + 0.2 * r.standard_normal(n)
        syl = (0.35 + 0.9 * _slow(r, n, r.uniform(3.5, 6.0))).clip(0, 1.2) ** 1.5              # syllables
        phr = (0.25 + 1.3 * _slow(r, n, 0.25) + r.uniform(-0.2, 0.3)).clip(0, 1)              # talking / listening
        f1 = (520 + 170 * _slow(r, n, 3.0)).clip(280, 900)
        f2 = (1500 + 420 * _slow(r, n, 3.0)).clip(900, 2500)
        v = (D.svf(src, f1, 0.28, SR, 1) + 0.5 * D.svf(src, f2, 0.22, SR, 1)) * 0.25
        v = D.lp(v, 3400) * syl * phr
        out += D.pan_stereo(v, r.uniform(-0.9, 0.9)) * r.uniform(0.5, 1.0)
    out /= max(1.0, np.abs(out).max())
    fade = np.minimum(1, tt / 0.8) * np.minimum(1, (seconds - tt) / 0.8)
    return amp * out * fade[:, None]
