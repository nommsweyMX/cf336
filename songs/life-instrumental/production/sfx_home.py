"""Sound effects, batch 2: the house, water and plumbing, electricity, the phone, the motif sounds and the
possum night. All synthesized (no samples). Same conventions as sfx_world: deterministic per `seed`, float
arrays at 44.1 kHz, mono (n,) unless the docstring says stereo (n, 2)."""
import numpy as np

import shared  # noqa: F401
import synth_dsp as D
from sfx_world import _click, _place, _rng, _slow

SR = D.SR


# ------------------------------------------------------------------ water and plumbing
def shower(seconds, amp=0.4, dip_at=None, dip_len=1.4, seed=0):
    """A running shower. With dip_at (seconds) the water pressure sags, the pipe shudders, and it recovers."""
    r = _rng(seed)
    n = int(seconds * SR)
    tt = np.arange(n) / SR
    press = np.ones(n)
    if dip_at is not None:
        u = (tt - dip_at) / dip_len
        press -= 0.72 * np.sin(np.pi * np.clip(u, 0, 1)) ** 0.6
    w = r.standard_normal(n)
    spray = D.svf(w, 1400 + 2400 * press, 1.3, SR, 2) * press ** 1.5
    patter = D.bp(w * (r.random(n) > 0.97), 400, 2600) * 2.2 * press
    low = D.bp(r.standard_normal(n), 120, 500) * 0.25 * press
    y = D.lp(spray, 9500) * 0.5 + patter + low
    if dip_at is not None:                                # the shudder in the wall as it comes back
        u = (tt - dip_at - dip_len * 0.75)
        y += np.where((u > 0) & (u < 0.5), np.sin(2 * np.pi * 86 * tt) * np.sin(2 * np.pi * 13 * tt) * np.exp(-u / 0.15), 0.0) * 0.6
    return amp * y * np.minimum(1, tt / 0.25) * np.minimum(1, (seconds - tt) / 0.35)


def pump(seconds, amp=0.4, failing=True, seed=0):
    """The building's water pump: a labouring motor hum with a rattle; failing=True makes it sag and stutter."""
    r = _rng(seed)
    n = int(seconds * SR)
    tt = np.arange(n) / SR
    u = tt / seconds
    speed = np.ones(n)
    if failing:
        speed -= 0.22 * np.clip((u - 0.45) / 0.55, 0, 1) ** 1.5 + 0.05 * _slow(r, n, 1.5) * np.clip((u - 0.3) / 0.3, 0, 1)
    ph = 2 * np.pi * np.cumsum(58.0 * speed) / SR
    hum = np.sin(ph) * 0.6 + np.sin(2 * ph) * 0.5 + np.sin(3 * ph + 0.5) * 0.22 + np.sin(4 * ph) * 0.2 + np.sin(6 * ph) * 0.08
    rattle = D.bp(r.standard_normal(n), 300, 2200) * (0.5 + 0.5 * np.sin(ph * 0.5)) ** 4 * 0.5
    whoosh = D.bp(r.standard_normal(n), 500, 3000) * 0.12
    gate = np.ones(n)
    if failing:
        st = (np.sin(2 * np.pi * 7.5 * tt) > 0.2) | (u < 0.72)
        gate = D.lp(st.astype(float), 60).clip(0, 1) * 0.6 + 0.4
    y = (hum * 0.45 + rattle + whoosh) * gate
    return amp * D.lp(y, 5000) * np.minimum(1, tt / 0.3) * np.minimum(1, (seconds - tt) / 0.4)


def pipe_knock(knocks=3, interval=0.21, amp=0.6, seed=0):
    """Water hammer: the pipe knocking in the wall."""
    r = _rng(seed)
    n = int((knocks * interval + 0.6) * SR)
    out = np.zeros(n)
    for k in range(knocks):
        m = int(0.5 * SR)
        mt = np.arange(m) / SR
        s = np.sin(2 * np.pi * (112 + 40 * np.exp(-mt / 0.01)) * mt) * np.exp(-mt / 0.07)
        s += 0.35 * np.sin(2 * np.pi * 690 * mt) * np.exp(-mt / 0.12) + 0.2 * np.sin(2 * np.pi * 1130 * mt) * np.exp(-mt / 0.07)
        s += _click(r, m, 300, 2500, 0.006) * 0.5
        _place(out, s, 0.01 + k * interval * (1 + 0.15 * k), 0.8 ** k)
    return amp * out * 0.8


def drip(amp=0.5, seed=0, pitch=1.0):
    """One drop of water falling into water: the rising 'plink' of the bubble."""
    r = _rng(seed)
    n = int(0.22 * SR)
    tt = np.arange(n) / SR
    f0 = r.uniform(620, 900) * pitch
    f = f0 * (1 + 1.5 * (1 - np.exp(-tt / 0.012)))
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt / 0.028) * np.minimum(1, tt / 0.002)
    y += _click(r, n, 1500, 6000, 0.002) * 0.25
    return amp * y


def drips(seconds, interval=0.9, amp=0.5, seed=0, regular=0.9):
    """A leak: drops at a nearly steady interval (regular=1 is metronomic)."""
    r = _rng(seed)
    n = int(seconds * SR)
    out = np.zeros(n)
    t = 0.05
    k = 0
    while t < seconds - 0.25:
        _place(out, drip(1.0, seed * 31 + k, pitch=1 + 0.08 * r.standard_normal()), t, r.uniform(0.7, 1.0))
        t += interval * (1 + (1 - regular) * r.standard_normal() * 0.6)
        k += 1
    return amp * out


def water_jug(amp=0.5, seed=0):
    """A heavy water jug carried and set down: slosh, glug, thud."""
    r = _rng(seed)
    n = int(1.6 * SR)
    tt = np.arange(n) / SR
    slosh = D.bp(r.standard_normal(n), 250, 1500) * (0.5 + 0.5 * np.sin(2 * np.pi * 2.6 * tt)) ** 3 * np.exp(-tt / 0.9) * 0.6
    out = slosh
    for k in range(4):
        m = int(0.12 * SR)
        mt = np.arange(m) / SR
        f = r.uniform(180, 320) * (1 + 0.9 * mt / 0.12)
        _place(out, np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * mt / 0.12) ** 2, 0.15 + 0.22 * k + 0.03 * r.standard_normal(), 0.5)
    thud = np.sin(2 * np.pi * (70 + 50 * np.exp(-np.arange(int(0.3 * SR)) / SR / 0.015)) * np.arange(int(0.3 * SR)) / SR) * np.exp(-np.arange(int(0.3 * SR)) / SR / 0.06)
    _place(out, thud + _click(r, int(0.3 * SR), 150, 1200, 0.02) * 0.6, 1.15, 1.0)
    return amp * out * np.minimum(1, (1.6 - tt) / 0.1)


# ------------------------------------------------------------------ electricity
def mains_hum(seconds, amp=0.3, sag=False, seed=0):
    """Mains hum from a tired transformer (60 Hz and its harmonics); sag=True lets the voltage droop."""
    r = _rng(seed)
    n = int(seconds * SR)
    tt = np.arange(n) / SR
    v = np.ones(n)
    if sag:
        v -= 0.35 * np.clip(np.sin(np.pi * tt / seconds), 0, 1) ** 2 + 0.05 * _slow(r, n, 3.0)
    ph = 2 * np.pi * 60.0 * tt
    y = np.sin(2 * ph) * 0.7 + np.sin(4 * ph + 0.3) * 0.35 + np.sin(6 * ph + 1.0) * 0.22 + np.sin(ph) * 0.3 + np.sin(8 * ph) * 0.1
    buzz = D.bp(np.sign(np.sin(2 * ph)) * 1.0, 800, 3500) * 0.12
    return amp * (y * 0.5 + buzz) * v * np.minimum(1, tt / 0.15) * np.minimum(1, (seconds - tt) / 0.2)


def arc(seconds=0.9, amp=0.5, seed=0, density=1.0):
    """A corroded connection arcing: sharp crackles riding the mains cycle."""
    r = _rng(seed)
    n = int(seconds * SR)
    tt = np.arange(n) / SR
    cyc = np.abs(np.sin(2 * np.pi * 60 * tt)) ** 6
    burst = (_slow(r, n, 9.0) > 0.4).astype(float)
    sp = (r.random(n) > 1 - 0.02 * density) * r.uniform(0.3, 1.0, n) * np.sign(r.standard_normal(n))
    y = D.hp(sp * cyc * burst, 900) * 4.0 + D.bp(r.standard_normal(n), 2500, 9000) * cyc * burst * 0.5
    y += np.sign(np.sin(2 * np.pi * 120 * tt)) * burst * 0.05
    return amp * 0.7 * D.lp(y, 11000) * np.minimum(1, (seconds - tt) / 0.05)


def flicker(amp=0.6, seed=0):
    """The lights flicker once: relay click, a buzzing dip, click back."""
    r = _rng(seed)
    n = int(0.75 * SR)
    tt = np.arange(n) / SR
    y = np.zeros(n)
    _place(y, _click(r, int(0.05 * SR), 900, 5000, 0.003) * 1.6, 0.02)
    dip = ((tt > 0.03) & (tt < 0.36)).astype(float)
    y += np.sign(np.sin(2 * np.pi * 120 * tt)) * D.lp(dip, 80).clip(0, 1) * 0.18 * (1 + 0.5 * np.sin(2 * np.pi * 23 * tt))
    y += D.bp(r.standard_normal(n), 2000, 8000) * dip * np.abs(np.sin(2 * np.pi * 60 * tt)) ** 8 * 0.5
    _place(y, _click(r, int(0.05 * SR), 700, 4000, 0.004) * 1.3, 0.37)
    _place(y, np.sin(2 * np.pi * 3900 * np.arange(int(0.3 * SR)) / SR) * np.exp(-np.arange(int(0.3 * SR)) / SR / 0.09) * 0.08, 0.38)  # filament ping
    return amp * D.lp(y, 9000)


def switch(amp=0.6, seed=0):
    """A flashlight switch: plastic click-clack."""
    r = _rng(seed)
    n = int(0.12 * SR)
    y = _click(r, n, 1800, 7500, 0.0025) * 1.2
    _place(y, _click(r, n, 900, 4000, 0.004)[: n - int(0.05 * SR)] * 0.8, 0.045)
    return amp * y


# ------------------------------------------------------------------ the phone
def ping(amp=0.5, kind=0):
    """A chat notification: a short two-note marimba-like blip. kind 0..3 are different pitches/shapes
    (generic tones, not any app's actual sound)."""
    n = int(0.5 * SR)
    tt = np.arange(n) / SR
    pairs = ((1318.5, 1760.0), (1568.0, 2093.0), (1760.0, 1318.5), (2093.0, 2637.0))[kind % 4]
    y = np.zeros(n)
    for k, f in enumerate(pairs):
        t0 = 0.085 * k
        e = np.where(tt >= t0, np.exp(-(tt - t0) / 0.07) * np.minimum(1, (tt - t0) / 0.002), 0.0)
        y += (np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(2 * np.pi * 2 * f * tt + 1.0) + 0.1 * np.sin(2 * np.pi * 3.01 * f * tt)) * e * (1.0 if k else 0.8)
    return amp * y * 0.6


def pings(count=6, spread=2.0, amp=0.5, seed=0, accelerate=True):
    """A group chat waking up: notifications arriving in a burst, faster and faster. Stereo (n, 2)."""
    r = _rng(seed)
    n = int((spread + 0.7) * SR)
    out = np.zeros((n, 2))
    for k in range(count):
        u = k / max(1, count - 1)
        t = spread * (u ** 0.6 if accelerate else u) + 0.02 * r.standard_normal()
        p = D.pan_stereo(ping(1.0, int(r.integers(0, 4))), r.uniform(-0.8, 0.8))
        i = max(0, int(t * SR))
        m = min(len(p), n - i)
        out[i:i + m] += p[:m] * r.uniform(0.6, 1.0)
    return amp * out


def phone_taps(seconds=1.5, amp=0.3, seed=0, rate=9.0):
    """Thumbs typing a message on a phone: soft, fast, glassy taps."""
    r = _rng(seed)
    n = int(seconds * SR)
    out = np.zeros(n)
    t = 0.01
    while t < seconds - 0.05:
        m = int(0.04 * SR)
        mt = np.arange(m) / SR
        k = _click(r, m, 1800, 6000, 0.0025) * 0.6 + np.sin(2 * np.pi * r.uniform(1100, 1500) * mt) * np.exp(-mt / 0.006) * 0.5
        _place(out, k, t, r.uniform(0.6, 1.0))
        t += float(np.clip(r.exponential(1 / rate), 0.05, 0.3))
    return amp * out


def ringback(rings=2, on=1.0, off=0.8, amp=0.35, answered=True, seed=0):
    """Calling ahead: the ringing tone in the earpiece (425 Hz), then the click of someone picking up."""
    r = _rng(seed)
    n = int((rings * (on + off) + 0.4) * SR)
    tt = np.arange(n) / SR
    y = np.zeros(n)
    for k in range(rings):
        t0 = k * (on + off)
        g = D.lp(((tt >= t0) & (tt < t0 + on)).astype(float), 200).clip(0, 1)
        y += np.sin(2 * np.pi * 425 * tt) * g
    y = D.bp(y + 0.01 * r.standard_normal(n), 300, 3200) * 0.6
    if answered:
        t0 = rings * (on + off) - off + 0.25
        _place(y, _click(r, int(0.06 * SR), 500, 3500, 0.004) * 1.5, t0)
        _place(y, D.bp(r.standard_normal(int(0.3 * SR)), 300, 3000) * 0.05, t0 + 0.03)
    return amp * y


def camera(amp=0.5, seed=0):
    """A phone-camera shutter: click, tiny whirr, clack."""
    r = _rng(seed)
    n = int(0.3 * SR)
    tt = np.arange(n) / SR
    y = _click(r, n, 1500, 8000, 0.003) * 1.2
    y += D.bp(r.standard_normal(n), 2500, 6000) * np.where((tt > 0.02) & (tt < 0.11), 0.12, 0.0)
    _place(y, _click(r, n, 900, 5000, 0.006)[: n - int(0.13 * SR)] * 1.0, 0.115)
    return amp * y


def card_beep(amp=0.4, double=True):
    """A card payment terminal accepting: one or two clean beeps."""
    n = int(0.5 * SR)
    tt = np.arange(n) / SR
    y = np.zeros(n)
    for k in range(2 if double else 1):
        t0 = 0.17 * k
        g = D.lp(((tt >= t0) & (tt < t0 + 0.11)).astype(float), 400).clip(0, 1)
        y += (np.sin(2 * np.pi * 2730 * tt) + 0.25 * np.sin(2 * np.pi * 5460 * tt)) * g
    return amp * y * 0.5


def knock(knocks=3, interval=0.19, amp=0.7, seed=0):
    """Knuckles on a wooden door: somebody arriving without calling ahead."""
    r = _rng(seed)
    n = int((knocks * interval + 0.4) * SR)
    out = np.zeros(n)
    for k in range(knocks):
        m = int(0.3 * SR)
        mt = np.arange(m) / SR
        s = np.sin(2 * np.pi * (165 + 70 * np.exp(-mt / 0.006)) * mt) * np.exp(-mt / 0.035)
        s += 0.5 * np.sin(2 * np.pi * 410 * mt) * np.exp(-mt / 0.02) + _click(r, m, 600, 3000, 0.004) * 0.7
        _place(out, s, 0.01 + k * interval + 0.01 * r.standard_normal(), r.uniform(0.8, 1.0))
    return amp * out * 0.8


# ------------------------------------------------------------------ small animals
def mosquito(seconds=2.5, amp=0.25, seed=0):
    """A mosquito circling an ankle: thin whine that swerves, comes close and leaves. Stereo (n, 2)."""
    r = _rng(seed)
    n = int(seconds * SR)
    tt = np.arange(n) / SR
    wander = _slow(r, n, 1.2)
    f = 610 * 2 ** (0.14 * wander + 0.03 * np.sin(2 * np.pi * 11 * tt))
    ph = 2 * np.pi * np.cumsum(f) / SR
    y = np.sin(ph) * 0.5 + np.sin(2 * ph) * 0.8 + np.sin(3 * ph) * 0.5 + np.sin(4 * ph) * 0.25
    near = (0.35 + 0.65 * np.sin(np.pi * tt / seconds) ** 2) * (0.75 + 0.25 * _slow(r, n, 2.0)).clip(0.2, 1.5)
    y = D.hp(y, 500) * near * 0.3
    pan = np.clip(0.9 * np.sin(2 * np.pi * 0.37 * tt + r.uniform(0, 6)) + 0.2 * wander, -1, 1)
    out = np.stack([y * np.cos((pan + 1) * np.pi / 4), y * np.sin((pan + 1) * np.pi / 4)], axis=1)
    fade = np.minimum(1, tt / 0.3) * np.minimum(1, (seconds - tt) / 0.4)
    return amp * out * fade[:, None]


def scrabble(seconds=1.6, amp=0.4, seed=0):
    """Claws scrabbling along a roof beam: bursts of quick dry scratches."""
    r = _rng(seed)
    n = int(seconds * SR)
    out = np.zeros(n)
    t = 0.03
    while t < seconds - 0.1:
        for j in range(int(r.integers(3, 8))):
            m = int(0.04 * SR)
            s = _click(r, m, 1500, 7000, 0.006, 0.002) + _click(r, m, 250, 900, 0.008) * 0.5
            _place(out, s, t, r.uniform(0.4, 1.0))
            t += r.uniform(0.035, 0.08)
        t += r.uniform(0.08, 0.35)
    return amp * out * 0.8


# ------------------------------------------------------------------ motif sounds
def mirror(seconds=3.0, amp=0.35, sharpen=True, midi=84, seed=0):
    """The mirror: a singing-glass tone. With sharpen=True it starts soft and flattering and grows edges
    (upper partials and a harder beat) until it is simply accurate."""
    r = _rng(seed)
    n = int(seconds * SR)
    tt = np.arange(n) / SR
    u = tt / seconds
    f0 = D.mtof(midi)
    y = np.zeros(n)
    for k, (mult, a) in enumerate(((1.0, 1.0), (2.32, 0.5), (4.25, 0.35), (6.63, 0.25), (9.38, 0.18))):
        grow = 1.0 if (k == 0 or not sharpen) else np.clip((u - 0.12 * k) / 0.5, 0, 1) ** 1.5
        beat = 1 + 0.25 * np.sin(2 * np.pi * (0.9 + 0.8 * k) * tt + r.uniform(0, 6))
        y += a * np.sin(2 * np.pi * f0 * mult * tt + r.uniform(0, 6)) * grow * beat
    rub = D.bp(r.standard_normal(n), f0 * 0.9, f0 * 1.1) * 0.35 * (1 + 0.5 * np.sin(2 * np.pi * 6.5 * tt))
    env = np.minimum(1, tt / (0.3 * seconds)) * np.minimum(1, (seconds - tt) / (0.18 * seconds))
    return amp * (y * 0.3 + rub) * env


def hammer(hits=3, interval=0.33, amp=0.7, seed=0):
    """Building: a hammer driving a nail into timber."""
    r = _rng(seed)
    n = int((hits * interval + 0.4) * SR)
    out = np.zeros(n)
    for k in range(hits):
        m = int(0.3 * SR)
        mt = np.arange(m) / SR
        s = np.sin(2 * np.pi * (130 + 90 * np.exp(-mt / 0.005)) * mt) * np.exp(-mt / 0.03)
        s += 0.45 * np.sin(2 * np.pi * (2550 + 120 * k) * mt) * np.exp(-mt / 0.035) + _click(r, m, 1200, 6000, 0.003) * 0.9
        _place(out, s, 0.01 + k * interval, r.uniform(0.85, 1.0))
    return amp * out * 0.8


def gears(seconds=2.0, amp=0.25, seed=0):
    """Small machinery: little gears turning with a faint whirr."""
    r = _rng(seed)
    n = int(seconds * SR)
    tt = np.arange(n) / SR
    out = D.bp(r.standard_normal(n), 2500, 5000) * 0.05 * (1 + 0.4 * np.sin(2 * np.pi * 31 * tt))
    for rate, f, a in ((6.0, 3100, 1.0), (9.0, 4300, 0.7), (4.0, 2200, 0.8)):
        t = r.uniform(0, 1 / rate)
        while t < seconds - 0.03:
            m = int(0.025 * SR)
            mt = np.arange(m) / SR
            _place(out, (np.sin(2 * np.pi * f * mt) * np.exp(-mt / 0.004) + _click(r, m, 2000, 8000, 0.0015) * 0.6), t, a * 0.6)
            t += 1 / rate
    return amp * out * np.minimum(1, tt / 0.2) * np.minimum(1, (seconds - tt) / 0.3)


def bell(midi=57, seconds=5.0, amp=0.5, seed=0):
    """A church bell: struck bronze with its hum, prime, minor-third tierce, quint and nominal."""
    r = _rng(seed)
    n = int(seconds * SR)
    tt = np.arange(n) / SR
    f = D.mtof(midi)
    parts = ((0.5, 0.45, 1.0), (1.0, 1.0, 0.75), (1.19, 0.7, 0.55), (1.5, 0.35, 0.4), (2.0, 0.85, 0.32),
             (2.51, 0.3, 0.2), (3.01, 0.4, 0.14), (4.1, 0.25, 0.09), (5.4, 0.15, 0.06))
    y = np.zeros(n)
    for mult, a, tau in parts:
        beat = 1 + 0.08 * np.sin(2 * np.pi * r.uniform(0.6, 2.2) * tt + r.uniform(0, 6))
        y += a * np.sin(2 * np.pi * f * mult * tt + r.uniform(0, 6)) * np.exp(-tt / (tau * seconds * 0.45)) * beat
    y += _click(r, n, 800, 6000, 0.006) * 0.6
    return amp * y * 0.3 * np.minimum(1, tt / 0.001) * np.minimum(1, (seconds - tt) / 0.3)


def heartbeat(beats=4, bpm=62.0, amp=0.8, seed=0):
    """Lub-dub, the same heart that opens and closes 'What You Repeat'."""
    r = _rng(seed)
    per = 60.0 / bpm
    n = int((beats * per + 0.3) * SR)
    out = np.zeros(n)
    for k in range(beats):
        for j, (f0, tau, a, dt) in enumerate(((58.0, 0.055, 1.0, 0.0), (72.0, 0.04, 0.7, min(0.3, per * 0.32)))):
            m = int(0.3 * SR)
            mt = np.arange(m) / SR
            s = np.sin(2 * np.pi * (f0 + 30 * np.exp(-mt / 0.012)) * mt) * np.exp(-mt / tau) + D.lp(r.standard_normal(m), 260) * D.env_exp(m, 0.025, 0.002) * 0.35
            _place(out, s * np.minimum(1, mt / 0.004), k * per + dt, a)
    return amp * D.lp(out, 700)


def bottle(amp=0.5, seed=0):
    """A cold glass bottle of Coca opened: the cap pried, the gas, the cap dropping on the table."""
    r = _rng(seed)
    n = int(1.1 * SR)
    tt = np.arange(n) / SR
    y = _click(r, n, 2000, 9000, 0.004) * 1.4
    fizz = D.bp(r.standard_normal(n), 2800, 9000) * np.where(tt > 0.012, np.exp(-(tt - 0.012) / 0.16), 0.0) * 0.6
    fizz += D.bp(r.standard_normal(n) * (r.random(n) > 0.96), 3000, 8500) * np.exp(-tt / 0.5) * 1.4
    y += fizz
    for k, t0 in enumerate((0.42, 0.5, 0.555)):           # the cap bouncing
        m = int(0.12 * SR)
        mt = np.arange(m) / SR
        c = (np.sin(2 * np.pi * 5200 * mt) + 0.6 * np.sin(2 * np.pi * 7900 * mt)) * np.exp(-mt / 0.012) + _click(r, m, 3000, 10000, 0.002)
        _place(y, c, t0, 0.5 * 0.6 ** k)
    return amp * D.lp(y, 11000) * 0.7


def tape(seconds=0.6, amp=0.4, seed=0):
    """Packing tape ripped off the roll."""
    r = _rng(seed)
    n = int(seconds * SR)
    u = np.arange(n) / n
    am = (np.sin(2 * np.pi * np.cumsum(70 + 60 * u) / SR) > -0.2).astype(float)
    y = D.bp(r.standard_normal(n), 1200, 7000) * am * (0.8 + 0.2 * _slow(r, n, 20)).clip(0.3, 1.4)
    y += np.sin(2 * np.pi * np.cumsum(900 + 700 * u) / SR) * 0.08 * am
    return amp * y * np.minimum(1, u / 0.04) * np.minimum(1, (1 - u) / 0.06) * 0.6


def applause(seconds=3.0, amp=0.35, people=40, seed=0):
    """A small crowd clapping (for the bay taking its bow). Stereo (n, 2)."""
    r = _rng(seed)
    n = int(seconds * SR)
    tt = np.arange(n) / SR
    out = np.zeros((n, 2))
    env = np.minimum(1, tt / 0.5) * np.minimum(1, ((seconds - tt) / (0.45 * seconds))) ** 1.3
    for p in range(people):
        rate = r.uniform(3.2, 5.2)
        t = r.uniform(0, 0.6)
        lo = r.uniform(700, 1400)
        pan = r.uniform(-1, 1)
        l, rr = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
        g = r.uniform(0.4, 1.0)
        while t < seconds - 0.05:
            m = int(0.03 * SR)
            c = _click(r, m, lo, lo * 3.2, 0.004) * g
            i = int(t * SR)
            k = min(m, n - i)
            out[i:i + k, 0] += c[:k] * l
            out[i:i + k, 1] += c[:k] * rr
            t += (1 / rate) * (1 + 0.08 * r.standard_normal())
    out /= max(1.0, np.abs(out).max() / 0.9)
    return amp * out * env[:, None]




# ------------------------------------------------------------------ more: office, shop, phone, heat, the patrol
def chime(amp=0.5, up=True):
    """A calendar / meeting reminder: two soft synthetic notes (generic, not any product's sound)."""
    n = int(1.3 * SR)
    tt = np.arange(n) / SR
    y = np.zeros(n)
    for k, f in enumerate((880.0, 1174.7) if up else (1174.7, 880.0)):
        t0 = 0.16 * k
        e = np.where(tt >= t0, np.exp(-(tt - t0) / 0.28) * np.minimum(1, (tt - t0) / 0.004), 0.0)
        y += (np.sin(2 * np.pi * f * tt) + 0.35 * np.sin(2 * np.pi * 3 * f * tt) * np.exp(-tt / 0.1) + 0.15 * np.sin(2 * np.pi * 4.02 * f * tt)) * e
    return amp * y * 0.45 * np.minimum(1, (1.3 - tt) / 0.15)


def till(amp=0.5, seed=0):
    """A corner-shop till: the bell, the drawer shooting out, coins settling."""
    r = _rng(seed)
    n = int(1.3 * SR)
    tt = np.arange(n) / SR
    y = np.zeros(n)
    for f, a, tau in ((2093, 1.0, 0.35), (4310, 0.5, 0.2), (5870, 0.3, 0.12), (8040, 0.2, 0.07)):
        y += a * np.sin(2 * np.pi * f * tt + r.uniform(0, 6)) * np.exp(-tt / tau)
    y = y * 0.35 + _click(r, n, 2500, 9000, 0.002) * 0.6
    drawer = D.bp(r.standard_normal(n), 150, 1500) * np.sin(np.pi * np.clip((tt - 0.1) / 0.32, 0, 1)) ** 0.5 * 0.5
    y += drawer
    _place(y, _click(r, int(0.15 * SR), 120, 1200, 0.02) * 1.2, 0.42)
    for _ in range(7):
        m = int(0.08 * SR)
        mt = np.arange(m) / SR
        f = r.uniform(4500, 9000)
        _place(y, (np.sin(2 * np.pi * f * mt) + 0.5 * np.sin(2 * np.pi * f * 1.4 * mt)) * np.exp(-mt / 0.012), r.uniform(0.44, 0.75), r.uniform(0.1, 0.3))
    return amp * y


_DTMF = {"1": (697, 1209), "2": (697, 1336), "3": (697, 1477), "4": (770, 1209), "5": (770, 1336), "6": (770, 1477),
         "7": (852, 1209), "8": (852, 1336), "9": (852, 1477), "0": (941, 1336), "*": (941, 1209), "#": (941, 1477)}


def dial(digits="3221", amp=0.3, tone=0.085, gap=0.06):
    """Dialing a number: touch-tones (the standard tone pairs), quick and practised."""
    n = int((len(digits) * (tone + gap) + 0.1) * SR)
    tt = np.arange(n) / SR
    y = np.zeros(n)
    for k, d in enumerate(digits):
        lo, hi = _DTMF[d]
        t0 = 0.02 + k * (tone + gap)
        g = D.lp(((tt >= t0) & (tt < t0 + tone)).astype(float), 500).clip(0, 1)
        y += (np.sin(2 * np.pi * lo * tt) + np.sin(2 * np.pi * hi * tt)) * g
    return amp * D.bp(y, 400, 3400) * 0.5


def heat(seconds, amp=0.2, seed=0):
    """Heat and glare: a thin high shimmer that trembles, like air over hot stone. Stereo (n, 2)."""
    r = _rng(seed)
    n = int(seconds * SR)
    tt = np.arange(n) / SR
    out = np.zeros((n, 2))
    for k, f in enumerate((5274.0, 5920.0, 6645.0, 7459.0, 8372.0)):
        trem = 0.5 + 0.5 * np.sin(2 * np.pi * r.uniform(5.5, 9.0) * tt + r.uniform(0, 6))
        wob = 2 ** (0.004 * np.sin(2 * np.pi * r.uniform(0.2, 0.6) * tt + r.uniform(0, 6)))
        s = np.sin(2 * np.pi * np.cumsum(f * wob) / SR) * trem * (0.6 if k else 1.0)
        out += D.pan_stereo(s, r.uniform(-0.9, 0.9))
    buzz = D.bp(r.standard_normal(n), 4500, 9000) * (0.5 + 0.5 * np.sin(2 * np.pi * 47 * tt)) ** 3 * (0.6 + 0.4 * _slow(r, n, 0.6)).clip(0, 1.4)
    out += np.stack([buzz, np.roll(buzz, 300)], axis=1) * 0.8
    env = np.minimum(1, tt / (0.3 * seconds)) * np.minimum(1, (seconds - tt) / (0.25 * seconds))
    return amp * out * env[:, None] * 0.3


def grind(seconds, amp=0.35, seed=0):
    """A low grinding tone that tightens: two close pitches rubbing, the filter slowly opening.
    (The cost that 'accumulates'; it is a tension tone, not a body sound.)"""
    r = _rng(seed)
    n = int(seconds * SR)
    tt = np.arange(n) / SR
    u = tt / seconds
    f1 = np.full(n, 55.0)
    f2 = 55.0 * 2 ** ((100 - 55 * u) / 1200)              # a semitone apart, closing in: the beat slows
    y = D.saw(f1, SR, 0.0) + D.saw(f2, SR, 0.37) + 0.5 * D.saw(f1 * 2.01, SR, 0.5)
    y = D.svf(y, 180 + 1500 * u ** 2, 0.5, SR, 0)
    y *= (0.8 + 0.2 * np.sin(2 * np.pi * (3 + 9 * u) * tt))
    env = u ** 1.3 * np.minimum(1, (seconds - tt) / 0.05)
    return amp * D.sat(y * 0.6, 1.5) * env


def shelf_crash(amp=0.7, seed=0):
    """A shelf coming down: one heavy thud, then things rolling and clattering."""
    r = _rng(seed)
    n = int(1.5 * SR)
    out = np.zeros(n)
    m = int(0.4 * SR)
    mt = np.arange(m) / SR
    _place(out, np.sin(2 * np.pi * (75 + 80 * np.exp(-mt / 0.01)) * mt) * np.exp(-mt / 0.09) + _click(r, m, 200, 2500, 0.03) * 0.9, 0.01)
    t = 0.12
    for k in range(14):
        mm = int(0.12 * SR)
        mmt = np.arange(mm) / SR
        f = r.uniform(300, 2400)
        c = np.sin(2 * np.pi * f * mmt) * np.exp(-mmt / r.uniform(0.01, 0.04)) + _click(r, mm, 500, 5000, 0.006)
        _place(out, c, t, 0.7 * 0.86 ** k * r.uniform(0.5, 1.0))
        t += r.uniform(0.03, 0.13)
    return amp * out * 0.8


def trickle(seconds=1.4, amp=0.35, seed=0):
    """A thin stream of liquid on porcelain."""
    r = _rng(seed)
    n = int(seconds * SR)
    tt = np.arange(n) / SR
    w = r.standard_normal(n)
    bub = (0.6 + 0.4 * _slow(r, n, 25)).clip(0.1, 1.6)
    y = D.bp(w, 1800, 7000) * bub * 0.5
    for _ in range(int(22 * seconds)):
        m = int(0.03 * SR)
        mt = np.arange(m) / SR
        f = r.uniform(1500, 4200) * (1 + 0.5 * mt / 0.03)
        _place(y, np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * mt / 0.03) ** 2, r.uniform(0, seconds - 0.05), r.uniform(0.15, 0.45))
    env = np.minimum(1, tt / 0.12) * np.minimum(1, (seconds - tt) / (0.3 * seconds)) ** 0.7
    return amp * y * env


def lock(amp=0.6, seed=0):
    """A deadbolt thrown: clack-clunk."""
    r = _rng(seed)
    n = int(0.3 * SR)
    y = _click(r, n, 1200, 6000, 0.003) * 1.0
    m = int(0.2 * SR)
    mt = np.arange(m) / SR
    _place(y, (np.sin(2 * np.pi * 240 * mt) * np.exp(-mt / 0.02) * 0.9 + _click(r, m, 400, 3500, 0.008) * 1.2 + np.sin(2 * np.pi * 2900 * mt) * np.exp(-mt / 0.015) * 0.3)[: n - int(0.09 * SR)], 0.085)
    return amp * y * 0.7


def truck(seconds=2.2, amp=0.4, seed=0, arrive=True):
    """A pickup truck pulling up (arrive=True) or pulling away: engine, then idling / fading."""
    r = _rng(seed)
    n = int(seconds * SR)
    tt = np.arange(n) / SR
    u = tt / seconds
    rpm = (38 + 22 * (1 - u) ** 2) if arrive else (30 + 34 * u ** 1.5)
    ph = 2 * np.pi * np.cumsum(rpm) / SR
    y = np.sin(ph) + 0.7 * np.sin(2 * ph + 0.3) + 0.5 * np.sin(3 * ph) + 0.35 * np.sin(4 * ph + 1) + 0.2 * np.sin(6 * ph)
    y *= 0.6 + 0.4 * np.sin(ph * 0.5) ** 2
    y += D.bp(r.standard_normal(n), 200, 2500) * 0.25 * (0.6 + 0.4 * np.sin(ph * 0.5))
    env = (np.minimum(1, tt / 0.5) * (1.0 - 0.45 * u)) if arrive else ((1 - u) ** 0.8 * np.minimum(1, tt / 0.15))
    return amp * D.lp(y, 3500) * env * 0.4 * np.minimum(1, (seconds - tt) / 0.25)


def horn(beeps=2, amp=0.45, seed=0):
    """A friendly double tap on a truck horn."""
    n = int((beeps * 0.24 + 0.15) * SR)
    tt = np.arange(n) / SR
    y = np.zeros(n)
    for k in range(beeps):
        t0 = 0.01 + k * 0.24
        g = D.lp(((tt >= t0) & (tt < t0 + 0.15)).astype(float), 300).clip(0, 1)
        y += (D.saw(np.full(n, 415.0), SR, 0.1) + D.saw(np.full(n, 523.0), SR, 0.6)) * g
    return amp * D.peq(D.lp(D.hp(y, 300), 4200), 1800, 5.0, 1.5) * 0.35


TEST_CASES = {"shower": dict(seconds=5.0, dip_at=2.0), "pump": dict(seconds=4.0), "drips": dict(seconds=4.0),
              "mains_hum": dict(seconds=2.0, sag=True), "heat": dict(seconds=4.0), "grind": dict(seconds=4.0)}
