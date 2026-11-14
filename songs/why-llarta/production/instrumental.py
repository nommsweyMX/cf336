"""WHAT YOU REPEAT: instrumental. Renders 10 dry instrument stems from score.py.

  python3 instrumental.py <outdir>

Stems (float32 WAV, 44.1 kHz stereo, full song length, sample 0 = song time 0):
  inst_heartbeat   - the lub-dub heart (solo intro, bridge, breakdown, racing build, outro, tail) + breath/exhale
  inst_kick        - house kick fused with a beatbox lip-pop kick
  inst_beatbox     - mouth percussion: "ts" hats, "pf" snare, inward "k", rolls, boom-tss ad-libs
  inst_perc        - claps, shaker, open hats, crash/ride accents
  inst_bass        - rolling deep-house bass (saw mid + sine sub)
  inst_keys        - Rhodes (verses) + organ-house stabs (drops)
  inst_pads        - supersaw pads + disco strings
  inst_synths      - glassy arpeggio + acid line
  inst_piano       - half-time bridge piano
  inst_fx          - risers, reverse crashes, impacts, downlifters
Also writes sidechain.json (kick times + stop windows) for the mix.
"""
import json
import os
import sys

import numpy as np
import soundfile as sf

import score as S
import synth_dsp as D

SR = S.SR
BEAT, BAR = S.BEAT, S.BAR
E8, S16 = BEAT / 2, BEAT / 4
N = int(round(S.SONG_SECONDS * SR))
rng = np.random.default_rng(124)
OUT = sys.argv[1] if len(sys.argv) > 1 else "."

STOPS = [(S.t(71, 2.5), S.t(72, 0)), (S.t(119, 3.0), S.t(120, 0))]


def jit(sd=0.002):
    return float(rng.normal(0, sd))


def noise(n):
    return rng.standard_normal(n)


def section_of(b):
    for name, s, n, e, _ in S.SECTIONS:
        if s <= b < s + n:
            return name, b - s, n
    return "tail", 0, 0


def chord(b):
    return S.CHORDS[S.chord_at(b)]


# =========================================================== sound design
def lub(amp=1.0, muffle=700):
    dur = 0.24
    n = int(dur * SR)
    body = D.sat(D.sine(70 * 0 + (47 + (92 - 47) * np.exp(-np.arange(n) / SR / 0.035)), n) * D.env_exp(n, 0.07, 0.006), 2.4)
    thump = D.lp(noise(n), 200) * D.env_exp(n, 0.028, 0.002) * 0.45
    return amp * D.lp(body + thump, muffle)


def dub(amp=0.7, muffle=800):
    dur = 0.17
    n = int(dur * SR)
    body = D.sat(D.sine(60 + 58 * np.exp(-np.arange(n) / SR / 0.025), n) * D.env_exp(n, 0.045, 0.004), 2.0)
    thump = D.lp(noise(n), 260) * D.env_exp(n, 0.02, 0.002) * 0.35
    return amp * D.lp(body + thump, muffle)


def lip_pop(amp=1.0):
    n = int(0.014 * SR)
    return amp * D.bp(noise(n) * D.env_exp(n, 0.0025, 0.0003), 250, 2600)


def bb_kick(amp=1.0):
    n = int(0.26 * SR)
    body = D.sat(D.sine(50 + 85 * np.exp(-np.arange(n) / SR / 0.022), n) * D.env_exp(n, 0.075, 0.0015), 2.6)
    out = 0.85 * D.lp(body, 1500)
    p = lip_pop(0.55)
    out[: len(p)] += p
    return amp * out


def house_kick(amp=1.0):
    n = int(0.5 * SR)
    tt = np.arange(n) / SR
    f = 46 + 120 * np.exp(-tt / 0.03) + 40 * np.exp(-tt / 0.006)
    e = np.minimum(1, tt / 0.0008) * (0.75 * np.exp(-tt / 0.22) + 0.25 * np.exp(-tt / 0.05))
    body = D.sat(D.sine(f, n) * e, 1.7)
    click = D.hp(noise(n) * D.env_exp(n, 0.0015, 0.0002), 2500) * 0.25
    return amp * D.lp(body + click, 6000)


def hat(amp=0.5, open_=False):
    n = int((0.42 if open_ else 0.16) * SR)
    x = noise(n)
    t_part = D.hp(x, 3200) * D.env_exp(n, 0.005, 0.0004) * 0.55
    s_part = D.bp(x, 5200, 10500) * D.env_exp(n, 0.13 if open_ else 0.034, 0.003)
    return amp * (t_part + s_part)


def snare_pf(amp=0.8):
    n = int(0.32 * SR)
    burst = D.bp(noise(n), 1100, 6200) * D.env_exp(n, 0.08, 0.0015)
    tone = D.sine(185, n) * D.env_exp(n, 0.035, 0.001) * 0.28
    out = burst + tone
    p = lip_pop(0.8)
    out[: len(p)] += p
    return amp * out


def k_snare(amp=0.6):
    n = int(0.16 * SR)
    x = D.bp(noise(n), 2000, 7200) * D.env_exp(n, 0.028, 0.0006)
    return amp * (x + D.sine(1250, n) * D.env_exp(n, 0.012, 0.0005) * 0.25)


def breath(dur, inhale=True, amp=0.12):
    n = int(dur * SR)
    y = D.bp(noise(n), 350, 3400)
    u = np.linspace(0, 1, n)
    e = (np.sin(np.pi * u) ** 1.5) * (0.35 + 0.65 * u) if inhale else (np.sin(np.pi * u) ** 1.1) * (1 - 0.55 * u)
    return amp * y * e


def clap(amp=0.7):
    n = int(0.45 * SR)
    out = np.zeros(n)
    for k, off in enumerate((0.0, 0.009, 0.019, 0.028)):
        i = int(off * SR)
        m = n - i
        out[i:] += D.env_exp(m, 0.006 if k < 3 else 0.16, 0.0005) * (1.0 if k < 3 else 0.9)
    x = D.bp(noise(n) * out, 900, 5200)
    return amp * D.peq(x, 1300, 4, 1.2)


def shaker(amp=0.3):
    n = int(0.09 * SR)
    return amp * D.hp(noise(n), 6000) * D.env_exp(n, 0.025, 0.006)


def openhat(amp=0.4):
    n = int(0.35 * SR)
    x = noise(n)
    return amp * (D.hp(x, 7500) * D.env_exp(n, 0.11, 0.001) + 0.4 * D.bp(x, 9000, 14000) * D.env_exp(n, 0.2, 0.001))


def crash(amp=0.6, seconds=2.6):
    n = int(seconds * SR)
    tt = np.arange(n) / SR
    x = D.hp(noise(n), 3000) * np.exp(-tt / 0.9)
    metal = sum(np.sin(2 * np.pi * f * tt + rng.uniform(0, 6)) for f in (3120, 4385, 5277, 6650, 7911, 9300)) / 6
    x += 0.25 * metal * np.exp(-tt / 0.5)
    return amp * D.lp(x, 12000)


def sub_boom(amp=0.9):
    n = int(1.6 * SR)
    tt = np.arange(n) / SR
    return amp * D.sat(D.sine(30 + 40 * np.exp(-tt / 0.08), n) * np.exp(-tt / 0.7), 1.4)


def riser(seconds, f0=300, f1=9000, amp=0.5):
    n = int(seconds * SR)
    u = np.linspace(0, 1, n)
    fc = f0 * (f1 / f0) ** (u ** 1.4)
    x = D.svf(noise(n), fc, 0.6, SR, 1)
    siren = D.sine(110 * 2 ** (3 * u ** 1.3), n) * 0.15
    return amp * (x * 0.8 + siren) * (u ** 2.2)


def downlifter(seconds, amp=0.4):
    n = int(seconds * SR)
    u = np.linspace(0, 1, n)
    fc = 9000 * (250 / 9000) ** u
    return amp * D.svf(noise(n), fc, 0.7, SR, 0) * (1 - u) ** 1.5


def reverse_crash(seconds, amp=0.45):
    x = crash(1.0, seconds)[::-1]
    u = np.linspace(0, 1, len(x))
    return amp * x * u ** 2


# ---------------------------------------------------------------- tonal
def bass_note(mid_midi, dur, cut=600, vel=1.0, sub=True):
    n = int((dur + 0.08) * SR)
    f = D.mtof(mid_midi)
    tt = np.arange(n) / SR
    osc = D.saw(np.full(n, f), SR, rng.uniform(0, 1))
    fc = cut * (1 + 2.5 * vel * np.exp(-tt / 0.07))
    mid = D.svf(osc, fc, 0.7, SR, 0) * 0.5
    sub_m = mid_midi - 12 if mid_midi >= 40 else mid_midi
    s = D.sine(D.mtof(sub_m), n) * 0.85 if sub else 0
    e = D.adsr(n, 0.004, 0.14, 0.72, 0.035, dur)
    return vel * D.sat((mid + s) * e, 1.3)


def bass_mid(root):
    m = root
    while m > 42:
        m -= 12
    while m < 31:
        m += 12
    return m


def rhodes(voicing, dur, vel=0.6):
    n = int((dur + 0.5) * SR)
    tt = np.arange(n) / SR
    out = np.zeros(n)
    for m in voicing:
        f = D.mtof(m)
        I = (1.4 + 1.2 * vel) * np.exp(-tt / 0.28) + 0.2
        y = np.sin(2 * np.pi * f * tt + I * np.sin(2 * np.pi * f * tt))
        y += 0.12 * np.sin(2 * np.pi * 7.02 * f * tt) * np.exp(-tt / 0.03)
        out += y
    e = D.adsr(n, 0.002, 1.8, 0.0, 0.25, dur)
    out = out * e / len(voicing)
    trem = 0.5 + 0.5 * np.sin(2 * np.pi * 4.3 * tt)
    st = np.stack([out * (0.65 + 0.35 * trem), out * (0.65 + 0.35 * (1 - trem))], axis=1)
    return vel * st


def organ(voicing, dur, vel=0.6):
    n = int((dur + 0.06) * SR)
    tt = np.arange(n) / SR
    out = np.zeros(n)
    for m in voicing:
        f = D.mtof(m)
        for h, a in ((0.5, 0.35), (1, 1.0), (2, 0.75), (3, 0.5), (4, 0.45), (6, 0.2), (8, 0.22)):
            if f * h < 15000:
                out += a * np.sin(2 * np.pi * f * h * tt * (1 + rng.normal(0, 0.0007)))
        out += 0.3 * np.sin(2 * np.pi * f * 3 * tt) * np.exp(-tt / 0.04)  # percussion
    e = D.adsr(n, 0.003, 0.22, 0.25, 0.03, dur)
    out = D.lp(out * e / len(voicing), 7000)
    return vel * D.sat(out * 0.6, 1.2)


def supersaw(voicing, dur, cut=3000, vel=0.5, attack=0.25, release=0.7, extra_top=True):
    notes = list(voicing) + ([voicing[-1] + 12] if extra_top else [])
    n = int((dur + release + 0.1) * SR)
    L = np.zeros(n)
    R = np.zeros(n)
    det = (-18, -11, -5, 0, 5, 11, 18)
    for m in notes:
        for k, c in enumerate(det):
            y = D.saw(np.full(n, D.mtof(m) * 2 ** (c / 1200)), SR, rng.uniform(0, 1))
            if k % 2:
                L += y * 0.8
                R += y * 0.2
            else:
                L += y * 0.2
                R += y * 0.8
    st = np.stack([L, R], axis=1) / (len(notes) * 3.5)
    st = D.lp(D.hp(st, 160), cut, order=4)
    e = D.adsr(n, attack, 1.0, 1.0, release, dur)
    return vel * st * e[:, None]


def strings(voicing, dur, vel=0.4, attack=0.18):
    notes = [m + 12 for m in voicing]
    n = int((dur + 0.6) * SR)
    tt = np.arange(n) / SR
    vib = 1 + 0.004 * np.sin(2 * np.pi * 5.2 * tt) * np.clip((tt - 0.3) / 0.3, 0, 1)
    L = np.zeros(n)
    R = np.zeros(n)
    for m in notes:
        for k, c in enumerate((-7, 0, 7)):
            y = D.saw(D.mtof(m) * 2 ** (c / 1200) * vib, SR, rng.uniform(0, 1))
            L += y * (0.7 if k != 1 else 0.5)
            R += y * (0.7 if k != 0 else 0.5)
    st = np.stack([L, R], axis=1) / (len(notes) * 2.5)
    st = D.lp(D.hp(st, 250), 5200, order=4)
    st = D.peq(st, 2800, 3, 0.8)
    return vel * st * D.adsr(n, attack, 1.0, 1.0, 0.5, dur)[:, None]


def string_stab(voicing, vel=0.35):
    n = int(0.25 * SR)
    tt = np.arange(n) / SR
    y = np.zeros(n)
    for m in voicing:
        for o in (12, 24):
            y += D.saw(np.full(n, D.mtof(m + o)), SR, rng.uniform(0, 1))
    y = D.lp(y / (2 * len(voicing)), 6000) * D.env_exp(n, 0.07, 0.004)
    return vel * y


def string_run(lo, hi, seconds, vel=0.4):
    n = int(seconds * SR)
    u = np.linspace(0, 1, n)
    f = D.mtof(lo + (hi - lo) * u ** 0.7)
    y = sum(D.saw(f * 2 ** (c / 1200), SR, rng.uniform(0, 1)) for c in (-8, 0, 8)) / 3
    return vel * D.lp(y, 6000) * np.sin(np.pi * u) ** 0.5


def pluck(midi, vel=0.4):
    n = int(0.55 * SR)
    tt = np.arange(n) / SR
    f = D.mtof(midi)
    y = np.sin(2 * np.pi * f * tt + 1.6 * np.exp(-tt / 0.05) * np.sin(2 * np.pi * 3.5 * f * tt))
    y += 0.3 * np.sin(2 * np.pi * 2 * f * tt) * np.exp(-tt / 0.06)
    return vel * y * D.env_exp(n, 0.15, 0.002)


def piano_note(midi, dur, vel=0.6):
    f0 = D.mtof(midi)
    n = int((dur + 1.2) * SR)
    tt = np.arange(n) / SR
    B = 0.00035
    tau0 = float(np.clip(3.2 * (261.6 / f0) ** 0.45, 0.8, 6.0))
    y = np.zeros(n)
    for k in range(1, 13):
        fk = k * f0 * np.sqrt(1 + B * k * k)
        if fk > 16000:
            break
        a = (1 / k ** 1.15) * (0.6 + 0.6 * vel if k > 3 else 1)
        tk = tau0 / (1 + 0.32 * k)
        for det in (-0.7, 0.7):
            y += a * np.sin(2 * np.pi * fk * 2 ** (det / 1200) * tt + rng.uniform(0, 6)) * np.exp(-tt / tk)
    ham = D.lp(noise(n) * D.env_exp(n, 0.004, 0.0003), 2500) * 0.3
    y = (y / 8 + ham) * np.minimum(1, tt / 0.002)
    rel = tt > dur
    y[rel] *= np.exp(-(tt[rel] - dur) / 0.35)
    return vel * D.lp(y, 3000 + 5000 * vel)


# =========================================================== arrangement
def arr(b):
    name, i, n = section_of(b)
    A = dict(kick=0.0, hb="kick", hats=0.0, ghost=0.0, snare=0.0, claps=0.0, shaker=0.0, openhat=0.0,
             bass=None, bass_cut=600, bass_vel=1.0, bass_pat="full", rhodes=0.0, organ=0.0, organ_pat="full", pad=0.0, pad_cut=1500,
             strings=0.0, stabs=0.0, arp=0.0, arp_oct=12, acid=0.0, acid_cut=500, piano=0.0, fill=None)
    if name in ("verse1", "verse2"):
        A.update(kick=0.78, hats=0.34, ghost=0.10, snare=0.6, bass="roll", bass_cut=380 + 10 * i, rhodes=0.5,
                 pad=0.10, pad_cut=1300, bass_vel=0.72, bass_pat="light" if i < 8 else "full")
        if i >= 8:
            A.update(organ=0.22, organ_pat="sparse", ghost=0.16)
        if name == "verse2":
            A.update(claps=0.3)
            if i >= 8:
                A.update(acid=0.16, acid_cut=360 + 30 * (i - 8))
        if i % 8 == 7:
            A["fill"] = "pf-k"
    elif name in ("pre1", "pre2"):
        A.update(kick=0.85 if i < 6 else 0, bass_vel=0.85, hats=0.4 if i < 6 else 0.25, ghost=0.12, snare=0.7 if i < 6 else 0,
                 bass="long", bass_cut=400 + 140 * i, rhodes=0.35 if i < 4 else 0.0,
                 pad=0.2 + 0.04 * i, pad_cut=1400 * 1.33 ** i, strings=0.12 + 0.04 * i,
                 arp=0.18 + 0.05 * (i - 3) if i >= 3 else 0.0)
        if name == "pre2":
            A.update(claps=0.25 if i < 6 else 0, pad=A["pad"] + 0.05)
    elif name in ("chorus1", "chorus2", "post2", "final", "tag") or (name == "post1" and i < 4):
        big = name in ("chorus2", "final", "tag")
        A.update(kick=1, hats=0.4, ghost=0.15, snare=0.72, claps=0.72, shaker=0.32, openhat=0.38,
                 bass="roll", bass_cut=1050 if not big else 1250, organ=0.55, pad=0.42 if not big else 0.48,
                 pad_cut=7000 if not big else 9000, arp=0.3, arp_oct=12 if not big else 24)
        if name.startswith("post"):
            A.update(pad=0.3, organ=0.5)
        if big:
            A.update(strings=0.3 if name == "chorus2" else 0.42, stabs=0.3 if name != "chorus2" else 0.0)
        if name in ("final", "tag"):
            A.update(acid=0.11, acid_cut=700)
        if i % 4 == 3 and name in ("chorus1", "chorus2", "final"):
            A["fill"] = "pf-k" if i == 7 else None
    elif name == "post1":  # bars 44-47 instrumental drop
        A.update(kick=1, hats=0.42, ghost=0.18, snare=0.7, claps=0.6, shaker=0.35, openhat=0.4, bass="roll",
                 bass_cut=900, organ=0.3, organ_pat="sparse", pad=0.12, pad_cut=3000, acid=0.45,
                 acid_cut=420 + 160 * (i - 4))
        if i == 7:
            A["fill"] = "k-roll"
    elif name == "verse3":
        A.update(kick=1 if i >= 6 else 0, hats=0.5, ghost=0.28, snare=0.8, bass="sub" if i < 6 else "roll",
                 bass_cut=700, acid=0.42, acid_cut=360 + 85 * i, hb="kick" if i >= 6 else "none")
    elif name == "bridge":
        A.update(hb="half", bass="halfsub", piano=1.0, pad=0.2, pad_cut=3400, strings=0.12 if i >= 8 else 0.0)
    elif name == "breakdown":
        A.update(hb="half_loud", pad=0.3, pad_cut=1300)
    elif name == "build":
        A.update(hb="race", pad=0.3 + 0.07 * i, pad_cut=1800 * 1.6 ** i, strings=0.2 + 0.06 * i,
                 arp=0.25 if i >= 2 else 0.0, bass="pulse", bass_cut=500 + 250 * i)
    elif name == "outro":
        if i < 2:
            A.update(kick=1, hats=0.38, ghost=0.12, snare=0.65, bass="roll", bass_cut=700, organ=0.3,
                     organ_pat="sparse", pad=0.3, pad_cut=3500)
        elif i < 4:
            A.update(kick=1, hats=0.3, pad=0.26, pad_cut=2500, hb="kick")
        else:
            A.update(hb="half_out", hats=0.22 * (1 - (i - 4) / 2) if i < 6 else 0.0, pad=0.22 - 0.035 * (i - 4),
                     pad_cut=1800 - 200 * (i - 4))
    return A


# =========================================================== render
def render():
    B = {k: D.Bus(k, N) for k in ("inst_heartbeat", "inst_kick", "inst_beatbox", "inst_perc", "inst_bass", "inst_keys",
                                  "inst_pads", "inst_synths", "inst_piano", "inst_fx")}
    hb, kick, bbx, perc, bass, keys, pads, syn, pno, fx = (B[k] for k in B)
    kick_times = []

    def heartbeat(tm, amp=1.0, gap=E8, muffle=700):
        hb.add(lub(1.0, muffle), tm, gain=amp)
        hb.add(dub(0.7, muffle + 100), tm + gap, gain=amp)

    def put_kick(tm, level=1.0, hb_layer=True):
        kick.add(house_kick(1.0), tm, gain=0.9 * level)
        kick.add(bb_kick(1.0), tm + 0.001, gain=0.35 * level)
        if hb_layer:
            hb.add(lub(0.5, 900), tm, gain=0.55 * level)
        kick_times.append(tm)

    # ---------------- intro (bars 0-7): the heartbeat doubles into the house tempo
    hb.add(breath(0.5, True, 0.10), 0.02)
    for k in range(8):
        heartbeat(S.t(0, 2 * k), amp=0.32 + 0.3 * k / 7, muffle=550 + 40 * k)
    for k in range(4):
        heartbeat(S.t(4, 2 * k), amp=0.66, muffle=900)
    for i in range(16):
        tb = S.t(4, i)
        a = 0.12 + 0.3 * i / 15
        bbx.add(hat(a), tb + E8 + jit(0.003), pan=0.15)
        if i >= 8:
            bbx.add(hat(a * 0.35), tb + 3 * S16 + jit(0.004), pan=-0.2)
    bbx.add(k_snare(0.45), S.t(6) - 2 * S16)
    bbx.add(k_snare(0.6), S.t(6) - S16)
    for b in (6, 7):
        for beat in range(4):
            tb = S.t(b, beat)
            put_kick(tb, 0.85 if b == 6 else 0.95)
            hb.add(dub(0.35, 900), tb + E8 + 0.01, gain=0.9)
            bbx.add(hat(0.42, open_=(beat == 3 and b == 6)), tb + E8 + jit(0.003), pan=0.15)
            bbx.add(hat(0.14), tb + 3 * S16 + jit(0.004), pan=-0.2)
            if beat in (1, 3) and not (b == 7 and beat == 3):
                bbx.add(snare_pf(0.78), tb + jit())
    for s, a in enumerate((0.45, 0.5, 0.6, 0.75)):
        bbx.add(k_snare(a), S.t(7, 3) + s * S16, pan=0.1 * (-1) ** s)
    pads.add(supersaw(chord(6)["voicing"], 2 * BAR, cut=650, vel=0.16, attack=1.2), S.t(6))
    fx.add(reverse_crash(2 * BAR, 0.3), S.t(8) - 2 * BAR)

    # ---------------- bars 8..139
    acid_bars = []
    for b in range(8, S.TOTAL_BARS):
        A = arr(b)
        name, i, n = section_of(b)
        ch = chord(b)
        v, root = ch["voicing"], ch["root"]
        # drums
        for beat in range(4):
            tb = S.t(b, beat)
            if A["kick"]:
                put_kick(tb + jit(0.001), float(A["kick"]), hb_layer=(A["hb"] == "kick"))
            if A["hats"]:
                bbx.add(hat(A["hats"], open_=(beat == 3 and i % 2 == 1 and A["fill"] is None)), tb + E8 + jit(0.003), pan=0.15)
            if A["ghost"]:
                bbx.add(hat(A["ghost"]), tb + 3 * S16 + jit(0.004), pan=-0.2)
                if name == "verse3":
                    bbx.add(hat(A["ghost"] * 0.8), tb + S16 + jit(0.004), pan=-0.3)
            if A["snare"] and beat in (1, 3) and not (A["fill"] and beat == 3):
                bbx.add(snare_pf(A["snare"]), tb + jit(0.002))
            if A["claps"] and beat in (1, 3):
                perc.add(clap(A["claps"]), tb + 0.004, pan=0.05)
            if A["openhat"]:
                perc.add(openhat(A["openhat"]), tb + E8 + jit(0.002), pan=-0.15)
            if A["shaker"]:
                for s in range(4):
                    perc.add(shaker(A["shaker"] * (1.0 if s == 2 else 0.55)), tb + s * S16 + jit(0.003), pan=0.35)
        if name == "verse3":  # inward-K accents
            for p in (7, 15):
                bbx.add(k_snare(0.42), S.t(b) + p * S16 + jit(), pan=0.2)
        if A["fill"] == "pf-k":
            tb = S.t(b, 3)
            bbx.add(snare_pf(0.8), tb)
            bbx.add(k_snare(0.5), tb + 2 * S16)
            bbx.add(k_snare(0.62), tb + 3 * S16)
        elif A["fill"] == "k-roll":
            for s, a in enumerate((0.45, 0.5, 0.6, 0.75)):
                bbx.add(k_snare(a), S.t(b, 3) + s * S16, pan=0.1 * (-1) ** s)

        # heartbeat modes
        if A["hb"] == "half":
            for k in (0, 2):
                heartbeat(S.t(b, k), amp=0.55, muffle=650)
        elif A["hb"] == "half_loud":
            for k in (0, 2):
                heartbeat(S.t(b, k), amp=0.72, muffle=560)
        elif A["hb"] == "half_out":
            for k in (0, 2):
                heartbeat(S.t(b, k), amp=0.6 - 0.05 * (i - 4), muffle=800 - 40 * (i - 4))
        elif A["hb"] == "race":
            if i == 0:
                hits = [k for k in range(4)]
            elif i == 1:
                hits = [k * 0.5 for k in range(8)]
            elif i == 2:
                hits = [k * 0.25 for k in range(16)]
            else:
                hits = [k * 0.25 for k in range(12)]  # stop at beat 3
            for j, k in enumerate(hits):
                amp = 0.55 + 0.35 * (k / 4)
                if i < 2:
                    heartbeat(S.t(b, k), amp=amp, gap=min(E8, 0.5 * BEAT * (1 if i == 0 else 0.5)), muffle=700 + 150 * i)
                else:
                    hb.add(lub(1.0, 900 + 100 * i), S.t(b, k), gain=amp)
            # snare roll on top of the race
            step = {0: 1.0, 1: 0.5, 2: 0.25, 3: 0.125}[i]
            k = 0.0
            while k < (4 if i < 3 else 3):
                bbx.add(snare_pf(0.35 + 0.12 * i + 0.08 * k) if i < 2 else k_snare(0.4 + 0.08 * i + 0.05 * k), S.t(b, k))
                k += step

        # bass
        mid = bass_mid(root)
        if A["bass"] == "roll":
            if A["bass_pat"] == "light":
                pat = [(2, 2, 0, 1.0), (10, 2, 0, 1.0), (14, 1.5, 0, 0.75)]
            else:
                pat = [(2, 2, 0, 1.0), (6, 1.5, 0, 0.9), (7, 1, 12, 0.65), (10, 2, 0, 1.0), (13, 1, 7, 0.6), (14, 1.5, 0, 0.9)]
            for pos, ln, iv, vel in pat:
                bass.add(bass_note(mid + iv, ln * S16, A["bass_cut"], vel * A["bass_vel"]), S.t(b) + pos * S16)
        elif A["bass"] == "long":
            bass.add(bass_note(mid, BAR - 0.05, A["bass_cut"], 0.85 * A["bass_vel"]), S.t(b))
        elif A["bass"] == "pulse":
            for k in range(8 if i < 3 else 6):
                bass.add(bass_note(mid, E8 * 0.8, A["bass_cut"], 0.75 + 0.03 * k), S.t(b, 0.5 * k))
        elif A["bass"] == "sub":
            bass.add(bass_note(mid, BAR - 0.05, 250, 0.7), S.t(b))
        elif A["bass"] == "halfsub" and (b - 92) % 2 == 0:
            bass.add(bass_note(mid, 2 * BAR - 0.1, 220, 0.6), S.t(b))

        # keys
        if A["rhodes"]:
            keys.add(rhodes(v, 1.5 * BEAT, A["rhodes"]), S.t(b, 0))
            keys.add(rhodes(v, 1.25 * BEAT, A["rhodes"] * 0.7), S.t(b, 2.5))
        if A["organ"]:
            pat = [0, 3, 6, 9, 12, 14] if A["organ_pat"] == "full" else [3, 11]
            accents = [1.0, 0.8, 0.9, 0.8, 0.9, 0.7]
            for k, p in enumerate(pat):
                keys.add(np.stack([organ(v, S16 * 1.2, A["organ"] * accents[k % 6])] * 2, axis=1), S.t(b) + p * S16)

        # pads / strings (chord segments: one bar each, release overlaps)
        if A["pad"]:
            pads.add(supersaw(v, BAR, cut=A["pad_cut"], vel=A["pad"], attack=0.08 if name in ("chorus1", "chorus2", "final", "tag") else 0.25), S.t(b))
        if A["strings"]:
            pads.add(strings(v, BAR, A["strings"]), S.t(b))
        if A["stabs"]:
            for p in (2, 6, 10, 14):
                pads.add(string_stab(v, A["stabs"]), S.t(b) + p * S16, pan=0.25 if p % 4 == 2 else -0.25)

        # arp
        if A["arp"]:
            tones = [m + A["arp_oct"] for m in v] + [v[0] + A["arp_oct"] + 12]
            order = [0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1, 2, 3]
            for s in range(16):
                m = tones[order[s] % len(tones)]
                syn.add(pluck(m, A["arp"] * (1.0 if s % 4 == 0 else 0.7)), S.t(b) + s * S16, pan=0.4 * np.sin(s * 0.7))

        # acid (collected, rendered as continuous lines below)
        if A["acid"]:
            acid_bars.append((b, A["acid"], A["acid_cut"], ch))

        # piano (bridge, half-time)
        if A["piano"]:
            lh = [root + 12] if root + 12 >= 48 else [root + 12, root + 24]
            for m in lh:
                pno.add(piano_note(m, 2 * BEAT, 0.42), S.t(b, 0), pan=-0.2)
            for m in v:
                pno.add(piano_note(m + 12, 1.8 * BEAT, 0.45), S.t(b, 0) + 0.01 * v.index(m), pan=0.15)
            for m in v[1:]:
                pno.add(piano_note(m + 12, 1.3 * BEAT, 0.3), S.t(b, 2) + 0.012 * v.index(m), pan=0.15)
            pno.add(piano_note(v[-1] + 24, 0.8 * BEAT, 0.28), S.t(b, 3.5), pan=0.3)

    # ---------------- acid lines (continuous, with slides)
    pat = [(0, 0, 1, 0), (2, 0, 0, 0), (3, 12, 1, 1), (5, 0, 0, 0), (6, 3, 0, 0), (8, 0, 1, 0),
           (10, 7, 0, 1), (11, 5, 0, 0), (12, 0, 1, 0), (14, 12, 0, 1), (15, 10, 0, 0)]
    runs = []
    for item in acid_bars:
        if runs and runs[-1][-1][0] == item[0] - 1:
            runs[-1].append(item)
        else:
            runs.append([item])
    for run in runs:
        b0 = run[0][0]
        t0 = S.t(b0)
        n = int((len(run) * BAR + 0.3) * SR)
        freq = np.zeros(n)
        cut = np.zeros(n)
        amp = np.zeros(n)
        for (b, lvl, base_cut, ch) in run:
            root = bass_mid(ch["root"]) + 12
            for pos, iv, acc, slide in pat:
                st = int((S.t(b) + pos * S16 - t0) * SR)
                ln = int(S16 * SR * (1.0 if slide else 0.55))
                f = D.mtof(root + iv)
                end = min(st + int(S16 * SR), n)
                freq[st:end] = f
                if slide and st > 0:
                    prev = freq[max(st - 1, 0)] or f
                    g = int(0.045 * SR)
                    freq[st:st + g] = np.geomspace(prev, f, g)
                tt = np.arange(end - st) / SR
                cut[st:end] = base_cut * (1 + (3.5 if acc else 1.8) * np.exp(-tt / (0.09 if acc else 0.06)))
                a = np.zeros(end - st)
                a[: min(ln, end - st)] = lvl * (1.0 if acc else 0.7)
                amp[st:end] = a
        freq[freq == 0] = 100.0
        amp = D.lp(amp, 300)
        y = D.svf(D.saw(freq, SR, 0.0), cut, 0.16, SR, 0)
        y = D.sat(y * 1.8, 2.0) * amp
        syn.add(y, t0, pan=-0.1)

    # ---------------- FX
    for b in (32, 72, 120):
        fx.add(crash(0.55), S.t(b), pan=0.1)
        fx.add(sub_boom(0.8), S.t(b))
    for b in (40, 80, 128):
        fx.add(crash(0.3), S.t(b), pan=-0.1)
    fx.add(riser(4 * BAR - 0.05, amp=0.45), S.t(28))
    fx.add(riser(S.t(71, 2.5) - S.t(66), amp=0.5), S.t(66))
    fx.add(riser(S.t(119, 3.0) - S.t(116), amp=0.55), S.t(116))
    fx.add(riser(2 * BAR, amp=0.25), S.t(6))
    fx.add(reverse_crash(BAR, 0.35), S.t(92) - BAR)
    fx.add(downlifter(2 * BAR, 0.35), S.t(92))
    fx.add(downlifter(2 * BAR, 0.3), S.t(132))
    fx.add(reverse_crash(BAR, 0.3), S.t(48) - BAR)
    fx.add(sub_boom(0.5), S.t(92))
    # disco string run into the final chorus
    pads.add(string_run(55, 79, BEAT * 0.9, 0.35), S.t(120), pan=0.0)

    # ---------------- ad-libs from the score
    for bar, beat, kind in S.ADLIBS:
        tb = S.t(bar, beat)
        if kind == "boom-tss-ka-tss":
            bbx.add(bb_kick(0.9), tb)
            bbx.add(hat(0.5), tb + E8 * 0.5)
            bbx.add(k_snare(0.65), tb + E8)
            bbx.add(hat(0.5), tb + E8 * 1.5)
        elif kind == "ka-tss":
            bbx.add(k_snare(0.65), tb)
            bbx.add(hat(0.5), tb + E8 * 0.5)
        elif kind == "boom-tss":
            bbx.add(bb_kick(0.9), tb)
            bbx.add(hat(0.5), tb + E8 * 0.5)
        elif kind == "lubdub2":
            heartbeat(tb, 0.8)
            heartbeat(tb + BEAT, 0.8)
        elif kind == "kroll":
            for s, a in enumerate((0.45, 0.5, 0.6, 0.75, 0.8, 0.85, 0.9, 0.95)):
                bbx.add(k_snare(a), tb + s * S16 / 1.0 * 0.5)

    # ---------------- tail: heart slows to a stop, then an exhale
    tm = S.t(S.TOTAL_BARS)
    for k in range(3):
        heartbeat(tm, amp=0.5 - 0.03 * k, muffle=760 - 30 * k)
        tm += 2 * BEAT
    for k, iv in enumerate([1.0, 1.08, 1.18, 1.3]):
        heartbeat(tm, amp=0.42 - 0.07 * k, gap=E8 + 0.02 * k, muffle=650 - 60 * k)
        tm += iv
    hb.add(breath(1.3, False, 0.09), tm + 0.3)

    # ---------------- the stops: true silence
    for a, z in STOPS:
        ia, iz = int(a * SR), int(z * SR)
        f = int(0.008 * SR)
        for bus in B.values():
            bus.x[ia - f:ia] *= np.linspace(1, 0, f, dtype=np.float32)[:, None]
            bus.x[ia:iz] = 0
    return B, kick_times


if __name__ == "__main__":
    import time

    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    B, kick_times = render()
    for k, bus in B.items():
        pk = float(np.max(np.abs(bus.x)))
        sf.write(os.path.join(OUT, k + ".wav"), bus.x, SR, subtype="FLOAT")
        print(f"{k:16s} peak {pk:6.3f}  rms {float(np.sqrt(np.mean(bus.x ** 2))):.4f}")
    json.dump({"kick_times": kick_times, "stops": STOPS}, open(os.path.join(OUT, "sidechain.json"), "w"))
    print(f"rendered {N / SR:.1f}s in {time.time() - t0:.0f}s")
