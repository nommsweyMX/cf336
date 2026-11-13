"""Synthesize heartbeat + human-beatbox samples for the WHY-LLARTA Suno song.

Concept: a resting heartbeat at 62 BPM is exactly half of the 124 BPM house tempo,
so the heart "doubles" into the four-on-the-floor kick. The beatbox layers are
mouth-style sounds (lip-pop kick, "ts" hats, "pf" snare, inward "k"), not drum
machine samples, so they read as a human body making the beat.

Outputs (WAV 44.1 kHz/16-bit stereo + MP3 320k):
  01_heartbeat_intro_62-to-124bpm  - breath, solo heartbeat, beatbox joins, doubles to 124
  02_heartbeat_beatbox_loop_124bpm - seamless 4-bar loop to run under the whole song
  03_heartbeat_outro_124-to-62bpm  - groove strips back, heart halves, slows, last beat, exhale
"""
import os
import subprocess
import sys

import numpy as np
from scipy import signal
from scipy.io import wavfile

SR = 44100
BPM = 124.0
BEAT = 60.0 / BPM          # 0.4839 s
BAR = 4 * BEAT             # 1.9355 s
E8 = BEAT / 2              # eighth note
S16 = BEAT / 4             # sixteenth note
OUT = sys.argv[1] if len(sys.argv) > 1 else "."
rng = np.random.default_rng(336)  # Casa Francisca 336 — deterministic renders


# ---------------------------------------------------------------- dsp helpers
def _sos(kind, f, order=2):
    return signal.butter(order, f, btype=kind, fs=SR, output="sos")


def lp(x, fc, order=2):
    return signal.sosfilt(_sos("low", fc, order), x)


def hp(x, fc, order=2):
    return signal.sosfilt(_sos("high", fc, order), x)


def bp(x, lo, hi, order=2):
    return signal.sosfilt(_sos("band", [lo, hi], order), x)


def env(n, tau, attack=0.002):
    t = np.arange(n) / SR
    a = np.clip(t / attack, 0, 1) if attack > 0 else 1.0
    return a * np.exp(-t / tau)


def sweep(dur, f_start, f_end, tau_f):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = f_end + (f_start - f_end) * np.exp(-t / tau_f)
    return np.sin(2 * np.pi * np.cumsum(f) / SR)


def sat(x, drive):
    return np.tanh(drive * x) / np.tanh(drive)


def noise(n):
    return rng.standard_normal(n)


# ---------------------------------------------------------------- the body
def lub(amp=1.0, muffle=700):
    """S1 heart sound: low pitch-dropping thump, saturated so it survives phone speakers."""
    dur = 0.24
    body = sweep(dur, 92, 47, 0.035) * env(int(dur * SR), 0.07, 0.006)
    body = sat(body, 2.4)
    n = len(body)
    thump = lp(noise(n), 200) * env(n, 0.028, 0.002) * 0.45
    return amp * lp(body + thump, muffle)


def dub(amp=0.7, muffle=800):
    """S2 heart sound: shorter, a little higher."""
    dur = 0.17
    body = sweep(dur, 118, 60, 0.025) * env(int(dur * SR), 0.045, 0.004)
    body = sat(body, 2.0)
    n = len(body)
    thump = lp(noise(n), 260) * env(n, 0.02, 0.002) * 0.35
    return amp * lp(body + thump, muffle)


# ---------------------------------------------------------------- the mouth
def lip_pop(amp=1.0):
    n = int(0.014 * SR)
    return amp * bp(noise(n) * env(n, 0.0025, 0.0003), 250, 2600)


def bb_kick(amp=1.0):
    """Beatbox 'B' kick fused with the heartbeat lub: lip pop + chesty sweep."""
    dur = 0.26
    n = int(dur * SR)
    body = sat(sweep(dur, 135, 50, 0.022) * env(n, 0.075, 0.0015), 2.6)
    out = 0.85 * lp(body, 1500)
    pop = lip_pop(0.55)
    out[: len(pop)] += pop
    return amp * out


def hat(amp=0.5, open_=False):
    """Beatbox 't' / 'ts' / 'tsss'."""
    dur = 0.42 if open_ else 0.16
    n = int(dur * SR)
    x = noise(n)
    t_part = hp(x, 3200) * env(n, 0.005, 0.0004) * 0.55
    s_part = bp(x, 5200, 10500) * env(n, 0.13 if open_ else 0.034, 0.003)
    return amp * (t_part + s_part)


def snare_pf(amp=0.8):
    """Beatbox 'Pf' snare: lip burst + breathy mid noise + a little body."""
    dur = 0.32
    n = int(dur * SR)
    burst = bp(noise(n), 1100, 6200) * env(n, 0.08, 0.0015)
    tone = np.sin(2 * np.pi * 185 * np.arange(n) / SR) * env(n, 0.035, 0.001) * 0.28
    out = burst + tone
    pop = lip_pop(0.8)
    out[: len(pop)] += pop
    return amp * out


def k_snare(amp=0.6):
    """Inward 'K' snare: sharp, papery."""
    n = int(0.16 * SR)
    x = bp(noise(n), 2000, 7200) * env(n, 0.028, 0.0006)
    ring = np.sin(2 * np.pi * 1250 * np.arange(n) / SR) * env(n, 0.012, 0.0005) * 0.25
    return amp * (x + ring)


def breath(dur, inhale=True, amp=0.12):
    n = int(dur * SR)
    y = bp(noise(n), 350, 3400)
    t = np.linspace(0, 1, n)
    e = (np.sin(np.pi * t) ** 1.5) * (0.35 + 0.65 * t) if inhale else (np.sin(np.pi * t) ** 1.1) * (1 - 0.55 * t)
    return amp * y * e


# ---------------------------------------------------------------- arranging
class Track:
    def __init__(self, seconds):
        self.n = int(seconds * SR)
        self.dry = np.zeros((self.n, 2))
        self.send = np.zeros((self.n, 2))  # reverb send

    def put(self, x, t, gain=1.0, pan=0.0, verb=0.1, jitter=0.0):
        if jitter:
            t = t + rng.normal(0, jitter)
        i = int(round(t * SR))
        if i >= self.n or i + len(x) <= 0:
            return
        j0 = max(0, -i)
        x = x[j0:]
        i = max(i, 0)
        m = min(len(x), self.n - i)
        l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
        seg = x[:m] * gain * np.sqrt(2)
        self.dry[i : i + m, 0] += seg * l
        self.dry[i : i + m, 1] += seg * r
        self.send[i : i + m, 0] += seg * l * verb
        self.send[i : i + m, 1] += seg * r * verb


def room_ir(seconds=0.55, decay=0.13):
    n = int(seconds * SR)
    pre = int(0.012 * SR)
    irs = []
    for _ in range(2):
        x = noise(n) * np.exp(-np.arange(n) / SR / decay)
        x = lp(hp(x, 250), 5200)
        x = np.concatenate([np.zeros(pre), x])
        irs.append(x / np.sqrt(np.sum(x ** 2)))
    return irs


IR = room_ir()


def render(track, tail_fade=0.0, loop=False):
    wet = np.stack([signal.fftconvolve(track.send[:, c], IR[c])[: track.n] for c in range(2)], axis=1)
    mix = track.dry + wet
    if loop:  # wrap reverb/decay tails that would cross the loop point back to the start
        pass
    mix = sat(mix * 1.1, 1.3)  # gentle glue
    if tail_fade:
        k = int(tail_fade * SR)
        mix[-k:] *= np.linspace(1, 0, k)[:, None] ** 2
    peak = np.max(np.abs(mix))
    return mix / peak * 10 ** (-1.0 / 20)  # -1 dBFS peak


def heartbeat(tr, t, amp=1.0, gap=E8, muffle=700, verb=0.05):
    tr.put(lub(1.0, muffle), t, gain=amp, verb=verb)
    tr.put(dub(0.7, muffle + 100), t + gap, gain=amp, verb=verb)


def groove_bar(tr, t0, level=1.0, snare=True, hats=True, ghosts=True, fill=None, kick_amp=1.0):
    """One bar of the heartbeat/beatbox house groove at 124 BPM."""
    for b in range(4):
        tb = t0 + b * BEAT
        # kick = heartbeat lub fused with a beatbox lip-pop kick
        tr.put(bb_kick(kick_amp), tb, gain=0.95 * level, verb=0.06, jitter=0.002)
        tr.put(lub(0.55, 900), tb, gain=level, verb=0.03)
        # the heart's "dub" becomes the off-beat ghost (deep-house off-beat feel)
        tr.put(dub(0.35, 900), tb + E8 + 0.01, gain=level, verb=0.03)
        if hats:
            open_ = (b == 3 and fill is None)
            tr.put(hat(0.42, open_=open_), tb + E8, gain=level, pan=0.15, verb=0.12, jitter=0.003)
            if ghosts:
                tr.put(hat(0.14), tb + S16 * 3, gain=level, pan=-0.2, verb=0.1, jitter=0.004)
        if snare and b in (1, 3) and not (fill and b == 3):
            tr.put(snare_pf(0.78), tb, gain=level, verb=0.16, jitter=0.002)
    if fill == "k-roll":
        tb = t0 + 3 * BEAT
        for s, a in zip(range(4), (0.45, 0.5, 0.6, 0.75)):
            tr.put(k_snare(a), tb + s * S16, gain=level, pan=0.1 * (-1) ** s, verb=0.15, jitter=0.002)
    elif fill == "pf-k":
        tb = t0 + 3 * BEAT
        tr.put(snare_pf(0.8), tb, gain=level, verb=0.16)
        tr.put(k_snare(0.5), tb + 2 * S16, gain=level, verb=0.15)
        tr.put(k_snare(0.62), tb + 3 * S16, gain=level, verb=0.15)


# ---------------------------------------------------------------- 01 intro
def make_intro():
    T0 = 0.55  # breath first, then the first heartbeat on the downbeat
    bars = 8
    tr = Track(T0 + bars * BAR + 0.12)
    tr.put(breath(0.5, inhale=True, amp=0.10), 0.02, verb=0.2)
    # bars 1-4: solo heartbeat at 62 BPM (lub on beats 1 & 3 of each 124-BPM bar), slow crescendo
    for k in range(8):
        t = T0 + k * 2 * BEAT
        amp = 0.32 + 0.3 * (k / 7)  # quieter than the groove so the doubling lifts
        heartbeat(tr, t, amp=amp, muffle=550 + 40 * k)
    # bars 5-6: heartbeat continues; beatbox hats creep in on the off-beats
    for k in range(4):
        t = T0 + 4 * BAR + k * 2 * BEAT
        heartbeat(tr, t, amp=0.66, muffle=900)
    for i in range(16):
        tb = T0 + 4 * BAR + i * BEAT
        a = 0.12 + 0.3 * (i / 15)
        tr.put(hat(a), tb + E8, pan=0.15, verb=0.12, jitter=0.003)
        if i >= 8:
            tr.put(hat(a * 0.35), tb + 3 * S16, pan=-0.2, verb=0.1, jitter=0.004)
    tr.put(k_snare(0.45), T0 + 6 * BAR - S16 * 2, verb=0.15)
    tr.put(k_snare(0.6), T0 + 6 * BAR - S16, verb=0.15)
    # bars 7-8: the heart doubles -> 124 BPM four-on-the-floor + full beatbox, fill into the song
    groove_bar(tr, T0 + 6 * BAR, level=0.95)
    groove_bar(tr, T0 + 7 * BAR, level=1.0, fill="k-roll")
    return render(tr, tail_fade=0.1)


# ---------------------------------------------------------------- 02 loop
def make_loop():
    bars = 4
    L = bars * BAR
    tr = Track(L + 1.0)  # render an extra second, then wrap the overhang onto the start
    for b in range(bars):
        groove_bar(tr, b * BAR, level=1.0, fill=("pf-k" if b == bars - 1 else None))
    wet = np.stack([signal.fftconvolve(tr.send[:, c], IR[c])[: tr.n] for c in range(2)], axis=1)
    mix = tr.dry + wet
    n = int(round(L * SR))
    out = mix[:n].copy()
    out[: tr.n - n] += mix[n:]
    out = sat(out * 1.1, 1.3)
    return out / np.max(np.abs(out)) * 10 ** (-1.0 / 20)


# ---------------------------------------------------------------- 03 outro
def make_outro():
    tr = Track(6 * BAR + 9.0)
    # bars 1-2: full groove
    groove_bar(tr, 0 * BAR, level=1.0)
    groove_bar(tr, 1 * BAR, level=0.95, fill="pf-k")
    # bars 3-4: snare drops out, hats thin out
    groove_bar(tr, 2 * BAR, level=0.85, snare=False, ghosts=False)
    groove_bar(tr, 3 * BAR, level=0.72, snare=False, ghosts=False)
    # bars 5-6: the heart halves back to 62 BPM, hats fade away
    t = 4 * BAR
    for k in range(4):
        heartbeat(tr, t + k * 2 * BEAT, amp=0.7 - 0.05 * k, muffle=900 - 50 * k)
    for i in range(8):
        tr.put(hat(0.3 * (1 - i / 8)), t + i * BEAT + E8, pan=0.15, verb=0.14)
    # then the heart slows down (ritardando) and gets quieter / more muffled
    t = 6 * BAR
    intervals = [1.0, 1.06, 1.14, 1.24, 1.36]
    for k, iv in enumerate(intervals):
        heartbeat(tr, t, amp=0.52 - 0.075 * k, gap=E8 + 0.02 * k, muffle=650 - 60 * k, verb=0.08)
        t += iv
    # last heartbeat ... then an exhale
    tr.put(breath(1.3, inhale=False, amp=0.09), t + 0.4, verb=0.25)
    return render(tr, tail_fade=1.2)


def save(name, x, title):
    wav = os.path.join(OUT, name + ".wav")
    mp3 = os.path.join(OUT, name + ".mp3")
    wavfile.write(wav, SR, (np.clip(x, -1, 1) * 32767).astype(np.int16))
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-codec:a", "libmp3lame", "-b:a", "320k",
         "-metadata", f"title={title}", "-metadata", "comment=124 BPM heartbeat/beatbox sample", mp3],
        check=True,
    )
    print(f"{name}: {len(x) / SR:.2f}s")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    save("01_heartbeat_intro_62-to-124bpm", make_intro(), "Heartbeat Intro (62 -> 124 BPM)")
    save("02_heartbeat_beatbox_loop_124bpm", make_loop(), "Heartbeat Beatbox Loop (124 BPM, 4 bars)")
    save("03_heartbeat_outro_124-to-62bpm", make_outro(), "Heartbeat Outro (124 -> 62 BPM)")
