"""The beat for "La Loca Tlacuaque de la Calle Cerrada", synthesized from score.py.

    python3 beat.py <outdir>

Writes one float32 stereo stem per instrument group (dry, full song length) plus sidechain.json
(kick times for the mix's pump). The arrangement is driven by the section TYPE in the score
(intro / verse / hook / bridge / outro), so it follows whatever structure the score has:

  verse   slow heavy boom-bap: kick, crack snare on 2 and 4, swung hats, 808, dark piano arpeggio,
          accordion licks at phrase ends
  hook    norteño bounce for the jingle: tuba on the beats, accordion on the off-beats, claps, güira,
          sleigh bells, and glockenspiel + toy piano doubling the sung melody from score.SUNG
  bridge  half-time stomp with patrol sirens
  intro / outro  vinyl crackle and the toy-piano jingle on its own
"""
import json
import os
import sys

import numpy as np
import soundfile as sf

import shared  # noqa: F401
import score as S
import sounds as X
import synth_dsp as D

SR = S.SR
N = int(round(S.SONG_SECONDS * SR))
rng = np.random.default_rng(90)
STEMS = ["kick", "snare", "hats", "bass", "piano", "accordion", "jingle", "fx"]


def jit(sd=0.003):
    return float(rng.normal(0, sd))


def hook_notes():
    """Sung hook notes on the timeline: [(time, dur_s, midi, bar)], from score.SUNG."""
    out = []
    for p in S.SUNG:
        for (b, d, m, _) in p["notes"]:
            out.append((S.t(p["bar"], b), d * S.BEAT, m, p["bar"] + int(b // 4)))
    return out


def first_hook_line():
    return S.SUNG[0] if S.SUNG else None


def render():
    B = {k: D.Bus(k, N) for k in STEMS}
    kick_times = []
    sung_bars = {}
    for (tm, d, m, bar) in hook_notes():
        sung_bars.setdefault(bar, []).append((tm, d, m))
    verse_count = 0
    last_type = None

    def k_hit(bar, slot, amp=1.0):
        tm = S.t16(bar, slot) + jit(0.0015)
        B["kick"].add(X.kick(amp), tm)
        kick_times.append(tm)

    def s_hit(bar, slot, amp=1.0, ghost=False, clap=0.0):
        tm = S.t16(bar, slot) + jit(0.002)
        B["snare"].add(X.snare(amp, ghost), tm, pan=0.04)
        if clap:
            B["snare"].add(X.clap(clap), tm - 0.004, pan=-0.12)
            B["snare"].add(X.clap(clap * 0.8), tm + 0.007, pan=0.16)

    def h_hit(bar, slot, amp=0.4, open_=False, pan=0.22):
        B["hats"].add(X.hat(amp, open_), S.t16(bar, slot) + jit(0.002), pan=pan)

    def sub(bar, slot, slots, midi, amp=1.0, glide=None):
        t0 = S.t16(bar, slot)
        dur = S.t16(bar, slot + slots) - t0 - 0.02
        B["bass"].add(X.sub808(midi, dur, amp, glide), t0)

    for bar in range(S.TOTAL_BARS):
        name, typ, i, n = S.section_of(bar)
        ch = S.chord_at(bar)
        root, voic = ch["root"], ch["voicing"]
        last_bar = i == n - 1
        if typ == "verse" and last_type != "verse":
            verse_count += 1
        last_type = typ
        phrase_end = (i % 4 == 3)

        # ------------------------------------------------------------ VERSE: boom-bap
        if typ == "verse":
            dropout = last_bar                          # the bar before a hook: beat stops after the snare
            kicks = [0, 6, 10] if i % 2 == 0 else [0, 7, 10, 11]
            if i % 8 == 7:
                kicks = [0, 6, 10]
            for s in kicks:
                if dropout and s > 12:
                    continue
                k_hit(bar, s, 1.0 if s in (0, 10) else 0.8)
            s_hit(bar, 4, 1.0)
            s_hit(bar, 12, 1.05, clap=0.25 if verse_count >= 2 else 0.0)
            if i % 2 == 1 and not dropout:
                s_hit(bar, 15, 0.5, ghost=True)
            if i % 4 == 2:
                s_hit(bar, 9, 0.45, ghost=True)
            # hats: eighths, with swung ghost sixteenths
            for s in range(16):
                if dropout and s > 12:
                    break
                if s % 2 == 0:
                    h_hit(bar, s, 0.36 if s % 4 == 0 else 0.27)
                elif s in (3, 7, 11, 15) and (i % 2 == 1 or s in (7, 15)):
                    h_hit(bar, s, 0.15)
            if phrase_end and not dropout:
                h_hit(bar, 14, 0.3, open_=True)
                for k in range(3):                     # trap roll into the next phrase
                    tm = S.t16(bar, 15) + k * (S.t16(bar, 16) - S.t16(bar, 15)) / 3
                    B["hats"].add(X.hat(0.16 + 0.05 * k), tm, pan=0.22)
            # 808: long notes under the kicks
            sub(bar, 0, 5.5, root, 1.0)
            if not dropout:
                sub(bar, 10, 5.5, root, 0.9)
                if i % 2 == 1:
                    sub(bar, 7, 2.5, root + (7 if i % 4 == 1 else 0), 0.6)
            # dark piano arpeggio in eighths, left hand on the downbeat
            top = {0: 69, 1: 70, 2: 70, 3: 69}[i % 4] if ch is not S.CHORDS.get("A") else 69
            lo, mid = voic[0], voic[1]
            fig = [lo, mid, top, mid, lo, mid, top if i % 4 != 3 else 67, mid]
            for e, m in enumerate(fig):
                if dropout and e >= 7:
                    break
                vel = 0.5 if e % 2 == 0 else 0.38
                B["piano"].add(X.piano(m, 0.30, vel + (0.08 if e == 0 else 0)), S.t(bar, e * 0.5) + jit(0.004),
                               pan=-0.25 + 0.06 * (e % 4))
            B["piano"].add(X.piano(root + 12, 1.6, 0.55), S.t(bar, 0) + jit(0.003), pan=-0.3)
            B["piano"].add(X.piano(root, 1.6, 0.5), S.t(bar, 0) + jit(0.003), pan=-0.3)
            # accordion: off-beat stabs from the second verse, a lick in thirds at the end of each 8 bars
            if verse_count >= 2 and not dropout:
                for s in ((6, 14) if i % 2 == 0 else (14,)):
                    B["accordion"].add(X.accordion([m + 12 for m in voic[:3]], 0.11, 0.30), S.t16(bar, s), pan=0.3)
            if i % 8 == 7 and not dropout:
                lick = [(8, 2, (69, 73)), (10, 1, (67, 70)), (11, 1, (65, 69)), (12, 2, (64, 67)), (14, 2, (61, 64))]
                for s, ln, ms in lick:
                    dur = S.t16(bar, s + ln) - S.t16(bar, s) - 0.02
                    B["accordion"].add(X.accordion(list(ms), dur, 0.42), S.t16(bar, s), pan=0.3)
            if verse_count >= 2 and i % 2 == 1 and not dropout:   # bajo-sexto answer
                for s, m in ((8, root + 12), (10, root + 19), (11, root + 24)):
                    B["accordion"].add(X.pluck(m, 0.16, 0.5), S.t16(bar, s), pan=-0.35)
            if dropout:
                B["fx"].add(X.reverse_swell(S.t16(bar, 16) - S.t16(bar, 13), 0.22), S.t16(bar, 13))

        # ------------------------------------------------------------ HOOK: norteño bounce + jingle
        elif typ == "hook":
            for s in (0, 4, 8, 12):
                k_hit(bar, s, 0.92 if s in (0, 8) else 0.72)
            if i % 2 == 1:
                k_hit(bar, 14, 0.6)
            s_hit(bar, 4, 0.95, clap=0.6)
            s_hit(bar, 12, 1.0, clap=0.6)
            if phrase_end:
                for k, s in enumerate((13, 14, 15)):
                    s_hit(bar, s, 0.45 + 0.15 * k, ghost=True)
            # güira (ch - k-ch) and sleigh bells on the eighths
            for s in range(16):
                if s % 4 == 0:
                    B["hats"].add(X.guira(0.30, long_=True), S.t16(bar, s) + jit(0.002), pan=-0.3)
                elif s % 4 in (2, 3):
                    B["hats"].add(X.guira(0.22 if s % 4 == 2 else 0.17), S.t16(bar, s) + jit(0.002), pan=-0.3)
                if s % 2 == 0:
                    B["hats"].add(X.sleigh(0.55 if s % 4 == 0 else 0.38), S.t16(bar, s) + jit(0.003), pan=0.35)
                    h_hit(bar, s, 0.16, pan=0.15)
            if phrase_end:
                h_hit(bar, 14, 0.26, open_=True)
            # tuba on the beats (root / fifth), 808 underneath on 1 and 3, accordion on the off-beats
            for bt, m in enumerate((root, root + 7, root, root + 7 if not phrase_end else root + 2)):
                B["bass"].add(X.tuba(m + (0 if m < 44 else -12), S.BEAT * 0.42, 0.75), S.t(bar, bt) + jit(0.002))
            sub(bar, 0, 3.5, root, 0.8)
            sub(bar, 8, 3.5, root, 0.7)
            for s in (2, 6, 10, 14):
                B["accordion"].add(X.accordion([m + 12 for m in voic], 0.12, 0.42), S.t16(bar, s) + jit(0.002), pan=0.28)
                B["accordion"].add(X.pluck(voic[0], 0.10, 0.35), S.t16(bar, s), pan=-0.35)
            # the jingle: glockenspiel + toy piano double the sung melody; arpeggio fill where nobody sings
            if bar in sung_bars:
                for (tm, d, m) in sung_bars[bar]:
                    B["jingle"].add(X.glock(m + 24, 0.42), tm, pan=0.2)
                    B["jingle"].add(X.toypiano(m + 12, 0.40), tm + 0.004, pan=-0.2)
            else:
                arp = [voic[0], voic[1], voic[2], voic[1] + 12, voic[2], voic[1], voic[0], voic[1]]
                for e, m in enumerate(arp):
                    B["jingle"].add(X.glock(m + 24, 0.26), S.t(bar, e * 0.5), pan=0.2)
                    if e % 2 == 0:
                        B["jingle"].add(X.toypiano(m + 12, 0.3), S.t(bar, e * 0.5), pan=-0.2)
            if i == 0:
                B["fx"].add(X.sub_drop(0.55), S.t(bar, 0))
                B["hats"].add(X.hat(0.5, open_=True, tau=0.5), S.t(bar, 0), pan=-0.1)

        # ------------------------------------------------------------ BRIDGE: half-time stomp, sirens
        elif typ == "bridge":
            k_hit(bar, 0, 1.0)
            k_hit(bar, 2, 0.7)
            if i % 2 == 1:
                k_hit(bar, 10, 0.8)
            s_hit(bar, 8, 1.1, clap=0.7)
            for s in (4, 12):                          # stomp-claps on 2 and 4, quieter
                B["snare"].add(X.clap(0.35), S.t16(bar, s), pan=0.0)
            for s in range(0, 16, 2):
                h_hit(bar, s, 0.22 if s % 4 == 0 else 0.14)
            if last_bar:                               # snare roll out of the bridge
                for k in range(8):
                    s_hit(bar, 8 + k, 0.35 + 0.08 * k, ghost=k < 6)
            sub(bar, 0, 7.5, root, 1.0)
            sub(bar, 8, 7.5, root, 0.8, glide=root + 12 if i % 2 == 1 else None)
            B["bass"].add(X.tuba(root, S.BEAT * 1.6, 0.6), S.t(bar, 0))
            B["bass"].add(X.tuba(root + 7, S.BEAT * 0.9, 0.5), S.t(bar, 2))
            B["accordion"].add(X.accordion(voic, S.BAR * 0.92, 0.26, bright=2200), S.t(bar, 0), pan=0.25)
            B["piano"].add(X.piano(root + 12, 2.2, 0.6), S.t(bar, 0), pan=-0.3)
            B["piano"].add(X.piano(voic[0], 1.0, 0.42), S.t(bar, 2), pan=-0.2)
            B["piano"].add(X.piano(voic[2], 1.0, 0.42), S.t(bar, 3), pan=-0.1)
            if i == 0:                                 # one whoop to open the bridge; later ones are score FX
                B["fx"].add(X.siren(S.BAR * 0.95, 0.30), S.t(bar, 0), pan=-0.5)

        # ------------------------------------------------------------ INTRO / OUTRO: the jingle alone
        else:
            line = first_hook_line()
            if line is not None and i % 2 == 0:
                for (b, d, m, _) in line["notes"]:
                    tm = S.t(bar, b * 2.0) if typ == "intro" else S.t(bar, b)   # intro: half speed, eerie
                    if tm < S.t(bar + (2 if typ == "intro" else 1), 0):
                        B["jingle"].add(X.toypiano(m + 12, 0.42), tm, pan=-0.15)
                        B["jingle"].add(X.glock(m + 24, 0.16), tm + (S.BEAT * 0.75), pan=0.3)
            B["piano"].add(X.piano(root + 12, 2.4, 0.42), S.t(bar, 0), pan=-0.3)
            B["piano"].add(X.piano(root, 2.4, 0.4), S.t(bar, 0), pan=-0.3)
            if typ == "intro" and last_bar:            # pick-up fill into the first verse
                for k, s in enumerate((10, 11, 12, 13, 14, 15)):
                    s_hit(bar, s, 0.4 + 0.1 * k, ghost=k < 4)
                k_hit(bar, 8, 0.9)
                B["fx"].add(X.scratch(0.35, 3), S.t16(bar, 12))
            if typ == "outro" and i == 0:
                B["fx"].add(X.sub_drop(0.6), S.t(bar, 0))

    # ---------------- vinyl crackle: intro, outro and tail loud; a whisper of it under the verses
    cr = X.crackle(S.SONG_SECONDS + 0.5, 1.0)[:N]
    g = np.zeros(N)
    for name, start, n, _, _ in S.SECTIONS:
        typ = S.section_of(start)[1]
        a, z = int(S.t(start) * SR), min(N, int(S.t(start + n) * SR))
        g[a:z] = {"intro": 0.05, "outro": 0.05, "verse": 0.016, "bridge": 0.012, "hook": 0.0}.get(typ, 0.01)
    g[:int(S.t(0) * SR)] = 0.05
    g[int(S.t(S.TOTAL_BARS) * SR):] = 0.05
    k = int(0.4 * SR)
    g = np.convolve(g, np.ones(k) / k, mode="same")
    B["fx"].add(cr * g, 0.0, pan=0.0)

    # ---------------- story sound effects from the score
    for f in S.FX:
        tm = S.t(f["bar"], f["beat"])
        amp, pan, kind = f.get("amp", 1.0), f.get("pan", 0.0), f["kind"]
        if kind == "glass":
            B["fx"].add(X.glass_crash(0.75 * amp, seed=f.get("seed", 0)), tm, pan=pan)
        elif kind == "splash":
            B["fx"].add(X.splash(0.7 * amp), tm, pan=pan)
        elif kind == "door":
            B["fx"].add(X.door_slam(0.8 * amp), tm, pan=pan)
        elif kind == "hiss":
            B["fx"].add(X.hiss(f.get("seconds", 0.7), 0.3 * amp), tm, pan=pan)
        elif kind == "siren":
            B["fx"].add(X.siren(f.get("seconds", 1.3), 0.3 * amp), tm, pan=pan)
        elif kind == "scratch":
            B["fx"].add(X.scratch(0.4 * amp, f.get("moves", 3)), tm, pan=pan)
        elif kind == "drop":
            B["fx"].add(X.sub_drop(0.7 * amp), tm, pan=pan)
        else:
            raise ValueError(f"unknown FX kind {kind!r}")

    # ---------------- ending: let the last sounds ring, then fade the crackle out
    fade = int(1.2 * SR)
    for bus in B.values():
        bus.x[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)[:, None] ** 2
    return B, sorted(kick_times)


if __name__ == "__main__":
    import time

    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(shared.WORK, "tlac", "inst")
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    B, kick_times = render()
    for k, bus in B.items():
        sf.write(os.path.join(out, f"inst_{k}.wav"), bus.x, SR, subtype="FLOAT")
        x = bus.x
        print(f"inst_{k:10s} peak {np.abs(x).max():5.2f}  rms {20 * np.log10(np.sqrt((x ** 2).mean()) + 1e-12):6.1f} dB")
    json.dump(dict(kick_times=kick_times), open(os.path.join(out, "sidechain.json"), "w"))
    print(f"{len(B)} stems, {N / SR:.1f} s, rendered in {time.time() - t0:.1f} s -> {out}")
