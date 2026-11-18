"""THE NEXT REPETITION: the arrangement. Renders every scene to 14 stems.

    python3 arrange.py <out_dir>

Themes: A = the Builder (the author), B = the Place (the six chapter titles as a wordless greeting),
C = Griff's five quiet notes, D = the clock / the practice (ostinato), E = the mirror, F = the possum.
"""
import json
import os
import sys

import numpy as np
import soundfile as sf

import shared
import score as S
import engine as E
import instruments as I
import sfx_world as SW
import sfx_home as SH
import synth_dsp as D
from player import Player

X = shared.tlac_sounds()
SR = S.SR
E._QUAL["7sus4"] = (0, 5, 7, 10)

STEMS = ["fx_time", "fx_place", "fx_house", "kick", "snare", "perc", "hats", "bass", "keys", "guitar",
         "mallets", "leads", "accordion", "pads"]
P = Player(S.TL, STEMS)
KICKS = []

A_MIN = ["D4:.5 A4:1.5 G4:.5 F4:.5 E4:1", "F4:.5 C5:1.5 Bb4:.5 A4:.5 G4:1", "A4:.5 D5:1.5 C5:.5 Bb4:.5 A4:.5 G4:.5", "F4:1 E4:1 D4:2"]
A_MAJ = ["D4:.5 A4:1.5 G4:.5 F#4:.5 E4:1", "F#4:.5 C5:1.5 B4:.5 A4:.5 G4:1", "A4:.5 D5:1.5 C5:.5 B4:.5 A4:.5 G4:.5", "D4:1.5 E4:.5 F#4:1 E4:1"]
B = ["G4:.5 B4:.5 D5:.5 B4:.5 E5:1 D5:1", "D5:.5 E5:.5 G5:2 r:1", "B4:1 A4:.5 G4:.5 A4:.5 D5:1.5",
     "D5:.5 D5:.5 E5:.5 G5:.5 F#5:1 D5:1", "E5:1 D5:.5 B4:1.5 r:.5 D4:.5", "G4:1 A4:.5 B4:.5 A4:.5 B4:1.5"]
GRIFF = ["D4:3 E4:1", "F#4:2 E4:2", "D4:4"]
LAUGH = "D6:.25 B5:.25 A5:.25 G5:.25 E5:.25 D5:.5"
OST = [50, 57, 62, 57, 55, 59, 62, 59]            # the practice, rebuilt by hand: D3 A3 D4 A3 G3 B3 D4 B3


# ------------------------------------------------------------------ helpers
def kick(sec, bar, beat, gain=1.0):
    P.hit("kick", X.kick(0.9), sec, bar, beat, gain, human=0.0)
    KICKS.append(P.t(sec, bar, beat))


def snare(sec, bar, beat, gain=1.0, ghost=False):
    P.hit("snare", X.snare(0.8, ghost=ghost), sec, bar, beat, gain, human=0.0)


def mar(sec, bar, spec, vel=0.6, tr=0, beat=0.0, pan=0.25, gain=1.0):
    P.line("mallets", I.marimba, sec, bar, spec, beat=beat, vel=vel, transpose=tr, sustained=False, pan=pan, gain=gain)


def gtr(sec, bar, spec, vel=0.6, tr=0, beat=0.0, pan=-0.2, gain=1.0, human=0.006):
    P.line("guitar", I.nylon, sec, bar, spec, beat=beat, vel=vel, transpose=tr, pan=pan, gain=gain, human=human)


def tpt(sec, bar, spec, vel=0.6, tr=0, beat=0.0, pan=0.0, mute=False, vib=1.0, gain=1.0, legato=0.95):
    P.line("leads", I.trumpet, sec, bar, spec, beat=beat, vel=vel, transpose=tr, pan=pan, mute=mute, vib=vib, gain=gain, legato=legato)


def flu(sec, bar, spec, vel=0.6, tr=0, beat=0.0, pan=-0.1, gain=1.0):
    P.line("leads", I.flute, sec, bar, spec, beat=beat, vel=vel, transpose=tr, pan=pan, gain=gain)


def pno(sec, bar, spec, vel=0.6, tr=0, beat=0.0, pan=0.0, gain=1.0, dark=False):
    P.line("keys", X.piano, sec, bar, spec, beat=beat, vel=vel, transpose=tr, pan=pan, gain=gain, dark=dark)


def rho(sec, bar, spec, vel=0.3, tr=0, beat=0.0, pan=0.3, gain=1.0):
    P.line("keys", I.rhodes, sec, bar, spec, beat=beat, vel=vel, transpose=tr, pan=pan, gain=gain)


def acc(sec, bar, spec, vel=0.5, tr=0, beat=0.0, pan=-0.35, gain=1.0):
    for off, beats, m, v in E.notes(spec):
        t0 = P.t(sec, bar, beat + off)
        P.stems["accordion"].add(X.accordion([m + tr], P.dur(sec, bar, beat + off, beats) * 0.95, v or vel), t0, gain, pan)


def pad(stem, inst, sec, bar, beat, beats, name, center=60, vel=0.4, pan=0.0, gain=1.0, **kw):
    P.chord(stem, inst, sec, bar, beat, beats, name=name, center=center, vel=vel, pan=pan, gain=gain, **kw)


def bass(sec, bar, beat, beats, name, vel=0.7, octave=2, fifth=False, inst=None):
    m = E.bass_note(name, octave) + (7 if fifth else 0)
    while m < 28:
        m += 12
    P.note("bass", inst or I.bass_ac, sec, bar, beat, beats, m, vel, human=0.004)


def tuba(sec, bar, beat, beats, pitch, vel=0.6):
    P.note("bass", X.tuba, sec, bar, beat, beats, pitch, vel, human=0.0)


def ost_half(name, third_first=False):
    root, iv, _ = E.chord(name)
    r = 50 + ((root - 2) % 12)
    return [r + iv[1] if third_first else r, r + 7, r + 12, r + 7]


def machine(sec, bar, chords, vel=0.5):
    """Theme D, machine form: hard-quantized eighths on a dry pluck and the piano's left hand."""
    a, b = (chords[0], chords[0]) if len(chords) == 1 else (chords[0], chords[1])
    seq = ost_half(a) + ost_half(b, third_first=(a == b))
    for k, m in enumerate(seq):
        v = vel * (1.0 if k % 2 == 0 else 0.8)
        P.note("guitar", lambda mm, d, vv: X.pluck(mm, d, vv, damp=0.7), sec, bar, k * 0.5, 0.45, m, v, human=0.0, pan=-0.3)
        P.note("keys", X.piano, sec, bar, k * 0.5, 0.45, m - 12, v * 0.7, human=0.0, pan=0.2)


def fx(stem, sig, at=None, sec=None, bar=0, beat=0.0, gain=1.0, pan=0.0, label=None, lead=0.0):
    return P.fx(stem, sig, at=at, sec=sec, bar=bar, beat=beat, gain=gain, pan=pan, label=label, lead=lead)


def laugh(sec, bar, beat, gain=1.0):
    """'The sound of someone laughing from a doorway', played, not voiced: marimba and staccato flute."""
    mar(sec, bar, LAUGH, vel=0.5, beat=beat, pan=0.5, gain=gain)
    flu(sec, bar, LAUGH, vel=0.35, beat=beat, pan=0.45, gain=gain * 0.7)
    P.log.append((round(P.t(sec, bar, beat), 3), "mallets", "doorway laugh (played on marimba and flute)"))


# ------------------------------------------------------------------ 1. Internal Clock (D minor, 120)
def s1():
    s = "s1"
    for b in range(15):
        for k, bt in enumerate((0, 2)):                       # a literal clock: one tick per second
            fx("fx_time", SW.tick(k == 0, 0.9, b), sec=s, bar=b, beat=bt, gain=1.0 if (b < 2 or b == 14) else 0.6)
    P.log.append((0.0, "fx_time", "clock tick (one per second)"))
    fx("fx_time", SW.typing(23.0, rate=10, amp=0.3, seed=1), sec=s, bar=1, label="keyboard typing")
    fx("fx_time", SH.chime(0.5), at=3.5, label="calendar chime")
    prog = {2: ["Dm"], 3: ["F"], 4: ["Dm/A", "Gm"], 5: ["Bb", "A7"], 6: ["Bb"], 7: ["C"], 8: ["Dm/A"], 9: ["A7"]}
    for b in range(10, 14):
        prog[b] = prog[b - 8]
    for b in range(2, 14):
        ch = prog[b]
        loud = b >= 10
        machine(s, b, ch, vel=0.6 if loud else 0.5)
        kick(s, b, 0)
        kick(s, b, 2, 0.85)
        for bt in (1, 3):                                     # the rubber stamp is the snare
            fx("fx_time", SW.stamp(0.8, b * 2 + bt), sec=s, bar=b, beat=bt, gain=0.9, label="rubber stamp" if (b == 2 and bt == 1) else None)
        P.note("bass", X.sub808, s, b, 0, 1.9, E.bass_note(ch[0], 1), 0.8, human=0.0)
        if len(ch) > 1:
            P.note("bass", X.sub808, s, b, 2, 1.9, E.bass_note(ch[1], 1), 0.8, human=0.0)
        P.pattern("hats", lambda: X.hat(0.35), s, 1, [0.5, 1.5, 2.5, 3.5], first_bar=b, human=0.0)
    for k, sp in enumerate(A_MIN):                            # Theme A, bare octaves: competent, hollow
        pno(s, 2 + k, sp, vel=0.75, human=0.0) if False else P.line("keys", X.piano, s, 2 + k, sp, vel=0.75, human=0.0, dark=False)
        P.line("keys", X.piano, s, 2 + k, sp, vel=0.6, transpose=-12, human=0.0, dark=False)
        pad("pads", I.strings, s, 2 + k, 0, 4, None, pitches=[E.midi(x) for x in (("D3", "A3"), ("F3", "C4"), ("D3", "A3"), ("Bb2", "F3"))[k]], vel=0.3)
    # bars 7-10: the bridge being built: strings climb on the left, brass descends on the right, they meet on A
    for k, (lo, hi) in enumerate((("D4", "F5"), ("E4", "E5"), ("F4", "D5"))):
        pad("pads", I.strings, s, 6 + k, 0, 4, None, pitches=[E.midi(lo)], vel=0.5, pan=-0.7)
        tpt(s, 6 + k, f"{hi}:4", vel=0.5, pan=0.7, vib=0.2)
    pad("pads", I.strings, s, 9, 0, 2, None, pitches=[E.midi("G4")], vel=0.5, pan=-0.7)
    pad("pads", I.strings, s, 9, 2, 2, None, pitches=[E.midi("A4")], vel=0.55, pan=-0.2)
    tpt(s, 9, "C#5:2 A4:2", vel=0.55, pan=0.5, vib=0.2)
    for b, n in ((6, 2), (8, 3)):                             # hammer on steel: "bridges between people"
        fx("fx_time", SH.hammer(n, 0.5, 0.7, b), sec=s, bar=b, beat=2, gain=0.8, pan=0.3, label="hammer")
    for k, sp in enumerate(A_MIN):                            # bars 11-14: Theme A, tutti
        tpt(s, 10 + k, sp, vel=0.75, vib=0.4)
        P.line("keys", X.piano, s, 10 + k, sp, vel=0.8, human=0.0, dark=False)
        P.line("keys", X.piano, s, 10 + k, sp, vel=0.65, transpose=-12, human=0.0, dark=False)
        ch = prog[10 + k]
        pad("keys", I.organ, s, 10 + k, 0, 4 if len(ch) == 1 else 2, ch[0], center=57, vel=0.35, gain=0.8)
        pad("pads", I.strings, s, 10 + k, 0, 4 if len(ch) == 1 else 2, ch[0], center=64, vel=0.5)
        if len(ch) > 1:
            pad("keys", I.organ, s, 10 + k, 2, 2, ch[1], center=57, vel=0.35, gain=0.8)
            pad("pads", I.strings, s, 10 + k, 2, 2, ch[1], center=64, vel=0.5)
    pno(s, 13, "r:2 D3+A3+D4:2", vel=0.8)
    fx("fx_place", X.reverse_swell(3.2, 0.5), at=30.0, lead=3.2, gain=0.7)


# ------------------------------------------------------------------ 2. "I said it was awful" (A Lydian vs D minor, 90)
def s2():
    s = "s2"
    fx("fx_place", SW.suitcase(5.0, 0.6, seed=2), at=30.3, pan=-0.3, label="suitcase wheels on cobblestones")
    fx("fx_place", SH.heat(23.0, 0.2, seed=1), at=30.5, gain=0.6, label="heat shimmer")
    fx("fx_place", SW.surf(9.0, 0.5, period=7.0, seed=3), at=33.0, gain=0.7, label="surf")
    for b in range(9):
        last = b == 8
        if not last:
            for bt, kind, hi in ((0, "open", False), (1, "slap", True), (1.5, "open", True), (2.5, "open", False), (3, "slap", True), (3.5, "open", True)):
                P.hit("perc", I.conga(kind, hi, 0.6), s, b, bt, 0.8, pan=-0.4 if hi else -0.2)
            for bt in (0, 0.5, 1, 1.5, 2, 3):                 # the claves already say "Ho-la bue-nos di-as"
                P.hit("perc", I.clave(0.5), s, b, bt, 0.7, pan=0.5)
            P.hit("perc", I.guiro(0.35, 0.3), s, b, 1, 0.8, pan=0.3)
            P.hit("perc", I.guiro(0.35, 0.16, up=False), s, b, 3, 0.7, pan=0.3)
            P.pattern("perc", lambda: I.cowbell(0.3), s, 1, [0, 2], first_bar=b, pan=-0.6, gain=0.6)
            P.pattern("hats", lambda: I.maraca(0.3), s, 1, [k * 0.5 for k in range(8)], first_bar=b, accents=[1, 0.6])
        for bt, bts in ((0, 1.4), (1.5, 0.9), (2.5, 0.5), (3, 0.9)):
            if not last or bt == 0:
                P.note("bass", I.bass_ac, s, b, bt, bts, "A1" if bt != 2.5 else "E2", 0.7)
    # the brochure: the place's greeting, a tone too bright, two trumpets in thirds and a glockenspiel
    hi1, lo1 = "A4:.5 C#5:.5 E5:.5 C#5:.5 F#5:1 E5:1", "F#4:.5 A4:.5 C#5:.5 A4:.5 D#5:1 C#5:1"
    hi2, lo2 = "E5:.5 F#5:.5 A5:2.6", "C#5:.5 D#5:.5 F#5:2.6"
    for b0, v, withmar in ((1, 0.75, False), (5, 0.6, True)):
        for k, (h, l) in enumerate(((hi1, lo1), (hi2, lo2))):
            tpt(s, b0 + k, h, vel=v, pan=0.55, vib=1.8)
            tpt(s, b0 + k, l, vel=v * 0.85, pan=-0.55, vib=1.8)
            P.line("mallets", X.glock, s, b0 + k, h, vel=0.4, transpose=12, sustained=False, pan=0.6)
            if withmar:
                mar(s, b0 + k, h, vel=0.55, pan=0.3)
            for name, bt in ((("A", 0), ("B/A", 2)) if k == 0 else (("B/A", 0), ("A", 2))):
                pad("pads", I.strings, s, b0 + k, bt, 2, name, center=69, vel=0.35)
                pad("guitar", I.nylon, s, b0 + k, bt + 1, 0.8, name, center=62, vel=0.4, poly=False, strum=0.012, pan=0.2)
    fx("fx_place", SH.applause(2.6, 0.3, seed=1), sec=s, bar=2, beat=2.3, gain=0.5, label="applause for the bay taking its bow")
    # his replies: the Builder's head, dry, low, clipped, a little ahead of the beat
    head = "D3:.25 A3:.75 G3:.25 F3:.25 E3:.5"
    for b, bt in ((3, 0), (3, 2), (7, 0)):
        tpt(s, b, head, vel=0.55, beat=bt - 0.05, mute=True, vib=0, tr=12, legato=0.6)
        for off, beats, m, _ in E.notes(head):
            tuba(s, b, bt - 0.05 + off, beats * 0.6, m - 12, 0.6)
    for b in (4, 7):                                          # the verdict
        bt = 0 if b == 4 else 2
        tpt(s, b, "Bb3:.5 A3:.5 D3:1", vel=0.6, beat=bt, mute=True, vib=0, tr=12, legato=0.7)
        for off, beats, m, _ in E.notes("Bb2:.5 A2:.5 D2:1"):
            tuba(s, b, bt + off, beats * 0.7, m, 0.7)
    fx("fx_time", SW.stamp(0.9, 99), sec=s, bar=4, beat=1.5, gain=1.0, label="rubber stamp: the verdict, \"awful\"")
    # Griff, quietly, under all of it
    for k, sp in enumerate(GRIFF):
        rho(s, 1 + 2 * k, sp, vel=0.3)
    rho(s, 7, "D4:8", vel=0.3)
    pad("guitar", I.nylon, s, 8, 0, 0.5, "A", center=62, vel=0.6, poly=False, strum=0.01)
    pad("guitar", I.nylon, s, 8, 1, 0.5, "A", center=62, vel=0.5, poly=False, strum=0.01)
    fx("fx_place", SW.zip_(0.5, 0.4, up=False), sec=s, bar=8, beat=1.6, pan=-0.3)
    fx("fx_place", SW.suitcase(2.4, 0.6, speed=1.6, seed=5), at=51.6, pan=0.3, label="suitcase wheels, leaving fast")


# ------------------------------------------------------------------ 3. It Decayed in Pieces (D minor thawing, 60)
def s3():
    s = "s3"
    t, k, gap = 54.0, 0, 1.0                                  # the old clock, far away, slowing, stopping
    while t < 71.5:
        fx("fx_time", D.lp(SW.tick(k % 2 == 0, 0.8, k), 1400), at=t, gain=max(0.15, 0.6 - 0.03 * k))
        gap *= 1.07
        t += gap
        k += 1
    P.log.append((54.0, "fx_time", "distant clock, slowing to a stop"))
    pad("pads", I.strings, s, 0, 0, 4, None, pitches=[50, 57], vel=0.2)
    thaw = ["D4:.5 A4:1.5 G4:.5 F4:.5 E4:1", "F#4:.5 C5:1.5 B4:.5 A4:.5 G4:1", "A4:.5 D5:1.5 C5:.5 B4:.5 A4:.5 G4:.5", "F#4:1 E4:1 D4:2"]
    for i, sp in enumerate(thaw):                             # Theme A alone at the piano; the F turns to F sharp
        pno(s, 1 + i, sp, vel=0.5, dark=True)
    for b, parts in ((1, (("Dm", 0, 3), ("G/D", 3, 1))), (2, (("D", 0, 2), ("G/D", 2, 2))), (3, (("Bm7", 0, 2), ("Am7", 2, 2))),
                     (4, (("D7", 0, 2), ("Bbmaj7", 2, 2))), (5, (("Bbmaj7", 0, 2), ("C9", 2, 2))), (6, (("D7sus4", 0, 2), ("D7", 2, 2)))):
        for name, bt, bts in parts:
            pno(s, b, f"{E.root_note(name, 2)}", vel=0.3) if False else P.note("keys", X.piano, s, b, bt, bts, E.bass_note(name, 2), 0.35, dark=True)
            if b >= 3:
                pad("pads", I.strings, s, b, bt, bts, name, center=62, vel=0.22 + 0.04 * (b - 3))
    mar(s, 1, "G5:.5 B5:.5 D6:1", vel=0.35, beat=3, pan=0.6, gain=0.6)     # a memory of a street corner
    laugh(s, 2, 3.0, gain=0.7)                                             # someone laughing from a doorway
    mar(s, 3, "G5:.5 B5:.5 D6:.5 B5:.5", vel=0.35, beat=2, pan=-0.5, gain=0.6)
    for k2, sp in enumerate(GRIFF):                           # Griff's cell; its F sharp is what turns the key
        rho(s, 1 + k2, sp, vel=0.32)
    tpt(s, 5, "r:.5 D4:.5 E4:.5 F4:1 D4:1.5", vel=0.4, mute=True, vib=0.3)   # "I wanted to give it another look"
    for k3, m in enumerate([50, 54, 57, 60, 62, 66, 69, 72, 74, 78, 81, 84]):
        P.note("guitar", I.harp, s, 6, 2 + k3 * 0.16, 2.0, m, 0.35 + 0.02 * k3, pan=-0.5 + 0.08 * k3)
    fx("fx_place", SW.footsteps(4, 0.62, 0.5, seed=3), at=62.6, gain=0.6, pan=-0.4, label="sandal footsteps (a street corner)")
    fx("fx_place", SW.suitcase(4.2, 0.5, speed=0.7, seed=7), at=77.4, pan=0.2, label="suitcase wheels, slow (the return)")


# ------------------------------------------------------------------ 4. The Absence of Pressure (G major, 6/8)
def s4():
    s = "s4"
    ch = ["Gmaj7", "Gmaj7", "Cmaj7", "Cmaj7", "Em9", "Am9", "Cmaj7", "Cadd9", "Gmaj7", "Gmaj7/B", "Cmaj7", "Cm6", "G/D", "Em7", "Cmaj7", "Cmaj7"]
    fx("fx_place", SW.surf(34.0, 0.5, period=8.0, seed=11, calm=0.3), at=80.5, gain=0.8, label="surf on the bay")
    th = 1 / 3
    for b, name in enumerate(ch):
        bloom = 8 <= b <= 11
        n_arp = 6 if b < 12 else (4 if b == 12 else (3 if b == 13 else (2 if b == 14 else 0)))
        vs = E.voicing(name, 62)
        for k in range(n_arp):                                # the harp stands in for the ostinato, and thins out
            idx = [0, 1, 2, 3, 2, 1][k]
            m = vs[idx % len(vs)] + 12 * (idx // len(vs))
            P.note("guitar", I.harp, s, b, k * th, 1.2, m, 0.42 if bloom else 0.36, pan=-0.35)
        if b < 12:
            P.note("guitar", I.nylon, s, b, 0, 1.0, vs[0], 0.4, pan=0.3)
            P.note("guitar", I.nylon, s, b, 1, 1.0, vs[2 % len(vs)], 0.35, pan=0.3)
        if b < 15:
            bass(s, b, 0, 1.9, name, vel=0.55)
        if b % 2 == 0 and b < 14:
            rho(s, b, "D4:4", vel=0.22)
        if 2 <= b <= 13:                                      # the broom is the only percussion
            fx("fx_place", SW.broom(1, 0.8, 0.45, seed=b), sec=s, bar=b, beat=1, gain=0.7, pan=0.5, label="broom on a sidewalk" if b == 2 else None)
        if b < 14:
            pad("pads", I.strings, s, b, 0, 2, name, center=64, vel=0.42 if bloom else 0.26)
        if bloom:
            pad("pads", I.choir, s, b, 0, 2, name, center=67, vel=0.4)
            for k in range(6):                                # marimba tremolo roll
                P.note("mallets", I.marimba, s, b, k * th, th, vs[-1] + 12, 0.3, sustained=False, pan=0.4)
    for b0, v, g in ((0, 0.45, 0.8), (8, 0.55, 0.9)):         # the place's greeting, granted in full
        tpt(s, b0, "G4:.333 B4:.333 D5:.334 B4:.667 E5:.333", vel=v, vib=0.5, gain=g)
        tpt(s, b0 + 1, "D5:2", vel=v, vib=0.6, gain=g)
        tpt(s, b0 + 2, "D5:.667 E5:.333 G5:1", vel=v, vib=0.5, gain=g)
        tpt(s, b0 + 3, "G5:2", vel=v, vib=0.8, gain=g)
    fl = ["D5:.333 A5:1 G5:.333 F#5:.334", "F#5:.333 C6:1 B5:.333 A5:.334", "A5:.333 D6:1 C6:.333 B5:.334", "G5:.667 F#5:.667 D5:.666"]
    for k, sp in enumerate(fl):                               # Theme A, floating, on flute and guitar
        flu(s, 4 + k, sp, vel=0.55)
        gtr(s, 4 + k, sp, vel=0.45, tr=-12, pan=0.2)
    flu(s, 12, "D5:.667 A5:3.3", vel=0.45)                    # only the rising fifth is left, hanging
    flu(s, 14, "D5:.667 A5:3.0", vel=0.35)
    fx("fx_house", SH.shower(5.0, 0.4, dip_at=2.0, seed=1), at=104.0, gain=0.8, pan=-0.4, label="shower: the water pressure dips")
    fx("fx_house", SH.flicker(0.7, seed=1), at=S.FLICKER_AT - 0.03, gain=1.0, label="the lights flicker once")


# ------------------------------------------------------------------ 5. Observation with Little Knives (E minor, 120)
def s5():
    s = "s5"
    prog = [["Em6"], ["Em6"], ["Am7"], ["B7b9"], ["Em6"], ["C9"], ["F#m7b5", "B7"], ["Em6"], ["Am7", "D7"], ["C", "G"]]
    fx("fx_place", SW.murmur(20.0, 0.3, people=7, seed=4), at=114.0, gain=0.35, label="cafe murmur (wordless)")
    for b in (0, 4, 8):                                       # espresso steam swells into the downbeat
        fx("fx_place", SW.espresso(1.8, 0.4, seed=b), sec=s, bar=b, beat=0, lead=1.8, gain=0.7, pan=0.5, label="espresso steam" if b == 0 else None)
    for b, ch in enumerate(prog):
        if b < 9:
            for bt in (1, 3):                                 # a kit made from the cafe: cup on saucer on 2 and 4
                fx("fx_place", SW.cup(0.5, seed=b * 2 + bt), sec=s, bar=b, beat=bt, gain=0.9, pan=-0.3, label="cup on saucer" if (b == 0 and bt == 1) else None)
            for bt in (0.5, 1.5, 2.5, 3.5):                   # teaspoon ticks
                P.hit("hats", I.triangle(0.2, open_=False), s, b, bt, 0.6, pan=0.4)
            P.hit("perc", I.guiro(0.3, 0.25), s, b, 0, 0.6, pan=0.3)
            if b >= 4:
                for bt, kind, hi in ((0, "open", False), (1, "mute", True), (1.5, "slap", True), (2.5, "open", True), (3, "open", False), (3.5, "mute", True)):
                    P.hit("perc", I.conga(kind, hi, 0.55), s, b, bt, 0.7, pan=-0.4 if hi else -0.2)
                for bt in (1.5, 3.5):
                    pad("keys", I.organ, s, b, bt, 0.4, ch[-1] if bt > 2 else ch[0], center=62, vel=0.3, gain=0.7)
        if b < 8:                                             # the walking bass: "the identity kept walking"
            c0, c1 = ch[0], ch[-1]
            r0, iv0, _ = E.chord(c0)
            base = E.bass_note(c0, 2)
            nxt = E.bass_note(prog[b + 1][0], 2)
            walk = [base, base + iv0[1], base + 7, nxt - 1] if c0 == c1 else [base, base + iv0[1], E.bass_note(c1, 2), nxt - 1]
            for k, m in enumerate(walk):
                P.note("bass", I.bass_ac, s, b, k, 0.95, m, 0.65)
        elif b == 8:                                          # ...and stops mid-step
            P.note("bass", I.bass_ac, s, b, 0, 0.95, E.bass_note("Am7", 2), 0.6)
            P.note("bass", I.bass_ac, s, b, 1, 0.95, E.bass_note("Am7", 2) + 3, 0.5)
    lines = {0: "E4:.5 B4:1.5 r:.5 A4:.25 B4:.25 G4:.5 E4:.5", 2: "G4:.5 D5:1.5 r:.5 C5:.25 D5:.25 B4:.5 A4:.5",
             4: "E4:.5 B4:1.5 A4:.25 G4:.25 F#4:.25 G4:.25 E4:1", 6: "B4:.5 E5:1.5 r:.5 D#5:.25 E5:.25 C5:.5 B4:.5",
             8: "A4:.5 E5:1.5 D5:.5 C5:.5 r:1"}
    for b, sp in lines.items():                               # the Builder's head, then chatter instead of the tune
        tpt(s, b, sp, vel=0.5, mute=True, vib=0.3, legato=0.7)
        mar(s, b, sp, vel=0.45, pan=0.25, gain=0.8)
    mar(s, 1, "r:2 E5:.25 G5:.25 B5:.5", vel=0.5)             # the 'catalogue' cell, a new specimen each time
    P.line("mallets", X.glock, s, 3, "r:2 A5:.25 C6:.25 E6:.5", vel=0.4, sustained=False, pan=0.5)
    P.line("mallets", X.toypiano, s, 5, "r:2 F#5:.25 A5:.25 C6:.5", vel=0.45, sustained=False, pan=-0.5)
    mar(s, 7, "r:2 E5:.25 G5:.25 B5:.5", vel=0.5)
    fx("fx_house", SH.ping(0.5, 0), at=117.0, pan=0.6, gain=0.8, label="message ping")
    fx("fx_house", SH.camera(0.5), sec=s, bar=3, beat=3.5, pan=-0.6, gain=0.7, label="phone camera")
    fx("fx_place", SW.footsteps(9, 0.5, 0.5, seed=8, stop=True), at=124.0, gain=0.7, pan=-0.5, label="sandal footsteps that keep walking, then stop")
    fx("fx_house", SH.pings(5, 1.6, 0.5, seed=2), sec=s, bar=8, beat=2, gain=0.7, label="a burst of pings")
    pad("keys", I.organ, s, 9, 0, 2, "C", center=60, vel=0.4)       # the organ says 'amen' for the group chat
    pad("keys", I.organ, s, 9, 2, 2, "G", center=59, vel=0.4)
    fx("fx_house", SH.gears(2.2, 0.3, seed=1), at=134.0, gain=0.8, pan=0.3, label="small gears turning (grand pause)")
    pad("pads", I.strings, s, 11, 0, 4, "B7sus4", center=59, vel=0.22)
    fx("fx_house", SH.mosquito(3.4, 0.3, seed=2), at=134.9, gain=0.9, label="mosquito")


# ------------------------------------------------------------------ 6. The Mirror Turns (B minor, 60)
def s6():
    s = "s6"
    fx("fx_house", SH.mirror(7.5, 0.4, sharpen=True, midi=90, seed=1), at=137.6, gain=0.9, label="the mirror: a glass tone that sharpens")
    fx("fx_house", SH.switch(0.6), at=140.0, gain=0.9, pan=0.2, label="flashlight click")
    for k in range(13):                                       # the leak: it shows the leak, it does not fix the pipe
        fx("fx_house", SH.drip(0.5, seed=k, pitch=1.0 + 0.03 * (k % 3)), at=141.0 + k, gain=0.6, pan=-0.5, label="leak dripping, one drop a second" if k == 0 else None)
    P.line("mallets", I.vibes, s, 1, "C#6:.5 F#5:1.5 G5:.5 A5:.5 B5:1", vel=0.4, sustained=False, pan=0.4)
    P.line("mallets", I.musicbox, s, 1, "C#6:.5 F#5:1.5 G5:.5 A5:.5 B5:1", vel=0.3, transpose=12, sustained=False, pan=0.5)
    gtr(s, 2, "B3:.5 F#4:1.5 E4:.5 D4:.5 C#4:1", vel=0.55, pan=0.0)       # the plain theme, alone and dry
    gtr(s, 3, "B3:.5 F#4:3.5", vel=0.55, pan=0.0)                           # theme and mirror meet on one note
    P.line("mallets", I.vibes, s, 3, "C#5:.5 F#4:3.5", vel=0.45, sustained=False, pan=0.0)
    pad("pads", I.strings, s, 3, 2, 2, "Bm", center=59, vel=0.2)
    gtr(s, 4, "D4:1 C#4:1.6", vel=0.5, pan=0.0)                             # the cadence starts; its last note never comes
    pad("pads", I.strings, s, 4, 0, 1, "Em7", center=59, vel=0.2)
    pad("pads", I.strings, s, 4, 1, 1.2, "F#7sus4", center=58, vel=0.2)
    pad("pads", I.strings, s, 6, 0, 4, "A7sus4", center=57, vel=0.25)       # after the silence, the door to D
    P.note("guitar", I.harp, s, 6, 2, 2, "A2", 0.4, pan=-0.2)
    fx("fx_place", SW.footsteps(2, 0.6, 0.5, seed=4), at=164.7, gain=0.7, pan=0.3, label="two footsteps")


# ------------------------------------------------------------------ 7. Económico, Not Barato (D major, 90)
def s7():
    s = "s7"
    prog = [["D"], ["D"], ["Gmaj9"], ["Em7", "A7sus4"], ["Bbmaj7"], ["A7sus4", "A7"], ["G", "A7sus4"], ["D"], ["Cmaj7"], ["G/B", "Em7"], ["G", "A7sus4"], ["A7sus4"]]
    fx("fx_place", SW.shutter(1.5, 0.5, seed=2), at=166.0, lead=1.5, gain=0.8, pan=0.4, label="a shop's metal shutter going up")
    for b, ch in enumerate(prog):
        full = b >= 7
        for k, name in enumerate(ch):
            bt, bts = (0, 4) if len(ch) == 1 else (2 * k, 2)
            if b < 11:
                bass(s, b, bt, 1.4, name, vel=0.65 if full else 0.55)
                if len(ch) == 1:
                    bass(s, b, 2.5, 1.2, name, vel=0.5, fifth=True)
            if b >= 4:
                pad("pads", I.strings, s, b, bt, bts, name, center=62, vel=0.42 if full else 0.3)
            for sb in ([1, 3] if len(ch) == 1 else [1]):      # soft strum on 2 and 4
                if b < 11:
                    pad("guitar", I.nylon, s, b, bt + sb if len(ch) == 1 else bt + 1, 0.8, name, center=57, vel=0.34 if full else 0.28, poly=False, strum=0.014, pan=0.35)
        if b < 11:
            P.hit("kick", I.bombo(0.8), s, b, 0, 0.8)
            P.hit("kick", I.bombo(0.6), s, b, 2.5, 0.6)
            P.pattern("hats", lambda: I.maraca(0.3), s, 1, [k * 0.5 for k in range(8)], first_bar=b, accents=[1, 0.6], gain=0.9 if full else 0.6)
        if b in (0, 1):                                       # the broom keeps time before the band does
            for bt in (1, 3):
                fx("fx_place", SW.broom(1, 0.8, 0.45, seed=20 + b * 2 + bt), sec=s, bar=b, beat=bt, gain=0.7, pan=-0.5, label="broom on a sidewalk" if (b == 0 and bt == 1) else None)
    tr = -5                                                   # the greeting in D: marimba calls, the guitar answers
    mar(s, 0, B[0], vel=0.6, tr=tr)
    mar(s, 1, B[1], vel=0.6, tr=tr)
    gtr(s, 2, "F4:.18 F#4:.82 E4:.5 D4:.5 E4:.5 A4:1.5", vel=0.6, pan=-0.15)   # almost the wrong word, corrected inside the note
    mar(s, 3, B[3], vel=0.6, tr=tr)
    fx("fx_house", SH.till(0.5, seed=1), at=173.3, gain=0.7, pan=0.5, label="corner-shop till")
    fx("fx_house", SH.mains_hum(7.0, 0.3, sag=True, seed=1), at=175.5, gain=0.9, label="mains hum, the voltage sagging (the CFE report)")
    for k, bt in enumerate((0.5, 1.5, 2.5, 3.5, 4.5, 5.5)):   # a corroded connection, crackling on the off-beats
        fx("fx_house", SH.arc(0.35, 0.5, seed=k), sec=s, bar=4, beat=bt, gain=0.6, pan=0.5 - 0.2 * k, label="electrical crackle" if k == 0 else None)
    gtr(s, 4, "D4:1 F4:1 Bb4:2", vel=0.5, pan=-0.15)
    gtr(s, 5, B[4], vel=0.6, tr=tr, pan=-0.15)
    fx("fx_time", SW.stamp(0.8, 7), at=182.0, gain=0.8, label="one stamp: the report is filed")
    gtr(s, 6, "D4:.5 E4:.5 F#4:.5 G4:.5 A4:1 C#5:1", vel=0.55, pan=-0.15)
    low = ["D3:.5 A3:1.5 G3:.5 F#3:.5 E3:1", "F#3:.5 C4:1.5 B3:.5 A3:.5 G3:1", "A3:.5 D4:1.5 C4:.5 B3:.5 A3:.5 G3:.5", "F#3:1 E3:3"]
    for k, sp in enumerate(low):                              # Theme A, complete, low, under the greeting: both at once
        gtr(s, 7 + k, sp, vel=0.7, pan=-0.1, gain=1.1)
        for off, beats, m, _ in E.notes(sp):
            P.stems["pads"].add(I.strings([m], P.dur(s, 7 + k, off, beats), 0.45, attack=0.06, release=0.2), P.t(s, 7 + k, off), 1.0, -0.3)
        mar(s, 7 + k, B[k], vel=0.55, tr=tr)
        acc(s, 7 + k, B[k], vel=0.42, tr=tr)
    pad("guitar", I.nylon, s, 11, 0, 3.5, "A7sus4", center=57, vel=0.5, poly=False, strum=0.03, pan=0.2)
    for k in range(4):                                        # the first questions arrive
        fx("fx_house", SH.ping(0.5, k), at=195.4 + 0.5 * k, pan=(-0.6, 0.6, -0.3, 0.4)[k], gain=0.6, label="message pings begin" if k == 0 else None)


# ------------------------------------------------------------------ 8. A One-Person Middleware Layer (D minor, 120)
def s8():
    s = "s8"
    prog = {0: ["Dm"], 1: ["F"], 2: ["Dm/A", "Gm"], 3: ["Bb", "A7"], 4: ["Dm"], 5: ["F"], 6: ["Dm/A", "Gm"], 7: ["Bb", "A7"], 8: ["Bb", "C"], 9: ["Dm/A"], 10: ["A7b9"]}
    for b in range(11):
        ch = prog[b]
        cut = b == 10
        for k, bt in enumerate((0, 2)):                       # the old clock, wearing sandals
            fx("fx_time", SW.tick(k == 0, 0.9, b), sec=s, bar=b, beat=bt, gain=0.6, label="clock tick" if (b == 0 and k == 0) else None)
        machine(s, b, ch, vel=0.55 + 0.01 * b)
        for bt in (0, 2):
            if not (cut and bt == 2):
                kick(s, b, bt)
        for bt in (1, 3):
            if not (cut and bt == 3):
                snare(s, b, bt, 0.8)
        for k in range(16 if not cut else 12):                # the place's instruments made to do the office's work
            P.hit("hats", X.guira(0.3, long_=(k % 4 == 2)), s, b, k * 0.25, 0.8 if k % 4 == 0 else 0.5, pan=0.4, human=0.0)
        for bt in ((0, 1.5, 3) if b % 2 == 0 else (1, 2)):    # 3-2 clave
            P.hit("perc", I.clave(0.5), s, b, bt, 0.7, pan=-0.5, human=0.0)
        for bt, kind, hi in ((0.5, "open", True), (1.5, "slap", True), (2.5, "open", False), (3.5, "open", True)):
            if not (cut and bt > 2):
                P.hit("perc", I.conga(kind, hi, 0.55), s, b, bt, 0.7, pan=0.2)
        P.note("bass", X.sub808, s, b, 0, 1.9, E.bass_note(ch[0], 1), 0.8, human=0.0)
        if len(ch) > 1:
            P.note("bass", X.sub808, s, b, 2, 1.9, E.bass_note(ch[1], 1), 0.8, human=0.0)
    for k, sp in enumerate(A_MIN):                            # Theme A, sure of itself
        gtr(s, k, sp, vel=0.75, pan=-0.1, human=0.0, gain=1.1)
        P.line("keys", X.piano, s, k, sp, vel=0.75, human=0.0, dark=False)
    # bars 5-8: other people start the phrase and he has to finish it; the last one he does not
    acc(s, 4, "D4:.5 A4:1.5", vel=0.5)
    tpt(s, 5, "F4:.5 C5:1.5", vel=0.5, mute=True, vib=0, pan=0.5)
    mar(s, 6, "A4:.5 D5:1.5", vel=0.6, pan=0.5)
    for b, fin in ((4, "r:2 G4:.5 F4:.5 E4:1"), (5, "r:2 Bb4:.5 A4:.5 G4:1"), (6, "r:2 C5:.5 Bb4:.5 A4:.5 G4:.5")):
        gtr(s, b, fin, vel=0.75, pan=-0.1, human=0.0, gain=1.1)
        P.line("keys", X.piano, s, b, fin, vel=0.75, human=0.0, dark=False)
    for off, beats, m, _ in E.notes("F2:1 E2:1"):
        tuba(s, 7, off, beats * 0.9, m, 0.7)
    for b in (8, 9, 10):                                      # everyone at once
        up = 2 * (b - 8)
        acc(s, b, "D4:.5 A4:1.5", vel=0.5, tr=up)
        tpt(s, b, "D4:.5 A4:1.2", vel=0.55, beat=1, tr=up + 5, mute=True, vib=0, pan=0.5)
        if b < 10:
            mar(s, b, "D5:.5 A5:1.2", vel=0.6, beat=2, tr=up, pan=0.5)
    fx("fx_house", SH.phone_taps(20.0, 0.3, seed=3, rate=10), at=199.0, gain=0.5, pan=-0.6, label="thumbs typing replies")
    t, k = 199.8, 0
    while t < 219.3:                                          # "any update?": pings that never agree with the bar line
        fx("fx_house", SH.ping(0.5, k), at=t, pan=-0.7 if k % 2 == 0 else 0.3, gain=0.55 + 0.012 * k, label="message pings, every five eighths" if k == 0 else None)
        t += 1.25
        k += 1
    t, k = 206.4, 0
    while t < 219.3:
        fx("fx_house", SH.ping(0.5, k + 1), at=t, pan=0.75, gain=0.5 + 0.02 * k, label="a second layer of pings, every seven eighths" if k == 0 else None)
        t += 1.75
        k += 1
    fx("fx_house", SH.knock(3, 0.19, 0.7, seed=1), sec=s, bar=6, beat=2, gain=0.9, pan=0.5, label="knock at the door (nobody called ahead)")
    fx("fx_house", SH.card_beep(0.4), sec=s, bar=8, beat=3, gain=0.6, pan=-0.4, label="card terminal")
    fx("fx_house", SH.grind(13.5, 0.4, seed=1), at=206.0, gain=0.9, label="a low grinding tone (the cost accumulating)")
    fx("fx_house", X.hiss(0.7, 0.35), at=219.5, gain=1.0, label="hiss: everything stops")
    n = int(1.6 * SR)                                         # a phone buzzing on a table, unanswered
    tt = np.arange(n) / SR
    buzz = D.sat(np.sin(2 * np.pi * 155 * tt) * 2, 1.5) * (((tt % 0.8) < 0.45).astype(float)) * 0.3
    fx("fx_house", D.lp(buzz, 1200) + D.bp(np.random.default_rng(5).standard_normal(n), 800, 2500) * np.abs(buzz) * 0.5, at=220.5, gain=0.8, pan=0.3, label="phone buzzing, unanswered")
    fx("fx_place", SW.surf(6.5, 0.5, period=6.0, seed=21, calm=0.2), at=221.0, gain=0.7, label="one wave")
    pad("pads", I.pad, s, 12, 0, 3.5, "Dmadd9", center=62, vel=0.3)
    for k in range(3):
        fx("fx_house", SH.ping(0.5, 2), at=223.5 + 0.9 * k, pan=0.5, gain=0.45 - 0.1 * k, label="pings nobody answers" if k == 0 else None)
    P.chord("pads", I.pad, s, 14, 0, 6, pitches=[45, 55, 57, 62], vel=0.3)


# ------------------------------------------------------------------ 9. Call Ahead (G major over D, 120)
def s9():
    s = "s9"
    fx("fx_house", SH.dial("322", 0.3), at=228.0, gain=0.9, pan=-0.2, label="dialing: he calls ahead")
    fx("fx_house", SH.ringback(1, 1.0, 0.5, 0.35, answered=True), at=228.9, gain=0.9, pan=-0.2, label="ringing tone, and someone picks up")
    mar(s, 1, "A4:.5 D5:1", vel=0.55, beat=1.5)               # the place answers: '¿bueno?'
    for b in range(1, 19):                                    # the practice rebuilt by hand, one note per bar
        k_max = min(b, 8)
        for k in range(k_max):
            if b == 18 and k > 3:
                break
            P.note("guitar", I.nylon, s, b, k * 0.5, 0.9, OST[k], 0.5 * (1.0 if k % 2 == 0 else 0.85), pan=-0.4, human=0.012)
        if b <= 8:                                            # one soft hammer tap under each new note: "Not announced. Rebuilt."
            fx("fx_time", SH.hammer(1, 0.3, 0.5, seed=b), sec=s, bar=b, beat=(k_max - 1) * 0.5, gain=0.5, pan=-0.4, label="soft hammer taps, one per new note" if b == 1 else None)
    prog = {2: ["G"], 3: ["Cmaj7"], 4: ["Em7"], 5: ["Am7", "D7"], 6: ["Cmaj7"], 7: ["G/D", "D7sus4"],
            8: ["G/D"], 9: ["Cmaj7"], 10: ["G/B", "Em7"], 11: ["Cadd9", "G/D"], 12: ["G/D"], 13: ["Cmaj7"], 14: ["G/B", "Em7"], 15: ["Cadd9", "G/D"]}
    for b, ch in prog.items():
        peak = 8 <= b <= 11
        for k, name in enumerate(ch):
            bt, bts = (0, 4) if len(ch) == 1 else (2 * k, 2)
            bass(s, b, bt, bts * 0.9, name, vel=0.7 if peak else 0.6)
            pad("pads", I.strings, s, b, bt, bts, name, center=64, vel=0.5 if peak else (0.22 if b < 4 else 0.32))
            if peak:
                pad("pads", I.choir, s, b, bt, bts, name, center=67, vel=0.3)
                pad("keys", I.rhodes, s, b, bt, bts, name, center=60, vel=0.3, poly=False, strum=0.01, pan=0.3)
        if b >= 4:
            P.hit("kick", I.bombo(0.8), s, b, 0, 0.9 if peak else 0.7)
            P.hit("perc", I.conga("open", False, 0.5), s, b, 2, 0.8 if peak else 0.6, pan=0.3)
            P.pattern("hats", lambda: I.maraca(0.3), s, 1, [k * 0.5 for k in range(8)], first_bar=b, accents=[1, 0.6], gain=0.8 if peak else 0.55)
        if peak:
            P.hit("snare", I.rim(0.5), s, b, 2, 0.8, pan=-0.2)
            P.hit("perc", I.conga("open", True, 0.5), s, b, 3.5, 0.6, pan=0.3)
    mar(s, 2, B[0], vel=0.6)                                  # the whole greeting, for the first time
    mar(s, 3, B[1], vel=0.6)
    gtr(s, 4, B[2], vel=0.65, pan=0.1)
    mar(s, 5, B[3], vel=0.6)
    acc(s, 5, B[3], vel=0.4)
    gtr(s, 6, B[4], vel=0.65, pan=0.1)
    tpt(s, 6, B[4], vel=0.3, mute=True, vib=0.3, gain=0.6)
    mar(s, 7, B[5] + " G4:3", vel=0.6)                        # 'que tengas un buen día': said together
    gtr(s, 7, B[5] + " G4:3", vel=0.65, pan=0.1)
    for k, sp in enumerate(A_MAJ):                            # Theme A rebuilt: plain, warm, the fullest moment of the piece
        last = k == 3
        gtr(s, 8 + k, sp, vel=0.8, pan=0.05, gain=1.25)
        for off, beats, m, _ in E.notes(sp):
            P.stems["pads"].add(I.strings([m + 12], P.dur(s, 8 + k, off, beats), 0.5, attack=0.05, release=0.25), P.t(s, 8 + k, off), 1.0, 0.3)
        tpt(s, 8 + k, sp, vel=0.35, vib=0.5, gain=0.55)
        if last:
            gtr(s, 12, "D4:1", vel=0.6, pan=0.05)
    mar(s, 12, A_MAJ[0], vel=0.6)                             # shared out: "let others carry their part"
    flu(s, 13, A_MAJ[1], vel=0.55, tr=12)
    acc(s, 14, A_MAJ[2], vel=0.5)
    gtr(s, 15, "D4:1.5 E4:.5 F#4:1 E4:1", vel=0.75, pan=0.05, gain=1.2)   # Griff's notes, on the author's own instrument
    gtr(s, 16, "D4:2", vel=0.65, pan=0.05)
    for k, sp in enumerate(GRIFF):
        rho(s, 12 + k, sp, vel=0.35)
    P.line("mallets", I.vibes, s, 13, "r:3 C#6:.5 F#5:1.5", vel=0.3, sustained=False, pan=0.5)   # the mirror, now merely useful
    for b in (16, 17):                                        # the wink: two bars of dance-pop for the Lady Gaga joke
        for bt in range(4):
            kick(s, b, bt)
            P.hit("hats", X.hat(0.4, open_=True), s, b, bt + 0.5, 0.7, pan=0.3, human=0.0)
        for bt in (1, 3):
            P.hit("snare", X.clap(0.7), s, b, bt, 0.8, human=0.0)
        for k, name in enumerate((("G", "D"), ("Em", "C"))[b - 16]):
            for e in range(4):
                P.note("bass", X.sub808, s, b, 2 * k + e * 0.5, 0.4, E.bass_note(name, 1), 0.7, human=0.0)
            pad("pads", I.pad, s, b, 2 * k, 2, name, center=64, vel=0.4, cut=3500.0, attack=0.02, release=0.2)
        rho(s, b, "D5:1.5 E5:.5 F#5:1 E5:1", vel=0.6, pan=0.0)
        P.line("mallets", X.glock, s, b, "D5:1.5 E5:.5 F#5:1 E5:1", vel=0.4, transpose=12, sustained=False, pan=0.4)
    fx("fx_place", X.scratch(0.5), at=263.5, gain=0.8, label="record scratch")
    pad("guitar", I.nylon, s, 18, 0, 3, "Dsus2", center=57, vel=0.55, poly=False, strum=0.03, pan=0.1)
    fx("fx_house", SH.scrabble(1.3, 0.45, seed=1), at=264.6, gain=0.9, pan=-0.6, label="scrabbling in the roof beam")


# ------------------------------------------------------------------ 10. La Loca Tlacuaque (G minor polka, 120)
def s10():
    s = "s10"
    fx("fx_house", I.bombo(0.9) * 0.9, at=266.5, gain=1.0, pan=-0.4, label="thud: she drops from the beam")
    fx("fx_house", SH.scrabble(0.9, 0.45, seed=3), at=266.7, gain=0.8, pan=-0.3)
    tuba(s, 0, 2, 0.5, "G2", 0.6)
    tuba(s, 0, 3, 0.5, "D2", 0.6)
    prog = {1: ["Gm", "Gm"], 2: ["D7", "D7"], 3: ["Gm", "Cm"], 4: ["Cm", "D7"], 5: ["G", "Gm"], 6: ["Eb7", "D7"], 9: ["G", "C"]}
    for b, ch in prog.items():
        for bt in range(4):
            if b == 6 and bt >= 2:
                break
            name = ch[bt // 2]
            root = E.bass_note(name, 2)
            tuba(s, b, bt, 0.45, root if bt % 2 == 0 else root - 5, 0.65)                 # oom
            P.stems["accordion"].add(X.accordion(E.voicing(name, 64), 0.16, 0.45), P.t(s, b, bt + 0.5), 1.0, -0.3)   # pah
            kick(s, b, bt, 0.6)
            P.hit("snare", X.snare(0.5, ghost=True), s, b, bt + 0.5, 0.7, human=0.0)
    F = "G4:.25 A4:.25 Bb4:.25 B4:.25 C5:.25 C#5:.25 D5:.5 r:.5 Eb5:.25 D5:.25 G5:.5"
    for tr, v in ((0, 0.65), (12, 0.0)):
        if v:
            mar(s, 1, F, vel=v, pan=0.3)
    P.line("mallets", X.toypiano, s, 1, F, vel=0.5, transpose=12, sustained=False, pan=-0.3)
    for k, bt in enumerate((1, 3)):                           # two glass apples go to their grave
        fx("fx_house", X.glass_crash(0.7, seed=k), sec=s, bar=2, beat=bt, gain=1.0, pan=-0.5 + k * 1.0, label=f"glass apple no. {k + 1}")
        P.line("mallets", X.glock, s, 2, "E6:.25 G6:.25 B6:.25" if k == 0 else "G6:.25 B6:.25 D7:.25", beat=bt + 0.25, vel=0.4, sustained=False, pan=0.5)
        P.line("mallets", X.toypiano, s, 2, "F6:.25 E6:.5", beat=bt + 1.0 if k == 0 else bt + 0.5, vel=0.45, sustained=False, pan=-0.4)
    for k in range(16):                                       # claws on tile
        P.hit("perc", I.rim(0.4), s, 3, k * 0.25, 0.7 if k % 4 == 0 else 0.4, pan=-0.6 + 0.08 * k, human=0.0)
    fx("fx_house", SH.trickle(0.9, 0.4, seed=1), sec=s, bar=3, beat=1, gain=0.9, pan=-0.7, label="a trickle, two inches from the toilet")
    fx("fx_house", SH.trickle(0.8, 0.4, seed=2), sec=s, bar=3, beat=2.5, gain=0.9, pan=0.7, label="...and in the sink")
    tuba(s, 3, 3, 0.9, "G1", 0.8)
    mar(s, 4, "G4:.25 A4:.25 Bb4:.25 C5:.25 D5:.25 Eb5:.25 F#5:.25 G5:.25 A5:.25 Bb5:.25 C6:.25 D6:.25", vel=0.65, pan=0.3)   # the run to the pool
    fx("fx_house", X.splash(0.7), sec=s, bar=5, beat=0, gain=1.0, pan=0.4, label="pool splash")
    P.chord("guitar", I.nylon, s, 5, 0, 1.5, name="G", center=62, vel=0.7, poly=False, strum=0.01)
    mar(s, 5, "r:2 G3:.25 A3:.25 Bb3:.25 B3:.25 C4:.25 C#4:.25 D4:.5", vel=0.6, pan=0.3)                # back in, wet
    fx("fx_house", X.door_slam(0.8), sec=s, bar=6, beat=2, gain=1.0, label="the door slams")
    fx("fx_house", SH.lock(0.6), sec=s, bar=6, beat=2.8, gain=0.9, pan=0.2, label="and locks")
    trem = I.strings([54, 57, 60, 63], P.dur(s, 7, 0, 7.5) - 0.6, 0.3, attack=0.3, release=0.4)   # the held breath under the sofa
    trem = trem * (0.55 + 0.45 * np.sign(np.sin(2 * np.pi * 11 * np.arange(len(trem)) / SR)))
    P.stems["pads"].add(trem, P.t(s, 7, 0), 0.7, 0.0)
    tuba(s, 7, 2, 1.5, "D2", 0.6)
    fx("fx_house", X.hiss(0.8, 0.35), at=281.5, gain=1.0, pan=-0.3, label="possum hiss, from under the sofa")
    for bt in (0, 0.5, 1, 1.5, 2, 3):                         # la patrulla verde knocks: in the rhythm of 'hola, buenos días'
        fx("fx_house", SH.knock(1, 0.2, 0.7, seed=int(bt * 2)), sec=s, bar=8, beat=bt, gain=0.9, pan=0.5, label="la patrulla verde knocks (in the greeting's rhythm)" if bt == 0 else None)
    tpt(s, 9, B[0], vel=0.65, vib=0.8)                        # and greets
    mar(s, 9, B[0], vel=0.6)
    fx("fx_house", X.hiss(0.5, 0.25), sec=s, bar=9, beat=3, gain=0.7, pan=0.7, label="one last hiss as she is carried out")
    for k in range(4):                                        # four Cocas: the count-in
        fx("fx_house", SH.bottle(0.5, seed=k), sec=s, bar=10, beat=k, gain=1.0, pan=-0.6 + 0.4 * k, label=f"Coca no. {k + 1}")
    tpt(s, 10, "D5:.3", vel=0.6, beat=3.5, mute=True, vib=0, legato=0.5)   # ...and one pollo
    P.log.append((round(P.t(s, 10, 3.5), 3), "leads", "one pollo (a muted-trumpet 'bok')"))


# ------------------------------------------------------------------ 11. The Next Repetition (Gadd9/D, 120)
def s11():
    s = "s11"
    fx("fx_place", SW.surf(15.0, 0.5, period=7.0, seed=31, calm=0.5), at=287.5, gain=0.7, label="surf, calm")
    fin = [(50, 0, 0.5), (57, 0.5, 0.5), (62, 1, 0.5), (57, 1.5, 0.5), (55, 2, 1), (59, 3, 1)]   # the practice in the greeting's rhythm
    for b in range(5):
        for k, (m, bt, bts) in enumerate(fin):
            if b == 4 and k >= 3:                             # it begins a fifth time and is left mid-figure
                break
            P.note("guitar", I.nylon, s, b, bt, bts * 1.6, m, 0.55 * (1.0 if k % 2 == 0 else 0.85), pan=-0.3, human=0.012)
    rho(s, 0, "D4:8", vel=0.3)
    rho(s, 2, "D4:8", vel=0.25)
    P.chord("pads", I.strings, s, 3, 0, 9, pitches=[50, 55, 57, 59, 66], vel=0.2, attack=1.5, release=3.0)
    fx("fx_house", SH.drip(0.5, seed=40), at=289.1, gain=0.6, pan=-0.5, label="the leak, still dripping")
    fx("fx_house", SH.ping(0.5, 0), at=291.5, gain=0.6, pan=0.6, label="a message ping...")
    gtr(s, 2, "G4:.5 B4:.5 D5:1", vel=0.6, pan=0.2)           # ...answered, with the greeting's first three notes
    laugh(s, 2, 2.0, gain=0.6)
    fx("fx_place", SW.broom(1, 0.9, 0.45, seed=77), at=297.5, gain=0.8, pan=0.4, label="one last sweep of the broom")


def render(out):
    os.makedirs(out, exist_ok=True)
    for f in (s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11):
        f()
        print("   rendered", f.__name__, flush=True)
    n = int(round(S.SONG_SECONDS * SR))
    for name, bus in P.stems.items():
        x = bus.x[:n]
        sf.write(os.path.join(out, name + ".wav"), x, SR, subtype="FLOAT")
        print(f"   {name:10s} peak {np.abs(x).max():5.2f}  rms {20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12):6.1f} dB")
    json.dump({"kick_times": sorted(KICKS)}, open(os.path.join(out, "sidechain.json"), "w"))
    cues = sorted(P.log)
    json.dump([{"t": t, "stem": st, "cue": c} for t, st, c in cues], open(os.path.join(out, "cues.json"), "w"), indent=1, ensure_ascii=False)
    print(f"   {len(cues)} labelled sound-effect cues, {len(set(c for _, _, c in cues))} distinct")


if __name__ == "__main__":
    render(sys.argv[1])
