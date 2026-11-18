"""Timeline, note and chord helpers for the life instrumental.

A piece is a list of Sections, each with its own tempo (optionally ramping), meter and bar count.
`Timeline.t(section, bar, beat)` gives the time in seconds, so the arrangement can be written in bars and
beats while the tempo changes underneath.
"""
import math
import re

import numpy as np

import shared  # noqa: F401
import synth_dsp as D

SR = D.SR
_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def midi(name):
    """'C#4' / 'Bb2' / 'D5' -> MIDI number (C4 = 60). Integers pass through."""
    if isinstance(name, (int, np.integer)):
        return int(name)
    m = re.fullmatch(r"([A-Ga-g])([#b]*)(-?\d+)", name.strip())
    if not m:
        raise ValueError(f"bad note name: {name!r}")
    pc = _PC[m.group(1).upper()] + m.group(2).count("#") - m.group(2).count("b")
    return 12 * (int(m.group(3)) + 1) + pc


def notes(spec):
    """'D4:1 F4:.5 r:.5 A4:2' -> [(beat_offset, dur_beats, midi)], rests ('r') advance time.
    A note may carry a velocity: 'A4:2:0.8'. Several notes at once: 'D4+F4+A4:2'."""
    out, pos = [], 0.0
    for tok in spec.split():
        parts = tok.split(":")
        dur = float(parts[1])
        vel = float(parts[2]) if len(parts) > 2 else None
        if parts[0] not in ("r", "R"):
            for nm in parts[0].split("+"):
                out.append((pos, dur, midi(nm), vel))
        pos += dur
    return out


def spec_beats(spec):
    return sum(float(tok.split(":")[1]) for tok in spec.split())


_QUAL = {
    "": (0, 4, 7), "m": (0, 3, 7), "7": (0, 4, 7, 10), "m7": (0, 3, 7, 10), "maj7": (0, 4, 7, 11),
    "dim": (0, 3, 6), "dim7": (0, 3, 6, 9), "m7b5": (0, 3, 6, 10), "sus4": (0, 5, 7), "sus2": (0, 2, 7),
    "6": (0, 4, 7, 9), "m6": (0, 3, 7, 9), "9": (0, 4, 7, 10, 14), "m9": (0, 3, 7, 10, 14),
    "maj9": (0, 4, 7, 11, 14), "add9": (0, 4, 7, 14), "madd9": (0, 3, 7, 14), "aug": (0, 4, 8),
    "7b9": (0, 4, 7, 10, 13), "5": (0, 7),
}


def chord(name):
    """'Dm7', 'Bbmaj7', 'A7/C#' -> (root pitch class, intervals tuple, bass pitch class)."""
    base, _, bass = name.partition("/")
    m = re.fullmatch(r"([A-G][#b]?)(.*)", base)
    if not m or m.group(2) not in _QUAL:
        raise ValueError(f"bad chord: {name!r}")
    root = (_PC[m.group(1)[0]] + (1 if "#" in m.group(1) else -1 if "b" in m.group(1)[1:] else 0)) % 12
    bpc = root
    if bass:
        bpc = (_PC[bass[0]] + (1 if "#" in bass else -1 if "b" in bass[1:] else 0)) % 12
    return root, _QUAL[m.group(2)], bpc


def voicing(name, center=60, n=None, spread=False):
    """Chord tones placed as close as possible around `center` (smooth voice leading comes for free when
    successive chords use the same center). spread=True opens the voicing (drop the second voice an octave)."""
    root, iv, _ = chord(name)
    out = []
    for i in iv:
        pc = (root + i) % 12
        k = pc + 12 * round((center - pc) / 12)
        out.append(int(k))
    out = sorted(set(out))
    if n and len(out) > n:
        out = out[:n] if n < 3 else [out[0]] + out[-(n - 1):]
    if spread and len(out) >= 3:
        out[1] -= 12
        out = sorted(out)
    return out


def bass_note(name, octave=2):
    """Bass note (slash bass if given) in the given octave."""
    return 12 * (octave + 1) + chord(name)[2]


def root_note(name, octave=2):
    return 12 * (octave + 1) + chord(name)[0]


class Section:
    def __init__(self, name, bars, bpm, beats=4, bpm_end=None, key="", stage="", note=""):
        self.name, self.bars, self.bpm, self.beats = name, bars, float(bpm), beats
        self.bpm_end = float(bpm_end) if bpm_end else float(bpm)
        self.key, self.stage, self.note = key, stage, note
        self.start = 0.0

    @property
    def total_beats(self):
        return self.bars * self.beats

    def beat_time(self, b):
        """Seconds from the section start to beat b (float), with a linear tempo ramp over the section."""
        b0, b1, B = self.bpm, self.bpm_end, self.total_beats
        if abs(b1 - b0) < 1e-9:
            return 60.0 * b / b0
        k = (b1 - b0) / B
        return 60.0 / k * math.log((b0 + k * b) / b0)

    @property
    def seconds(self):
        return self.beat_time(self.total_beats)


class Timeline:
    def __init__(self, sections, t0=0.0, tail=0.0):
        self.sections = list(sections)
        self.by = {}
        t = t0
        for s in self.sections:
            assert s.name not in self.by, s.name
            s.start = t
            self.by[s.name] = s
            t += s.seconds
        self.t0, self.end, self.total = t0, t, t + tail

    def t(self, sec, bar=0, beat=0.0):
        """Time of (bar, beat) in a section; both zero-based, beat may be fractional or run past the bar.
        Bars past the end of the section continue at its final tempo."""
        s = self.by[sec]
        b = bar * s.beats + beat
        if b <= s.total_beats:
            return s.start + s.beat_time(max(b, -s.total_beats) if b >= 0 else 0.0) + (60.0 * b / s.bpm if b < 0 else 0.0)
        return s.start + s.seconds + 60.0 * (b - s.total_beats) / s.bpm_end

    def beat_len(self, sec, bar=0, beat=0.0):
        """Length of one beat (seconds) at that point."""
        return self.t(sec, bar, beat + 1) - self.t(sec, bar, beat)

    def section_at(self, t):
        for s in reversed(self.sections):
            if t >= s.start - 1e-9:
                return s
        return self.sections[0]

    def table(self):
        return [(s.name, round(s.start, 3), round(s.start + s.seconds, 3), s.bars, s.beats, s.bpm, s.bpm_end, s.key, s.stage)
                for s in self.sections]


def stems(names, seconds):
    n = int(round(seconds * SR))
    return {k: D.Bus(k, n) for k in names}


def swing(beat, amount=0.0, grid=0.5):
    """Delay every second `grid` subdivision by `amount` (0..0.33 of the grid pair)."""
    k = beat / grid
    if abs(k - round(k)) < 1e-6 and int(round(k)) % 2 == 1:
        return beat + amount * grid * 2
    return beat
