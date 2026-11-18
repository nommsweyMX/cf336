"""The player: puts notes, chords, drum hits and sound effects on the stems, in bars and beats.

    P = Player(timeline, stem_names)
    P.note("leads", I.trumpet, "return", bar=2, beat=1.5, beats=2, pitch="A4", vel=0.7)
    P.line("leads", I.flute, "bay", bar=0, spec="D5:1 F5:.5 A5:2.5")
    P.chord("keys", I.rhodes, "bay", bar=0, beat=0, beats=4, name="Dm9")
    P.hit("drums", X.kick(), "old_life", bar=0, beat=0)
    P.fx("fx_place", S.surf(20), at=101.3, gain=0.8)
"""
import numpy as np

import engine as E
import synth_dsp as D

SR = D.SR


class Player:
    def __init__(self, timeline, stem_names, seed=7):
        self.tl = timeline
        self.stems = E.stems(stem_names, timeline.total)
        self.rng = np.random.default_rng(seed)
        self.log = []                                      # (time, stem, label) for the cue sheet

    # ---- time
    def t(self, sec, bar=0, beat=0.0):
        return self.tl.t(sec, bar, beat)

    def dur(self, sec, bar, beat, beats):
        return self.tl.t(sec, bar, beat + beats) - self.tl.t(sec, bar, beat)

    def _human(self, human):
        return float(self.rng.normal(0, human)) if human else 0.0

    # ---- pitched
    def note(self, stem, inst, sec, bar, beat, beats, pitch, vel=0.6, pan=0.0, gain=1.0, human=0.004, legato=1.0, sustained=True, **kw):
        """One note. `inst(midi, dur_seconds, vel, **kw)` for sustained instruments; for struck ones
        (marimba, glock...) pass sustained=False and it is called as inst(midi, vel, **kw)."""
        m = E.midi(pitch)
        t0 = self.t(sec, bar, beat) + self._human(human)
        d = self.dur(sec, bar, beat, beats) * legato
        v = float(np.clip(vel * (1 + self.rng.normal(0, 0.05)), 0.05, 1.2))
        y = inst(m, d, v, **kw) if sustained else inst(m, v, **kw)
        self.stems[stem].add(y, t0, gain, pan)
        return t0

    def line(self, stem, inst, sec, bar, spec, beat=0.0, vel=0.6, transpose=0, **kw):
        """A melody from a note spec ('D4:1 F4:.5 r:.5 A4:2'); per-note velocities override `vel`."""
        for off, beats, m, v in E.notes(spec):
            self.note(stem, inst, sec, bar, beat + off, beats, m + transpose, v if v is not None else vel, **kw)

    def chord(self, stem, inst, sec, bar, beat, beats, name=None, pitches=None, center=60, vel=0.5, pan=0.0, gain=1.0,
              poly=True, strum=0.0, n=None, spread=False, human=0.003, legato=1.0, sustained=True, **kw):
        """A chord by name (voiced around `center`) or explicit pitches. poly=True means the instrument takes
        a list of notes (strings, pad, choir, accordion); otherwise one call per note, with optional strum
        (seconds between strings, negative = downward)."""
        ps = [E.midi(p) for p in pitches] if pitches else E.voicing(name, center, n=n, spread=spread)
        t0 = self.t(sec, bar, beat) + self._human(human)
        d = self.dur(sec, bar, beat, beats) * legato
        if poly:
            self.stems[stem].add(inst(ps, d, vel, **kw), t0, gain, pan)
            return ps
        order = ps if strum >= 0 else ps[::-1]
        for k, m in enumerate(order):
            v = float(np.clip(vel * (1 + self.rng.normal(0, 0.06)), 0.05, 1.2))
            y = inst(m, d, v, **kw) if sustained else inst(m, v, **kw)
            self.stems[stem].add(y, t0 + abs(strum) * k, gain, pan)
        return ps

    def arp(self, stem, inst, sec, bar, name, pattern, step=0.5, beats=None, center=60, vel=0.5, octave_up_last=False,
            start_beat=0.0, sustained=True, ring=1.0, **kw):
        """Arpeggiate a chord: `pattern` is a list of chord-tone indices (None = rest), one per `step` beats."""
        ps = E.voicing(name, center)
        for k, idx in enumerate(pattern):
            if idx is None:
                continue
            m = ps[idx % len(ps)] + 12 * (idx // len(ps))
            self.note(stem, inst, sec, bar, start_beat + k * step, (beats or step) * ring, m, vel * (1.0 if k % 2 == 0 else 0.85), sustained=sustained, **kw)

    # ---- unpitched
    def hit(self, stem, sig, sec, bar, beat, gain=1.0, pan=0.0, human=0.003):
        self.stems[stem].add(sig, self.t(sec, bar, beat) + self._human(human), gain, pan)

    def pattern(self, stem, make, sec, bars, beats, gain=1.0, pan=0.0, accents=None, human=0.003, first_bar=0):
        """Repeat a drum pattern: `make()` builds one hit; `beats` are the positions inside each bar."""
        for b in range(first_bar, first_bar + bars):
            for k, bt in enumerate(beats):
                a = accents[k % len(accents)] if accents else 1.0
                self.hit(stem, make(), sec, b, bt, gain * a, pan, human)

    # ---- sound effects
    def fx(self, stem, sig, at=None, sec=None, bar=0, beat=0.0, gain=1.0, pan=0.0, label=None, lead=0.0):
        """Place an effect at an absolute time (`at`) or at a bar/beat. `lead` seconds starts it early so that
        its main event (not its first sample) lands on the time given."""
        t0 = (at if at is not None else self.t(sec, bar, beat)) - lead
        self.stems[stem].add(np.asarray(sig), t0, gain, pan)
        if label:
            self.log.append((round(t0 + lead, 3), stem, label))
        return t0
