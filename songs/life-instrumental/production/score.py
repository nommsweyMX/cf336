"""THE NEXT REPETITION: the plan of the piece. Eleven scenes, each a stage of the life the book tells
(plus the possum night), with its own tempo and key. Everything else reads from here."""
import numpy as np

import engine as E

SR = E.SR
TITLE = "The Next Repetition"
FILE_BASE = "The_Next_Repetition"
BEAT = 0.5                                # nominal beat for delay times (120 bpm)

#        name   bars bpm  beats  key                 stage of the life
SCENES = [
    ("s1",  15, 120, 4, "D minor",          "1. Internal Clock: the builder, before Vallarta"),
    ("s2",   9,  90, 4, "A Lydian / D minor", "2. \"I said it was awful\": the first visit"),
    ("s3",   7,  60, 4, "D minor thawing",   "3. It Decayed in Pieces: away, and the slow return"),
    ("s4",  16,  60, 2, "G major (6/8)",     "4. The Absence of Pressure: nothing pushes"),
    ("s5",  12, 120, 4, "E minor",           "5. Observation with Little Knives: the cafe table"),
    ("s6",   7,  60, 4, "B minor",           "6. The Mirror Turns: \"I am in it\""),
    ("s7",  12,  90, 4, "D major",           "7. Economico, Not Barato: the work of living there"),
    ("s8",  15, 120, 4, "D minor",           "8. A One-Person Middleware Layer: the helper trap"),
    ("s9",  19, 120, 4, "G major over D",    "9. Call Ahead: the builder, rebuilt"),
    ("s10", 11, 120, 4, "G minor / G major", "10. La Loca Tlacuaque: the possum night"),
    ("s11",  6, 120, 4, "G major over D",    "11. The Next Repetition"),
]
TL = E.Timeline([E.Section(n, bars, bpm, beats=bt, key=k, stage=st) for n, bars, bpm, bt, k, st in SCENES], t0=0.0, tail=3.0)
SONG_SECONDS = TL.total

# For the shared mastering chain: sections on one global bar index.
SECTIONS, _b = [], 0
for _n, _bars, *_ in SCENES:
    SECTIONS.append((_n, _b, _bars, 1.0, ""))
    _b += _bars


def t(bar):
    """Seconds at a global bar index (bars counted through all scenes)."""
    for name, b0, nb, *_ in SECTIONS:
        if bar < b0 + nb or name == SECTIONS[-1][0]:
            return TL.t(name, bar - b0, 0.0)


FLICKER_AT = 111.03                       # scene 4: the lights flicker once (done to the whole mix)


def mix_automation(mix):
    """Whole-mix gestures: the single light flicker in scene 4 dips everything for a blink."""
    g = np.ones(len(mix), dtype=np.float32)
    a, z, r = int(FLICKER_AT * SR), int((FLICKER_AT + 0.13) * SR), int(0.006 * SR)
    g[a:z] = 0.12
    g[a - r:a] = np.linspace(1, 0.12, r)
    g[z:z + r] = np.linspace(0.12, 1, r)
    return mix * g[:, None]
