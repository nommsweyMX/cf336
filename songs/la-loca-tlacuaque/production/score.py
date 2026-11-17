"""The score for "La Loca Tlacuaque de la Calle Cerrada": the single source of truth that the beat,
the vocals and the mix all read. Nothing else decides tempo, structure, lyrics or placement.

Time grid
    t(bar, beat)    straight time in seconds (beat may be fractional)
    t16(bar, slot)  swung sixteenth-note slot (0..15 inside the bar; slots <0 or >15 cross bar lines).
                    Odd slots are delayed by SWING, so rap syllables and hats share the same pocket.
"""
SR = 44100
TITLE = "La Loca Tlacuaque de la Calle Cerrada"
BPM = 90.0
BEAT = 60.0 / BPM
BAR = 4 * BEAT
SWING = 0.57            # position of the off sixteenth inside its eighth (0.5 = straight)
T0 = 0.35               # lead-in before bar 0
KEY = "D minor"
LYRICS_FINAL = False    # flipped to True once the final lyric is in


def t(bar, beat=0.0):
    return T0 + bar * BAR + beat * BEAT


def t16(bar, slot):
    b, s = divmod(int(slot), 16)
    pair, odd = divmod(s, 2)
    return T0 + (bar + b) * BAR + pair * (BEAT / 2) + odd * (BEAT / 2) * SWING


# ------------------------------------------------------------------ harmony
CHORDS = {  # name -> bass note (MIDI) and a close voicing around middle C
    "Dm": dict(root=38, voicing=[62, 65, 69]),
    "C":  dict(root=36, voicing=[60, 64, 67]),
    "Bb": dict(root=34, voicing=[58, 62, 65]),
    "A":  dict(root=33, voicing=[57, 61, 64]),
    "A7": dict(root=33, voicing=[57, 61, 64, 67]),
    "Gm": dict(root=31, voicing=[55, 58, 62]),
}
PROG = {  # 4-bar loops by section type (section names start with one of these words)
    "intro":  ["Dm", "Dm", "Bb", "A"],
    "verse":  ["Dm", "Dm", "Bb", "A"],      # dark loop
    "hook":   ["Dm", "C", "Bb", "A7"],      # Andalusian cadence: the jingle walks down it
    "bridge": ["Dm", "Dm", "Gm", "A"],
    "outro":  ["Dm", "Dm", "Bb", "A"],
}


def section_of(bar):
    """-> (name, type, index of the bar inside the section, section length)"""
    for name, start, n, _, _ in SECTIONS:
        if start <= bar < start + n:
            typ = next((k for k in PROG if name.startswith(k)), "verse")
            return name, typ, bar - start, n
    return "tail", "outro", 0, 1


def chord_at(bar):
    _, typ, i, _ = section_of(bar)
    return CHORDS[PROG[typ][i % 4]]


# ------------------------------------------------------------------ voices (Kokoro v1.0 voice ids)
VOICES = {
    "lead": "em_alex",                               # the rapper: pitched/shaped deep and raspy by the vocal engine
    "gang": ["em_santa", "em_alex", "ef_dora", "am_onyx", "am_adam"],   # crowd shouts, answers, hook chant
    "kids": ["ef_dora", "af_heart", "af_bella"],     # the schoolyard-jingle layer (shifted up by the vocal engine)
}

# ------------------------------------------------------------------ helpers
def _N(spec):
    """'beat dur midi syl; ...' -> [(beat, dur_beats, midi, syllable)]"""
    out = []
    for part in spec.split(";"):
        b, d, m, s = part.split()
        out.append((float(b), float(d), int(m), s))
    return out


RAP, ADLIBS, SUNG, SPOKEN, FX = [], [], [], [], []


def _R(bar, text, anchor=12, lo=-3, hi=15, tts=None, punch=True, role="lead"):
    """One rap line. Its LAST STRESSED syllable lands on sixteenth slot `anchor` of `bar` (12 = beat 4, the
    snare). Earlier syllables fill the swung sixteenths before it (no earlier than slot `lo`, which may be
    negative = pickup in the previous bar); the last syllable must END by slot `hi`.
    punch=True: the gang doubles the rhyme (from the last stressed syllable on)."""
    RAP.append(dict(id=f"r{bar}", bar=bar, text=text, tts_text=tts, anchor=anchor, lo=lo, hi=hi, punch=punch,
                    role=role, lang="es"))


def _A(bar, slot, text, slots=3, role="gang", style="shout", pan=0.0, tts=None):
    """Ad-lib / crowd answer: starts on swung slot `slot` of `bar`, must end within `slots` sixteenths."""
    ADLIBS.append(dict(id=f"a{bar}_{slot}", bar=bar, slot=slot, slots=slots, text=text, tts_text=tts, role=role,
                       style=style, pan=pan, lang="es"))


def _H(bar, text, spec, role="gang", tts=None, sid=None):
    """One sung hook line: notes are (beat offset from the bar, duration in beats, MIDI note, syllable)."""
    SUNG.append(dict(id=sid or f"h{bar}", bar=bar, role=role, lang="es", text=text, tts_text=tts, notes=_N(spec)))


def _S(bar, beat, text, max_beats, style="spoken", role="lead", pan=0.0, tts=None):
    """Free-rhythm spoken line: starts at (bar, beat), must end within max_beats."""
    SPOKEN.append(dict(id=f"s{bar}_{beat}", bar=bar, beat=beat, max_beats=max_beats, text=text, tts_text=tts,
                       style=style, role=role, pan=pan, lang="es"))


def _F(bar, beat, kind, **kw):
    FX.append(dict(bar=bar, beat=beat, kind=kind, **kw))


# ------------------------------------------------------------------ SAMPLE CONTENT (placeholder until LYRICS_FINAL)
# The data format above is final; the lines below are a development sample and will be replaced.
SECTIONS = [  # (name, start_bar, n_bars, energy 0..1, notes)
    ("intro", 0, 4, 0.2, "toy-piano jingle, vinyl crackle, spoken intro"),
    ("verse1", 4, 8, 0.7, "boom-bap, dark piano loop, 808"),
    ("hook1", 12, 8, 1.0, "jingle hook: gang + kids chant, glockenspiel, sleigh bells, claps"),
    ("outro", 20, 2, 0.2, "spoken, toy piano"),
]
TOTAL_BARS = 22
TAIL_SECONDS = 3.0
SONG_SECONDS = T0 + TOTAL_BARS * BAR + TAIL_SECONDS

_S(1, 0.0, "¿Oíste eso, carnal? Algo se mueve en la viga...", 7.0)
_S(3, 0.0, "¿En la viga? ¡A la verga!", 3.5)
_A(3, 14, "¡Aguas!", slots=2)

_R(4, "Se metió por la viga, la viga de arriba,")
_R(5, "hocico de rata y tremenda barriga,")
_R(6, "dos manzanas de vidrio, las joyas del comedor,")
_R(7, "se las llevó al panteón, y luego vino el olor:")
_A(7, 6, "¡Crash!", slots=2)
_R(8, "cagó a dos pulgadas de la taza, ¡qué puntería!")
_A(8, 14, "¡No mames!", slots=3)
_R(9, "y se mió en el lavabo la cochina porquería,")
_R(10, "tenía la taza enfrente y le dio al azulejo,")
_R(11, "¡méndiga ojete, ni pa' cagar tienes reflejo!")
_A(11, 14, "¡Fuchi!", slots=2)

# hook: schoolyard taunt in D minor (F F D G F D = "nyah nyah" on the tonic)
_H(12, "¡Culón, colito! ¡Culón, colito!", "0 .5 62 Cu; .5 .5 65 lón; 1 .5 65 co; 1.5 .5 67 li; 2 .5 65 to; "
       "2.5 .25 62 cu; 2.75 .25 65 lón; 3 .25 65 co; 3.25 .25 67 li; 3.5 .5 62 to")
_H(13, "Menea la cola poquito, poquito,", "0 .5 62 Me; .5 .5 62 ne; 1 .5 65 a; 1.5 .5 65 la; 2 .5 67 co; 2.5 .5 65 la; "
       "3 .25 62 po; 3.25 .25 65 qui; 3.5 .25 62 to; 3.75 .25 60 po; 4 .25 62 qui; 4.25 .5 62 to", sid="h13")
_H(14, "la loca tlacuaque de la Calle Cerrada,", "0.5 .5 62 la; 1 .5 65 lo; 1.5 .5 65 ca; 2 .5 67 tla; 2.5 .5 69 cua; 3 .5 67 que; "
       "3.5 .25 65 de; 3.75 .25 65 la; 4 .5 67 Ca; 4.5 .5 65 lle; 5 .5 64 Ce; 5.5 .5 65 rra; 6 1 62 da")
_A(16, 0, "¿Quién se metió?", slots=6, role="lead")
_A(16, 8, "¡La tlacuaque!", slots=6)
_A(17, 0, "¿Quién se cagó?", slots=6, role="lead")
_A(17, 8, "¡La tlacuaque!", slots=6)
_H(18, "¡La loca tlacuaque de la Calle Cerrada!", "0.5 .5 62 la; 1 .5 65 lo; 1.5 .5 65 ca; 2 .5 67 tla; 2.5 .5 69 cua; 3 .5 67 que; "
       "3.5 .25 65 de; 3.75 .25 65 la; 4 .5 67 Ca; 4.5 .5 65 lle; 5 .5 64 Ce; 5.5 .5 65 rra; 6 1 62 da")

_S(20, 0.0, "Cuatro Cocas... un pollo... y una loca tlacuaque de la Calle Cerrada.", 7.5)
