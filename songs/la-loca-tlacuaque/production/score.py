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
LYRICS_FINAL = True     # the final lyric is in


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
    """Ad-lib / crowd answer: starts on swung slot `slot` of `bar`, must end within `slots` sixteenths.

    The window may run under the next line's pickup syllables: ad-libs are panned to the sides."""
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


# ------------------------------------------------------------------ structure
SECTIONS = [  # (name, start_bar, n_bars, energy 0..1, notes)
    ("intro",  0,  4, 0.2, "vinyl crackle, toy-piano jingle at half speed, whisper, spoken intro, drum pick-up"),
    ("verse1", 4,  8, 0.7, "the break-in, the two glass apples, the bathroom crimes"),
    ("hook1",  12, 8, 1.0, "jingle: 4 sung lines, 3 question/answer bars, title line"),
    ("verse2", 20, 8, 0.75, "the pool, the last-second return, locked in, under the sofa"),
    ("hook2",  28, 8, 1.0, "jingle, new questions"),
    ("bridge", 36, 8, 0.8, "half-time stomp, sirens: la patrulla verde, 4 Cocas and 1 pollo"),
    ("verse3", 44, 8, 0.8, "the capture; they leave with 4 Cocas, 1 pollo and 1 loca tlacuaque"),
    ("hook3",  52, 8, 1.0, "final jingle: the count"),
    ("outro",  60, 4, 0.2, "spoken, toy piano, whisper, last hiss"),
]
TOTAL_BARS = 64
TAIL_SECONDS = 3.0
SONG_SECONDS = T0 + TOTAL_BARS * BAR + TAIL_SECONDS

# ------------------------------------------------------------------ intro
_A(0, 4, "Culón, colito... culón, colito...", slots=12, style="whisper")
_S(1, 0.0, "¿Oíste eso, carnal? Algo se mueve en la viga...", 7.0)
_A(3, 0, "¿En la viga? ¡A la verga!", slots=8, pan=-0.3)
_A(3, 9, "¡Aguas!", slots=4, role="lead")

# ------------------------------------------------------------------ verse 1
_R(4, "Se metió por la viga, la viga de arriba,")
_R(5, "hocico de rata y tremenda barriga,")
_R(6, "dos manzanas de vidrio, las joyas del comedor,")
_F(6, 3.5, "glass", pan=-0.4, seed=0)                     # apple number one
_F(6, 3.875, "glass", pan=0.4, seed=1, amp=0.9)           # apple number two
_R(7, "se las llevó al panteón, y luego vino el olor...")
_A(7, 14, "¡Ahí nos vidrios!", slots=6, pan=-0.45)
_R(8, "se cagó a dos pulgadas de la taza, ¡qué puntería!")
_R(9, "tenía el trono enfrente y le atinó al azulejo,")
_A(9, 14, "¡Y ni pagó piso!", slots=6, pan=0.45)
_R(10, "y se mió en el lavabo la cochina porquería,", tts="y se meó en el lavabo la cochina porquería,")
_R(11, "¡mapache de la pulga, ni mi tío bien pedo!")

# ------------------------------------------------------------------ hook (a schoolyard taunt walking down Dm - C - Bb - A7)
# Written in the gang's octave; the kids' layer sings it an octave up. Stressed syllables sit on the beats.
H_L1 = ("-0.25 .25 53 cu; 0 .75 53 lón; .75 .25 50 co; 1 .5 55 li; 1.5 .25 53 to; "
        "1.75 .25 50 cu; 2 .75 53 lón; 2.75 .25 50 co; 3 .5 55 li; 3.5 .25 50 to")
H_L2 = ("-0.25 .25 52 me; 0 .5 55 ne; .5 .25 55 a; .75 .25 52 la; 1 .5 57 co; 1.5 .25 55 la; "
        "1.75 .25 52 po; 2 .5 55 qui; 2.5 .25 52 to; 2.75 .25 52 po; 3 .5 55 qui; 3.5 .25 52 to")
H_L3 = ("-0.25 .25 50 la; 0 .5 53 lo; .5 .25 53 ca; .75 .25 50 tla; 1 .25 55 cua; 1.25 .25 53 que; 1.5 .25 50 de; "
        "1.75 .25 50 la; 2 .5 53 ca; 2.5 .25 53 lle; 2.75 .25 50 ce; 3 .5 55 rra; 3.5 .25 53 da")
H_L4 = ("-0.25 .25 49 cu; 0 .75 52 lón; .75 .25 49 co; 1 .5 55 li; 1.5 .25 52 to; 1.75 .25 49 co; 2 .5 55 li; "
        "2.5 .25 52 ta; 2.75 .25 49 pe; 3 .5 52 la; 3.5 .5 50 da")
H_TITLE = ("-0.25 .25 52 la; 0 .5 55 lo; .5 .25 55 ca; .75 .25 52 tla; 1 .25 57 cua; 1.25 .25 55 que; 1.5 .25 52 de; "
           "1.75 .25 52 la; 2 .5 55 ca; 2.5 .25 52 lle; 2.75 .25 49 ce; 3 .5 52 rra; 3.5 .5 50 da")
H_TITLE_Y = "-0.75 .25 52 y; -0.5 .25 52 u; " + H_TITLE.replace("-0.25 .25 52 la;", "-0.25 .25 52 na;", 1)
ANS_TLAC = {0: "2.25 .25 50 la; 2.5 .5 53 tla; 3 .5 57 cua; 3.5 .25 53 que",       # over Dm
            1: "2.25 .25 48 la; 2.5 .5 52 tla; 3 .5 55 cua; 3.5 .25 52 que",       # over C
            2: "2.25 .25 50 la; 2.5 .5 53 tla; 3 .5 58 cua; 3.5 .25 53 que"}       # over Bb


def _hook(b, questions, answers, last_text, last_spec):
    """8 bars: four sung lines, three question (rapper) / answer (gang + kids) bars, the title line."""
    _H(b + 0, "¡Culón, colito! ¡Culón, colito!", H_L1)
    _A(b + 0, 13, "¡Menéale!", slots=5, role="lead", pan=0.2)
    _H(b + 1, "Menea la cola poquito, poquito,", H_L2)
    _H(b + 2, "la loca tlacuaque de la Calle Cerrada,", H_L3)
    _H(b + 3, "¡culón, colito, colita pelada!", H_L4)
    _A(b + 3, 14, "¡Fuchi!", slots=3, pan=-0.4)
    for k, (q, a) in enumerate(zip(questions, answers)):
        text, tts = q if isinstance(q, tuple) else (q, None)
        _R(b + 4 + k, text, anchor=4, lo=-2, hi=7, punch=False, tts=tts)
        _H(b + 4 + k, a[0], a[1], sid=f"h{b + 4 + k}a")
    _H(b + 7, last_text, last_spec)


_hook(12, ["¿Quién se metió?", "¿Quién se cagó?", ("¿Quién se mió?", "¿Quién se meó?")],
      [("¡La tlacuaque!", ANS_TLAC[0]), ("¡La tlacuaque!", ANS_TLAC[1]), ("¡La tlacuaque!", ANS_TLAC[2])],
      "¡La loca tlacuaque de la Calle Cerrada!", H_TITLE)

# ------------------------------------------------------------------ verse 2
_R(20, "Oyó la chancla y se peló pa' la alberca,")
_F(20, 3.625, "splash", pan=0.3)
_R(21, "flotando de muertita, la rata más terca,")
_R(22, "pero al cuarto pa' las doce, justo antes del portazo,")
_R(23, "se regresó la cabrona y se coló de un chingazo.")
_F(23, 3.6, "door", pan=-0.2)                               # slam, then the lock clicks on the next downbeat
_R(24, "Clic, hizo la chapa: quedó bien encerrada,")
_A(24, 14, "¡No mames!", slots=4, pan=0.45)
_R(25, "en la casa, sin salida, en la Calle Cerrada,")
_R(26, "debajo del sillón, la muy desgraciada,")
_R(27, "enseñando los dientes con la cola enroscada.")
_F(27, 3.45, "hiss", seconds=0.6, pan=-0.3)

_hook(28, ["¿Quién se peló?", "¿Quién se regresó?", "¿Quién se escondió?"],
      [("¡La tlacuaque!", ANS_TLAC[0]), ("¡La tlacuaque!", ANS_TLAC[1]), ("¡La tlacuaque!", ANS_TLAC[2])],
      "¡La loca tlacuaque de la Calle Cerrada!", H_TITLE)

# ------------------------------------------------------------------ bridge: la patrulla verde
_A(36, 2, "¡Uiu, uiu!", slots=6, pan=0.4)
_H(37, "¡Verde, verde, la patrulla verde!", "0 .5 57 ver; .5 .5 53 de; 1 .25 57 ver; 1.25 .25 53 de; 1.5 .25 50 la; "
       "1.75 .25 53 pa; 2 .5 57 tru; 2.5 .5 55 lla; 3 .25 53 ver; 3.25 .25 50 de")
_A(37, 14, "¡La verde!", slots=4, pan=-0.4)
_R(38, "De verde hasta los guantes, porque la loca muerde,", lo=1)
_R(39, "se bajaron de la troca, más verdes que un nopal,")
_R(40, "con cuatro Cocas heladas y un pollo asado, carnal,")
_A(40, 14, "¡Cuatro!", slots=4, pan=0.45)
_R(41, "¿era pa' carnada o el lonche del personal?")
_A(41, 14, "¡Ja!", slots=2, pan=-0.45)
_H(42, "¡Verde, verde, la patrulla verde!", "0 .5 55 ver; .5 .5 50 de; 1 .25 55 ver; 1.25 .25 50 de; 1.5 .25 50 la; "
       "1.75 .25 55 pa; 2 .5 58 tru; 2.5 .5 57 lla; 3 .5 55 ver; 3.5 .5 50 de")
_F(43, 0.0, "siren", seconds=2.4, pan=0.4)

# ------------------------------------------------------------------ verse 3
_R(44, "La sacaron del sillón jalándola del rabito,")
_A(44, 14, "¡Jálale!", slots=4, pan=0.45)
_R(45, "se hizo la muertita... ¡ay, qué pinche teatrito!")
_R(46, "con la lengua de fuera y el culón colito,")
_R(47, "pa' la jaula, mi reina, ¡ya llegó tu taxito!")
_R(48, "Trapeador con dientes, peluche de basurero,")
_R(49, "te vas con los de verde sin bañarte primero.")
_R(50, "Se fueron como vinieron: cuatro Cocas, un pollo,")
_R(51, "¡y una loca tlacuaque de pilón! Se acabó el rollo.")

# ------------------------------------------------------------------ final hook: the count
_hook(52, ["¿Y quién se la llevó?", "¿Con cuántas Cocas?", "¿Y cuántos pollos?"],
      [("¡La patrulla verde!", "1.5 .25 50 la; 1.75 .25 53 pa; 2 .5 57 tru; 2.5 .5 55 lla; 3 .5 57 ver; 3.5 .5 53 de"),
       ("¡Cuatro!", "2.5 .5 55 cua; 3 .25 52 tro"),
       ("¡Uno!", "2.5 .5 58 u; 3 .25 53 no")],
      "¡Y una loca tlacuaque de la Calle Cerrada!", H_TITLE_Y)

# ------------------------------------------------------------------ outro
_S(60, 0.5, "Y que alguien tape el pinche hoyo de la viga, carnal.", 6.5)
_A(62, 0, "¡A la verga!", slots=5, pan=-0.3)
_A(62, 8, "Culón, colito... culón, colito...", slots=14, style="whisper")
_F(63, 2.5, "hiss", seconds=0.9, pan=0.2)

# ------------------------------------------------------------------ checks
assert len({x["id"] for x in RAP + ADLIBS + SUNG + SPOKEN}) == len(RAP + ADLIBS + SUNG + SPOKEN), "duplicate ids"
