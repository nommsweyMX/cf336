"""WHAT YOU REPEAT: the score. This file is the single source of truth for the production.

Everything (instrumental, vocals, mix) is placed on one global timeline:
    time(bar, beat) = T0 + bar * BAR + beat * BEAT        (seconds)
T0 is a short breath lead-in before the first heartbeat (bar 0, beat 0).
All MIDI note numbers here are ABSOLUTE (already transposed for the G-minor key change).
C4 = 60. Lead range is G3-G4 (a baritone/tenor male lead).
"""

SR = 44100
BPM = 124.0
BEAT = 60.0 / BPM          # 0.48387 s
BAR = 4 * BEAT             # 1.93548 s
T0 = 0.55                  # breath lead-in (s)
TOTAL_BARS = 140
TAIL_SECONDS = 11.0        # heartbeat-only tail after the last bar (slows to a stop + exhale)
SONG_SECONDS = T0 + TOTAL_BARS * BAR + TAIL_SECONDS   # ~282.5 s


def t(bar, beat=0.0):
    """Absolute time in seconds of (bar, beat)."""
    return T0 + bar * BAR + beat * BEAT


# ------------------------------------------------------------------ sections
# (name, start_bar, n_bars, energy 0-1, notes for the arrangement)
SECTIONS = [
    ("intro",        0,  8, 0.35, "breath, solo heartbeat 62 BPM x4 bars, beatbox hats creep in, heart doubles to 124 in bars 6-7, k-roll fill"),
    ("verse1",       8, 16, 0.55, "spoken deadpan lead; kick+hats+pf snare, rolling sub bass, Rhodes/organ stabs, filtered"),
    ("pre1",        24,  8, 0.65, "sung; filter opens, strings swell, snare roll + arp rise in last 2 bars"),
    ("chorus1",     32,  8, 0.95, "FULL DROP; Spanish ladder call-and-response; claps, supersaw pads, arps"),
    ("post1",       40,  8, 0.90, "post-chorus 4 bars vocal + 4 bars instrumental drop with acid line"),
    ("verse2",      48, 16, 0.60, "spoken; groove + acid hint"),
    ("pre2",        64,  8, 0.75, "sung; bigger build; total STOP on bar 71 beats 2.5-4"),
    ("chorus2",     72,  8, 1.00, "FULL DROP, thicker"),
    ("post2",       80,  4, 0.90, "post-chorus vocal"),
    ("verse3",      84,  8, 0.70, "chanted; drums thin to beatbox + acid synth; kick slams back last 2 bars"),
    ("bridge",      92, 16, 0.40, "HALF-TIME: piano, pads, heartbeat (62 BPM) is the only percussion"),
    ("breakdown",  108,  8, 0.20, "whispered; heartbeat + sustained pad only"),
    ("build",      116,  4, 0.80, "heartbeat races (doubles each bar), riser, snare roll, STOP on bar 119 beat 3-4"),
    ("final",      120,  8, 1.00, "KEY CHANGE to G minor; biggest drop, disco strings"),
    ("tag",        128,  4, 1.00, "Otra vez tag, G minor"),
    ("outro",      132,  8, 0.30, "layers peel away bar by bar; whispered; heart halves to 62"),
]
# After bar 140: TAIL_SECONDS of heartbeat only, slowing to a stop, then an exhale.

# ------------------------------------------------------------------ harmony
# voicing = keys/pad chord tones (MIDI), root = bass pitch class as MIDI in octave 2 (36-47)
CHORDS = {
    "Fm9":    {"root": 41, "voicing": [56, 60, 63, 67]},      # Ab C Eb G over F
    "Fm":     {"root": 41, "voicing": [56, 60, 65]},          # Ab C F
    "Dbmaj7": {"root": 37, "voicing": [53, 56, 60]},          # F Ab C over Db
    "Db":     {"root": 37, "voicing": [53, 56, 61]},          # F Ab Db
    "Ebadd9": {"root": 39, "voicing": [55, 58, 63, 65]},      # G Bb Eb F
    "Eb":     {"root": 39, "voicing": [55, 58, 63]},          # G Bb Eb
    "Cm7":    {"root": 36, "voicing": [55, 58, 60, 63]},      # G Bb C Eb
    "C7sus4": {"root": 36, "voicing": [53, 55, 58, 60]},      # F G Bb C
    "C7":     {"root": 36, "voicing": [52, 55, 58, 60]},      # E G Bb C
    "Ab":     {"root": 44, "voicing": [56, 60, 63]},          # Ab C Eb
    "Abmaj7": {"root": 44, "voicing": [55, 60, 63]},          # G C Eb over Ab
    "Dsus4":  {"root": 38, "voicing": [55, 57, 62]},          # G A D
    "D7":     {"root": 38, "voicing": [54, 57, 60, 62]},      # F# A C D
    "Gm":     {"root": 43, "voicing": [58, 62, 67]},          # Bb D G
    "Gm9":    {"root": 43, "voicing": [58, 62, 65, 69]},      # Bb D F A
    "Bb":     {"root": 46, "voicing": [58, 62, 65]},          # Bb D F
    "F":      {"root": 41, "voicing": [57, 60, 65]},          # A C F
    "Ebmaj7": {"root": 39, "voicing": [55, 58, 62]},          # G Bb D over Eb
}

_VERSE = ["Fm9", "Fm9", "Dbmaj7", "Dbmaj7", "Ebadd9", "Ebadd9", "Cm7", "Cm7"]
_PRE = ["Dbmaj7", "Dbmaj7", "Ebadd9", "Ebadd9", "Fm9", "Fm9", "C7sus4", "C7"]
_CHORUS = ["Fm", "Db", "Ab", "Eb"]
_BRIDGE = ["Dbmaj7", "Dbmaj7", "Abmaj7", "Abmaj7", "Ebadd9", "Ebadd9", "Fm9", "Fm9"]
_FINAL = ["Gm", "Ebmaj7", "Bb", "F"]

CHORD_BY_BAR = (
    ["Fm9"] * 8                                   # intro 0-7 (pad enters only bars 6-7, filtered)
    + _VERSE * 2                                  # verse1 8-23
    + _PRE                                        # pre1 24-31
    + _CHORUS * 2                                 # chorus1 32-39
    + _CHORUS * 2                                 # post1 40-47
    + _VERSE * 2                                  # verse2 48-63
    + _PRE                                        # pre2 64-71
    + _CHORUS * 2                                 # chorus2 72-79
    + _CHORUS                                     # post2 80-83
    + ["Fm9"] * 4 + ["Dbmaj7", "Dbmaj7", "Ebadd9", "C7"]   # verse3 84-91
    + _BRIDGE * 2                                 # bridge 92-107
    + ["Fm9"] * 8                                 # breakdown 108-115
    + ["Ebmaj7", "F", "Dsus4", "D7"]              # build 116-119 (pivot to G minor: bVI bVII V)
    + _FINAL * 2                                  # final 120-127
    + _FINAL                                      # tag 128-131
    + ["Gm9"] * 4 + ["Ebmaj7", "Ebmaj7", "Gm9", "Gm9"]   # outro 132-139
)
assert len(CHORD_BY_BAR) == TOTAL_BARS, len(CHORD_BY_BAR)


def chord_at(bar):
    return CHORD_BY_BAR[min(max(int(bar), 0), TOTAL_BARS - 1)]


# ------------------------------------------------------------------ voices (Kokoro v1.0)
VOICES = {
    "lead": {"voice": "am_michael", "note": "the narrator; sings English AND Spanish (gringo accent is part of the joke)"},
    "locals": {"voices": ["em_alex", "ef_dora", "em_santa"], "lang": "es", "note": "the town answering back"},
    "gang": {"voices": ["am_adam", "am_onyx", "am_puck", "am_liam", "af_heart", "af_bella"], "note": "arena crowd / group chat"},
}

# ------------------------------------------------------------------ sung phrases
# Each phrase: id, bar (absolute start bar), role, lang, text (for TTS), tts_text (optional override),
# notes: list of (beat_offset_from_phrase_bar, dur_beats, midi, syllable). Syllable count == len(notes).
# role: lead | crowd | backing.  harmony=True -> also render a diatonic third ABOVE in the current key.
def _tr(notes, semis):
    return [(b, d, m + semis, s) for (b, d, m, s) in notes]


def _N(spec):
    """spec: 'beat dur midi syl; ...'"""
    out = []
    for item in spec.split(";"):
        item = item.strip()
        if not item:
            continue
        b, d, m, s = item.split(None, 3)
        out.append((float(b), float(d), int(m), s))
    return out


# chorus building blocks (F minor)
CH = {
    "c1_lead": _N("0 .5 60 Ho; .5 .5 60 la; 1 .5 63 bue; 1.5 .5 60 nos; 2 .5 65 dí; 2.5 .5 63 as"),
    "c1_echo": _N("3 .5 60 Ho; 3.5 .5 56 la"),
    "c2_lead": _N("0 .5 61 Có; .5 .5 60 mo; 1 .5 60 es; 1.5 1 61 tás"),
    "c2_echo": _N("2.5 .375 56 Có; 2.875 .375 56 mo; 3.25 .25 60 es; 3.5 .5 61 tás"),
    "c3_lead": _N("0 .5 63 Bien; .5 .5 60 gra; 1 .5 58 cias; 1.5 .5 56 y; 2 1 60 tú"),
    "c3_echo": _N("3 .5 60 y; 3.5 .5 60 tú"),
    "c4_lead": _N("0 .5 58 Muy; .5 .5 60 bien; 1 .5 61 tam; 1.5 .5 63 bién; 2 .5 65 gra; 2.5 .5 63 cias"),
    "c4_echo": _N("3 .5 63 gra; 3.5 .5 60 cias"),
    "c5_lead": _N("0 .5 60 You; .5 .5 60 are; 1 .5 63 not; 1.5 .5 60 what; 2 .5 58 you; 2.5 1.25 56 say"),
    "c6_lead": _N("0 .5 60 You; .5 .5 60 are; 1 .5 63 what; 1.5 .5 65 you; 2 .5 63 re; 2.5 1.25 60 peat"),
    "c7_lead": _N("0 .5 56 Said; .5 .5 60 once; 1.25 .25 60 it's; 1.5 .5 58 a; 2 .5 56 sen; 2.5 1 55 tence"),
    "c8_lead": _N("0 .5 55 Lived; .5 .5 56 dai; 1 .5 58 ly; 1.5 .5 60 a; 2 .5 63 heart; 2.5 .5 60 beat"),
    "c8_echo": _N("3 .25 53 ba; 3.25 .25 53 dum; 3.5 .25 53 ba; 3.75 .25 53 dum"),
}


def _chorus(start_bar, semis=0, final=False):
    P = []
    s = semis
    P.append(dict(id=f"ch{start_bar}_1", bar=start_bar + 0, role="lead", lang="es", text="¡Hola, buenos días!", notes=_tr(CH["c1_lead"], s)))
    P.append(dict(id=f"ch{start_bar}_1e", bar=start_bar + 0, role="crowd", lang="es", text="¡Hola!", notes=_tr(CH["c1_echo"], s)))
    P.append(dict(id=f"ch{start_bar}_2", bar=start_bar + 1, role="lead", lang="es", text="¿Cómo estás?", notes=_tr(CH["c2_lead"], s)))
    P.append(dict(id=f"ch{start_bar}_2e", bar=start_bar + 1, role="crowd", lang="es", text="¿Cómo estás?", notes=_tr(CH["c2_echo"], s)))
    P.append(dict(id=f"ch{start_bar}_3", bar=start_bar + 2, role="lead", lang="es", text="Bien, gracias, ¿y tú?", notes=_tr(CH["c3_lead"], s)))
    if final:  # the narrator turns the question on himself
        P.append(dict(id=f"ch{start_bar}_3y", bar=start_bar + 2, role="lead", lang="es", text="¿Y yo?",
                      notes=_tr(_N("3 .5 60 Y; 3.5 .5 63 yo"), s)))
    else:
        P.append(dict(id=f"ch{start_bar}_3e", bar=start_bar + 2, role="crowd", lang="es", text="¿Y tú?", notes=_tr(CH["c3_echo"], s)))
    P.append(dict(id=f"ch{start_bar}_4", bar=start_bar + 3, role="lead", lang="es", text="¡Muy bien también, gracias!", notes=_tr(CH["c4_lead"], s)))
    P.append(dict(id=f"ch{start_bar}_4e", bar=start_bar + 3, role="crowd", lang="es", text="¡Gracias!", notes=_tr(CH["c4_echo"], s)))
    if final:
        P.append(dict(id=f"ch{start_bar}_5", bar=start_bar + 4, role="lead", lang="en", harmony=True, text="I am not what I say",
                      notes=_tr(_N("0 .5 60 I; .5 .5 60 am; 1 .5 63 not; 1.5 .5 60 what; 2 .5 58 I; 2.5 .75 56 say"), s)))
        P.append(dict(id=f"ch{start_bar}_5e", bar=start_bar + 4, role="crowd", lang="en", text="what you say",
                      notes=_tr(_N("3.25 .25 60 what; 3.5 .25 60 you; 3.75 .5 56 say"), s)))
        P.append(dict(id=f"ch{start_bar}_6", bar=start_bar + 5, role="lead", lang="en", harmony=True, text="I am what I repeat",
                      notes=_tr(_N("0 .5 60 I; .5 .5 60 am; 1 .5 63 what; 1.5 .5 65 I; 2 .5 63 re; 2.5 .75 60 peat"), s)))
        P.append(dict(id=f"ch{start_bar}_6e", bar=start_bar + 5, role="crowd", lang="en", text="what you repeat",
                      notes=_tr(_N("3.25 .25 60 what; 3.5 .25 60 you; 3.75 .25 63 re; 4 .5 60 peat"), s)))
    else:
        P.append(dict(id=f"ch{start_bar}_5", bar=start_bar + 4, role="lead", lang="en", harmony=True, text="You are not what you say", notes=_tr(CH["c5_lead"], s)))
        P.append(dict(id=f"ch{start_bar}_6", bar=start_bar + 5, role="lead", lang="en", harmony=True, text="You are what you repeat", notes=_tr(CH["c6_lead"], s)))
    P.append(dict(id=f"ch{start_bar}_7", bar=start_bar + 6, role="lead", lang="en", harmony=True, text="Said once, it's a sentence", notes=_tr(CH["c7_lead"], s)))
    P.append(dict(id=f"ch{start_bar}_8", bar=start_bar + 7, role="lead", lang="en", harmony=True, text="Lived daily, a heartbeat", notes=_tr(CH["c8_lead"], s)))
    P.append(dict(id=f"ch{start_bar}_8e", bar=start_bar + 7, role="crowd", lang="en", text="ba dum, ba dum", notes=_tr(CH["c8_echo"], s)))
    return P


def _pre(start_bar, lines):
    melodies = [
        "0.5 .5 56 L1; 1 .5 56 L2; 1.5 .5 58 L3; 2 .5 60 L4; 2.5 1 60 L5; 4 .5 58 L6; 4.5 .5 56 L7; 5 1 58 L8; 6 1.5 56 L9",
        "0.5 .5 58 L1; 1 .5 58 L2; 1.5 .5 60 L3; 2 .5 61 L4; 2.5 1 61 L5; 4 1 60 L6; 5 1 58 L7; 6 1.5 55 L8",
        "0.5 .5 60 L1; 1 .5 60 L2; 1.5 .5 61 L3; 2 1 63 L4; 3 .5 63 L5; 3.5 .5 65 L6; 4 .5 63 L7; 4.5 2.5 60 L8",
        None,  # last line given explicitly per pre-chorus
    ]
    P = []
    for i, (text, syls, explicit) in enumerate(lines):
        if explicit:
            notes = _N(explicit)
        else:
            base = _N(melodies[i])
            assert len(base) == len(syls), (text, len(base), len(syls))
            notes = [(b, d, m, syls[k]) for k, (b, d, m, _) in enumerate(base)]
        P.append(dict(id=f"pre{start_bar}_{i + 1}", bar=start_bar + 2 * i, role="lead", lang="en", text=text, notes=notes))
    return P


SUNG = []
# Pre-chorus 1 (bars 24-31)
SUNG += _pre(24, [
    ("It's not an escape, it's a mirror", ["It's", "not", "an", "es", "cape", "it's", "a", "mir", "ror"], None),
    ("Flattering till I stop posing", ["Flat", "ter", "ing", "till", "I", "stop", "pos", "ing"], None),
    ("It never asks what I believe", ["It", "ne", "ver", "asks", "what", "I", "be", "lieve"], None),
    ("It asks me what I repeat", None, "0.5 .5 63 It; 1 .5 65 asks; 1.5 .5 63 me; 2 .5 65 what; 2.5 .5 67 I; 3 1 65 re; 4 3 64 peat"),
])
SUNG += _chorus(32)
# Post-chorus 1 (bars 40-43); bars 44-47 instrumental
POST = [
    ("¡Qué bueno!", "es", "0 .5 63 Qué; .5 .5 60 bue; 1 1 56 no"),
    ("Otra vez", "es", "0 .5 60 O; .5 .5 63 tra; 1 1.5 65 vez"),
    ("¡Qué bueno!", "es", "0 .5 63 Qué; .5 .5 60 bue; 1 1 56 no"),
    ("Mañana, otra vez", "es", "0 .5 60 Ma; .5 .5 58 ña; 1 .5 56 na; 1.5 .5 60 o; 2 .5 63 tra; 2.5 1.5 65 vez"),
]
for start in (40, 80):
    for i, (text, lang, spec) in enumerate(POST):
        SUNG.append(dict(id=f"post{start}_{i + 1}", bar=start + i, role="lead", lang=lang, text=text, notes=_N(spec)))
    for i in (0, 2):  # (ba-dum, ba-dum) answer after ¡Qué bueno!
        SUNG.append(dict(id=f"post{start}_{i + 1}e", bar=start + i, role="crowd", lang="en", text="ba dum, ba dum",
                         notes=_N("2.5 .25 53 ba; 2.75 .25 53 dum; 3 .25 53 ba; 3.25 .25 53 dum")))
# Pre-chorus 2 (bars 64-71) ends in a stop
SUNG += _pre(64, [
    ("Insight's a flashlight, not a repair", ["In", "sight's", "a", "flash", "light", "not", "a", "re", "pair"], None),
    ("It shows the leak, won't fix the pipe", ["It", "shows", "the", "leak", "won't", "fix", "the", "pipe"], None),
    ("The sentence isn't I see it", ["The", "sen", "tence", "is", "n't", "I", "see", "it"], None),
    ("The sentence is I am in it", None, "0.5 .5 63 The; 1 .5 65 sen; 1.5 .5 63 tence; 2 .5 65 is; 2.5 .5 67 I; 3 .5 65 am; 3.5 .5 67 in; 4 2.5 64 it"),
])
SUNG += _chorus(72)
# Bridge (bars 92-107), half-time, tender
BRIDGE = [
    (92, "Griff, you got here first", None, "0 1 56 Griff; 1 .5 58 you; 1.5 .5 60 got; 2 1 61 here; 3 3 60 first"),
    (94, "You said I'd like it. I called it the worst", None,
     "0 .5 60 You; .5 .5 60 said; 1 .5 63 I'd; 1.5 .5 60 like; 2 1 56 it; 3 .5 56 I; 3.5 .5 60 called; 4 .5 58 it; 4.5 .5 56 the; 5 2 55 worst"),
    (96, "I came back slowly, not right away", None,
     "0 .5 55 I; .5 .5 56 came; 1 .5 58 back; 1.5 1 60 slow; 2.5 .5 58 ly; 3 .5 56 not; 3.5 .5 58 right; 4 .5 60 a; 4.5 2.5 63 way"),
    (98, "You never claimed it, you lived it every day", None,
     "0 .5 60 You; .5 .5 60 ne; 1 .5 61 ver; 1.5 1 63 claimed; 2.5 .5 60 it; 3 .5 60 you; 3.5 .5 61 lived; 4 .5 60 it; 4.5 .5 58 ev; 5 .5 56 ery; 5.5 2 56 day"),
    (100, "You brought me to Vallarta", "You brought me to Vayarta",
     "0 .5 56 You; .5 .5 60 brought; 1 .5 63 me; 1.5 .5 61 to; 2 .5 60 Va; 2.5 1 63 yar; 3.5 2.5 60 ta"),
    (102, "And the diva in the meat dress too", None,
     "0 .5 60 And; .5 .5 60 the; 1 .5 63 di; 1.5 .5 60 va; 2 .5 58 in; 2.5 .5 56 the; 3 .5 58 meat; 3.5 .5 60 dress; 4 2.5 56 too"),
    (104, "After I heard her first", None, "0 .5 55 Af; .5 .5 56 ter; 1 .5 58 I; 1.5 1 60 heard; 2.5 .5 58 her; 3 3 55 first"),
]
for bar, text, tts, spec in BRIDGE:
    d = dict(id=f"bridge{bar}", bar=bar, role="lead", lang="en", text=text, notes=_N(spec))
    if tts:
        d["tts_text"] = tts
    SUNG.append(d)
# Build (bars 116-119): chant "Repeat (repeat)" over the pivot to G minor
for i, bar in enumerate((116, 117)):
    SUNG.append(dict(id=f"build{bar}", bar=bar, role="lead", lang="en", text="Repeat", notes=_N("0 .5 62 re; .5 1.5 65 peat")))
    SUNG.append(dict(id=f"build{bar}e", bar=bar, role="crowd", lang="en", text="repeat", notes=_N("2 .5 62 re; 2.5 1.5 65 peat")))
SUNG.append(dict(id="build118", bar=118, role="lead", lang="en", text="Repeat, repeat", notes=_N("0 .5 62 re; .5 .5 66 peat; 1 .5 62 re; 1.5 1.5 69 peat")))
# Final chorus (bars 120-127) up a whole step
SUNG += _chorus(120, semis=2, final=True)
# Tag (bars 128-131), G minor
SUNG.append(dict(id="tag1e", bar=128, role="crowd", lang="es", text="Otra vez", notes=_N("0 .5 62 O; .5 .5 65 tra; 1 1 67 vez")))
SUNG.append(dict(id="tag1", bar=128, role="lead", lang="en", harmony=True, text="I am what I repeat",
                 notes=_N("2 .5 62 I; 2.5 .5 62 am; 3 .5 65 what; 3.5 .5 67 I; 4 .5 65 re; 4.5 2.5 62 peat")))
SUNG.append(dict(id="tag2e", bar=130, role="crowd", lang="es", text="Otra vez", notes=_N("0 .5 62 O; .5 .5 65 tra; 1 1 67 vez")))
SUNG.append(dict(id="tag2", bar=130, role="lead", lang="en", harmony=True, text="Lived daily, a heartbeat",
                 notes=_N("2 .5 57 Lived; 2.5 .5 58 dai; 3 .5 60 ly; 3.5 .5 62 a; 4 .5 65 heart; 4.5 1 62 beat")))
SUNG.append(dict(id="tag2e2", bar=130, role="crowd", lang="es", text="¡Otra vez!", notes=_N("5.5 .5 62 O; 6 .5 65 tra; 6.5 1.5 67 vez")))

# ------------------------------------------------------------------ spoken / chanted / whispered lines
# style: spoken (deadpan talk-over) | chant (rhythmic, doubled by gang) | whisper | aside (quiet, panned)
# max_beats: the line must finish inside this many beats (speed it up if needed).
SPOKEN = []


def _S(bar, beat, text, style="spoken", role="lead", lang="en", max_beats=7.0, tts_text=None, pan=0.0):
    d = dict(id=f"sp{bar}_{beat}", bar=bar, beat=beat, text=text, style=style, role=role, lang=lang, max_beats=max_beats, pan=pan)
    if tts_text:
        d["tts_text"] = tts_text
    SPOKEN.append(d)


# Verse 1 (bars 8-23): one line per 2 bars
for i, line in enumerate([
    "Every night the bay takes a bow",
    "Like a soprano milking applause",
    "The light flatters linen, and me, for now",
    "Soft on the faces, soft on the flaws",
    "Same pattern in different sandals",
    "Dog-rescue tote, corruption theory too",
    "Ya soy local, said too soon",
    "Toddler in a parked car going vroom",
]):
    _S(8 + 2 * i, 0.0, line)
# Verse 2 (bars 48-63)
_S(48, 0.0, "I cared in January", max_beats=3.75)
_S(49, 0.0, "I meant to help in March", max_beats=3.75)
_S(50, 0.0, "I'd follow up in June", max_beats=3.75)
_S(51, 0.0, "Still involved by October", max_beats=3.75)
_S(52, 0.0, "The group chat's very concerned", max_beats=5.5)
_S(53, 2.0, "Following!", style="chant", role="gang", max_beats=1.75)
_S(54, 0.0, "Someone sweeps before the door opens")
_S(56, 0.0, "And I called it atmosphere")
_S(58, 0.0, "I'm a builder, not a spectator", max_beats=5.5)
_S(59, 2.5, "that's my line", style="aside", max_beats=1.5, pan=0.45)
_S(60, 0.0, "I used to build. Now I describe it.", max_beats=6.0)
_S(61, 3.0, "Fine.", style="aside", max_beats=1.0, pan=-0.45)
# Verse 3 (bars 84-91): chanted, doubled
_S(84, 0.0, "Call ahead", style="chant", max_beats=2.0)
_S(85, 0.0, "Ask clearly", style="chant", max_beats=2.0)
_S(86, 0.0, "Answer honestly", style="chant", max_beats=3.0)
_S(88, 0.0, "Belonging's not a mood", style="chant", max_beats=3.5)
_S(89, 0.0, "it's a loop", style="chant", role="gang", max_beats=2.0)
_S(90, 0.0, "No está caro", style="chant", lang="es", max_beats=2.25)
_S(90, 2.5, "está económico", style="spoken", role="locals", lang="es", max_beats=1.5)
_S(91, 1.0, "Eso dicen", style="spoken", role="locals", lang="es", max_beats=2.0)
# Bridge aside under the held "way"
_S(97, 1.0, "That would have been honest", style="aside", max_beats=2.75, pan=-0.35)
# Breakdown (bars 108-115): whispered over heartbeat
_S(108, 0.0, "It's also, inconveniently, about me", style="whisper", max_beats=7.0)
_S(110, 0.0, "The mirror isn't cruel", style="whisper", max_beats=6.0)
_S(112, 0.0, "It's worse than cruel", style="whisper", max_beats=6.0)
_S(114, 0.0, "It's accurate", style="whisper", max_beats=6.0)
# Build: whispered heartbeat count-in
_S(119, 0.0, "ba dum, ba dum", style="whisper", max_beats=2.5)
# Outro (bars 132-139)
_S(132, 0.0, "Que tengas un buen día", style="whisper", lang="es", max_beats=6.0)
_S(134, 0.0, "Not with the claim", style="whisper", max_beats=4.0)
_S(135, 2.0, "With the next repetition", style="whisper", max_beats=5.5)
_S(137, 2.0, "hola, buenos días", style="whisper", role="locals", lang="es", max_beats=4.0)
_S(139, 0.0, "ba dum... ba dum...", style="whisper", max_beats=4.0)

# ------------------------------------------------------------------ instrumental ad-lib cues (beatbox / heartbeat answers)
# (bar, beat, kind)  kind: boom-tss-ka-tss | ka-tss | boom-tss | lubdub2 (two heartbeats) | kroll
ADLIBS = [
    (41, 2.5, "boom-tss-ka-tss"), (43, 3.0, "boom-tss-ka-tss"),
    (81, 2.5, "boom-tss-ka-tss"), (83, 3.0, "boom-tss-ka-tss"),
    (84, 2.5, "ka-tss"), (85, 2.5, "boom-tss"), (87, 0.0, "lubdub2"), (87, 2.0, "kroll"),
]

if __name__ == "__main__":
    import collections
    print(f"song length {SONG_SECONDS:.1f}s ({int(SONG_SECONDS // 60)}:{SONG_SECONDS % 60:04.1f}); {TOTAL_BARS} bars")
    print(f"{len(SUNG)} sung phrases, {len(SPOKEN)} spoken lines")
    for p in SUNG:
        assert all(len(n) == 4 for n in p["notes"])
    c = collections.Counter(p["role"] for p in SUNG)
    print("roles:", dict(c))
    lo = min(n[2] for p in SUNG for n in p["notes"]); hi = max(n[2] for p in SUNG for n in p["notes"])
    print("melody range MIDI", lo, hi)
