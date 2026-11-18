"""LA LOCA TLACUAQUE DE LA CALLE CERRADA - vocal stems, rendered locally (no Suno).

    cd /home/user/cf336/songs/la-loca-tlacuaque/production
    WHY_LLARTA_WORK=<work folder> python3 vocals.py [--no-asr] [--only ID[,ID...]]

Reads score.py (RAP, ADLIBS, SUNG, SPOKEN, VOICES, t(), t16(), SONG_SECONDS) and writes six dry stereo
float32 stems (44.1 kHz, exactly round(SONG_SECONDS*44100) samples, sample 0 = score time 0, peak <= -1 dBFS,
digital silence wherever the score has no vocal) to <work>/tlac/stems/:
    vox_rap.wav     the rapper's RAP lines (deep raspy lead, grid-locked flow)
    vox_punch.wav   two gang voices doubling every punch=True rhyme, from the rhyme stress to the end of the
                    line (RAP lines with a role other than 'lead' are rapped by the gang into this stem)
    vox_hook.wav    sung hook, gang layer (four men in unison at the written octave: one leads, three behind)
    vox_kids.wav    sung hook, kids layer (VOICES['kids'] an octave up with raised formants)
    vox_adlibs.wav  ad-libs and crowd answers (gang shouts / whispers, and the rapper)
    vox_spoken.wav  SPOKEN lines (the rapper)
    vocals_report.json
Isolated per-phrase renders go to <work>/tlac/vox_debug/<stem>/, plus rap_clean_twin.wav (the rap stem
without the rasp, same time map: a timing reference).
--only renders a subset: the flow is still computed for the whole verse (the chosen lines sit where they
would in the full song) but stems and report go to <work>/tlac/vox_debug/partial/; the full stems are untouched.
A stem with no content in the score is still written (silent).

Everything is driven by score.py data; nothing here knows a line id, a bar or a text.
Method: rap_engine.py (flow, rapper voice, rasp, closed timing loop, punch-ins, hook) and rap_text.py.

Takes are chosen by measurement where the engine offers alternatives (deterministic, cached):
  rap     a line Whisper does not fully understand is rendered again with the decoder at another pitch offset
          (RAP_RETAKES); the clearest take is kept unless its timing is worse.
  hook    the four gang voices (three kids) are balanced by one of a few presets (who leads, how far the
          others sit behind); the preset whose mix the local Whisper model understands best is used.
  ad-libs the phrase must be over inside its window. If it is not at the style's own speed, Kokoro speaks it
          faster (the speeds of SPEED_GRID); a take that is at most MAX_TRIM too long is trimmed (pauses
          first, the stressed vowel last), a longer one gives way to the next speed; among the first
          MAX_TAKES takes that fit, the one Whisper understands best is used. Every squeezed ad-lib is
          listed in the report with the slots it would need at its natural speed.
The choice uses the Whisper model whenever it is installed, with or without --no-asr (that flag only skips
the report's intelligibility checks), so both kinds of run give the same stems.

Report (vocals_report.json): per rap line the flow (syllable, slot, stressed, rhyme), planned slot times,
vowel-onset errors measured on the finished stem by three detectors (start of periodicity = the shared
engine's detector, where the vowel follows an unvoiced consonant or a rest; energy mid-rise in 100-3500 Hz
and in 300-3500 Hz = the audit's definition, wherever the envelope rises 6 dB into the vowel), the same on
the clean twin, the rhyme stress, punch-in against lead, median F0, rasp metrics, ASR on the isolated render
and on the line cut from the stem, raw-TTS recall; the equivalent per hook line, ad-lib and spoken line; and
a summary with every quality target.
Caches: <work>/tlac/cache (neural passes), <work>/tlac/memo (finished renders, ASR). Delete memo to force a
re-render; results are deterministic either way.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, butter, sosfilt, sosfiltfilt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import shared  # noqa: E402
import score  # noqa: E402
import vox_engine as ve  # noqa: E402
import vox_onset as vo  # noqa: E402
import rap_engine as R  # noqa: E402

TLAC = os.path.join(shared.WORK, "tlac")
STEM_DIR = os.path.join(TLAC, "stems")
DEBUG_DIR = os.path.join(TLAC, "vox_debug")
CACHE_DIR = os.path.join(TLAC, "cache")       # neural passes (vox_engine cache)
MEMO_DIR = os.path.join(TLAC, "memo")         # finished renders + ASR results
SR = score.SR
NSAMP = int(round(score.SONG_SECONDS * SR))
STEMS = ("rap", "punch", "hook", "kids", "adlibs", "spoken")
LEAD = score.VOICES["lead"]
GANG = list(score.VOICES["gang"])
KIDS = list(score.VOICES["kids"])

# loudness of one isolated render (dBFS RMS over its active part) and its peak ceiling
LVL = dict(rap=-18.0, punch=-25.0, gang_rap=-22.0, hook_front=-19.5, kids=-24.0, adlib_gang=-22.5,
           adlib_lead=-19.0, spoken=-19.0, whisper=-22.0)
PEAK_CAP = dict(rap=-4.0, punch=-10.0, gang_rap=-8.0, hook_front=-6.0, kids=-9.0, adlib_gang=-8.0,
                adlib_lead=-4.5, spoken=-4.5, whisper=-7.0)
# Hook balance presets: (name, who leads [0 = the rapper's own voice if it is in the gang, else the first
# Spanish voice; 1 = the next Spanish voice], level of the other Spanish voices, level of the English voices),
# dB relative to the leading voice. Tried in this order; the first one with the best Whisper score is used.
GANG_PRESETS = (("lead", 0, -6.0, -10.0), ("second", 1, -6.0, -10.0), ("lead, others far", 0, -12.0, -15.0),
                ("second, others far", 1, -12.0, -15.0))
KIDS_PRESETS = (("trio", 0, -6.0, -9.0), ("front, others far", 0, -12.0, -15.0))
HOOK_BUS_CONTRAST = False      # optional: one common attack shaping for the unison on top of each voice's own.
                               # Measured on the final lyric: no gain (gang 96.5 -> 95.9 % of mix onsets within
                               # +-20 ms; the mix's late readings come from the voices' phases, not from missing dips)
KIDS_OCTAVE = 1
KIDS_FORMANTS = (1.12, 1.16, 1.20)     # the leading (Spanish) kid has the smallest shift: clearest vowels
HOOK_END_GUARD_S = 0.003       # every hook voice is silent this long before the written end of its last note
HOOK_END_FADE_S = 0.015
HOOK_MAX_OFFSET_MS = 8.0       # timing spread of the gang (the leading voice at 0)
KIDS_MAX_OFFSET_MS = 4.0
PUNCH_OFFSETS_MS = (0.0, 4.0)
TOL_MS = 20.0                  # rap / hook vowel onsets
RHYME_TOL_MS = 15.0
SPEECH_TOL_MS = 25.0
RAP_RETAKES = (dict(neural_shift=2), dict(neural_shift=4))   # alternative takes of a rap line (RAPPER overrides)
RETAKE_BELOW = 0.95            # a line is taken again if its first take scores below this (take_score recall)
RETAKE_COST_SLACK = 300.0      # a retake may not be worse in timing than the first take by more than this (readings_cost)
SPEED_GRID = (1.0, 1.12, 1.25, 1.35, 1.5, 1.65, 1.8, 2.0)   # Kokoro speaking rates tried for a spoken phrase
MAX_TRIM = 0.12                # share of a phrase's frames that may be trimmed before a faster rate is preferred
MAX_TAKES = 3                  # fitting speeds compared by Whisper
END_BEFORE_DEADLINE_S = 0.012     # a phrase is 40 dB down this long before its deadline
QUIET_BEFORE_DEADLINE_S = 0.0     # optional: ... and 20 dB down this long before it. Measured with 0.060: the
                                  # 2- and 3-slot shouts then have to lose a third of their frames ("¡Cuatro!"
                                  # 11 -> 7) and Whisper recall falls to zero; such endings are flagged instead
END_REL_DB = -50.0             # "ended" = 50 dB below the phrase's peak
SHOUT = dict(speed=1.12, f0_scale=1.18, f0_range=1.25)
LEAD_SHOUT = dict(speed=1.05, f0_scale=1.12, f0_range=1.15)
STEM_CEILING_DB = -1.2         # peak of every stem; ad-lib phrases are held under it one by one


# ----------------------------------------------------------------------------- small DSP helpers
def to44(x):
    return resample_poly(np.asarray(x, np.float64), 147, 80)


def active_rms_db(x, sr=ve.KSR):
    hop = int(0.02 * sr)
    n = len(x) // hop
    if n == 0:
        return -120.0
    fr = np.sqrt(np.mean(x[:n * hop].reshape(n, hop) ** 2, axis=1) + 1e-20)
    thr = fr.max() * 10 ** (-30 / 20)
    act = fr[fr > thr]
    return 20 * np.log10(np.sqrt(np.mean(act ** 2)) + 1e-12)


def limit(x, ceiling_db, sr=ve.KSR):
    """Transparent peak limiter (5 ms look-ahead, ~60 ms release)."""
    from scipy.ndimage import maximum_filter1d, minimum_filter1d
    c = 10 ** (ceiling_db / 20)
    a = np.abs(x)
    if a.max() <= c:
        return x
    w = int(0.005 * sr)
    pk = maximum_filter1d(a, size=2 * w + 1)
    g = np.minimum(1.0, c / (pk + 1e-12))
    g = minimum_filter1d(g, size=2 * w + 1)
    k = int(0.004 * sr)
    g = np.convolve(g, np.ones(k) / k, mode="same")
    rel = np.exp(-1.0 / (0.06 * sr))
    out = g.copy()
    for i in range(1, len(out)):
        if out[i] > out[i - 1]:
            out[i] = out[i - 1] * rel + out[i] * (1 - rel)
    return np.clip(x * out, -c, c)


_SOS_LOWCUT = butter(2, 55.0, "highpass", fs=ve.KSR, output="sos")


def low_cut(x):
    """55 Hz high-pass (the lowest voice here is 80 Hz): pulse-train synthesis and the gains that ride on it
    leave thumps below 30 Hz at voice onsets. Causal, so nothing sounds before the render starts; whatever
    rings after its last sample is faded out over 2 ms, so the render stays inside its own span."""
    x = np.asarray(x, np.float64)
    nz = np.nonzero(x)[0]
    if len(nz) == 0:
        return x
    y = sosfilt(_SOS_LOWCUT, x)
    y[:nz[0]] = 0.0
    n = min(48, nz[-1] - nz[0])
    if n > 0:
        y[nz[-1] + 1 - n:nz[-1] + 1] *= np.linspace(1.0, 0.0, n)
    y[nz[-1] + 1:] = 0.0
    return y


def norm_to(x, kind, sr=ve.KSR, trim=0.0):
    y = low_cut(x)
    y = y * 10 ** ((LVL[kind] + trim - active_rms_db(y, sr)) / 20)
    return limit(y, PEAK_CAP[kind] + trim, sr)


def pan_gains(p):
    a = (np.clip(p, -1, 1) + 1) * np.pi / 4
    return np.cos(a) * np.sqrt(2), np.sin(a) * np.sqrt(2)   # unity at centre, constant power


def env_frames(x, sr, hop_s=0.003):
    hop = max(1, int(hop_s * sr))
    n = len(x) // hop
    return np.sqrt(np.mean(x[:n * hop].reshape(n, hop) ** 2, axis=1) + 1e-20), hop


def envelope_onset_end(x, sr, rel_db=-32.0, hop_s=0.003):
    fr, hop = env_frames(x, sr, hop_s)
    thr = fr.max() * 10 ** (rel_db / 20)
    idx = np.nonzero(fr > thr)[0]
    if len(idx) == 0:
        return 0, len(x)
    return int(idx[0] * hop), int((idx[-1] + 1) * hop)


def bright(x, sr=ve.KSR, db=4.0, fc=3200.0):
    """High shelf for the kids layer (the filter's ringing after the last sample of the render is cut, so
    the render stays inside its own span)."""
    x = np.asarray(x, np.float64)
    hi = sosfilt(butter(1, fc, "highpass", fs=sr, output="sos"), x)
    y = x + (10 ** (db / 20) - 1.0) * hi
    nz = np.nonzero(x)[0]
    if len(nz):
        y[:nz[0]] = 0.0
        y[nz[-1] + 1:] = 0.0
    return y


def fade_in_at_first_sound(x, sr=ve.KSR, ms=6.0):
    """Raised-cosine fade-in that starts exactly on the first non-zero sample (no step out of silence)."""
    x = np.array(x, np.float64)
    nz = np.nonzero(x)[0]
    if len(nz):
        n = int(ms / 1000.0 * sr)
        seg = np.arange(nz[0], min(len(x), nz[0] + n))
        x[seg] *= 0.5 - 0.5 * np.cos(np.pi * (seg - nz[0] + 0.5) / n)
    return x


def start_level_db(x):
    """Level reached in the first 0.5 ms after the first non-zero sample, dB re the peak (a hard step out of
    digital silence reads about -15 dB or more; a faded start -35 dB or less)."""
    nz = np.nonzero(x)[0]
    if len(nz) == 0:
        return None
    m = max(1, int(0.0005 * SR))
    return float(20 * np.log10(np.max(np.abs(x[nz[0]:nz[0] + m])) / (np.max(np.abs(x)) + 1e-12) + 1e-12))


def r1(v):
    return None if v is None else round(float(v), 1)


def context_of(c):
    """Class of the sound right before a vowel (cons_before): how hard its onset is to place and to measure."""
    if c is None:
        return "vowel-initial / hiatus"
    if c in "jw":
        return "after glide"
    if c in ve.UNV_OBS:
        return "after unvoiced"
    if c == "ð":
        return "after ð"
    return "after voiced consonant"


def compact(rd):
    """Per-syllable detector readings -> lists (ill-defined mid-rise readings are dropped, their flags kept)."""
    out = dict(per=[q["per"] for q in rd])
    for b in R.MR_BANDS:
        out[b] = [None if q[b + "_flag"] else q[b] for q in rd]
    out["flags"] = [" ".join(f"{b}:{q[b + '_flag']}" for b in R.MR_BANDS if q[b + "_flag"]) for q in rd]
    return out


def energy_of(q):
    """The energy mid-rise reading of one syllable: wide band, else formant band (well-defined ones only)."""
    for b in R.MR_BANDS:
        if q[b] is not None and not q[b + "_flag"]:
            return q[b]
    return None


# ----------------------------------------------------------------------------- the renderer
class Vocals:
    def __init__(self, asr=True, only=None):
        self.E = R.RapEngine(cache_dir=CACHE_DIR, memo_dir=MEMO_DIR)
        self.cfg = dict(R.RAPPER, voice=LEAD)
        self.stems = {k: np.zeros((NSAMP, 2), np.float64) for k in STEMS}
        self.twin = np.zeros(NSAMP, np.float64)         # the rap stem without the rasp (timing reference only)
        self.regions = {k: [] for k in STEMS}          # (start_s, end_s, id) where the score puts sound
        self.steps = []                                 # level 0.5 ms into every placed render (dB re its peak)
        self.report = dict(rap=[], hook=[], adlibs=[], spoken=[], stems={}, notes=[])
        self.only = set(only) if only else None
        self.times = {}
        import vox_asr
        self.vox_asr = vox_asr
        self.have_asr = bool(vox_asr.available())      # take selection (hook balance, ad-lib speed)
        self.asr = bool(asr and self.have_asr)         # the report's intelligibility checks
        if not self.have_asr:
            print("ASR model missing: no intelligibility checks, takes are chosen by rule")
            self.report["notes"].append("ASR model missing: takes chosen by rule (first preset / slowest speed that fits)")
        for d in STEMS:
            os.makedirs(os.path.join(DEBUG_DIR, d), exist_ok=True)

    # --- plumbing
    def want(self, i):
        return self.only is None or i in self.only

    def dbg(self, sub, name, x, sr=ve.KSR):
        sf.write(os.path.join(DEBUG_DIR, sub, name + ".wav"), np.asarray(x, np.float32), sr, subtype="FLOAT")

    def place(self, stem, x24, start_s, pan, ident):
        """Add a mono 24 kHz render to a stem with its sample 0 at song time start_s."""
        x44 = to44(R.edge_fade(x24))
        i0 = int(round(start_s * SR))
        a, b = max(0, i0), min(NSAMP, i0 + len(x44))
        if b <= a:
            self.report["notes"].append(f"{ident}: render lies outside the song ({start_s:.2f} s)")
            return
        if a > i0 or b < i0 + len(x44):
            cut = x44[a - i0:b - i0].copy()
            n = min(len(cut) // 2, int(0.003 * SR))
            if n and a > i0:
                cut[:n] *= np.linspace(0, 1, n)
            if n and b < i0 + len(x44):
                cut[-n:] *= np.linspace(1, 0, n)
        else:
            cut = x44
        gl, gr = pan_gains(pan)
        self.stems[stem][a:b, 0] += gl * cut
        self.stems[stem][a:b, 1] += gr * cut
        # does the render start with a step? Its level in the first 0.5 ms after it leaves silence (the
        # resampler rings below -70 dB before the first sound, which is not counted), dB re its peak
        pk = float(np.max(np.abs(cut))) + 1e-12
        loud = np.nonzero(np.abs(cut) > pk * 10 ** (-70 / 20))[0]
        if len(loud):
            m = max(1, int(0.0005 * SR))
            self.steps.append((float(20 * np.log10(np.max(np.abs(cut[loud[0]:loud[0] + m])) / pk + 1e-12)), ident))

    def place_twin(self, x24, start_s):
        x44 = to44(R.edge_fade(x24))
        i0 = int(round(start_s * SR))
        a, b = max(0, i0), min(NSAMP, i0 + len(x44))
        if b > a:
            self.twin[a:b] += x44[a - i0:b - i0]

    def transcribe(self, x, lang, sr=ve.KSR):
        x = np.asarray(x, np.float32)

        def run():
            return np.zeros(1), dict(hyp=self.vox_asr.transcribe(x, sr, lang))
        _, m = self.E.memo("asr", hashlib.sha1(x.tobytes()).hexdigest() + f"|{sr}|{lang}", run)
        return m["hyp"]

    def asr_check(self, x, lang, ref, sr=ve.KSR):
        if not self.asr:
            return None
        hyp = self.transcribe(x, lang, sr)
        return dict(hyp=hyp, **R.asr_scores(ref, hyp, lang))

    def asr_raw(self, text, lang, voice, ref):
        """The same words as plain Kokoro speech (same Mexican phonemes, natural rhythm, speed 1.0)."""
        if not self.asr:
            return None
        ph = self.E.phonemes(text, lang)
        v = voice.split("|")[0]

        def run():
            a, _ = self.E.k.create(ph, v, speed=1.0, is_phonemes=True)
            return np.zeros(1), dict(hyp=self.vox_asr.transcribe(a, ve.KSR, lang))
        _, m = self.E.memo("rawtts", dict(ph=ph, voice=v, lang=lang), run)
        return dict(hyp=m["hyp"], **R.asr_scores(ref, m["hyp"], lang))

    def take_score(self, x24, lang, ref, span=None):
        """How well Whisper understands a candidate take: (recall, char_sim), the mean of two listening
        conditions: the whole 24 kHz render as it is, and the line as an outside audit cuts it from a
        44.1 kHz stem (`span` = (start, end) of that cut in seconds from the render's first sample, 8 ms
        fades; without a span: 0.2 s of silence before and 0.1 s after).
        One transcription of a short sung or shouted phrase is a coin toss near the edge; two are steadier."""
        if not self.have_asr:
            return None
        x24 = np.asarray(x24, np.float64)
        if span is None:
            v2 = np.concatenate([np.zeros(int(0.2 * ve.KSR)), x24, np.zeros(int(0.1 * ve.KSR))])
        else:
            a, b = int(round(span[0] * ve.KSR)), int(round(span[1] * ve.KSR))
            v2 = np.concatenate([np.zeros(max(0, -a)), x24[max(0, a):max(0, b)], np.zeros(max(0, b - len(x24)))])
            n = min(len(v2) // 2, int(0.008 * ve.KSR))
            if n:
                v2[:n] *= np.linspace(0, 1, n)
                v2[-n:] *= np.linspace(1, 0, n)
        s1 = R.asr_scores(ref, self.transcribe(x24, lang), lang)
        s2 = R.asr_scores(ref, self.transcribe(to44(v2), lang, SR), lang)
        return (round(0.5 * (s1["recall"] + s2["recall"]), 3), round(0.5 * (s1["char_sim"] + s2["char_sim"]), 3))

    @staticmethod
    def best_take(scores):
        """Index of the best (recall, char_sim); char_sim only counts in steps of 0.1; ties: the first."""
        if not scores or scores[0] is None:
            return 0
        return max(range(len(scores)), key=lambda i: (scores[i][0], round(scores[i][1], 1), -i))

    # ------------------------------------------------------------------------- rap
    def gang_males(self):
        m = [v for v in GANG if v[1] == "m"]
        return m if len(m) >= 2 else list(GANG)

    def punch_voices(self, idx):
        cand = [v for v in self.gang_males() if v != LEAD] or self.gang_males()
        es = [v for v in cand if v[0] == "e"]
        en = [v for v in cand if v[0] != "e"]
        if es and en:
            return [es[idx % len(es)], en[idx % len(en)]]
        return [cand[idx % len(cand)], cand[(idx + 1) % len(cand)]] if len(cand) > 1 else cand[:1]

    def rap_entry(self, ln, f, plan, stem):
        E = self.E
        n = len(plan["sylls"])
        return dict(id=ln["id"], stem=stem, bar=ln["bar"], role=ln["role"], text=ln["text"], tts_text=ln.get("tts_text"),
                    phonemes=plan["ph"], anchor=ln["anchor"], lo=ln["lo"], hi=ln["hi"], n_syllables=len(f["T"]),
                    flow=R.flow_table(f), crowded=bool(f["crowded"]), thirty_second_pairs=f["pairs"],
                    flow_log=f["log"], first_slot=f["pos"][0], slot_times_s=[round(t, 4) for t in f["T"]],
                    note_durs_s=[round(d, 4) for d in f["D"]], target_midi=[round(m, 2) for m in plan["midi"]],
                    measurable=[bool(E.measurable(plan, k)) for k in range(n)],
                    cons_before=[E.cons_before(plan, k) for k in range(n)],
                    context=[context_of(E.cons_before(plan, k)) for k in range(n)],
                    rhyme_syllable=int(f["rhyme"]), planned_end_s=round(f["end_s"], 4),
                    deadline_s=round(f["deadline_s"], 4))

    def render_rap(self, idx, ln, f):
        E = self.E
        seed = R.seed_of(ln["id"])
        lead = ln["role"] == "lead"
        take = None
        if lead:
            r = E.render_rap(f, self.cfg, seed=seed)
            if self.have_asr and RAP_RETAKES:
                # a line Whisper does not fully understand is taken again with the decoder speaking at another
                # pitch offset (a different articulation of the same plan); the clearest take is kept, unless
                # its timing is worse
                def cost_of(q):
                    return float(q["meta"]["loop"]["costs"][q["meta"]["loop"]["kept"]])
                def span_of(q):
                    return (f["T"][0] - 0.22 - q["R0"], f["deadline_s"] + 0.02 - q["R0"])
                sc = self.take_score(r["audio"], ln["lang"], ln["text"], span_of(r))
                tried = [dict(take="first", recall=sc[0], char_sim=sc[1], timing_cost=round(cost_of(r), 1))]
                best, used = (sc, r), "first"
                if sc[0] < RETAKE_BELOW:
                    for alt in RAP_RETAKES:
                        r2 = E.render_rap(f, dict(self.cfg, **alt), seed=seed)
                        s2 = self.take_score(r2["audio"], ln["lang"], ln["text"], span_of(r2))
                        name = ", ".join(f"{k_}={v_}" for k_, v_ in alt.items())
                        tried.append(dict(take=name, recall=s2[0], char_sim=s2[1], timing_cost=round(cost_of(r2), 1)))
                        if (cost_of(r2) <= cost_of(r) + RETAKE_COST_SLACK
                                and (s2[0], round(s2[1], 1)) > (best[0][0], round(best[0][1], 1))):
                            best, used = (s2, r2), name
                r = best[1]
                take = dict(used=used, tried=tried)
            x = norm_to(r["audio"], "rap")
            self.dbg("rap", ln["id"], x)
            self.place("rap", x, r["R0"], 0.0, ln["id"])
            clean = None
            if r.get("clean") is not None:          # same time map without the rasp: the timing reference
                clean = norm_to(r["clean"], "rap")
                self.place_twin(clean, r["R0"])
                self.dbg("rap", ln["id"] + "_clean", clean)
            stem, mix = "rap", x
            voices = [LEAD]
        else:
            # a RAP line given to the gang: two gang voices rap it on the grid (no rasp), into vox_punch
            voices = self.punch_voices(idx)
            mix, clean = None, None
            for j, v in enumerate(voices):
                r = E.render_punch(f, v, self.cfg, seed=seed + j, detune=(-6.0, 6.0)[j % 2], first=0)
                x = norm_to(r["audio"], "gang_rap")
                self.dbg("punch", f"{ln['id']}_{v}", x)
                self.place("punch", x, r["R0"] + 0.004 * j, (-0.3, 0.3)[j % 2], ln["id"])
                mix = x if mix is None else mix[:len(x)] + x[:len(mix)]
            stem = "punch"
        plan = r["plan"]
        e = self.rap_entry(ln, f, plan, f"vox_{stem}")
        e["voices"] = voices
        e["take"] = take
        # what the timing loop saw on the clean render it decided on (the stem is measured again in write())
        e["loop"] = dict(r["meta"]["loop"], readings=compact(r["meta"]["readings"]),
                         moved_ms=[t["second_pass_ms"] for t in r["meta"]["timing"]],
                         decoder_ms=[t["measured_src_ms"] for t in r["meta"]["timing"]])
        e["render_onset_err_ms"] = r["meta"]["verify_per_ms"]
        e["consonant_ducked_db"] = r["meta"]["duck_db"]
        # audible start and end of the isolated render (-40 dB re its peak): deadline and overlap checks
        on, end = envelope_onset_end(mix, ve.KSR, rel_db=-40.0)
        e["first_sound_s"] = round(r["R0"] + on / ve.KSR, 4)
        e["end_s"] = round(r["R0"] + end / ve.KSR, 4)
        e["deadline_margin_ms"] = r1(1000 * (f["deadline_s"] - e["end_s"]))
        mt = r["meta"]["rasp"] if (lead and r["meta"].get("rasp")) else R.rasp_metrics(mix)
        e["f0_median_hz"] = mt["f0_med"]
        e["f0_target_median_hz"] = r1(r["meta"]["f0_target_med"])
        e["rasp"] = {k: mt[k] for k in ("hnr_db", "periodicity", "shr_db", "nhr_1_4k_db")}
        if clean is not None:
            mc = R.rasp_metrics(clean)
            e["clean_twin"] = dict(rasp={k: mc[k] for k in ("hnr_db", "periodicity", "shr_db", "nhr_1_4k_db")},
                                   f0_median_hz=mc["f0_med"], asr=self.asr_check(clean, ln["lang"], ln["text"]))
        e["asr"] = self.asr_check(mix, ln["lang"], ln["text"])
        e["asr_raw_tts"] = self.asr_raw(f["tts_text"], ln["lang"], LEAD, ln["text"])
        self.regions[stem].append((min(E.sound_start(plan), e["first_sound_s"], f["T"][0] - R.LINE_PRE_MAX_S) - 0.006,
                                   f["deadline_s"] + 0.006, ln["id"]))
        if lead and ln.get("punch"):
            pv = self.punch_voices(idx)
            k = int(f["rhyme"])
            pe = dict(voices=pv, from_syllable=k, syllables=f["strings"][k:], lead_in_syllables=None, renders=[])
            for j, v in enumerate(pv):
                rp = E.render_punch(f, v, self.cfg, seed=seed + 11 + j, detune=(-7.0, 7.0)[j % 2])
                xp = norm_to(rp["audio"], "punch")
                off = PUNCH_OFFSETS_MS[j % 2] / 1000.0
                pan = (-0.45, 0.45)[j % 2]
                self.dbg("punch", f"{ln['id']}_{v}", xp)
                self.place("punch", xp, rp["R0"] + off, pan, ln["id"] + "_punch")
                on, pend = envelope_onset_end(xp, ve.KSR, rel_db=-40.0)
                rd = rp["meta"]["readings"][rp["rhyme_index"]:]
                pe["lead_in_syllables"] = int(rp["rhyme_index"])
                pe["renders"].append(dict(voice=v, pan=pan, offset_ms=PUNCH_OFFSETS_MS[j % 2], detune_cents=(-7.0, 7.0)[j % 2],
                                          onset_err_ms=[q["per"] for q in rd], rise_err_ms=[energy_of(q) for q in rd],
                                          loop_costs=rp["meta"]["loop"]["costs"], consonant_db=rp["meta"].get("punch_consonant_db"),
                                          gate_opens_ms=r1(1000 * (rp["gate_s"] + off - f["T"][k])),
                                          start_s=round(rp["R0"] + off + on / ve.KSR, 4),
                                          end_s=round(rp["R0"] + off + pend / ve.KSR, 4)))
                self.regions["punch"].append((rp["gate_s"] + off - 0.006, f["deadline_s"] + 0.012, ln["id"] + "_punch"))
            e["punch"] = pe
        self.report["rap"].append(e)
        return e

    # ------------------------------------------------------------------------- hook
    def hook_voices(self, layer):
        """(Spanish voices, the others) of a hook layer; the rapper's own voice comes first if it is there."""
        if layer == "gang":
            m = [v for v in GANG if v[1] == "m"]
            f = [v for v in GANG if v[1] != "m"]
            v = (m + f)[:4] if len(m) >= 3 else (m + f)[:max(3, min(4, len(m) + len(f)))]
        else:
            v = list(KIDS)
        es = [x for x in v if x[0] == "e"]
        en = [x for x in v if x[0] != "e"]
        if not es:
            es, en = en[:1], en[1:]
        ana = es[0]                                    # the voice whose natural speech shapes the duration plan
        if LEAD in es:
            es = [LEAD] + [x for x in es if x != LEAD]
        return es, en, ana

    def hook_entry(self, p, plan, stem, voices):
        E = self.E
        n = len(plan["sylls"])
        notes = [nt for s in plan["sylls"] for nt in s["notes"][:1]]
        return dict(id=p["id"], stem=stem, bar=p["bar"], role=p["role"], text=p["text"], tts_text=p.get("tts_text"),
                    phonemes=plan["ph"], n_notes=len(p["notes"]), n_syllables=n,
                    natural_syllables=plan["natural_syllables"], reconciliation=plan["slog"],
                    syllables=E.syl_strings(plan["chars"], plan["sylls"]), score_syllables=[nt[3] for nt in p["notes"]],
                    voices=voices, note_times_s=[round(nt["T"], 4) for nt in notes],
                    note_durs_s=[round(nt["D"], 4) for nt in notes], note_midi=[nt["m"] for nt in notes],
                    measurable=[bool(E.measurable(plan, k)) for k in range(n)],
                    cons_before=[E.cons_before(plan, k) for k in range(n)],
                    context=[context_of(E.cons_before(plan, k)) for k in range(n)])

    def render_hook(self, p):
        E = self.E
        # seeded by the line's content, not its id: a hook line that comes back later is the same render
        seed = R.seed_of(f"{p.get('tts_text') or p['text']}|{p['role']}|{[(round(b, 4), d, m) for (b, d, m, _) in p['notes']]}")
        last = max(score.t(p["bar"], b + d) for (b, d, _, _) in p["notes"])
        out = []
        layers = []
        if p["role"] in ("gang", "lead"):
            layers.append("gang")
        if p["role"] in ("gang", "kids"):
            layers.append("kids")
        if not layers:
            layers = ["gang"]
        for layer in layers:
            rng = np.random.default_rng(seed + (0 if layer == "gang" else 7))
            if layer == "gang" and p["role"] == "lead":
                es, en, ana = [LEAD], [], LEAD
            else:
                es, en, ana = self.hook_voices(layer)
            voices = es + en
            n = len(voices)
            if layer == "gang":
                octave, stem, kind, presets = 0, "hook", "hook_front", GANG_PRESETS
                fsh = [1.0] * n
                dets = [(-3.0, 3.0)[j % 2] if j < len(es) else (-9.0, 9.0)[j % 2] for j in range(n)]
                offs = [float((0.0, 3.0)[j % 2]) if j < len(es) else float(rng.uniform(4.0, HOOK_MAX_OFFSET_MS)) for j in range(n)]
                side = (-0.5, 0.4, 0.75, -0.75)
            else:
                octave, stem, kind, presets = KIDS_OCTAVE, "kids", "kids", KIDS_PRESETS
                fsh = [KIDS_FORMANTS[j % len(KIDS_FORMANTS)] for j in range(n)]
                dets = [0.0 if j == 0 else (-6.0, 6.0)[j % 2] for j in range(n)]
                offs = [0.0 if j == 0 else float(rng.uniform(1.0, KIDS_MAX_OFFSET_MS)) for j in range(n)]
                side = (-0.5, 0.5, -0.8, 0.8)
            plan = E.hook_plan(score, p, ana)
            e = self.hook_entry(p, plan, f"vox_{stem}", voices)
            e["layer"] = layer
            raw, rs, tcs = [], [], []
            for j, v in enumerate(voices):
                r = E.render_hook(plan, v, seed=seed + j, detune=float(dets[j]), octave=octave, formant_shift=fsh[j])
                raw.append(bright(r["audio"]) if layer == "kids" else np.asarray(r["audio"], np.float64))
                rd = r["meta"]["readings"]
                tcs.append(r["meta"]["tc"])

                def notes(a_=r["audio"], d_=float(dets[j])):
                    return np.zeros(1), R.pitch_check_notes(a_, plan, octave, d_)
                _, nc = E.memo("notepitch", hashlib.sha1(np.asarray(r["audio"], np.float32).tobytes()).hexdigest(), notes)
                rs.append(dict(voice=v, detune_cents=r1(dets[j]), offset_ms=r1(offs[j]), octave=octave, formant_shift=fsh[j],
                               pitch=r["meta"]["pitch"], note_cents=nc,
                               onset_err_ms=[q["per"] for q in rd], rise_err_ms=[energy_of(q) for q in rd],
                               loop_costs=r["meta"]["loop"]["costs"], consonant_ducked_db=r["meta"]["duck_db"]))

            def balance(preset):
                """Per-voice level (dB re the leading voice) and the mono mix of a preset."""
                _, who, es_db, en_db = preset
                front = es[min(who, len(es) - 1)]
                lv = [0.0 if v == front else (es_db if j < len(es) else en_db) for j, v in enumerate(voices)]
                xs = [norm_to(raw[j], kind, trim=lv[j]) for j in range(n)]
                shs = [int(round(offs[j] / 1000.0 * ve.KSR)) for j in range(n)]
                # the voices behind sing a few ms late: all of them are shut by the written end of the line
                n_end = int(round((last - HOOK_END_GUARD_S - plan["R0"]) * ve.KSR))
                nf = int(HOOK_END_FADE_S * ve.KSR)
                for j in range(n):
                    e_ = n_end - shs[j]
                    if 0 < e_ < len(xs[j]):
                        xs[j] = xs[j].copy()
                        seg = np.arange(max(0, e_ - nf), e_)
                        xs[j][seg] *= 0.5 + 0.5 * np.cos(np.pi * (seg - (e_ - nf) + 0.5) / nf)
                        xs[j][e_:] = 0.0
                L = max(len(x) + sh for x, sh in zip(xs, shs))

                def summed():
                    m = np.zeros(L)
                    for x, sh in zip(xs, shs):
                        m[sh:sh + len(x)] += x
                    return m
                mix = summed()
                if n > 1 and HOOK_BUS_CONTRAST:
                    # every voice has its attacks on the notes, but their consonants differ in length, so in
                    # the unison the dips before the vowels fill each other up: give the whole group the
                    # same attack (the engine's onset_contrast on the mix, placed by the leading voice)
                    cg = E.onset_contrast(plan, mix, tcs[voices.index(front)])
                    xs = [x * cg[sh:sh + len(x)] for x, sh in zip(xs, shs)]
                    mix = summed()
                return front, lv, xs, mix
            cands = [pr for pr in presets if pr[1] < len(es)] if n > 1 else presets[:1]
            built = [balance(pr) for pr in cands]
            first = min(score.t(p["bar"], b) for (b, _, _, _) in p["notes"])
            span = (first - 0.2 - plan["R0"], last + 0.1 - plan["R0"])
            scores = [self.take_score(b[3], p["lang"], p["text"], span) for b in built] if len(built) > 1 else [None]
            pick = self.best_take(scores)
            front, lv, xs, mix = built[pick]
            e["balance"] = dict(preset=cands[pick][0], leads=front,
                                tried=[dict(preset=c[0], recall=s and s[0], char_sim=s and s[1]) for c, s in zip(cands, scores)])
            others = [v for v in voices if v != front]
            pan_of = {front: 0.0, **{v: float(side[i % len(side)]) for i, v in enumerate(others)}}
            for j, v in enumerate(voices):
                self.dbg(stem, f"{p['id']}_{v}", xs[j])
                self.place(stem, xs[j], plan["R0"] + offs[j] / 1000.0, pan_of[v], p["id"])
                rs[j].update(pan=round(pan_of[v], 2), level_db=r1(lv[j]))
            e["renders"] = rs
            on, end = envelope_onset_end(mix, ve.KSR, rel_db=-40.0)
            e["first_sound_s"] = round(plan["R0"] + on / ve.KSR, 4)
            e["end_s"] = round(plan["R0"] + end / ve.KSR, 4)
            e["written_end_s"] = round(last, 4)
            e["asr"] = self.asr_check(mix, p["lang"], p["text"])
            if layer == "gang":
                e["asr_raw_tts"] = self.asr_raw(p.get("tts_text") or p["text"], p["lang"], ana, p["text"])
            self.regions[stem].append((min(E.sound_start(plan), e["first_sound_s"], first - R.LINE_PRE_MAX_S) - 0.006, last, p["id"]))
            self.report["hook"].append(e)
            out.append(e)
        return out

    # ------------------------------------------------------------------------- speech path
    def fit_speech(self, text, lang, voices, mode, window, par, seed, ref):
        """A spoken phrase that is over inside `window` seconds (counted from its envelope onset):
        40 dB down END_BEFORE_DEADLINE_S before the end of the window (and, if QUIET_BEFORE_DEADLINE_S is set,
        20 dB down that long before it), so the deadline never cuts a word.
        If the phrase fits at the style's own speed it is used as it is. Otherwise Kokoro speaks it at the
        faster speeds of SPEED_GRID. A speed whose take is too long may be trimmed by at most MAX_TRIM of its
        frames (pauses, then the longest sounds, the stressed vowel last); otherwise the next speed is tried.
        The first MAX_TAKES fitting takes are compared by Whisper (take_score) and the best is used (ties:
        the slowest); whispers, which Whisper cannot read, take the slowest. If not even the fastest speed
        fits, it is trimmed as far as it takes (reported as forced).
        Returns (list of arrays 24 kHz, info)."""
        E = self.E
        kw = dict(cfg=self.cfg, seed=seed, f0_scale=par["f0_scale"], f0_range=par["f0_range"])
        grid = [float(par["speed"])] + [s for s in SPEED_GRID if s > par["speed"] + 1e-6]

        def lens(xs):
            L = min(map(len, xs))
            mix = np.sum([x[:L] for x in xs], axis=0)
            fr, hop = env_frames(mix, ve.KSR)
            pk = fr.max()
            on = np.nonzero(fr > pk * 10 ** (-32 / 20))[0][0]
            e20 = np.nonzero(fr > pk * 10 ** (-20 / 20))[0][-1]
            e40 = np.nonzero(fr > pk * 10 ** (-40 / 20))[0][-1]
            return (e20 + 1 - on) * hop / ve.KSR, (e40 + 1 - on) * hop / ve.KSR, mix

        def excess(l20, l40):
            return max(l20 - (window - QUIET_BEFORE_DEADLINE_S), l40 - (window - END_BEFORE_DEADLINE_S))

        def run():
            takes, attempts, nat = [], [], None
            for s in grid:
                fastest = s >= grid[-1] - 1e-6
                xs, sp = E.speech_once(text, lang, voices, s, mode, **kw)
                l20, l40, mix = lens(xs)
                frames = int(np.sum(sp["dur"][1:-1]))
                if nat is None:
                    nat = dict(speed=s, frames=frames, len_s=round(l40, 3), len_to_minus20_s=round(l20, 3))
                ex = excess(l20, l40)
                attempts.append(dict(speed=s, frames=frames, len_s=round(l40, 3), fits=bool(ex <= 0)))
                trimmed, forced = None, False
                if ex > 0:
                    last_resort = fastest and not takes
                    limit = frames - 2 if last_resort else max(1, int(MAX_TRIM * frames))
                    cut, found, tried = int(np.ceil(ex / ve.FR)), False, None
                    while cut <= limit:
                        xs2, sp2 = E.speech_once(text, lang, voices, s, mode, max_frames=frames - cut, **kw)
                        got = int(np.sum(sp2["dur"][1:-1]))
                        l20, l40, mix2 = lens(xs2)
                        ex2 = excess(l20, l40)
                        attempts.append(dict(speed=s, frames=got, trimmed_from=frames, len_s=round(l40, 3), fits=bool(ex2 <= 0)))
                        tried = (xs2, sp2, mix2, (frames, got))
                        if ex2 <= 0:
                            found = True
                            break
                        if got > frames - cut:               # nothing left to trim
                            break
                        cut += max(1, int(np.ceil(ex2 / ve.FR)))
                    if not found and not last_resort:
                        continue
                    if tried is not None:
                        xs, sp, mix, trimmed = tried
                    forced = not found
                takes.append(dict(speed=s, xs=xs, ph=sp["ph"], trimmed=trimmed, mix=mix, forced=forced,
                                  len_s=round(l40, 3), len_to_minus20_s=round(l20, 3)))
                if len(takes) >= MAX_TAKES or (s == grid[0] and trimmed is None):
                    break                                    # the style's own speed fits: nothing to choose
            scores = [None] * len(takes)
            if len(takes) > 1 and mode != "whisper":
                for i_, t in enumerate(takes):
                    on_ = envelope_onset_end(t["mix"], ve.KSR)[0] / ve.KSR
                    scores[i_] = self.take_score(t["mix"], lang, ref, (on_ - 0.05, on_ + window + 0.02))
            pick = self.best_take(scores)
            t = takes[pick]
            L = max(map(len, t["xs"]))
            arr = np.stack([np.pad(x, (0, L - len(x))) for x in t["xs"]])
            info = dict(speed=t["speed"], ph=t["ph"], trimmed=t["trimmed"], forced=bool(t["forced"]), natural=nat,
                        len_s=t["len_s"], len_to_minus20_s=t["len_to_minus20_s"], attempts=attempts,
                        takes=[dict(speed=q["speed"], trimmed=q["trimmed"], recall=s_ and s_[0], char_sim=s_ and s_[1])
                               for q, s_ in zip(takes, scores)])
            return arr, info
        key = dict(text=text, lang=lang, voices=voices, mode=mode, window=round(window, 5), par=par, seed=seed, ref=ref,
                   cfg=self.cfg if mode == "rapper" else None, asr=self.have_asr, scoring="audit-cut",
                   fit=(SPEED_GRID, MAX_TRIM, MAX_TAKES, QUIET_BEFORE_DEADLINE_S, END_BEFORE_DEADLINE_S,
                        R.SPEECH_HOLD_S, R.SPEECH_RELEASE_S, R.SPEECH_FADE_IN_S))
        arr, info = E.memo("speech", key, run)
        return [np.asarray(a, np.float64) for a in np.atleast_2d(arr)], info

    def render_phrase(self, item, kind, target, deadline):
        """An ad-lib or a spoken line: starts on `target`, over before `deadline`."""
        text = item.get("tts_text") or item["text"]
        style, role = item["style"], item["role"]
        seed = R.seed_of(f"{text}|{role}|{style}")          # a repeated ad-lib is the same render
        if role == "lead":
            voices = [LEAD]
            mode = "whisper" if style == "whisper" else "rapper"
            par = dict(LEAD_SHOUT) if style == "shout" else dict(speed=1.0, f0_scale=1.0, f0_range=1.0)
            lvl = "adlib_lead" if kind == "adlibs" else "spoken"
        else:
            pool = sorted(self.gang_males(), key=lambda v: 0 if v[0] == "e" else 1)
            k = seed % max(1, len(pool))
            es = [v for v in pool if v[0] == "e"]
            en = [v for v in pool if v[0] != "e"]
            if kind == "adlibs" or style == "shout":
                voices = (es[:2] + ([en[k % len(en)]] if en else []))[:3] or pool[:3]
            else:
                voices = [es[seed % len(es)]] if es else [pool[k]]      # one voice: a Spanish one if there is one
            mode = "whisper" if style == "whisper" else "plain"
            par = dict(SHOUT) if style == "shout" else dict(speed=1.0, f0_scale=1.0, f0_range=1.0)
            lvl = "adlib_gang"
        if style == "whisper":
            par = dict(speed=0.95, f0_scale=1.0, f0_range=1.0)
            lvl = "whisper"
        stem = kind
        window = deadline - target
        xs, info = self.fit_speech(text, item["lang"], voices, mode, window, par, seed, item["text"])
        n = len(voices)
        base_pan = float(item.get("pan", 0.0))
        pans = [base_pan] if n == 1 else [float(np.clip(base_pan + d, -1, 1)) for d in ([0.0, -0.4, 0.4] if n == 3 else [-0.3, 0.3])]
        outs = [norm_to(x, lvl, trim=0.0 if (v[0] == "e" or n == 1) else -2.0) for v, x in zip(voices, xs)]
        L = min(map(len, outs))
        mix = np.sum([x[:L] for x in outs], axis=0)
        on, _ = envelope_onset_end(mix, ve.KSR)
        start = target - on / ve.KSR
        # safety net: a 20 ms fade that reaches zero 5 ms before the deadline (does nothing to a phrase that fits)
        Lm = max(map(len, outs))
        edge = np.ones(Lm)
        z1 = int(round((deadline - start) * ve.KSR)) - int(0.005 * ve.KSR)
        z0 = z1 - int(0.020 * ve.KSR)
        if z0 < Lm:
            edge[max(0, z1):] = 0.0
            seg = np.arange(max(0, z0), min(Lm, max(0, z1)))
            edge[seg] *= 0.5 + 0.5 * np.cos(np.pi * (seg - z0) / max(1, z1 - z0))
        outs = [x * edge[:len(x)] for x in outs]
        # the voices of one phrase add up: hold the phrase (not the whole stem) under the stem ceiling
        gl = np.array([pan_gains(p)[0] for p in pans])
        gr = np.array([pan_gains(p)[1] for p in pans])
        pk = max(np.max(np.abs(sum(g * x[:L] for g, x in zip(gl, outs)))), np.max(np.abs(sum(g * x[:L] for g, x in zip(gr, outs)))))
        ceil = 10 ** ((STEM_CEILING_DB - 0.8) / 20)          # 0.8 dB for inter-sample peaks after resampling
        pg = min(1.0, ceil / (pk + 1e-12))
        outs = [x * pg for x in outs]
        for j, (v, xn) in enumerate(zip(voices, outs)):
            self.dbg(stem, f"{item['id']}_{v}", xn)
            self.place(stem, xn, start, pans[j], item["id"])
        mix = np.sum([x[:L] for x in outs], axis=0)
        fr, hop = env_frames(mix, ve.KSR)
        pkf = fr.max()
        on2 = np.nonzero(fr > pkf * 10 ** (-32 / 20))[0][0] * hop
        end2 = (np.nonzero(fr > pkf * 10 ** (END_REL_DB / 20))[0][-1] + 1) * hop
        a_, b_ = int(round((deadline - 0.060 - start) * ve.KSR)) // hop, int(round((deadline - 0.030 - start) * ve.KSR)) // hop
        before = float(fr[max(0, a_):max(a_ + 1, b_)].max()) if a_ < len(fr) else 0.0
        lvl_before = r1(20 * np.log10(before / pkf + 1e-9))
        fit_log = []
        if abs(info["speed"] - info["natural"]["speed"]) > 1e-6:
            fit_log.append(f"speed {info['speed']:.2f}")
        if info["trimmed"]:
            fit_log.append(f"durations trimmed {info['trimmed'][0]}->{info['trimmed'][1]} frames")
        if info["forced"]:
            fit_log.append("does not fit: cut by the deadline fade")
        e = dict(id=item["id"], stem=f"vox_{stem}", bar=item["bar"], style=style, role=role, text=item["text"],
                 tts_text=item.get("tts_text"), voices=voices, pans=[round(p, 2) for p in pans], mode=mode,
                 phonemes=info["ph"], target_onset_s=round(target, 4), onset_s=round(start + on2 / ve.KSR, 4),
                 onset_err_ms=r1(1000 * (start + on2 / ve.KSR - target)), end_s=round(start + end2 / ve.KSR, 4),
                 deadline_s=round(deadline, 4), margin_ms=r1(1000 * (deadline - (start + end2 / ve.KSR))),
                 window_s=round(window, 3), speed=info["speed"], fit_log=fit_log, attempts=info["attempts"],
                 takes=info["takes"], natural=info["natural"], length_s=info["len_s"],
                 trimmed_share=round(1 - info["trimmed"][1] / info["trimmed"][0], 3) if info["trimmed"] else 0.0,
                 level_before_deadline_db=lvl_before, phrase_gain_db=r1(20 * np.log10(pg)))
        if kind == "adlibs":
            e["slot"], e["slots"] = item["slot"], item["slots"]
            six = window / max(1, item["slots"])
            e["slots_needed_at_natural_speed"] = int(np.ceil((info["natural"]["len_to_minus20_s"] + QUIET_BEFORE_DEADLINE_S) / six))
        else:
            e["beat"], e["max_beats"] = item["beat"], item["max_beats"]
        e["squeezed"] = bool(info["forced"] or info["trimmed"] or info["speed"] >= 1.6)
        e["cut_off_suspect"] = bool(lvl_before is not None and lvl_before > -20.0)
        if mode == "rapper":
            mt = R.rasp_metrics(mix)
            e["f0_median_hz"] = mt["f0_med"]
            e["rasp"] = {k: mt[k] for k in ("hnr_db", "periodicity", "shr_db", "nhr_1_4k_db")}
        e["asr"] = self.asr_check(mix, item["lang"], item["text"])
        if mode != "whisper":
            e["asr_raw_tts"] = self.asr_raw(text, item["lang"], voices[0], item["text"])
        self.regions[stem].append((target - 0.02, deadline, item["id"]))
        self.report[kind].append(e)
        return e

    # ------------------------------------------------------------------------- run
    def run(self):
        t0 = time.time()
        E = self.E
        ts = time.time()
        flows = R.flow(score, score.RAP, E, LEAD) if score.RAP else []
        self.times["flow_s"] = round(time.time() - ts, 1)
        order = sorted(range(len(score.RAP)), key=lambda i: score.RAP[i]["bar"] * 16 + score.RAP[i]["anchor"])
        todo = [i for i in order if self.want(score.RAP[i]["id"])]
        ts = time.time()
        for n, i in enumerate(todo):
            tl = time.time()
            ln, f = score.RAP[i], flows[i]
            e = self.render_rap(order.index(i), ln, f)
            a = e.get("asr") or {}
            print(f"[rap {n + 1}/{len(todo)}] {ln['id']:6s} {time.time() - tl:5.1f}s syl {e['n_syllables']:2d}"
                  f"{' CROWDED' if e['crowded'] else ''} loop {e['loop']['costs']} f0 {e['f0_median_hz']} asr {a.get('recall', '-')} "
                  f"(raw {(e.get('asr_raw_tts') or {}).get('recall', '-')}) {a.get('hyp', '')}", flush=True)
        self.times["rap_s"] = round(time.time() - ts, 1)
        ts = time.time()
        sung = [p for p in score.SUNG if self.want(p["id"])]
        for n, p in enumerate(sung):
            tl = time.time()
            for e in self.render_hook(p):
                a = e.get("asr") or {}
                med = [r["pitch"]["med_cents"] for r in e["renders"]]
                print(f"[hook {n + 1}/{len(sung)}] {p['id']:6s} {e['layer']:5s} {time.time() - tl:5.1f}s notes {e['n_notes']} "
                      f"syl {e['natural_syllables']} {e['reconciliation'] or ''} balance {e['balance']['preset']!r} ({e['balance']['leads']}) "
                      f"cents {med} asr {a.get('recall', '-')} {a.get('hyp', '')}", flush=True)
        self.times["hook_s"] = round(time.time() - ts, 1)
        ts = time.time()
        ads = [a for a in score.ADLIBS if self.want(a["id"])]
        for n, a in enumerate(ads):
            tl = time.time()
            e = self.render_phrase(a, "adlibs", score.t16(a["bar"], a["slot"]), score.t16(a["bar"], a["slot"] + a["slots"]))
            q = e.get("asr") or {}
            print(f"[adlib {n + 1}/{len(ads)}] {a['id']:8s} {a['role']:5s} {time.time() - tl:5.1f}s onset {e['onset_err_ms']:+.1f} ms "
                  f"margin {e['margin_ms']:.0f} ms speed {e['speed']:.2f} {e['fit_log']} asr {q.get('recall', '-')} {q.get('hyp', '')}",
                  flush=True)
        sps = [s for s in score.SPOKEN if self.want(s["id"])]
        for n, s in enumerate(sps):
            tl = time.time()
            e = self.render_phrase(s, "spoken", score.t(s["bar"], s["beat"]), score.t(s["bar"], s["beat"] + s["max_beats"]))
            q = e.get("asr") or {}
            print(f"[spoken {n + 1}/{len(sps)}] {s['id']:8s} {time.time() - tl:5.1f}s onset {e['onset_err_ms']:+.1f} ms "
                  f"margin {e['margin_ms']:.0f} ms speed {e['speed']:.2f} {e['fit_log']} asr {q.get('recall', '-')} {q.get('hyp', '')}",
                  flush=True)
        self.times["speech_s"] = round(time.time() - ts, 1)
        self.write(flows, t0)

    # ------------------------------------------------------------------------- measurements on the stems
    def stem_timing(self, mono):
        """Re-measure every vowel onset on the finished stems with the three detectors (R.line_readings)."""
        x16 = {k: vo.to16(v, SR) for k, v in mono.items()}
        envs = {k: {b: R.band_env(x16[k], b) for b in R.MR_BANDS} for k in x16}

        def read(name, Ts, Ds, ok):
            return compact(R.line_readings(x16[name], Ts, Ds, ok, envs[name]))

        def put(e, rd, pre):
            e[pre + "_onset_err_ms"] = rd["per"]
            e[pre + "_wide_err_ms"] = rd["wide"]
            e[pre + "_form_err_ms"] = rd["form"]
            e[pre + "_rise_err_ms"] = [w if w is not None else f for w, f in zip(rd["wide"], rd["form"])]
            e[pre + "_ill_defined"] = rd["flags"]

        def one(rd, k):
            vals = {d: rd[d][k] for d in ("per", "wide", "form")}
            have = [v for v in vals.values() if v is not None]
            en = [v for v in (vals["wide"], vals["form"]) if v is not None]
            return dict(vals, value=r1(np.mean(en)) if en else vals["per"], verified=bool(have),
                        worst=r1(max(have, key=abs)) if have else None)
        for e in self.report["rap"]:
            name = e["stem"][4:]
            if name not in x16:
                continue
            rd = read(name, e["slot_times_s"], e["note_durs_s"], e["measurable"])
            put(e, rd, "stem")
            k = e["rhyme_syllable"]
            e["rhyme_err_ms"] = one(rd, k)
            if name == "rap" and "twin" in x16:
                rt_ = read("twin", e["slot_times_s"], e["note_durs_s"], e["measurable"])
                put(e, rt_, "twin")
                e["twin_rhyme_err_ms"] = one(rt_, k)
            if e.get("punch") and "punch" in x16:
                # the doubles against the lead, both on their stems (periodicity only where the lead's plan
                # has a voicing boundary: behind the gate a voiced consonant is already periodic)
                rp = read("punch", e["slot_times_s"][k:], e["note_durs_s"][k:], e["measurable"][k:])
                pr = one(rp, 0)
                lead = e["rhyme_err_ms"]
                d = None
                for band in ("wide", "form", "per"):
                    if pr[band] is not None and lead[band] is not None:
                        d = dict(ms=r1(pr[band] - lead[band]), detector=band)
                        break
                e["punch"]["stem_rhyme_err_ms"] = pr
                e["punch"]["stem_wide_err_ms"], e["punch"]["stem_form_err_ms"] = rp["wide"], rp["form"]
                e["punch"]["minus_lead"] = d
        for e in self.report["hook"]:
            name = e["stem"][4:]
            if name in x16:
                put(e, read(name, e["note_times_s"], e["note_durs_s"], e["measurable"]), "stem")

    def stem_phrases(self, mono):
        """Onset and end of every ad-lib / spoken line measured on the finished stem."""
        for kind in ("adlibs", "spoken"):
            x = mono[kind]
            for e in self.report[kind]:
                a = max(0, int((e["target_onset_s"] - 0.12) * SR))
                b = min(NSAMP, int((e["deadline_s"] + 0.08) * SR))
                seg = np.abs(x[a:b])
                if seg.max() <= 0:
                    e["stem_onset_err_ms"] = None
                    continue
                d0 = max(0, int((e["target_onset_s"] - 0.02) * SR) - a)
                d1 = int((e["deadline_s"]) * SR) - a
                pk = seg[d0:d1].max()
                hop = int(0.003 * SR)
                nfr = len(seg) // hop
                fr = np.sqrt(np.mean(seg[:nfr * hop].reshape(nfr, hop) ** 2, axis=1))
                top = fr[d0 // hop:max(d0 // hop + 1, d1 // hop)].max()
                on = np.nonzero(fr > top * 10 ** (-32 / 20))[0]
                e["stem_onset_err_ms"] = r1(1000 * ((a + on[0] * hop) / SR - e["target_onset_s"])) if len(on) else None
                tail = seg[max(0, d1):min(len(seg), d1 + int(0.05 * SR))]
                e["stem_level_after_deadline_db"] = r1(20 * np.log10((tail.max() if len(tail) else 0.0) / pk + 1e-12))
                e["stem_start_level_db"] = r1(start_level_db(x[a:b]))

    def stem_asr(self, mono):
        """Whisper on every line cut from the finished stems (the way an outside audit hears them)."""
        if not self.asr:
            return

        def cut(x, a, b):
            y = x[max(0, int(a * SR)):min(len(x), int(b * SR))].astype(np.float64).copy()
            n = min(len(y) // 2, int(0.008 * SR))
            if n:
                y[:n] *= np.linspace(0, 1, n)
                y[-n:] *= np.linspace(1, 0, n)
            return y

        def check(x, a, b, lang, ref):
            hyp = self.transcribe(cut(x, a, b), lang, SR)
            return dict(hyp=hyp, **R.asr_scores(ref, hyp, lang))
        rap = sorted(self.report["rap"], key=lambda e: e["slot_times_s"][0])
        for i, e in enumerate(rap):
            T0 = e["slot_times_s"][0]
            a = T0 - 0.22
            if i > 0:
                p = rap[i - 1]
                a = max(a, min(T0 - 0.06, p["slot_times_s"][-1] + p["note_durs_s"][-1]))
            b = e["deadline_s"] + 0.02
            if i + 1 < len(rap):
                b = min(b, rap[i + 1]["slot_times_s"][0] - 0.05)
            e["asr_stem"] = check(mono[e["stem"][4:]], a, b, "es" if "lang" not in e else e["lang"], e["text"])
        for e in self.report["hook"]:
            e["asr_stem"] = check(mono[e["stem"][4:]], e["note_times_s"][0] - 0.2, e["note_times_s"][-1] + e["note_durs_s"][-1] + 0.1,
                                  "es", e["text"])
        for kind in ("adlibs", "spoken"):
            for e in self.report[kind]:
                e["asr_stem"] = check(mono[kind], e["target_onset_s"] - 0.05, e["deadline_s"] + 0.02, "es", e["text"])

    def write(self, flows, t0):
        partial = self.only is not None
        out_dir = os.path.join(DEBUG_DIR, "partial") if partial else STEM_DIR
        os.makedirs(out_dir, exist_ok=True)
        lim = 10 ** (STEM_CEILING_DB / 20)
        sos30 = butter(2, 30.0, "lowpass", fs=SR, output="sos")
        final = {}
        for k, st in self.stems.items():
            if not np.all(np.isfinite(st)):
                raise RuntimeError(f"stem {k} has NaN/inf")
            pk = float(np.max(np.abs(st)))
            g = 1.0
            if pk > lim:
                g = lim / pk
            y = (st * g).astype(np.float32)
            assert y.shape == (NSAMP, 2)
            final[k] = y
            sf.write(os.path.join(out_dir, f"vox_{k}.wav"), y, SR, subtype="FLOAT")
            mono = y.mean(axis=1).astype(np.float64)
            hop = SR // 10
            n = len(mono) // hop
            fr = np.sqrt(np.mean(mono[:n * hop].reshape(n, hop) ** 2, axis=1))
            act = fr > 10 ** (-60 / 20)
            # anything outside the regions where the score puts this stem's vocals?
            mask = np.zeros(NSAMP, bool)
            for (a, b, _) in self.regions[k]:
                mask[max(0, int(a * SR)):min(NSAMP, int(np.ceil(b * SR)))] = True
            outside = np.abs(y[~mask]).max() if (~mask).any() else 0.0
            sub = sosfiltfilt(sos30, mono) if pk > 0 else np.zeros(1)
            rms = float(np.sqrt(np.mean(mono[np.abs(mono) > 0] ** 2))) if pk > 0 else 0.0
            self.report["stems"][f"vox_{k}"] = dict(
                peak_dbfs=round(20 * np.log10(np.max(np.abs(y)) + 1e-12), 2),
                rms_dbfs=round(20 * np.log10(np.sqrt(np.mean(y.astype(np.float64) ** 2)) + 1e-12), 2),
                active_rms_dbfs=round(20 * np.log10(np.sqrt(np.mean(fr[act] ** 2)) + 1e-12), 2) if act.any() else None,
                active_seconds=round(float(act.sum() * hop / SR), 1), gain_applied_db=round(20 * np.log10(g), 2),
                samples=int(y.shape[0]), channels=2, sr=SR, nan=False,
                clipped_samples=int(np.sum(np.abs(y) >= 1.0)),
                outside_score_regions_peak_dbfs=round(20 * np.log10(outside + 1e-12), 1),
                max_sample_step=round(float(np.max(np.abs(np.diff(y, axis=0)))), 4) if len(y) > 1 else 0.0,
                dc=[round(float(y[:, 0].mean()), 6), round(float(y[:, 1].mean()), 6)],
                dc_re_active_rms_db=round(20 * np.log10(abs(float(mono[np.abs(mono) > 0].mean())) / rms + 1e-12), 1) if pk > 0 else None,
                below_30hz_peak_dbfs=round(20 * np.log10(np.max(np.abs(sub)) + 1e-12), 1),
                silent=bool(pk == 0.0))
        mono = {k: final[k].mean(axis=1).astype(np.float64) for k in STEMS}
        tm = {k: mono[k] for k in ("rap", "punch", "hook", "kids")}
        if np.any(self.twin):
            tm["twin"] = self.twin
            sf.write(os.path.join(DEBUG_DIR, "partial" if partial else "", "rap_clean_twin.wav"),
                     (self.twin * min(1.0, 0.89 / (np.max(np.abs(self.twin)) + 1e-12))).astype(np.float32), SR, subtype="FLOAT")
        ts = time.time()
        self.stem_timing(tm)
        self.stem_phrases(mono)
        self.stem_asr(mono)
        self.times["stem_checks_s"] = round(time.time() - ts, 1)
        self.times["total_s"] = round(time.time() - t0, 1)
        self.report["engine"] = dict(
            rapper=self.cfg, voices=dict(lead=LEAD, gang=GANG, kids=KIDS), version=R.VERSION,
            kids_octave=KIDS_OCTAVE, kids_formants=KIDS_FORMANTS, punch_up_semitones=R.PUNCH_UP,
            timing_loop=dict(passes=R.LOOP_PASSES, tol_ms=R.LOOP_TOL_MS, max_step_ms=R.LOOP_MAX_STEP_MS),
            onset_contrast=dict(wanted_db=R.CONTRAST_DB, max_db=R.CONTRAST_MAX_DB),
            punch=dict(lead_in_syllables=R.PUNCH_LEAD_IN, fade_s=R.PUNCH_FADE_S, earliest_before_anchor_s=R.PUNCH_PRE_MAX_S),
            hook=dict(full_vowels=R.HOOK_FULL_VOWELS, vowel_share=R.HOOK_NUC_SHARE, gang_presets=GANG_PRESETS, kids_presets=KIDS_PRESETS,
                      bus_contrast=HOOK_BUS_CONTRAST),
            speech=dict(speed_grid=SPEED_GRID, max_trim=MAX_TRIM, max_takes=MAX_TAKES),
            rap_retakes=dict(alternatives=RAP_RETAKES, below=RETAKE_BELOW, cost_slack=RETAKE_COST_SLACK),
            levels_dbfs_rms=LVL, memo_hits=self.E.memo_hits, memo_misses=self.E.memo_miss, asr_for_takes=self.have_asr)
        self.report["score"] = dict(title=score.TITLE, bpm=score.BPM, swing=score.SWING, lyrics_final=score.LYRICS_FINAL,
                                    song_seconds=score.SONG_SECONDS, rap_lines=len(score.RAP), sung_lines=len(score.SUNG),
                                    adlibs=len(score.ADLIBS), spoken=len(score.SPOKEN), partial=sorted(self.only) if partial else None)
        self.report["render_seconds"] = self.times
        worst = max(self.steps, default=(None, None))
        self.report["start_level_worst"] = dict(db_re_peak_after_0p5ms=r1(worst[0]), id=worst[1])
        self.report["summary"] = summarize(self.report)
        with open(os.path.join(out_dir, "vocals_report.json"), "w") as fh:
            json.dump(self.report, fh, indent=1, ensure_ascii=False, default=R._jsonable)
        print(json.dumps(self.report["summary"], indent=1, ensure_ascii=False, default=R._jsonable))
        print(f"stems -> {out_dir}   ({self.times['total_s']:.0f} s, cache hits {self.E.memo_hits}, misses {self.E.memo_miss})")


# ----------------------------------------------------------------------------- summary
def _stats(vals, tol):
    vals = [v for v in vals if v is not None]
    if not vals:
        return dict(n=0)
    a = np.abs(vals)
    return dict(n=len(vals), within=round(float(np.mean(a <= tol)), 3), tol_ms=tol, mean_abs=round(float(a.mean()), 1),
                median=round(float(np.median(vals)), 1), p90_abs=round(float(np.percentile(a, 90)), 1),
                max_abs=round(float(a.max()), 1))


DETECTORS = (("periodicity", "onset"), ("energy_wide", "wide"), ("energy_formant", "form"))


def _onset_block(entries, pre, tol, label):
    """Timing statistics of a set of lines measured on a stem (pre = 'stem' or 'twin')."""
    d = {}
    tot = sum(len(e[pre + "_onset_err_ms"]) for e in entries)
    for name, key in DETECTORS:
        d[name] = _stats([v for e in entries for v in e[f"{pre}_{key}_err_ms"]], tol)
    per_syl = []
    for e in entries:
        for k in range(len(e[pre + "_onset_err_ms"])):
            vals = [e[f"{pre}_{key}_err_ms"][k] for _, key in DETECTORS]
            vals = [v for v in vals if v is not None]
            en = [e[f"{pre}_{key}_err_ms"][k] for key in ("wide", "form") if e[f"{pre}_{key}_err_ms"][k] is not None]
            per_syl.append((e, k, vals, en))
    meas = [q for q in per_syl if q[2]]
    d["syllables"] = tot
    d["measurable"] = len(meas)
    d["coverage"] = round(len(meas) / max(1, tot), 3)
    # the strict figure: a syllable counts only if EVERY detector that can read it puts it inside the tolerance
    d["all_detectors_within"] = round(float(np.mean([all(abs(v) <= tol for v in q[2]) for q in meas])), 3) if meas else None
    # the lenient figure the audit also quotes: the kinder of the two energy bands
    kinder = [min(q[3], key=abs) for q in per_syl if q[3]]
    d["energy_kinder_band"] = _stats(kinder, tol)
    ctx = {}
    for e, k, vals, en in meas:
        ctx.setdefault(e["context"][k], []).append(all(abs(v) <= tol for v in vals))
    d["all_detectors_within_by_context"] = {c: dict(n=len(v), within=round(float(np.mean(v)), 3)) for c, v in sorted(ctx.items())}
    d["unmeasurable_by_context"] = {}
    for e, k, vals, en in per_syl:
        if not vals:
            c = e["context"][k]
            d["unmeasurable_by_context"][c] = d["unmeasurable_by_context"].get(c, 0) + 1
    d["violations"] = sorted(
        [dict(id=e["id"], syllable=(e["flow"][k]["syl"] if "flow" in e else e["syllables"][k]),
              slot=(e["flow"][k]["slot"] if "flow" in e else None), context=e["context"][k],
              periodicity=e[pre + "_onset_err_ms"][k], energy_wide=e[pre + "_wide_err_ms"][k],
              energy_formant=e[pre + "_form_err_ms"][k])
         for e, k, vals, en in meas if any(abs(v) > tol for v in vals)],
        key=lambda q: -max(abs(v) for v in (q["periodicity"], q["energy_wide"], q["energy_formant"]) if v is not None))
    return d


def summarize(rep):
    s = {}
    rap = [e for e in rep["rap"] if "stem_onset_err_ms" in e]
    lead = [e for e in rap if e["stem"] == "vox_rap"]
    if rap:
        s["rap_onsets"] = _onset_block(rap, "stem", TOL_MS, "rap")
        tw = [e for e in rap if "twin_onset_err_ms" in e]
        if tw:
            t = _onset_block(tw, "twin", TOL_MS, "twin")
            s["rap_onsets_clean_twin"] = {k: t[k] for k in ("periodicity", "energy_wide", "energy_formant", "coverage",
                                                             "all_detectors_within", "energy_kinder_band")}
        rh = [e["rhyme_err_ms"] for e in rap]
        ver = [r for r in rh if r["verified"]]
        s["rhyme_stress"] = dict(
            lines=len(rh), verified=len(ver), not_verified=[e["id"] for e in rap if not e["rhyme_err_ms"]["verified"]],
            periodicity=_stats([r["per"] for r in rh], RHYME_TOL_MS), energy_wide=_stats([r["wide"] for r in rh], RHYME_TOL_MS),
            energy_formant=_stats([r["form"] for r in rh], RHYME_TOL_MS),
            all_detectors_within=round(float(np.mean([abs(r["worst"]) <= RHYME_TOL_MS for r in ver])), 3) if ver else None,
            outside=[dict(id=e["id"], text=e["text"], syllable=e["flow"][e["rhyme_syllable"]]["syl"],
                          **{k: e["rhyme_err_ms"][k] for k in ("per", "wide", "form")})
                     for e in rap if e["rhyme_err_ms"]["verified"] and abs(e["rhyme_err_ms"]["worst"]) > RHYME_TOL_MS])
        s["crowded_lines"] = [dict(id=e["id"], text=e["text"], n_syllables=e["n_syllables"], pairs=e["thirty_second_pairs"],
                                   log=e["flow_log"]) for e in rap if e["crowded"]]
        s["rap_deadline_violations"] = [dict(id=e["id"], text=e["text"], margin_ms=e["deadline_margin_ms"])
                                        for e in rap if e["deadline_margin_ms"] < 0]
        # overlaps: a line's audible end (-40 dB) against the next line's first vowel slot and first sound
        by_time = sorted(rap, key=lambda e: e["slot_times_s"][0])
        pairs = [(a, b) for a, b in zip(by_time[:-1], by_time[1:]) if a["stem"] == b["stem"]]
        s["rap_overlaps"] = [dict(line=a["id"], next=b["id"], overlap_ms=r1(1000 * (a["end_s"] - b["slot_times_s"][0])))
                             for a, b in pairs if b["slot_times_s"][0] < a["end_s"]]
        s["rap_tail_under_next_first_sound"] = [dict(line=a["id"], next=b["id"], ms=r1(1000 * (a["end_s"] - b["first_sound_s"])))
                                                for a, b in pairs if b["first_sound_s"] < a["end_s"]]
        s["rap_min_gap_to_next_line_ms"] = r1(min([1000 * (b["slot_times_s"][0] - a["end_s"]) for a, b in pairs], default=0.0)) if pairs else None
        if lead:
            f0 = [e["f0_median_hz"] for e in lead if e["f0_median_hz"]]
            s["rapper_f0_median_hz"] = dict(median=round(float(np.median(f0)), 1), min=min(f0), max=max(f0))
            tw = [e for e in lead if e.get("clean_twin")]
            if tw:
                def mean(get):
                    v = [get(e) for e in tw]
                    v = [x for x in v if x is not None]
                    return round(float(np.mean(v)), 2) if v else None
                keys = ("hnr_db", "periodicity", "shr_db", "nhr_1_4k_db")
                s["rasp"] = dict(n_lines=len(tw), clean={k: mean(lambda e, k=k: e["clean_twin"]["rasp"][k]) for k in keys},
                                 raspy={k: mean(lambda e, k=k: e["rasp"][k]) for k in keys},
                                 rougher_in_lines={k: int(sum((e["rasp"][k] < e["clean_twin"]["rasp"][k]) if k in ("hnr_db", "periodicity")
                                                              else (e["rasp"][k] > e["clean_twin"]["rasp"][k]) for e in tw)) for k in keys},
                                 asr_recall_clean=mean(lambda e: (e["clean_twin"].get("asr") or {}).get("recall")),
                                 asr_recall_raspy=mean(lambda e: (e.get("asr") or {}).get("recall")))
        a = [e for e in lead if e.get("asr")]
        if a:
            def block(key):
                rec = float(np.mean([e[key]["recall"] for e in a]))
                raw = float(np.mean([e["asr_raw_tts"]["recall"] for e in a]))
                return dict(recall=round(rec, 3), raw_tts_recall=round(raw, 3), difference=round(rec - raw, 3),
                            strict_recall=round(float(np.mean([e[key]["strict"] for e in a])), 3),
                            char_sim=round(float(np.mean([e[key]["char_sim"] for e in a])), 3),
                            lines_within_0p10_of_raw=int(sum(e[key]["recall"] >= e["asr_raw_tts"]["recall"] - 0.10 for e in a)),
                            weak=[dict(id=e["id"], text=e["text"], recall=e[key]["recall"], raw=e["asr_raw_tts"]["recall"], hyp=e[key]["hyp"])
                                  for e in a if e[key]["recall"] < 0.7])
            s["rap_asr"] = block("asr_stem") if all("asr_stem" in e for e in a) else block("asr")
            s["rap_asr"]["measured_on"] = "lines cut from the stem" if all("asr_stem" in e for e in a) else "isolated renders"
            s["rap_asr_isolated_renders"] = {k: v for k, v in block("asr").items() if k != "weak"}
        tk = [e for e in lead if e.get("take")]
        if tk:
            used = {}
            for e in tk:
                used[e["take"]["used"]] = used.get(e["take"]["used"], 0) + 1
            s["rap_takes"] = dict(lines=len(tk), taken_again=int(sum(len(e["take"]["tried"]) > 1 for e in tk)), used=used)
        pun = [e for e in rap if e.get("punch") and e["punch"].get("stem_rhyme_err_ms")]
        if pun:
            d = [e["punch"]["minus_lead"]["ms"] for e in pun if e["punch"]["minus_lead"]]
            s["punch_ins"] = dict(
                lines=len(pun), rhyme_onset_energy_wide=_stats([e["punch"]["stem_rhyme_err_ms"]["wide"] for e in pun], TOL_MS),
                rhyme_onset_energy_formant=_stats([e["punch"]["stem_rhyme_err_ms"]["form"] for e in pun], TOL_MS),
                rhyme_onset_periodicity=_stats([e["punch"]["stem_rhyme_err_ms"]["per"] for e in pun], TOL_MS),
                all_syllables_energy_wide=_stats([v for e in pun for v in e["punch"]["stem_wide_err_ms"]], TOL_MS),
                minus_lead=_stats(d, TOL_MS), minus_lead_not_measurable=[e["id"] for e in pun if not e["punch"]["minus_lead"]],
                minus_lead_outside=[dict(id=e["id"], text=e["text"], **e["punch"]["minus_lead"]) for e in pun
                                    if e["punch"]["minus_lead"] and abs(e["punch"]["minus_lead"]["ms"]) > TOL_MS],
                gate_opens_ms_before_anchor=dict(earliest=min(r["gate_opens_ms"] for e in pun for r in e["punch"]["renders"]),
                                                 latest=max(r["gate_opens_ms"] for e in pun for r in e["punch"]["renders"])))
    hook = [e for e in rep["hook"] if "stem_onset_err_ms" in e]
    for layer in ("gang", "kids"):
        hs = [e for e in hook if e["layer"] == layer]
        if not hs:
            continue
        d = dict(onsets=_onset_block(hs, "stem", TOL_MS, layer))
        meds = [r["pitch"]["med_cents"] for e in hs for r in e["renders"] if r["pitch"]["med_cents"] is not None]
        if meds:
            d["pitch_median_abs_cents"] = dict(median=round(float(np.median(meds)), 1), worst=round(float(np.max(meds)), 1))
        for kind in ("short", "long"):
            blk = dict(n=sum(len(r["note_cents"][kind]["whole"]) for e in hs for r in e["renders"] if r.get("note_cents")))
            for win in ("whole", "vowel"):
                c = [v for e in hs for r in e["renders"] if r.get("note_cents") for v in r["note_cents"][kind][win]]
                ok = [v for v in c if v is not None]
                blk[f"{win}_window"] = dict(unvoiced=len(c) - len(ok), median_abs_cents=round(float(np.median(np.abs(ok))), 1) if ok else None,
                                            over_50_cents=int(sum(abs(v) > 50 for v in ok)))
            sh = [v for e in hs for r in e["renders"] if r.get("note_cents") for v in r["note_cents"][kind]["vowel_share"]]
            blk["whole_window_less_than_half_vowel"] = int(sum(v < 0.5 for v in sh))
            d[f"{kind}_notes"] = blk
        d["rings_past_written_end_ms"] = dict(max=r1(max(1000 * (e["end_s"] - e["written_end_s"]) for e in hs)))
        d["balance_used"] = {}
        for e in hs:
            d["balance_used"][e["balance"]["preset"]] = d["balance_used"].get(e["balance"]["preset"], 0) + 1
        a = [e for e in hs if e.get("asr")]
        if a:
            key = "asr_stem" if all("asr_stem" in e for e in a) else "asr"
            full = [e for e in a if len(e["text"].split()) >= 3]
            d["asr_measured_on"] = "lines cut from the stem" if key == "asr_stem" else "isolated mixes"
            d["asr_recall"] = round(float(np.mean([e[key]["recall"] for e in a])), 3)
            d["asr_recall_full_lines"] = round(float(np.mean([e[key]["recall"] for e in full])), 3) if full else None
            d["asr_lines"] = [dict(id=e["id"], text=e["text"], recall=e[key]["recall"], raw=(e.get("asr_raw_tts") or {}).get("recall"),
                                   hyp=e[key]["hyp"]) for e in a]
            if layer == "gang":
                d["asr_raw_tts_recall"] = round(float(np.mean([e["asr_raw_tts"]["recall"] for e in a])), 3)
                d["asr_raw_tts_recall_full_lines"] = round(float(np.mean([e["asr_raw_tts"]["recall"] for e in full])), 3) if full else None
                can = [e for e in full if e["asr_raw_tts"]["recall"] >= 0.75]
                d["full_lines_where_raw_reaches_0p75"] = len(can)
                d["of_those_reaching_0p75"] = int(sum(e[key]["recall"] >= 0.75 for e in can))
                d["of_those_below_0p75"] = [dict(id=e["id"], text=e["text"], recall=e[key]["recall"], hyp=e[key]["hyp"])
                                            for e in can if e[key]["recall"] < 0.75]
        d["reconciliations"] = [dict(id=e["id"], notes=e["n_notes"], natural_syllables=e["natural_syllables"], log=e["reconciliation"])
                                for e in hs if e["reconciliation"]]
        s[f"hook_{layer}"] = d
    for kind in ("adlibs", "spoken"):
        it = rep[kind]
        if not it:
            continue
        oe = [abs(e["onset_err_ms"]) for e in it]
        so = [abs(e["stem_onset_err_ms"]) for e in it if e.get("stem_onset_err_ms") is not None]
        d = dict(n=len(it), onset_abs_err_ms=dict(mean=round(float(np.mean(oe)), 2), max=round(float(np.max(oe)), 2)),
                 stem_onset_abs_err_ms=dict(mean=round(float(np.mean(so)), 2), max=round(float(np.max(so)), 2)) if so else None,
                 onset_violations=[e["id"] for e in it if abs(e["onset_err_ms"]) > SPEECH_TOL_MS
                                   or (e.get("stem_onset_err_ms") is not None and abs(e["stem_onset_err_ms"]) > SPEECH_TOL_MS)],
                 deadline_violations=[e["id"] for e in it if e["margin_ms"] < 0],
                 min_margin_ms=float(min(e["margin_ms"] for e in it)),
                 start_level_worst_db=max([e["stem_start_level_db"] for e in it if e.get("stem_start_level_db") is not None], default=None),
                 squeezed=[dict(id=e["id"], text=e["text"], window_s=e["window_s"], natural_s=e["natural"]["len_s"], speed=e["speed"],
                                trimmed_share=e["trimmed_share"], slots=e.get("slots"), slots_needed=e.get("slots_needed_at_natural_speed"),
                                log=e["fit_log"]) for e in it if e["squeezed"]],
                 cut_off_suspects=[dict(id=e["id"], text=e["text"], level_before_deadline_db=e["level_before_deadline_db"])
                                   for e in it if e["cut_off_suspect"]])
        a = [e for e in it if e.get("asr")]
        if a:
            key = "asr_stem" if all("asr_stem" in e for e in a) else "asr"
            d["asr_recall"] = round(float(np.mean([e[key]["recall"] for e in a])), 3)
            raw = [e["asr_raw_tts"]["recall"] for e in a if e.get("asr_raw_tts")]
            d["asr_raw_tts_recall"] = round(float(np.mean(raw)), 3) if raw else None
            d["asr_char_sim"] = round(float(np.mean([e[key]["char_sim"] for e in a])), 3)
            d["asr_lines"] = [dict(id=e["id"], text=e["text"], recall=e[key]["recall"], raw=(e.get("asr_raw_tts") or {}).get("recall"),
                                   hyp=e[key]["hyp"]) for e in a]
        rp = [e["f0_median_hz"] for e in it if e.get("f0_median_hz")]
        if rp:
            d["rapper_f0_median_hz"] = round(float(np.median(rp)), 1)
        s[kind] = d
    st = rep["stems"]
    s["stems"] = dict(max_peak_dbfs=max(v["peak_dbfs"] for v in st.values()), clipped_samples=sum(v["clipped_samples"] for v in st.values()),
                      outside_score_regions_peak_dbfs=max(v["outside_score_regions_peak_dbfs"] for v in st.values()),
                      gain_applied_db={k: v["gain_applied_db"] for k, v in st.items() if v["gain_applied_db"] != 0.0},
                      dc_re_active_rms_db={k: v["dc_re_active_rms_db"] for k, v in st.items() if not v["silent"]},
                      below_30hz_peak_dbfs={k: v["below_30hz_peak_dbfs"] for k, v in st.items() if not v["silent"]},
                      silent=[k for k, v in st.items() if v["silent"]], start_level_worst=rep.get("start_level_worst"))
    s["render_seconds"] = rep["render_seconds"]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-asr", action="store_true", help="skip the report's intelligibility checks (the stems do not change)")
    ap.add_argument("--only", default=None, help="comma-separated ids (r4,h12,a8_14,s1_0.0): partial debug render")
    a = ap.parse_args()
    Vocals(asr=not a.no_asr, only=a.only.split(",") if a.only else None).run()


if __name__ == "__main__":
    main()
