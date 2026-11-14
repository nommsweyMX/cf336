"""Neural speech-to-singing engine built on a locally re-wired Kokoro v1.0 graph.

Kokoro predicts, per phoneme token, a duration in 25 ms alignment frames, then an F0 curve and an
energy curve (N) at 12.5 ms, and renders audio with its neural decoder. kokoro_surgery.py exposes
those quantities as graph inputs, so instead of time-stretching/pitch-shifting finished speech
(WORLD) we *drive the decoder directly*:
  1. analyse the phrase as natural speech      -> natural phoneme durations
  2. syllabify the phonemes against the notes  -> onset / nucleus / coda tokens per note
  3. plan frame durations so every vowel nucleus starts on its note (consonants before it)
  4. re-run the prosody predictor on that plan -> natural voicing decisions + energy
  5. replace F0 by the melody, sustain energy through vowels, decode the SOURCE plan (every vowel capped
     at ~175 ms, rests capped) -> natural articulation, no re-articulated long vowels
  6. closed-loop timing: measure where the decoder really starts each vowel (sustained periodicity, the
     same detector the audit uses) and where it really ends before an unvoiced consonant
  7. WORLD analysis of the decoder output, time-mapped onto the score: measured vowel onsets land exactly
     on the note times, consonants keep their measured length just before them, held vowels are filled by
     slowly stretching the stable middle and pulling it towards its mean spectrum; re-pitched to the exact
     melody (glides, short scoops, vibrato, drift); aperiodicity capped below ~3 kHz only, with a breath
     floor above it (no pulse-train buzz); unvoiced consonants come from the (time-mapped) decoder audio.
Spoken lines use steps 1 (+uniform duration compression when they must fit a slot) and 4 directly.
"""
import hashlib
import os

import numpy as np
import onnxruntime as rt
from scipy.signal import butter, sosfiltfilt

# Working folder for models, render cache and stems (git-ignored). Override with WHY_LLARTA_WORK.
SCRATCH = os.environ.get("WHY_LLARTA_WORK", os.path.join(os.path.dirname(os.path.abspath(__file__)), "work"))
MODEL_DIR = os.environ.get("VOX_MODEL_DIR", os.path.join(SCRATCH, "models"))
KSR = 24000          # Kokoro sample rate
HOP = 600            # samples per alignment frame
FR = HOP / KSR       # 0.025 s
F0R = FR / 2         # 0.0125 s per F0/N point
PAD_LEAD = int(os.environ.get("VOX_PAD_LEAD", 20))   # the decoder needs ~12-16 frames of warm-up before it can sound
PAD_TAIL = int(os.environ.get("VOX_PAD_TAIL", 16))
# Kokoro renders every phoneme ~3 alignment frames (75 ms) BEFORE its aligned slot (the official kokoro
# pipeline also offsets its word timestamps by -3 frames). Plans are made in acoustic time and the
# alignment is delayed by LAG frames, so the sound lands where it was planned.
LAG = int(os.environ.get("VOX_LAG", 3))
GATE_RELEASE = 0.03
NEURAL_TOP_M = 60      # highest note (MIDI) the decoder itself is asked to sing, male voices
NEURAL_TOP_F = 68
ONSET_MIN = int(os.environ.get("VOX_ONSET_MIN", 2))     # min frames for an obstruent onset consonant
SUNG_VOWEL_SUB = {"ᵻ": "i"}                             # reduced vowels sung as full vowels
SEG_FRAMES = int(os.environ.get("VOX_SEG", 5))         # max frames per vowel token before it is re-articulated
SEG_PATTERN = os.environ.get("VOX_SEG_PATTERN", "rep")
NUC_SRC = int(os.environ.get("VOX_NUC_SRC", 7))       # longest vowel (frames) the decoder sings; held in WORLD
GAP_SRC = 4                                             # longest rest (frames) the decoder is given
UNV_OBS = set("ptkfθsʃhxç")
ONSET_MAX_S = 0.085
XOVER_HZ = 1200.0
CONS_LIFTER_MS = float(os.environ.get("VOX_CONS_LIFTER_MS", 2.5))
HOLD_MEAN = float(os.environ.get("VOX_HOLD_MEAN", 0.7))
SCOOP_MIN_NOTE = 0.25                                   # s; shorter notes start on pitch
AP_CAP_HI_HZ = (2500.0, 3500.0)                         # vowel aperiodicity cap fades out over this band
AP_FLOOR_HZ = tuple(float(v) for v in os.environ.get("VOX_AP_FLOOR_HZ", "2500,5000").split(","))                          # voiced-frame aperiodicity floor ramps up over this band
AP_FLOOR_MAX = float(os.environ.get("VOX_AP_FLOOR", 0.95))

VOWELS = set("aeiouyɑɐɒæɔəɘɚɛɜɝɞɤɨɪʊʉʌʏøœɯɵᵻ")
LENGTH = "ː"
STRESS = set("ˈˌ")
PUNCT = set(",.!?;:—…\"()¡¿-–")
SONORANT = set("mnŋlɹɾjwɲʎʝrɫʋ")
VOICED_OBS = set("bdɡvðzʒβɣ")
DIPH_EN = {"aɪ", "eɪ", "oʊ", "aʊ", "ɔɪ"}


def cls_of(c):
    if c in VOWELS:
        return "V"
    if c == LENGTH:
        return "L"
    if c in STRESS:
        return "S"
    if c == " ":
        return "W"
    if c in PUNCT:
        return "P"
    return "C"


def _interp_cubic(x, pos):
    """Catmull-Rom interpolation of x at fractional sample positions pos."""
    x = np.asarray(x, np.float64)
    n = len(x)
    i = np.floor(pos).astype(np.int64)
    f = pos - i
    xp = np.pad(x, (2, 3))
    def g(o):
        return xp[np.clip(i + o + 2, 0, n + 4)]
    p0, p1, p2, p3 = g(-1), g(0), g(1), g(2)
    return p1 + 0.5 * f * (p2 - p0 + f * (2 * p0 - 5 * p1 + 4 * p2 - p3 + f * (3 * (p1 - p2) + p3 - p0)))


def midi_hz(m):
    return 440.0 * 2.0 ** ((np.asarray(m, dtype=np.float64) - 69.0) / 12.0)


# ----------------------------------------------------------------------------- syllabification
def _onset_count(cs):
    """How many of the trailing consonants (chars) of a word-internal cluster start the next syllable."""
    n = len(cs)
    if n == 0:
        return 0
    k = 1
    if n >= 2 and cs[-2:] in ("tʃ", "dʒ"):
        k = 2
        return k
    if n >= 2:
        a, b = cs[-2], cs[-1]
        if (b in "ɹlɾ" and a in "pbtdkɡfvθβɣ") or (b in "jw") or (a == "s" and b in "ptkmnlw"):
            k = 2
            if n >= 3 and cs[-3] == "s" and a in "ptk" and b in "ɹlj":
                k = 3
    return k


def syllabify(chars, notes, lang):
    """chars: phoneme chars (no pads). notes: list of note dicts (T, D, midi).
    Returns list of syllables {onset, nucleus, coda, gap, notes} (indices into chars) and a log."""
    log = []
    runs, i = [], 0
    while i < len(chars):
        if chars[i] in VOWELS:
            j, idx = i, []
            while j < len(chars) and (chars[j] in VOWELS or chars[j] == LENGTH):
                idx.append(j)
                j += 1
            runs.append(idx)
            i = j
        else:
            i += 1
    nuclei = []
    for run in runs:
        vs = [x for x in run if chars[x] in VOWELS]
        units, p = [], 0
        while p < len(vs):
            if lang == "en" and p + 1 < len(vs) and vs[p + 1] == vs[p] + 1 + (chars[vs[p] + 1] == LENGTH) \
                    and chars[vs[p]] + chars[vs[p + 1]] in DIPH_EN:
                units.append([vs[p], vs[p + 1]])
                p += 2
            else:
                units.append([vs[p]])
                p += 1
        for x in run:
            if chars[x] == LENGTH:
                best = max((u for u in units if u[0] < x), key=lambda u: u[0])
                best.append(x)
        for u in units:
            u.sort()
        nuclei.extend(units)
    N = len(notes)
    # reconcile vowel-nucleus count with the note count
    while len(nuclei) < N:
        cand = [k for k, u in enumerate(nuclei) if sum(chars[x] in VOWELS for x in u) >= 2]
        if not cand:
            break
        k = cand[0]
        u = nuclei[k]
        v = [x for x in u if chars[x] in VOWELS]
        a = [x for x in u if x < v[1]]
        b = [x for x in u if x >= v[1]]
        nuclei[k:k + 1] = [a, b]
        log.append(f"split diphthong at {k}")
    while len(nuclei) > N:
        # merge two adjacent nuclei with nothing between them, else the weakest vowel into its left neighbour
        merged = False
        for k in range(len(nuclei) - 1):
            if nuclei[k + 1][0] == nuclei[k][-1] + 1:
                nuclei[k:k + 2] = [nuclei[k] + nuclei[k + 1]]
                merged = True
                log.append(f"merged adjacent vowels at {k}")
                break
        if not merged:
            weak = [k for k in range(1, len(nuclei)) if chars[nuclei[k][0]] in "əɐᵻɪ"]
            k = weak[-1] if weak else len(nuclei) - 1
            nuclei[k - 1:k + 1] = [list(range(nuclei[k - 1][0], nuclei[k][-1] + 1))]
            log.append(f"merged weak syllable {k} into previous")
    # note groups: if fewer nuclei than notes -> melisma on the longest notes' syllables
    groups = [[n] for n in notes[:len(nuclei)]]
    if len(nuclei) < N:
        groups = [[] for _ in nuclei]
        extra = N - len(nuclei)
        # give the extra notes to the last syllables (sustained word endings), one each
        owner = list(range(len(nuclei))) + [len(nuclei) - 1] * extra
        owner.sort()
        for n, o in zip(notes, owner):
            groups[o].append(n)
        log.append(f"melisma: {extra} extra note(s)")
    sylls = []
    for k, nu in enumerate(nuclei):
        sylls.append(dict(nucleus=nu, onset=[], coda=[], gap=[], notes=groups[k]))
    # consonants before first nucleus
    pre = list(range(0, nuclei[0][0]))
    sylls[0]["onset"] = [x for x in pre if cls_of(chars[x]) == "C"]
    sylls[0]["lead"] = [x for x in pre if cls_of(chars[x]) != "C"]
    for k in range(len(nuclei) - 1):
        seg = list(range(nuclei[k][-1] + 1, nuclei[k + 1][0]))
        bnd = [x for x in seg if cls_of(chars[x]) in "WP"]
        if bnd:
            sylls[k]["coda"] = [x for x in seg if x < bnd[0] and cls_of(chars[x]) == "C"]
            sylls[k + 1]["onset"] = [x for x in seg if x > bnd[-1] and cls_of(chars[x]) == "C"]
        else:
            cons = [x for x in seg if cls_of(chars[x]) == "C"]
            n_on = _onset_count("".join(chars[x] for x in cons))
            sylls[k]["coda"] = cons[:len(cons) - n_on]
            sylls[k + 1]["onset"] = cons[len(cons) - n_on:]
        used = set(sylls[k]["coda"]) | set(sylls[k + 1]["onset"])
        sylls[k]["gap"] = [x for x in seg if x not in used]
    post = list(range(nuclei[-1][-1] + 1, len(chars)))
    sylls[-1]["coda"] = [x for x in post if cls_of(chars[x]) == "C"]
    sylls[-1]["gap"] = [x for x in post if cls_of(chars[x]) != "C"]
    return sylls, log


# ----------------------------------------------------------------------------- engine
class Engine:
    def __init__(self, model_dir=MODEL_DIR, threads=4, cache_dir=None):
        from kokoro_onnx import Kokoro
        from kokoro_surgery import build
        self.k = Kokoro(os.path.join(model_dir, "kokoro-v1.0.onnx"), os.path.join(model_dir, "voices-v1.0.bin"))
        paths = build(os.path.join(model_dir, "kokoro-v1.0.onnx"), os.path.join(model_dir, "ctl"))
        so = rt.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.s = {n: rt.InferenceSession(p, so, providers=["CPUExecutionProvider"]) for n, p in paths.items()}
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        self._ph = {}
        self._ana = {}

    # --- basics
    def phonemes(self, text, lang):
        key = (text, lang)
        if key not in self._ph:
            ph = self.k.tokenizer.phonemize(text, "en-us" if lang == "en" else "es")
            self._ph[key] = " ".join(ph.split())
        return self._ph[key]

    def tokens(self, ph):
        toks = self.k.tokenizer.tokenize(ph)
        chars = [c for c in ph if c in self.k.tokenizer.vocab]
        assert len(toks) == len(chars)
        return toks, chars

    def style(self, voice, ntok, blend=None):
        v = self.k.voices[voice][min(ntok, 510) - 1].astype(np.float32)
        if blend:  # (other_voice, amount) small timbre variation for doubles
            o = self.k.voices[blend[0]][min(ntok, 510) - 1].astype(np.float32)
            v = (1 - blend[1]) * v + blend[1] * o
        return v

    def _feed(self, toks, style, speed=1.0):
        return {"tokens": np.array([[0, *toks, 0]], np.int64), "style": style,
                "speed": np.array([speed], np.float32)}

    def _cached(self, tag, arrays, fn):
        if not self.cache_dir:
            return fn()
        h = hashlib.sha1(tag.encode())
        for a in arrays:
            a = np.ascontiguousarray(a)
            h.update(str(a.dtype).encode() + str(a.shape).encode())
            h.update(a.tobytes())
        path = os.path.join(self.cache_dir, h.hexdigest() + ".npz")
        if os.path.exists(path):
            z = np.load(path)
            return [z[f"a{i}"] for i in range(len(z.files))]
        out = fn()
        np.savez(path, **{f"a{i}": o for i, o in enumerate(out)})
        return out

    def analyse(self, toks, style, speed=1.0):
        feed = self._feed(toks, style, speed)
        out = self._cached("ana", [feed["tokens"], style, feed["speed"]],
                           lambda: [np.asarray(o) for o in self.s["ana"].run(None, feed)])
        a, d, f0, n = out
        return a.ravel(), d.ravel().astype(np.int64), f0.ravel(), n.ravel()

    def run_dur(self, toks, style, dur):
        feed = self._feed(toks, style)
        feed["dur_in"] = np.asarray(dur, np.float32)[None]
        out = self._cached("dur", [feed["tokens"], style, feed["dur_in"]],
                           lambda: [np.asarray(o) for o in self.s["dur"].run(None, feed)])
        a, f0, n = out
        return a.ravel(), f0.ravel(), n.ravel()

    def run_sing(self, toks, style, dur, f0, n):
        feed = self._feed(toks, style)
        feed["dur_in"] = np.asarray(dur, np.float32)[None]
        feed["f0_in"] = np.asarray(f0, np.float32)[None]
        feed["n_in"] = np.asarray(n, np.float32)[None]
        out = self._cached("sing", [feed["tokens"], style, feed["dur_in"], feed["f0_in"], feed["n_in"]],
                           lambda: [np.asarray(self.s["sing"].run(None, feed)[0])])
        return out[0].ravel()

    # ------------------------------------------------------------------------- singing
    def plan_sung(self, text, lang, notes, ana_voice, ana_speed=1.0, tail=None, vowel_sub=True):
        """notes: list of (T_abs, D_sec, midi). Returns a plan dict (shared by every voice singing it).
        tail: optional continuation text appended as zero-duration tokens: it is never heard, but it keeps
        the last sung syllable from being utterance-final for the text encoder (avoids creaky phrase-final
        devoicing on held notes)."""
        ph = self.phonemes(text, lang)
        if vowel_sub:
            ph = "".join(SUNG_VOWEL_SUB.get(c, c) for c in ph)
            if lang == "en":   # sung diction: word-final flaps become real t's ("it asks", not "i-dasks")
                ph = ph.replace("ɾ ", "t ")
        _, chars_main = self.tokens(ph)
        if tail:
            ph = ph + " " + self.phonemes(tail, lang)
        toks, chars = self.tokens(ph)
        st = self.style(ana_voice, len(toks))
        _, nat, _, _ = self.analyse(toks, st, ana_speed)
        nat = nat[1:-1]  # drop pads -> aligned with chars
        nd = [dict(T=T, D=D, m=m) for (T, D, m) in notes]
        sylls, slog = syllabify(chars_main, nd, lang)
        nS = len(sylls)
        Ts = [s["notes"][0]["T"] for s in sylls]
        # --- consonant frame budgets
        on_f, co_f = [], []
        for k, s in enumerate(sylls):
            on = [int(np.clip(nat[x], ONSET_MIN if chars[x] not in SONORANT else 1, 3)) for x in s["onset"]]
            span_prev = (Ts[k] - Ts[k - 1]) if k > 0 else 1.0
            cap = max(len(on), min(5, int(round(0.38 * span_prev / FR))))
            while sum(on) > cap:
                on[int(np.argmax(on))] -= 1
            on_f.append(on)
            co = [int(np.clip(nat[x], 1, 3)) for x in s["coda"]]
            last = s["notes"][-1]
            nxt = Ts[k + 1] if k + 1 < nS else last["T"] + last["D"]
            span = nxt - s["notes"][0]["T"]
            ccap = max(len(co), min(3, int(round(0.3 * span / FR))))
            while sum(co) > ccap:
                co[int(np.argmax(co))] -= 1
            co_f.append(co)
        pad_lead = PAD_LEAD
        # --- choose the frame phase that best fits all vowel onsets onto the 25 ms grid
        lead_s = (pad_lead + sum(on_f[0])) * FR
        best = None
        for ph_ms in np.arange(0.0, 25.0, 0.5):
            R0 = Ts[0] - lead_s - ph_ms / 1000.0
            vf = [int(round((T - R0) / FR)) for T in Ts]
            err = [R0 + v * FR - T for v, T in zip(vf, Ts)]
            score = max(abs(e) for e in err)
            if best is None or score < best[0] - 1e-9:
                best = (score, R0, vf, err)
        _, R0, VF, verr = best
        # --- assign frames to every token
        ntok = len(chars)
        dur = np.zeros(ntok + 2, np.int64)
        dur[0] = VF[0] - sum(on_f[0])
        assert dur[0] >= 3, dur[0]
        rest_mask = []   # (start_frame, end_frame) silent regions, to gate after synthesis
        nuc_frames = []  # (k, start, end)
        nuc_alloc = []   # per syllable {token: frames} of its nucleus
        gaps = []        # (k, token that carries the rest, frames)
        coda_used = []   # per syllable coda frames actually given
        for k, s in enumerate(sylls):
            for x, f in zip(s["onset"], on_f[k]):
                dur[x + 1] = f
        cur = dur[0]
        for k, s in enumerate(sylls):
            cur = VF[k]
            last = s["notes"][-1]
            end_note = int(round((last["T"] + last["D"] - R0) / FR))
            if k + 1 < nS:
                next_on = VF[k + 1] - sum(on_f[k + 1])
                rest = (Ts[k + 1] - (last["T"] + last["D"])) >= 0.12
                end_k = min(end_note, next_on) if rest else next_on
            else:
                next_on = None
                rest = True
                end_k = end_note
            co = list(co_f[k])
            nuc = end_k - cur - sum(co)
            while nuc < 2 and sum(co) > len(co):
                co[int(np.argmax(co))] -= 1
                nuc = end_k - cur - sum(co)
            if nuc < 1:
                nuc = 1
            # distribute nucleus frames
            vs = [x for x in s["nucleus"] if chars[x] in VOWELS]
            ls = [x for x in s["nucleus"] if chars[x] == LENGTH]
            alloc = {}
            rem = nuc
            if len(vs) >= 2:
                off = int(np.clip(round(0.22 * nuc), 1, 5)) if nuc >= 3 else 0
                alloc[vs[-1]] = off
                rem -= off
                for x in vs[1:-1]:
                    alloc[x] = 0
            if ls and rem >= 2:
                w0 = max(1, nat[vs[0]])
                wl = max(1, sum(nat[x] for x in ls))
                lf = int(round(rem * wl / (w0 + wl)))
                lf = int(np.clip(lf, 1, rem - 1))
                alloc[vs[0]] = rem - lf
                for j, x in enumerate(ls):
                    alloc[x] = lf if j == 0 else 0
            else:
                alloc[vs[0]] = rem
                for x in ls:
                    alloc[x] = 0
            for x, f in alloc.items():
                dur[x + 1] = f
            nuc_alloc.append(dict(alloc))
            nuc_frames.append((k, cur, cur + nuc))
            cur += nuc
            for x, f in zip(s["coda"], co):
                dur[x + 1] = f
            coda_used.append(list(co))
            cur += sum(co)
            gap = (next_on - cur) if next_on is not None else 0
            if gap > 0:
                rest_mask.append((cur, next_on))
                tgt = [x for x in s["gap"] if cls_of(chars[x]) in "WP"]
                if tgt:
                    gtok = tgt[0]
                elif sylls[k + 1]["onset"]:
                    gtok = sylls[k + 1]["onset"][0]
                else:
                    gtok = s["nucleus"][-1]
                dur[gtok + 1] += gap
                gaps.append((k, gtok, int(gap)))
                cur = next_on
            elif gap < 0:
                raise RuntimeError(f"negative gap at syllable {k}: {gap}")
        tail_pad = PAD_TAIL
        rest_mask.append((cur, cur + tail_pad + 2 + LAG))
        dur[-1] = tail_pad + 2
        assert int(dur.sum()) == cur + tail_pad + 2
        lead_frames = int(dur[0])
        dur[0] += LAG
        plan = dict(text=text, lang=lang, ph=ph, toks=toks, chars=chars, sylls=sylls, slog=slog, R0=R0, tail=tail,
                    VF=VF, vowel_err=verr, dur=dur, rest_mask=rest_mask, nuc_frames=nuc_frames,
                    notes=nd, onset_frames=on_f, coda_frames=co_f, ntok0=len(toks), lead_frames=lead_frames,
                    nuc_alloc=nuc_alloc, gaps=gaps, chars0=chars, coda_used=coda_used)
        return plan

    def expand(self, plan, seg=None, pattern=None):
        """Split long vowel runs into repeated tokens of <= seg frames.
        The decoder treats long runs of identical aligned features as pauses (breathy noise / fade);
        re-articulating the same vowel token every ~125 ms keeps it phonated without audible re-attacks."""
        seg = seg or SEG_FRAMES
        pattern = pattern or SEG_PATTERN
        vocab = self.k.tokenizer.vocab
        toks, chars, dur = [], [], [0]
        dur = [int(plan["dur"][0])]
        for i, c in enumerate(plan["chars"]):
            f = int(plan["dur"][i + 1])
            if cls_of(c) in "VL" and f > seg + 2:
                r = int(np.ceil(f / seg))
                parts = [f // r + (1 if j < f % r else 0) for j in range(r)]
                if pattern == "rep" or cls_of(c) == "L":
                    seq = [c] * r
                elif pattern == "alt":
                    seq = [c if j % 2 == 0 else LENGTH for j in range(r)]
                else:  # "len": vowel then length marks
                    seq = [c] + [LENGTH] * (r - 1)
                for cc, ff in zip(seq, parts):
                    toks.append(vocab[cc])
                    chars.append(cc)
                    dur.append(ff)
            else:
                toks.append(plan["toks"][i])
                chars.append(c)
                dur.append(f)
        dur.append(int(plan["dur"][-1]))
        out = dict(plan)
        out.update(toks=toks, chars=chars, dur=np.array(dur, np.int64), chars0=plan["chars"], dur0=plan["dur"])
        return out

    def pitch_curve(self, plan, t, detune_cents=0.0, octave=0, seed=0, vib_depth=0.22, scoop=0.4,
                    expressive=True):
        """Target pitch (MIDI float) at absolute song times t (s).
        Legato note changes glide over ~55 ms around the vowel onset; notes after a rest scoop up from
        -40 cents over 50 ms; notes >= 0.35 s get 5.2-5.6 Hz vibrato (+-22 cents) fading in after 150 ms;
        a slow +-6 cent drift keeps it from sounding like a sine. expressive=False -> glides only."""
        rng = np.random.default_rng(seed)
        notes = [n for s in plan["sylls"] for n in s["notes"]]
        T = np.array([n["T"] for n in notes])
        D = np.array([n["D"] for n in notes])
        M = np.array([n["m"] for n in notes], dtype=np.float64)
        idx = np.clip(np.searchsorted(T, t, side="right") - 1, 0, len(T) - 1)
        pitch = M[idx].copy()
        for j in range(1, len(T)):
            legato = T[j] - (T[j - 1] + D[j - 1]) < 0.12
            if legato and M[j] != M[j - 1]:
                a, b = T[j] - 0.035, T[j] + 0.02
                w = np.clip((t - a) / (b - a), 0, 1)
                w = w * w * (3 - 2 * w)
                sel = (t >= a) & (t < b + 0.01)
                pitch[sel] = M[j - 1] + (M[j] - M[j - 1]) * w[sel]
        if expressive:
            for j in range(len(T)):
                legato = j > 0 and T[j] - (T[j - 1] + D[j - 1]) < 0.12
                depth = 0.12 if legato else scoop
                # the scoop takes at most 22% of the note (<= 50 ms) and is skipped on notes < half a beat
                sl = min(0.05, 0.22 * D[j])
                if D[j] < SCOOP_MIN_NOTE:
                    depth, sl = 0.0, 0.01
                w = np.clip((t - T[j]) / sl, 0, 1)
                sel = (t >= T[j] - 0.05) & (t < T[j] + sl)
                pitch[sel] -= depth * (1 - (w[sel] * w[sel] * (3 - 2 * w[sel])))
                rate = 5.2 + 0.4 * rng.random()
                if D[j] >= 0.35:
                    tt = t - T[j]
                    ramp = np.clip((tt - 0.15) / 0.22, 0, 1) * np.clip((D[j] - tt) / 0.06, 0, 1)
                    sel = (tt >= 0) & (tt < D[j])
                    pitch[sel] += vib_depth * ramp[sel] * np.sin(2 * np.pi * rate * tt[sel])
            ph1, ph2 = rng.random(2) * 2 * np.pi
            pitch += 0.04 * np.sin(2 * np.pi * 0.7 * t + ph1) + 0.025 * np.sin(2 * np.pi * 1.9 * t + ph2)
        pitch += detune_cents / 100.0 + 12 * octave
        return pitch

    def render_neural(self, plan, voice, neural_shift=-12, style_shift=0, blend=None, energy_boost=0.0):
        """Neural pass: the decoder sings the melody transposed into the voice's comfortable speaking
        range (default an octave down) with exact phoneme timing. Returns audio + per-12.5ms masks."""
        toks, chars, dur = plan["toks"], plan["chars"], plan["dur"]
        nt = int(np.clip(plan.get("ntok0", len(toks)) + style_shift, 1, 510))
        st = self.style(voice, nt, blend)
        _, f0p, n_p = self.run_dur(toks, st, dur)
        nf = len(f0p)
        assert nf == 2 * int(dur.sum()), (nf, dur.sum())
        tok_of = np.repeat(np.arange(len(dur)), 2 * dur)
        tok_of = tok_of[np.minimum(np.arange(nf) + 2 * LAG, nf - 1)]   # token sounding at each F0 point
        cls = np.array(["X"] + [cls_of(c) for c in chars] + ["X"])
        chs = np.array(["<"] + list(chars) + [">"])
        nucl = np.zeros(nf, bool)
        for (k, a, b) in plan["nuc_frames"]:
            nucl[2 * a:2 * b] = True
        silent = np.zeros(nf, bool)
        for (a, b) in plan["rest_mask"]:
            silent[2 * a:2 * b] = True
        c = cls[tok_of]
        ch = chs[tok_of]
        son = np.isin(ch, list(SONORANT))
        voiced = (nucl | (c == "V") | (c == "L") | son | (f0p > 45.0)) & (c != "X") & ~silent
        unv_obs = (c == "C") & ~son & ~np.isin(ch, list(VOICED_OBS))
        voiced &= ~(unv_obs & (f0p <= 45.0))
        t = plan["R0"] + (np.arange(nf) + 0.5) * F0R
        pitch = self.pitch_curve(plan, t, expressive=False) + neural_shift
        f0 = np.where(voiced, midi_hz(pitch), 0.0).astype(np.float32)
        n = n_p.astype(np.float64).copy()
        allnuc = n[nucl]
        pref = np.percentile(allnuc, 80) if allnuc.size else 0.0
        for (k, a, b) in plan["nuc_frames"]:
            a2, b2 = 2 * a, 2 * b
            if b2 - a2 < 4:
                continue
            ref = max(np.percentile(n[a2:b2], 80), pref - 1.5) + energy_boost
            lo, hi = a2 + 2, b2 - (3 if k == len(plan["sylls"]) - 1 else 1)
            n[lo:hi] = np.maximum(n[lo:hi], ref)
        n = np.convolve(np.pad(n, 1, mode="edge"), np.ones(3) / 3, mode="valid")
        audio = self.run_sing(toks, st, dur, f0, n.astype(np.float32))
        return dict(audio=audio.astype(np.float64), f0_fed=f0, voiced=voiced, nucl=nucl, f0_pred=f0p, n=n,
                    neural_shift=neural_shift)

    def gate(self, plan, audio, sr=KSR):
        g = np.ones(len(audio))
        hop = HOP * sr / KSR
        r = int(0.012 * sr)
        rel = int(GATE_RELEASE * sr)
        for (a, b) in plan["rest_mask"]:
            s0, s1 = int(a * hop), min(len(audio), int(b * hop))
            s0 = min(s1, s0 + rel)          # let the coda's release ring out before the rest
            g[s0:s1] = 0.0
            if s0 > 0:
                lo = max(0, s0 - r)
                g[lo:s0] = np.minimum(g[lo:s0], np.linspace(1, 0, s0 - lo))
            if s1 < len(audio):
                e = min(len(audio), s1 + r)
                g[s1:e] = np.minimum(g[s1:e], np.linspace(0, 1, e - s1))
        lead = int(max(0, int(plan["lead_frames"]) - 2) * hop)   # leave room for the first consonant
        g[:lead] = 0.0
        r2 = int(0.015 * sr)
        g[lead:lead + r2] = np.minimum(g[lead:lead + r2], np.linspace(0, 1, r2))
        return audio * g

    def level_syllables(self, plan, y, amount=0.8, max_cut=9.0, max_boost=12.0):
        """Vocal-rider: speech declination makes later syllables 10-15 dB quieter than the first; a sung pop
        line is even. Each syllable (onset..coda) is moved `amount` of the way to the phrase's median
        nucleus level, with 15 ms crossfades between syllable gains."""
        lv = []
        for (k, a, b) in plan["nuc_frames"]:
            seg = y[a * HOP:b * HOP]
            lv.append(10 * np.log10(np.mean(seg ** 2) + 1e-12) if len(seg) else -120.0)
        lv = np.array(lv)
        ok = lv > -100
        if ok.sum() < 2:
            return y
        tgt = np.median(lv[ok])
        gdb = np.where(ok, np.clip(amount * (tgt - lv), -max_cut, max_boost), 0.0)
        # gain breakpoints at syllable starts (first onset consonant)
        starts = []
        for k, s_ in enumerate(plan["sylls"]):
            starts.append((plan["VF"][k] - sum(plan["onset_frames"][k])) * HOP)
        g = np.zeros(len(y))
        for k in range(len(starts)):
            a = 0 if k == 0 else starts[k]
            b = len(y) if k + 1 == len(starts) else starts[k + 1]
            g[a:b] = gdb[k]
        r = int(0.015 * KSR)
        g = np.convolve(np.pad(g, r, mode="edge"), np.ones(2 * r + 1) / (2 * r + 1), mode="valid")
        return y * 10 ** (g / 20)

    def world_analyse(self, nr):
        """WORLD analysis of a neural render, seeded with the F0 we asked the decoder for."""
        import pyworld as pw
        x = nr["audio"]
        fp = 5.0
        t5 = np.arange(int(len(x) / KSR / (fp / 1000.0)) + 1) * (fp / 1000.0)
        i12 = np.clip((t5 / F0R).astype(int), 0, len(nr["f0_fed"]) - 1)
        fed = nr["f0_fed"][i12].astype(np.float64)
        f0 = fed.copy()
        v = fed > 0
        if v.any():
            ref = pw.stonemask(x, fed, t5, KSR)
            ok = v & (ref > 0) & (np.abs(np.log2(np.maximum(ref, 1) / np.maximum(fed, 1))) < 0.12)
            f0[ok] = ref[ok]
        sp = pw.cheaptrick(x, f0, t5, KSR)
        ap = pw.d4c(x, f0, t5, KSR)
        return dict(t=t5, f0=f0, sp=sp, ap=ap, v=v, nucl=nr["nucl"][i12], i12=i12)

    def world_synth(self, plan, wa, detune_cents=0.0, octave=0, seed=0, vib_depth=0.22, ap_cap=0.35,
                    time_offset=0.0, formant_shift=1.0):
        """Resynthesise at the exact target melody (+ expressive curve), keeping the decoder's spectral
        envelope (formants) and articulation. ap is capped inside vowel nuclei for a clean, tuned tone."""
        import pyworld as pw
        t = wa["t"]
        pitch = self.pitch_curve(plan, plan["R0"] + t, detune_cents, octave, seed, vib_depth=vib_depth)
        f0 = np.where(wa["v"], midi_hz(pitch), 0.0)
        ap = wa["ap"].copy()
        if ap_cap is not None:
            # clean, tuned vowel tone below ~3 kHz; above it keep the decoder's own (d4c) aperiodicity, plus a
            # breath floor, so the top band is not a pure pulse train (the 'buzzy/metallic' WORLD sound)
            nb = ap.shape[1]
            fq = np.arange(nb) / (nb - 1) * KSR / 2
            lo, hi = AP_CAP_HI_HZ
            cap = ap_cap + (1 - ap_cap) * np.clip((fq - lo) / (hi - lo), 0, 1)
            floor = AP_FLOOR_MAX * np.clip((fq - AP_FLOOR_HZ[0]) / (AP_FLOOR_HZ[1] - AP_FLOOR_HZ[0]), 0, 1)
            ap[wa["nucl"]] = np.minimum(ap[wa["nucl"]], cap)
            ap[wa["v"]] = np.maximum(ap[wa["v"]], floor)
        sp = wa["sp"]
        if CONS_LIFTER_MS > 0:
            # voiced consonants (and vowel edges) were analysed where the decoder is not cleanly periodic at the
            # f0 it was given, so their envelope keeps ripple at the DECODER's (transposed) harmonics, which the
            # resynthesis would turn into a second, wrong pitch. Cepstrally smooth those frames' envelopes.
            sel = ~wa["nucl"]
            if sel.any():
                nb = sp.shape[1]
                c = np.fft.irfft(np.log(sp[sel] + 1e-16), axis=1)
                q = int(CONS_LIFTER_MS / 1000.0 * KSR)
                lift = np.zeros(c.shape[1])
                lift[:q] = 1.0
                lift[-q + 1:] = 1.0
                sp = sp.copy()
                sp[sel] = np.exp(np.fft.rfft(c * lift, axis=1).real[:, :nb])
        if formant_shift != 1.0:
            nb = sp.shape[1]
            src = np.clip(np.arange(nb) / formant_shift, 0, nb - 1)
            sp = np.stack([np.exp(np.interp(src, np.arange(nb), np.log(row + 1e-16))) for row in sp])
        y = pw.synthesize(f0, np.ascontiguousarray(sp), np.ascontiguousarray(ap), KSR, 5.0)
        return y, f0

    # ------------------------------------------------------------------------- closed-loop timing
    def bounds(self, plan, nuc_len=None):
        """Per-syllable frame boundaries (acoustic frames from the render start): onset start c, vowel onset v,
        nucleus end z, coda end g, and the rest after it. nuc_len overrides nucleus lengths, gap_cap caps rests."""
        out = []
        sylls = plan["sylls"]
        c = plan["VF"][0] - sum(plan["onset_frames"][0])
        gapmap = {k: g for (k, _, g) in plan["gaps"]}
        for k in range(len(sylls)):
            v = c + sum(plan["onset_frames"][k])
            nl = nuc_len[k] if nuc_len is not None else plan["nuc_frames"][k][2] - plan["nuc_frames"][k][1]
            z = v + nl
            g = z + sum(plan["coda_used"][k])
            gap = gapmap.get(k, 0)
            out.append(dict(c=c, v=v, z=z, g=g, gap=gap))
            c = g + gap
        return out

    def source_plan(self, plan):
        """The plan the decoder actually sings: identical consonants, but every vowel nucleus capped at NUC_SRC
        frames (a natural ~175 ms vowel; it is held to the note length later in the WORLD domain, so the
        decoder never has to re-articulate a long vowel) and rests capped at GAP_SRC frames."""
        if "src" in plan:
            return plan["src"]
        dur = plan["dur"].copy()
        nalloc, nlen, offg = [], [], []
        chars = plan["chars"]
        for k, alloc in enumerate(plan["nuc_alloc"]):
            tot = sum(alloc.values())
            keys = list(alloc)
            vs_ = [x for x in plan["sylls"][k]["nucleus"] if chars[x] in VOWELS]
            if tot > NUC_SRC and len(plan["sylls"][k]["notes"]) == 1 and len(vs_) >= 2 and alloc.get(vs_[-1], 0) > 0:
                # diphthong: keep a real off-glide (up to 2 frames) and give the rest to the first vowel
                off = int(min(alloc[vs_[-1]], 2))
                a2 = {x: 0 for x in keys}
                a2[vs_[-1]] = off
                a2[vs_[0]] = NUC_SRC - off
            elif tot > NUC_SRC and len(plan["sylls"][k]["notes"]) == 1:
                vals = np.array([alloc[x] for x in keys], float)
                new = np.where(vals > 0, np.maximum(1, np.round(vals * NUC_SRC / tot)), 0)
                while new.sum() > NUC_SRC:
                    new[int(np.argmax(new))] -= 1
                while new.sum() < NUC_SRC:
                    new[int(np.argmax(vals - new * tot / NUC_SRC))] += 1
                a2 = {x: int(v) for x, v in zip(keys, new)}
            else:
                a2 = dict(alloc)
            for x in keys:
                dur[x + 1] -= alloc[x] - a2[x]
            nalloc.append(a2)
            nlen.append(sum(a2.values()))
            vs = [x for x in plan["sylls"][k]["nucleus"] if chars[x] in VOWELS]
            offg.append(a2.get(vs[-1], 0) if len(vs) >= 2 else 0)
        gaps = []
        for (k, gtok, g) in plan["gaps"]:
            g2 = min(g, GAP_SRC)
            dur[gtok + 1] -= g - g2
            gaps.append((k, gtok, g2))
        src = dict(plan)
        src.update(dur=dur, gaps=gaps, nuc_alloc=nalloc)
        b = self.bounds(src, nuc_len=nlen)
        src["nuc_frames"] = [(k, q["v"], q["z"]) for k, q in enumerate(b)]
        src["VF"] = [q["v"] for q in b]
        rm = [(q["g"], q["g"] + q["gap"]) for q in b if q["gap"] > 0]
        tail = int(plan["dur"][-1])
        rm.append((b[-1]["g"], b[-1]["g"] + tail + LAG))
        src["rest_mask"] = rm
        assert int(dur.sum()) == b[-1]["g"] + tail + LAG, (dur.sum(), b[-1]["g"], tail)
        # notes in source time (for the decoder's own pitch curve); rests keep a > 0.12 s gap
        sy = []
        for k, (s_, q) in enumerate(zip(plan["sylls"], b)):
            n0 = s_["notes"][0]
            T = plan["R0"] + q["v"] * FR
            D = (q["g"] - q["v"]) * FR if q["gap"] > 0 else (q["g"] + q["gap"] - q["v"]) * FR
            if q["gap"] > 0:
                D -= 0.13
            sy.append(dict(s_, notes=[dict(n0, T=T, D=max(0.02, D))] + [dict(n) for n in s_["notes"][1:]]))
        src["sylls"] = sy
        src["bnd"] = b
        src["offglide"] = offg
        plan["src"] = src
        return src

    def measurable(self, plan, k):
        """Can the vowel onset of syllable k be seen acoustically (follows silence or an unvoiced obstruent)?"""
        s_ = plan["sylls"][k]
        ch = plan["chars"]
        if s_["onset"]:
            return ch[s_["onset"][-1]] in UNV_OBS
        if k == 0:
            return True
        rest = {kk: g for (kk, _, g) in plan["gaps"]}.get(k - 1, 0) > 0
        if rest:
            return True
        cod = plan["sylls"][k - 1]["coda"]
        return bool(cod) and ch[cod[-1]] in UNV_OBS

    def after_unvoiced(self, plan, k):
        """Is the sound right before syllable k's vowel an unvoiced obstruent (its onset, or the previous coda)?"""
        s_ = plan["sylls"][k]
        ch = plan["chars"]
        if s_["onset"]:
            return ch[s_["onset"][-1]] in UNV_OBS
        if k == 0 or {kk: g for (kk, _, g) in plan["gaps"]}.get(k - 1, 0) > 0:
            return False
        cod = plan["sylls"][k - 1]["coda"]
        return bool(cod) and ch[cod[-1]] in UNV_OBS

    def measure_vowels(self, src, audio):
        """Measured vowel onset (s, source timeline) per syllable, None where not measurable."""
        import vox_onset as vo
        x16 = vo.to16(audio, KSR)
        out = []
        self._mfeat = {}
        for k, q in enumerate(src["bnd"]):
            if not self.measurable(src, k):
                out.append(None)
                continue
            T = q["v"] * FR
            m = vo.vowel_onset(x16, T, (q["z"] - q["v"]) * FR)
            ts, per, edb = vo.features(x16, T - 0.16, T + 0.14)
            _, _, en = vo.features(x16, T + 0.03, T + max(0.06, min((q["z"] - q["v"]) * FR, 0.3)))
            ref = float(np.percentile(en, 90))
            self._mfeat[k] = (ts, edb, ref)
            out.append(m if (m is not None and -0.06 <= m - T <= 0.13) else None)
        return out

    def before_unvoiced(self, plan, k):
        """Is syllable k's vowel followed directly by an unvoiced obstruent (coda, or the next onset)?"""
        s_ = plan["sylls"][k]
        ch = plan["chars"]
        if s_["coda"]:
            return ch[s_["coda"][0]] in UNV_OBS
        if k + 1 < len(plan["sylls"]) and {kk: g for (kk, _, g) in plan["gaps"]}.get(k, 0) == 0:
            on = plan["sylls"][k + 1]["onset"]
            return bool(on) and ch[on[0]] in UNV_OBS
        return False

    def measure_offsets(self, src, audio, meas):
        """Measured end of the vowel's sustained periodicity (s, source timeline) where an unvoiced consonant
        follows it; None elsewhere."""
        import vox_onset as vo
        x16 = vo.to16(audio, KSR)
        out = []
        for k, q in enumerate(src["bnd"]):
            if not self.before_unvoiced(src, k):
                out.append(None)
                continue
            mk = meas[k] if meas[k] is not None else q["v"] * FR
            z = q["z"] * FR
            ts, per, edb = vo.features(x16, mk, z + 0.12)
            ref = float(np.percentile(edb[:max(5, int((z - mk) * 1000))], 90))
            runs = vo.voiced_runs(ts, per, edb, ref, minlen=15)
            cand = [r for r in runs if r[0] <= mk + 0.03 and r[1] > mk + 0.04]
            if not cand or cand[0][1] >= ts[-1] - 0.002:
                out.append(None)
                continue
            e = float(cand[0][1])
            out.append(e if -0.06 <= e - z <= 0.10 and e > mk + 0.03 else None)
        return out

    def effective_onsets(self, src, meas, v_src, fp=0.005):
        """Where the resynthesis will actually start its vowel: first instant (sustained 20 ms) that is voiced in
        the WORLD voicing track AND has low-band energy within 18 dB of the note's sustain. WORLD makes every
        voiced frame periodic, so this (not the decoder's own, sometimes creaky, periodicity) is what lands on
        the beat in the output."""
        out = list(meas)
        for k, mk in enumerate(meas):
            if k not in self._mfeat:
                continue
            ts, edb, ref = self._mfeat[k]
            vi = v_src[np.clip(np.round(ts / fp).astype(int), 0, len(v_src) - 1)]
            ok = vi & (edb > ref - 18.0)
            j, n, hit = 0, len(ok), None
            while j < n:
                if ok[j]:
                    e = j
                    while e < n and ok[e]:
                        e += 1
                    if e - j >= 20 or e == n:
                        hit = ts[j]
                        break
                    j = e
                else:
                    j += 1
            if hit is None or hit <= ts[0] + 0.005:
                continue
            T = src["bnd"][k]["v"] * FR
            if -0.06 <= hit - T <= 0.13:
                out[k] = float(hit)
        return out

    def time_map(self, plan, src, meas, offs=None):
        """Piecewise map target time -> source time (s, render-local). Measured vowel onsets land on the note
        times; consonants keep their measured length (capped) just before them; vowels are compressed or held
        to fill the note (see render_sung); rests are stretched (they are gated anyway).
        meas: measured source vowel onsets (or None); offs: measured source voicing offsets before an unvoiced
        coda / next onset (or None)."""
        tb, sb = self.bounds(plan), src["bnd"]
        n = len(tb)
        offs = offs or [None] * n
        T = [q["v"] * FR for q in tb]
        # ---- source boundaries: onset start sc, vowel m, nucleus end sz, coda end sg
        m = []
        for k in range(n):
            mk = meas[k] if meas[k] is not None else sb[k]["v"] * FR
            zlim = max(sb[k]["z"] * FR, offs[k] or 0.0, (sb[k]["g"] - 1) * FR)
            m.append(min(mk, zlim - 0.02))
        sc, sz, sg = [], [], []
        for k in range(n):
            r = sb[k]
            z = offs[k] if offs[k] is not None else r["z"] * FR
            z = max(z, m[k] + 0.03)
            lco = (r["g"] - r["z"]) * FR
            g = max(r["g"] * FR, z + lco) if offs[k] is not None else max(r["g"] * FR, z)
            sz.append(z)
            sg.append(g)
        for k in range(n):
            c = min(sb[k]["c"] * FR, m[k])
            if k > 0:
                c = min(max(c, sg[k - 1] if sb[k - 1]["gap"] == 0 else c), m[k])
                sg[k - 1] = min(sg[k - 1], c) if sb[k - 1]["gap"] == 0 else min(sg[k - 1], c)
                sz[k - 1] = min(sz[k - 1], sg[k - 1])
            sc.append(c)
        # ---- target boundaries
        tc = []
        for k in range(n):
            lon = m[k] - sc[k]
            # a decoder that lingers on a consonant (e.g. a long stop closure) is tightened to at most
            # max(planned length, ONSET_MAX_S) so it does not eat into the previous held vowel
            lon_pl = (sb[k]["v"] - sb[k]["c"]) * FR
            lon = min(lon, max(lon_pl, ONSET_MAX_S))
            on_ch = [src["chars"][x] for x in src["sylls"][k]["onset"]]
            if on_ch and all(c_ == "h" for c_ in on_ch):
                lon = min(lon, lon_pl + 0.01)       # a breathy 'h' is weak: keep it short
            lb = 0.02 if k == 0 else T[k - 1] + max(0.04, 0.45 * (T[k] - T[k - 1]))
            tc.append(max(T[k] - lon, lb))
        t_end = float(int(plan["dur"].sum()) * FR)
        s_end = float(int(src["dur"].sum()) * FR)
        segs = [(0.0, tc[0], 0.0, sc[0], None)]
        for k in range(n):
            q = tb[k]
            lcod_s = sg[k] - sz[k]
            # a vowel that dies early (creaky/breathy phrase-final tail) does not shorten the sung vowel: the
            # coda keeps at most its planned length (+10 ms) on the target side and the tail is compressed
            lcod = min(lcod_s, (sb[k]["g"] - sb[k]["z"]) * FR + 0.01)
            if k + 1 < n:
                tg = tc[k + 1] if q["gap"] == 0 else min(q["g"] * FR, tc[k + 1])
            else:
                tg = q["g"] * FR
            # the vowel keeps at least ~45% of the note before its coda; a long coda is compressed instead
            tmin = T[k] + max(0.03, 0.45 * (tg - T[k]) if lcod > 0 else 0.03)
            tz = min(q["z"] * FR if (q["gap"] > 0 or k + 1 == n) else tg, tg - lcod)
            tz = min(max(tz, tmin), tg)
            segs.append((tc[k], T[k], sc[k], m[k], None))
            segs.append((T[k], tz, m[k], sz[k], ("nuc", src["offglide"][k] * FR)))
            segs.append((tz, tg, sz[k], sg[k], None))
            if k + 1 < n:
                segs.append((tg, tc[k + 1], sg[k], sc[k + 1], None))
            else:
                segs.append((tg, t_end, sg[k], max(sg[k] + 1e-3, s_end), None))

        holds = []

        def f(tq):
            tq = np.asarray(tq, np.float64)
            out = np.zeros_like(tq)
            for (t0, t1, s0, s1, kind) in segs:
                sel = (tq >= t0) & (tq < t1) if t1 > t0 else np.zeros(tq.shape, bool)
                if not sel.any():
                    continue
                u = tq[sel] - t0
                Lt, Ls = t1 - t0, s1 - s0
                if kind is None or Lt <= Ls + 0.01:
                    out[sel] = s0 + u * (Ls / Lt)
                    continue
                B = min(max(0.03, kind[1] + 0.015), 0.4 * Ls)
                A = min(0.06, 0.4 * Ls)
                Lm = Ls - A - B
                if Lm < 0.03:
                    out[sel] = s0 + u * (Ls / Lt)
                    continue
                Mt = Lt - A - B
                # head and tail (consonant transitions, diphthong off-glide) at natural speed; the stable middle
                # is stretched slowly and (in render_sung) pulled towards its mean spectrum -> a held vowel
                y = np.empty_like(u)
                h = u < A
                y[h] = u[h]
                e = u >= A + Mt
                y[e] = Ls - (Lt - u[e])
                mid = ~h & ~e
                y[mid] = A + Lm * (u[mid] - A) / Mt
                out[sel] = s0 + y
                if (t0, t1) not in [hh[:2] for hh in holds]:
                    holds.append((t0 + A, t1 - B, s0 + A, s1 - B))
            out[tq >= segs[-1][1]] = segs[-1][3]
            return out
        f(np.array([0.0]))
        holds.clear()
        f(np.arange(0, t_end, 0.0025))
        return f, segs, m, list(holds)

    def render_sung(self, plan, voice, detune_cents=0.0, octave=0, seed=0, blend=None, energy_boost=0.0,
                    vib_depth=0.22, neural_shift=None, style_shift=0, retune=True, level=True, closed_loop=True):
        """Full sung render (24 kHz) on the TARGET timeline (sample 0 = plan['R0']).
        The decoder sings the source plan (natural-length vowels); its measured vowel onsets are then mapped
        onto the score's note times in the WORLD domain."""
        import pyworld as pw
        if neural_shift is None:
            # The decoder is asked to sing a few semitones below the target so the phrase's top note stays in
            # the voice's comfortable range (male top <= C4+, female <= Ab4); WORLD then moves it to pitch.
            top = max(n["m"] for n in plan["notes"]) + 12 * octave
            ceiling = NEURAL_TOP_M if voice[1] == "m" else NEURAL_TOP_F
            neural_shift = int(np.clip(ceiling - top, -12, 0))
        src = self.source_plan(plan)
        nr = self.render_neural(src, voice, neural_shift, style_shift, blend, energy_boost)
        x = nr["audio"]
        meas = self.measure_vowels(src, x) if closed_loop else [None] * len(src["bnd"])
        offs = self.measure_offsets(src, x, meas) if closed_loop else None
        fmap, segs, m_used, holds = self.time_map(plan, src, meas, offs)
        wa = self.world_analyse(nr)
        fp = 0.005
        # voicing follows the measured vowel onsets (consonant noise stays unvoiced, early vowels voiced)
        v_src = wa["v"].copy()
        for k, mk in enumerate(meas):
            if mk is None:
                continue
            q = src["bnd"][k]
            a = min(q["c"] * FR, mk - 0.005)
            i0, i1 = int(np.floor(a / fp)), int(np.floor(mk / fp))
            if self.after_unvoiced(src, k):
                v_src[max(0, i0):max(0, i1)] = False
            iv = int(np.floor(q["v"] * FR / fp))
            if i1 < iv:
                v_src[i1:iv] = True
        for k, ok in enumerate(offs or []):
            if ok is None:
                continue
            q = src["bnd"][k]
            iz, io = int(np.floor(q["z"] * FR / fp)), int(np.floor(ok / fp))
            if io > iz:
                v_src[iz:io] = True              # vowel still sounding past its planned end
            else:
                v_src[io:iz] = False
        if closed_loop:
            meas = self.effective_onsets(src, meas, v_src)
            fmap, segs, m_used, holds = self.time_map(plan, src, meas, offs)

        def synth(fmap, holds):
            n_t = int(plan["dur"].sum()) * HOP
            t5 = np.arange(int(n_t / KSR / fp) + 1) * fp
            s5 = fmap(t5) / fp
            nsrc = len(wa["f0"])
            i0 = np.clip(np.floor(s5).astype(int), 0, nsrc - 1)
            i1 = np.clip(i0 + 1, 0, nsrc - 1)
            w = np.clip(s5 - i0, 0, 1)[:, None]
            sp = np.exp((1 - w) * np.log(wa["sp"][i0] + 1e-16) + w * np.log(wa["sp"][i1] + 1e-16))
            ap = (1 - w) * wa["ap"][i0] + w * wa["ap"][i1]
            # held vowels: crossfade (40 ms) into the mean spectrum of the stable middle, keeping 30% of its slow
            # natural evolution -> no re-articulation, no flutter
            lsp = np.log(sp + 1e-16)
            for (ta, tb_, sa, sb_) in holds:
                ja, jb = int(np.ceil(ta / fp)), int(np.floor(tb_ / fp))
                ka, kb = int(np.floor(sa / fp)), int(np.ceil(sb_ / fp)) + 1
                if jb - ja < 4 or kb - ka < 2:
                    continue
                msp = np.mean(np.log(wa["sp"][ka:kb] + 1e-16), axis=0)
                map_ = np.mean(wa["ap"][ka:kb], axis=0)
                jj = np.arange(ja, jb)
                r = int(round(0.04 / fp))
                g = np.minimum(1.0, np.minimum(jj - ja, jb - 1 - jj) / max(1, r))[:, None] * HOLD_MEAN
                lsp[ja:jb] = (1 - g) * lsp[ja:jb] + g * msp
                ap[ja:jb] = (1 - g) * ap[ja:jb] + g * map_
            sp = np.exp(lsp)
            inn = np.clip(np.round(s5).astype(int), 0, nsrc - 1)
            wt = dict(t=t5, sp=sp, ap=ap, v=v_src[inn], nucl=wa["nucl"][inn])
            if not retune:
                raise NotImplementedError
            y, f0t = self.world_synth(plan, wt, detune_cents, octave, seed, vib_depth)
            y = np.pad(y, (0, max(0, n_t - len(y))))[:n_t]
            # source audio warped onto the target timeline (used for the unvoiced stretches: s, t, k, f, breaths,
            # which keep the decoder's crisp bursts and fricatives)
            ss = fmap(np.arange(n_t) / KSR) * KSR
            xw = _interp_cubic(x, ss)
            vs = np.interp(np.arange(n_t) / KSR, t5, wt["v"].astype(float))
            k = int(0.010 * KSR)
            vs = np.convolve(vs, np.ones(k) / k, mode="same")
            # unvoiced stretches: the decoder's own audio above XOVER_HZ (crisp s/t/k/f) + WORLD's noise
            # excitation below it. The decoder sings at a transposed pitch, so its low band must not leak in
            # (breathy 'h', partly voiced codas would otherwise add a second, wrong pitch).
            sos_hp = butter(4, XOVER_HZ, "highpass", fs=KSR, output="sos")
            sos_lp = butter(4, XOVER_HZ, "lowpass", fs=KSR, output="sos")
            # ... and only where the decoder really is noisy there: if it is still periodic (a vowel ringing into
            # a coda slot), WORLD's noise from the same envelope is used instead of the decoder audio
            nb_ = wa["ap"].shape[1]
            fq_ = np.arange(nb_) / (nb_ - 1) * KSR / 2
            apm = wa["ap"][:, (fq_ >= 1000) & (fq_ <= 4000)].mean(axis=1)
            use_x = np.clip((apm[inn] - 0.25) / 0.3, 0, 1)
            ux = np.interp(np.arange(n_t) / KSR, t5, use_x)
            ux = np.convolve(ux, np.ones(k) / k, mode="same")
            hx, hy = sosfiltfilt(sos_hp, xw), sosfiltfilt(sos_hp, y)
            unv = ux * hx + (1 - ux) * hy + sosfiltfilt(sos_lp, y)
            y = vs * y + (1 - vs) * unv
            y = self.gate(plan, y)
            if level:
                y = self.level_syllables(plan, y)
            return y, f0t, t5, wt

        y, f0t, t5, wt = synth(fmap, holds)
        # second loop: anything the decoder-side measurement missed (or got wrong) is re-measured on the
        # finished render and corrected once more (up to 2 passes)
        corr = [0.0] * len(meas)
        for _ in range(2 if closed_loop else 0):
            ver = self.verify_onsets(plan, y)
            bad = [kk for kk, e in enumerate(ver) if e is not None and abs(e) > 6.0]
            if not bad:
                break
            meas = list(meas)
            for kk in bad:
                base = meas[kk] if meas[kk] is not None else src["bnd"][kk]["v"] * FR
                meas[kk] = base + ver[kk] / 1000.0
                corr[kk] += ver[kk]
            fmap, segs, m_used, holds = self.time_map(plan, src, meas, offs)
            y, f0t, t5, wt = synth(fmap, holds)
        tb = self.bounds(plan)
        timing = []
        for kk, q in enumerate(src["bnd"]):
            timing.append(dict(measured_src_ms=None if meas[kk] is None else round(1000 * (meas[kk] - q["v"] * FR), 1),
                               second_pass_ms=round(corr[kk], 1) if corr[kk] else 0.0))
        return dict(audio=y.astype(np.float32), f0_target5=f0t, t5=t5, nucl5=wt["nucl"], neural=x,
                    f0_fed=nr["f0_fed"], nucl=nr["nucl"], meas=meas, timing=timing, segs=segs)

    def verify_onsets(self, plan, audio, extra_delay=0.0):
        """Measured (audio) minus target vowel onset, ms, per syllable (None where not measurable)."""
        import vox_onset as vo
        x16 = vo.to16(audio, KSR)
        tb = self.bounds(plan)
        out = []
        for k, q in enumerate(tb):
            if not self.measurable(plan, k):
                out.append(None)
                continue
            T = q["v"] * FR + extra_delay
            mm = vo.vowel_onset(x16, T, (q["z"] - q["v"]) * FR)
            out.append(None if mm is None else round(1000 * (mm - T), 1))
        return out

    # ------------------------------------------------------------------------- speech
    def speech_plan(self, text, lang, voice, max_len=None, speed=1.0, max_speed=1.35, inner_pause_cap=6,
                    allow_compress=False):
        """Natural-speech durations (pads trimmed), compressed to fit max_len seconds of audible speech."""
        ph = self.phonemes(text, lang)
        toks, chars = self.tokens(ph)
        st = self.style(voice, len(toks))

        def durs(sp):
            _, d, _, _ = self.analyse(toks, st, sp)
            d = d.copy()
            # trailing punctuation pauses removed, inner pauses capped
            j = len(chars) - 1
            while j >= 0 and cls_of(chars[j]) in "PW":
                d[j + 1] = 0
                j -= 1
            for i, c in enumerate(chars):
                if cls_of(c) == "P" and d[i + 1] > inner_pause_cap:
                    d[i + 1] = inner_pause_cap
            return d

        d = durs(speed)
        log = []
        if max_len is not None:
            sp = speed
            while sum(d[1:-1]) * FR > max_len and sp < max_speed - 1e-6:
                sp = min(max_speed, sp + 0.05)
                d = durs(sp)
                log.append(f"speed {sp:.2f}")
            body = d[1:-1].astype(np.float64)
            if allow_compress and body.sum() * FR > max_len:
                # stress marks / word spaces are silent bookkeeping tokens: give them no time first
                for i, c in enumerate(chars):
                    if cls_of(c) in "SW":
                        body[i] = 0
                # duration-domain time compression (min 1 frame per sounding token)
                target = int(np.floor(max_len / FR))
                f = target / body.sum()
                nb = np.where(body > 0, np.maximum(1, np.round(body * f)), 0)
                if nb.sum() > target:
                    log.append("frame floor reached")
                while nb.sum() > target and (nb > 1).any():
                    nb[np.argmax(nb)] -= 1
                d[1:-1] = nb.astype(np.int64)
                log.append(f"compressed x{f:.2f}")
                speed_used = sp
            else:
                speed_used = sp
        else:
            speed_used = speed
        d[0] = PAD_LEAD
        d[-1] = PAD_TAIL
        return dict(text=text, lang=lang, ph=ph, toks=toks, chars=chars, dur=d, log=log, speed=speed_used)

    def render_speech(self, splan, voice, f0_mode="natural", f0_scale=1.0, f0_flatten=0.0, blend=None):
        toks, dur = splan["toks"], splan["dur"]
        st = self.style(voice, len(toks), blend)
        a, f0p, n = self.run_dur(toks, st, dur)
        if f0_mode == "natural" and f0_scale == 1.0 and f0_flatten == 0.0:
            out = a
        else:
            f0 = f0p.astype(np.float64).copy()
            v = f0 > 45
            if v.any():
                lf = np.log(f0[v])
                med = np.median(lf)
                lf = med + (1 - f0_flatten) * (lf - med) + np.log(f0_scale)
                f0[v] = np.exp(lf)
            f0[~v] = 0
            out = self.run_sing(toks, st, dur, f0.astype(np.float32), n)
        lead = max(0, int(dur[0]) - LAG - 2) * HOP
        out = out.copy()
        out[:lead] = 0
        tail = int(dur[-1]) * HOP
        out[len(out) - tail:] *= np.linspace(1, 0, tail) ** 2
        return out.astype(np.float32)
