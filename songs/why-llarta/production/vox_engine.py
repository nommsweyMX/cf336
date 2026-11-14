"""Neural speech-to-singing engine built on a locally re-wired Kokoro v1.0 graph.

Kokoro predicts, per phoneme token, a duration in 25 ms alignment frames, then an F0 curve and an
energy curve (N) at 12.5 ms, and renders audio with its neural decoder. kokoro_surgery.py exposes
those quantities as graph inputs, so instead of time-stretching/pitch-shifting finished speech
(WORLD) we *drive the decoder directly*:
  1. analyse the phrase as natural speech      -> natural phoneme durations
  2. syllabify the phonemes against the notes  -> onset / nucleus / coda tokens per note
  3. plan frame durations so every vowel nucleus starts on its note (consonants before it)
  4. re-run the prosody predictor on that plan -> natural voicing decisions + energy
  5. replace F0 by the melody (glides, scoops, vibrato, drift), sustain energy through vowels
  6. decode -> sung audio, 24 kHz.
Spoken lines use steps 1 (+uniform duration compression when they must fit a slot) and 4 directly.
"""
import hashlib
import os

import numpy as np
import onnxruntime as rt

SCRATCH = "/tmp/claude-0/-home-user-cf336/c7c99ebc-57ee-5e7f-9a20-477ea4b0b187/scratchpad"
MODEL_DIR = os.environ.get("VOX_MODEL_DIR", os.path.join(SCRATCH, "models"))
KSR = 24000          # Kokoro sample rate
HOP = 600            # samples per alignment frame
FR = HOP / KSR       # 0.025 s
F0R = FR / 2         # 0.0125 s per F0/N point
PAD_LEAD = int(os.environ.get("VOX_PAD_LEAD", 20))   # the decoder needs ~12-16 frames of warm-up before it can sound
PAD_TAIL = int(os.environ.get("VOX_PAD_TAIL", 16))
SEG_FRAMES = int(os.environ.get("VOX_SEG", 5))         # max frames per vowel token before it is re-articulated
SEG_PATTERN = os.environ.get("VOX_SEG_PATTERN", "rep")

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
    def plan_sung(self, text, lang, notes, ana_voice, ana_speed=1.0, tail=None):
        """notes: list of (T_abs, D_sec, midi). Returns a plan dict (shared by every voice singing it).
        tail: optional continuation text appended as zero-duration tokens: it is never heard, but it keeps
        the last sung syllable from being utterance-final for the text encoder (avoids creaky phrase-final
        devoicing on held notes)."""
        ph = self.phonemes(text, lang)
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
            on = [int(np.clip(nat[x], 1, 3)) for x in s["onset"]]
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
            nuc_frames.append((k, cur, cur + nuc))
            cur += nuc
            for x, f in zip(s["coda"], co):
                dur[x + 1] = f
            cur += sum(co)
            gap = (next_on - cur) if next_on is not None else 0
            if gap > 0:
                rest_mask.append((cur, next_on))
                tgt = [x for x in s["gap"] if cls_of(chars[x]) in "WP"]
                if tgt:
                    dur[tgt[0] + 1] += gap
                elif sylls[k + 1]["onset"]:
                    dur[sylls[k + 1]["onset"][0] + 1] += gap
                else:
                    dur[s["nucleus"][-1] + 1] += gap
                    nuc_frames[-1] = (k, nuc_frames[-1][1], nuc_frames[-1][2])
                cur = next_on
            elif gap < 0:
                raise RuntimeError(f"negative gap at syllable {k}: {gap}")
        tail_pad = PAD_TAIL
        rest_mask.append((cur, cur + tail_pad + 2))
        dur[-1] = tail_pad + 2
        assert int(dur.sum()) == cur + tail_pad + 2
        plan = dict(text=text, lang=lang, ph=ph, toks=toks, chars=chars, sylls=sylls, slog=slog, R0=R0, tail=tail,
                    VF=VF, vowel_err=verr, dur=dur, rest_mask=rest_mask, nuc_frames=nuc_frames,
                    notes=nd, onset_frames=on_f, coda_frames=co_f, ntok0=len(toks))
        return self.expand(plan)

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

    def pitch_curve(self, plan, nframes, detune_cents=0.0, octave=0, seed=0, vib_depth=0.22, scoop=0.4):
        """Target pitch (MIDI float) per 12.5 ms point. Times relative to plan R0."""
        rng = np.random.default_rng(seed)
        t = plan["R0"] + (np.arange(nframes) + 0.5) * F0R
        notes = [n for s in plan["sylls"] for n in s["notes"]]
        T = np.array([n["T"] for n in notes])
        D = np.array([n["D"] for n in notes])
        M = np.array([n["m"] for n in notes], dtype=np.float64)
        # step function at vowel onsets; transitions: glide if legato, scoop after rests
        idx = np.clip(np.searchsorted(T, t, side="right") - 1, 0, len(T) - 1)
        pitch = M[idx].copy()
        for j in range(1, len(T)):
            legato = T[j] - (T[j - 1] + D[j - 1]) < 0.12
            if legato and M[j] != M[j - 1]:
                a, b = T[j] - 0.035, T[j] + 0.02
                w = np.clip((t - a) / (b - a), 0, 1)
                w = w * w * (3 - 2 * w)
                sel = (t >= a) & (t < b + F0R)
                pitch[sel] = M[j - 1] + (M[j] - M[j - 1]) * w[sel]
        for j in range(len(T)):
            legato = j > 0 and T[j] - (T[j - 1] + D[j - 1]) < 0.12
            depth = 0.12 if legato else scoop
            w = np.clip((t - T[j]) / 0.05, 0, 1)
            sel = (t >= T[j] - 0.05) & (t < T[j] + 0.05)
            pitch[sel] -= depth * (1 - (w[sel] * w[sel] * (3 - 2 * w[sel])))
            # vibrato on long notes
            if D[j] >= 0.35:
                rate = 5.2 + 0.4 * rng.random()
                ph0 = rng.random() * 2 * np.pi
                tt = t - T[j]
                ramp = np.clip((tt - 0.15) / 0.22, 0, 1) * np.clip((D[j] - tt) / 0.06, 0, 1)
                sel = (tt >= 0) & (tt < D[j])
                pitch[sel] += vib_depth * ramp[sel] * np.sin(2 * np.pi * rate * tt[sel] + ph0 * 0)
        ph1, ph2 = rng.random(2) * 2 * np.pi
        pitch += 0.04 * np.sin(2 * np.pi * 0.7 * t + ph1) + 0.025 * np.sin(2 * np.pi * 1.9 * t + ph2)
        pitch += detune_cents / 100.0 + 12 * octave
        return pitch

    def render_sung(self, plan, voice, detune_cents=0.0, octave=0, seed=0, blend=None, energy_boost=0.0,
                    vib_depth=0.22):
        toks, chars, dur = plan["toks"], plan["chars"], plan["dur"]
        st = self.style(voice, plan.get("ntok0", len(toks)), blend)
        _, f0p, n_p = self.run_dur(toks, st, dur)
        nf = len(f0p)
        assert nf == 2 * int(dur.sum()), (nf, dur.sum())
        tok_of = np.repeat(np.arange(len(dur)), 2 * dur)  # token index per F0 point (0 = pad)
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
        # unvoiced obstruents stay unvoiced unless the predictor itself voiced them
        unv_obs = (c == "C") & ~son & ~np.isin(ch, list(VOICED_OBS))
        voiced &= ~(unv_obs & (f0p <= 45.0))
        pitch = self.pitch_curve(plan, nf, detune_cents, octave, seed, vib_depth=vib_depth)
        f0 = np.where(voiced, midi_hz(pitch), 0.0).astype(np.float32)
        # energy: sustain through vowel nuclei
        n = n_p.astype(np.float64).copy()
        allnuc = n[nucl]
        pref = np.percentile(allnuc, 80) if allnuc.size else 0.0
        for (k, a, b) in plan["nuc_frames"]:
            a2, b2 = 2 * a, 2 * b
            if b2 - a2 < 4:
                continue
            seg = n[a2:b2]
            ref = max(np.percentile(seg, 80), pref - 1.5) + energy_boost
            lo, hi = a2 + 2, b2 - (3 if k == len(plan["sylls"]) - 1 else 1)
            n[lo:hi] = np.maximum(n[lo:hi], ref)
        n = np.convolve(np.pad(n, 1, mode="edge"), np.ones(3) / 3, mode="valid")
        audio = self.run_sing(toks, st, dur, f0, n.astype(np.float32))
        # gate rests / pads to true silence with short ramps
        g = np.ones(len(audio))
        for (a, b) in plan["rest_mask"]:
            s0, s1 = a * HOP, min(len(audio), b * HOP)
            g[s0:s1] = 0.0
            r = int(0.012 * KSR)
            if s0 > 0:
                g[max(0, s0 - r):s0] = np.minimum(g[max(0, s0 - r):s0], np.linspace(1, 0, s0 - max(0, s0 - r)))
            if s1 < len(audio):
                e = min(len(audio), s1 + r)
                g[s1:e] = np.minimum(g[s1:e], np.linspace(0, 1, e - s1))
        lead = int(plan["dur"][0]) * HOP
        g[:lead] = 0.0
        r = int(0.008 * KSR)
        g[lead:lead + r] = np.minimum(g[lead:lead + r], np.linspace(0, 1, r))
        audio = audio * g
        return dict(audio=audio.astype(np.float32), f0_target=f0, pitch=pitch, voiced=voiced, nucl=nucl,
                    f0_pred=f0p, n=n)

    # ------------------------------------------------------------------------- speech
    def speech_plan(self, text, lang, voice, max_len=None, speed=1.0, max_speed=1.35, inner_pause_cap=6):
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
            if body.sum() * FR > max_len:
                # duration-domain time compression (min 1 frame per sounding token)
                target = int(np.floor(max_len / FR))
                f = target / body.sum()
                nb = np.where(body > 0, np.maximum(1, np.round(body * f)), 0)
                while nb.sum() > target and (nb > 1).any():
                    nb[np.argmax(nb)] -= 1
                d[1:-1] = nb.astype(np.int64)
                log.append(f"compressed x{f:.2f}")
                speed_used = sp
            else:
                speed_used = sp
        else:
            speed_used = speed
        d[0] = 4
        d[-1] = 5
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
        lead = int(dur[0]) * HOP
        out = out.copy()
        out[:lead] = 0
        tail = int(dur[-1]) * HOP
        out[len(out) - tail:] *= np.linspace(1, 0, tail) ** 2
        return out.astype(np.float32)
