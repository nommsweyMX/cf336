"""WHAT YOU REPEAT - vocal stems, rendered locally.

    cd /home/user/cf336/songs/why-llarta/production && python3 vocals.py [--no-asr] [--only ID[,ID...]]

Reads score.py (SUNG, SPOKEN, VOICES, t(), SONG_SECONDS) and writes five dry stereo float32 WAV stems
(44.1 kHz, exactly round(SONG_SECONDS*44100) samples, sample 0 = song time 0) to STEM_DIR:
    vox_lead.wav  vox_harmony.wav  vox_crowd.wav  vox_spoken.wav  vox_whisper.wav
plus vocals_report.json, and isolated per-phrase renders in DEBUG_DIR.

Method (see vox_engine.py): Kokoro v1.0 with its duration / F0 / energy predictions exposed as graph
inputs (kokoro_surgery.py). Each sung phrase is syllabified from its phonemes against the score's notes,
every token gets a frame budget so each vowel nucleus starts on its note (consonants before it), the
decoder sings the melody a few semitones down (its comfortable range) with natural-length vowels; the
vowel onsets it actually produced are measured and mapped onto the note times in the WORLD domain
(closed loop, re-verified on the finished render), held vowels are sustained in WORLD, and the voiced
parts are re-pitched to the exact melody while unvoiced consonants come from the decoder. Every sung
vowel onset is re-measured on the finished stems (report: stem_vowel_onset_err_ms, timing_violations).
Spoken lines use the natural Kokoro prosody at the slowest speed that ends (-40 dB) 30 ms before the
deadline (light WORLD compression only as a last resort), with an edge fade that reaches zero 10 ms
before it. Whispers are WORLD resyntheses with no voicing.
"""
import argparse
import json
import os
import sys
import time
import zlib

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, butter, sosfilt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score  # noqa: E402
import vox_engine as ve  # noqa: E402

# Working folder for models, render cache and stems (git-ignored). Override with WHY_LLARTA_WORK.
SCRATCH = os.environ.get("WHY_LLARTA_WORK", os.path.join(os.path.dirname(os.path.abspath(__file__)), "work"))
STEM_DIR = os.environ.get("VOX_STEM_DIR", os.path.join(SCRATCH, "song", "stems"))
DEBUG_DIR = os.environ.get("VOX_DEBUG_DIR", os.path.join(SCRATCH, "song", "vox_debug"))
SR = score.SR
NSAMP = int(round(score.SONG_SECONDS * SR))
LEAD = score.VOICES["lead"]["voice"]
LOCALS = score.VOICES["locals"]["voices"]
GANG = score.VOICES["gang"]["voices"]

# per-render loudness targets (dBFS RMS over the active part of one isolated render)
LVL = dict(lead=-18.0, double=-30.0, harmony=-25.5, crowd=-27.5, crowd_oct=-37.0, crowd_front=3.0,
           spoken=-19.0, chant=-22.5, aside=-25.0, whisper=-21.0, whisper_locals=-24.0)
DOUBLE_SECTIONS = ("chorus1", "chorus2", "final", "tag")
# vowel onsets are placed closed-loop in vox_engine.render_sung (measured on the decoder output and mapped
# onto the note times), so no global offset is applied.
SUNG_SHIFT = 0.0
TIMING_TOL_MS = 15.0
END_REL_DB = -50.0        # a spoken line "ends" when it falls 50 dB below its peak
END_MARGIN_S = 0.03       # ... at least 30 ms before its deadline (the edge fade reaches zero at -10 ms)
MIN_TSM = 0.9             # strongest WORLD time compression allowed before Kokoro is asked to talk faster
FIT_REL_DB = -40.0        # words must be over (40 dB down) before the deadline fade starts
MAX_KOKORO_SPEED = 1.75
CROWD_MAX_OFFSET_MS = 10.0


def section_of(bar):
    for name, start, n, _, _ in score.SECTIONS:
        if start <= bar < start + n:
            return name
    return "outro"


def seed_of(s):
    return zlib.crc32(s.encode()) & 0xFFFF


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


PEAK_CAP = dict(lead=-4.0, double=-14.0, harmony=-10.5, crowd=-12.5, crowd_front=-9.5, crowd_oct=-21.0, spoken=-4.5, chant=-8.0,
                aside=-10.0, whisper=-6.0, whisper_locals=-9.0)


def limit(x, ceiling_db, sr=ve.KSR):
    """Transparent peak limiter (5 ms look-ahead, ~60 ms release) so a single WORLD/decoder transient does
    not set the level of a whole stem."""
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
    # release: let gain recover no faster than ~60 ms
    rel = np.exp(-1.0 / (0.06 * sr))
    out = g.copy()
    for i in range(1, len(out)):
        if out[i] > out[i - 1]:
            out[i] = out[i - 1] * rel + out[i] * (1 - rel)
    y = x * out
    return np.clip(y, -c, c)


def norm_to(x, db, sr=ve.KSR, kind=None):
    y = x * 10 ** ((db - active_rms_db(x, sr)) / 20)
    if kind is not None:
        y = limit(y, PEAK_CAP[kind], sr)
    return y


def pan_gains(p):
    a = (np.clip(p, -1, 1) + 1) * np.pi / 4
    return np.cos(a) * np.sqrt(2), np.sin(a) * np.sqrt(2)   # unity at centre, constant power


def place(stem, x44, start_s, pan=0.0):
    """Add mono x44 to stereo stem with sample 0 of x44 at song time start_s."""
    i0 = int(round(start_s * SR))
    a, b = max(0, i0), min(NSAMP, i0 + len(x44))
    if b <= a:
        return
    seg = x44[a - i0:b - i0]
    gl, gr = pan_gains(pan)
    stem[a:b, 0] += gl * seg
    stem[a:b, 1] += gr * seg


def envelope_onset_end(x, sr, rel_db=-32.0, hop_s=0.003):
    hop = max(1, int(hop_s * sr))
    n = len(x) // hop
    fr = np.sqrt(np.mean(x[:n * hop].reshape(n, hop) ** 2, axis=1) + 1e-20)
    thr = fr.max() * 10 ** (rel_db / 20)
    idx = np.nonzero(fr > thr)[0]
    if len(idx) == 0:
        return 0, len(x)
    return int(idx[0] * hop), int((idx[-1] + 1) * hop)


# ----------------------------------------------------------------------------- music helpers
F_MINOR = [5, 7, 8, 10, 0, 1, 3]     # F G Ab Bb C Db Eb
G_MINOR = [7, 9, 10, 0, 2, 3, 5]     # G A Bb C D Eb F


def third_above(m, bar):
    scale = F_MINOR if bar < 116 else G_MINOR
    pcs = sorted(set(scale))
    if m % 12 not in pcs:  # chromatic note: harmonise from the scale tone below
        base = m - 1
        while base % 12 not in pcs:
            base -= 1
    else:
        base = m
    # walk up two scale steps
    x, steps = base, 0
    while steps < 2:
        x += 1
        if x % 12 in pcs:
            steps += 1
    return x


# ----------------------------------------------------------------------------- pitch / timing check
def pitch_check(audio, res, plan):
    """Median |cents| and % of voiced nucleus frames within 50 cents, measured with harvest on the render."""
    import pyworld as pw
    f, _ = pw.harvest(audio.astype(np.float64), ve.KSR, f0_floor=60, f0_ceil=1000, frame_period=5.0)
    tg = res["f0_target5"]
    n = min(len(f), len(tg))
    sel = res["nucl5"][:n] & (tg[:n] > 0)
    det = sel & (f[:n] > 0)
    if det.sum() == 0:
        return dict(med_cents=None, within50=0.0, voiced=0.0)
    c = np.abs(1200 * np.log2(f[:n][det] / tg[:n][det]))
    return dict(med_cents=round(float(np.median(c)), 1), within50=round(float(np.mean(c < 50)), 3),
                voiced=round(float(det.sum() / max(1, sel.sum())), 3))


def world_tsm(x, factor, sr=ve.KSR):
    """WORLD time-scale modification (factor < 1 shortens), pitch and formants kept."""
    import pyworld as pw
    x = np.asarray(x, np.float64)
    f0, t = pw.harvest(x, sr, f0_floor=60, f0_ceil=600, frame_period=5.0)
    sp = pw.cheaptrick(x, f0, t, sr)
    ap = pw.d4c(x, f0, t, sr)
    n = len(f0)
    m = max(2, int(round(n * factor)))
    src = np.linspace(0, n - 1, m)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, n - 1)
    w = (src - i0)[:, None]
    sp2 = np.exp((1 - w) * np.log(sp[i0] + 1e-16) + w * np.log(sp[i1] + 1e-16))
    ap2 = (1 - w) * ap[i0] + w * ap[i1]
    f02 = f0[np.round(src).astype(int)]
    y = pw.synthesize(f02, np.ascontiguousarray(sp2), np.ascontiguousarray(ap2), sr, 5.0)
    return y


class Vocals:
    def __init__(self, asr=True, only=None):
        self.E = ve.Engine(cache_dir=os.path.join(DEBUG_DIR, "cache"))
        self.stems = {k: np.zeros((NSAMP, 2), np.float64) for k in ("lead", "harmony", "crowd", "spoken", "whisper")}
        self.report = dict(sung=[], spoken=[], stems={}, notes=[])
        self.asr = asr
        self.only = set(only) if only else None
        if asr:
            import vox_asr
            self.vox_asr = vox_asr
            if not vox_asr.available():
                print("ASR model missing; skipping intelligibility checks")
                self.asr = False
        for d in ("lead", "harmony", "crowd", "spoken", "whisper"):
            os.makedirs(os.path.join(DEBUG_DIR, d), exist_ok=True)

    def dbg(self, sub, name, x, sr=ve.KSR):
        sf.write(os.path.join(DEBUG_DIR, sub, name + ".wav"), np.asarray(x, np.float32), sr, subtype="FLOAT")

    def asr_check(self, x, lang, ref, sr=ve.KSR):
        if not self.asr:
            return None
        hyp = self.vox_asr.transcribe(x, sr, lang)
        rec, acc = self.vox_asr.word_scores(ref, hyp)
        return dict(hyp=hyp, recall=round(rec, 3), word_acc=round(acc, 3))

    def asr_raw(self, text, lang, voice, ref):
        if not self.asr:
            return None
        a, _ = self.E.k.create(text, voice, speed=1.0, lang="en-us" if lang == "en" else "es")
        return self.asr_check(a, lang, ref)

    # ------------------------------------------------------------------------- sung
    def notes_abs(self, p, transform=None):
        out = []
        for (b, d, m, _) in p["notes"]:
            mm = transform(m) if transform else m
            out.append((score.t(p["bar"], b), d * score.BEAT, mm))
        return out

    def plan_for(self, p, ana_voice, transform=None):
        text = p.get("tts_text") or p["text"]
        return self.E.plan_sung(text, p["lang"], self.notes_abs(p, transform), ana_voice)

    def sung_entry(self, p, plan, stem, voices):
        syl = []
        for s, n in zip(plan["sylls"], p["notes"]):
            c0 = plan.get("chars0", plan["chars"])
            syl.append("".join(c0[x] for x in s["onset"]) + "|" + "".join(c0[x] for x in s["nucleus"]) + "|"
                       + "".join(c0[x] for x in s["coda"]))
        verr = [round(1000 * e, 1) for e in plan["vowel_err"]]
        return dict(id=p["id"], stem=stem, bar=p["bar"], role=p["role"], lang=p["lang"], text=p["text"],
                    tts_text=p.get("tts_text"), phonemes=plan["ph"], n_notes=len(p["notes"]),
                    n_syllables=len(plan["sylls"]), syllables=syl, score_syllables=[n[3] for n in p["notes"]],
                    syllabify_log=plan["slog"], voices=voices, plan_grid_err_ms=verr,
                    note_times_s=[round(n["T"], 4) for s_ in plan["sylls"] for n in s_["notes"][:1]],
                    note_durs_s=[round(n["D"], 4) for s_ in plan["sylls"] for n in s_["notes"][:1]],
                    measurable=[bool(self.E.measurable(plan, k)) for k in range(len(plan["sylls"]))])

    def timing_entry(self, e, r, plan, label):
        """Closed-loop timing of one render: decoder lateness that was corrected, and the vowel onset error
        measured on the finished render (ms; None = no acoustic boundary to measure)."""
        ver = self.E.verify_onsets(plan, r["audio"])
        e.setdefault("render_timing", {})[label] = dict(
            decoder_lateness_ms=[t["measured_src_ms"] for t in r["timing"]], measured_err_ms=ver)
        return ver

    def render_lead(self, p):
        sec = section_of(p["bar"])
        seed = seed_of(p["id"])
        plan = self.plan_for(p, LEAD)
        r = self.E.render_sung(plan, LEAD, seed=seed)
        x = norm_to(r["audio"], LVL["lead"], kind="lead")
        self.dbg("lead", p["id"], x)
        place(self.stems["lead"], to44(x), plan["R0"] + SUNG_SHIFT, 0.0)
        e = self.sung_entry(p, plan, "vox_lead", [LEAD])
        e["section"] = sec
        e["pitch"] = pitch_check(r["audio"], r, plan)
        e["vowel_onset_err_ms"] = self.timing_entry(e, r, plan, LEAD)
        if sec in DOUBLE_SECTIONS:
            dd = []
            for j, (cents, off_ms, pan, sshift) in enumerate([(+6.0, 9.0, -0.2, 17), (-6.0, 13.0, 0.2, -17)]):
                rd = self.E.render_sung(plan, LEAD, detune_cents=cents, seed=seed + 101 + j, style_shift=sshift,
                                        vib_depth=0.18)
                self.timing_entry(e, rd, plan, f"double{j}")
                xd = norm_to(rd["audio"], LVL["double"], kind="double")
                self.dbg("lead", f"{p['id']}_dbl{j}", xd)
                place(self.stems["lead"], to44(xd), plan["R0"] + SUNG_SHIFT + off_ms / 1000.0, pan)
                dd.append(dict(detune_cents=cents, offset_ms=off_ms, pan=pan))
            e["double"] = dd
        e["asr"] = self.asr_check(r["audio"], p["lang"], p["text"])
        e["asr_raw_tts"] = self.asr_raw(p.get("tts_text") or p["text"], p["lang"], LEAD, p["text"])
        self.report["sung"].append(e)
        if p.get("harmony"):
            self.render_harmony(p)
        return e

    def render_harmony(self, p):
        seed = seed_of(p["id"] + "h")
        tr = lambda m: third_above(m, p["bar"])  # noqa: E731
        plan = self.plan_for(p, LEAD, tr)
        hv = GANG[0]  # am_adam
        out = []
        e = self.sung_entry(p, plan, "vox_harmony", [LEAD, hv])
        for j, (v, pan, cents) in enumerate([(LEAD, -0.3, -4.0), (hv, 0.3, 4.0)]):
            r = self.E.render_sung(plan, v, detune_cents=cents, seed=seed + j, vib_depth=0.16,
                                   style_shift=9 if v == LEAD else 0)
            ver = self.timing_entry(e, r, plan, v)
            if j == 0:
                e["vowel_onset_err_ms"] = ver
            x = norm_to(r["audio"], LVL["harmony"], kind="harmony")
            self.dbg("harmony", f"{p['id']}_{v}", x)
            place(self.stems["harmony"], to44(x), plan["R0"] + SUNG_SHIFT + 0.004 * j, pan)
            out.append(dict(voice=v, pan=pan, pitch=pitch_check(r["audio"], r, plan)))
        e["harmony_midi"] = [tr(n[2]) for n in p["notes"]]
        e["renders"] = out
        e["asr"] = self.asr_check(r["audio"], p["lang"], p["text"]) if self.asr else None
        self.report["sung"].append(e)

    def crowd_voices(self, p):
        k = seed_of(p["id"])
        if p["lang"] == "es":
            g = [GANG[k % 4], GANG[(k + 1) % 4]]
            return LOCALS + g          # 3 locals + 2 gang
        rot = [GANG[(k + i) % len(GANG)] for i in range(len(GANG))]
        males = [v for v in rot if v[1] == "m"][:3]
        fem = [v for v in rot if v[1] == "f"][:1]
        return males + fem             # 4 gang voices

    def render_crowd(self, p):
        voices = self.crowd_voices(p)
        seed = seed_of(p["id"])
        rng = np.random.default_rng(seed)
        plan = self.plan_for(p, voices[0])
        n = len(voices)
        pans = np.linspace(-0.7, 0.7, n)
        rng.shuffle(pans)
        jc = int(np.argmin(np.abs(pans)))
        pans[[0, jc]] = pans[[jc, 0]]          # the leading voice sits nearest the centre
        dets = np.linspace(-10, 10, n)
        rng.shuffle(dets)
        # small offsets (0-10 ms) so the stack still lands together; the first voice (a local for Spanish lines)
        # leads the stack, 3 dB up, for intelligibility
        offs = rng.uniform(0, CROWD_MAX_OFFSET_MS, n)
        offs[0] = 0.0
        out, best = [], None
        e = self.sung_entry(p, plan, "vox_crowd", voices + ["am_onyx"])
        for j, v in enumerate(voices):
            r = self.E.render_sung(plan, v, detune_cents=float(dets[j]), seed=seed + j, vib_depth=0.15)
            ver = self.timing_entry(e, r, plan, v)
            if j == 0:
                e["vowel_onset_err_ms"] = ver
            if j == 0:
                x = norm_to(r["audio"], LVL["crowd"] + LVL["crowd_front"], kind="crowd_front")
            else:
                x = norm_to(r["audio"], LVL["crowd"], kind="crowd")
            self.dbg("crowd", f"{p['id']}_{v}", x)
            place(self.stems["crowd"], to44(x), plan["R0"] + SUNG_SHIFT + offs[j] / 1000.0, float(pans[j]))
            pc = pitch_check(r["audio"], r, plan)
            out.append(dict(voice=v, detune_cents=round(float(dets[j]), 1), offset_ms=round(float(offs[j]), 1),
                            pan=round(float(pans[j]), 2), pitch=pc))
            best = r["audio"] if best is None else best[:len(r["audio"])] + r["audio"][:len(best)]
        # quiet octave-below double for weight (deep male voice)
        ov = "am_onyx"
        r = self.E.render_sung(plan, ov, octave=-1, seed=seed + 50, vib_depth=0.1)
        # weight only: low-passed so it carries no consonants of its own, and well under the stack
        x = sosfilt(butter(4, 1500, "lowpass", fs=ve.KSR, output="sos"), np.asarray(r["audio"], np.float64))
        x = norm_to(x, LVL["crowd_oct"], kind="crowd_oct")
        self.dbg("crowd", f"{p['id']}_{ov}_oct", x)
        place(self.stems["crowd"], to44(x), plan["R0"] + SUNG_SHIFT + 0.006, 0.0)
        out.append(dict(voice=ov, octave=-1, pitch=pitch_check(r["audio"], r, plan)))
        e["renders"] = out
        e["asr"] = self.asr_check(best, p["lang"], p["text"])
        self.report["sung"].append(e)

    # ------------------------------------------------------------------------- spoken
    def spoken_voices(self, ln):
        k = seed_of(ln["id"])
        if ln["style"] == "chant":
            if ln["role"] == "gang":
                return ["am_adam", "am_puck", "af_heart"]
            return [LEAD, "am_adam", "am_onyx"]
        if ln["role"] == "locals":
            if ln["style"] == "whisper":
                return ["ef_dora", "em_alex"]
            return [LOCALS[k % len(LOCALS)]]
        if ln["role"] == "gang":
            return [GANG[k % len(GANG)]]
        return [LEAD]

    def render_spoken(self, ln):
        text = ln.get("tts_text") or ln["text"]
        voices = self.spoken_voices(ln)
        target = score.t(ln["bar"], ln["beat"])
        deadline = score.t(ln["bar"], ln["beat"] + ln["max_beats"])
        style = ln["style"]
        speed = {"spoken": 1.0, "chant": 1.08, "aside": 1.05, "whisper": 0.95}[style]
        budget = deadline - target - 0.05
        stem = "whisper" if style == "whisper" else "spoken"
        # Fit: the slowest Kokoro speed whose audible length (to -40 dB below peak) ends >= 30 ms before the
        # deadline; light WORLD compression (>= x0.9) is allowed from speed 1.35 on, stronger only at 1.75.
        avail = deadline - END_MARGIN_S - target
        sp, lens, chosen = None, [], None
        speeds = list(np.round(np.arange(speed, MAX_KOKORO_SPEED + 1e-6, 0.05), 3))
        for s_ in speeds:
            sp = self.E.speech_plan(text, ln["lang"], voices[0], speed=float(s_))
            xs = []
            for j, v in enumerate(voices):
                if style == "chant":
                    x = self.E.render_speech(sp, v, f0_mode="world", f0_flatten=0.45)
                else:
                    x = self.E.render_speech(sp, v)
                if style == "whisper":
                    x = whisperize(x)
                xs.append(x)
            mix = np.sum([x[:min(map(len, xs))] for x in xs], axis=0)
            on, _ = envelope_onset_end(mix, ve.KSR)
            _, end = envelope_onset_end(mix, ve.KSR, rel_db=FIT_REL_DB)
            dur_s = (end - on) / ve.KSR
            lens.append((float(s_), round(dur_s, 3)))
            fac = avail / dur_s
            if fac >= 1.0 or (s_ >= 1.35 - 1e-6 and fac >= MIN_TSM) or s_ == speeds[-1]:
                chosen = float(s_)
                if s_ != speed:
                    sp["log"].append(f"speed {s_:.2f}")
                if fac < 1.0:
                    xs = [world_tsm(x, fac) for x in xs]
                    sp["log"].append(f"WORLD time-compress x{fac:.2f}")
                    mix = np.sum([x[:min(map(len, xs))] for x in xs], axis=0)
                    on, _ = envelope_onset_end(mix, ve.KSR)
                    _, end = envelope_onset_end(mix, ve.KSR, rel_db=FIT_REL_DB)
                    lens.append(("tsm", round((end - on) / ve.KSR, 3)))
                break
        sp["speed"] = chosen
        start = target - on / ve.KSR
        gains = {"spoken": LVL["spoken"], "chant": LVL["chant"], "aside": LVL["aside"],
                 "whisper": LVL["whisper_locals"] if ln["role"] == "locals" else LVL["whisper"]}
        lvl = gains[style]
        if style == "aside":
            lvl = LVL["spoken"] - 6.0
        capk = style if style != "whisper" else ("whisper_locals" if ln["role"] == "locals" else "whisper")
        pans = {1: [ln["pan"]], 2: [-0.25, 0.25], 3: [0.0, -0.35, 0.35]}[len(voices)]
        if style in ("spoken", "aside") and len(voices) == 1:
            pans = [ln["pan"]]
        # edges: 4 ms raised-cosine fade-in at the first audible sound (no hard step), and a 20 ms raised-cosine
        # fade that reaches zero 10 ms before the deadline, so no tail can pass it
        on50, _ = envelope_onset_end(mix, ve.KSR, rel_db=END_REL_DB)
        n_dl = int(round((deadline - start) * ve.KSR))
        L = max(len(x) for x in xs)
        edge = np.ones(L)
        a0 = max(0, on50 - int(0.002 * ve.KSR))
        rin = int(0.004 * ve.KSR)
        edge[:a0] = 0.0
        edge[a0:a0 + rin] = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, rin))[:max(0, min(rin, L - a0))]
        z1 = n_dl - int(0.010 * ve.KSR)
        z0 = z1 - int(0.020 * ve.KSR)
        if z0 < L:
            edge[max(0, z1):] = 0.0
            seg = np.arange(max(0, z0), min(L, max(0, z1)))
            edge[seg] *= 0.5 + 0.5 * np.cos(np.pi * (seg - z0) / max(1, z1 - z0))
        outs = []
        for j, (v, x) in enumerate(zip(voices, xs)):
            xn = norm_to(x, lvl, kind=capk) * edge[:len(x)]
            outs.append(xn)
            self.dbg(stem, f"{ln['id']}_{v}", xn)
            place(self.stems[stem], to44(xn), start, pans[j])
        # measured on the final (faded) placement
        mix = np.sum([x[:min(map(len, outs))] for x in outs], axis=0)
        on2, _ = envelope_onset_end(mix, ve.KSR)
        _, end2 = envelope_onset_end(mix, ve.KSR, rel_db=END_REL_DB)
        e = dict(id=ln["id"], stem=f"vox_{stem}", bar=ln["bar"], beat=ln["beat"], style=style, role=ln["role"],
                 lang=ln["lang"], text=ln["text"], tts_text=ln.get("tts_text"), voices=voices, pans=pans,
                 phonemes=sp["ph"], target_onset_s=round(target, 4),
                 onset_s=round(start + on2 / ve.KSR, 4), onset_err_ms=round(1000 * (start + on2 / ve.KSR - target), 1),
                 end_s=round(start + end2 / ve.KSR, 4), deadline_s=round(deadline, 4),
                 margin_ms=round(1000 * (deadline - (start + end2 / ve.KSR)), 1), speed=sp["speed"],
                 fit_log=sp["log"], attempts=lens)
        e["asr"] = self.asr_check(mix, ln["lang"], ln["text"])
        if style != "whisper":
            e["asr_raw_tts"] = self.asr_raw(text, ln["lang"], voices[0], ln["text"])
        self.report["spoken"].append(e)
        return e

    # ------------------------------------------------------------------------- run
    def run(self):
        t0 = time.time()
        sung = [p for p in score.SUNG if not self.only or p["id"] in self.only]
        spoken = [s for s in score.SPOKEN if not self.only or s["id"] in self.only]
        for i, p in enumerate(sung):
            ts = time.time()
            if p["role"] == "lead":
                e = self.render_lead(p)
            else:
                self.render_crowd(p)
                e = self.report["sung"][-1]
            a = e.get("asr") or {}
            print(f"[{i + 1}/{len(sung)}] {p['id']:10s} {p['role']:5s} {time.time() - ts:5.1f}s "
                  f"onset err {[v for v in e.get('vowel_onset_err_ms', []) if v is not None]} "
                  f"asr {a.get('recall', '-')} {a.get('hyp', '')}", flush=True)
        for i, ln in enumerate(spoken):
            ts = time.time()
            e = self.render_spoken(ln)
            a = e.get("asr") or {}
            print(f"[{i + 1}/{len(spoken)}] {ln['id']:10s} {ln['style']:7s} {time.time() - ts:5.1f}s onset {e['onset_err_ms']:+.1f}ms "
                  f"margin {e['margin_ms']:.0f}ms asr {a.get('recall', '-')} {a.get('hyp', '')}", flush=True)
        self.write()
        print(f"done in {time.time() - t0:.0f}s")

    def stem_timing(self, stems):
        """Re-measure every measurable vowel onset on the finished stems (same detector as the audit)."""
        import vox_onset as vo
        x16 = {k: vo.to16(st.mean(axis=1), SR) for k, st in stems.items() if k in ("lead", "harmony", "crowd")}
        viol = []
        for e in self.report["sung"]:
            k = e["stem"][4:]
            errs = []
            for T, D, ok in zip(e["note_times_s"], e["note_durs_s"], e["measurable"]):
                if not ok:
                    errs.append(None)
                    continue
                m = vo.vowel_onset(x16[k], T, D)
                errs.append(None if m is None else round(1000 * (m - T), 1))
            e["stem_vowel_onset_err_ms"] = errs
            for j, v in enumerate(errs):
                if v is not None and abs(v) > TIMING_TOL_MS:
                    viol.append(dict(id=e["id"], stem=e["stem"], note=j, syllable=e["score_syllables"][j], err_ms=v))
        self.report["timing_violations"] = viol
        if viol:
            print(f"WARNING: {len(viol)} sung vowel onsets measured outside +-{TIMING_TOL_MS:.0f} ms on the stems:")
            for v in viol:
                print("   ", v)

    def write(self):
        os.makedirs(STEM_DIR, exist_ok=True)
        lim = 10 ** (-1.0 / 20)
        for k, st in self.stems.items():
            pk = float(np.max(np.abs(st)))
            g = 1.0
            if pk > lim * 0.98:
                g = lim * 0.98 / pk
            y = (st * g).astype(np.float32)
            assert y.shape == (NSAMP, 2)
            sf.write(os.path.join(STEM_DIR, f"vox_{k}.wav"), y, SR, subtype="FLOAT")
            mono = y.mean(axis=1)
            hop = SR // 10
            n = len(mono) // hop
            fr = np.sqrt(np.mean(mono[:n * hop].reshape(n, hop) ** 2, axis=1))
            act = fr > 10 ** (-60 / 20)
            self.report["stems"][f"vox_{k}"] = dict(
                peak_dbfs=round(20 * np.log10(np.max(np.abs(y)) + 1e-12), 2),
                rms_dbfs=round(20 * np.log10(np.sqrt(np.mean(y.astype(np.float64) ** 2)) + 1e-12), 2),
                active_rms_dbfs=round(20 * np.log10(np.sqrt(np.mean(fr[act] ** 2)) + 1e-12), 2) if act.any() else None,
                active_seconds=round(float(act.sum() * hop / SR), 1), gain_applied_db=round(20 * np.log10(g), 2),
                samples=int(y.shape[0]), channels=2, sr=SR)
        self.stem_timing({k: st for k, st in self.stems.items()})
        self.report["summary"] = summarize(self.report)
        with open(os.path.join(STEM_DIR, "vocals_report.json"), "w") as f:
            json.dump(self.report, f, indent=1, ensure_ascii=False, default=float)


def whisperize(x):
    """Whisper: WORLD resynthesis of the spoken render with all frames unvoiced (noise excitation through the
    voice's own spectral envelope), then a little breathy tilt."""
    import pyworld as pw
    x = np.asarray(x, np.float64)
    f0, t = pw.harvest(x, ve.KSR, f0_floor=60, f0_ceil=500, frame_period=5.0)
    sp = pw.cheaptrick(x, f0, t, ve.KSR)
    ap = np.ones_like(sp)
    y = pw.synthesize(np.zeros_like(f0), sp, ap, ve.KSR, 5.0)[:len(x)]
    y = np.pad(y, (0, len(x) - len(y)))
    hp = butter(2, 180, "highpass", fs=ve.KSR, output="sos")
    y = sosfilt(hp, y)
    hs = butter(1, 2500, "highpass", fs=ve.KSR, output="sos")
    y = y + 0.5 * sosfilt(hs, y)
    # keep the speech's own gaps silent (noise floor of the vocoder)
    env = np.abs(x)
    k = int(0.01 * ve.KSR)
    env = np.convolve(env, np.ones(k) / k, mode="same")
    gate = np.clip(env / (env.max() * 10 ** (-45 / 20)), 0, 1)
    return y * gate


def summarize(rep):
    s = {}
    sung = [e for e in rep["sung"]]
    for key, name in (("vowel_onset_err_ms", "render"), ("stem_vowel_onset_err_ms", "stem")):
        for stem in ("vox_lead", "vox_harmony", "vox_crowd"):
            ev = [v for e in sung if e["stem"] == stem for v in (e.get(key) or []) if v is not None]
            if ev:
                a = np.abs(ev)
                s[f"measured_vowel_onset_{name}_{stem}"] = dict(
                    n=len(ev), signed_mean=round(float(np.mean(ev)), 1), mean_abs=round(float(a.mean()), 1),
                    max_abs=round(float(a.max()), 1), within15=round(float(np.mean(a <= 15)), 3))
    s["timing_violations"] = len(rep.get("timing_violations", []))
    pcs = [e["pitch"] for e in sung if e.get("pitch")] + [r["pitch"] for e in sung for r in e.get("renders", []) if r.get("pitch")]
    meds = [p["med_cents"] for p in pcs if p["med_cents"] is not None]
    if meds:
        s["pitch_median_abs_cents"] = dict(median=round(float(np.median(meds)), 1), worst=round(float(np.max(meds)), 1),
                                           mean_within50=round(float(np.mean([p["within50"] for p in pcs])), 3))
    sp = rep["spoken"]
    if sp:
        oe = [abs(e["onset_err_ms"]) for e in sp]
        s["spoken_onset_abs_err_ms"] = dict(mean=round(float(np.mean(oe)), 2), max=round(float(np.max(oe)), 2))
        s["spoken_deadline_violations"] = [e["id"] for e in sp if e["margin_ms"] < 0]
        s["spoken_min_margin_ms"] = float(min(e["margin_ms"] for e in sp))
    a = [e for e in sung + sp if e.get("asr")]
    if a:
        s["asr_mean_recall"] = round(float(np.mean([e["asr"]["recall"] for e in a])), 3)
        s["asr_weak"] = [dict(id=e["id"], stem=e["stem"], text=e["text"], hyp=e["asr"]["hyp"], recall=e["asr"]["recall"],
                              raw_tts_recall=(e.get("asr_raw_tts") or {}).get("recall"))
                         for e in a if e["asr"]["recall"] < 0.75]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-asr", action="store_true")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    Vocals(asr=not a.no_asr, only=a.only.split(",") if a.only else None).run()


if __name__ == "__main__":
    main()
