"""LA LOCA TLACUAQUE DE LA CALLE CERRADA - vocal stems, rendered locally (no Suno).

    cd /home/user/cf336/songs/la-loca-tlacuaque/production
    WHY_LLARTA_WORK=<work folder> python3 vocals.py [--no-asr] [--only ID[,ID...]]

Reads score.py (RAP, ADLIBS, SUNG, SPOKEN, VOICES, t(), t16(), SONG_SECONDS) and writes six dry stereo
float32 stems (44.1 kHz, exactly round(SONG_SECONDS*44100) samples, sample 0 = score time 0) to
<work>/tlac/stems/:
    vox_rap.wav     the rapper's RAP lines (deep raspy lead, grid-locked flow)
    vox_punch.wav   two gang voices doubling every punch=True rhyme (RAP lines with another role than
                    'lead' are rapped by the gang into this stem as well)
    vox_hook.wav    sung hook, gang layer (men in unison at the written octave)
    vox_kids.wav    sung hook, kids layer (an octave up, raised formants)
    vox_adlibs.wav  ad-libs and crowd answers (gang shouts and the rapper)
    vox_spoken.wav  the rapper's SPOKEN lines
    vocals_report.json
Isolated per-phrase renders go to <work>/tlac/vox_debug/<stem>/. With --only the flow is still computed for
the whole verse (so the chosen lines sit where they would in the full song), but the partial stems and
report are written to <work>/tlac/vox_debug/partial/ and the full stems are left alone.

Everything is driven by score.py data; nothing here knows a line id, a bar or a text.
Method: see rap_engine.py (flow, rapper voice, rasp, punch-ins, hook, speech path).
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, butter, sosfilt

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
LVL = dict(rap=-18.0, punch=-25.0, gang_rap=-22.0, hook_front=-21.0, hook=-24.0, kids=-26.0, adlib_gang=-22.5,
           adlib_lead=-19.0, spoken=-19.0, whisper=-22.0)
PEAK_CAP = dict(rap=-4.0, punch=-10.0, gang_rap=-8.0, hook_front=-7.0, hook=-10.0, kids=-11.0, adlib_gang=-8.0,
                adlib_lead=-4.5, spoken=-4.5, whisper=-7.0)
KIDS_OCTAVE = 1
KIDS_FORMANTS = (1.16, 1.12, 1.20)
HOOK_MAX_OFFSET_MS = 9.0       # timing spread of the gang (the first voice leads at 0)
KIDS_MAX_OFFSET_MS = 4.0
PUNCH_OFFSETS_MS = (0.0, 5.0)
TOL_MS = 20.0                  # rap / hook vowel onsets
RHYME_TOL_MS = 15.0
SPEECH_TOL_MS = 25.0
FIT_REL_DB = -40.0             # words must be over (40 dB down) before the deadline fade starts
END_REL_DB = -50.0             # "ended" = 50 dB below the phrase's peak
END_MARGIN_S = 0.03
MIN_TSM = 0.9
MAX_KOKORO_SPEED = 1.75
SHOUT = dict(speed=1.12, f0_scale=1.18, f0_range=1.25)
LEAD_SHOUT = dict(speed=1.05, f0_scale=1.12, f0_range=1.15)


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


def norm_to(x, kind, sr=ve.KSR, trim=0.0):
    y = x * 10 ** ((LVL[kind] + trim - active_rms_db(x, sr)) / 20)
    return limit(y, PEAK_CAP[kind] + trim, sr)


def pan_gains(p):
    a = (np.clip(p, -1, 1) + 1) * np.pi / 4
    return np.cos(a) * np.sqrt(2), np.sin(a) * np.sqrt(2)   # unity at centre, constant power


def envelope_onset_end(x, sr, rel_db=-32.0, hop_s=0.003):
    hop = max(1, int(hop_s * sr))
    n = len(x) // hop
    fr = np.sqrt(np.mean(x[:n * hop].reshape(n, hop) ** 2, axis=1) + 1e-20)
    thr = fr.max() * 10 ** (rel_db / 20)
    idx = np.nonzero(fr > thr)[0]
    if len(idx) == 0:
        return 0, len(x)
    return int(idx[0] * hop), int((idx[-1] + 1) * hop)


def world_tsm(x, factor, sr=ve.KSR):
    """WORLD time-scale modification (factor < 1 shortens), pitch and formants kept."""
    import pyworld as pw
    x = np.asarray(x, np.float64)
    f0, t = pw.harvest(x, sr, f0_floor=50, f0_ceil=600, frame_period=5.0)
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
    return pw.synthesize(f02, np.ascontiguousarray(sp2), np.ascontiguousarray(ap2), sr, 5.0)


def bright(x, sr=ve.KSR, db=4.0, fc=3200.0):
    """High shelf for the kids layer."""
    hi = sosfilt(butter(1, fc, "highpass", fs=sr, output="sos"), x)
    return x + (10 ** (db / 20) - 1.0) * hi


def r1(v):
    return None if v is None else round(float(v), 1)


# ----------------------------------------------------------------------------- the renderer
class Vocals:
    def __init__(self, asr=True, only=None):
        self.E = R.RapEngine(cache_dir=CACHE_DIR, memo_dir=MEMO_DIR)
        self.cfg = dict(R.RAPPER, voice=LEAD)
        self.stems = {k: np.zeros((NSAMP, 2), np.float64) for k in STEMS}
        self.regions = {k: [] for k in STEMS}          # (start_s, end_s, id) where the score puts sound
        self.edges = []                                 # level at the edges of every placed render
        self.report = dict(rap=[], hook=[], adlibs=[], spoken=[], stems={}, notes=[])
        self.only = set(only) if only else None
        self.asr = asr
        self.times = {}
        if asr:
            import vox_asr
            self.vox_asr = vox_asr
            if not vox_asr.available():
                print("ASR model missing; skipping intelligibility checks")
                self.asr = False
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
        pk = np.max(np.abs(cut)) + 1e-12
        m = max(1, int(0.0005 * SR))
        self.edges.append(max(np.max(np.abs(cut[:m])), np.max(np.abs(cut[-m:]))) / pk)

    def asr_check(self, x, lang, ref, sr=ve.KSR):
        if not self.asr:
            return None
        x = np.asarray(x, np.float32)

        def run():
            return np.zeros(1), dict(hyp=self.vox_asr.transcribe(x, sr, lang))
        key = hashlib.sha1(x.tobytes()).hexdigest() + f"|{sr}|{lang}"
        _, m = self.E.memo("asr", key, run)
        return dict(hyp=m["hyp"], **R.asr_scores(ref, m["hyp"], lang))

    def asr_raw(self, text, lang, voice, ref):
        """The same words as plain Kokoro speech (same Mexican phonemes, natural rhythm, speed 1.0)."""
        if not self.asr:
            return None
        ph = self.E.phonemes(text, lang)
        a, _ = self.E.k.create(ph, voice.split("|")[0], speed=1.0, is_phonemes=True)
        return self.asr_check(a, lang, ref)

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
        return dict(id=ln["id"], stem=stem, bar=ln["bar"], role=ln["role"], text=ln["text"], tts_text=ln.get("tts_text"),
                    phonemes=plan["ph"], anchor=ln["anchor"], lo=ln["lo"], hi=ln["hi"], n_syllables=len(f["T"]),
                    flow=R.flow_table(f), crowded=bool(f["crowded"]), thirty_second_pairs=f["pairs"],
                    flow_log=f["log"], first_slot=f["pos"][0], slot_times_s=[round(t, 4) for t in f["T"]],
                    note_durs_s=[round(d, 4) for d in f["D"]], target_midi=[round(m, 2) for m in plan["midi"]],
                    measurable=[bool(E.measurable(plan, k)) for k in range(len(plan["sylls"]))],
                    rise_measurable=[bool(not E.measurable(plan, k) and E.has_consonant_before(plan, k))
                                     for k in range(len(plan["sylls"]))],
                    rhyme_syllable=int(f["rhyme"]), planned_end_s=round(f["end_s"], 4),
                    deadline_s=round(f["deadline_s"], 4))

    def render_rap(self, idx, ln, f):
        E = self.E
        seed = R.seed_of(ln["id"])
        lead = ln["role"] == "lead"
        if lead:
            r = E.render_rap(f, self.cfg, seed=seed)
            x = norm_to(r["audio"], "rap")
            self.dbg("rap", ln["id"], x)
            self.place("rap", x, r["R0"], 0.0, ln["id"])
            stem, mix = "rap", x
            voices = [LEAD]
        else:
            # a RAP line given to the gang: two gang voices rap it on the grid (no rasp), into vox_punch
            voices = self.punch_voices(idx)
            mix = None
            for j, v in enumerate(voices):
                r = E.render_punch(dict(f, rhyme=0), v, self.cfg, seed=seed + j, detune=(-6.0, 6.0)[j % 2], pitch_up=R.PUNCH_UP)
                x = norm_to(r["audio"], "gang_rap")
                self.dbg("punch", f"{ln['id']}_{v}", x)
                self.place("punch", x, r["R0"] + 0.004 * j, (-0.3, 0.3)[j % 2], ln["id"])
                mix = x if mix is None else mix[:len(x)] + x[:len(mix)]
            stem = "punch"
        plan = r["plan"]
        e = self.rap_entry(ln, f, plan, f"vox_{stem}")
        e["voices"] = voices
        e["render_onset_err_ms"] = r["meta"]["verify_per_ms"]
        e["render_rise_err_ms"] = r["meta"]["verify_rise_ms"]
        e["second_pass_ms"] = [t["second_pass_ms"] for t in r["meta"]["timing"]]
        # audible end of the isolated render against the deadline and the next line
        _, end = envelope_onset_end(mix, ve.KSR, rel_db=-40.0)
        e["end_s"] = round(r["R0"] + end / ve.KSR, 4)
        e["deadline_margin_ms"] = r1(1000 * (f["deadline_s"] - e["end_s"]))
        mt = R.rasp_metrics(mix)
        e["f0_median_hz"] = mt["f0_med"]
        e["f0_target_median_hz"] = r1(r["meta"]["f0_target_med"])
        e["rasp"] = {k: mt[k] for k in ("hnr_db", "periodicity", "shr_db", "nhr_1_4k_db")}
        e["asr"] = self.asr_check(mix, ln["lang"], ln["text"])
        e["asr_raw_tts"] = self.asr_raw(f["tts_text"], ln["lang"], LEAD, ln["text"])
        self.regions[stem].append((f["T"][0] - 0.16, f["deadline_s"], ln["id"]))
        if lead and ln.get("punch"):
            pv = self.punch_voices(idx)
            pe = dict(voices=pv, from_syllable=int(f["rhyme"]), syllables=f["strings"][f["rhyme"]:], renders=[])
            for j, v in enumerate(pv):
                rp = E.render_punch(f, v, self.cfg, seed=seed + 11 + j, detune=(-7.0, 7.0)[j % 2])
                xp = norm_to(rp["audio"], "punch")
                off = PUNCH_OFFSETS_MS[j % 2] / 1000.0
                pan = (-0.45, 0.45)[j % 2]
                self.dbg("punch", f"{ln['id']}_{v}", xp)
                self.place("punch", xp, rp["R0"] + off, pan, ln["id"] + "_punch")
                on, pend = envelope_onset_end(xp, ve.KSR, rel_db=-40.0)
                pe["renders"].append(dict(voice=v, pan=pan, offset_ms=PUNCH_OFFSETS_MS[j % 2], detune_cents=(-7.0, 7.0)[j % 2],
                                          onset_err_ms=rp["meta"]["verify_per_ms"], rise_err_ms=rp["meta"]["verify_rise_ms"],
                                          start_s=round(rp["R0"] + off + on / ve.KSR, 4),
                                          end_s=round(rp["R0"] + off + pend / ve.KSR, 4)))
            pe["plan_measurable"] = [bool(E.measurable(rp["plan"], k)) for k in range(len(rp["plan"]["sylls"]))]
            pe["rise_measurable"] = [bool(not E.measurable(rp["plan"], k) and E.has_consonant_before(rp["plan"], k))
                                     for k in range(len(rp["plan"]["sylls"]))]
            e["punch"] = pe
            self.regions["punch"].append((f["T"][f["rhyme"]] - 0.16, f["deadline_s"], ln["id"] + "_punch"))
        self.report["rap"].append(e)
        return e

    # ------------------------------------------------------------------------- hook
    def hook_gang(self):
        m = [v for v in GANG if v[1] == "m"]
        f = [v for v in GANG if v[1] != "m"]
        v = (m + f)[:4] if len(m) >= 3 else (m + f)[:max(3, min(4, len(m) + len(f)))]
        return sorted(v, key=lambda x: 0 if x[0] == "e" else 1)        # Spanish voices lead

    def hook_entry(self, p, plan, stem, voices):
        E = self.E
        notes = [n for s in plan["sylls"] for n in s["notes"][:1]]
        return dict(id=p["id"], stem=stem, bar=p["bar"], role=p["role"], text=p["text"], tts_text=p.get("tts_text"),
                    phonemes=plan["ph"], n_notes=len(p["notes"]), n_syllables=len(plan["sylls"]),
                    natural_syllables=plan["natural_syllables"], reconciliation=plan["slog"],
                    syllables=E.syl_strings(plan["chars"], plan["sylls"]), score_syllables=[n[3] for n in p["notes"]],
                    voices=voices, note_times_s=[round(n["T"], 4) for n in notes],
                    note_durs_s=[round(n["D"], 4) for n in notes], note_midi=[n["m"] for n in notes],
                    measurable=[bool(E.measurable(plan, k)) for k in range(len(plan["sylls"]))],
                    rise_measurable=[bool(not E.measurable(plan, k) and E.has_consonant_before(plan, k))
                                     for k in range(len(plan["sylls"]))])

    def render_hook(self, p):
        E = self.E
        seed = R.seed_of(p["id"])
        rng = np.random.default_rng(seed)
        last = max(score.t(p["bar"], b + d) for (b, d, _, _) in p["notes"])
        first = min(score.t(p["bar"], b) for (b, _, _, _) in p["notes"])
        out = []
        layers = []
        if p["role"] in ("gang", "lead"):
            layers.append("gang")
        if p["role"] in ("gang", "kids"):
            layers.append("kids")
        if not layers:
            layers = ["gang"]
        for layer in layers:
            if layer == "gang":
                voices = self.hook_gang() if p["role"] != "lead" else [LEAD]
                n = len(voices)
                pans = [0.0] if n == 1 else list(np.linspace(-0.75, 0.75, n))
                # Spanish voices (first in the list) take the inner positions
                order = sorted(range(n), key=lambda i: abs(pans[i]))
                pan_of = {voices[j]: float(pans[order[j]]) for j in range(n)}
                dets = np.linspace(-9, 9, n) if n > 1 else np.zeros(1)
                rng.shuffle(dets)
                offs = rng.uniform(2.0, HOOK_MAX_OFFSET_MS, n)
                offs[0] = 0.0
                octave, stem, fsh = 0, "hook", [1.0] * n
            else:
                voices = list(KIDS)
                n = len(voices)
                pans = [0.0] if n == 1 else list(np.linspace(-0.5, 0.5, n))
                order = sorted(range(n), key=lambda i: abs(pans[i]))
                pan_of = {voices[j]: float(pans[order[j]]) for j in range(n)}
                dets = np.linspace(-6, 6, n) if n > 1 else np.zeros(1)
                rng.shuffle(dets)
                offs = rng.uniform(0.0, KIDS_MAX_OFFSET_MS, n)
                offs[0] = 0.0
                octave, stem = KIDS_OCTAVE, "kids"
                fsh = [KIDS_FORMANTS[j % len(KIDS_FORMANTS)] for j in range(n)]
            plan = E.hook_plan(score, p, voices[0])
            e = self.hook_entry(p, plan, f"vox_{stem}", voices)
            e["layer"] = layer
            mix = None
            rs = []
            for j, v in enumerate(voices):
                r = E.render_hook(plan, v, seed=seed + j, detune=float(dets[j]), octave=octave, formant_shift=fsh[j])
                x = r["audio"]
                if layer == "kids":
                    x = norm_to(bright(x), "kids")
                else:
                    x = norm_to(x, "hook_front" if j == 0 else "hook", trim=0.0 if v[0] == "e" else -1.5)
                self.dbg(stem, f"{p['id']}_{v}", x)
                self.place(stem, x, r["R0"] + offs[j] / 1000.0, pan_of[v], p["id"])
                sh = int(round(offs[j] / 1000.0 * ve.KSR))
                xs = np.concatenate([np.zeros(sh), x])
                mix = xs if mix is None else mix[:len(xs)] + xs[:len(mix)]
                rs.append(dict(voice=v, pan=round(pan_of[v], 2), detune_cents=r1(dets[j]), offset_ms=r1(offs[j]),
                               octave=octave, formant_shift=fsh[j], pitch=r["meta"]["pitch"],
                               onset_err_ms=r["meta"]["verify_per_ms"], rise_err_ms=r["meta"]["verify_rise_ms"]))
            e["renders"] = rs
            _, end = envelope_onset_end(mix, ve.KSR, rel_db=-40.0)
            e["end_s"] = round(plan["R0"] + end / ve.KSR, 4)
            e["asr"] = self.asr_check(mix, p["lang"], p["text"])
            if layer == "gang":
                e["asr_raw_tts"] = self.asr_raw(p.get("tts_text") or p["text"], p["lang"], voices[0], p["text"])
            self.regions[stem].append((first - 0.2, last + 0.12, p["id"]))
            self.report["hook"].append(e)
            out.append(e)
        return out

    # ------------------------------------------------------------------------- speech path
    def fit_speech(self, text, lang, voices, mode, avail, par, seed):
        """The slowest Kokoro speed whose audible length (to -40 dB) fits `avail` seconds; light WORLD
        compression (>= x0.9) is allowed from speed 1.35 on, stronger only at the top speed.
        Returns (list of arrays 24 kHz, info)."""
        E = self.E

        def run():
            s = float(par["speed"])
            attempts, log = [], []
            while True:
                xs, sp = E.speech_once(text, lang, voices, s, mode, self.cfg, seed, par["f0_scale"], par["f0_range"])
                L = min(map(len, xs))
                mix = np.sum([x[:L] for x in xs], axis=0)
                on, _ = envelope_onset_end(mix, ve.KSR)
                _, end = envelope_onset_end(mix, ve.KSR, rel_db=FIT_REL_DB)
                dur = (end - on) / ve.KSR
                attempts.append((round(s, 3), round(dur, 3)))
                fac = avail / max(dur, 1e-3)
                if fac >= 1.0 or (s >= 1.35 - 1e-6 and fac >= MIN_TSM) or s >= MAX_KOKORO_SPEED - 1e-6:
                    break
                # durations scale about 1/speed: jump most of the way, then step
                s = min(MAX_KOKORO_SPEED, max(s + 0.05, np.floor(s / fac * 0.97 * 20) / 20))
            if s != par["speed"]:
                log.append(f"speed {s:.2f}")
            if fac < 1.0:
                xs = [world_tsm(x, fac) for x in xs]
                log.append(f"WORLD time-compress x{fac:.2f}")
            L = max(map(len, xs))
            arr = np.stack([np.pad(x, (0, L - len(x))) for x in xs])
            return arr, dict(speed=s, attempts=attempts, log=log, ph=sp["ph"])
        key = dict(text=text, lang=lang, voices=voices, mode=mode, avail=round(avail, 5), par=par, seed=seed,
                   cfg=self.cfg if mode == "rapper" else None)
        arr, info = E.memo("speech", key, run)
        return [np.asarray(a, np.float64) for a in np.atleast_2d(arr)], info

    def render_phrase(self, item, kind, target, deadline):
        """An ad-lib or a spoken line: starts on `target`, over before `deadline`."""
        text = item.get("tts_text") or item["text"]
        style, role = item["style"], item["role"]
        seed = R.seed_of(item["id"])
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
                voices = [pool[k]]
            mode = "whisper" if style == "whisper" else "plain"
            par = dict(SHOUT) if style == "shout" else dict(speed=1.0, f0_scale=1.0, f0_range=1.0)
            lvl = "adlib_gang"
        if style == "whisper":
            par = dict(speed=0.95, f0_scale=1.0, f0_range=1.0)
            lvl = "whisper"
        stem = kind
        avail = deadline - END_MARGIN_S - target
        xs, info = self.fit_speech(text, item["lang"], voices, mode, avail, par, seed)
        L = min(map(len, xs))
        mix = np.sum([x[:L] for x in xs], axis=0)
        on, _ = envelope_onset_end(mix, ve.KSR)
        start = target - on / ve.KSR
        # edges: 4 ms fade-in at the first audible sound, 20 ms fade reaching zero 10 ms before the deadline
        on50, _ = envelope_onset_end(mix, ve.KSR, rel_db=END_REL_DB)
        n_dl = int(round((deadline - start) * ve.KSR))
        Lm = max(map(len, xs))
        edge = np.ones(Lm)
        a0 = max(0, on50 - int(0.002 * ve.KSR))
        rin = int(0.004 * ve.KSR)
        edge[:a0] = 0.0
        seg = np.arange(a0, min(Lm, a0 + rin))
        edge[seg] = 0.5 - 0.5 * np.cos(np.pi * (seg - a0) / rin)
        z1 = n_dl - int(0.010 * ve.KSR)
        z0 = z1 - int(0.020 * ve.KSR)
        if z0 < Lm:
            edge[max(0, z1):] = 0.0
            seg = np.arange(max(0, z0), min(Lm, max(0, z1)))
            edge[seg] *= 0.5 + 0.5 * np.cos(np.pi * (seg - z0) / max(1, z1 - z0))
        n = len(voices)
        base_pan = float(item.get("pan", 0.0))
        pans = [base_pan] if n == 1 else [float(np.clip(base_pan + d, -1, 1)) for d in ([0.0, -0.4, 0.4] if n == 3 else [-0.3, 0.3])]
        outs = []
        for j, (v, x) in enumerate(zip(voices, xs)):
            xn = norm_to(x, lvl, trim=0.0 if (v[0] == "e" or n == 1) else -2.0) * edge[:len(x)]
            outs.append(xn)
            self.dbg(stem, f"{item['id']}_{v}", xn)
            self.place(stem, xn, start, pans[j], item["id"])
        L = min(map(len, outs))
        mix = np.sum([x[:L] for x in outs], axis=0)
        on2, _ = envelope_onset_end(mix, ve.KSR)
        _, end2 = envelope_onset_end(mix, ve.KSR, rel_db=END_REL_DB)
        e = dict(id=item["id"], stem=f"vox_{stem}", bar=item["bar"], style=style, role=role, text=item["text"],
                 tts_text=item.get("tts_text"), voices=voices, pans=[round(p, 2) for p in pans], mode=mode,
                 phonemes=info["ph"], target_onset_s=round(target, 4), onset_s=round(start + on2 / ve.KSR, 4),
                 onset_err_ms=r1(1000 * (start + on2 / ve.KSR - target)), end_s=round(start + end2 / ve.KSR, 4),
                 deadline_s=round(deadline, 4), margin_ms=r1(1000 * (deadline - (start + end2 / ve.KSR))),
                 speed=info["speed"], fit_log=info["log"], attempts=info["attempts"])
        if kind == "adlibs":
            e["slot"], e["slots"] = item["slot"], item["slots"]
        else:
            e["beat"], e["max_beats"] = item["beat"], item["max_beats"]
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
            pm = [v for v in e["render_onset_err_ms"] if v is not None]
            print(f"[rap {n + 1}/{len(todo)}] {ln['id']:6s} {time.time() - tl:5.1f}s syl {e['n_syllables']:2d}"
                  f"{' CROWDED' if e['crowded'] else ''} onsets {pm} f0 {e['f0_median_hz']} asr {a.get('recall', '-')} "
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
                      f"syl {e['natural_syllables']} {e['reconciliation'] or ''} cents {med} asr {a.get('recall', '-')} {a.get('hyp', '')}",
                      flush=True)
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
        ts = time.time()
        self.rasp_reference(flows, todo)
        self.times["rasp_reference_s"] = round(time.time() - ts, 1)
        self.write(flows, t0)

    def rasp_reference(self, flows, todo, n=8):
        """Clean-vs-raspy A/B on up to n rap lines: the same lines rendered without the rasp recipe."""
        E = self.E
        ids = [i for i in todo if score.RAP[i]["role"] == "lead"][:n]
        if not ids:
            return
        clean_cfg = dict(self.cfg, rasp=None)
        rows = []
        by_id = {e["id"]: e for e in self.report["rap"]}
        for i in ids:
            ln, f = score.RAP[i], flows[i]
            r = E.render_rap(f, clean_cfg, seed=R.seed_of(ln["id"]))
            x = norm_to(r["audio"], "rap")
            mt = R.rasp_metrics(x)
            a = self.asr_check(x, ln["lang"], ln["text"])
            e = by_id[ln["id"]]
            rows.append(dict(id=ln["id"], clean={k: mt[k] for k in ("hnr_db", "periodicity", "shr_db", "nhr_1_4k_db", "f0_med")},
                             raspy=dict(e["rasp"], f0_med=e["f0_median_hz"]),
                             asr_clean=a and a["recall"], asr_raspy=e["asr"] and e["asr"]["recall"]))

        def mean(side, k):
            v = [r[side][k] for r in rows if r[side].get(k) is not None]
            return round(float(np.mean(v)), 2) if v else None
        self.report["rasp_ab"] = dict(
            lines=rows, n=len(rows), recipe=self.cfg.get("rasp"),
            clean={k: mean("clean", k) for k in ("hnr_db", "periodicity", "shr_db", "nhr_1_4k_db", "f0_med")},
            raspy={k: mean("raspy", k) for k in ("hnr_db", "periodicity", "shr_db", "nhr_1_4k_db", "f0_med")},
            asr_recall_clean=round(float(np.mean([r["asr_clean"] for r in rows])), 3) if self.asr else None,
            asr_recall_raspy=round(float(np.mean([r["asr_raspy"] for r in rows])), 3) if self.asr else None)

    # ------------------------------------------------------------------------- measurements on the stems
    def stem_timing(self, stems16):
        """Re-measure every vowel onset on the finished stems: the periodicity detector of the shared engine
        where the vowel follows an unvoiced consonant or a rest, the energy-rise detector where it follows a
        voiced consonant."""
        for e in self.report["rap"]:
            x16 = stems16[e["stem"][4:]]
            per, rise = [], []
            for T, D, ok, rk in zip(e["slot_times_s"], e["note_durs_s"], e["measurable"], e["rise_measurable"]):
                m = vo.vowel_onset(x16, T, D) if ok else None
                per.append(None if m is None else r1(1000 * (m - T)))
                m = R.rise_onset(x16, T) if rk else None
                rise.append(None if m is None else r1(1000 * (m - T)))
            e["stem_onset_err_ms"] = per
            e["stem_rise_err_ms"] = rise
            k = e["rhyme_syllable"]
            if per[k] is not None:
                e["rhyme_err_ms"] = dict(value=per[k], detector="periodicity")
            elif rise[k] is not None:
                e["rhyme_err_ms"] = dict(value=rise[k], detector="energy-rise")
            else:
                e["rhyme_err_ms"] = dict(value=None, detector=None)
        for e in self.report["hook"]:
            x16 = stems16[e["stem"][4:]]
            per, rise = [], []
            for T, D, ok, rk in zip(e["note_times_s"], e["note_durs_s"], e["measurable"], e["rise_measurable"]):
                m = vo.vowel_onset(x16, T, D) if ok else None
                per.append(None if m is None else r1(1000 * (m - T)))
                m = R.rise_onset(x16, T) if rk else None
                rise.append(None if m is None else r1(1000 * (m - T)))
            e["stem_onset_err_ms"] = per
            e["stem_rise_err_ms"] = rise

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
                on = np.nonzero(fr > np.sqrt(np.mean(seg[d0:d1] ** 2)) * 0 + fr[d0 // hop:d1 // hop].max() * 10 ** (-32 / 20))[0]
                e["stem_onset_err_ms"] = r1(1000 * ((a + on[0] * hop) / SR - e["target_onset_s"])) if len(on) else None
                tail = seg[max(0, d1):min(len(seg), d1 + int(0.05 * SR))]
                e["stem_level_after_deadline_db"] = r1(20 * np.log10((tail.max() if len(tail) else 0.0) / pk + 1e-12))

    def write(self, flows, t0):
        partial = self.only is not None
        out_dir = os.path.join(DEBUG_DIR, "partial") if partial else STEM_DIR
        os.makedirs(out_dir, exist_ok=True)
        lim = 10 ** (-1.0 / 20)
        final = {}
        for k, st in self.stems.items():
            if not np.all(np.isfinite(st)):
                raise RuntimeError(f"stem {k} has NaN/inf")
            pk = float(np.max(np.abs(st)))
            g = 1.0
            if pk > lim * 0.98:
                g = lim * 0.98 / pk
            y = (st * g).astype(np.float32)
            assert y.shape == (NSAMP, 2)
            final[k] = y
            sf.write(os.path.join(out_dir, f"vox_{k}.wav"), y, SR, subtype="FLOAT")
            mono = y.mean(axis=1)
            hop = SR // 10
            n = len(mono) // hop
            fr = np.sqrt(np.mean(mono[:n * hop].reshape(n, hop) ** 2, axis=1))
            act = fr > 10 ** (-60 / 20)
            # anything outside the regions where the score puts this stem's vocals?
            mask = np.zeros(NSAMP, bool)
            for (a, b, _) in self.regions[k]:
                mask[max(0, int(a * SR)):min(NSAMP, int(np.ceil(b * SR)))] = True
            outside = np.abs(y[~mask]).max() if (~mask).any() else 0.0
            self.report["stems"][f"vox_{k}"] = dict(
                peak_dbfs=round(20 * np.log10(np.max(np.abs(y)) + 1e-12), 2),
                rms_dbfs=round(20 * np.log10(np.sqrt(np.mean(y.astype(np.float64) ** 2)) + 1e-12), 2),
                active_rms_dbfs=round(20 * np.log10(np.sqrt(np.mean(fr[act] ** 2)) + 1e-12), 2) if act.any() else None,
                active_seconds=round(float(act.sum() * hop / SR), 1), gain_applied_db=round(20 * np.log10(g), 2),
                samples=int(y.shape[0]), channels=2, sr=SR, nan=False,
                clipped_samples=int(np.sum(np.abs(y) >= 1.0)),
                outside_score_regions_peak_dbfs=round(20 * np.log10(outside + 1e-12), 1),
                max_sample_step=round(float(np.max(np.abs(np.diff(y, axis=0)))), 4) if len(y) > 1 else 0.0,
                silent=bool(pk == 0.0))
        self.stem_timing({k: vo.to16(final[k].mean(axis=1), SR) for k in ("rap", "punch", "hook", "kids")})
        self.stem_phrases({k: final[k].mean(axis=1) for k in ("adlibs", "spoken")})
        self.times["total_s"] = round(time.time() - t0, 1)
        self.report["engine"] = dict(rapper=self.cfg, voices=dict(lead=LEAD, gang=GANG, kids=KIDS), version=R.VERSION,
                                     kids_octave=KIDS_OCTAVE, kids_formants=KIDS_FORMANTS, punch_up_semitones=R.PUNCH_UP,
                                     memo_hits=self.E.memo_hits, memo_misses=self.E.memo_miss)
        self.report["score"] = dict(title=score.TITLE, bpm=score.BPM, swing=score.SWING, lyrics_final=score.LYRICS_FINAL,
                                    song_seconds=score.SONG_SECONDS, rap_lines=len(score.RAP), sung_lines=len(score.SUNG),
                                    adlibs=len(score.ADLIBS), spoken=len(score.SPOKEN), partial=sorted(self.only) if partial else None)
        self.report["render_seconds"] = self.times
        self.report["edge_level_max_rel_db"] = r1(20 * np.log10(max(self.edges) + 1e-12)) if self.edges else None
        self.report["summary"] = summarize(self.report, flows)
        with open(os.path.join(out_dir, "vocals_report.json"), "w") as fh:
            json.dump(self.report, fh, indent=1, ensure_ascii=False, default=R._jsonable)
        s = self.report["summary"]
        print(json.dumps(s, indent=1, ensure_ascii=False, default=R._jsonable))
        print(f"stems -> {out_dir}   ({self.times['total_s']:.0f} s, cache hits {self.E.memo_hits}, misses {self.E.memo_miss})")


def _stats(vals, tol):
    vals = [v for v in vals if v is not None]
    if not vals:
        return dict(n=0)
    a = np.abs(vals)
    return dict(n=len(vals), within=round(float(np.mean(a <= tol)), 3), tol_ms=tol, mean_abs=round(float(a.mean()), 1),
                max_abs=round(float(a.max()), 1), signed_mean=round(float(np.mean(vals)), 1))


def summarize(rep, flows):
    s = {}
    rap = rep["rap"]
    lead = [e for e in rap if e["stem"] == "vox_rap"]
    if rap:
        s["rap_onsets_periodicity"] = _stats([v for e in rap for v in e["stem_onset_err_ms"]], TOL_MS)
        s["rap_onsets_energy_rise"] = _stats([v for e in rap for v in e["stem_rise_err_ms"]], TOL_MS)
        s["rap_onsets_all"] = _stats([v for e in rap for v in e["stem_onset_err_ms"] + e["stem_rise_err_ms"]], TOL_MS)
        s["rap_onset_violations"] = [dict(id=e["id"], syllable=e["flow"][j]["syl"], slot=e["flow"][j]["slot"], err_ms=v, detector=d)
                                     for e in rap for d, key in (("periodicity", "stem_onset_err_ms"), ("energy-rise", "stem_rise_err_ms"))
                                     for j, v in enumerate(e[key]) if v is not None and abs(v) > TOL_MS]
        rh = [e["rhyme_err_ms"] for e in rap]
        s["rhyme_stress"] = dict(_stats([r["value"] for r in rh], RHYME_TOL_MS),
                                 by_periodicity=sum(r["detector"] == "periodicity" for r in rh),
                                 by_energy_rise=sum(r["detector"] == "energy-rise" for r in rh),
                                 unmeasurable=[e["id"] for e in rap if e["rhyme_err_ms"]["value"] is None],
                                 outside=[dict(id=e["id"], err_ms=e["rhyme_err_ms"]["value"], detector=e["rhyme_err_ms"]["detector"])
                                          for e in rap if e["rhyme_err_ms"]["value"] is not None
                                          and abs(e["rhyme_err_ms"]["value"]) > RHYME_TOL_MS])
        s["crowded_lines"] = [dict(id=e["id"], text=e["text"], n_syllables=e["n_syllables"], pairs=e["thirty_second_pairs"],
                                   log=e["flow_log"]) for e in rap if e["crowded"]]
        s["rap_deadline_violations"] = [dict(id=e["id"], margin_ms=e["deadline_margin_ms"]) for e in rap if e["deadline_margin_ms"] < 0]
        # overlaps: a line's audible end (-40 dB) against the next line's first slot
        by_time = sorted(rap, key=lambda e: e["slot_times_s"][0])
        ov = []
        for a, b in zip(by_time[:-1], by_time[1:]):
            gap = b["slot_times_s"][0] - a["end_s"]
            if gap < 0:
                ov.append(dict(line=a["id"], next=b["id"], overlap_ms=r1(-1000 * gap)))
        s["rap_overlaps"] = ov
        s["rap_min_gap_to_next_line_ms"] = r1(min([1000 * (b["slot_times_s"][0] - a["end_s"]) for a, b in zip(by_time[:-1], by_time[1:])],
                                                  default=0.0)) if len(by_time) > 1 else None
        if lead:
            f0 = [e["f0_median_hz"] for e in lead if e["f0_median_hz"]]
            s["rapper_f0_median_hz"] = dict(median=round(float(np.median(f0)), 1), min=min(f0), max=max(f0))
        a = [e for e in lead if e.get("asr")]
        if a:
            rec = float(np.mean([e["asr"]["recall"] for e in a]))
            raw = float(np.mean([e["asr_raw_tts"]["recall"] for e in a]))
            s["rap_asr"] = dict(recall=round(rec, 3), raw_tts_recall=round(raw, 3), difference=round(rec - raw, 3),
                                char_sim=round(float(np.mean([e["asr"]["char_sim"] for e in a])), 3),
                                raw_char_sim=round(float(np.mean([e["asr_raw_tts"]["char_sim"] for e in a])), 3),
                                strict_recall=round(float(np.mean([e["asr"]["strict"] for e in a])), 3),
                                weak=[dict(id=e["id"], recall=e["asr"]["recall"], raw=e["asr_raw_tts"]["recall"], hyp=e["asr"]["hyp"])
                                      for e in a if e["asr"]["recall"] < 0.6])
        pun = [r for e in rap if e.get("punch") for r in e["punch"]["renders"]]
        if pun:
            s["punch_onsets"] = _stats([v for r in pun for v in (r["onset_err_ms"] + r["rise_err_ms"])], TOL_MS)
    if rep.get("rasp_ab"):
        ab = rep["rasp_ab"]
        s["rasp"] = dict(n_lines=ab["n"], clean=ab["clean"], raspy=ab["raspy"], asr_recall_clean=ab["asr_recall_clean"],
                         asr_recall_raspy=ab["asr_recall_raspy"])
    hook = rep["hook"]
    for layer in ("gang", "kids"):
        hs = [e for e in hook if e["layer"] == layer]
        if not hs:
            continue
        d = dict(onsets_periodicity=_stats([v for e in hs for v in e["stem_onset_err_ms"]], TOL_MS),
                 onsets_energy_rise=_stats([v for e in hs for v in e["stem_rise_err_ms"]], TOL_MS),
                 violations=[dict(id=e["id"], syllable=e["syllables"][j], err_ms=v, detector=dn)
                             for e in hs for dn, key in (("periodicity", "stem_onset_err_ms"), ("energy-rise", "stem_rise_err_ms"))
                             for j, v in enumerate(e[key]) if v is not None and abs(v) > TOL_MS])
        meds = [r["pitch"]["med_cents"] for e in hs for r in e["renders"] if r["pitch"]["med_cents"] is not None]
        if meds:
            d["pitch_median_abs_cents"] = dict(median=round(float(np.median(meds)), 1), worst=round(float(np.max(meds)), 1))
        a = [e for e in hs if e.get("asr")]
        if a:
            d["asr_recall"] = round(float(np.mean([e["asr"]["recall"] for e in a])), 3)
            d["asr_lines"] = [dict(id=e["id"], recall=e["asr"]["recall"], raw=(e.get("asr_raw_tts") or {}).get("recall"),
                                   hyp=e["asr"]["hyp"]) for e in a]
            if layer == "gang":
                d["asr_raw_tts_recall"] = round(float(np.mean([e["asr_raw_tts"]["recall"] for e in a])), 3)
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
                 time_compressed=[dict(id=e["id"], speed=e["speed"], log=e["fit_log"]) for e in it if e["fit_log"]])
        a = [e for e in it if e.get("asr")]
        if a:
            d["asr_recall"] = round(float(np.mean([e["asr"]["recall"] for e in a])), 3)
            raw = [e["asr_raw_tts"]["recall"] for e in a if e.get("asr_raw_tts")]
            d["asr_raw_tts_recall"] = round(float(np.mean(raw)), 3) if raw else None
            d["asr_char_sim"] = round(float(np.mean([e["asr"]["char_sim"] for e in a])), 3)
        rp = [e["f0_median_hz"] for e in it if e.get("f0_median_hz")]
        if rp:
            d["rapper_f0_median_hz"] = round(float(np.median(rp)), 1)
        s[kind] = d
    st = rep["stems"]
    s["stems"] = dict(max_peak_dbfs=max(v["peak_dbfs"] for v in st.values()), clipped_samples=sum(v["clipped_samples"] for v in st.values()),
                      outside_score_regions_peak_dbfs=max(v["outside_score_regions_peak_dbfs"] for v in st.values()),
                      silent=[k for k, v in st.items() if v["silent"]], edge_level_max_rel_db=rep.get("edge_level_max_rel_db"))
    s["render_seconds"] = rep["render_seconds"]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-asr", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated ids (r4,h12,a8_14,s1_0.0): partial debug render")
    a = ap.parse_args()
    Vocals(asr=not a.no_asr, only=a.only.split(",") if a.only else None).run()


if __name__ == "__main__":
    main()
