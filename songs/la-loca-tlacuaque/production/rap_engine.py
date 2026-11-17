"""Rap / chant vocal engine for "La Loca Tlacuaque de la Calle Cerrada".

A thin layer over the shared neural singing engine (songs/why-llarta/production/vox_engine.py, which is
used read-only): Kokoro-82M with its phoneme durations / F0 / energy driven from a plan, plus the WORLD
vocoder. This module adds everything a grid-locked Mexican rap needs.

API
    E = RapEngine(cache_dir=...)                    the engine (subclass of vox_engine.Engine)
    E.phonemes(text, "es")                          Mexican phonemes (θ->s, ʎ->ʝ, rising diphthongs as glides)
    E.natural(text, lang, voice)                    natural-speech analysis: token durations, per-token F0
    E.line_syllables(text, lang, voice)             syllables + stress + natural onset / duration / F0
    flow(score, lines, E, voice)                    slot of every syllable of every RAP line (rap_text.flow_line:
                                                    anchor rule, natural-rhythm spreading, 32nd pairs, no overlaps)
    E.rap_melody(info, fl, voice_cfg)               per-syllable speech-like pitch targets (fractional MIDI)
    E.render_rap(score, fl, cfg, seed)              one rap line on the grid, rapper voice (+ rasp)   -> dict
    E.render_punch(score, fl, voices, cfg, seed)    gang doubles of the rhyme (stressed syllable -> line end)
    E.render_hook(score, line, voices, ...)         sung hook line: gang layer or kids layer (octave / formants)
    E.render_adlib(...), E.render_spoken(...)       speech path: fitted to a deadline, shout / spoken / whisper,
                                                    gang voices or the rapper voice (E.rapperize)
    RAPPER                                          the rapper voice recipe (Kokoro blend, pitch, formants, rasp)
    rasp_metrics(x, sr)                             HNR, periodicity, sub-harmonic ratio, 1-4 kHz noise ratio
    rise_onset(x16, T)                              secondary vowel-onset detector (energy rise) for syllables
                                                    whose onset the periodicity detector cannot see

Every render returns audio at 24 kHz with sample 0 at `R0` (song time, s), a plan, and measurements.
All randomness is seeded; results are cached on disk (neural passes by the base engine, finished renders
by RapEngine.memo) so re-runs are fast and bit-identical.
"""
import hashlib
import json
import os
import zlib

import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt

import shared  # noqa: F401  (adds the shared engine folder to sys.path)
import vox_engine as ve
import vox_onset as vo
import rap_text as rt
from rap_text import slot_time

KSR, FR, HOP, F0R, LAG = ve.KSR, ve.FR, ve.HOP, ve.F0R, ve.LAG
VERSION = "rap-2"            # bump to invalidate the finished-render cache

# ----------------------------------------------------------------------------- the rapper
RASP = dict(
    jitter_st=0.30,          # cycle-to-cycle F0 jitter: random pitch offset per 5 ms frame, semitones (1 sigma)
    wobble_st=0.25,          # slower (8-25 Hz) irregular pitch wobble, semitones
    sub=0.45,                # level of the period-doubled (F0/2) component relative to the voice
    sub_floor=0.35,          # the growl comes and goes: sub level moves between sub_floor*sub and sub
    ap=0.50,                 # aperiodicity floor in the 1-4 kHz band of voiced frames (0 clean .. 1 noise)
    shimmer=0.16,            # amplitude roughness: 20-70 Hz random AM depth (1 sigma)
    drive=1.6,               # tanh saturation drive
    sat_mix=0.5,             # wet share of the saturator
)
RAPPER = dict(
    voice="em_alex",         # Kokoro base voice (score.VOICES['lead'] overrides)
    blend=("am_onyx", 0.30),  # chest: deep English voice blended into the style vector
    recite=43.0,             # reciting tone (MIDI, 43 = G2 = 98 Hz): the line's median pitch
    neural_shift=3,          # the decoder speaks this many semitones above the target; WORLD brings it down
    formant_shift=0.94,      # < 1 = longer vocal tract (bigger man)
    compress=0.55,           # natural-speech pitch excursions are scaled by this ...
    max_dev=3.2,             # ... and soft-limited to +- this many semitones around the reciting tone
    declination=0.5,         # share of the natural downward trend that is kept
    stress_up=0.7,           # primary-stressed syllables a bit higher (semitones)
    rhyme_up=0.8,            # the rhyme stress a bit more
    final_fall=2.2,          # fall over the syllables after the rhyme stress / inside a final stressed syllable
    level=0.6,               # vocal-rider amount (0 = natural declination of loudness, 1 = flat)
    rasp=RASP,
)
SPEECH_RANGE = 0.8           # spoken lines keep this share of their natural pitch range
PUNCH_UP = 5.0               # gang doubles sit this many semitones above the rapper (shout register)
REST_MIN = 0.125             # s: a gap at least this long after a syllable is a rest (gated)


def seed_of(s):
    return zlib.crc32(str(s).encode()) & 0xFFFF


def _smoothstep(w):
    w = np.clip(w, 0.0, 1.0)
    return w * w * (3 - 2 * w)


# ----------------------------------------------------------------------------- rasp (WORLD domain)
def fx_synth(f0, sp, ap, fx, seed=0, fs=KSR):
    """WORLD synthesis with the optional rasp recipe fx['rasp'] (see RASP):
    jitter + wobble on F0, an aperiodicity floor in 1-4 kHz, a period-doubled copy (same envelope at F0/2,
    high-passed) whose level wanders, 20-70 Hz amplitude roughness, light asymmetric saturation."""
    import pyworld as pw
    f0 = np.ascontiguousarray(f0, dtype=np.float64)
    sp = np.ascontiguousarray(sp)
    ap = np.ascontiguousarray(ap)
    r = (fx or {}).get("rasp")
    if not r:
        return pw.synthesize(f0, sp, ap, fs, 5.0)
    rng = np.random.default_rng(1000003 * (int(seed) + 1) + 17)
    nfr = len(f0)
    v = f0 > 0
    fps = 200.0
    j = rng.standard_normal(nfr) * r.get("jitter_st", 0.0)
    wb = rng.standard_normal(nfr)
    wb = sosfiltfilt(butter(2, [8.0, 25.0], "bandpass", fs=fps, output="sos"), wb)
    wb *= r.get("wobble_st", 0.0) / (np.std(wb) + 1e-9)
    f0j = np.where(v, f0 * 2.0 ** ((j + wb) / 12.0), 0.0)
    nb = ap.shape[1]
    fq = np.arange(nb) / (nb - 1) * fs / 2
    band = np.clip((fq - 700.0) / 500.0, 0, 1) * np.clip((5000.0 - fq) / 1000.0, 0, 1)
    ap2 = ap.copy()
    ap2[v] = np.maximum(ap2[v], r.get("ap", 0.0) * band)
    y = pw.synthesize(f0j, sp, ap2, fs, 5.0)
    n = len(y)
    tt = np.arange(n) / fs
    tf = np.arange(nfr) / fps
    slow = sosfiltfilt(butter(2, 3.0, "lowpass", fs=fps, output="sos"), rng.standard_normal(nfr))
    slow = np.clip(0.5 + 0.5 * slow / (np.std(slow) + 1e-9) / 1.5, 0, 1)
    if r.get("sub", 0.0) > 0:
        ys = pw.synthesize(f0j * 0.5, sp, ap2, fs, 5.0)
        ys = sosfilt(butter(2, 110.0, "highpass", fs=fs, output="sos"), ys)[:n]
        g = r["sub"] * (r.get("sub_floor", 0.3) + (1 - r.get("sub_floor", 0.3)) * slow)
        gs = np.interp(tt, tf, g)
        y = (y + gs * ys) / np.sqrt(1.0 + gs ** 2)
    if r.get("shimmer", 0.0) > 0:
        am = sosfiltfilt(butter(2, [20.0, 70.0], "bandpass", fs=fps, output="sos"), rng.standard_normal(nfr))
        am = np.clip(am / (np.std(am) + 1e-9), -2.5, 2.5) * r["shimmer"]
        y = y * (1.0 + np.interp(tt, tf, am * v))
    if r.get("sat_mix", 0.0) > 0:
        ref = np.percentile(np.abs(y), 99.5) + 1e-9
        d = r.get("drive", 1.5)
        u = y / ref
        s = (np.tanh(d * (u + 0.12)) - np.tanh(d * 0.12)) / np.tanh(d)
        m = r["sat_mix"]
        y = ref * ((1 - m) * u + m * s)
    return y


# ----------------------------------------------------------------------------- measurements
def rasp_metrics(x, sr=KSR, f0_lo=60.0, f0_hi=260.0):
    """Roughness of a voice recording, over its voiced frames:
      hnr_db        harmonics-to-noise ratio from the normalised autocorrelation peak (40 ms frames)
      periodicity   median of that peak (the onset detector needs > 0.6)
      shr_db        sub-harmonic-to-harmonic ratio: spectral energy at (k - 1/2) F0 against k F0, k = 1..10
      nhr_1_4k_db   noise(between harmonics)-to-harmonic energy in 1-4 kHz
      f0_med        median F0 (harvest), Hz"""
    import pyworld as pw
    x = np.asarray(x, np.float64)
    f0, t = pw.harvest(x, sr, f0_floor=f0_lo / 1.4, f0_ceil=f0_hi * 1.6, frame_period=10.0)
    W = int(0.064 * sr)
    win = np.hanning(W)
    nfft = 8192
    fq = np.fft.rfftfreq(nfft, 1 / sr)
    hnr, per, shr, nhr = [], [], [], []
    env = np.sqrt(np.convolve(x ** 2, np.ones(W) / W, mode="same"))
    thr = np.percentile(env, 95) * 10 ** (-20 / 20)
    for i, f in enumerate(f0):
        c = int(t[i] * sr)
        if f <= 0 or c - W // 2 < 0 or c + W // 2 > len(x) or env[c] < thr:
            continue
        seg = x[c - W // 2:c - W // 2 + W]
        s = seg * win
        ac = np.correlate(s, s, "full")[W - 1:]
        wac = np.correlate(win, win, "full")[W - 1:]
        rn = ac / (wac + 1e-12)
        rn /= rn[0] + 1e-12
        lmin, lmax = int(sr / (f * 1.25)), int(sr / (f * 0.8))
        rmax = float(np.clip(rn[lmin:lmax + 1].max(), 1e-4, 0.9999))
        per.append(rmax)
        hnr.append(10 * np.log10(rmax / (1 - rmax)))
        S = np.abs(np.fft.rfft(s, nfft)) ** 2
        bw = 0.2 * f
        eh = es = 0.0
        for k in range(1, 11):
            eh += S[(fq > k * f - bw) & (fq < k * f + bw)].sum()
            es += S[(fq > (k - 0.5) * f - bw) & (fq < (k - 0.5) * f + bw)].sum()
        shr.append(10 * np.log10((es + 1e-20) / (eh + 1e-20)))
        sel = (fq >= 1000) & (fq <= 4000)
        d = np.abs(((fq / f) + 0.5) % 1.0 - 0.5)          # distance to the nearest harmonic, in F0
        nh = S[sel & (d > 0.3)].sum() / 0.4
        hh = S[sel & (d <= 0.2)].sum() / 0.4
        nhr.append(10 * np.log10((nh + 1e-20) / (hh + 1e-20)))
    if not hnr:
        return dict(hnr_db=None, periodicity=None, shr_db=None, nhr_1_4k_db=None, f0_med=None, frames=0)
    return dict(hnr_db=round(float(np.mean(hnr)), 2), periodicity=round(float(np.median(per)), 3),
                shr_db=round(float(np.mean(shr)), 2), nhr_1_4k_db=round(float(np.mean(nhr)), 2),
                f0_med=round(float(np.median(f0[f0 > 0])), 1), frames=len(hnr))


def pitch_check(audio, f0_target5, nucl5):
    """Median |cents| between the rendered pitch (harvest) and the target inside the vowel nuclei."""
    import pyworld as pw
    tg = np.asarray(f0_target5, np.float64)
    top = float(tg.max()) if len(tg) else 0.0
    if top <= 0:
        return dict(med_cents=None, within50=0.0, voiced=0.0)
    f, _ = pw.harvest(np.asarray(audio, np.float64), KSR, f0_floor=max(40.0, 0.5 * float(tg[tg > 0].min())),
                      f0_ceil=max(600.0, 1.6 * top), frame_period=5.0)
    n = min(len(f), len(tg))
    sel = np.asarray(nucl5[:n], bool) & (tg[:n] > 0)
    det = sel & (f[:n] > 0)
    if det.sum() == 0:
        return dict(med_cents=None, within50=0.0, voiced=0.0)
    c = np.abs(1200 * np.log2(f[:n][det] / tg[:n][det]))
    return dict(med_cents=round(float(np.median(c)), 1), within50=round(float(np.mean(c < 50)), 3),
                voiced=round(float(det.sum() / max(1, sel.sum())), 3))


_SOS_RISE = butter(4, [300, 3000], "bandpass", fs=vo.FS, output="sos")


def rise_onset(x16, T, before=0.075, after=0.09, min_rise_db=4.0):
    """Secondary vowel-onset estimate for syllables that follow a voiced consonant (no voicing boundary):
    the instant the 300-3000 Hz envelope (8 ms RMS) has risen half-way (in dB, the rise counted up to 12 dB)
    from the dip of the consonant to the vowel's level, i.e. it comes within 2..6 dB of the vowel.
    None if there is no dip of at least min_rise_db near T."""
    fs = vo.FS
    i0, i1 = int((T - before - 0.03) * fs), int((T + after + 0.05) * fs)
    if i0 < 0 or i1 > len(x16):
        return None
    seg = sosfiltfilt(_SOS_RISE, x16[i0:i1])
    k = int(0.008 * fs)
    e = 10 * np.log10(np.convolve(seg ** 2, np.ones(k) / k, mode="same") + 1e-14)
    tt = (i0 + np.arange(len(e))) / fs
    pre = (tt >= T - before) & (tt <= T + 0.03)
    if not pre.any():
        return None
    jd = np.nonzero(pre)[0][int(np.argmin(e[pre]))]
    post = (tt > tt[jd]) & (tt <= tt[jd] + 0.11)
    if not post.any():
        return None
    top = np.percentile(e[post], 90)
    dip = e[jd]
    if top - dip < min_rise_db:
        return None
    half = top - 0.5 * min(top - dip, 12.0)
    idx = np.nonzero(post & (e >= half))[0]
    if len(idx) == 0:
        return None
    return float(tt[idx[0]])


# ----------------------------------------------------------------------------- engine
class RapEngine(ve.Engine):
    def __init__(self, *a, memo_dir=None, **kw):
        super().__init__(*a, **kw)
        self.memo_dir = memo_dir
        if memo_dir:
            os.makedirs(memo_dir, exist_ok=True)
        self.memo_hits = 0
        self.memo_miss = 0

    # --- finished-render cache
    def memo(self, tag, key, fn):
        """Disk cache for finished renders: key is any JSON-able description of the inputs."""
        if not self.memo_dir:
            return fn()
        h = hashlib.sha1((VERSION + "|" + tag + "|" + json.dumps(key, sort_keys=True, default=_jsonable)).encode())
        path = os.path.join(self.memo_dir, f"{tag}-{h.hexdigest()}.npz")
        if os.path.exists(path):
            z = np.load(path, allow_pickle=False)
            self.memo_hits += 1
            return z["audio"], json.loads(str(z["meta"]))
        audio, meta = fn()
        self.memo_miss += 1
        tmp = path + ".tmp.npz"
        np.savez(tmp, audio=np.asarray(audio, np.float32), meta=json.dumps(meta, default=_jsonable))
        os.replace(tmp, path)
        return np.asarray(audio, np.float32), json.loads(json.dumps(meta, default=_jsonable))

    # --- text
    def phonemes(self, text, lang):
        ph = super().phonemes(text, lang)
        return rt.mexican(ph) if lang == "es" else ph

    def style(self, voice, ntok, blend=None):
        """voice may be a recipe 'em_alex|am_onyx:0.3|em_santa:0.2' (base | other:amount ...)."""
        if "|" in voice:
            base, *others = voice.split("|")
            v = super().style(base, ntok)
            tot = 0.0
            acc = np.zeros_like(v)
            for o in others:
                name, amt = o.split(":")
                acc += float(amt) * super().style(name, ntok)
                tot += float(amt)
            v = ((1 - tot) * v + acc).astype(np.float32)
            if blend:
                v = ((1 - blend[1]) * v + blend[1] * super().style(blend[0], ntok)).astype(np.float32)
            return v
        return super().style(voice, ntok, blend)

    def natural(self, text, lang, voice, speed=1.0):
        """Natural-speech analysis of a phrase: per-token durations (frames) and F0 (Hz, NaN = unvoiced)."""
        ph = self.phonemes(text, lang)
        toks, chars = self.tokens(ph)
        st = self.style(voice, len(toks))
        _, d, f0, _ = self.analyse(toks, st, speed)
        nf = len(f0)
        tok_of = np.repeat(np.arange(len(d)), 2 * d)
        tok_of = tok_of[np.minimum(np.arange(nf) + 2 * LAG, nf - 1)]
        tf0 = np.full(len(chars), np.nan)
        for i in range(len(chars)):
            sel = (tok_of == i + 1) & (f0 > 45.0)
            if sel.any():
                tf0[i] = float(np.median(f0[sel]))
        start = np.concatenate([[0], np.cumsum(d)])[1:-1] - d[0]       # frame at which each char starts
        return dict(ph=ph, toks=toks, chars=chars, nat=d[1:-1].astype(int), start=start.astype(int), tok_f0=tf0,
                    f0=f0, tok_of=tok_of)

    def line_syllables(self, text, lang, voice):
        """Syllables of a phrase with stress, natural onset (s), natural duration (s) and natural F0 (Hz)."""
        a = self.natural(text, lang, voice)
        chars = a["chars"]
        syls, joins = rt.syllabify_es(chars)
        if not syls:
            raise ValueError(f"no vowel in {text!r} ({a['ph']!r})")
        tau = []
        for s in syls:
            i = s["nuc"][0]
            if i > 0 and chars[i - 1] in "ˈˌ":
                i -= 1
            tau.append(a["start"][i] * FR)
        end = len(chars)
        while end > 0 and ve.cls_of(chars[end - 1]) in "PW":
            end -= 1
        t_end = (a["start"][end - 1] + a["nat"][end - 1]) * FR
        nd = [(tau[k + 1] if k + 1 < len(syls) else t_end) - tau[k] for k in range(len(syls))]
        f0 = []
        for s in syls:
            idx = list(s["nuc"])
            if idx[0] > 0 and chars[idx[0] - 1] in "ˈˌ":
                idx = [idx[0] - 1] + idx
            vals = [a["tok_f0"][x] for x in idx if not np.isnan(a["tok_f0"][x])]
            f0.append(float(np.median(vals)) if vals else np.nan)
        return dict(text=text, lang=lang, ph=a["ph"], chars=chars, syls=syls, joins=joins, tau=tau, natdur=nd,
                    f0=f0, nat=a["nat"])

    # --- syllable structure for the duration plan
    @staticmethod
    def build_sylls(chars, nuclei):
        """vox_engine syllable dicts (onset / nucleus / coda / gap token indices) from given nuclei."""
        sylls = [dict(nucleus=list(nu), onset=[], coda=[], gap=[], notes=[]) for nu in nuclei]
        pre = list(range(0, nuclei[0][0]))
        sylls[0]["onset"] = [x for x in pre if ve.cls_of(chars[x]) == "C"]
        sylls[0]["lead"] = [x for x in pre if ve.cls_of(chars[x]) != "C"]
        for k in range(len(nuclei) - 1):
            seg = list(range(nuclei[k][-1] + 1, nuclei[k + 1][0]))
            bnd = [x for x in seg if ve.cls_of(chars[x]) in "WP"]
            if bnd:
                sylls[k]["coda"] = [x for x in seg if x < bnd[0] and ve.cls_of(chars[x]) == "C"]
                sylls[k + 1]["onset"] = [x for x in seg if x > bnd[-1] and ve.cls_of(chars[x]) == "C"]
            else:
                cons = [x for x in seg if ve.cls_of(chars[x]) == "C"]
                n_on = ve._onset_count("".join(chars[x] for x in cons))
                sylls[k]["coda"] = cons[:len(cons) - n_on]
                sylls[k + 1]["onset"] = cons[len(cons) - n_on:]
            used = set(sylls[k]["coda"]) | set(sylls[k + 1]["onset"])
            sylls[k]["gap"] = [x for x in seg if x not in used]
        post = list(range(nuclei[-1][-1] + 1, len(chars)))
        sylls[-1]["coda"] = [x for x in post if ve.cls_of(chars[x]) == "C"]
        sylls[-1]["gap"] = [x for x in post if ve.cls_of(chars[x]) != "C"]
        return sylls

    @staticmethod
    def syl_strings(chars, sylls):
        out = []
        for s in sylls:
            out.append("".join(chars[x] for x in s["onset"]) + "".join(chars[x] for x in s["nucleus"] if chars[x] not in " ˈˌ")
                       + "".join(chars[x] for x in s["coda"]))
        return out

    def plan_notes(self, text, lang, groups, ana_voice, nuclei, first=0, ana_speed=1.0, slog=None):
        """Duration plan (the same structure vox_engine.plan_sung returns) for explicit syllables.
        nuclei: token-index lists of ALL syllable nuclei of the text (rap_text.syllabify_es / apply_joins).
        groups: for every SOUNDING syllable (those from index `first` on) its notes [(T, D, midi[, bend])].
        first > 0: the syllables before it get zero frames: they are never heard but the text encoder still
        sees the whole line (used for the punch-ins, which only sound from the rhyme stress on)."""
        ph = self.phonemes(text, lang)
        toks, chars = self.tokens(ph)
        st = self.style(ana_voice, len(toks))
        _, nat, _, _ = self.analyse(toks, st, ana_speed)
        nat = nat[1:-1]
        sylls = self.build_sylls(chars, nuclei)[first:]
        assert len(sylls) == len(groups), (len(sylls), len(groups))
        nd = []
        for s, g in zip(sylls, groups):
            s["notes"] = [dict(T=float(n[0]), D=float(n[1]), m=float(n[2]), b=float(n[3]) if len(n) > 3 else 0.0)
                          for n in g]
            nd.extend(s["notes"])
        nS = len(sylls)
        Ts = [s["notes"][0]["T"] for s in sylls]
        on_f, co_f = [], []
        for k, s in enumerate(sylls):
            on = [int(np.clip(nat[x], ve.ONSET_MIN if chars[x] not in ve.SONORANT else 1, 3)) for x in s["onset"]]
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
        lead_s = (ve.PAD_LEAD + sum(on_f[0])) * FR
        best = None
        for ph_ms in np.arange(0.0, 25.0, 0.5):
            R0 = Ts[0] - lead_s - ph_ms / 1000.0
            vf = [int(round((T - R0) / FR)) for T in Ts]
            err = [R0 + v * FR - T for v, T in zip(vf, Ts)]
            sc = max(abs(e) for e in err)
            if best is None or sc < best[0] - 1e-9:
                best = (sc, R0, vf, err)
        _, R0, VF, verr = best
        # two syllables may never share a frame (thirty-second notes on the 25 ms grid)
        for k in range(1, nS):
            while VF[k] < VF[k - 1] + 1 + sum(on_f[k]) and sum(on_f[k]) > len(on_f[k]):
                on_f[k][int(np.argmax(on_f[k]))] -= 1
            if VF[k] < VF[k - 1] + 1 + sum(on_f[k]):
                VF[k] = VF[k - 1] + 1 + sum(on_f[k])
        ntok = len(chars)
        dur = np.zeros(ntok + 2, np.int64)
        dur[0] = VF[0] - sum(on_f[0])
        assert dur[0] >= 3, dur[0]
        rest_mask, nuc_frames, nuc_alloc, gaps, coda_used = [], [], [], [], []
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
            while nuc < 1 and sum(co) > 0:          # very short syllable: the coda gives way
                co[int(np.argmax(co))] -= 1
                nuc = end_k - cur - sum(co)
            if nuc < 1:
                nuc = 1
            vs = [x for x in s["nucleus"] if chars[x] in ve.VOWELS]
            ls = [x for x in s["nucleus"] if chars[x] == ve.LENGTH]
            alloc = {}
            rem = nuc
            if len(vs) >= 2:
                contiguous = all(chars[x] in ve.VOWELS or chars[x] == ve.LENGTH for x in range(vs[0], vs[-1] + 1))
                if contiguous and chars[vs[-1]] in "ɪʊiu" and len(vs) == 2:
                    off = int(np.clip(round(0.22 * nuc), 1, 5)) if nuc >= 3 else 0     # falling diphthong
                    alloc[vs[-1]] = off
                    alloc[vs[0]] = rem - off
                else:
                    # synalepha / syneresis: share the nucleus like natural speech does (stress mark frames
                    # belong to the vowel they precede)
                    w = []
                    for x in vs:
                        ww = float(max(1, nat[x]))
                        if x > 0 and chars[x - 1] in "ˈˌ":
                            ww += float(nat[x - 1])
                        w.append(ww)
                    w = np.array(w) / np.sum(w)
                    fr = np.maximum(1 if rem >= len(vs) else 0, np.floor(w * rem)).astype(int)
                    while fr.sum() > rem and fr.max() > 0:
                        fr[int(np.argmax(fr))] -= 1
                    while fr.sum() < rem:
                        fr[int(np.argmax(w * rem - fr))] += 1
                    for x, f in zip(vs, fr):
                        alloc[x] = int(f)
                for x in ls:
                    alloc[x] = 0
            elif ls and rem >= 2:
                w0 = max(1, nat[vs[0]])
                wl = max(1, sum(nat[x] for x in ls))
                lf = int(np.clip(int(round(rem * wl / (w0 + wl))), 1, rem - 1))
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
                tgt = [x for x in s["gap"] if ve.cls_of(chars[x]) in "WP"]
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
                raise RuntimeError(f"negative gap at syllable {k}: {gap} in {text!r}")
        tail_pad = ve.PAD_TAIL
        rest_mask.append((cur, cur + tail_pad + 2 + LAG))
        dur[-1] = tail_pad + 2
        assert int(dur.sum()) == cur + tail_pad + 2
        lead_frames = int(dur[0])
        dur[0] += LAG
        return dict(text=text, lang=lang, ph=ph, toks=toks, chars=chars, sylls=sylls, slog=list(slog or []), R0=R0,
                    exact_v=[(T - R0) / FR for T in Ts], rise_loop=True, tail=None, VF=VF, vowel_err=verr, dur=dur, rest_mask=rest_mask, nuc_frames=nuc_frames,
                    notes=nd, onset_frames=on_f, coda_frames=co_f, ntok0=len(toks), lead_frames=lead_frames,
                    nuc_alloc=nuc_alloc, gaps=gaps, chars0=chars, coda_used=coda_used, first=first)

    # --- hooks into the shared sung path
    def pitch_curve(self, plan, t, detune_cents=0.0, octave=0, seed=0, vib_depth=0.22, scoop=0.4, expressive=True):
        if not plan.get("rap"):
            return super().pitch_curve(plan, t, detune_cents, octave, seed, vib_depth, scoop, expressive)
        # speech-like: every syllable has a target (fractional MIDI) and a bend over its length; fast glides
        # (about 65 ms, centred just before the vowel onset) join them; no scoops, no vibrato
        notes = [s["notes"][0] for s in plan["sylls"]]
        T = np.array([n["T"] for n in notes])
        D = np.array([n["D"] for n in notes])
        M = np.array([n["m"] for n in notes], dtype=np.float64)
        B = np.array([n.get("b", 0.0) for n in notes], dtype=np.float64)
        idx = np.clip(np.searchsorted(T, t, side="right") - 1, 0, len(T) - 1)
        pitch = M[idx] + B[idx] * np.clip((t - T[idx]) / np.maximum(D[idx], 1e-3), 0, 1)
        for j in range(1, len(T)):
            a, b = T[j] - 0.045, T[j] + 0.02
            sel = (t >= a) & (t < b)
            if sel.any():
                prev = M[j - 1] + B[j - 1] * np.clip((t[sel] - T[j - 1]) / max(D[j - 1], 1e-3), 0, 1)
                curv = M[j] + B[j] * np.clip((t[sel] - T[j]) / max(D[j], 1e-3), 0, 1)
                pitch[sel] = prev + (curv - prev) * _smoothstep((t[sel] - a) / (b - a))
        if expressive:
            rng = np.random.default_rng(seed)
            p1, p2 = rng.random(2) * 2 * np.pi
            pitch = pitch + 0.10 * np.sin(2 * np.pi * 0.9 * t + p1) + 0.06 * np.sin(2 * np.pi * 2.3 * t + p2)
        return pitch + detune_cents / 100.0 + 12 * octave

    def world_synth(self, plan, wa, detune_cents=0.0, octave=0, seed=0, vib_depth=0.22, ap_cap=0.35,
                    time_offset=0.0, formant_shift=1.0):
        """As vox_engine.Engine.world_synth, plus the per-plan voice effects in plan['fx']:
        formant_shift (spectral envelope warp) and rasp (see fx_synth)."""
        fx = plan.get("fx") or {}
        formant_shift = fx.get("formant_shift", formant_shift)
        t = wa["t"]
        pitch = self.pitch_curve(plan, plan["R0"] + t, detune_cents, octave, seed, vib_depth=vib_depth)
        f0 = np.where(wa["v"], ve.midi_hz(pitch), 0.0)
        ap = wa["ap"].copy()
        if ap_cap is not None:
            nb = ap.shape[1]
            fq = np.arange(nb) / (nb - 1) * KSR / 2
            lo, hi = ve.AP_CAP_HI_HZ
            cap = ap_cap + (1 - ap_cap) * np.clip((fq - lo) / (hi - lo), 0, 1)
            floor = ve.AP_FLOOR_MAX * np.clip((fq - ve.AP_FLOOR_HZ[0]) / (ve.AP_FLOOR_HZ[1] - ve.AP_FLOOR_HZ[0]), 0, 1)
            ap[wa["nucl"]] = np.minimum(ap[wa["nucl"]], cap)
            ap[wa["v"]] = np.maximum(ap[wa["v"]], floor)
        sp = wa["sp"]
        if ve.CONS_LIFTER_MS > 0:
            sel = ~wa["nucl"]
            if sel.any():
                nb = sp.shape[1]
                c = np.fft.irfft(np.log(sp[sel] + 1e-16), axis=1)
                q = int(ve.CONS_LIFTER_MS / 1000.0 * KSR)
                lift = np.zeros(c.shape[1])
                lift[:q] = 1.0
                lift[-q + 1:] = 1.0
                sp = sp.copy()
                sp[sel] = np.exp(np.fft.rfft(c * lift, axis=1).real[:, :nb])
        if formant_shift != 1.0:
            sp = warp_formants(sp, formant_shift)
        y = fx_synth(f0, sp, ap, fx, seed)
        return y, f0

    def bounds(self, plan, nuc_len=None):
        """As the base engine, but on the TARGET timeline the vowel onsets are the exact note times
        (plan['exact_v'], fractional frames) instead of their 25 ms frame-grid approximation: the swung
        sixteenth grid does not fit a 25 ms raster, and the time map / verification use these values."""
        out = super().bounds(plan, nuc_len)
        ex = plan.get("exact_v")
        if ex is not None and nuc_len is None and "bnd" not in plan:
            for q, v in zip(out, ex):
                q["v"] = float(v)
        return out

    def verify_onsets(self, plan, audio, extra_delay=0.0):
        """Measured minus target vowel onset (ms) per syllable. Where the periodicity detector has no
        boundary to see (the vowel follows a voiced consonant) and plan['rise_loop'] is set, the energy-rise
        detector is used instead, so the engine's second timing loop also corrects those syllables."""
        out = super().verify_onsets(plan, audio, extra_delay)
        if plan.get("rise_loop"):
            rise = self.verify_rise(plan, audio, extra_delay)
            for k, e in enumerate(out):
                if e is None and rise[k] is not None and abs(rise[k]) <= 45.0:
                    out[k] = rise[k]
        return out

    def has_consonant_before(self, plan, k):
        s_ = plan["sylls"][k]
        if s_["onset"]:
            return True
        return k > 0 and bool(plan["sylls"][k - 1]["coda"])

    def verify_rise(self, plan, audio, extra_delay=0.0):
        """Energy-rise onset error (ms) of every syllable whose vowel follows a VOICED consonant."""
        x16 = vo.to16(audio, KSR)
        out = []
        for k, q in enumerate(self.bounds(plan)):
            if self.measurable(plan, k) or not self.has_consonant_before(plan, k):
                out.append(None)
                continue
            T = q["v"] * FR + extra_delay
            m = rise_onset(x16, T)
            out.append(None if m is None else round(1000 * (m - T), 1))
        return out

    def level_syllables(self, plan, y, amount=0.8, max_cut=9.0, max_boost=12.0):
        return super().level_syllables(plan, y, plan.get("level", amount), max_cut, max_boost)

    # ------------------------------------------------------------------------- rap
    def rap_melody(self, f0_syl, stress, rhyme, cfg=RAPPER):
        """Speech-like pitch targets for a rap line: the natural F0 of every syllable, relative to the line's
        median, with part of the declination removed, compressed, stressed syllables raised, the tail falling,
        soft-limited to +-max_dev semitones around the reciting tone. Returns (midi[], bend[])."""
        f = np.array(f0_syl, dtype=np.float64)
        n = len(f)
        ok = ~np.isnan(f)
        if ok.any():                       # half-devoiced syllables report junk far below the voice's range
            ok &= ~(np.nan_to_num(f, nan=1e9) < 0.62 * np.median(f[ok]))
        if ok.sum() == 0:
            z = np.zeros(n)
        else:
            z = np.full(n, np.nan)
            z[ok] = 12 * np.log2(f[ok] / np.median(f[ok]))
            z = np.interp(np.arange(n), np.nonzero(ok)[0], z[ok])
        k = np.arange(n, dtype=np.float64)
        if n >= 3:
            slope = np.polyfit(k, z, 1)[0]
            z = z - (1 - cfg["declination"]) * slope * (k - k.mean())
        z = cfg["compress"] * z
        st = np.array(stress)
        z = z + cfg["stress_up"] * (st == 2) + 0.4 * cfg["stress_up"] * (st == 1)
        z[rhyme] += cfg["rhyme_up"]
        z = z - np.median(z)
        post = n - 1 - rhyme
        for j in range(1, post + 1):
            z[rhyme + j] = min(z[rhyme + j], z[rhyme] - cfg["final_fall"] * j / post - 0.5)
        z = cfg["max_dev"] * np.tanh(z / cfg["max_dev"])
        bend = np.zeros(n)
        bend[-1] = -cfg["final_fall"] * (1.0 if post == 0 else 0.5)
        return cfg["recite"] + z, bend


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(type(o))


def warp_formants(sp, shift):
    """Scale the frequency axis of a WORLD spectral envelope (shift < 1 lowers every formant)."""
    nb = sp.shape[1]
    src = np.clip(np.arange(nb) / shift, 0, nb - 1)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, nb - 1)
    w = src - i0
    ls = np.log(sp + 1e-16)
    return np.exp((1 - w) * ls[:, i0] + w * ls[:, i1])


def edge_fade(x, sr=KSR, ms=3.0):
    """Raised-cosine fades at both ends of a render (no clicks when it is placed in a stem)."""
    x = np.array(x, dtype=np.float64)
    n = min(len(x) // 2, int(ms / 1000.0 * sr))
    if n > 0:
        r = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))
        x[:n] *= r
        x[-n:] *= r[::-1]
    return x


# ----------------------------------------------------------------------------- flow of the whole verse
def flow(score, lines, E, voice):
    """Slots, times and note lengths of every syllable of every RAP line.
    Lines are laid out in time order; a line may not start before the previous line's last syllable has had
    its slot, and a line's last syllable is cut where the next line's first syllable needs room.
    Returns a list (same order as `lines`) of dicts:
      id, bar, text, info (line_syllables), syls (after merging), pos[], T[] (s), D[] (s), rhyme, stress[],
      crowded, pairs, log, end_s (planned end of the last syllable), deadline_s (t16(bar, hi))"""
    order = sorted(range(len(lines)), key=lambda i: lines[i]["bar"] * 16 + lines[i]["anchor"])
    out = [None] * len(lines)
    prev_free = None
    for i in order:
        ln = lines[i]
        text = ln.get("tts_text") or ln["text"]
        info = E.line_syllables(text, ln["lang"], voice)
        start_min = None if prev_free is None else int(prev_free - ln["bar"] * 16)
        f = rt.flow_line(info["syls"], info["joins"], info["tau"], info["natdur"], ln["anchor"], ln["lo"], ln["hi"],
                         start_min)
        syls = f["syls"]
        n = len(syls)
        T = [slot_time(score, ln["bar"], p) for p in f["pos"]]
        if any(T[k + 1] <= T[k] for k in range(n - 1)):
            raise RuntimeError(f"{ln['id']}: slots not increasing: {f['pos']}")
        deadline = score.t16(ln["bar"], ln["hi"])
        D = []
        for k in range(n - 1):
            ioi = T[k + 1] - T[k]
            d = ioi
            if syls[k]["punct"] and ioi >= 0.27:
                d = max(0.13, ioi - REST_MIN - 0.02)            # the comma is a short rest
            elif ioi > 0.40:
                d = 0.36                                        # nobody holds a rap syllable for 3 slots
            D.append(d)
        r = f["rhyme"]
        want = (2.0 if r == n - 1 else 1.4) * score.BEAT / 4
        D.append(max(0.06, min(want, deadline - 0.045 - T[-1])))
        f0 = [float(np.nanmedian([info["f0"][j] for j in s["orig"]])) if not all(np.isnan(info["f0"][j]) for j in s["orig"])
              else np.nan for s in syls]
        strings = E.syl_strings(info["chars"], E.build_sylls(info["chars"], [s["nuc"] for s in syls]))
        out[i] = dict(id=ln["id"], bar=ln["bar"], text=ln["text"], tts_text=text, lang=ln["lang"], info=info, syls=syls,
                      pos=f["pos"], T=T, D=D, rhyme=r, stress=[s["stress"] for s in syls], f0_nat=f0,
                      strings=strings, crowded=f["crowded"], pairs=f["pairs"], merged=f["merged"], log=list(f["log"]),
                      anchor=ln["anchor"], lo=ln["lo"], hi=ln["hi"], deadline_s=deadline, start_min=f["start_min"],
                      late=bool(T[-1] + 0.06 > deadline - 0.045))
        last = f["pos"][-1]
        prev_free = ln["bar"] * 16 + int(np.floor(last + 1e-9)) + 1
    # a line's tail ends where the next line needs room for its first consonants
    for a, b in zip(order[:-1], order[1:]):
        fa, fb = out[a], out[b]
        first = E.build_sylls(fb["info"]["chars"], [s["nuc"] for s in fb["syls"]])[0]
        room = 0.03 + 0.03 * min(2, len(first["onset"]))
        limit = fb["T"][0] - room
        end = fa["T"][-1] + fa["D"][-1]
        if end > limit:
            fa["D"][-1] = max(0.05, limit - fa["T"][-1])
            fa["log"].append(f"tail shortened to {fa['D'][-1] * 1000:.0f} ms for the pickup of {fb['id']}")
            if fb["T"][0] <= fa["T"][-1]:
                fa["log"].append(f"OVERLAP with {fb['id']}")
                fa["overlap"] = True
    for f in out:
        f["end_s"] = f["T"][-1] + f["D"][-1]
    return out


def flow_table(f):
    """[(syllable, slot, stressed)] for the report."""
    return [dict(syl=s, slot=(int(p) if float(p).is_integer() else round(float(p), 2)), stressed=bool(st == 2),
                 rhyme=bool(k == f["rhyme"]))
            for k, (s, p, st) in enumerate(zip(f["strings"], f["pos"], f["stress"]))]


# ----------------------------------------------------------------------------- renders
def _voice_of(cfg):
    return cfg["voice"]


class _Renders:
    """Mixed into RapEngine below (kept apart only for readability)."""

    # --- rap
    def rap_plan(self, f, cfg, first=0, pitch_up=0.0, fx=None, level=None):
        midi, bend = self.rap_melody(f["f0_nat"], f["stress"], f["rhyme"], cfg)
        groups = [[(f["T"][k], f["D"][k], midi[k] + pitch_up, bend[k])] for k in range(first, len(f["T"]))]
        plan = self.plan_notes(f["tts_text"], f["lang"], groups, cfg["voice"], [s["nuc"] for s in f["syls"]], first=first)
        plan["rap"] = True
        plan["fx"] = fx if fx is not None else dict(formant_shift=cfg["formant_shift"], rasp=cfg.get("rasp"))
        plan["level"] = cfg["level"] if level is None else level
        plan["midi"] = [float(m) for m in midi]
        return plan

    def _sung_cached(self, tag, plan, voice, key, **kw):
        """render_sung with the finished audio cached on disk. Returns (audio float64, meta)."""
        def run():
            r = self.render_sung(plan, voice, **kw)
            ver = self.verify_onsets(plan, r["audio"])
            f0t = r["f0_target5"]
            vv = f0t[f0t > 0]
            meta = dict(verify_ms=ver, verify_per_ms=ve.Engine.verify_onsets(self, plan, r["audio"]),
                        verify_rise_ms=self.verify_rise(plan, r["audio"]), timing=r["timing"],
                        f0_target_med=float(np.median(vv)) if len(vv) else None,
                        pitch=pitch_check(r["audio"], f0t, r["nucl5"]))
            return r["audio"], meta
        k = dict(key=key, voice=voice, kw=kw, text=plan["text"], dur=plan["dur"].tolist(),
                 notes=[(round(n["T"], 6), round(n["D"], 6), round(n["m"], 4), round(n.get("b", 0.0), 4)) for n in plan["notes"]],
                 fx=plan.get("fx"), rap=bool(plan.get("rap")), level=plan.get("level"), R0=round(plan["R0"], 6),
                 rise=bool(plan.get("rise_loop")))
        a, meta = self.memo(tag, k, run)
        return edge_fade(a), meta

    def render_rap(self, f, cfg=RAPPER, seed=0):
        """One rap line in the rapper's voice. Returns dict(audio 24 kHz, R0, plan, meta)."""
        plan = self.rap_plan(f, cfg)
        a, meta = self._sung_cached("rap", plan, cfg["voice"], dict(cfg={k: v for k, v in cfg.items()}),
                                    vib_depth=0.0, seed=int(seed), blend=cfg.get("blend"),
                                    neural_shift=int(cfg["neural_shift"]))
        return dict(audio=a, R0=plan["R0"], plan=plan, meta=meta)

    def render_punch(self, f, voice, cfg=RAPPER, seed=0, detune=0.0, pitch_up=PUNCH_UP, formant_shift=1.0):
        """A gang voice doubling the rhyme: only the syllables from the rhyme stress on are sounded."""
        plan = self.rap_plan(f, cfg, first=f["rhyme"], pitch_up=pitch_up, fx=dict(formant_shift=formant_shift), level=0.5)
        a, meta = self._sung_cached("punch", plan, voice, dict(first=f["rhyme"]), vib_depth=0.0, seed=int(seed),
                                    detune_cents=float(detune), neural_shift=0)
        return dict(audio=a, R0=plan["R0"], plan=plan, meta=meta)

    # --- hook
    def hook_plan(self, score, p, ana_voice):
        """Duration plan of a SUNG line: Spanish syllables reconciled with the score's notes.
        Every reconciliation (merge / melisma) is logged in plan['slog']."""
        text = p.get("tts_text") or p["text"]
        info = self.line_syllables(text, p["lang"], ana_voice)
        notes = [(score.t(p["bar"], b), d * score.BEAT, m) for (b, d, m, _) in p["notes"]]
        N = len(notes)
        syls, joins, nd = info["syls"], info["joins"], info["natdur"]
        use, log, cur = set(), [], rt.apply_joins(info["syls"], set())
        kind_of = dict(joins)
        for kinds in (("auto",), ("opt",), ("hiatus",)):
            cand = sorted([k for k, kind in joins if kind in kinds], key=lambda k: min(nd[k], nd[k + 1]))
            for k in cand:
                if len(cur) <= N:
                    break
                if (k - 1) in use or (k + 1) in use:
                    continue
                use.add(k)
                cur = rt.apply_joins(syls, use)
                log.append(f"merged syllables {k}+{k + 1} ({kind_of[k]}) to fit {N} notes")
        nuclei = [s["nuc"] for s in cur]
        while len(nuclei) > N:
            k = len(nuclei) - 1
            nuclei[k - 1:k + 1] = [list(range(nuclei[k - 1][0], nuclei[k][-1] + 1))]
            log.append(f"LOSSY: syllable {k} folded into {k - 1} (more syllables than notes)")
        groups = [[n] for n in notes[:len(nuclei)]]
        if len(nuclei) < N:
            extra = N - len(nuclei)
            owner = sorted(list(range(len(nuclei))) + [len(nuclei) - 1] * extra)
            groups = [[] for _ in nuclei]
            for n, o in zip(notes, owner):
                groups[o].append(n)
            log.append(f"melisma: {extra} extra note(s) on the last syllable (fewer syllables than notes)")
        plan = self.plan_notes(text, p["lang"], groups, ana_voice, nuclei, slog=log)
        plan["natural_syllables"] = len(syls)
        return plan

    def render_hook(self, plan, voice, seed=0, detune=0.0, octave=0, formant_shift=1.0, vib_depth=0.0):
        p2 = {k: v for k, v in plan.items() if k != "src"}
        p2["fx"] = dict(formant_shift=formant_shift)
        a, meta = self._sung_cached("hook", p2, voice, dict(), vib_depth=float(vib_depth), seed=int(seed),
                                    detune_cents=float(detune), octave=int(octave))
        return dict(audio=a, R0=p2["R0"], plan=p2, meta=meta)

    # --- speech path (ad-libs, spoken)
    def speech_once(self, text, lang, voices, speed, mode, cfg=RAPPER, seed=0, f0_scale=1.0, f0_range=1.0):
        """Every voice speaks the phrase with the SAME token durations (tight unison).
        mode: 'plain' (Kokoro prosody, pitch scaled by f0_scale / range by f0_range), 'rapper' (the rapper's
        deep raspy voice), 'whisper'. Returns list of float64 arrays (24 kHz)."""
        base = voices[0].split("|")[0]
        sp = self.speech_plan(text, lang, base, speed=float(speed))
        outs = []
        for j, v in enumerate(voices):
            if mode == "rapper":
                x = self.render_speech_rapper(sp, cfg, seed + j, f0_scale=f0_scale, f0_range=f0_range)
            else:
                natural = abs(f0_scale - 1.0) < 1e-9 and abs(f0_range - 1.0) < 1e-9
                x = self.render_speech(sp, v, f0_mode="natural" if natural else "world", f0_scale=f0_scale,
                                       f0_flatten=1.0 - f0_range)
                if mode == "whisper":
                    x = whisperize(x)
            outs.append(np.asarray(x, np.float64))
        return outs, sp

    def render_speech_rapper(self, splan, cfg=RAPPER, seed=0, f0_scale=1.0, f0_range=1.0):
        """A spoken phrase in the rapper's voice: Kokoro speaks it `neural_shift` semitones above the target
        with its natural (slightly compressed) intonation; WORLD brings it down to the reciting register with
        the formant shift and the rasp. Timing is exactly Kokoro's."""
        toks, dur = splan["toks"], splan["dur"]
        st = self.style(cfg["voice"], len(toks), cfg.get("blend"))
        _, f0p, n = self.run_dur(toks, st, dur)
        v = f0p > 45.0
        if not v.any():
            return self.render_speech(splan, cfg["voice"])
        lf = np.log(f0p[v].astype(np.float64))
        tgt = np.log(float(ve.midi_hz(cfg["recite"])) * f0_scale) + SPEECH_RANGE * f0_range * (lf - np.median(lf))
        up = 2.0 ** (cfg["neural_shift"] / 12.0)
        f0d = np.zeros(len(f0p), np.float32)
        f0d[v] = np.exp(tgt) * up
        x = self.run_sing(toks, st, dur, f0d, n).astype(np.float64)
        wa = self.world_analyse(dict(audio=x, f0_fed=f0d, nucl=v))
        f0o = np.where(wa["v"], wa["f0"] / up, 0.0)
        ap = wa["ap"].copy()
        nb = ap.shape[1]
        fq = np.arange(nb) / (nb - 1) * KSR / 2
        floor = ve.AP_FLOOR_MAX * np.clip((fq - ve.AP_FLOOR_HZ[0]) / (ve.AP_FLOOR_HZ[1] - ve.AP_FLOOR_HZ[0]), 0, 1)
        ap[wa["v"]] = np.maximum(ap[wa["v"]], floor)
        sp = warp_formants(wa["sp"], cfg["formant_shift"])
        y = fx_synth(f0o, sp, ap, dict(rasp=cfg.get("rasp")), seed)
        n_t = len(x)
        y = np.pad(y, (0, max(0, n_t - len(y))))[:n_t]
        # unvoiced stretches keep the decoder's crisp consonants above 1.2 kHz
        tt = np.arange(n_t) / KSR
        vs = np.interp(tt, wa["t"], wa["v"].astype(float))
        k = int(0.010 * KSR)
        vs = np.convolve(vs, np.ones(k) / k, mode="same")
        hp = butter(4, ve.XOVER_HZ, "highpass", fs=KSR, output="sos")
        lp = butter(4, ve.XOVER_HZ, "lowpass", fs=KSR, output="sos")
        y = vs * y + (1 - vs) * (sosfiltfilt(hp, x) + sosfiltfilt(lp, y))
        lead = max(0, int(dur[0]) - LAG - 2) * HOP
        y[:lead] = 0
        tail = int(dur[-1]) * HOP
        y[len(y) - tail:] *= np.linspace(1, 0, tail) ** 2
        return y


class RapEngine(RapEngine, _Renders):       # noqa: F811
    pass


def whisperize(x):
    """Whisper: WORLD resynthesis with every frame unvoiced (noise through the voice's own envelope)."""
    import pyworld as pw
    x = np.asarray(x, np.float64)
    f0, t = pw.harvest(x, KSR, f0_floor=60, f0_ceil=500, frame_period=5.0)
    sp = pw.cheaptrick(x, f0, t, KSR)
    ap = np.ones_like(sp)
    y = pw.synthesize(np.zeros_like(f0), sp, ap, KSR, 5.0)[:len(x)]
    y = np.pad(y, (0, len(x) - len(y)))
    y = sosfilt(butter(2, 180, "highpass", fs=KSR, output="sos"), y)
    y = y + 0.5 * sosfilt(butter(1, 2500, "highpass", fs=KSR, output="sos"), y)
    env = np.abs(x)
    k = int(0.01 * KSR)
    env = np.convolve(env, np.ones(k) / k, mode="same")
    gate = np.clip(env / (env.max() * 10 ** (-45 / 20) + 1e-12), 0, 1)
    return y * gate


# ----------------------------------------------------------------------------- intelligibility scoring
_NUM_ES = {"0": "cero", "1": "uno", "2": "dos", "3": "tres", "4": "cuatro", "5": "cinco", "6": "seis", "7": "siete",
           "8": "ocho", "9": "nueve", "10": "diez"}


def es_phonetic(word):
    """Spelling-independent form of a (lower-case, accent-free) Spanish word as Mexican Spanish sounds it:
    b=v, s=z=c(e,i), ll=y, silent h, qu/c=k, j=g(e,i). 'taza' and 'tasa', 'hocico' and 'ocico' become equal."""
    w = _NUM_ES.get(word, word)
    w = w.replace("ch", "C").replace("qu", "k").replace("ge", "xe").replace("gi", "xi")
    w = w.replace("gue", "ge").replace("gui", "gi").replace("ce", "se").replace("ci", "si")
    w = w.replace("c", "k").replace("z", "s").replace("ll", "y").replace("v", "b").replace("h", "").replace("j", "x")
    w = w.replace("C", "ch")
    if w == "y":
        w = "i"
    return w


def _lev(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
            prev = cur
    return d[len(b)]


def asr_scores(ref, hyp, lang="es"):
    """recall: share of the reference words found in the hypothesis (bag of words) after the phonetic
    normalisation above; strict: the same on plain spelling (vox_asr.word_scores); char_sim: 1 - edit distance
    of the two phonetic strings without spaces / reference length (robust to word-boundary differences)."""
    import vox_asr
    strict, _ = vox_asr.word_scores(ref, hyp)
    r = vox_asr.norm_words(ref)
    h = vox_asr.norm_words(hyp)
    if lang == "es":
        r = [es_phonetic(w) for w in r]
        h = [es_phonetic(w) for w in h]
    if not r:
        return dict(recall=1.0, strict=1.0, char_sim=1.0)
    pool, hit = list(h), 0
    for w in r:
        if w in pool:
            pool.remove(w)
            hit += 1
    rs, hs = "".join(r), "".join(h)
    return dict(recall=round(hit / len(r), 3), strict=round(strict, 3),
                char_sim=round(max(0.0, 1.0 - _lev(rs, hs) / max(1, len(rs))), 3))
