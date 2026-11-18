"""Rap / chant vocal engine for "La Loca Tlacuaque de la Calle Cerrada".

A layer over the shared neural singing engine (songs/why-llarta/production/vox_engine.py, used read-only):
Kokoro-82M with its phoneme durations / F0 / energy driven from a plan, plus the WORLD vocoder. This module
adds what a grid-locked Mexican rap needs. Everything is driven by score.py data.

API
    E = RapEngine(cache_dir=..., memo_dir=...)      subclass of vox_engine.Engine (cache_dir: neural passes;
                                                    memo_dir: finished renders, position-independent)
  text
    E.phonemes(text, "es")                          Mexican phonemes (rap_text.mexican: θ->s, ʎ->ʝ, glides);
                                                    lang "es+full": no vowel reduction (sung lines)
    E.line_syllables(text, lang, voice)             syllables + stress + natural onset / duration / F0
    flow(score, lines, E, voice)                    slot of every syllable of every RAP line (rap_text.flow_line:
                                                    anchor rule, natural-rhythm spreading by DP, synalepha only
                                                    when needed, 32nd pairs when crowded, no overlaps)
    flow_table(f)                                   [(syllable, slot, stressed, rhyme)] for the report
  plans
    E.plan_notes(text, lang, groups, voice, nuclei) duration plan for explicit syllables on explicit times
                                                    (natural-proportional frames; exact, non-quantised targets;
                                                    a glide + stress mark before a vowel capped at 50 ms)
    E.rap_melody(f0_syl, stress, rhyme, cfg)        speech-like pitch targets (fractional MIDI) + bends
  renders (24 kHz mono, sample 0 at R0 seconds of song time; dict(audio, R0, plan, meta))
    E.render_sung(plan, voice, ...)                 the closed timing loop (see below); every render goes through it
    E.render_rap(f, cfg, seed)                      one rap line, rapper voice; also the clean twin (no rasp)
    E.render_punch(f, voice, cfg, seed, ...)        a gang voice on the rhyme: sung from two syllables earlier,
                                                    gated open at the rhyme syllable's consonant (first=0: the
                                                    whole line)
    E.hook_plan(score, line, voice)                 SUNG line: syllables reconciled with the notes (logged)
    E.render_hook(plan, voice, seed, detune, octave, formant_shift)   gang (octave 0) or kids (octave 1)
    E.speech_once(text, lang, voices, speed, mode, ..., max_frames)   speech path for ad-libs / spoken lines:
                                                    'plain' | 'rapper' | 'whisper', same durations for all voices
    E.render_speech_rapper(splan, cfg, seed)        spoken phrase in the rapper's voice
  voice
    RAPPER, RASP                                    the rapper recipe (pitch, formants, intonation, rasp)
    fx_synth(f0, sp, ap, fx, seed)                  WORLD synthesis with the rasp recipe (DC-free)
  measurement
    rasp_metrics(x, sr)                             HNR, periodicity, sub-harmonic ratio, 1-4 kHz noise ratio, F0
    band_env / mid_rise(env, T, T_prev, T_next)     energy mid-rise vowel-onset detector (the audit's definition)
    line_readings(x16, Ts, Ds, per_ok)              every vowel onset by three detectors: periodicity (shared
                                                    vox_onset), mid-rise in 100-3500 Hz and in 300-3500 Hz
    combine_readings(r) / readings_cost(rd)         the loop's error per syllable / what it minimises
    E.onset_readings(plan, audio)                   line_readings of a render
    asr_scores(ref, hyp)                            word recall after Mexican-Spanish phonetic normalisation
    pitch_check(audio, f0_target, nucl)             median |cents| in the nuclei
    pitch_check_notes(audio, plan, ...)             per-note cents, short and long notes apart

How a rap line is made
  1. Kokoro analyses the line as natural speech: phoneme durations and F0 per syllable.
  2. flow(): the last primary-stressed syllable goes on the anchor slot; the syllables before it are laid on
     the swung sixteenths by dynamic programming over (slot - scaled natural onset)^2 + gap costs (stressed and
     pre-comma syllables may take two slots) + a small penalty for stressed syllables on odd slots.
  3. plan_notes(): every vowel onset is a note at t16(slot); frames between two onsets are shared the way
     natural speech shares them (coda, word space, onset consonants and the stress mark keep their natural
     length scaled by the local tempo; the vowel takes the rest).
  4. The decoder speaks the plan at the speech-like pitch curve (targets + fast glides, no vibrato, floor at
     80 Hz) with Kokoro's own energy contour; WORLD analysis; a time map puts every vowel onset on its slot.
  5. The closed timing loop (render_sung): the finished clean render is read by three onset detectors per
     syllable, syllables that are off are moved in the time map (and the voicing with them), and it is
     synthesised again, up to LOOP_PASSES times; the best pass is kept. Before the reading, every voiced
     consonant that is as loud as its vowel is turned down a few dB (onset_contrast), so each syllable has
     an attack on its slot that a listener - and a detector - can find.
  6. One more synthesis with the same time map adds the rasp (WORLD domain: jitter, sub-harmonic,
     aperiodicity, shimmer, saturation) and the formant shift; the clean twin is kept as a timing reference.

All randomness is seeded; results are cached on disk, so re-runs are fast and bit-identical.
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
VERSION = "rap-12"            # bump to invalidate the finished-render cache

# ----------------------------------------------------------------------------- the rapper
RASP = dict(
    jitter_st=0.30,          # cycle-to-cycle F0 jitter: random pitch offset per 5 ms frame, semitones (1 sigma)
    wobble_st=0.25,          # slower (8-25 Hz) irregular pitch wobble, semitones
    sub=0.45,                # level of the period-doubled (F0/2) component relative to the voice
    sub_floor=0.35,          # the growl comes and goes: sub level moves between sub_floor*sub and sub
    ap=0.50,                 # aperiodicity floor in the 1-4 kHz band of voiced frames (0 clean .. 1 noise)
    shimmer=0.16,            # amplitude roughness: 20-70 Hz random AM depth (1 sigma)
    attack=0.2,              # share of the rasp present in the first 30 ms of a voiced stretch (full after 90 ms)
    drive=1.6,               # tanh saturation drive
    sat_mix=0.5,             # wet share of the saturator
)
RAPPER = dict(
    voice="em_alex",         # Kokoro base voice (score.VOICES['lead'] overrides)
    blend=None,              # optional (voice, amount) blended into the style vector (am_onyx costs intelligibility)
    recite=43.0,             # reciting tone (MIDI, 43 = G2 = 98 Hz): the line's median pitch
    neural_shift=3,          # the decoder speaks this many semitones above the target; WORLD brings it down
    formant_shift=0.94,      # < 1 = longer vocal tract (bigger man)
    compress=0.55,           # natural-speech pitch excursions are scaled by this ...
    max_dev=3.2,             # ... and soft-limited to +- this many semitones around the reciting tone
    declination=0.5,         # share of the natural downward trend that is kept
    stress_up=0.7,           # primary-stressed syllables a bit higher (semitones)
    rhyme_up=1.2,            # the rhyme stress a bit more
    final_fall=2.2,          # fall over the syllables after the rhyme stress / inside a final stressed syllable
    floor_below=3.5,         # the voice never drops more than this below the reciting tone (80 Hz: lower than
                             # that the fry turns into single pulses and no pitch/onset detector follows it)
    level=0.5,               # vocal-rider amount (0 = natural declination of loudness, 1 = flat)
    speech_energy=True,      # keep Kokoro's own energy contour (accents) instead of the sung path's sustain
    rasp=RASP,
)
SPEECH_HOLD_S = 0.010        # spoken phrases: time after the last phoneme's nominal end before the release starts
SPEECH_RELEASE_S = 0.035     # ... length of that release
SPEECH_FADE_IN_S = 0.006     # ... fade-in from the first sounding sample
SPEECH_RANGE = 0.8           # spoken lines keep this share of their natural pitch range
HOOK_FULL_VOWELS = os.environ.get("RAP_HOOK_FULL", "1") == "1"     # sung lines: no vowel reduction (see phonemes)
HOOK_NUC_SHARE = float(os.environ.get("RAP_HOOK_NUC_SHARE", 0.5))  # sung lines: the vowel keeps half of every interval
SHORT_NOTE_S = 0.2           # sung notes shorter than this get a 36 ms pitch glide that ends at the vowel onset
PUNCH_UP = 5.0               # gang doubles sit this many semitones above the rapper (shout register)
PUNCH_LEAD_IN = 2            # syllables a punch-in sings (inaudibly) before the rhyme stress
PUNCH_FADE_S = 0.012         # fade-in of the punch-in gate
PUNCH_PRE_MAX_S = 0.080      # the gate never opens earlier than this before the anchor
PUNCH_CONTRAST_DB = 9.0      # behind the gate, the rhyme syllable's consonant stays this far under its vowel ...
PUNCH_CONTRAST_MAX_DB = 9.0  # ... by turning it down at most this much
REST_MIN = 0.125             # s: a gap at least this long after a syllable is a rest (gated)
CONTRAST_DB = float(os.environ.get("RAP_CONTRAST_DB", 8.0))       # wanted rise from a voiced consonant into its vowel
CONTRAST_MAX_DB = float(os.environ.get("RAP_CONTRAST_MAX", 6.0))  # most a consonant is turned down for it (0 = off)
GATE_PRE_S = 0.030           # the phrase gate opens this long before the first onset consonant ...
VOWEL_PRE_S = 0.012          # ... and this long before a phrase-initial vowel
LINE_PRE_MAX_S = 0.090       # ... but never earlier than this before the first vowel (a long 's' is faded in)
END_AFTER_S = 0.005          # ... and is shut this long after the written end of the last note
END_FADE_S = 0.025
LOOP_PASSES = int(os.environ.get("RAP_LOOP_PASSES", 5))    # corrections of the closed timing loop (render_sung)
LOOP_GOOD_COST = 30.0        # readings_cost at which the loop stops (about: nothing more than 13 ms off)
LOOP_TOL_MS = 4.0            # syllables whose combined reading is inside this are left alone
LOOP_MAX_STEP_MS = 70.0      # largest correction of one pass
GLIDE_MAX_FRAMES = int(os.environ.get("RAP_GLIDE_MAX", 2))   # glide + stress mark before a nucleus (25 ms frames)
NUC_SHARE = float(os.environ.get("RAP_NUC_SHARE", 0.0))       # optional minimum vowel share of an inter-onset interval (0.45 measured worse: ASR -0.02)
NAT_ALLOC = os.environ.get("RAP_NAT_ALLOC", "1") == "1"      # natural-proportional frame allocation (see plan_notes)
SPACE_FRAMES = os.environ.get("RAP_SPACE_FRAMES", "1") == "1"  # word spaces keep (scaled) natural time


def seed_of(s):
    return zlib.crc32(str(s).encode()) & 0xFFFF


def _smoothstep(w):
    w = np.clip(w, 0.0, 1.0)
    return w * w * (3 - 2 * w)


# ----------------------------------------------------------------------------- rasp (WORLD domain)
_SOS_DC = butter(2, 65.0, "highpass", fs=KSR, output="sos")


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
    # the gravel builds up over the first ~90 ms of every voiced stretch: a clean attack, then the growl
    run, since = 0, np.zeros(nfr)
    for i in range(nfr):
        run = run + 1 if v[i] else 0
        since[i] = run
    att = r.get("attack", 1.0)
    w = np.clip(att + (1 - att) * (since * 5.0 - 30.0) / 60.0, att, 1.0)
    j = rng.standard_normal(nfr) * r.get("jitter_st", 0.0)
    wb = rng.standard_normal(nfr)
    wb = sosfiltfilt(butter(2, [8.0, 25.0], "bandpass", fs=fps, output="sos"), wb)
    wb *= r.get("wobble_st", 0.0) / (np.std(wb) + 1e-9)
    f0j = np.where(v, f0 * 2.0 ** (w * (j + wb) / 12.0), 0.0)
    nb = ap.shape[1]
    fq = np.arange(nb) / (nb - 1) * fs / 2
    band = np.clip((fq - 700.0) / 500.0, 0, 1) * np.clip((5000.0 - fq) / 1000.0, 0, 1)
    ap2 = ap.copy()
    ap2[v] = np.maximum(ap2[v], r.get("ap", 0.0) * w[v][:, None] * band[None, :])
    y = pw.synthesize(f0j, sp, ap2, fs, 5.0)
    n = len(y)
    tt = np.arange(n) / fs
    tf = np.arange(nfr) / fps
    slow = sosfiltfilt(butter(2, 3.0, "lowpass", fs=fps, output="sos"), rng.standard_normal(nfr))
    slow = np.clip(0.5 + 0.5 * slow / (np.std(slow) + 1e-9) / 1.5, 0, 1)
    if r.get("sub", 0.0) > 0:
        ys = pw.synthesize(f0j * 0.5, sp, ap2, fs, 5.0)
        ys = sosfilt(butter(2, 110.0, "highpass", fs=fs, output="sos"), ys)[:n]
        g = r["sub"] * (r.get("sub_floor", 0.3) + (1 - r.get("sub_floor", 0.3)) * slow) * w
        gs = np.interp(tt, tf, g)
        y = (y + gs * ys) / np.sqrt(1.0 + gs ** 2)
    if r.get("shimmer", 0.0) > 0:
        am = sosfiltfilt(butter(2, [20.0, 70.0], "bandpass", fs=fps, output="sos"), rng.standard_normal(nfr))
        am = np.clip(am / (np.std(am) + 1e-9), -2.5, 2.5) * r["shimmer"]
        y = y * (1.0 + np.interp(tt, tf, am * v * w))
    if r.get("sat_mix", 0.0) > 0:
        ref = np.percentile(np.abs(y), 99.5) + 1e-9
        d = r.get("drive", 1.5)
        u = y / ref
        s = (np.tanh(d * (u + 0.12)) - np.tanh(d * 0.12)) / np.tanh(d)
        m = r["sat_mix"]
        y = ref * ((1 - m) * u + m * s)
        # the asymmetric curve leaves DC and infrasonic rumble (the voice's floor is 80 Hz): zero-phase high-pass
        y = sosfiltfilt(_SOS_DC, y)
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


def pitch_check_notes(audio, plan, octave=0, detune_cents=0.0, short_s=0.2):
    """Per-note pitch error of a sung render (cents against the written note, + octave and detune), split
    into short (< short_s) and longer notes. Measured twice:
      'whole'  median F0 from max(40 ms, 30 %) after the note's start to 30 ms before its end: the window an
               outside listener would take. On a sixteenth note followed by p / t / k most of that window is
               the next consonant's closure, where a pitch tracker reports nothing or an octave error;
      'vowel'  the same inside the syllable's planned vowel only (from 30 % of the vowel to 15 ms before its
               end): the pitch the voice actually sings.
    None = fewer than 3 voiced frames in the window. Also the share of each whole-note window that the plan
    gives to the vowel ('vowel_share')."""
    import pyworld as pw
    if not plan.get("notes") or plan.get("rap"):
        return None
    top = max(n["m"] for n in plan["notes"]) + 12 * octave
    f, t = pw.harvest(np.asarray(audio, np.float64), KSR, f0_floor=60.0, f0_ceil=max(600.0, 1.7 * float(ve.midi_hz(top))),
                      frame_period=5.0)
    out = {k: dict(whole=[], vowel=[], vowel_share=[]) for k in ("short", "long")}

    def cents(a, b, m):
        v = f[(t >= a) & (t <= b)]
        v = v[v > 0]
        if len(v) < 3:
            return None
        return round(float(1200 * np.log2(np.median(v) / ve.midi_hz(m + 12 * octave + detune_cents / 100.0))), 1)
    for s_, (k, na, nb) in zip(plan["sylls"], plan["nuc_frames"]):
        if len(s_["notes"]) != 1:
            continue                                       # melisma: several notes on one vowel
        n = s_["notes"][0]
        T, D, L = n["T"] - plan["R0"], n["D"], (nb - na) * FR
        a, b = T + max(0.04, 0.3 * D), T + D - 0.03
        kind = "short" if D < short_s else "long"
        out[kind]["whole"].append(cents(a, b, n["m"]))
        out[kind]["vowel"].append(cents(T + 0.3 * L, T + L - 0.015, n["m"]))
        out[kind]["vowel_share"].append(round(float(np.clip((T + L - a) / max(1e-3, b - a), 0, 1)), 2))
    return out


# ----------------------------------------------------------------------------- energy mid-rise onset detector
MR_FS = vo.FS
_SOS_MR = {"wide": butter(4, [100, 3500], "bandpass", fs=MR_FS, output="sos"),     # voice bar + formants
           "form": butter(4, [300, 3500], "bandpass", fs=MR_FS, output="sos")}     # formants only
MR_BANDS = ("wide", "form")


def band_env(x16, band="wide", win_ms=24.0):
    """Amplitude envelope of one band of a 16 kHz signal: sqrt of the Hann-smoothed (24 ms) band energy."""
    y = sosfiltfilt(_SOS_MR[band], x16)
    n = int(win_ms / 1000.0 * MR_FS) | 1
    w = np.hanning(n + 2)[1:-1]
    w /= w.sum()
    return np.sqrt(np.maximum(np.convolve(y * y, w, mode="same"), 0.0))


def _mr_cross(sub, dip, S):
    """Last upward crossing of the half-height dip + (S - dip) / 2 on the way to the peak at the end of `sub`
    (hysteresis: counted from the last sample below 60 % of that height)."""
    thr = dip + 0.5 * (S - dip)
    below = np.nonzero(sub < dip + 0.3 * (S - dip))[0]
    if len(below) == 0:
        return None
    j0 = int(below[-1])
    up = np.nonzero(sub[j0:] >= thr)[0]
    if len(up) == 0:
        return None
    return j0 + int(up[0]) - 1 if up[0] > 0 else j0


def mid_rise(env, T, T_prev=None, T_next=None, min_rise_db=6.0):
    """Vowel onset as the ear hears a beat: the MIDDLE of the last rise of the band envelope that at least
    doubles it (>= 6 dB) on the way to the vowel's peak. This is the definition of the independent audit
    (audit/onset.py), re-implemented here so the timing loop corrects what the audit measures; the
    periodicity detector of the shared engine only sees vowels after an unvoiced consonant or a rest.
      peak  = max of env in [T - 50 ms, min(T + 110 ms, T_next - 40 ms)]
      dip   = min of env in [max(T_prev + 30 ms, T - 220 ms), peak]; refined to the 50 ms before the crossing
      onset = half-height crossing of the last such rise (a later rise that again doubles the envelope wins)
    Returns None (no rise of 6 dB: no acoustic boundary) or dict(t, rise_db, flag) where flag is '' or
    'slow' (more than 60 ms from the crossing to 80 % of the vowel level) / 'shelf' (the envelope sags again
    3 dB or more after the crossing and the peak is 50 ms or more away): ill-defined onsets."""
    fs = MR_FS
    hi = T + 0.110 if T_next is None else min(T + 0.110, T_next - 0.040)
    hi = max(hi, T + 0.040)
    a, b = int(max(0.0, T - 0.050) * fs), int(hi * fs)
    if b <= a or b > len(env):
        return None
    ipk = a + int(np.argmax(env[a:b]))
    S = float(env[ipk])
    lo = T - 0.220 if T_prev is None else max(T_prev + 0.030, T - 0.220)
    il = int(max(0.0, lo) * fs)
    if il >= ipk or S <= 0:
        return None
    seg = env[il:ipk + 1]
    base, res = 0, None
    while True:
        sub = seg[base:]
        dip = float(sub.min())
        if 20 * np.log10((S + 1e-12) / (dip + 1e-12)) < min_rise_db:
            break
        j = _mr_cross(sub, dip, S)
        if j is None:
            break
        for _ in range(3):
            d2 = float(sub[max(0, j - int(0.050 * fs)):j + 1].min())
            if d2 <= dip * 1.02 + 1e-12:
                break
            dip = d2
            j2 = _mr_cross(sub, dip, S)
            if j2 is None:
                break
            j = j2
        rise = 20 * np.log10((S + 1e-12) / (dip + 1e-12))
        if rise < min_rise_db:
            break
        thr = dip + 0.5 * (S - dip)
        y0, y1 = sub[j], sub[min(j + 1, len(sub) - 1)]
        fr = 0.0 if y1 == y0 else float(np.clip((thr - y0) / (y1 - y0), 0, 1))
        res = dict(t=(il + base + j + fr) / fs, rise_db=float(rise), dip=dip)
        nb = base + j + int(0.020 * fs)
        if nb >= len(seg) - 2:
            break
        base = nb
    if res is None:
        return None
    i_on = max(0, int(round(res["t"] * fs)) - il)
    after = seg[i_on:]
    top = np.nonzero(after >= res["dip"] + 0.8 * (S - res["dip"]))[0]
    flag = "slow" if (len(top) == 0 or top[0] / fs > 0.060) else ""
    k20 = i_on + int(0.020 * fs)
    post = float(20 * np.log10((S + 1e-12) / (seg[k20:].min() + 1e-12))) if k20 < len(seg) - 1 else 0.0
    if post >= 3.0 and (ipk / fs - res["t"]) >= 0.050:
        flag += "shelf"
    return dict(t=float(res["t"]), rise_db=round(res["rise_db"], 1), flag=flag)


def line_readings(x16, Ts, Ds, per_ok, envs=None):
    """Every vowel onset of a phrase read by the three detectors, as error in ms against the target times Ts
    (s, in x16's timeline): 'per' = start of sustained periodicity (shared vox_onset detector; only where
    per_ok: after an unvoiced consonant or a rest), 'wide' / 'form' = energy mid-rise in 100-3500 Hz /
    300-3500 Hz (None: no 6 dB rise; '<band>_flag' marks ill-defined ones). A mid-rise reading that lies more
    than min(100 ms, 0.6 x the gap to the previous syllable) early, or within 15 ms of the previous
    syllable's reading, belongs to another syllable and is dropped."""
    if envs is None:
        envs = {b: band_env(x16, b) for b in MR_BANDS}
    out = [dict() for _ in Ts]
    for b in MR_BANDS:
        last = None
        for k, T in enumerate(Ts):
            r = mid_rise(envs[b], T, Ts[k - 1] if k else None, Ts[k + 1] if k + 1 < len(Ts) else None)
            if r is not None:
                lim = 0.100 if k == 0 else min(0.100, 0.6 * (T - Ts[k - 1]))
                if r["t"] < T - lim or (last is not None and abs(r["t"] - last) < 0.015):
                    r = None
            if r is not None:
                last = r["t"]
            out[k][b] = None if r is None else round(1000 * (r["t"] - T), 1)
            out[k][b + "_flag"] = "" if r is None else r["flag"]
            out[k][b + "_rise_db"] = None if r is None else r["rise_db"]
    for k, (T, D) in enumerate(zip(Ts, Ds)):
        m = vo.vowel_onset(x16, T, D) if per_ok[k] else None
        out[k]["per"] = None if m is None else round(1000 * (m - T), 1)
    return out


def combine_readings(r):
    """One timing error (ms) per syllable out of its detector readings, or None if nothing can be read.
    The energy mid-rise leads (it is what a listener's beat follows and what the audit measures): the mean of
    the two bands where they agree within 40 ms (so both can end inside a +-20 ms tolerance), else the band
    closer to the periodicity reading (or the smaller one). Where periodicity agrees within 30 ms the result is pulled a third of the way to it, so
    both definitions end up inside the tolerance; where it does not (a creaky, aperiodic vowel start, or weak
    voicing long before the vowel's energy) the energy reading is used alone and the voicing follows it.
    Ill-defined ('slow' / 'shelf') mid-rise readings are used only when there is nothing else."""
    P = r.get("per")
    good = [r[b] for b in MR_BANDS if r.get(b) is not None and not r.get(b + "_flag")]
    weak = [r[b] for b in MR_BANDS if r.get(b) is not None and r.get(b + "_flag")]
    if len(good) == 2:
        if abs(good[0] - good[1]) <= 40.0:
            E_ = 0.5 * (good[0] + good[1])
        elif P is not None:
            E_ = min(good, key=lambda v: abs(v - P))
        else:
            E_ = min(good, key=abs)
    elif len(good) == 1:
        E_ = good[0]
    else:
        E_ = None
    if E_ is None:
        if P is not None:
            return float(P)
        if weak:
            return 0.5 * float(min(weak, key=abs))
        return None
    if P is None:
        return float(E_)
    if abs(E_ - P) <= 30.0:
        return float(0.65 * E_ + 0.35 * P)
    return float(E_)


def readings_cost(rd):
    """What the timing loop minimises: squared excess over 8 ms of every well-defined reading."""
    c = 0.0
    for r in rd:
        vals = [r["per"]] + [r[b] for b in MR_BANDS if not r.get(b + "_flag")]
        for v in vals:
            if v is not None:
                c += max(0.0, min(abs(v), 80.0) - 8.0) ** 2
    return c


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
        """Phoneme string of a text. lang 'es' -> Mexican Spanish (rap_text.mexican).
        lang '<lang>+full' -> the same with FULL vowels for singing (rap_text.full_vowels): every unstressed
        syllable gets a secondary-stress mark, so the decoder does not reduce it."""
        base, _, mod = lang.partition("+")
        ph = super().phonemes(text, base)
        if base == "es":
            ph = rt.mexican(ph)
        if mod == "full":
            ph = rt.full_vowels(ph, self.tokens(ph)[1])
        return ph

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
            out.append("".join(chars[x] for x in s["onset"] if chars[x] not in " ˈˌ")
                       + "".join(chars[x] for x in s["nucleus"] if chars[x] not in " ˈˌ")
                       + "".join(chars[x] for x in s["coda"]))
        return out

    def plan_notes(self, text, lang, groups, ana_voice, nuclei, first=0, ana_speed=1.0, slog=None, nuc_share=None):
        """Duration plan (the same structure vox_engine.plan_sung returns) for explicit syllables.
        nuclei: token-index lists of ALL syllable nuclei of the text (rap_text.syllabify_es / apply_joins).
        groups: for every SOUNDING syllable (those from index `first` on) its notes [(T, D, midi[, bend])].
        first > 0: the syllables before it get zero frames: they are never heard but the text encoder still
        sees the whole line (used for the punch-ins, which only sound from the rhyme stress on).

        Frames are shared out like natural speech does it (NAT_ALLOC): inside every inter-onset interval the
        coda, the word space (one frame at most), the next onset consonants and the stress mark keep their
        natural durations, compressed by the interval's tempo (x0.5 .. 1, never stretched) and the vowel
        takes the rest. vox_engine.plan_sung gives
        stress marks and spaces no time at all, which is fine for held notes but makes fast isochronous
        syllables run together (measured: Whisper recall on 16 rap lines)."""
        ph = self.phonemes(text, lang)
        toks, chars = self.tokens(ph)
        st = self.style(ana_voice, len(toks))
        _, nat, _, _ = self.analyse(toks, st, ana_speed)
        nat = nat[1:-1]
        all_sylls = self.build_sylls(chars, nuclei)
        sylls = all_sylls[first:]
        assert len(sylls) == len(groups), (len(sylls), len(groups))
        nd = []
        for s, g in zip(sylls, groups):
            s["notes"] = [dict(T=float(n[0]), D=float(n[1]), m=float(n[2]), b=float(n[3]) if len(n) > 3 else 0.0)
                          for n in g]
            nd.extend(s["notes"])
        nS = len(sylls)
        Ts = [s["notes"][0]["T"] for s in sylls]
        if NAT_ALLOC:
            # the stress mark (and the word space) before a vowel are timed with its onset consonants
            for k, s in enumerate(sylls):
                a0 = s["nucleus"][0]
                on = list(s["onset"])
                if a0 > 0 and chars[a0 - 1] in "ˈˌ":
                    on = on + [a0 - 1]
                if SPACE_FRAMES and k > 0:
                    lo_ = sylls[k - 1]["nucleus"][-1] + 1
                    sp = [x for x in range(lo_, a0) if chars[x] == " "]
                    if sp and nat[sp[-1]] > 0:
                        on = [sp[-1]] + on
                s["onset"] = on

        def floor_of(x, room=True):
            c = chars[x]
            if ve.cls_of(c) != "C":
                return 0
            return ve.ONSET_MIN if (c not in ve.SONORANT and room and not NAT_ALLOC) else 1

        def nat_fr(x, scale):
            c = chars[x]
            k_ = ve.cls_of(c)
            if k_ == "C":
                return int(np.clip(round(nat[x] * scale), 2 if (c in ve.UNV_OBS and scale >= 0.8) else 1, 4))
            if k_ == "S":
                return int(np.clip(round(nat[x] * scale), 0, 3))
            return int(np.clip(round(nat[x] * scale), 0, 1))          # a word space: at most one frame

        def shrink(fr, idx, need):
            """Take `need` frames away from the longest tokens (down to their floors)."""
            fr = list(fr)
            while need > 0:
                cand = [j for j in range(len(fr)) if fr[j] > floor_of(idx[j])]
                if not cand:
                    break
                j = max(cand, key=lambda j: (fr[j] - floor_of(idx[j]), fr[j]))
                fr[j] -= 1
                need -= 1
            return fr

        def cap_glide(fr, idx):
            """A glide (j, w) right before the nucleus, plus the stress mark, gets GLIDE_MAX_FRAMES in all: a
            long glide is as loud as the vowel and is heard (and measured) as the vowel starting early. The
            frames it gives up go to the vowel before it."""
            fr = list(fr)
            cons = [j for j, x in enumerate(idx) if ve.cls_of(chars[x]) == "C"]
            if not cons or chars[idx[cons[-1]]] not in "jw":
                return fr
            grp = [cons[-1]] + [j for j, x in enumerate(idx) if j > cons[-1] and chars[x] in "ˈˌ"]
            while sum(fr[j] for j in grp) > GLIDE_MAX_FRAMES:
                j = max(grp, key=lambda j: (fr[j], -j))
                if fr[j] <= (1 if j == cons[-1] else 0):
                    break
                fr[j] -= 1
            return fr

        share = NUC_SHARE if nuc_share is None else float(nuc_share)

        def nuc_want(F):
            """Frames the vowel must keep out of an inter-onset interval of F frames."""
            if F < 5:
                return 1
            return max(2, int(round(share * F)))

        if NAT_ALLOC:
            on0 = cap_glide([nat_fr(x, 1.0) for x in sylls[0]["onset"]], sylls[0]["onset"])
        else:
            on0 = [int(np.clip(nat[x], ve.ONSET_MIN if chars[x] not in ve.SONORANT else 1, 3)) for x in sylls[0]["onset"]]
            while sum(on0) > max(len(on0), 5):
                on0[int(np.argmax(on0))] -= 1
        lead_s = (ve.PAD_LEAD + sum(on0)) * FR
        best = None
        for ph_ms in np.arange(0.0, 25.0, 0.5):
            R0 = Ts[0] - lead_s - ph_ms / 1000.0
            vf = [int(round((T - R0) / FR)) for T in Ts]
            err = [R0 + v * FR - T for v, T in zip(vf, Ts)]
            sc = max(abs(e) for e in err)
            if best is None or sc < best[0] - 1e-9:
                best = (sc, R0, vf, err)
        _, R0, VF, verr = best
        VF = list(VF)
        # every interval must hold its vowel and one frame per consonant (thirty-second notes on a 25 ms grid)
        for k in range(1, nS):
            ncons = sum(ve.cls_of(chars[x]) == "C" for x in sylls[k - 1]["coda"] + sylls[k]["onset"])
            if VF[k] < VF[k - 1] + 1 + ncons:
                VF[k] = VF[k - 1] + 1 + ncons
        on_f = [on0] + [None] * (nS - 1)
        co_f = [None] * nS
        nuc_n = [None] * nS
        rests = [False] * nS
        for k, s in enumerate(sylls):
            last = s["notes"][-1]
            end_note = int(round((last["T"] + last["D"] - R0) / FR))
            vs_ = [x for x in s["nucleus"] if chars[x] in ve.VOWELS or chars[x] == ve.LENGTH]
            nat_n = max(1.0, float(sum(nat[x] for x in vs_)))
            co_idx = list(s["coda"])
            if k + 1 < nS:
                on_idx = list(sylls[k + 1]["onset"])
                rest = (Ts[k + 1] - (last["T"] + last["D"])) >= 0.12
            else:
                on_idx, rest = [], True
            rests[k] = rest
            if NAT_ALLOC:
                if rest:
                    on = cap_glide([nat_fr(x, 1.0) for x in on_idx], on_idx)
                    limit = (VF[k + 1] - sum(on)) if k + 1 < nS else end_note
                    if k + 1 < nS and limit < VF[k] + 1 + len(co_idx):
                        on = shrink(on, on_idx, VF[k] + 1 + len(co_idx) - limit)
                        limit = VF[k + 1] - sum(on)
                    F1 = max(1 + len(co_idx), min(end_note, limit) - VF[k])
                    sc_ = float(np.clip(F1 / (nat_n + sum(nat[x] for x in co_idx)), 0.5, 1.0))
                    co = [nat_fr(x, sc_) for x in co_idx]
                    nuc = F1 - sum(co)
                    if nuc < (2 if F1 >= 5 else 1):
                        co = shrink(co, co_idx, (2 if F1 >= 5 else 1) - nuc)
                        nuc = F1 - sum(co)
                else:
                    F = VF[k + 1] - VF[k]
                    idx = co_idx + on_idx
                    sc_ = float(np.clip(F / (nat_n + sum(nat[x] for x in idx)), 0.5, 1.0))
                    fr = [nat_fr(x, sc_) for x in idx]
                    fr = fr[:len(co_idx)] + cap_glide(fr[len(co_idx):], on_idx)
                    nuc = F - sum(fr)
                    want = nuc_want(F)
                    if nuc < want:
                        fr = shrink(fr, idx, want - nuc)
                        nuc = F - sum(fr)
                    co, on = fr[:len(co_idx)], fr[len(co_idx):]
            else:
                on = None
                if k + 1 < nS:
                    on = [int(np.clip(nat[x], ve.ONSET_MIN if chars[x] not in ve.SONORANT else 1, 3)) for x in on_idx]
                    cap = max(len(on), min(5, int(round(0.38 * (Ts[k + 1] - Ts[k]) / FR))))
                    while sum(on) > cap:
                        on[int(np.argmax(on))] -= 1
                    while VF[k + 1] < VF[k] + 1 + sum(on) and sum(on) > len(on):
                        on[int(np.argmax(on))] -= 1
                co = [int(np.clip(nat[x], 1, 3)) for x in co_idx]
                nxt = Ts[k + 1] if k + 1 < nS else last["T"] + last["D"]
                ccap = max(len(co), min(3, int(round(0.3 * (nxt - Ts[k]) / FR))))
                while sum(co) > ccap:
                    co[int(np.argmax(co))] -= 1
                next_on = (VF[k + 1] - sum(on)) if k + 1 < nS else None
                end_k = end_note if next_on is None else (min(end_note, next_on) if rest else next_on)
                nuc = end_k - VF[k] - sum(co)
                while nuc < 2 and sum(co) > len(co):
                    co[int(np.argmax(co))] -= 1
                    nuc = end_k - VF[k] - sum(co)
                while nuc < 1 and sum(co) > 0:
                    co[int(np.argmax(co))] -= 1
                    nuc = end_k - VF[k] - sum(co)
            if k + 1 < nS:
                on_f[k + 1] = on
            co_f[k] = co
            nuc_n[k] = max(1, int(nuc))
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
            co = list(co_f[k])
            nuc = nuc_n[k]
            next_on = (VF[k + 1] - sum(on_f[k + 1])) if k + 1 < nS else None
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
                tgt = [x for x in s["gap"] if ve.cls_of(chars[x]) in "WP" and x not in sylls[k + 1]["onset"]]
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
                    exact_v=[(T - R0) / FR for T in Ts], rise_loop=True, tail=None,
                    VF=VF, vowel_err=verr, dur=dur, rest_mask=rest_mask, nuc_frames=nuc_frames,
                    notes=nd, onset_frames=on_f, coda_frames=co_f, ntok0=len(toks), lead_frames=lead_frames,
                    nuc_alloc=nuc_alloc, gaps=gaps, chars0=chars, coda_used=coda_used, first=first)

    # --- which vowel onsets can be seen acoustically (the onset list may hold a space / stress mark)
    def _last_cons(self, plan, k):
        ch = plan["chars"]
        on = [x for x in plan["sylls"][k]["onset"] if ve.cls_of(ch[x]) == "C"]
        return ch[on[-1]] if on else None

    def _rest_before(self, plan, k):
        return {kk: g for (kk, _, g) in plan["gaps"]}.get(k - 1, 0) > 0

    def measurable(self, plan, k):
        c = self._last_cons(plan, k)
        if c is not None:
            return c in ve.UNV_OBS
        if k == 0 or self._rest_before(plan, k):
            return True
        cod = plan["sylls"][k - 1]["coda"]
        return bool(cod) and plan["chars"][cod[-1]] in ve.UNV_OBS

    def after_unvoiced(self, plan, k):
        c = self._last_cons(plan, k)
        if c is not None:
            return c in ve.UNV_OBS
        if k == 0 or self._rest_before(plan, k):
            return False
        cod = plan["sylls"][k - 1]["coda"]
        return bool(cod) and plan["chars"][cod[-1]] in ve.UNV_OBS

    def before_unvoiced(self, plan, k):
        s_ = plan["sylls"][k]
        ch = plan["chars"]
        if s_["coda"]:
            return ch[s_["coda"][0]] in ve.UNV_OBS
        if k + 1 < len(plan["sylls"]) and {kk: g for (kk, _, g) in plan["gaps"]}.get(k, 0) == 0:
            on = [x for x in plan["sylls"][k + 1]["onset"] if ve.cls_of(ch[x]) == "C"]
            return bool(on) and ch[on[0]] in ve.UNV_OBS
        return False

    # --- hooks into the shared sung path
    def pitch_curve(self, plan, t, detune_cents=0.0, octave=0, seed=0, vib_depth=0.22, scoop=0.4, expressive=True):
        if not plan.get("rap"):
            # chant: the shared engine's curve (55 ms glides around the vowel onset, scoops and vibrato only
            # on long notes, slow drift); around notes shorter than SHORT_NOTE_S the glide is redone shorter
            # and placed before the vowel onset, so a sixteenth note sits on its pitch from its first cycles
            pitch = super().pitch_curve(plan, t, detune_cents, octave, seed, vib_depth, scoop, expressive)
            notes = [n for s_ in plan["sylls"] for n in s_["notes"]]
            for j in range(1, len(notes)):
                p0, p1 = notes[j - 1], notes[j]
                legato = p1["T"] - (p0["T"] + p0["D"]) < 0.12
                if not legato or p0["m"] == p1["m"] or min(p0["D"], p1["D"]) >= SHORT_NOTE_S:
                    continue
                a, b = p1["T"] - 0.032, p1["T"] + 0.004
                sel = (t >= p1["T"] - 0.035) & (t < p1["T"] + 0.03)
                if sel.any():
                    w = _smoothstep((t[sel] - a) / (b - a))
                    old_w = _smoothstep((t[sel] - (p1["T"] - 0.035)) / 0.055)
                    pitch[sel] += (p1["m"] - p0["m"]) * (w - old_w)
            return pitch
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
        if plan.get("pitch_floor") is not None:
            pitch = np.maximum(pitch, plan["pitch_floor"])
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
        lifter_ms = plan.get("lifter_ms", ve.CONS_LIFTER_MS)
        if lifter_ms and lifter_ms > 0:
            sel = ~wa["nucl"]
            if sel.any():
                nb = sp.shape[1]
                c = np.fft.irfft(np.log(sp[sel] + 1e-16), axis=1)
                q = int(lifter_ms / 1000.0 * KSR)
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

    def gate(self, plan, audio, sr=KSR, open_s=None):
        """As the shared engine's gate (rests are muted, a coda may ring 30 ms into a rest), except at both
        ends of the phrase:
        start: the gate opens (12 ms raised-cosine fade) at `open_s` (render time, s): 30 ms before the first
               syllable's onset consonants as the time map places them (never more than LINE_PRE_MAX_S
               before the first vowel), or 12 ms before a phrase-initial vowel (the shared gate opens 50 ms
               before the PLANNED consonant, which lets a phrase-initial vowel swell in up to 40 ms early);
        end:   the fade (25 ms) reaches zero END_AFTER_S after the last note's written end, so the phrase is
               40 dB down at its written end (the shared gate lets it ring about 45 ms longer)."""
        g = np.ones(len(audio))
        hop = HOP * sr / KSR
        r = int(0.012 * sr)
        rel = int(ve.GATE_RELEASE * sr)
        last = plan["sylls"][-1]["notes"][-1]
        end_s = last["T"] + last["D"] - plan["R0"]
        nrm = len(plan["rest_mask"])
        for i, (a, b) in enumerate(plan["rest_mask"]):
            s1 = min(len(audio), int(b * hop))
            if i == nrm - 1:
                s0 = min(s1, int(round((end_s + END_AFTER_S) * sr)))
                rr = int(END_FADE_S * sr)
            else:
                s0 = min(s1, int(a * hop) + rel)
                rr = r
            g[s0:s1] = 0.0
            if s0 > 0:
                lo = max(0, s0 - rr)
                g[lo:s0] = np.minimum(g[lo:s0], 0.5 + 0.5 * np.cos(np.pi * (np.arange(lo, s0) - lo + 0.5) / max(1, s0 - lo)))
            if s1 < len(audio):
                e = min(len(audio), s1 + r)
                g[s1:e] = np.minimum(g[s1:e], np.linspace(0, 1, e - s1))
        if open_s is None:
            lead = int(max(0, int(plan["lead_frames"]) - 2) * hop)
            r2 = int(0.015 * sr)
        else:
            lead = max(0, int(round(open_s * sr)))
            r2 = r
        g[:lead] = 0.0
        ramp = 0.5 - 0.5 * np.cos(np.pi * (np.arange(r2) + 0.5) / r2)
        g[lead:lead + r2] = np.minimum(g[lead:lead + r2], ramp[:max(0, len(g) - lead)])
        return audio * g

    def onset_contrast(self, plan, y, tcs, pad=0.25):
        """Gain curve (linear, len(y)) that gives every syllable an audible attack.
        A vowel that follows a voiced consonant, a glide or another vowel has no silence before it; if the
        sound before it is nearly as loud as the vowel itself (nasal and lateral murmurs come out of the
        vocoder at vowel level, where in a human voice they sit 6-10 dB lower) there is no beat to hear and
        no acoustic onset to measure. Where the envelope (100-3500 Hz or 300-3500 Hz, whichever shows less)
        rises by less than CONTRAST_DB from the consonant into the vowel, the consonant (at most the last 70 ms before the vowel onset) is turned
        down by the missing amount, at most CONTRAST_MAX_DB, with 10 ms / 14 ms ramps; the ramp back up ends
        on the vowel onset. Syllables after an unvoiced consonant or a rest are left alone."""
        n = len(plan["sylls"])
        if CONTRAST_MAX_DB <= 0 or n == 0:
            return np.ones(len(y))
        fs = MR_FS
        tb = self.bounds(plan)
        Ts = [q["v"] * FR for q in tb]
        z = np.zeros(int(pad * KSR))
        g_db = np.zeros(len(y))
        tt = np.arange(len(y)) / KSR
        todo = []
        for k, T in enumerate(Ts):
            if self.after_unvoiced(plan, k):
                continue
            first = k == 0 or self._rest_before(plan, k)
            if first and self._last_cons(plan, k) is None:
                continue                                   # a vowel after silence
            tc = max(float(tcs[k]), T - 0.070)
            if T - tc < 0.030:
                tc = T - 0.030 if not first else tc        # hiatus: the end of the previous vowel dips
            if T - tc < 0.012:
                continue
            todo.append((k, T, tc))
        if not todo:
            return np.ones(len(y))
        for rnd in range(2):
            x16 = vo.to16(np.concatenate([z, y * 10 ** (g_db / 20.0), z]), KSR)
            envs = [band_env(x16, b) for b in MR_BANDS]
            for (k, T, tc) in todo:
                Tn = Ts[k + 1] if k + 1 < n else None
                hi = T + 0.110 if Tn is None else max(T + 0.040, min(T + 0.110, Tn - 0.040))
                a_, b_ = int((tc + 0.008 + pad) * fs), int((T - 0.002 + pad) * fs)
                rise = 1e9                                  # the smaller rise of the two bands
                for env in envs:
                    pk = float(env[int((T - 0.010 + pad) * fs):int((hi + pad) * fs)].max())
                    dip = float(env[a_:max(a_ + 1, b_)].min())
                    rise = min(rise, 20 * np.log10((pk + 1e-12) / (dip + 1e-12)))
                have = -float(g_db[min(len(g_db) - 1, int((0.5 * (tc + T)) * KSR))])
                need = float(np.clip(have + CONTRAST_DB - rise, 0.0, CONTRAST_MAX_DB))
                if need < 0.5 or need <= have + 0.25:
                    continue
                down = _smoothstep((tt - (tc - 0.004)) / 0.010)
                up = _smoothstep((tt - (T - 0.014)) / 0.014)
                sel = (tt >= tc - 0.004) & (tt <= T)
                g_db[sel] = np.minimum(g_db[sel], -need * (down[sel] - up[sel]))
        return 10 ** (g_db / 20.0)

    def onset_readings(self, plan, audio, pad=0.25):
        """Detector readings (line_readings) for every syllable of a render of `plan` (sample 0 = plan['R0'])."""
        z = np.zeros(int(pad * KSR))
        x16 = vo.to16(np.concatenate([z, np.asarray(audio, np.float64), z]), KSR)
        tb = self.bounds(plan)
        return line_readings(x16, [q["v"] * FR + pad for q in tb], [(q["z"] - q["v"]) * FR for q in tb],
                             [self.measurable(plan, k) for k in range(len(tb))])

    def render_sung(self, plan, voice, detune_cents=0.0, octave=0, seed=0, blend=None, energy_boost=0.0,
                    vib_depth=0.22, neural_shift=None, style_shift=0, passes=None):
        """The shared engine's sung render with this module's closed timing loop.
        As vox_engine.Engine.render_sung up to the first synthesis (decoder sings the source plan, vowel
        onsets measured on the decoder audio by periodicity, WORLD time map onto the note times). Then, up to
        `passes` times: the finished CLEAN render (no rasp) is read by three detectors per syllable
        (periodicity where there is a voicing boundary, energy mid-rise in two bands everywhere: see
        line_readings / combine_readings), every syllable that is off by more than LOOP_TOL_MS is moved in the
        time map (by its whole error; if a pass makes the line worse it is retried from the best state with
        half the step), the voicing track follows the moved vowel onsets (an aperiodic, creaky vowel start becomes
        voiced; weak voicing long before the vowel's energy becomes part of the consonant), and the render is
        synthesised again. The pass with the lowest readings_cost is kept; the rasp (plan['fx']['rasp']) is
        applied in one last synthesis with exactly that time map.
        Returns dict(audio, clean (the same render without rasp, or None), readings (of the clean render),
        passes, timing, tc (target-time start of every syllable's onset consonants, s from R0), ...)."""
        if neural_shift is None:
            top = max(n["m"] for n in plan["notes"]) + 12 * octave
            ceiling = ve.NEURAL_TOP_M if voice.split("|")[0][1] == "m" else ve.NEURAL_TOP_F
            neural_shift = int(np.clip(ceiling - top, -12, 0))
        passes = LOOP_PASSES if passes is None else int(passes)
        src = self.source_plan(plan)
        nr = self.render_neural(src, voice, neural_shift, style_shift, blend, energy_boost)
        x = nr["audio"]
        meas = self.measure_vowels(src, x)
        offs = self.measure_offsets(src, x, meas)
        wa = self.world_analyse(nr)
        fp = 0.005
        fx = plan.get("fx") or {}
        plan_c = dict(plan, fx=dict(fx, rasp=None)) if fx.get("rasp") else plan

        def voicing(ms):
            """WORLD voicing track on the source timeline, following the (corrected) vowel onsets."""
            v = wa["v"].copy()
            for k, mk in enumerate(ms):
                if mk is None:
                    continue
                q = src["bnd"][k]
                a_ = min(q["c"] * FR, mk - 0.005)
                i0, i1 = int(np.floor(a_ / fp)), int(np.floor(mk / fp))
                if self.after_unvoiced(src, k):
                    v[max(0, i0):max(0, i1)] = False
                iv = int(np.floor(q["v"] * FR / fp))
                if i1 < iv:
                    v[max(0, i1):iv] = True
            for k, ok in enumerate(offs or []):
                if ok is None:
                    continue
                q = src["bnd"][k]
                iz, io = int(np.floor(q["z"] * FR / fp)), int(np.floor(ok / fp))
                if io > iz:
                    v[iz:io] = True
                else:
                    v[io:iz] = False
            return v

        sos_hp = butter(4, ve.XOVER_HZ, "highpass", fs=KSR, output="sos")
        sos_lp = butter(4, ve.XOVER_HZ, "lowpass", fs=KSR, output="sos")
        nb_ = wa["ap"].shape[1]
        fq_ = np.arange(nb_) / (nb_ - 1) * KSR / 2
        apm = wa["ap"][:, (fq_ >= 1000) & (fq_ <= 4000)].mean(axis=1)
        n_t = int(plan["dur"].sum()) * HOP
        t5 = np.arange(int(n_t / KSR / fp) + 1) * fp
        nsrc = len(wa["f0"])
        lsp_src = np.log(wa["sp"] + 1e-16)

        def synth(pl, fmap, holds, v_src, segs, cgain=None):
            s5 = fmap(t5) / fp
            i0 = np.clip(np.floor(s5).astype(int), 0, nsrc - 1)
            i1 = np.clip(i0 + 1, 0, nsrc - 1)
            w = np.clip(s5 - i0, 0, 1)[:, None]
            lsp = (1 - w) * lsp_src[i0] + w * lsp_src[i1]
            ap = (1 - w) * wa["ap"][i0] + w * wa["ap"][i1]
            for (ta, tb_, sa, sb_) in holds:
                ja, jb = int(np.ceil(ta / fp)), int(np.floor(tb_ / fp))
                ka, kb = int(np.floor(sa / fp)), int(np.ceil(sb_ / fp)) + 1
                if jb - ja < 4 or kb - ka < 2:
                    continue
                msp = np.mean(lsp_src[ka:kb], axis=0)
                map_ = np.mean(wa["ap"][ka:kb], axis=0)
                jj = np.arange(ja, jb)
                r = int(round(0.04 / fp))
                g = np.minimum(1.0, np.minimum(jj - ja, jb - 1 - jj) / max(1, r))[:, None] * ve.HOLD_MEAN
                lsp[ja:jb] = (1 - g) * lsp[ja:jb] + g * msp
                ap[ja:jb] = (1 - g) * ap[ja:jb] + g * map_
            inn = np.clip(np.round(s5).astype(int), 0, nsrc - 1)
            wt = dict(t=t5, sp=np.exp(lsp), ap=ap, v=v_src[inn], nucl=wa["nucl"][inn])
            y, f0t = self.world_synth(pl, wt, detune_cents, octave, seed, vib_depth)
            y = np.pad(y, (0, max(0, n_t - len(y))))[:n_t]
            tt = np.arange(n_t) / KSR
            xw = ve._interp_cubic(x, fmap(tt) * KSR)
            vs = np.interp(tt, t5, wt["v"].astype(float))
            k = int(0.010 * KSR)
            vs = np.convolve(vs, np.ones(k) / k, mode="same")
            ux = np.interp(tt, t5, np.clip((apm[inn] - 0.25) / 0.3, 0, 1))
            ux = np.convolve(ux, np.ones(k) / k, mode="same")
            hx, hy = sosfiltfilt(sos_hp, xw), sosfiltfilt(sos_hp, y)
            unv = ux * hx + (1 - ux) * hy + sosfiltfilt(sos_lp, y)
            y = vs * y + (1 - vs) * unv
            tcs = [sg[0] for sg in segs[1::4]]            # (tc[k], T[k], ...) segments of the time map
            first_on = [x_ for x_ in plan["sylls"][0]["onset"] if ve.cls_of(plan["chars"][x_]) == "C"]
            t_first = self.bounds(plan)[0]["v"] * FR
            # a phrase that starts with a vowel has no consonant to let in: the gate opens right on it,
            # wherever the loop has moved the vowel (the weak voicing before its attack stays out)
            y = self.gate(pl, y, open_s=max(tcs[0] - GATE_PRE_S, t_first - LINE_PRE_MAX_S) if first_on
                          else t_first - VOWEL_PRE_S)
            y = self.level_syllables(pl, y)
            if cgain is None:
                cgain = self.onset_contrast(pl, y, tcs)
            return y * cgain, f0t, wt, cgain

        meas = self.effective_onsets(src, meas, voicing(meas))
        meas0 = list(meas)
        best, hist, gain, frozen = None, [], 1.0, set()
        for it in range(passes + 1):
            fmap, segs, _, holds = self.time_map(plan, src, meas, offs)
            v_src = voicing(meas)
            y, f0t, wt, cg = synth(plan_c, fmap, holds, v_src, segs)
            rd = self.onset_readings(plan, y)
            err = [combine_readings(r) for r in rd]
            cost = readings_cost(rd)
            hist.append(round(cost, 1))
            if best is None or cost < best["cost"] - 1e-9:
                best = dict(cost=cost, meas=list(meas), y=y, f0t=f0t, wt=wt, segs=segs, rd=rd, err=err, fmap=fmap,
                            holds=holds, v_src=v_src, it=it, cg=cg)
                gain = min(1.0, gain * 1.25) if it else 1.0
            else:
                # the step made the line worse (syllables pull at each other, or a move hit a limit of the
                # time map): go back to the best state, leave alone the syllables that did not respond, and
                # take half the step with the rest
                for k, e0 in enumerate(best["err"]):
                    if e0 is not None and abs(e0) > LOOP_TOL_MS and (err[k] is None or abs(err[k]) > abs(e0) - 2.0):
                        frozen.add(k)
                gain *= 0.5
            bad = [k for k, e in enumerate(best["err"]) if e is not None and abs(e) > LOOP_TOL_MS and k not in frozen]
            if it == passes or not bad or best["cost"] <= LOOP_GOOD_COST:
                break
            meas = list(best["meas"])
            for k in bad:
                base = meas[k] if meas[k] is not None else src["bnd"][k]["v"] * FR
                meas[k] = base + gain * float(np.clip(best["err"][k], -LOOP_MAX_STEP_MS, LOOP_MAX_STEP_MS)) / 1000.0
        clean = None
        y, f0t, wt = best["y"], best["f0t"], best["wt"]
        if plan_c is not plan:
            clean = y
            y, f0t, wt, _ = synth(plan, best["fmap"], best["holds"], best["v_src"], best["segs"], best["cg"])
        timing = []
        for k, q in enumerate(src["bnd"]):
            m0, m1 = meas0[k], best["meas"][k]
            b0 = m0 if m0 is not None else q["v"] * FR
            timing.append(dict(measured_src_ms=None if m0 is None else round(1000 * (m0 - q["v"] * FR), 1),
                               second_pass_ms=0.0 if m1 is None else round(1000 * (m1 - b0), 1)))
        tc = [sg[0] for sg in best["segs"][1::4]]
        cgd = 20 * np.log10(best["cg"] + 1e-9)
        duck = [round(float(-cgd[int(np.clip((0.5 * (a_ + q["v"] * FR) if q["v"] * FR - a_ >= 0.03 else q["v"] * FR - 0.016)
                                              * KSR, 0, len(cgd) - 1))]), 1)
                for a_, q in zip(tc, self.bounds(plan))]
        return dict(duck_db=duck, audio=y.astype(np.float32), clean=None if clean is None else clean.astype(np.float32),
                    f0_target5=f0t, t5=t5, nucl5=wt["nucl"], neural=x, f0_fed=nr["f0_fed"], nucl=nr["nucl"],
                    meas=best["meas"], timing=timing, segs=best["segs"], readings=best["rd"], tc=tc,
                    loop=dict(costs=hist, kept=best["it"]))

    def verify_onsets(self, plan, audio, extra_delay=0.0):
        """Periodicity-detector error (ms) per syllable (shared engine's definition; None = no boundary)."""
        return ve.Engine.verify_onsets(self, plan, audio, extra_delay)

    def cons_before(self, plan, k):
        """The consonant right before syllable k's vowel (its onset, else the previous coda), or None."""
        c = self._last_cons(plan, k)
        if c is None and k > 0 and plan["sylls"][k - 1]["coda"] and not self._rest_before(plan, k):
            c = plan["chars"][plan["sylls"][k - 1]["coda"][-1]]
        return c

    def render_neural(self, plan, voice, neural_shift=-12, style_shift=0, blend=None, energy_boost=0.0):
        """As the base engine; with plan['speech_energy'] the decoder keeps Kokoro's own energy contour
        (stressed syllables louder) instead of the sung path's 'sustain every vowel at the phrase level'."""
        if not plan.get("speech_energy"):
            return super().render_neural(plan, voice, neural_shift, style_shift, blend, energy_boost)
        toks, chars, dur = plan["toks"], plan["chars"], plan["dur"]
        nt = int(np.clip(plan.get("ntok0", len(toks)) + style_shift, 1, 510))
        st = self.style(voice, nt, blend)
        _, f0p, n_p = self.run_dur(toks, st, dur)
        nf = len(f0p)
        assert nf == 2 * int(dur.sum()), (nf, dur.sum())
        tok_of = np.repeat(np.arange(len(dur)), 2 * dur)
        tok_of = tok_of[np.minimum(np.arange(nf) + 2 * LAG, nf - 1)]
        cls = np.array(["X"] + [ve.cls_of(c) for c in chars] + ["X"])
        chs = np.array(["<"] + list(chars) + [">"])
        nucl = np.zeros(nf, bool)
        for (k, a, b) in plan["nuc_frames"]:
            nucl[2 * a:2 * b] = True
        silent = np.zeros(nf, bool)
        for (a, b) in plan["rest_mask"]:
            silent[2 * a:2 * b] = True
        c = cls[tok_of]
        ch = chs[tok_of]
        son = np.isin(ch, list(ve.SONORANT))
        voiced = (nucl | (c == "V") | (c == "L") | son | (f0p > 45.0)) & (c != "X") & ~silent
        unv_obs = (c == "C") & ~son & ~np.isin(ch, list(ve.VOICED_OBS))
        voiced &= ~(unv_obs & (f0p <= 45.0))
        t = plan["R0"] + (np.arange(nf) + 0.5) * F0R
        pitch = self.pitch_curve(plan, t, expressive=False) + neural_shift
        f0 = np.where(voiced, ve.midi_hz(pitch), 0.0).astype(np.float32)
        n = n_p.astype(np.float64) + energy_boost
        audio = self.run_sing(toks, st, dur, f0, n.astype(np.float32))
        return dict(audio=audio.astype(np.float64), f0_fed=f0, voiced=voiced, nucl=nucl, f0_pred=f0p, n=n,
                    neural_shift=neural_shift)

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
        room = 0.075 + 0.03 * min(2, len(first["onset"]))      # gate release (42 ms) + the next line's consonants
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
        plan["speech_energy"] = bool(cfg.get("speech_energy", False))
        plan["pitch_floor"] = float(cfg["recite"] - cfg.get("floor_below", 3.5) + pitch_up)
        if cfg.get("lifter_ms") is not None:
            plan["lifter_ms"] = float(cfg["lifter_ms"])
        plan["midi"] = [float(m) for m in midi]
        return plan

    def _sung_cached(self, tag, plan, voice, key, **kw):
        """render_sung with the finished audio cached on disk. Returns (audio float64, meta); for a raspy plan
        meta['_clean'] is the same render without the rasp (same time map: the timing reference)."""
        def run():
            r = self.render_sung(plan, voice, **kw)
            f0t = r["f0_target5"]
            vv = f0t[f0t > 0]
            rd = r["readings"]
            meta = dict(readings=rd, verify_per_ms=[q["per"] for q in rd], loop=r["loop"], timing=r["timing"],
                        duck_db=r["duck_db"],
                        tc=[round(float(v), 4) for v in r["tc"]],
                        rasp=rasp_metrics(r["audio"]) if tag in ("rap", "gangrap") else None,
                        f0_target_med=float(np.median(vv)) if len(vv) else None,
                        pitch=pitch_check(r["audio"], f0t, r["nucl5"]))
            if r["clean"] is not None:
                n_ = min(len(r["audio"]), len(r["clean"]))
                return np.stack([r["audio"][:n_], r["clean"][:n_]]), meta
            return r["audio"], meta
        # the key is independent of where in the song the phrase sits: a hook line repeated at another bar
        # (same text, same notes) is rendered once
        k = dict(key=key, voice=voice, kw=kw, text=plan["text"], dur=plan["dur"].tolist(),
                 notes=[(round(n["T"] - plan["R0"], 5), round(n["D"], 5), round(n["m"], 4), round(n.get("b", 0.0), 4))
                        for n in plan["notes"]],
                 fx=plan.get("fx"), rap=bool(plan.get("rap")), level=plan.get("level"),
                 se=bool(plan.get("speech_energy")), lift=plan.get("lifter_ms"), floor=plan.get("pitch_floor"),
                 loop=(LOOP_PASSES, LOOP_TOL_MS, LOOP_MAX_STEP_MS, LOOP_GOOD_COST, 3), con=(CONTRAST_DB, CONTRAST_MAX_DB),
                 gl=(GLIDE_MAX_FRAMES, GATE_PRE_S, LINE_PRE_MAX_S, END_AFTER_S, END_FADE_S, SHORT_NOTE_S), ph=plan["ph"])
        if not [x_ for x_ in plan["sylls"][0]["onset"] if ve.cls_of(plan["chars"][x_]) == "C"]:
            k["vowel_start"] = VOWEL_PRE_S        # (only these plans depend on it: the others keep their cache)
        a, meta = self.memo(tag, k, run)
        if a.ndim == 2:                       # [raspy, clean twin]
            meta = dict(meta, _clean=edge_fade(a[1]))
            a = a[0]
        return edge_fade(a), meta

    def render_rap(self, f, cfg=RAPPER, seed=0):
        """One rap line in the rapper's voice. Returns dict(audio 24 kHz, R0, plan, meta)."""
        plan = self.rap_plan(f, cfg)
        a, meta = self._sung_cached("rap", plan, cfg["voice"], dict(cfg={k: v for k, v in cfg.items()}),
                                    vib_depth=0.0, seed=int(seed), blend=cfg.get("blend"),
                                    neural_shift=int(cfg["neural_shift"]))
        return dict(audio=a, clean=meta.pop("_clean", None), R0=plan["R0"], plan=plan, meta=meta)

    def render_punch(self, f, voice, cfg=RAPPER, seed=0, detune=0.0, pitch_up=PUNCH_UP, formant_shift=1.0, first=None,
                     lead_in=None):
        """A gang voice doubling the rhyme (first=None), or rapping the whole line (first=0, for RAP lines the
        score gives to the gang).
        The double is HEARD from the rhyme stress on, but it is SUNG from `lead_in` syllables earlier: a
        syllable in the middle of a phrase is articulated like the lead's and can be read by the timing loop
        (the first syllable of a render cannot: nothing precedes it). The lead-in is then gated away: the
        gate opens with a PUNCH_FADE_S fade at the rhyme syllable's own onset consonants, never earlier than
        PUNCH_PRE_MAX_S before the anchor.
        Returns dict(audio, R0, plan, meta, rhyme_index (of the rhyme syllable inside the plan), gate_s)."""
        whole = first is not None
        r0 = int(first) if whole else int(f["rhyme"])
        lead_in = PUNCH_LEAD_IN if lead_in is None else int(lead_in)
        start = r0 if whole else max(0, r0 - lead_in)
        plan = self.rap_plan(f, cfg, first=start, pitch_up=pitch_up, fx=dict(formant_shift=formant_shift), level=0.5)
        a, meta = self._sung_cached("gangrap" if (whole and r0 == 0) else "punch", plan, voice, dict(first=start),
                                    vib_depth=0.0, seed=int(seed), detune_cents=float(detune), neural_shift=0)
        k = r0 - start
        gate_s = None
        if not whole:
            T = plan["sylls"][k]["notes"][0]["T"] - plan["R0"]
            g0 = max(float(meta["tc"][k]) - PUNCH_FADE_S, T - PUNCH_PRE_MAX_S)
            i0, n = int(round(g0 * KSR)), int(round(PUNCH_FADE_S * KSR))
            a = np.array(a, np.float64)
            a[:max(0, i0)] = 0.0
            seg = np.arange(max(0, i0), min(len(a), i0 + n))
            a[seg] *= 0.5 - 0.5 * np.cos(np.pi * (seg - i0 + 0.5) / n)
            gate_s = plan["R0"] + g0
            # Behind the gate the rhyme syllable's own consonant now rises out of silence. If it is voiced
            # and nearly as loud as the vowel (a trilled r, a nasal), that rise - not the vowel - is the
            # attack one hears, a consonant's length before the beat: keep it PUNCH_CONTRAST_DB under the
            # vowel (onset_contrast did the same inside the phrase, against the dip of the consonant; here
            # it is the consonant's loudest moment that counts).
            Tn = plan["sylls"][k + 1]["notes"][0]["T"] - plan["R0"] if k + 1 < len(plan["sylls"]) else None
            hi = T + 0.110 if Tn is None else max(T + 0.040, min(T + 0.110, Tn - 0.040))
            pad = 0.25
            z = np.zeros(int(pad * KSR))
            x16 = vo.to16(np.concatenate([z, a, z]), KSR)
            c0, c1 = g0 + PUNCH_FADE_S + 0.004, T - 0.016
            rise = 1e9
            if c1 - c0 >= 0.008:
                for b in MR_BANDS:
                    env = band_env(x16, b)
                    pk = float(env[int((T - 0.010 + pad) * MR_FS):int((hi + pad) * MR_FS)].max())
                    shelf = float(env[int((c0 + pad) * MR_FS):int((c1 + pad) * MR_FS)].max())
                    rise = min(rise, 20 * np.log10((pk + 1e-12) / (shelf + 1e-12)))
            need = float(np.clip(PUNCH_CONTRAST_DB - rise, 0.0, PUNCH_CONTRAST_MAX_DB))
            if need >= 0.5:
                tt = np.arange(len(a)) / KSR
                a = a * 10 ** (-need * (1.0 - _smoothstep((tt - (T - 0.016)) / 0.016)) / 20.0)
            meta = dict(meta, punch_consonant_db=round(-need, 1))
        return dict(audio=a, R0=plan["R0"], plan=plan, meta=meta, rhyme_index=k, gate_s=gate_s)

    @staticmethod
    def sound_start(plan):
        """Song time (s) before which a render of this plan is silent (the gate opens 2 frames before the
        first consonant)."""
        return plan["R0"] + max(0, int(plan["lead_frames"]) - 2) * FR

    # --- hook
    def hook_plan(self, score, p, ana_voice):
        """Duration plan of a SUNG line: Spanish syllables reconciled with the score's notes.
        Every reconciliation (merge / melisma) is logged in plan['slog']."""
        text = p.get("tts_text") or p["text"]
        lang = p["lang"] + ("+full" if HOOK_FULL_VOWELS else "")
        info = self.line_syllables(text, lang, ana_voice)
        notes = [(score.t(p["bar"], b), d * score.BEAT, m) for (b, d, m, _) in p["notes"]]
        N = len(notes)
        syls, joins, nd = info["syls"], info["joins"], info["natdur"]
        use, log, cur = set(), [], rt.apply_joins(info["syls"], set())
        kind_of = dict(joins)
        for kinds in (("same", "auto"), ("opt",), ("hiatus",)):
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
        plan = self.plan_notes(text, lang, groups, ana_voice, nuclei, slog=log, nuc_share=HOOK_NUC_SHARE)
        plan["natural_syllables"] = len(syls)
        return plan

    def render_hook(self, plan, voice, seed=0, detune=0.0, octave=0, formant_shift=1.0, vib_depth=0.0):
        p2 = {k: v for k, v in plan.items() if k != "src"}
        p2["fx"] = dict(formant_shift=formant_shift)
        a, meta = self._sung_cached("hook", p2, voice, dict(), vib_depth=float(vib_depth), seed=int(seed),
                                    detune_cents=float(detune), octave=int(octave))
        return dict(audio=a, R0=p2["R0"], plan=p2, meta=meta)

    # --- speech path (ad-libs, spoken)
    def speech_once(self, text, lang, voices, speed, mode, cfg=RAPPER, seed=0, f0_scale=1.0, f0_range=1.0,
                    max_frames=None):
        """Every voice speaks the phrase with the SAME token durations (tight unison).
        mode: 'plain' (Kokoro prosody, pitch scaled by f0_scale / range by f0_range), 'rapper' (the rapper's
        deep raspy voice), 'whisper'. max_frames: trim the phoneme durations to this many 25 ms frames
        (longest sounds first, see trim_durations). Returns (list of float64 arrays at 24 kHz, speech plan)."""
        base = voices[0].split("|")[0]
        sp = self.speech_plan(text, lang, base, speed=float(speed))
        if max_frames is not None:
            sp["dur"] = trim_durations(sp["dur"], sp["chars"], int(max_frames))
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
            x = np.array(x, np.float64)
            # Kokoro lets the last sound ring on into its 0.4 s tail pad; close it right after the last
            # phoneme (SPEECH_HOLD_S, then a SPEECH_RELEASE_S raised-cosine release) so a phrase's audible
            # length is what its duration plan says (and can be fitted to a deadline)
            d = sp["dur"]
            a = (int(d[0]) + int(np.sum(d[1:-1])) - LAG) * HOP + int(SPEECH_HOLD_S * KSR)
            b = a + int(SPEECH_RELEASE_S * KSR)
            if a < len(x):
                seg = np.arange(max(0, a), min(len(x), b))
                x[seg] *= 0.5 + 0.5 * np.cos(np.pi * (seg - a) / max(1, b - a))
                x[min(len(x), b):] = 0.0
            # ... and open it with a short fade that starts on the first sounding sample (the speech path
            # zeroes its lead-in hard, which would leave a step)
            nz = np.nonzero(x)[0]
            if len(nz):
                n_in = int(SPEECH_FADE_IN_S * KSR)
                seg = np.arange(nz[0], min(len(x), nz[0] + n_in))
                x[seg] *= 0.5 - 0.5 * np.cos(np.pi * (seg - nz[0] + 0.5) / n_in)
            outs.append(x)
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


def trim_durations(dur, chars, max_frames):
    """Shorten a speech duration plan (frames per token, pads at both ends) to max_frames of speech.
    Order: pauses (punctuation, word spaces) down to one frame; then 'water-filling' of the longest sounds
    that are not a primary-stressed vowel (a drawn-out final vowel goes first), every sounding token
    keeping at least one frame; then the stressed vowels, down to two thirds of their length; only if that
    is still not enough, whatever is longest."""
    d = np.array(dur, dtype=np.int64)
    body = d[1:-1].copy()
    if body.sum() <= max_frames:
        return d
    sounding = np.array([ve.cls_of(c) in "VCL" for c in chars])
    prot = np.zeros(len(chars), bool)
    for i, c in enumerate(chars):                       # a primary-stressed vowel is the word: protect it
        if c == "ˈ":
            j = i + 1
            while j < len(chars) and ve.cls_of(chars[j]) in "VL":
                prot[j] = True
                j += 1
    for i, c in enumerate(chars):                       # pauses first
        if ve.cls_of(c) in "PW" and body[i] > 1 and body.sum() > max_frames:
            body[i] = max(1, body[i] - (body.sum() - max_frames))
    floor = np.where(prot, np.maximum(1, np.ceil(2.0 * body / 3.0)), 1).astype(np.int64)
    for sel in (sounding & ~prot, sounding & prot):
        while body.sum() > max_frames:
            cand = [i for i in np.nonzero(sel)[0] if body[i] > floor[i]]
            if not cand:
                break
            body[max(cand, key=lambda i: (body[i], i))] -= 1          # longest first; ties: the later sound
    while body.sum() > max_frames:
        z = [i for i in range(len(body)) if body[i] > (1 if sounding[i] else 0)]
        if not z:
            break
        body[max(z, key=lambda i: body[i])] -= 1
    d[1:-1] = body
    return d


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
           "8": "ocho", "9": "nueve", "10": "diez", "11": "once", "12": "doce", "13": "trece", "14": "catorce",
           "15": "quince", "20": "veinte", "100": "cien"}


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
