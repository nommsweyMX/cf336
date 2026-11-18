"""WHAT YOU REPEAT: mix + master + export.

  python3 mix.py <inst_dir> <vox_dir> <release_dir>

- Auto-gain every stem to a target "active RMS", then per-stem EQ / compression / sidechain pump /
  vocal ducking / reverb + delay sends (printed into each stem so the stems sum to the mix).
- Master: glue compressor -> tilt EQ -> lookahead limiter, loudness-matched to TARGET_LUFS, true peak <= -1 dBTP.
- Exports: full mix and instrumental as WAV (24-bit), FLAC (24-bit), MP3 (320k); every channel as a FLAC stem.
"""
import json
import math
import os
import re
import subprocess
import sys

import numba
import numpy as np
import soundfile as sf

import score as S
import synth_dsp as D

SR = S.SR
TARGET_LUFS = -10.5
CEILING_DB = -1.2
LAST_STATS = {}
TITLE = "What You Repeat"

# Fader rides on the lead (and its harmony) per section, in dB: up where the arrangement is densest,
# down in the sparse bridge, so the words sit at a similar level against the music throughout.
LEAD_RIDE = {"pre1": -0.5, "pre2": -0.5, "post1": 1.0, "post2": 1.0, "chorus2": 0.5, "bridge": -2.0,
             "final": 2.0, "tag": 2.5}

# name -> processing. active_rms: target RMS (dBFS) over the stem's active passages.
SPEC = {
    "inst_heartbeat": dict(order=1, label="heartbeat", active_rms=-19, hp=25, rev=("room", 0.10)),
    "inst_kick":      dict(order=2, label="kick", active_rms=-15.5, hp=28, eq=[("peq", 55, 1.5, 1.0), ("peq", 320, -3, 1.2)]),
    "inst_beatbox":   dict(order=3, label="beatbox", active_rms=-22.5, hp=140, rev=("room", 0.12)),
    "inst_perc":      dict(order=4, label="claps_shaker_hats", active_rms=-27, hp=280, rev=("room", 0.10), eq=[("shelf", 9000, -2.0, True)]),
    "inst_bass":      dict(order=5, label="bass", active_rms=-17.5, hp=28, sc=0.6, eq=[("peq", 220, -2, 1.0)]),
    "inst_keys":      dict(order=6, label="keys_rhodes_organ", active_rms=-24, hp=160, sc=0.4, duck=1.5, carve=6.0, rev=("hall", 0.18), delay=(0.75, 0.3, 0.12)),
    "inst_pads":      dict(order=7, label="pads_strings", active_rms=-24.5, hp=180, sc=0.5, duck=1.5, carve=6.0, rev=("hall", 0.22)),
    "inst_synths":    dict(order=8, label="arp_acid", active_rms=-24.5, hp=150, sc=0.3, duck=1.5, carve=6.0, rev=("hall", 0.14), delay=(0.75, 0.35, 0.2)),
    "inst_piano":     dict(order=9, label="piano", active_rms=-21, hp=110, rev=("hall", 0.32), duck=1.0, carve=4.0, eq=[("peq", 300, -3.5, 0.9), ("peq", 2500, 1.5, 0.9), ("shelf", 6000, 2.0, True)]),
    "inst_fx":        dict(order=10, label="fx", active_rms=-25, hp=30, rev=("hall", 0.25)),
    "vox_lead":       dict(order=11, label="vocal_lead", peak_lim=16, active_rms=-14.0, hp=140, comp=(-24, 3.0, 4, 90), ride=LEAD_RIDE,
                           eq=[("peq", 240, -4.0, 0.8), ("peq", 3100, 3.0, 0.8), ("shelf", 9500, -1.5, True)],
                           rev=("plate", 0.17), delay=(0.5, 0.22, 0.10)),
    "vox_harmony":    dict(order=12, label="vocal_harmony", peak_lim=16, active_rms=-22.5, hp=180, comp=(-26, 3.0, 4, 90), ride=LEAD_RIDE,
                           eq=[("peq", 300, -3.0, 0.9), ("peq", 3000, 2, 0.8)], rev=("plate", 0.25)),
    "vox_crowd":      dict(order=13, label="vocal_crowd_echoes", peak_lim=16, active_rms=-15.5, hp=150, comp=(-24, 2.5, 5, 100),
                           eq=[("peq", 300, -3.0, 0.9), ("peq", 2500, 2, 0.8)], rev=("hall", 0.22)),
    "vox_spoken":     dict(order=14, label="vocal_spoken_chant", peak_lim=16, active_rms=-16.5, hp=120, comp=(-24, 3.0, 4, 90),
                           eq=[("peq", 220, -2.5, 0.9), ("peq", 3500, 2.5, 0.9)], rev=("room", 0.12), delay=(0.75, 0.18, 0.08)),
    "vox_whisper":    dict(order=15, label="vocal_whisper", peak_lim=16, active_rms=-18.5, hp=160, comp=(-28, 2.5, 5, 120),
                           eq=[("peq", 5000, 2, 0.8)], rev=("hall", 0.30)),
}
REVERBS = {
    "room": dict(seconds=0.7, decay=0.16, lo=250, hi=7000, predelay=0.008),
    "plate": dict(seconds=2.2, decay=0.5, lo=300, hi=9000, predelay=0.018),
    "hall": dict(seconds=3.2, decay=0.85, lo=250, hi=6000, predelay=0.03),
}


# ------------------------------------------------------------------ dynamics
@numba.njit(cache=True)
def _comp_gain(x, sr, thr_db, ratio, att_ms, rel_ms, knee):
    n = x.shape[0]
    g = np.ones(n)
    aa = math.exp(-1.0 / (att_ms * 1e-3 * sr))
    ar = math.exp(-1.0 / (rel_ms * 1e-3 * sr))
    env = -120.0
    for i in range(n):
        lvl = max(abs(x[i, 0]), abs(x[i, 1]))
        ldb = 20.0 * math.log10(lvl + 1e-9)
        if ldb > env:
            env = aa * env + (1 - aa) * ldb
        else:
            env = ar * env + (1 - ar) * ldb
        over = env - thr_db
        if over <= -knee / 2:
            gr = 0.0
        elif over >= knee / 2:
            gr = over * (1 - 1 / ratio)
        else:
            gr = (1 - 1 / ratio) * (over + knee / 2) ** 2 / (2 * knee)
        g[i] = 10.0 ** (-gr / 20.0)
    return g


@numba.njit(cache=True)
def _limiter_gain(peak, sr, ceil, look_ms, rel_ms):
    """peak: per-sample (true-)peak magnitude. Returns a gain curve that guarantees peak*gain <= ceil.
    forward sliding-min over L, then a causal box average over L (never exceeds the requirement at a peak),
    then a one-pole release that can only rise as fast as rel_ms allows."""
    n = peak.shape[0]
    raw = np.ones(n)
    for i in range(n):
        if peak[i] > ceil:
            raw[i] = ceil / peak[i]
    L = max(int(look_ms * 1e-3 * sr), 1)
    fm = np.ones(n)
    dq = np.empty(n, dtype=np.int64)
    head = 0
    tail = 0
    j = 0
    for i in range(n):
        while j < n and j <= i + L:
            while tail > head and raw[dq[tail - 1]] >= raw[j]:
                tail -= 1
            dq[tail] = j
            tail += 1
            j += 1
        while dq[head] < i:
            head += 1
        fm[i] = raw[dq[head]]
    box = np.ones(n)
    acc = 0.0
    for i in range(n):
        acc += fm[i]
        if i >= L:
            acc -= fm[i - L]
            box[i] = acc / L
        else:
            box[i] = (acc + (L - 1 - i)) / L
    g = np.ones(n)
    ar = math.exp(-1.0 / (rel_ms * 1e-3 * sr))
    cur = 1.0
    for i in range(n):
        up = cur + (1.0 - cur) * (1.0 - ar)
        cur = box[i] if box[i] < up else up
        g[i] = cur
    return g


def true_peak(x, os_factor=4):
    """Per-sample true-peak estimate (max |x| over the 4x-oversampled neighbourhood)."""
    from scipy.signal import resample_poly
    n = x.shape[0]
    pk = np.zeros(n)
    for c in range(x.shape[1]):
        u = np.abs(resample_poly(x[:, c], os_factor, 1))[: n * os_factor].reshape(n, os_factor).max(axis=1)
        pk = np.maximum(pk, u)
    pk = np.maximum(pk, np.abs(x).max(axis=1))
    pk[1:] = np.maximum(pk[1:], pk[:-1])  # cover the inter-sample span on both sides
    return pk


def soft_clip(x, thr):
    """Transparent below thr, smooth tanh knee above (ceiling ~ thr + (1-thr))."""
    a = np.abs(x)
    over = a > thr
    y = x.copy()
    y[over] = np.sign(x[over]) * (thr + (1 - thr) * np.tanh((a[over] - thr) / (1 - thr)))
    return y


def active_rms_db(x):
    m = np.abs(x).mean(axis=1) if x.ndim == 2 else np.abs(x)
    blk = int(0.05 * SR)
    nb = len(m) // blk
    e = (x[: nb * blk] ** 2).reshape(nb, blk, -1).mean(axis=(1, 2))
    edb = 10 * np.log10(e + 1e-12)
    if edb.max() < -90:
        return None
    act = edb > edb.max() - 30
    return 10 * np.log10(e[act].mean())


def pump_curve(kick_times, n, depth, release=0.11):
    imp = np.zeros(n)
    for tk in kick_times:
        i = int(tk * SR)
        if 0 <= i < n:
            imp[i] = 1.0
    k = int(0.45 * SR)
    tt = np.arange(k) / SR
    kern = np.minimum(1, tt / 0.003) * np.exp(-tt / release)
    from scipy.signal import fftconvolve
    p = np.minimum(1.0, fftconvolve(imp, kern)[:n])
    return (1 - depth * p).astype(np.float32)


def env_follow(x, ms=60):
    m = np.abs(x).max(axis=1)
    return D.lp(m, 1000 / ms / 2 * 1.0)


def ride_curve(ride, n, ramp_s=0.25):
    """Per-section gain (dB) as a curve, with short ramps that end at each section's downbeat."""
    g_db = np.zeros(n, dtype=np.float64)
    for name, s, nb, _, _ in S.SECTIONS:
        a, z = int(S.t(s) * SR), min(n, int(S.t(s + nb) * SR))
        g_db[a:z] = ride.get(name, 0.0)
    k = int(ramp_s * SR)
    g_db = np.convolve(np.pad(g_db, (0, k), mode="edge"), np.ones(k) / k, mode="full")[k:k + n]
    return (10 ** (g_db / 20)).astype(np.float32)


def stop_gate(n, tail_s):
    g = np.ones(n, dtype=np.float32)
    for a, z in STOPS:
        ia, iz = int(a * SR), int(z * SR)
        tl = int(tail_s * SR)
        g[ia:ia + tl] = np.minimum(g[ia:ia + tl], np.linspace(1, 0, tl) ** 2)
        g[ia + tl:iz] = 0
        f = int(0.004 * SR)
        g[iz - f:iz] = np.linspace(0, 1, f)
        g[iz:iz] = 1
    return g


def lufs(path):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-af", "ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True).stderr
    summ = r[r.rfind("Summary:"):]
    I = float(re.search(r"I:\s+(-?[\d.]+) LUFS", summ).group(1))
    tp = float(re.search(r"Peak:\s+(-?[\d.]+|-inf) dBFS", summ).group(1))
    lra = float(re.search(r"LRA:\s+(-?[\d.]+) LU", summ).group(1))
    return I, tp, lra


# ------------------------------------------------------------------ main
def process_stem(name, x, spec, kick_times, lead_env):
    n = len(x)
    ar = active_rms_db(x)
    if ar is None:
        return None
    x = x * 10 ** ((spec["active_rms"] - ar) / 20)
    x = D.hp(x, spec["hp"]).astype(np.float32)
    for kind, f, g, q in spec.get("eq", []):
        x = (D.peq(x, f, g, q) if kind == "peq" else D.shelf(x, f, g, q)).astype(np.float32)
    if "comp" in spec:
        thr, ratio, att, rel = spec["comp"]
        g = _comp_gain(x.astype(np.float64), SR, float(thr), float(ratio), float(att), float(rel), 6.0)
        x = (x * g[:, None]).astype(np.float32)
        x *= 10 ** ((spec["active_rms"] - active_rms_db(x)) / 20)  # makeup back to target
    if spec.get("peak_lim"):
        # fast peak limiter for consonant bursts (plosives sit 20+ dB over the vowels in TTS speech)
        ceil = 10 ** ((spec["active_rms"] + spec["peak_lim"]) / 20)
        g = _limiter_gain(np.abs(x).max(axis=1).astype(np.float64), SR, float(ceil), 1.5, 40.0)
        x = (x * g[:, None]).astype(np.float32)
    if spec.get("ride"):
        x = (x * ride_curve(spec["ride"], n)[:, None]).astype(np.float32)
    wet = np.zeros_like(x)
    if "delay" in spec:
        beats, fb, mixl = spec["delay"]
        wet += D.pingpong(x, beats * S.BEAT, fb) * mixl
    if "rev" in spec:
        kind, send = spec["rev"]
        wet += D.convolve_stereo(x, IRS[kind]) * send
    y = x + wet
    if spec.get("sc"):
        y *= pump_curve(kick_times, n, spec["sc"])[:, None]
    y *= stop_gate(n, 0.25 if name.startswith("vox") else 0.06)[:, None]
    y = y.astype(np.float32)
    if lead_env is None or not (spec.get("duck") or spec.get("carve")):
        return y, y
    # Make room for the vocal in the full mix only: a small broadband duck plus a dynamic cut of the
    # 0.7-4.5 kHz band (where the words live), both driven by the vocal envelope. The instrumental
    # master uses the un-ducked version.
    out = y.copy()
    if spec.get("carve"):
        band = D.bp(y, 700, 4500, order=2).astype(np.float32)
        gc = (10 ** (-spec["carve"] * lead_env / 20)).astype(np.float32)
        out -= ((1 - gc)[:, None] * band)
    if spec.get("duck"):
        out *= (10 ** (-spec["duck"] * lead_env / 20))[:, None].astype(np.float32)
    return out, y


def master(mix, gain_db, stats=None):
    x = (mix * 10 ** (gain_db / 20)).astype(np.float64)
    x = D.hp(x, 22)
    g = _comp_gain(x, SR, -12.0, 1.8, 15.0, 160.0, 8.0)
    x = x * g[:, None]
    x = D.shelf(x, 10000, 0.0, True)
    x = D.shelf(x, 60, -0.8, False)
    ceil = 10 ** (CEILING_DB / 20)
    pre = np.abs(x).max()
    x = soft_clip(x / (ceil * 1.26), 0.7) * (ceil * 1.26)   # shaves the top ~2 dB of transients smoothly
    lg = _limiter_gain(true_peak(x), SR, ceil, 3.0, 80.0)
    x = x * lg[:, None]
    x = np.clip(x, -ceil, ceil)
    if stats is not None:
        grdb = -20 * np.log10(lg)
        stats.update(glue_gr_max=float(-20 * np.log10(g.min())), lim_gr_max=float(grdb.max()),
                     lim_gr_mean_db=float(grdb.mean()), lim_gr_p95=float(np.percentile(grdb, 95)),
                     pre_peak_db=float(20 * np.log10(pre)))
    f = int(0.01 * SR)
    x[:f] *= np.linspace(0, 1, f)[:, None]
    f = int(1.5 * SR)
    x[-f:] *= np.linspace(1, 0, f)[:, None] ** 2
    return x


def export(x, base, tags):
    wav, flac, mp3 = base + ".wav", base + ".flac", base + ".mp3"
    sf.write(wav, x, SR, subtype="PCM_24")
    meta = sum((["-metadata", f"{k}={v}"] for k, v in tags.items()), [])
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-c:a", "flac", "-compression_level", "8", *meta, flac], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-codec:a", "libmp3lame", "-b:a", "320k", "-id3v2_version", "3", *meta, mp3], check=True)
    return wav, flac, mp3


def loudness_matched_master(mix, tmp_path):
    gain = 0.0
    for it in range(5):
        st = {}
        y = master(mix, gain, st)
        sf.write(tmp_path, y, SR, subtype="PCM_24")
        I, tp, lra = lufs(tmp_path)
        print(f"   master pass {it}: gain {gain:+.2f} dB -> {I:.2f} LUFS, TP {tp:.2f} dBTP, LRA {lra:.1f} | "
              f"limiter GR max {st['lim_gr_max']:.1f} dB, p95 {st['lim_gr_p95']:.1f}, mean {st['lim_gr_mean_db']:.2f}; glue GR max {st['glue_gr_max']:.1f}")
        LAST_STATS.update(st)
        if abs(I - TARGET_LUFS) < 0.25 and tp <= -1.0:
            break
        gain += (TARGET_LUFS - I) * 1.8
    return y, gain, (I, tp, lra)


if __name__ == "__main__":
    inst_dir, vox_dir, rel = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(os.path.join(rel, "stems"), exist_ok=True)
    sc = json.load(open(os.path.join(inst_dir, "sidechain.json")))
    kick_times, STOPS = sc["kick_times"], sc["stops"]
    IRS = {k: D.reverb_ir(seed=i, **v) for i, (k, v) in enumerate(REVERBS.items())}
    n = int(round(S.SONG_SECONDS * SR))

    raw = {}
    for name in SPEC:
        d = inst_dir if name.startswith("inst") else vox_dir
        p = os.path.join(d, name + ".wav")
        if os.path.exists(p):
            x, sr = sf.read(p, dtype="float32", always_2d=True)
            assert sr == SR, (p, sr)
            x = x[:n] if len(x) >= n else np.pad(x, ((0, n - len(x)), (0, 0)))
            if x.shape[1] == 1:
                x = np.repeat(x, 2, axis=1)
            raw[name] = x
        else:
            print(f"   (missing {name})")

    lead_env = None
    if "vox_lead" in raw or "vox_spoken" in raw:
        v = sum(raw[k] * w for k, w in (("vox_lead", 1.0), ("vox_spoken", 1.0), ("vox_crowd", 0.7)) if k in raw)
        e = env_follow(v, 120)
        la = int(0.04 * SR)  # 40 ms look-ahead so the cut is in place before each word
        e = np.concatenate([e[la:], np.zeros(la, e.dtype)])
        e = e / (np.percentile(e[e > 1e-4], 95) + 1e-9)
        lead_env = np.clip(e, 0, 1)

    stems, stems_inst = {}, {}
    for name, spec in sorted(SPEC.items(), key=lambda kv: kv[1]["order"]):
        if name in raw:
            r = process_stem(name, raw[name], spec, kick_times, lead_env)
            if r is not None:
                stems[name], u = r
                if name.startswith("inst"):
                    stems_inst[name] = u
                print(f"   {name:15s} -> {spec['label']}")

    tags = {"title": TITLE, "album": "WHY-LLARTA", "genre": "Dance / Deep House", "date": "2026",
            "comment": "124 BPM, F minor -> G minor. Synthesized and produced with open-source tools."}
    report = {}
    for version, src in (("full", stems), ("instrumental", stems_inst)):
        keys = list(src)
        if version == "full" and not any(k.startswith("vox") for k in stems):
            continue
        mix = sum(src[k] for k in keys)
        tmp = os.path.join(rel, f"_tmp_{version}.wav")
        y, gain, (I, tp, lra) = loudness_matched_master(mix, tmp)
        os.remove(tmp)
        base = os.path.join(rel, TITLE.replace(" ", "_") + ("" if version == "full" else "_Instrumental"))
        t = dict(tags, title=TITLE + ("" if version == "full" else " (Instrumental)"))
        export(y, base, t)
        report[version] = dict(lufs=I, true_peak=tp, lra=lra, master_gain_db=gain, **LAST_STATS)
        if version == "full" or not any(k.startswith("vox") for k in stems):
            stem_gain = 10 ** (gain / 20)
            pk = max(float(np.max(np.abs(src[k]))) for k in keys) * stem_gain
            if pk > 0.98:  # keep every stem un-clipped; one common scale so they still sum to the mix
                stem_gain *= 0.98 / pk
            for k in keys:
                spec = SPEC[k]
                sp = os.path.join(rel, "stems", f"{spec['order']:02d}_{spec['label']}.flac")
                sf.write(sp, np.clip(src[k] * stem_gain, -1, 1), SR, subtype="PCM_24", format="FLAC")
            report["stems_gain_db"] = 20 * math.log10(stem_gain)
            report["stems_note"] = "stems are pre-master (before glue comp + limiter); they sum to the mix at a common gain of stems_gain_db"
    json.dump(report, open(os.path.join(rel, "master_report.json"), "w"), indent=1)
    print(json.dumps(report, indent=1))
