"""LA LOCA TLACUAQUE DE LA CALLE CERRADA: mix + master + export.

    python3 mix.py <inst_dir> <vox_dir> <release_dir>

Uses the mixing and mastering chain written for "What You Repeat" (auto-gain to a target level per stem,
EQ, compression, kick sidechain, vocal-keyed mid cut on the busy instruments, reverbs and delay printed
into each stem, glue compressor, true-peak limiter, loudness matching) with this song's own settings.
Exports the full song and the instrumental as WAV (24-bit), FLAC (24-bit) and MP3 (320k), and every
channel as a 24-bit FLAC stem.
"""
import importlib.util
import json
import math
import os
import sys

import numpy as np
import soundfile as sf

import shared
import score as S
import synth_dsp as D

_spec = importlib.util.spec_from_file_location("wl_mix", os.path.join(shared.SHARED, "mix.py"))
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)          # its "import score" resolves to THIS song's score (already imported)

SR = S.SR
M.TARGET_LUFS = -10.0
M.CEILING_DB = -1.2
M.STOPS = []

# Fader rides per section (dB), keyed by section name prefix.
RAP_RIDE = {}
HOOK_RIDE = {}

SPEC = {
    "inst_kick":      dict(order=1, label="kick", active_rms=-14.5, hp=28, eq=[("peq", 60, 1.5, 1.0), ("peq", 320, -3, 1.2)]),
    "inst_snare":     dict(order=2, label="snare_claps", active_rms=-18.0, hp=120, rev=("room", 0.14),
                           eq=[("peq", 210, 1.5, 1.0), ("shelf", 7000, 1.5, True)]),
    "inst_hats":      dict(order=3, label="hats_guira_sleighbells", active_rms=-28.5, hp=450, duck=1.0, rev=("room", 0.08),
                           eq=[("shelf", 9000, -3.0, True)]),
    "inst_bass":      dict(order=4, label="808_tuba", active_rms=-15.0, hp=26, sc=0.45, eq=[("peq", 230, -2, 1.0)]),
    "inst_piano":     dict(order=5, label="piano", active_rms=-21.5, hp=110, sc=0.15, duck=1.5, carve=7.0, rev=("hall", 0.20),
                           eq=[("peq", 300, -2.5, 0.9)]),
    "inst_accordion": dict(order=6, label="accordion_bajo", active_rms=-23.5, hp=170, duck=2.0, carve=8.0, rev=("room", 0.16)),
    "inst_jingle":    dict(order=7, label="glockenspiel_toy_piano", active_rms=-26.5, hp=350, duck=1.5, carve=6.0, rev=("hall", 0.24),
                           delay=(0.75, 0.25, 0.10)),
    "inst_fx":        dict(order=8, label="fx", active_rms=-24.0, hp=30, rev=("hall", 0.22)),
    "vox_rap":        dict(order=9, label="vocal_rap", active_rms=-11.0, hp=85, drive=1.6, comp=(-24, 3.5, 3, 80), peak_lim=15,
                           ride=RAP_RIDE, eq=[("peq", 240, -3.0, 0.9), ("peq", 2800, 3.0, 0.8), ("shelf", 9500, -1.0, True)],
                           rev=("room", 0.10), delay=(0.5, 0.18, 0.06)),
    "vox_punch":      dict(order=10, label="vocal_punch_ins", active_rms=-18.5, hp=150, comp=(-24, 3.0, 4, 90), peak_lim=15,
                           eq=[("peq", 300, -3.0, 0.9), ("peq", 3000, 2.0, 0.8)], rev=("plate", 0.14)),
    "vox_hook":       dict(order=11, label="vocal_hook_gang", active_rms=-11.0, hp=120, comp=(-24, 3.0, 4, 90), peak_lim=15,
                           ride=HOOK_RIDE, eq=[("peq", 280, -3.0, 0.9), ("peq", 3000, 3.0, 0.8), ("shelf", 7000, 2.0, True)],
                           rev=("plate", 0.18)),
    "vox_kids":       dict(order=12, label="vocal_hook_kids", active_rms=-22.0, hp=260, comp=(-24, 3.0, 4, 90), peak_lim=15,
                           ride=HOOK_RIDE, eq=[("peq", 3500, 1.5, 0.8)], rev=("plate", 0.24)),
    "vox_adlibs":     dict(order=13, label="vocal_adlibs_crowd", active_rms=-15.0, hp=140, comp=(-24, 3.0, 4, 90), peak_lim=15,
                           eq=[("peq", 300, -2.5, 0.9), ("peq", 2800, 2.0, 0.8)], rev=("hall", 0.18), delay=(0.75, 0.2, 0.08)),
    "vox_spoken":     dict(order=14, label="vocal_spoken", active_rms=-13.5, hp=85, drive=1.4, comp=(-24, 3.0, 4, 90), peak_lim=15,
                           eq=[("peq", 240, -2.5, 0.9), ("peq", 3000, 2.5, 0.9)], rev=("room", 0.12)),
}
KEY_STEMS = (("vox_rap", 1.0), ("vox_hook", 1.0), ("vox_spoken", 1.0), ("vox_adlibs", 0.6), ("vox_kids", 0.5))


def expand_ride(ride):
    """Section-prefix rides -> the per-section-name dict the shared ride_curve expects."""
    out = {}
    for name, *_ in S.SECTIONS:
        for k, v in ride.items():
            if name.startswith(k):
                out[name] = v
    return out


def grit(x, drive):
    """Soft saturation that keeps the level: adds growl harmonics to the voice."""
    pk = np.percentile(np.abs(x[np.abs(x) > 1e-4]), 99.5) if np.any(np.abs(x) > 1e-4) else 1.0
    y = np.tanh(x / pk * drive) / np.tanh(drive) * pk
    return (0.6 * y + 0.4 * x).astype(np.float32)


def main(inst_dir, vox_dir, rel):
    os.makedirs(os.path.join(rel, "stems"), exist_ok=True)
    kick_times = json.load(open(os.path.join(inst_dir, "sidechain.json")))["kick_times"]
    M.IRS = {k: D.reverb_ir(seed=i, **v) for i, (k, v) in enumerate(M.REVERBS.items())}
    n = int(round(S.SONG_SECONDS * SR))

    raw = {}
    for name in SPEC:
        p = os.path.join(inst_dir if name.startswith("inst") else vox_dir, name + ".wav")
        if not os.path.exists(p):
            print(f"   (missing {name})")
            continue
        x, sr = sf.read(p, dtype="float32", always_2d=True)
        assert sr == SR, (p, sr)
        x = x[:n] if len(x) >= n else np.pad(x, ((0, n - len(x)), (0, 0)))
        if x.shape[1] == 1:
            x = np.repeat(x, 2, axis=1)
        if np.abs(x).max() < 1e-6:
            print(f"   (silent {name})")
            continue
        raw[name] = x

    lead_env = None
    keys = [(k, w) for k, w in KEY_STEMS if k in raw]
    if keys:
        v = sum(raw[k] * w for k, w in keys)
        e = M.env_follow(v, 120)
        la = int(0.04 * SR)
        e = np.concatenate([e[la:], np.zeros(la, e.dtype)])
        e = e / (np.percentile(e[e > 1e-4], 95) + 1e-9)
        lead_env = np.clip(e, 0, 1)

    stems, stems_inst = {}, {}
    for name, spec in sorted(SPEC.items(), key=lambda kv: kv[1]["order"]):
        if name not in raw:
            continue
        spec = dict(spec)
        if "ride" in spec:
            spec["ride"] = expand_ride(spec["ride"])
            if not spec["ride"]:
                del spec["ride"]
        x = raw[name]
        if spec.get("drive"):
            x = grit(x, spec["drive"])
        r = M.process_stem(name, x, spec, kick_times, lead_env)
        if r is None:
            continue
        stems[name], u = r
        if name.startswith("inst"):
            stems_inst[name] = u
        print(f"   {name:15s} -> {spec['label']}")

    tags = {"title": S.TITLE, "genre": "Hip-Hop / Rap", "date": "2026", "language": "spa",
            "comment": f"{S.BPM:.0f} BPM, {S.KEY}. Synthesized and produced with open-source tools."}
    base_name = "La_Loca_Tlacuaque_de_la_Calle_Cerrada"
    report = {}
    for version, src in (("full", stems), ("instrumental", stems_inst)):
        keys = list(src)
        if version == "full" and not any(k.startswith("vox") for k in stems):
            continue
        mix = sum(src[k] for k in keys)
        tmp = os.path.join(rel, f"_tmp_{version}.wav")
        y, gain, (I, tp, lra) = M.loudness_matched_master(mix, tmp)
        os.remove(tmp)
        t = dict(tags, title=S.TITLE + ("" if version == "full" else " (Instrumental)"))
        M.export(y, os.path.join(rel, base_name + ("" if version == "full" else "_Instrumental")), t)
        report[version] = dict(lufs=I, true_peak=tp, lra=lra, master_gain_db=gain, **M.LAST_STATS)
        if version == "full" or not any(k.startswith("vox") for k in stems):
            stem_gain = 10 ** (gain / 20)
            pk = max(float(np.max(np.abs(src[k]))) for k in keys) * stem_gain
            if pk > 0.98:
                stem_gain *= 0.98 / pk
            for k in keys:
                sp = os.path.join(rel, "stems", f"{SPEC[k]['order']:02d}_{SPEC[k]['label']}.flac")
                sf.write(sp, np.clip(src[k] * stem_gain, -1, 1), SR, subtype="PCM_24", format="FLAC")
            report["stems_gain_db"] = 20 * math.log10(stem_gain)
            report["stems_note"] = ("stems are pre-master (before the glue compressor and limiter); "
                                    "they sum to the mix at a common gain of stems_gain_db")
    json.dump(report, open(os.path.join(rel, "master_report.json"), "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
