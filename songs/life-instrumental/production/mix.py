"""THE NEXT REPETITION: mix + master + export.

    python3 mix.py <stems_dir> <release_dir>

Reuses the mixing and mastering chain written for "What You Repeat" (auto-gain per stem, EQ, compression,
kick sidechain, a lead-keyed mid cut on the pads, reverbs and delay printed into each stem, glue
compressor, true-peak limiter, loudness matching), with this piece's own settings. Exports the piece as WAV
(24-bit), FLAC (24-bit) and MP3 (320k), and every channel as a 24-bit FLAC stem.
"""
import json
import math
import os
import sys

import numpy as np
import soundfile as sf

import shared
import score as S
import synth_dsp as D

M = shared.master_chain()                # its "import score" resolves to THIS piece's score (already imported)

SR = S.SR
M.TARGET_LUFS = -13.0                    # a dynamic piece: quieter than the two songs, so the soft scenes stay soft
M.CEILING_DB = -1.2
M.STOPS = []

# Stem settings. `active_rms` is the level the stem is brought to while it sounds (dBFS), before the rides.
SPEC = {
    "fx_time":      dict(order=1, label="fx_clock_office", active_rms=-22.0, hp=120, rev=("room", 0.10)),
    "fx_place":     dict(order=2, label="fx_bay_street_cafe", active_rms=-24.0, hp=60, rev=("hall", 0.10)),
    "fx_house":     dict(order=3, label="fx_house_phone_possum", active_rms=-22.0, hp=40, rev=("room", 0.14)),
    "kick":         dict(order=4, label="kick_low_drums", active_rms=-16.0, hp=28, eq=[("peq", 60, 1.5, 1.0), ("peq", 320, -3, 1.2)]),
    "snare":        dict(order=5, label="snare_rim_claps", active_rms=-20.0, hp=120, rev=("room", 0.14)),
    "perc":         dict(order=6, label="latin_percussion", active_rms=-22.0, hp=90, rev=("room", 0.16)),
    "hats":         dict(order=7, label="hats_shakers", active_rms=-29.0, hp=450, rev=("room", 0.08), eq=[("shelf", 9000, -2.0, True)]),
    "bass":         dict(order=8, label="bass_tuba", active_rms=-16.5, hp=28, sc=0.35, eq=[("peq", 230, -2, 1.0)]),
    "keys":         dict(order=9, label="piano_rhodes_organ", active_rms=-20.0, hp=70, sc=0.1, duck=0.8, carve=3.0, rev=("hall", 0.20),
                         eq=[("peq", 300, -2.0, 0.9)]),
    "guitar":       dict(order=10, label="nylon_guitar_harp", active_rms=-19.0, hp=70, rev=("plate", 0.18), eq=[("peq", 250, -1.5, 1.0), ("peq", 3200, 1.5, 0.8)]),
    "mallets":      dict(order=11, label="marimba_vibes_bells", active_rms=-20.0, hp=110, rev=("hall", 0.22), delay=(0.75, 0.2, 0.07)),
    "leads":        dict(order=12, label="trumpet_flute", active_rms=-18.0, hp=160, comp=(-22, 2.5, 8, 120), rev=("hall", 0.24),
                         delay=(0.75, 0.18, 0.05), eq=[("peq", 2800, 1.5, 0.8), ("shelf", 9000, -1.5, True)]),
    "accordion":    dict(order=13, label="accordion", active_rms=-22.5, hp=170, duck=1.0, carve=4.0, rev=("room", 0.16)),
    "pads":         dict(order=14, label="strings_choir_pads", active_rms=-22.0, hp=110, sc=0.12, duck=1.5, carve=5.0, rev=("hall", 0.30),
                         eq=[("peq", 350, -2.0, 0.9)]),
}
# The melody carriers: while they sound, the pads, accordion and keys dip a little in the 0.7-4.5 kHz band.
KEY_STEMS = (("leads", 1.0), ("mallets", 0.7), ("guitar", 0.6))


def main(stems_dir, rel):
    os.makedirs(os.path.join(rel, "stems"), exist_ok=True)
    kick_times = json.load(open(os.path.join(stems_dir, "sidechain.json")))["kick_times"]
    rides = json.load(open(os.path.join(stems_dir, "rides.json"))) if os.path.exists(os.path.join(stems_dir, "rides.json")) else {}
    M.IRS = {k: D.reverb_ir(seed=i, **v) for i, (k, v) in enumerate(M.REVERBS.items())}
    n = int(round(S.SONG_SECONDS * SR))

    raw = {}
    for name in SPEC:
        p = os.path.join(stems_dir, name + ".wav")
        if not os.path.exists(p):
            print(f"   (missing {name})")
            continue
        x, sr = sf.read(p, dtype="float32", always_2d=True)
        assert sr == SR, (p, sr)
        x = x[:n] if len(x) >= n else np.pad(x, ((0, n - len(x)), (0, 0)))
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

    stems = {}
    for name, spec in sorted(SPEC.items(), key=lambda kv: kv[1]["order"]):
        if name not in raw:
            continue
        spec = dict(spec)
        if rides.get(name):
            spec["ride"] = rides[name]
        r = M.process_stem(name, raw[name], spec, kick_times, lead_env)
        if r is None:
            continue
        stems[name] = r[0]
        print(f"   {name:10s} -> {spec['label']}")

    mix = sum(stems.values())
    if hasattr(S, "mix_automation"):                      # whole-mix gestures written in the score (dips, cuts)
        mix = S.mix_automation(mix)
    tmp = os.path.join(rel, "_tmp.wav")
    y, gain, (I, tp, lra) = M.loudness_matched_master(mix, tmp)
    os.remove(tmp)
    tags = {"title": S.TITLE, "genre": "Instrumental", "date": "2026",
            "comment": "Instrumental. Synthesized and produced with open-source tools."}
    M.export(y, os.path.join(rel, S.FILE_BASE), tags)
    report = {"master": dict(lufs=I, true_peak=tp, lra=lra, master_gain_db=gain, **M.LAST_STATS)}
    stem_gain = 10 ** (gain / 20)
    pk = max(float(np.max(np.abs(v))) for v in stems.values()) * stem_gain
    if pk > 0.98:
        stem_gain *= 0.98 / pk
    for k, v in stems.items():
        sp = os.path.join(rel, "stems", f"{SPEC[k]['order']:02d}_{SPEC[k]['label']}.flac")
        sf.write(sp, np.clip(v * stem_gain, -1, 1), SR, subtype="PCM_24", format="FLAC")
    report["stems_gain_db"] = 20 * math.log10(stem_gain)
    report["stems_note"] = ("stems are pre-master (before the whole-mix gestures, the glue compressor and the limiter); "
                            "they sum to the mix at a common gain of stems_gain_db")
    json.dump(report, open(os.path.join(rel, "master_report.json"), "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
