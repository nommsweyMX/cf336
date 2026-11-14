# "What You Repeat": the finished track

A complete local render of the song, with no Suno. Everything you hear was synthesized by the Python code in [`../production/`](../production/): the drums, heartbeat, beatbox, bass, keys, pads, synths, piano and FX, every vocal, the mix and the master.

**4:42 · 124 BPM · F minor, rising to G minor for the final chorus · 44.1 kHz**

## Files

| File | What it is |
|---|---|
| `What_You_Repeat.wav` | Full song, mastered. 24-bit WAV, the master copy |
| `What_You_Repeat.flac` | The same, lossless FLAC (24-bit, about half the size) |
| `What_You_Repeat.mp3` | The same, MP3 320 kbps, for your phone |
| `What_You_Repeat_Instrumental.{wav,flac,mp3}` | The same song with no vocals, mastered on its own. Use it to sing over or to record your own voice |
| `stems/01_heartbeat.flac` … `stems/15_vocal_whisper.flac` | **The 15 channels**, one FLAC per channel (24-bit stereo, full length, all starting at 0:00) |
| `master_report.json` | Measured loudness, true peak and limiter figures for both masters |

### The 15 channels

| # | Channel | What's on it |
|---|---|---|
| 01 | `heartbeat` | The lub-dub heart: solo at 62 BPM in the intro, doubling into the groove, alone in the bridge, racing before the last chorus, slowing to a stop at the end |
| 02 | `kick` | Four-on-the-floor house kick |
| 03 | `beatbox` | Mouth percussion: lip-pop kicks, "ts" hats, "pf" and inward "k" snares, rolls |
| 04 | `claps_shaker_hats` | Claps, shaker, closed and open hats |
| 05 | `bass` | Rolling sub bass, sidechained to the kick |
| 06 | `keys_rhodes_organ` | Rhodes chords and house organ stabs |
| 07 | `pads_strings` | Supersaw pads, strings and disco string stabs |
| 08 | `arp_acid` | Plucked arpeggio and the 303-style acid line |
| 09 | `piano` | Bridge piano |
| 10 | `fx` | Risers, crashes, sub booms, downlifters, reverse crashes |
| 11 | `vocal_lead` | Lead vocal (sung lines, with a double-track in the choruses) |
| 12 | `vocal_harmony` | Harmony a third above the lead on the hook lines |
| 13 | `vocal_crowd_echoes` | The crowd answering the call-and-response Spanish ladder, "otra vez", "repeat" |
| 14 | `vocal_spoken_chant` | Deadpan spoken verses, chanted verse 3, the locals' asides |
| 15 | `vocal_whisper` | The whispered breakdown and outro |

**Using the stems:** drop all 15 into any DAW at 0:00, all faders at 0 dB. They add up to the full mix *before* the master bus.
- Each stem already has its own EQ, compression, sidechain pumping, reverb and delay printed in. The busy instruments also carry the dips that make room for the vocal.
- All 15 were turned down by the same 4.9 dB so none clips on its own. Their sum runs over 0 dBFS, which is fine on a DAW's floating-point mix bus.
- To get back to the finished master, put a gentle bus compressor and a limiter on the master and bring it up to about −10.5 LUFS.
- To put your own voice on the song, mute `11_vocal_lead` and record yourself in its place. The instrumental master has none of the vocal dips, so it is the better backing to sing over.

## Song map

| Time | Section | What happens |
|---|---|---|
| 0:00 | Intro | A breath, a solo heartbeat at 62 BPM; the beatbox creeps in and the heart doubles into the 124 BPM groove |
| 0:16 | Verse 1 | Spoken, deadpan, over a filtered groove |
| 0:47 | Pre-chorus | Sung; the filter opens and the strings swell |
| 1:02 | **Chorus 1** | Full drop; the Spanish small-talk ladder as call-and-response; the hook "You are not what you say, you are what you repeat" |
| 1:18 | Post-chorus | "¡Qué bueno!", then an acid-line instrumental drop |
| 1:33 | Verse 2 | Spoken |
| 2:04 | Pre-chorus 2 | Bigger build, ending in total silence for a beat and a half |
| 2:20 | **Chorus 2** | Full drop |
| 2:35 | Post-chorus 2 | |
| 2:43 | Verse 3 | Chanted, over beatbox and acid only |
| 2:59 | Bridge | Half-time piano and pads; the heartbeat is the only percussion |
| 3:30 | Breakdown | Whispered over the heartbeat and a held pad |
| 3:45 | Build | The heartbeat races, riser, snare roll, then silence |
| 3:53 | **Final chorus** | Key change up to G minor, the biggest drop, disco strings |
| 4:08 | Tag | "Otra vez" |
| 4:16 | Outro | Layers peel away; whispers; the heart halves back to 62 BPM, slows, stops, exhale |

## How it was made

- **Score:** [`production/score.py`](../production/score.py) holds the whole song: tempo, sections, chords, every sung note with its syllable, every spoken line with its beat and every beatbox cue. Everything else reads from it.
- **Instrumental:** [`production/instrumental.py`](../production/instrumental.py) synthesizes every sound from oscillators, filters and noise (numpy, scipy, numba). It uses no samples or loops from anywhere.
- **Vocals:** [`production/vocals.py`](../production/vocals.py) turns speech into singing:
  1. The open-source **Kokoro-82M** text-to-speech model (Apache-2.0) is modified so its phoneme timing, pitch and energy can be set from outside.
  2. The **WORLD** vocoder then retunes each vowel to its exact note, with scoops, glides and vibrato, and places it on the beat.
  3. A speech-recognition model (Whisper-small, run locally) checks that the words are understandable.

  Different Kokoro voices play the lead, the harmony, the crowd and the "locals".
- **Mix and master:** [`production/mix.py`](../production/mix.py) handles:
  - per-channel level, EQ and compression;
  - kick sidechain pumping;
  - a vocal-keyed cut in the 0.7–4.5 kHz band of the keys, pads, synths and piano, so the words come through;
  - fader rides on the lead per section;
  - plate, hall and room reverbs and a ping-pong delay;
  - a glue compressor and a true-peak lookahead limiter;
  - matching the loudness (ITU-R BS.1770) to the targets below.

## Measured specs

| | Full song | Instrumental |
|---|---|---|
| Integrated loudness | −10.7 LUFS | −10.6 LUFS |
| True peak (WAV / FLAC) | −1.1 dBTP | −1.0 dBTP |
| True peak (MP3, after encoding) | −0.6 dBTP | −0.6 dBTP |
| Loudness range | 5.6 LU | 6.1 LU |
| Limiter gain reduction | mean 0.4 dB, max 2.6 dB | mean 0.45 dB, max 3.3 dB |
| Length | 4:42.5 | 4:42.5 |

Loudness by section, from the quiet heartbeat intro to the key-change final chorus (median short-term LUFS):

| Section | LUFS | | Section | LUFS |
|---|---|---|---|---|
| intro | −18.4 | | verse 3 | −11.2 |
| verse 1 | −11.7 | | bridge | −11.0 |
| pre-chorus 1 | −10.0 | | breakdown | −14.3 |
| chorus 1 | −9.3 | | build | −9.2 |
| post-chorus 1 | −10.2 | | **final chorus** | **−8.7** |
| verse 2 | −11.6 | | tag | −8.9 |
| pre-chorus 2 | −10.0 | | outro | −13.2 |
| chorus 2 | −9.1 | | | |
| post-chorus 2 | −9.7 | | | |

Both stops (at 2:19 and 3:52) are true digital silence. The song fades to silence after the last heartbeat and exhale.

Both masters sit at a loud club level. Streaming services turn them down to about −14 LUFS, which is normal.

**Vocal checks** (all measured on the rendered audio):
- **Timing:** every sung vowel lands within ±15 ms of its beat.
- **Pitch:** the median error is 3.3 cents.
- **Spoken lines:** none runs past its slot.
- **Words:** speech recognition run on the *finished master* recovers about 78–80% of the lyric words, about the same as on the isolated vocals. The mix doesn't bury the words; what is lost comes from the singing voice itself.

## Be aware

- **The voices are synthetic, not yours.** They are stock Kokoro voices. Nothing here imitates you; making the song sound like you needs a recording of you, so use the instrumental or the stem route above (or the Suno Voices route in [`../README.md`](../README.md)).
- **It sounds like a vocoder singer.** Speech-to-singing with a TTS model is a real technique, but it is not a trained singing voice. Expect a slightly synthetic, robotic quality, Eurovision-meets-Daft-Punk rather than a human session singer.
- **Some lines are hard to make out.** The hardest line is the hook's second half, "Said once, it's a sentence", which the recognizer still often hears as "said one's…". The spoken verses come through best. Short crowd shouts ("¡Hola!", "¿Y tú?") are also hard to catch as a stack of voices.
- **Checked by measurement, not by ear.** All checks were done with meters and analysis: loudness, true peak, timing of every sung vowel against the beat, pitch in cents, spectral balance per section, and silence where the score says silence. Nobody listened to it during production, so please listen before you share it.

## Rebuilding

```
cd songs/why-llarta/production
python3 instrumental.py <inst_dir>               # ~25 s
python3 vocals.py                                 # ~7 min; needs kokoro-onnx, pyworld, onnx, sherpa-onnx and the model files
python3 mix.py <inst_dir> <vox_dir> ../release    # ~3 min; needs ffmpeg
```

The vocal step uses a git-ignored working folder, `production/work/`; set `WHY_LLARTA_WORK` to put it somewhere else. It must contain:
- `models/kokoro-v1.0.onnx` and `models/voices-v1.0.bin`, from the kokoro-onnx GitHub releases;
- for the speech check, `models/asr/sherpa-onnx-whisper-small/`, from the sherpa-onnx GitHub releases.

The stems are written to `work/song/stems/`. Run `vocals.py --no-asr` to skip the speech check.
