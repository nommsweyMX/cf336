# "La Loca Tlacuaque de la Calle Cerrada": the finished track

A complete local render of the song, with no Suno. Everything you hear was synthesized by the Python code in [`../production/`](../production/): the drums, 808, tuba, piano, accordion, bells and sound effects, every voice, the mix and the master.

**2:54 · 90 BPM · D minor · 44.1 kHz · explicit Mexican street slang**

## Files

| File | What it is |
|---|---|
| `La_Loca_Tlacuaque_de_la_Calle_Cerrada.wav` | Full song, mastered. 24-bit WAV, the master copy |
| `La_Loca_Tlacuaque_de_la_Calle_Cerrada.flac` | The same, lossless FLAC (24-bit) |
| `La_Loca_Tlacuaque_de_la_Calle_Cerrada.mp3` | The same, MP3 320 kbps, for your phone |
| `La_Loca_Tlacuaque_de_la_Calle_Cerrada_Instrumental.{wav,flac,mp3}` | The beat with no voices, mastered on its own. Use it to rap over |
| `stems/01_kick.flac` … `stems/14_vocal_spoken.flac` | **The 14 channels**, one FLAC per channel (24-bit stereo, full length, all starting at 0:00) |
| `master_report.json` | Measured loudness, true peak and limiter figures for both masters |

### The 14 channels

| # | Channel | What's on it |
|---|---|---|
| 01 | `kick` | Boom-bap kick in the verses, a kick on every beat in the hooks, a half-time stomp in the bridge |
| 02 | `snare_claps` | Snare on 2 and 4, ghost notes, claps in the hooks, the roll into verse 3 |
| 03 | `hats_guira_sleighbells` | Swung hi-hats, güira and sleigh bells (the bells only in the hooks) |
| 04 | `808_tuba` | Sliding 808 sub bass, plus a tuba on the beats in the hooks |
| 05 | `piano` | Dark piano arpeggio in the verses, bass notes in the intro and outro |
| 06 | `accordion_bajo` | Accordion stabs and licks, and the plucked bajo |
| 07 | `glockenspiel_toy_piano` | The jingle: glockenspiel and toy piano playing the hook melody with the voices |
| 08 | `fx` | Vinyl crackle and the story: two glass apples breaking, the pool splash, the door and lock, the possum's hiss, the patrol siren |
| 09 | `vocal_rap` | The rapper: every verse line and the "¿Quién se…?" questions |
| 10 | `vocal_punch_ins` | Two gang voices doubling the rhyme at the end of each verse line |
| 11 | `vocal_hook_gang` | The gang singing "¡Culón, colito!", the answers and the bridge chant |
| 12 | `vocal_hook_kids` | Kids' voices singing the hook an octave up, kept low |
| 13 | `vocal_adlibs_crowd` | Side shouts ("¡Ahí nos vidrios!", "¡No mames!", "¡Fuchi!"…) and the whispers |
| 14 | `vocal_spoken` | The spoken lines that open and close the song |

**Using the stems:** drop all 14 into any DAW at 0:00, all faders at 0 dB. They add up to the full mix *before* the master bus.
- Each stem already has its own EQ, compression, sidechain pumping, reverb and delay printed in. The piano, accordion, bells and hats also carry the dips that make room for the voices.
- All 14 were turned down by the same 4.8 dB so none clips on its own. Their sum runs over 0 dBFS, which is fine on a DAW's floating-point mix bus.
- To get back to the finished master, put a gentle bus compressor and a limiter on the master and bring it up to about −10 LUFS.
- To put your own voice on it, mute `09_vocal_rap` and record yourself in its place, or rap over the instrumental. The instrumental master has none of the vocal dips.

## Song map

| Time | Section | What happens |
|---|---|---|
| 0:00 | Intro | Whispered "culón, colito…", toy piano, "¿Oíste eso, carnal?", the gang: "¿En la viga? ¡A la verga!" |
| 0:11 | Verse 1 | In through the roof beam; two glass apples break; two inches from the toilet; the sink |
| 0:32 | **Hook 1** | "¡Culón, colito!" jingle, then question and answer: "¿Quién se metió?" "¡La tlacuaque!" |
| 0:54 | Verse 2 | The pool splash, back in at the last second, the door slams and locks, under the sofa, hiss |
| 1:15 | **Hook 2** | Jingle; "¿Quién se peló? ¿Quién se regresó? ¿Quién se escondió?" |
| 1:36 | Bridge | Siren, half-time stomp: "¡Verde, verde, la patrulla verde!", four Cocas and one pollo |
| 1:58 | Verse 3 | The capture; they leave as they came, plus one loca tlacuaque |
| 2:19 | **Hook 3** | The count: "¿Y quién se la llevó?" "¡La patrulla verde!" "¿Con cuántas Cocas?" "¡Cuatro!" |
| 2:40 | Outro | "Y que alguien tape el pinche hoyo de la viga, carnal." Whisper, one last hiss |

The lyric is the one in [`../SUNO_PASTE.md`](../SUNO_PASTE.md), with two differences in this render:
- The rapper says "se meó" where the sheet reads "se mió", so the word comes out as two syllables.
- The side shout "¡Menéale!" is only in the last hook, so the first two hooks state "¡Culón, colito!" with nothing on top.

## How it was made

- **Score:** [`production/score.py`](../production/score.py) holds the whole song: tempo and swing, sections, chords, every rap line with the beat its rhyme lands on, every sung note with its syllable, every shout and sound effect. Everything else reads from it.
- **Beat:** [`production/sounds.py`](../production/sounds.py) and [`production/beat.py`](../production/beat.py) synthesize every sound from oscillators, filters and noise. No samples or loops from anywhere.
- **Voices:** [`production/vocals.py`](../production/vocals.py), with [`rap_engine.py`](../production/rap_engine.py) and [`rap_text.py`](../production/rap_text.py):
  1. The open-source **Kokoro-82M** text-to-speech model (Apache-2.0) speaks the Spanish, with its phoneme timing set from outside and the Castilian sounds swapped for Mexican ones.
  2. Each rap syllable is placed on the swung sixteenth grid, with the line's rhyme on the snare.
  3. The **WORLD** vocoder drops the voice to about 98 Hz and adds the rasp; for the hook it tunes each vowel to its note.
  4. A speech-recognition model (Whisper-small, run locally) checks each line and, where one comes out unclear, picks the clearer of a few takes.
- **Mix and master:** [`production/mix.py`](../production/mix.py) reuses the chain built for the other song in this repository:
  - per-channel level, EQ and compression;
  - kick sidechain on the bass;
  - a voice-keyed cut in the 0.7–4.5 kHz band of the piano, accordion and bells, so the words come through;
  - room, plate and hall reverbs and a ping-pong delay;
  - a glue compressor and a true-peak lookahead limiter;
  - matching the loudness (ITU-R BS.1770) to the targets below.

## Measured specs

| | Full song | Instrumental |
|---|---|---|
| Integrated loudness | −10.0 LUFS | −10.1 LUFS |
| True peak (WAV / FLAC) | −1.1 dBTP | −1.0 dBTP |
| True peak (MP3, after encoding) | −0.8 dBTP | −0.4 dBTP |
| Loudness range | 2.1 LU | 7.2 LU |
| Limiter gain reduction | mean 0.6 dB, max 3.0 dB | mean 0.4 dB, max 4.6 dB |
| Length | 2:54.0 | 2:54.0 |

Loudness by section (median short-term LUFS):

| Intro | Verse 1 | Hook 1 | Verse 2 | Hook 2 | Bridge | Verse 3 | Hook 3 | Outro |
|---|---|---|---|---|---|---|---|---|
| −13.4 | −9.6 | −9.8 | −9.6 | −9.8 | −9.9 | −9.7 | −9.7 | −14.2 |

The WAV and the FLAC are sample-identical. Both masters sit at a loud level; streaming services turn them down to about −14 LUFS, which is normal.

**Balance:** while it sounds, the rapper sits 2.4–3.1 dB above the beat and the hook gang 2.8 dB above it. The kids' layer is about 10.5 dB under the gang.

**Voice checks** (all measured on the rendered audio):
- **Rap timing:** of 451 syllables, 96.6–97.8% start within ±20 ms of their grid slot, depending on the detector; the mean error is about 5 ms.
- **Rhymes:** the rhyme of all 37 verse and question lines lands on its beat: 36 within ±15 ms, one 16 ms late.
- **Rapper's voice:** median pitch 98.6 Hz. The rasp shows in the numbers (harmonics-to-noise ratio 3.3 dB, against 8.5 dB for the same lines rendered clean).
- **Hook:** median pitch error 3.5 cents for the gang and 2.3 for the kids; 95–97% of sung vowels start within ±20 ms of the beat.
- **Words in the verses:** speech recognition on the *finished master* recovers 79% of the verse words, against 81% on the voices alone. The beat does not bury the rap.
- **Words in the hook:** on the gang vocal alone the recognizer gets 83–90% of the words. On the finished master it mostly labels the hook as music and writes nothing. Where it does write (the opening lines of hook 2), it gets 67–78%.

## Be aware

- **The voices are synthetic.** They are stock Kokoro voices, lowered and roughened. They do not imitate you or any real artist.
- **It sounds like a machine rapping.** The timing is tight and the voice is deep and rough, but it is text-to-speech pushed onto a beat. Expect a robotic edge, not a human MC with breath and swagger.
- **The hook is the part I could least verify.** The checker hears the words when the gang sings alone, but with the bells and accordion on top it usually refuses to transcribe. Listen to the hook first.
- **Some lines are hard to make out.** The weakest are:
  - "y se mió en el lavabo la cochina porquería";
  - "tenía el trono enfrente y le atinó al azulejo";
  - "De verde hasta los guantes, porque la loca muerde" (the most crowded line);
  - "¿Quién se mió?".

  The recognizer does not know the word "tlacuaque" and writes "tracuaque" or "la coja".
- **Short shouts are a coin toss.** "¡Fuchi!", "¡No mames!" and "¡La verde!" come through; "¡Jálale!", "¡Ja!" and the whispers do not register as words.
- **Checked by measurement, not by ear.** Loudness, true peak, timing of every syllable, pitch, band balance per section and the word checks were all done with meters and analysis. Nobody listened to it during production, so please listen before you share it.

## Rebuilding

```
cd songs/la-loca-tlacuaque/production
export WHY_LLARTA_WORK=<work folder>
python3 beat.py <inst_dir>                                        # ~4 min
python3 vocals.py                                                 # 2 min with a warm cache, over 20 min cold
python3 mix.py <inst_dir> $WHY_LLARTA_WORK/tlac/stems ../release  # ~4 min; needs ffmpeg
```

The code imports the shared synthesis, voice and mastering modules from [`../../why-llarta/production/`](../../why-llarta/production/). The work folder must contain the same models as for that song:
- `models/kokoro-v1.0.onnx` and `models/voices-v1.0.bin`, from the kokoro-onnx GitHub releases;
- `models/asr/sherpa-onnx-whisper-small/`, from the sherpa-onnx GitHub releases.
