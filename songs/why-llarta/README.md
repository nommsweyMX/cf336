# "What You Repeat": a Suno song from *WHY-LLARTA*

A Eurovision grand-final anthem on a deep house / melodic techno chassis, in your own voice, with a heartbeat and beatbox pulse that opens the song, runs underneath it, and closes it.

**The idea:** the six chapter titles are one complete Spanish small-talk exchange (*¡Hola, buenos días! → ¿Cómo estás? → Bien, gracias, ¿y tú? → ¡Muy bien también, gracias! → Qué bueno → Que tengas un buen día*). That becomes the call-and-response chorus. The book's thesis, *"You are not what you say. You are what you repeat"*, is the hook, and the form of a house track (repetition) proves it. A resting heartbeat at 62 BPM is exactly half of the 124 BPM house tempo, so the heart doubles into the kick at the start and halves back to a stop at the end.

| File | What it is |
|---|---|
| [`SUNO_PASTE.md`](SUNO_PASTE.md) | **Copy-paste sheet:** title, style prompt, exclude list, lyrics (two versions), settings, section map, and the source of every borrowed line |
| [`VOICE_SAMPLE_SCRIPT.md`](VOICE_SAMPLE_SCRIPT.md) | What to sing and how to record it so Suno's **Voices** feature sounds like you |
| `audio/01_heartbeat_intro_62-to-124bpm` | 16 s: breath, solo heartbeat, beatbox creeps in, the heart doubles into the 124 BPM groove, fill into the song |
| `audio/02_heartbeat_beatbox_loop_124bpm` | 7.7 s seamless 4-bar loop to run underneath the whole song (Suno Studio / any DAW) |
| `audio/03_heartbeat_outro_124-to-62bpm` | 20.6 s: the groove strips back, the heart halves, slows, last beat, exhale (for lining up under the outro in a DAW) |
| `audio/04_heartbeat_tail_only_62bpm` | 12 s: heartbeat only, slowing to a stop, then an exhale. The easy way to end any take |
| `audio/05_voice_recording_guide_124bpm_Fm-to-Gm` | 49 s headphone guide for recording your voice sample: count-in, groove, F minor, then G minor |
| `audio/make_samples.py` | Regenerates all of the audio (`python3 make_samples.py <outdir>`, needs numpy, scipy and ffmpeg) |

Every sample comes as **WAV** (upload these to Suno) and **MP3** (for previewing on your phone). All are 44.1 kHz stereo, peaking at -1 dBFS.

## Step by step

### 1. Make your Voice (once)
Follow [`VOICE_SAMPLE_SCRIPT.md`](VOICE_SAMPLE_SCRIPT.md): about 90 s of you singing the hook, the Spanish ladder and a couple of deadpan verse lines, recorded dry over the guide track in headphones. Then go to **Create → Add Voice → Create Voice**, upload it, speak the verification phrase, and save.

### 2. Start the song on the heartbeat
1. **Upload** `audio/01_heartbeat_intro_62-to-124bpm.wav` (Create, then upload audio; confirm you own it, which you do).
2. On the uploaded clip choose **Extend**, extending from the **end** (about 0:16).
3. Paste from `SUNO_PASTE.md`: **Lyrics 4a** (it starts at Verse 1, because the clip *is* the intro), the **Style**, and the **Exclude** list.
4. Select **your Voice**. Suggested settings: **Audio Influence 75-85%**, **Style Influence 70-75%**, **Weirdness 50-55%**, **Variety Off**. These are community starting points, not official values.
5. Generate **4 to 6 takes**. Keep a take if:
   - the heartbeat flows straight into the groove,
   - the chorus echoes alternate between lead and crowd,
   - Verse 3 is actually chanted, not skipped,
   - the breakdown is whispered over only the heartbeat,
   - the heartbeat races into a stop before the last chorus,
   - and the final chorus lifts in key.

   If one section misses, use v6's plain-language **section edit** on that section only (e.g. *"raise this final chorus a whole step to G minor, keep the lyrics"*) rather than regenerating everything.

No upload? Use **Lyrics 4b**, which starts with its own heartbeat intro tag.

### 3. End it on the heartbeat
Suno won't reliably finish on a solo heartbeat, so add one yourself:

- **Easiest (any computer with ffmpeg):** download your take and run:
  ```
  ffmpeg -i "what-you-repeat.mp3" -i audio/04_heartbeat_tail_only_62bpm.wav \
    -filter_complex "[0:a][1:a]acrossfade=d=1.5:c1=exp:c2=qsin" -b:a 320k what-you-repeat-final.mp3
  ```
  This crossfades the last 1.5 s of the song into the heartbeat. Use `d=3` for a longer blend.
- **Suno Studio (Premier):** drop `03_heartbeat_outro_124-to-62bpm.wav` on a new track so its groove lines up with the song's last bars, and fade the song out under it.
- **Any DAW / phone editor:** same idea. Line up the outro, or just place the tail after the last note.

### 4. Beatbox all the way through
The style prompt asks for a *"human beatbox vocal percussion layer throughout"*. The lyrics also carry `(boom-tss, ka-tss)` and `(ba-dum, ba-dum)` ad-libs, so the beatbox comes out in your voice. To *guarantee* it in Suno Studio (Premier), put `02_heartbeat_beatbox_loop_124bpm.wav` on its own track, duplicate it across the song, and sit it well under the mix (around -10 to -14 dB). If you want the beatbox to be literally your mouth, record 4 bars over the guide track and use it the same way.

### If something gets flagged
- **"Eurovision" stripped from the style:** use `Euro-pop song-contest grand-final anthem` instead.
- **The meat-dress line flagged:** use `And the dance-floor diva too` instead.
- **"Vallarta" sung as va-LAR-ta:** type `Vayarta` in Suno's lyric box only.
- **"bow" rhyming with "go":** type `bough` in Suno's lyric box only.
