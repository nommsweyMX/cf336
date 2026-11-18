# Your voice sample for Suno "Voices"

Suno can sing a song in **your own voice** through its **Voices** feature (launched with v5.5 in March 2026, and on the iOS and Android apps since August 2026). You give it one recording of you singing, then confirm it is you by reading a short phrase aloud. Nobody can make that sample for you: the verification step checks that the singing and the speaking come from the same person. This script makes the recording as useful as possible for *this* song.

> **Reliability:** this comes from research done on 2026-11-13. Suno's own pages could not be opened directly, so the details come from search snippets of them plus third-party guides. Two points are **unconfirmed**: whether Voices works on every **v6** model (the FAQ still names v5.5), and how much the **Free** plan gets (sources disagree; Pro and Premier have full access).

---

## What to record (about 75 to 90 seconds, one take, no music in the file)

Play `audio/05_voice_recording_guide_124bpm_Fm-to-Gm.mp3` in **wired headphones** while you record. It gives you a 1-bar count-in, the 124 BPM heartbeat/beatbox groove, and a soft F-minor chord. At bar 17 it steps up to G minor, which is where the song's final-chorus key change goes. Any melody you like is fine: Suno learns your **timbre and delivery**, not a tune. Staying in the song's tempo and register just gives it the most relevant data.

Loop the guide if you need more time. Leave about 1 second of silence between parts.

**Part 1: sung, your natural chest voice, over the F-minor pad (about 25 s)**

```
You are not what you say
You are what you repeat
Said once, it's a sentence
Lived daily, a heartbeat
(again, with a little more push the second time)
```

**Part 2: sung, Spanish, open vowels, lean into "días", "estás" and "gracias" (about 15 s)**

```
¡Hola, buenos días! ¿Cómo estás?
Bien, gracias, ¿y tú?
¡Muy bien también, gracias!
¡Qué bueno! Otra vez
```

**Part 3: half-spoken, deadpan, the way you would say it at dinner (about 15 s)**

This teaches Suno your dry speaking delivery for the verses.

```
Every night the bay takes a bow
Like a soprano milking applause
The light flatters linen, and me, for now
Ya soy local, said too soon
```

**Part 4: sung higher, when the guide lifts to G minor; this is your top range for the key change (about 20 s)**

```
I am not what I say
I am what I repeat
Bien, gracias, ¿y tú? ¿Y yo?
Lived daily, a heartbeat
```

Skip whispering and beatboxing in this sample. The song's tags ask Suno for those, and mixing them into the Voice sample tends to blur the model.

---

## How to record it

- **Room:** the quietest, softest room you have. A closet full of clothes is ideal. Avoid tiled bathrooms and kitchens, and turn off fans and the AC.
- **Phone:** hold it 15 to 20 cm (6 to 8 in) from your mouth, slightly to the side so "p" and "b" sounds don't pop. Use the built-in voice recorder. On iPhone Voice Memos, turn off **Enhance Recording**. Turn off any noise suppression, reverb or auto-tune.
- **Headphones:** wired, at low volume, so the guide track does **not** leak into the mic. The file must be your voice only.
- **Level:** loud parts should never distort. If your recorder shows a meter, keep peaks well below the top.
- Record **2 or 3 takes** and keep the best one. Trim the silence at the start and end. Export **WAV or MP3**, which are the safe formats. Officially a sample can be 15 s to 4 min, and Suno uses the best 2 minutes, but guides say clips under about 20 s often fail.

## Making the Voice in Suno

1. Go to **Create**, then **Add Voice**, then **Create Voice**.
2. Choose **Upload** and pick your file. You can also record live with the mic, but the guide track can't play in your ears that way.
3. If the clip is longer than 2 minutes, select the best 2-minute section.
4. **Verification:** Suno shows a short random phrase. **Speak** it in your normal voice. Don't sing it, don't put on an accent, and don't slow it down. This is the most common place it fails.
5. Name the Voice, tick the box confirming the rights to the voice are yours, and click **Save**.
6. When you generate: **Add Voice**, pick yours, then check the **model dropdown** still allows it. If the Voice greys out on a v6 model, switch to whichever model the dropdown accepts with your Voice.

The Voice is private to your account and nobody else can select it.
