"""THE NEXT REPETITION: MP4 export (H.264 video + AAC audio).

    python3 make_mp4.py <master.wav> <out.mp4> [work_dir]

The picture is plain on purpose: the title, the name of the scene that is playing, a moving waveform and a
progress line, so the file plays and uploads anywhere a video is expected.
"""
import os
import subprocess
import sys

import score as S

FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def main(wav, out, work):
    os.makedirs(work, exist_ok=True)
    total = S.SONG_SECONDS

    def txt(name, text):
        p = os.path.join(work, name + ".txt")
        open(p, "w").write(text)
        return p

    chain = ["[0:a]aformat=channel_layouts=mono,showwaves=s=1280x280:mode=cline:rate=20:colors=0xE8B04A:scale=sqrt,format=rgba[w]",
             "color=c=0x0F1418:s=1280x720:r=20[bg]", "color=c=0xE8B04A:s=1280x6:r=20[bar]",
             "[bg][w]overlay=0:330:shortest=1[v0]",
             f"[v0][bar]overlay=x='-1280+1280*t/{total:.3f}':y=714:shortest=1[v1]"]
    draw = [f"drawtext=fontfile={FONT_B}:textfile={txt('title', S.TITLE)}:fontcolor=0xF4EDE0:fontsize=58:x=(w-text_w)/2:y=96",
            f"drawtext=fontfile={FONT}:textfile={txt('sub', 'an instrumental in eleven scenes')}:fontcolor=0x8FA3AD:fontsize=24:x=(w-text_w)/2:y=176"]
    for row in S.TL.table():
        name, a, z, stage = row[0], row[1], row[2], row[8]
        if name == S.SCENES[-1][0]:
            z = total
        draw.append(f"drawtext=fontfile={FONT}:textfile={txt(name, stage)}:fontcolor=0xE8B04A:fontsize=32:x=(w-text_w)/2:y=250:"
                    f"enable='between(t,{a:.3f},{z:.3f})':alpha='min(1,(t-{a:.3f})/0.6)'")
    chain.append("[v1]" + ",".join(draw) + ",format=yuv420p[v]")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-filter_complex", ";".join(chain), "-map", "[v]", "-map", "0:a",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "30", "-maxrate", "1400k", "-bufsize", "2800k", "-pix_fmt", "yuv420p", "-r", "20",
           "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", "-metadata", f"title={S.TITLE}", "-t", f"{total:.3f}", out]
    subprocess.run(cmd, check=True)
    print("wrote", out, f"{os.path.getsize(out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(sys.argv[2]), "_mp4_work"))
