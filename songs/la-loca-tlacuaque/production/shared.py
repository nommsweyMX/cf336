"""Path setup: this song reuses the DSP toolkit, the vocal engine and the mastering chain written for
"What You Repeat" (songs/why-llarta/production). Import this module first; the shared folder is appended
to sys.path, so this folder's own score.py always wins."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SHARED = os.path.normpath(os.path.join(HERE, "..", "..", "why-llarta", "production"))
if SHARED not in sys.path:
    sys.path.append(SHARED)
# Working folder for models, render cache and intermediate stems (git-ignored).
WORK = os.environ.setdefault("WHY_LLARTA_WORK", os.path.join(HERE, "work"))
