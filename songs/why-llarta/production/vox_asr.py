"""Local intelligibility check with sherpa-onnx Whisper (multilingual 'small', int8)."""
import os
import re
import unicodedata

import numpy as np
from scipy.signal import resample_poly

SCRATCH = "/tmp/claude-0/-home-user-cf336/c7c99ebc-57ee-5e7f-9a20-477ea4b0b187/scratchpad"
ASR_DIR = os.environ.get("VOX_ASR_DIR", os.path.join(SCRATCH, "models", "asr", "sherpa-onnx-whisper-small"))

_REC = {}


def _recognizer(lang):
    import sherpa_onnx
    if lang not in _REC:
        _REC[lang] = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=os.path.join(ASR_DIR, "small-encoder.int8.onnx"),
            decoder=os.path.join(ASR_DIR, "small-decoder.int8.onnx"),
            tokens=os.path.join(ASR_DIR, "small-tokens.txt"),
            language=lang, task="transcribe", num_threads=4)
    return _REC[lang]


def available():
    return os.path.exists(os.path.join(ASR_DIR, "small-encoder.int8.onnx"))


def transcribe(audio, sr, lang="en"):
    x = np.asarray(audio, dtype=np.float64)
    if sr != 16000:
        from math import gcd
        g = gcd(int(sr), 16000)
        x = resample_poly(x, 16000 // g, int(sr) // g)
    pk = np.max(np.abs(x)) + 1e-9
    x = (x / pk * 0.5).astype(np.float32)
    # pad with silence so short phrases are decoded fully
    pad = np.zeros(int(0.4 * 16000), np.float32)
    x = np.concatenate([pad, x, pad])
    rec = _recognizer(lang)
    st = rec.create_stream()
    st.accept_waveform(16000, x)
    rec.decode_stream(st)
    return st.result.text.strip()


def norm_words(s):
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.replace("'", "").replace("’", "")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return s.split()


# words ASR may legitimately spell differently
_EQUIV = {"vayarta": "vallarta", "ok": "okay", "everyday": "every day", "heart beat": "heartbeat",
          "badum": "ba dum", "ba-dum": "ba dum"}


def word_scores(ref, hyp):
    """(recall of reference words in hypothesis [bag of words], word accuracy = 1 - WER)."""
    r = norm_words(ref)
    h = norm_words(" ".join(_EQUIV.get(w, w) for w in norm_words(hyp)))
    h = norm_words(" ".join(h).replace("every day", "everyday").replace("heart beat", "heartbeat"))
    r = norm_words(" ".join(r).replace("every day", "everyday").replace("heart beat", "heartbeat"))
    if not r:
        return 1.0, 1.0
    pool = list(h)
    hit = 0
    for w in r:
        if w in pool:
            pool.remove(w)
            hit += 1
    # Levenshtein on words
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
            prev = cur
    return hit / len(r), max(0.0, 1.0 - d[len(h)] / len(r))
