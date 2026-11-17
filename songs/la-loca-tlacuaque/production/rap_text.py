"""Text side of the rap engine (no audio, no models): Mexican phoneme fix, Spanish syllables + stress from the
Kokoro/espeak phoneme string, and the FLOW: which swung sixteenth every rap syllable lands on.

    mexican(ph)                    Castilian espeak phonemes -> Mexican (θ->s, ʎ->ʝ, unstressed i/u + vowel -> j/w)
    syllabify_es(chars)            -> (syllables, joins)   one syllable per vowel nucleus; joins = possible mergers
    apply_joins(syls, which)       merge syllable k+1 into k (synalepha / syneresis)
    rhyme_index(syls)              index of the LAST primary-stressed syllable (the rhyme stress)
    flow_line(...)                 slots for one line (anchor rule, natural-rhythm spreading, 32nd pairs)
    slot_time(score, bar, pos)     swung time of a (possibly fractional) sixteenth position

`chars` is the list of phoneme characters that Kokoro tokenises (Engine.tokens(ph)[1]); every index in a
syllable refers to that list, so the same structure drives the duration plan of the neural engine.
"""
import numpy as np

VOWELS = set("aeiouyɑɐɒæɔəɘɚɛɜɝɞɤɨɪʊʉʌʏøœɯɵᵻ")
LENGTH = "ː"
PUNCT = set(",.!?;:—…\"()¡¿-–")
GLIDE_OFF = "ɪʊ"          # espeak writes Spanish falling diphthongs as aɪ eɪ oɪ aʊ eʊ


def mexican(ph):
    """espeak-ng 'es' is Castilian: z/ce/ci -> θ and ll -> ʎ. Mexican Spanish has neither (seseo, yeismo).
    Also normalises the few rising diphthongs espeak leaves as two vowels (vidrio -> βˈiðɾio, luego -> luˈeɣo)
    to glide + vowel, like it already writes most of them (metió -> metjˈo, cuatro -> kwˈatɾo)."""
    ph = ph.replace("θ", "s").replace("ʎ", "ʝ")
    out = list(ph)
    for i, c in enumerate(out):
        if c not in "iu":
            continue
        if i > 0 and out[i - 1] in "ˈˌ":
            continue                                  # stressed i / u: a real nucleus (tenía, río, baúl)
        j = i + 1
        if j < len(out) and out[j] in "ˈˌ":
            j += 1
        if j < len(out) and out[j] in VOWELS and out[j] != c and out[j] not in GLIDE_OFF:
            out[i] = "j" if c == "i" else "w"
    return "".join(out)


def _cls(c):
    if c in VOWELS:
        return "V"
    if c == LENGTH:
        return "L"
    if c in "ˈˌ":
        return "S"
    if c == " ":
        return "W"
    if c in PUNCT:
        return "P"
    return "C"


def syllabify_es(chars):
    """One syllable per vowel nucleus (a following ɪ/ʊ or unstressed i/u off-glide belongs to it).
    Returns (syls, joins):
      syls[k]  = dict(nuc=[token indices], stress=2 primary / 1 secondary / 0, word=word number,
                      punct=True if punctuation follows the syllable's word and the syllable is word-final)
      joins    = [(k, kind)] where syllable k+1 may be merged into k:
                 'auto'   synalepha between words, no primary stress involved (or the same vowel twice)
                 'opt'    synalepha between words involving a stressed vowel
                 'hiatus' two vowels inside one word (syneresis: last resort)"""
    n = len(chars)
    word_of, w = [], 0
    for c in chars:
        word_of.append(w)
        if c == " ":
            w += 1
    syls, i = [], 0
    while i < n:
        c = chars[i]
        if c in VOWELS:
            idx, j = [i], i + 1
            while j < n and chars[j] == LENGTH:
                idx.append(j)
                j += 1
            if j < n and c not in GLIDE_OFF and (chars[j] in GLIDE_OFF or (chars[j] in "iu" and chars[j] != c)):
                idx.append(j)
                j += 1
            st, b = 0, i - 1
            while b >= 0 and _cls(chars[b]) in "CS":
                if chars[b] == "ˈ":
                    st = 2
                    break
                if chars[b] == "ˌ":
                    st = max(st, 1)
                b -= 1
            syls.append(dict(nuc=idx, stress=st, word=word_of[i]))
            i = j
        else:
            i += 1
    for k, s in enumerate(syls):
        e = s["nuc"][-1] + 1
        nxt = syls[k + 1]["nuc"][0] if k + 1 < len(syls) else n
        s["punct"] = any(_cls(chars[x]) == "P" for x in range(e, nxt))
    joins = []
    for k in range(len(syls) - 1):
        a, b = syls[k], syls[k + 1]
        between = chars[a["nuc"][-1] + 1:b["nuc"][0]]
        if any(_cls(c) in "CP" for c in between):
            continue
        if " " in between:
            same = chars[a["nuc"][-1]] == chars[b["nuc"][0]]
            strong = (a["stress"] == 2) + (b["stress"] == 2)
            kind = "auto" if (strong == 0 or (same and strong < 2)) else "opt"
        else:
            kind = "hiatus"
        joins.append((k, kind))
    return syls, joins


def apply_joins(syls, ks):
    """Merge syllable k+1 into syllable k for every k in ks (indices into the ORIGINAL list)."""
    out, skip = [], False
    ks = set(ks)
    for k, s in enumerate(syls):
        if skip:
            skip = False
        if out and (k - 1) in ks:
            p = out[-1]
            p["nuc"] = list(range(p["nuc"][0], s["nuc"][-1] + 1))
            p["stress"] = max(p["stress"], s["stress"])
            p["punct"] = s["punct"]
            p["joined"] = p.get("joined", 0) + 1
            p["orig"] = p["orig"] + [k]
        else:
            out.append(dict(s, nuc=list(s["nuc"]), orig=[k]))
    return out


def rhyme_index(syls):
    """Index of the last primary-stressed syllable (falls back to secondary stress, then the penultimate)."""
    for want in (2, 1):
        ks = [k for k, s in enumerate(syls) if s["stress"] >= want]
        if ks:
            return ks[-1]
    return max(0, len(syls) - 2)


def syl_text(chars, syls, k):
    """Printable phoneme text of syllable k (its consonants up to the next nucleus are split naively)."""
    a = syls[k - 1]["nuc"][-1] + 1 if k > 0 else 0
    b = syls[k + 1]["nuc"][0] if k + 1 < len(syls) else len(chars)
    pre = "".join(chars[a:syls[k]["nuc"][0]])
    post = "".join(chars[syls[k]["nuc"][-1] + 1:b])
    # consonants after the nucleus up to a word break belong here; the rest to the next syllable
    if k + 1 < len(syls):
        cut = post.find(" ")
        post = post[:cut] if cut >= 0 else ""
    if k > 0:
        cut = pre.rfind(" ")
        if cut >= 0:
            pre = pre[cut + 1:]
        else:
            # word-internal cluster: keep the last consonant (plus a liquid/glide cluster) as this onset
            body = [c for c in pre if _cls(c) == "C"]
            keep = 1 if body else 0
            if len(body) >= 2 and (body[-1] in "ɾlrjw" or "".join(body[-2:]) == "tʃ"):
                keep = 2
            if len(body) >= 3 and body[-1] in "jw" and body[-2] in "ɾl":
                keep = 3
            pre = "".join(body[len(body) - keep:]) if keep else ""
    nuc = "".join(chars[x] for x in syls[k]["nuc"])
    txt = (pre + nuc + post)
    return "".join(c for c in txt if _cls(c) not in "PS").strip().replace(" ", "‿")


# ----------------------------------------------------------------------------- the grid
def slot_time(score, bar, pos):
    """Swung time (s) of sixteenth position `pos` (fractional = between two swung slots)."""
    i = int(np.floor(pos + 1e-9))
    f = pos - i
    t0 = score.t16(bar, i)
    if f < 1e-9:
        return t0
    return t0 + f * (score.t16(bar, i + 1) - t0)


# ----------------------------------------------------------------------------- flow
MAX_STRETCH = 2.0          # a line never spreads wider than 2 slots per syllable on average
W_POS = 0.5                # weight of (slot - scaled natural onset)^2


def _gap_cost(d, stress, punct):
    """Cost of giving a syllable d slots before the next one starts."""
    if d == 1:
        return 0.9 if punct else (0.3 if stress == 2 else 0.0)
    if d == 2:
        return 0.0 if (punct or stress == 2) else (0.35 if stress == 1 else 0.7)
    if d == 3:
        return 0.6 if punct else (2.5 if stress == 2 else 5.0)
    if d == 4:
        return 2.0 if punct else 9.0
    return 1e6


def _spread(s0, anchor, tau, stress, punct):
    """Slots for syllables 0..L-1 (strictly increasing, first = s0, all < anchor) that follow the natural
    speech rhythm tau (onset times incl. the anchor syllable's, len L+1), by dynamic programming."""
    L = len(stress)
    span = anchor - s0
    T = tau[L] - tau[0]
    p = [s0 + (tau[i] - tau[0]) / T * span for i in range(L)] if T > 0 else [s0 + i for i in range(L)]
    INF = 1e18

    def pos_cost(i, s):
        c = W_POS * (s - p[i]) ** 2
        if stress[i] == 2:
            c += 0.4 * (s % 2) - 0.15 * (s % 4 == 0)
        return c
    best = [dict() for _ in range(L)]
    best[0][s0] = (0.0, None)
    for i in range(1, L):
        for s in range(s0 + i, anchor - (L - i) + 1):
            m = (INF, None)
            for sp_, (c0, _) in best[i - 1].items():
                d = s - sp_
                if d < 1 or d > 4:
                    continue
                c = c0 + _gap_cost(d, stress[i - 1], punct[i - 1]) + pos_cost(i, s)
                if c < m[0]:
                    m = (c, sp_)
            if m[1] is not None:
                best[i][s] = m
    end = (INF, None)
    for s, (c0, _) in best[L - 1].items():
        d = anchor - s
        if 1 <= d <= 4:
            c = c0 + _gap_cost(d, stress[L - 1], punct[L - 1])
            if c < end[0]:
                end = (c, s)
    if end[1] is None:                       # cannot happen while span <= 4 L; be safe
        return [s0 + int(round(i * span / L)) for i in range(L)], p
    out, s = [], end[1]
    for i in range(L - 1, -1, -1):
        out.append(s)
        s = best[i][s][1]
    return out[::-1], p


def _pick_pairs(K, natdur, stress, punct):
    """Choose K disjoint adjacent syllable pairs (i, i+1) to share one slot (thirty-second notes): the
    fastest unstressed pairs of natural speech first. Returns the list of first indices, or None."""
    L = len(natdur)
    INF = 1e18

    def cost(i):
        c = natdur[i] + natdur[i + 1]
        for j in (i, i + 1):
            c += {0: 0.0, 1: 0.12, 2: 0.6}[stress[j]]
        if punct[i]:
            c += 0.5
        return c
    # dp[i][k] = min cost using syllables < i with k pairs
    dp = [[INF] * (K + 1) for _ in range(L + 1)]
    ch = [[None] * (K + 1) for _ in range(L + 1)]
    dp[0][0] = 0.0
    for i in range(L):
        for k in range(K + 1):
            if dp[i][k] >= INF:
                continue
            if dp[i][k] < dp[i + 1][k]:
                dp[i + 1][k], ch[i + 1][k] = dp[i][k], (i, k, False)
            if k < K and i + 1 < L:
                c = dp[i][k] + cost(i)
                if c < dp[i + 2][k + 1]:
                    dp[i + 2][k + 1], ch[i + 2][k + 1] = c, (i, k, True)
    if dp[L][K] >= INF:
        return None
    pairs, i, k = [], L, K
    while i > 0:
        pi, pk, used = ch[i][k]
        if used:
            pairs.append(pi)
        i, k = pi, pk
    return sorted(pairs)


def flow_line(syls, joins, tau, natdur, anchor=12, lo=-3, hi=15, start_min=None):
    """Slots for one rap line.
    syls/joins: syllabify_es output.  tau[k]: natural-speech onset (s) of syllable k's vowel.
    natdur[k]: natural duration (s) of syllable k.  start_min: earliest slot the previous line leaves free.
    Returns dict(syls (after merging), pos [slot per syllable, x.5 = thirty-second], rhyme, crowded,
    pairs, merged, notes [log])."""
    log = []
    lo_eff = lo if start_min is None else max(lo, start_min)
    tau, natdur = list(tau), list(natdur)

    def merged(ks):
        s2 = apply_joins(syls, ks)
        t2, d2, k2 = [], [], 0
        for k in range(len(syls)):
            if k > 0 and (k - 1) in ks:
                d2[-1] += natdur[k]
            else:
                t2.append(tau[k])
                d2.append(natdur[k])
        return s2, t2, d2

    r0 = rhyme_index(syls)
    use = set(k for k, kind in joins if kind == "auto" and k + 1 < r0)
    cur, ctau, cdur = merged(use)
    r = rhyme_index(cur)
    avail = anchor - lo_eff
    crowded = False
    # a line that would need pickup slots first uses its optional synalephas (cagó‿a, llevó‿al): what fast
    # speech does anyway, and the previous rhyme gets room to ring
    soft = anchor - max(lo_eff, 0)
    if r > soft:
        cand = [k for k, kind in joins if kind == "opt" and k + 1 < r0]
        cand.sort(key=lambda k: min(natdur[k], natdur[k + 1]))
        for k in cand:
            if r <= soft:
                break
            if (k - 1) in use or (k + 1) in use:
                continue
            use.add(k)
            cur, ctau, cdur = merged(use)
            r = rhyme_index(cur)
            log.append(f"synalepha at syllable {k} (stressed vowel) to stay near the downbeat")
    if r > avail:
        # optional synalepha (stressed vowels), then syneresis inside words, weakest (shortest) first
        for kinds in (("opt",), ("auto", "opt"), ("hiatus",)):
            cand = [k for k, kind in joins if kind in kinds and k not in use and (k + 1 < r0 or kinds == ("auto", "opt"))
                    and k + 1 <= r0 and not (k + 1 == r0 and kinds != ("auto", "opt"))]
            cand.sort(key=lambda k: min(natdur[k], natdur[k + 1]))
            for k in cand:
                if r <= avail:
                    break
                if (k - 1) in use or (k + 1) in use:
                    continue                      # no three-vowel pile-ups
                use.add(k)
                cur, ctau, cdur = merged(use)
                r = rhyme_index(cur)
                log.append(f"extra merge at syllable {k} ({dict(joins)[k]})")
            if r <= avail:
                break
    n = len(cur)
    stress = [s["stress"] for s in cur]
    punct = [s["punct"] for s in cur]
    pos = [None] * n
    pos[r] = float(anchor)
    pairs = []
    if r == 0:
        pass
    elif r <= avail:
        if r <= anchor - max(lo_eff, 0):
            s0 = max(lo_eff, 0, anchor - int(MAX_STRETCH * r))
            if s0 % 2 and anchor - (s0 + 1) >= r and s0 > max(lo_eff, 0):
                s0 += 1
            if anchor - s0 == r:
                sl = list(range(s0, anchor))
            else:
                sl, _ = _spread(s0, anchor, ctau[:r + 1], stress[:r], punct[:r])
        else:
            sl = list(range(anchor - r, anchor))          # pickup into the previous bar
        for i, s in enumerate(sl):
            pos[i] = float(s)
    else:
        crowded = True
        K = r - avail
        pk = _pick_pairs(K, cdur[:r], stress[:r], punct[:r]) if avail > 0 else None
        if pk is None:
            # hopeless: more than two syllables per slot; spread them evenly and report
            log.append(f"OVERCROWDED: {r} syllables before the anchor in {avail} slots")
            for i in range(r):
                pos[i] = lo_eff + i * (anchor - lo_eff) / r
        else:
            pairs = pk
            s, i = lo_eff, 0
            while i < r:
                pos[i] = float(s)
                if i in pk:
                    pos[i + 1] = s + 0.5
                    i += 2
                else:
                    i += 1
                s += 1
            log.append(f"crowded: {K} thirty-second pair(s) at syllables {pk}")
    # after the rhyme stress: one slot each, squeezed if the line would pass `hi`
    post = n - 1 - r
    if post:
        room = hi - anchor - 1            # slots available for the post-stress syllables (each needs >= its own)
        step = 1.0 if post <= max(room, 0) else max(0.5, room / post) if room > 0 else 0.5
        if step < 1.0:
            crowded = True
            log.append(f"crowded tail: {post} syllables after the rhyme stress, {max(room, 0)} slots before hi")
        for j in range(post):
            pos[r + 1 + j] = anchor + 1 + j * step if step == 1.0 else anchor + (j + 1) * step
    return dict(syls=cur, pos=pos, rhyme=r, crowded=crowded, pairs=pairs, merged=sorted(use), log=log,
                n_syllables=n, start_min=lo_eff)
