"""Full-song lyrics from LRCLIB and the structure they reveal: hook, repetition, density, line length, rhymes."""
import os
import re
import statistics
import time
from collections import Counter

import requests

from .scout import norm

LRCLIB = "https://lrclib.net/api"
HEADERS = {"User-Agent": "ComfyUI-HitMaker"}
TIME_RE = re.compile(r"^\[(\d+):(\d+(?:\.\d+)?)\](.*)$")
SECTION_RE = re.compile(r"^\[([^\]]+)\]$")
VOCABLES = {"uh", "oh", "eh", "ah", "ey", "yeah", "ye", "wo", "woh", "na", "la", "mmm", "mm", "ooh", "ay", "ja", "prr", "brr", "yeh"}
# Function words that only occur in English, to measure code-switching in Spanish lyrics.
ENGLISH = {"the", "you", "your", "my", "baby", "and", "with", "what", "that", "this", "girl", "love", "all", "night", "shawty", "know", "want", "just", "like", "got"}


def strip_title(title):
    return re.sub(r"\s*[\(\[].*?[\)\]]|\s+(feat|ft)\..*$", "", title, flags=re.I).strip()


def api(endpoint, **params):
    """LRCLIB GET; it answers 503/429 when busy, so back off and retry a few times."""
    for attempt in range(4):
        r = requests.get(f"{LRCLIB}/{endpoint}", params=params, headers=HEADERS)
        if r.status_code not in (429, 503):
            return r
        time.sleep(2 ** attempt)
    return r


def fetch(artist, title, duration):
    """Synced lyrics for a track, or None. Tries the exact match first, then the closest search hit by duration."""
    r = api("get", artist_name=artist, track_name=title, duration=duration)
    if r.status_code == 200 and r.json().get("syncedLyrics"):
        return r.json()
    hits = []
    for name in dict.fromkeys([title, strip_title(title)]):
        r = api("search", track_name=name, artist_name=artist)
        if r.status_code == 200:
            hits += r.json()
    hits = [h for h in hits if h.get("syncedLyrics") and norm(artist) in norm(h["artistName"]) and abs(h["duration"] - duration) <= 15]
    return min(hits, key=lambda h: abs(h["duration"] - duration)) if hits else None


def parse(synced):
    lines = []
    for raw in synced.splitlines():
        m = TIME_RE.match(raw.strip())
        if m and m.group(3).strip():
            lines.append((int(m.group(1)) * 60 + float(m.group(2)), m.group(3).strip()))
    return lines


def syllables(line):
    return len(re.findall(r"[aeiouáéíóúü]+", line.lower()))


def rhyme(line):
    """Last two vowels of the line, a rough assonance key that works for Spanish and English."""
    vowels = re.findall(r"[aeiou]", norm(line))
    return "".join(vowels[-2:]) if len(vowels) >= 2 else None


def stats(synced, duration):
    return line_stats(parse(synced), duration)


def text_stats(text, duration, first_vocal=0.0):
    """The same stats for plain lyrics with [Section] tags. Lines are spread evenly from the first
    vocal to the end, so the times are estimates that assume the vocal runs through the rest of the song."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    sung = [line for line in lines if not SECTION_RE.match(line)]
    step = (duration - first_vocal) / max(1, len(sung))
    result = line_stats([(first_vocal + i * step, line) for i, line in enumerate(sung)], duration)
    result["sections"] = [SECTION_RE.match(line).group(1) for line in lines if SECTION_RE.match(line)]
    return result


def line_stats(lines, duration):
    if not lines:
        return None
    keys = [norm(text) for _, text in lines]
    words = [w for k in keys for w in k.split()]
    sung = [(t, text, k) for (t, text), k in zip(lines, keys) if not set(k.split()) <= VOCABLES]
    counts = Counter(k for _, _, k in sung)
    order = list(dict.fromkeys(k for _, _, k in sung))
    # On a tie the earliest line wins: it is the one that opens the chorus.
    hook_key, hook_count = max(counts.items(), key=lambda kv: (kv[1], -order.index(kv[0])))
    first_repeat = next((t for i, (t, _, k) in enumerate(sung) if k in {s[2] for s in sung[:i]}), None)
    span = (lines[-1][0] - lines[0][0]) or 1
    return {
        "duration": duration,
        "lines": len(sung),
        "words_per_minute": round(len(words) / span * 60),
        "first_vocal": round(lines[0][0], 1),
        "hook": next(text for _, text, k in sung if k == hook_key),
        "hook_repeats": hook_count,
        "hook_first_at": round(next(t for t, _, k in sung if k == hook_key), 1),
        "first_repeat_at": round(first_repeat, 1) if first_repeat is not None else None,
        "repeated_line_share": round(sum(c for c in counts.values() if c > 1) / len(sung), 2),
        "vocable_line_share": round(1 - len(sung) / len(lines), 2),
        "syllables_per_line": statistics.median(syllables(text) for _, text, _ in sung),
        "rhymes": [r for r, _ in Counter(rhyme(text) for _, text, _ in sung if rhyme(text)).most_common(3)],
        "english_word_share": round(sum(w in ENGLISH for w in words) / max(1, len(words)), 2),
    }


def attach(tracks):
    """Fetch lyrics for scouted tracks and save them as .lrc beside each preview."""
    for t in tracks:
        found = fetch(t["artist"], t["title"], t["duration"])
        if not found:
            t["lyrics_file"] = None
            continue
        path = os.path.splitext(t["file"])[0] + ".lrc"
        with open(path, "w") as f:
            f.write(found["syncedLyrics"])
        t["lyrics_file"] = path
        t["lyrics_stats"] = stats(found["syncedLyrics"], found["duration"])
    return tracks
