import json
import re
import statistics
from collections import Counter
from difflib import SequenceMatcher

from .scout import norm

NOTES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
MAJOR = {0: "I", 1: "bII", 2: "II", 3: "bIII", 4: "III", 5: "IV", 6: "#IV", 7: "V", 8: "bVI", 9: "VI", 10: "bVII", 11: "VII"}
MINOR = {0: "I", 1: "bII", 2: "II", 3: "III", 4: "#III", 5: "IV", 6: "#IV", 7: "V", 8: "VI", 9: "#VI", 10: "VII", 11: "#VII"}
CHORD_RE = re.compile(r"^([A-G])([#b♯♭]?)(.*)$")
NUMERAL_RE = re.compile(r"^(b|#)?(i{1,3}|iv|vi{0,2}|v)(°|7)?$", re.I)


def extract_json(text):
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                return decoder.raw_decode(text[i:])[0]
            except json.JSONDecodeError:
                continue
    return None


def pitch(root, accidental):
    return (NOTES[root] + {"#": 1, "♯": 1, "b": -1, "♭": -1}.get(accidental, 0)) % 12


def parse_key(key):
    m = CHORD_RE.match((key or "").strip())
    if not m:
        return None
    return pitch(m.group(1), m.group(2)), re.match(r"\s*(m|min|minor)\b", m.group(3), re.I) is not None


def roman(chord, key):
    if NUMERAL_RE.match(chord.strip()):
        return chord.strip()
    m = CHORD_RE.match(chord.strip())
    if not m or key is None:
        return None
    tonic, minor = key
    quality = m.group(3).split("/")[0]
    numeral = (MINOR if minor else MAJOR)[(pitch(m.group(1), m.group(2)) - tonic) % 12]
    if quality.startswith("m") and not quality.startswith("maj"):
        numeral = numeral.lower()
        quality = quality[1:]
    if "dim" in quality:
        numeral = numeral.lower() + "°"
    elif "7" in quality:
        numeral += "7"
    return numeral


def progression(chords, key):
    numerals = [roman(c, key) for c in chords]
    numerals = [n for n in numerals if n]
    # Collapse repeats so "Am Am F G" and "Am F G" count as the same loop.
    numerals = [n for i, n in enumerate(numerals) if i == 0 or n != numerals[i - 1]]
    # A whole score repeats its loop many times; keep the shortest period that repeats at the start.
    for period in range(2, 9):
        if len(numerals) >= 2 * period and numerals[period:2 * period] == numerals[:period]:
            return "-".join(numerals[:period])
    return "-".join(numerals[:8])


def counts(values, n=8):
    return Counter(v for v in values if v).most_common(n)


def as_list(value):
    if isinstance(value, list):
        return [str(v).strip().lower() for v in value]
    if isinstance(value, str):
        return [v.strip().lower() for v in re.split(r"[,;/]", value) if v.strip()]
    return []


def build(profiles):
    bpms = [float(p["bpm"]) for p in profiles if isinstance(p.get("bpm"), (int, float)) and p["bpm"] > 0]
    keys = [parse_key(p.get("key")) for p in profiles]
    progressions = [progression(p.get("chords") or [], k) for p, k in zip(profiles, keys)]
    return {
        "tracks": len(profiles),
        "bpm": {"median": statistics.median(bpms), "min": min(bpms), "max": max(bpms), "values": bpms} if bpms else None,
        "keys": counts(p.get("key") for p in profiles),
        "minor_share": round(sum(1 for k in keys if k and k[1]) / max(1, sum(1 for k in keys if k)), 2),
        "progressions": counts(progressions),
        "time_signatures": counts(p.get("time_signature") for p in profiles),
        "instruments": counts((i for p in profiles for i in as_list(p.get("instruments"))), 15),
        "vocals": counts(", ".join(map(str, v.values())) if isinstance(v, dict) else str(v) for v in (p.get("vocals", "") for p in profiles)),
        "moods": counts((m for p in profiles for m in as_list(p.get("mood"))), 12),
        "production": [p.get("production") for p in profiles if p.get("production")],
        "hooks": [p.get("hook") for p in profiles if p.get("hook")],
        "themes": counts((t for p in profiles for t in as_list(p.get("themes"))), 15),
        "languages": counts(str(p.get("language", "")).lower() for p in profiles),
    }


def lyric_match(sung, intended):
    """Share of the intended lyric words heard, in order, in the transcription (1.0 = every word)."""
    strip = lambda t: norm(re.sub(r"\[[^\]]*\]", " ", t)).split()
    intended, sung = strip(intended), strip(sung)
    matched = sum(b.size for b in SequenceMatcher(None, intended, sung, autojunk=False).get_matching_blocks())
    return round(matched / max(1, len(intended)), 3)


def lyric_dna(stats):
    """Aggregate the per-song lyric stats from hitmaker.lyrics."""
    if not stats:
        return {"songs": 0}
    numeric = ["duration", "lines", "words_per_minute", "first_vocal", "hook_repeats", "hook_first_at", "first_repeat_at",
               "repeated_line_share", "vocable_line_share", "syllables_per_line", "english_word_share"]
    result = {"songs": len(stats)}
    for key in numeric:
        values = [s[key] for s in stats if s.get(key) is not None]
        result[key] = statistics.median(values) if values else None
    result["rhymes"] = counts((r for s in stats for r in s["rhymes"]), 6)
    result["hooks"] = [s["hook"] for s in stats]
    return result


def abc_harmony(abc):
    """Key, chord symbols in order and tempo from an ABC score."""
    key = re.search(r"^K:\s*(\S+)", abc, re.M)
    tempo = re.search(r"^Q:\s*(?:\d+/\d+=)?(\d+)", abc, re.M)
    return key.group(1) if key else None, re.findall(r'"([A-G][^"]*)"', abc), int(tempo.group(1)) if tempo else None


def abc_first_vocal(abc):
    """Seconds before the first sung note, counted in bars of the Vocal voice."""
    meter = re.search(r"^M:\s*(\d+)/(\d+)", abc, re.M)
    _, _, tempo = abc_harmony(abc)
    if not meter or not tempo:
        return None
    bar_seconds = int(meter.group(1)) * 4 / int(meter.group(2)) * 60 / tempo
    voice, bars = None, 0
    for line in abc.splitlines():
        if line.startswith("V:"):
            voice = line[2:].split()[0]
            continue
        if voice != "Vocal" or not line.strip() or line[0] in "%XTMLQK":
            continue
        for bar in filter(None, (b.strip() for b in line.split("|"))):
            notes = re.sub(r'"[^"]*"', "", bar)
            rest = re.fullmatch(r"Z(\d*)", notes)
            if rest:
                bars += int(rest.group(1) or 1)
            elif re.search(r"[A-Ga-g]", notes):
                return round(bars * bar_seconds, 1)
            else:
                bars += 1
    return None
