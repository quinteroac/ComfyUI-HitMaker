import os
import re
import unicodedata

import requests


def norm(text):
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9]+", "_", norm(text)).strip("_")[:60]


def deezer_genres():
    return {g["id"]: g["name"] for g in requests.get("https://api.deezer.com/genre").json()["data"]}


def deezer_top(genre, top):
    genres = deezer_genres()
    wanted = norm(genre)
    matches = [(gid, name) for gid, name in genres.items() if wanted in norm(name) or norm(name) in wanted]
    if not matches:
        raise ValueError(f"Unknown Deezer genre '{genre}'. Available: {', '.join(genres.values())}")
    gid, name = matches[0]
    tracks = requests.get(f"https://api.deezer.com/chart/{gid}/tracks", params={"limit": top}).json()["data"]
    out = []
    for t in tracks:
        detail = requests.get(f"https://api.deezer.com/track/{t['id']}").json()
        out.append({
            "source": f"deezer:{name}",
            "rank": t["position"],
            "title": t["title"],
            "artist": t["artist"]["name"],
            "duration": t["duration"],
            "explicit": t["explicit_lyrics"],
            "deezer_bpm": detail.get("bpm") or None,
            "release_date": detail.get("release_date"),
            "preview_url": t["preview"],
        })
    return out


def deezer_playlist(query, top):
    """For genres without a chart: the best-matching Deezer playlist, preferring Deezer's own editors,
    with its tracks ordered by Deezer's popularity rank."""
    found = requests.get("https://api.deezer.com/search/playlist", params={"q": query, "limit": 25}).json()["data"]
    if not found:
        raise ValueError(f"No Deezer playlist matches '{query}'.")
    playlist = next((p for p in found if "deezer" in p["user"]["name"].lower()), found[0])
    tracks = requests.get(f"https://api.deezer.com/playlist/{playlist['id']}/tracks", params={"limit": 200}).json()["data"]
    tracks = sorted((t for t in tracks if t.get("preview")), key=lambda t: -t["rank"])[:top]
    return [{
        "source": f"deezer-playlist:{playlist['title']} ({playlist['user']['name']})",
        "rank": i + 1,
        "title": t["title"],
        "artist": t["artist"]["name"],
        "duration": t["duration"],
        "explicit": t["explicit_lyrics"],
        "preview_url": t["preview"],
    } for i, t in enumerate(tracks)]


def apple_top(genre, top, country):
    feed = requests.get(f"https://rss.marketingtools.apple.com/api/v2/{country}/music/most-played/100/songs.json").json()
    wanted = norm(genre)
    out = []
    for item in feed["feed"]["results"]:
        if not any(wanted in norm(g["name"]) for g in item.get("genres", [])):
            continue
        found = requests.get("https://itunes.apple.com/lookup", params={"id": item["id"], "country": country}).json()["results"]
        if not found or not found[0].get("previewUrl"):
            continue
        out.append({
            "source": f"apple:{country}",
            "rank": len(out) + 1,
            "title": item["name"],
            "artist": item["artistName"],
            "duration": found[0].get("trackTimeMillis", 0) // 1000,
            "explicit": item.get("contentAdvisoryRating", "").lower().startswith("expl"),
            "release_date": item.get("releaseDate"),
            "preview_url": found[0]["previewUrl"],
        })
        if len(out) >= top:
            break
    if not out:
        raise ValueError(f"No songs tagged '{genre}' in the Apple Music top 100 for '{country}'.")
    return out


def download_previews(tracks, folder):
    os.makedirs(folder, exist_ok=True)
    for t in tracks:
        ext = ".m4a" if "apple" in t["source"] else ".mp3"
        path = os.path.join(folder, f"{t['rank']:02d}_{safe_name(t['artist'])}-{safe_name(t['title'])}{ext}")
        with open(path, "wb") as f:
            f.write(requests.get(t.pop("preview_url")).content)
        t["file"] = path
    return tracks
