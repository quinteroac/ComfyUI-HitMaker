"""Hit Maker: study what is charting in a genre, then write and produce an original song with YuE2.

Every step reads and writes output/hitmaker/<run>/, so an agent can inspect and edit between steps.
"""
import argparse
import json
import os
import shutil
import time

from hitmaker import dna, lyrics, scout
from hitmaker.comfy import INPUT_DIR, OUTPUT_DIR, ROOT, Comfy, output_file, stage_input
from moss import runner

PROFILE_QUESTION = """Analyze this track as a music producer. Answer with one JSON object only, using these keys:
"genre", "subgenre", "bpm" (number), "key" (for example "A minor"), "time_signature",
"chords" (chord names, not Roman numerals: the main loop once, at most 8, for example ["Am", "F", "C", "G"]),
"structure" (list of {"section", "start", "end"} in seconds), "instruments" (list),
"vocals" (gender, register, style and effects), "mood" (list),
"production" (one sentence on mix, drums, bass and effects), "hook" (what makes the catchiest part memorable),
"language", "themes" (list of lyric topics)."""
LYRICS_QUESTION = "Transcribe the lyrics of this song."


def run_dir(run):
    return os.path.join(OUTPUT_DIR, "hitmaker", run)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


class Listener:
    """MOSS-Music in this process: bitsandbytes int8 segfaults inside the ComfyUI server, so ComfyUI only frees the GPU."""

    def __init__(self, args):
        Comfy(args.server).free()
        path = args.model if os.path.isdir(args.model) else os.path.join(ROOT, "models", "LLM", args.model)
        self.model, self.processor = runner.load(path, args.precision)
        self.max_new_tokens = args.max_new_tokens

    def profile(self, path, transcribe=True):
        wav = runner.read_audio(path)
        # The Thinking model can loop on repeated choruses or long chord lists; the penalties keep it out of loops without dropping real repeats.
        thinking, answer = runner.ask(self.model, self.processor, wav, PROFILE_QUESTION, self.max_new_tokens, repetition_penalty=1.05)
        if dna.extract_json(answer) is None:
            # A stronger penalty breaks the reasoning trace, so retry with another seed instead.
            thinking, answer = runner.ask(self.model, self.processor, wav, PROFILE_QUESTION, self.max_new_tokens, repetition_penalty=1.05, seed=1)
        profile = dna.extract_json(answer) or {}
        result = {"profile": profile, "profile_raw": answer if not profile else None, "reasoning": thinking}
        if transcribe:
            # Past the sung words the transcription drifts into invented lines, so budget it by duration.
            lyric_tokens = 300 + 6 * len(wav) // runner.SAMPLE_RATE
            result["lyrics"] = runner.ask(self.model, self.processor, wav, LYRICS_QUESTION, lyric_tokens, repetition_penalty=1.1)[1]
        return result


def cmd_scout(args):
    folder = run_dir(args.run)
    if args.source == "deezer":
        tracks = scout.deezer_top(args.genre, args.top)
    elif args.source == "deezer-playlist":
        tracks = scout.deezer_playlist(args.genre, args.top)
    else:
        tracks = scout.apple_top(args.genre, args.top, args.country)
    tracks = lyrics.attach(scout.download_previews(tracks, os.path.join(folder, "refs")))
    save_json(os.path.join(folder, "run.json"), {"genre": args.genre, "source": args.source, "date": time.strftime("%Y-%m-%d"), "tracks": tracks})
    for t in tracks:
        print(f"{t['rank']:>3}  {t['artist']} - {t['title']}{'' if t['lyrics_file'] else '  (no lyrics)'}")
    print(f"\n{len(tracks)} previews in {folder}/refs")


def cmd_lyrics(args):
    path = os.path.join(run_dir(args.run), "run.json")
    data = load_json(path)
    lyrics.attach(data["tracks"])
    save_json(path, data)
    for t in data["tracks"]:
        st = t.get("lyrics_stats")
        print(f"{t['rank']:>3}  {t['artist']} - {t['title']}: " + (f"{st['lines']} lines, {st['words_per_minute']} wpm, hook x{st['hook_repeats']} at {st['hook_first_at']} s" if st else "no lyrics"))


def cmd_listen(args):
    path = os.path.join(run_dir(args.run), "run.json")
    data = load_json(path)
    todo = [t for t in data["tracks"] if args.force or not t.get("profile")]
    listener = Listener(args) if todo else None
    for t in todo:
        print(f"Listening: {t['artist']} - {t['title']}", flush=True)
        # Scouted lyrics beat a transcription of a 30 s preview, so only transcribe when there are none.
        t.update(listener.profile(t["file"], transcribe=not t.get("lyrics_file")))
        save_json(path, data)
        print(json.dumps(t["profile"] or t["profile_raw"], ensure_ascii=False), flush=True)


def cmd_dna(args):
    folder = run_dir(args.run)
    data = load_json(os.path.join(folder, "run.json"))
    profiles = [t["profile"] for t in data["tracks"] if t.get("profile")]
    result = dna.build(profiles)
    result["genre"] = data["genre"]
    result["lyrics"] = dna.lyric_dna([t["lyrics_stats"] for t in data["tracks"] if t.get("lyrics_stats")])
    result["lyrics"]["files"] = {f"{t['artist']} - {t['title']}": t.get("lyrics_file") or t.get("lyrics", "") for t in data["tracks"]}
    save_json(os.path.join(folder, "dna.json"), result)
    print(json.dumps({k: v for k, v in result.items() if k != "lyrics"}, indent=2, ensure_ascii=False))


def song_path(args):
    return os.path.join(run_dir(args.run), "songs", args.song, "song.json")


def yue2_loader(args):
    return {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": args.ckpt}}


def cmd_plan(args):
    song = load_json(song_path(args))
    comfy = Comfy(args.server)
    for seed in args.seeds:
        graph = {
            "1": yue2_loader(args),
            "2": {"class_type": "YuE2GenerateABC", "inputs": {
                "clip": ["1", 1], "style": song["style"], "lyrics": song["lyrics"], "seed": seed, "mode": song.get("mode", "full"),
                "max_abc_tokens": 8192, "temperature": 0.7, "top_p": 0.9, "top_k": 30, "repetition_penalty": 1.005, "penalty_window": 100}},
            "3": {"class_type": "PreviewAny", "inputs": {"source": ["2", 0]}},
        }
        abc = comfy.run(graph)["3"]["text"][0]
        path = os.path.join(os.path.dirname(song_path(args)), f"plan_s{seed}.abc")
        with open(path, "w") as f:
            f.write(abc)
        print(path)


def cmd_author(args):
    folder = os.path.dirname(song_path(args))
    os.makedirs(folder, exist_ok=True)
    demo = os.path.join(folder, "author_demo" + os.path.splitext(args.audio)[1].lower())
    shutil.copyfile(args.audio, demo)
    with open(args.lyrics) as f:
        text = f.read().strip()

    staged = stage_input(demo, f"hitmaker_{args.run}_{args.song}_demo")
    graph = {
        "1": {"class_type": "LoadAudio", "inputs": {"audio": staged}},
        "2": {"class_type": "AudioEncoderLoader", "inputs": {"audio_encoder_name": args.sheetsage}},
        "3": {"class_type": "SheetSage2AudioToABC", "inputs": {"audio_encoder": ["2", 0], "audio": ["1", 0], "mode": "full"}},
        "4": {"class_type": "PreviewAny", "inputs": {"source": ["3", 0]}},
    }
    try:
        abc = Comfy(args.server).run(graph)["4"]["text"][0]
    finally:
        os.remove(os.path.join(INPUT_DIR, staged))
    abc_path = os.path.join(folder, "author.abc")
    with open(abc_path, "w") as f:
        f.write(abc)

    duration = len(runner.read_audio(demo)) / runner.SAMPLE_RATE
    heard = Listener(args).profile(demo, transcribe=False)
    song = {"title": args.title or args.song, "style": "", "lyrics": text, "mode": "full",
            "author": {"demo": demo, "abc": abc_path, "duration": round(duration, 1), "lyric_stats": lyrics.text_stats(text, duration, dna.abc_first_vocal(abc) or 0.0), **heard}}
    save_json(song_path(args), song)
    key, chords, tempo = dna.abc_harmony(abc)
    print(f"{abc_path}\nkey {key}, {tempo or '?'} BPM, chords {' '.join(chords[:8])}")
    print(json.dumps(heard["profile"] or heard["profile_raw"], ensure_ascii=False))


def cmd_diagnose(args):
    song = load_json(song_path(args))
    author = song["author"]
    target = load_json(os.path.join(run_dir(args.run), "dna.json"))
    with open(author["abc"]) as f:
        key, chords, tempo = dna.abc_harmony(f.read())
    mine, genre = author["lyric_stats"], target["lyrics"]
    rows = [
        ("tempo (BPM)", tempo or (author.get("profile") or {}).get("bpm"), target["bpm"]["median"] if target.get("bpm") else None),
        ("duration (s)", author["duration"], genre.get("duration")),
        ("first vocal (s)", mine["first_vocal"], genre.get("first_vocal")),
        ("hook first at (s)", mine["hook_first_at"], genre.get("hook_first_at")),
        ("first repeated line (s)", mine["first_repeat_at"], genre.get("first_repeat_at")),
        ("hook repeats", mine["hook_repeats"], genre.get("hook_repeats")),
        ("repeated line share", mine["repeated_line_share"], genre.get("repeated_line_share")),
        ("syllables per line", mine["syllables_per_line"], genre.get("syllables_per_line")),
        ("words per minute", mine["words_per_minute"], genre.get("words_per_minute")),
    ]
    table = []
    print(f"{'':<26}{'yours':>9}{'genre':>9}")
    for name, value, median in rows:
        verdict = "?" if value is None or not median else "ok" if abs(value - median) <= 0.15 * median else "higher" if value > median else "lower"
        value, median = (round(v, 1) if isinstance(v, float) else v for v in (value, median))
        table.append({"metric": name, "yours": value, "genre": median, "verdict": verdict})
        print(f"{name:<26}{str(value):>9}{str(median):>9}  {verdict}")

    progression = dna.progression(chords, dna.parse_key(key))
    genre_progressions = [p for p, _ in target.get("progressions", [])]
    minor = (dna.parse_key(key) or (None, None))[1]
    result = {"table": table, "key": key, "minor": minor, "genre_minor_share": target.get("minor_share"),
              "progression": progression, "genre_progressions": genre_progressions,
              "hook": mine["hook"], "genre_hooks": genre.get("hooks", []), "sections": mine.get("sections", [])}
    save_json(os.path.join(os.path.dirname(song_path(args)), "diagnosis.json"), result)
    print(f"\nkey {key} ({'minor' if minor else 'major'}; genre minor share {target.get('minor_share')})")
    print(f"progression {progression}\ngenre progressions {', '.join(genre_progressions)}")
    print(f"hook \"{mine['hook']}\" x{mine['hook_repeats']}, sections {' '.join(mine.get('sections', []))}")


def cmd_render(args):
    path = song_path(args)
    song = load_json(path)
    abc = ""
    if args.abc:
        with open(args.abc) as f:
            abc = f.read()
    comfy = Comfy(args.server)
    for seed in args.seeds:
        graph = {
            "1": yue2_loader(args),
            "2": {"class_type": "YuE2GenerateMusic", "inputs": {
                "clip": ["1", 1], "style": song["style"], "lyrics": song["lyrics"], "abc": abc, "seed": seed,
                "mode": song.get("mode", "full"), "max_duration": song.get("max_duration", 240),
                "temperature": 1.0, "top_p": 0.95, "top_k": 100, "repetition_penalty": 1.2}},
            "3": {"class_type": "EmptyYuE2LatentAudio", "inputs": {"seconds": ["2", 1], "batch_size": 1}},
            "4": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["2", 0]}},
            "5": {"class_type": "KSampler", "inputs": {
                "model": ["1", 0], "positive": ["2", 0], "negative": ["4", 0], "latent_image": ["3", 0],
                "seed": seed, "steps": 32, "cfg": 1.0, "sampler_name": "dpm_2", "scheduler": "sgm_uniform", "denoise": 1.0}},
            "6": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
            "7": {"class_type": "SaveAudio", "inputs": {"audio": ["6", 0], "filename_prefix": f"hitmaker/{args.run}/songs/{args.song}/take_s{seed}"}},
        }
        start = time.time()
        file = output_file(comfy.run(graph)["7"]["audio"][0])
        song.setdefault("takes", []).append({"file": file, "seed": seed, "abc": args.abc, "seconds": round(time.time() - start)})
        save_json(path, song)
        print(f"{file}  ({round(time.time() - start)} s)", flush=True)


SUNO_LIMITS = {"title": 100, "style": 1000, "exclude": 1000, "lyrics": 5000}


def cmd_suno(args):
    path = song_path(args)
    song = load_json(path)
    suno = song.get("suno") or {}
    fields = {
        "title": song["title"],
        "style": suno.get("style", song["style"]),
        "exclude": suno.get("exclude", ""),
        "lyrics": suno.get("lyrics", song["lyrics"]),
    }
    over = [f"{k} has {len(v)} characters (Suno allows {SUNO_LIMITS[k]})" for k, v in fields.items() if len(v) > SUNO_LIMITS[k]]
    if over:
        raise SystemExit("Shorten the Suno prompt: " + "; ".join(over))

    lines = [f"# {song['title']} — Suno (Custom mode)", ""]
    for key, label in [("title", "Song Title"), ("style", "Style of Music"), ("exclude", "Exclude Styles"), ("lyrics", "Lyrics")]:
        if fields[key]:
            lines += [f"## {label} ({len(fields[key])}/{SUNO_LIMITS[key]})", "", "```", fields[key], "```", ""]
    lines += ["## Advanced", "",
              f"- Model: {suno.get('model', 'latest (v5 or newer)')}",
              f"- Vocal gender: {suno.get('vocal_gender', 'as in the style')}",
              f"- Weirdness: {suno.get('weirdness', 50)} %",
              f"- Style influence: {suno.get('style_influence', 60)} %", ""]
    takes = [t for t in song.get("takes", []) if t.get("engine", "yue2") == "yue2" and t.get("critique")]
    if takes:
        best = max(takes, key=lambda t: t["critique"]["lyric_match"])
        lines += ["## Optional: keep the YuE2 melody and chords", "",
                  f"Upload `{best['file']}` and use **Cover** with the same style and lyrics to re-produce the planned song in Suno.", ""]
    lines += ["Generate in Suno, download the takes you like, then:", "", "```bash",
              f"python hit_maker.py import {args.run} {args.song} <file> [<file> ...]", "```"]
    out = os.path.join(os.path.dirname(path), "suno_prompt.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n{out}")


def cmd_import(args):
    path = song_path(args)
    song = load_json(path)
    folder = os.path.dirname(path)
    takes = song.setdefault("takes", [])
    for src in args.files:
        n = sum(1 for t in takes if t.get("engine") == args.engine) + 1
        dest = os.path.join(folder, f"{args.engine}_{n:02d}{os.path.splitext(src)[1].lower()}")
        shutil.copyfile(src, dest)
        takes.append({"file": dest, "engine": args.engine, "source": os.path.abspath(src)})
        print(dest)
    save_json(path, song)


def cmd_critique(args):
    path = song_path(args)
    song = load_json(path)
    target = load_json(os.path.join(run_dir(args.run), "dna.json"))
    target_key = dna.parse_key(target["keys"][0][0]) if target.get("keys") else None
    todo = [t for t in song.get("takes", []) if args.force or "critique" not in t]
    listener = Listener(args) if todo else None
    for take in todo:
        print(f"Critiquing {os.path.basename(take['file'])}", flush=True)
        heard = listener.profile(take["file"])
        p = heard["profile"]
        bpm = p.get("bpm") if isinstance(p.get("bpm"), (int, float)) else None
        intended = song["suno"]["lyrics"] if take.get("engine") == "suno" and "suno" in song else song["lyrics"]
        heard["lyric_match"] = dna.lyric_match(heard["lyrics"], intended)
        heard["bpm_delta"] = round(bpm - target["bpm"]["median"], 1) if bpm and target.get("bpm") else None
        heard["mode_matches"] = (dna.parse_key(p.get("key")) or (None, None))[1] == target_key[1] if target_key else None
        take["critique"] = heard
        save_json(path, song)

    print(f"\n{'take':<24}{'lyrics':>8}{'bpm Δ':>8}  key / progression")
    for take in sorted(song.get("takes", []), key=lambda t: -(t.get("critique") or {}).get("lyric_match", 0)):
        c = take.get("critique") or {}
        p = c.get("profile") or {}
        print(f"{os.path.basename(take['file']):<24}{c.get('lyric_match', '-'):>8}{str(c.get('bpm_delta', '-')):>8}  {p.get('key', '?')} / {' '.join(p.get('chords') or [])}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--model", default="MOSS-Music-8B-Thinking", help="Folder name in models/LLM, or a path")
    parser.add_argument("--precision", default="int8", choices=["int8", "nf4", "bf16"])
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--ckpt", default="yue2_3b_int8_convrot.safetensors")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("scout", help="Download 30 s previews of the current top tracks of a genre")
    p.add_argument("run")
    p.add_argument("genre")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--source", default="deezer", choices=["deezer", "deezer-playlist", "apple"],
                   help="deezer-playlist: for genres without a chart, the genre is a playlist search such as \"bedroom pop latino\"")
    p.add_argument("--country", default="mx", help="Apple Music storefront")
    p.set_defaults(fn=cmd_scout)

    p = sub.add_parser("lyrics", help="Fetch full synced lyrics from LRCLIB for a scouted run (scout already does this)")
    p.add_argument("run")
    p.set_defaults(fn=cmd_lyrics)

    p = sub.add_parser("listen", help="Profile every reference track with MOSS-Music")
    p.add_argument("run")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_listen)

    p = sub.add_parser("dna", help="Aggregate the profiles into dna.json")
    p.add_argument("run")
    p.set_defaults(fn=cmd_dna)

    for name, fn, help_text in [("plan", cmd_plan, "Let YuE2 write ABC plans (melody + chords) for songs/<song>/song.json"),
                                ("render", cmd_render, "Render takes from an ABC plan, or without one"),
                                ("critique", cmd_critique, "Listen to every take and rank it against dna.json")]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("run")
        p.add_argument("song")
        if name != "critique":
            p.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")], default=[1])
        if name == "render":
            p.add_argument("--abc", help="ABC plan file; omit to render without a plan")
        if name == "critique":
            p.add_argument("--force", action="store_true")
        p.set_defaults(fn=fn)

    p = sub.add_parser("author", help="Bring in your own song: demo audio + lyrics; transcribes melody and chords and profiles the demo")
    p.add_argument("run")
    p.add_argument("song")
    p.add_argument("audio", help="Demo recording (any quality)")
    p.add_argument("lyrics", help="Text file with the lyrics, ideally with [Verse]/[Chorus] tags")
    p.add_argument("--title")
    p.add_argument("--sheetsage", default="sheetsage2_bf16.safetensors")
    p.set_defaults(fn=cmd_author)

    p = sub.add_parser("diagnose", help="Compare an author song with the run's dna.json")
    p.add_argument("run")
    p.add_argument("song")
    p.set_defaults(fn=cmd_diagnose)

    p = sub.add_parser("suno", help="Write suno_prompt.md for generating the song by hand in Suno")
    p.add_argument("run")
    p.add_argument("song")
    p.set_defaults(fn=cmd_suno)

    p = sub.add_parser("import", help="Add songs generated elsewhere (Suno) as takes for critique")
    p.add_argument("run")
    p.add_argument("song")
    p.add_argument("files", nargs="+")
    p.add_argument("--engine", default="suno")
    p.set_defaults(fn=cmd_import)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
