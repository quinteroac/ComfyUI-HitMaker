# ComfyUI-HitMaker

An agent toolkit for writing songs that fit what is charting now, running locally on a consumer GPU (tested on a 16 GB RTX 5060 Ti).

It studies a genre's current top tracks (audio previews and full synced lyrics), builds a "DNA" of what they share, then writes and produces an original song with **YuE2** in ComfyUI, or prepares a prompt for **Suno**. In author mode it takes your own demo and lyrics, compares them with the genre and helps you turn them into a stronger song while keeping them yours.

The songwriting decisions are made by an AI agent (the included [Claude Code](https://claude.com/claude-code) skill); the CLI does the I/O and runs the models.

## What it does

| Command | |
|---|---|
| `scout <run> <genre>` | Top tracks of a genre from Deezer charts, a Deezer playlist search (`--source deezer-playlist`, for genres without a chart) or Apple Music (`--source apple --country mx`). Downloads the official 30 s previews and the full synced lyrics from [LRCLIB](https://lrclib.net). |
| `lyrics <run>` | Fetches lyrics for a run scouted before lyrics support. |
| `listen <run>` | Profiles every preview with MOSS-Music: genre, BPM, key, chords, instruments, vocals, mood, production, hook. |
| `dna <run>` | Aggregates the profiles and the lyrics into `dna.json`: tempo, mode, chord loops as Roman numerals, sound, and the lyric blueprint (when the hook lands, how often it repeats, syllables per line, words per minute, rhymes). |
| `plan <run> <song>` | YuE2 writes ABC scores (melody + chords) for `songs/<song>/song.json`. |
| `render <run> <song> --abc plan.abc --seeds 1,2` | YuE2 renders full songs from an (edited) ABC score. |
| `critique <run> <song>` | Listens to every take and ranks it: share of the intended lyrics heard, tempo against the genre, key and chords. |
| `suno <run> <song>` / `import <run> <song> files...` | Writes a copy-paste prompt for Suno's Custom mode, and brings the songs you generated there back as takes. |
| `author <run> <song> demo.m4a lyrics.txt` | Author mode: transcribes your demo's melody and chords with SheetSage2, profiles it and measures your lyrics. |
| `diagnose <run> <song>` | Compares your song with `dna.json` and prints the gaps. |

Everything is written to `ComfyUI/output/hitmaker/<run>/` so you, or the agent, can read and edit it between steps.

The pack also adds a **MOSS-Music Listen** node (`HitMaker` category): ask any question about an `AUDIO` input.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/quinteroac/ComfyUI-HitMaker
pip install -r ComfyUI-HitMaker/requirements.txt
```

Use a recent ComfyUI: YuE2 and SheetSage2 are native nodes there.

### Models

| Model | Where | Size |
|---|---|---|
| [MOSS-Music-8B-Thinking](https://huggingface.co/OpenMOSS-Team/MOSS-Music-8B-Thinking) (whole repo folder) | `models/LLM/MOSS-Music-8B-Thinking/` | 18 GB, loaded in int8 (~11 GB VRAM) |
| [yue2_3b_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/YuE2) | `models/checkpoints/` | 3.7 GB |
| [sheetsage2_bf16.safetensors](https://huggingface.co/Comfy-Org/YuE2) (author mode) | `models/audio_encoders/` | 1.3 GB |

```bash
hf download OpenMOSS-Team/MOSS-Music-8B-Thinking --local-dir ComfyUI/models/LLM/MOSS-Music-8B-Thinking
```

### Agent skill

```bash
ln -s "$PWD/ComfyUI-HitMaker/skill" ~/.claude/skills/hit-maker
```

Then ask Claude Code for a hit ("Hit Maker: a romantic reggaeton for Mexico") or to improve your song ("here is my demo and lyrics, help me make it a hit").

## Notes

- **MOSS-Music runs out of the ComfyUI process.** bitsandbytes' int8 matmul segfaults inside the ComfyUI server, so the node uses a worker process (`moss/runner.py`) and the CLI loads the model in its own process after asking ComfyUI to free the GPU. `moss/` vendors the upstream modeling code (it encodes audio in 4 s windows like training did) with small patches for transformers 5.
- **Trust SheetSage2 for key, chords and melody.** MOSS-Music's key estimates are often off by a semitone or more; it is good at sound, mood, production and lyrics.
- **Previews are 30 s.** Song structure and length come from the full lyrics and the store durations.
- **Originality.** The agent takes only abstract traits from the charts (tempo, mode, progression family, sound, structure, themes) and never reuses melodies, lyric lines, titles or hooks. Lyrics and previews are downloaded for local analysis only; respect the terms of Deezer, Apple and LRCLIB.
- No telemetry. Network access is limited to the chart, preview and lyrics lookups you run.

## Licenses

The code is Apache-2.0. `moss/` is vendored from [OpenMOSS/MOSS-Music](https://github.com/OpenMOSS/MOSS-Music) (Apache-2.0). Model weights keep their own licenses: **YuE2 is CC BY-NC 4.0 (non-commercial)**; MOSS-Music is Apache-2.0. For commercial releases, use another engine (for example MiniMax Music 3 or ACE-Step 1.5, both native in ComfyUI) or Suno under its terms.
