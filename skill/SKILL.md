---
name: hit-maker
description: Write and produce an original song that fits what is charting now in a music genre, or help an author turn their own demo into a hit. Scouts the genre's top tracks and full lyrics, profiles them with MOSS-Music, writes lyrics and a chord-annotated ABC plan, renders takes with YuE2 in ComfyUI (or prepares a Suno prompt) and ranks them by listening back. Use when asked for a hit, a song in the style of a genre's current charts, to improve the user's own song, or to run Hit Maker.
---

# Hit Maker

The ComfyUI-HitMaker pack drives everything through ComfyUI's API with one CLI:

```bash
cd ComfyUI/custom_nodes/ComfyUI-HitMaker
python hit_maker.py <command> ...
```

Use the same Python environment that runs ComfyUI.

Every step reads and writes `ComfyUI/output/hitmaker/<run>/`. Read the JSON between steps; you are the songwriter and producer, the CLI only does the I/O.

Requirements: ComfyUI running at `http://127.0.0.1:8188` for `plan` and `render` (`listen` and `critique` run MOSS-Music in the CLI process and only ask ComfyUI to free the GPU), `yue2_3b_int8_convrot.safetensors` in checkpoints and `MOSS-Music-8B-Thinking` in `models/LLM`. Check `nvidia-smi` first: YuE2 and MOSS-Music each need most of a 16 GB card, so stop if another job holds the GPU and tell the user.

## 0. Brief

Ask for whatever the user has not given: genre, language, market (Deezer charts follow the server's region; Apple needs a country code such as `mx`, `us`, `es`), vocal type, theme or story, and how many takes. Offer defaults they can accept in one word.

## 1. Scout and listen

```bash
python hit_maker.py scout <run> "<genre>" --top 10 [--source apple --country mx]
python hit_maker.py listen <run>          # ~1–2 min per track, run in the background
python hit_maker.py dna <run>
```

`scout` downloads the official 30 s previews into `refs/` and the full synced lyrics from LRCLIB as `.lrc` files beside them (`hit_maker.py lyrics <run>` backfills an older run). `listen` asks MOSS-Music for a JSON profile of each preview, and transcribes the preview only when no lyrics were found. `dna` aggregates BPM, keys, chord progressions as Roman numerals, instruments, vocals, moods, hooks and themes into `dna.json`.

`dna.json` → `lyrics` describes the whole songs, not just the previews: median duration, lines, words per minute, when the first vocal and the hook land, how often the hook repeats, the share of repeated lines, syllables per line, rhyme vowels and English share. Use it as the song's blueprint: land the hook near `hook_first_at`, repeat the hook about `hook_repeats` times, keep lines near `syllables_per_line`, and match the density (`words_per_minute`). Read the `.lrc` files for vocabulary, imagery and point of view, but never reuse their lines, titles or hooks.

Read `dna.json` and the per-track `reasoning`. Previews are 30 s, so song length and structure come from the lyrics and the Deezer/Apple durations, not from the audio. Treat any single profile with doubt: drop outliers (half/double-time BPM, a key that contradicts the chords) before you draw conclusions, and tell the user what the charts have in common in a few lines.

## 2. Write the song

Write `songs/<song>/song.json` in the run folder:

```json
{"title": "...", "style": "...", "lyrics": "...", "mode": "full", "max_duration": 180}
```

- **Originality first.** Take only abstract traits from the DNA: tempo range, mode, a progression family, instrumentation, vocal treatment, hook devices, themes. Never reuse a reference's melody, lyric lines, title or artist name, and never use YuE2's cover path on charting songs.
- **style:** `Language, genre, vocal character, BPM, key, instruments, production` in one line, in English, for example `Spanish, reggaeton, romantic male vocal with light autotune, 94 BPM, A minor, dembow drums, 808 bass, plucked synth, polished modern production`.
- **lyrics:** only the sung words, with `[Verse]`, `[Pre-Chorus]`, `[Chorus]`, `[Bridge]`, `[Outro]` tags. Put the hook in the first chorus line and repeat the chorus word for word. Keep lines singable: similar syllable counts within a section, stressed vowels on the rhyme.

Show the user the title, the style line and the lyrics before rendering, and adjust on their notes.

## 3. Plan the music (ABC)

```bash
python hit_maker.py plan <run> <song> --seeds 1,2,3
```

Each `plan_s<seed>.abc` is YuE2's melody with chord symbols. Read them, pick the one whose chords follow the DNA's progression family and whose chorus melody sits highest, and edit it if needed: change chord symbols (`"Am"`, `"F"`…), transpose to the target key, or lift the chorus. Keep note counts per lyric line intact when you edit chords, and save your edit as a new file such as `plan_final.abc`.

## 4. Render and critique

```bash
python hit_maker.py render <run> <song> --abc <plan.abc> --seeds 11,12,13,14   # minutes per take, run in the background
python hit_maker.py critique <run> <song>
```

Always send the user the takes to listen to before acting on the numbers below; they choose by ear, and a blurry vocal can be the point in genres such as bedroom pop or lo-fi. Ask before spending a render on a metric-driven fix. MOSS-Music's key is often off (a semitone or more): check key, chords and melody of a take with SheetSage2 (`SheetSage2AudioToABC`) instead.

`critique` listens to every take and prints a ranking: `lyrics` is the share of intended words MOSS-Music heard (intelligibility), `bpm Δ` is the distance from the charts' median tempo, plus the key and chord loop it heard. Use it to shortlist, not as the verdict:

- Below ~0.6 lyric match, the vocal is mumbling: simplify the lines or slow the style's BPM, then render again.
- If the heard chords drift from the plan, render more seeds from the same ABC before editing it.

## Author mode: turn the user's own song into a hit

Use it when the user already wrote a song and wants help, not a new one. The song stays theirs: you are a hit doctor and producer, not a ghostwriter.

1. **Brief.** Ask for a demo recording (phone voice + guitar is fine), the lyrics as text with `[Verse]`/`[Chorus]` tags if they have them, the target genre and market, and what must not change (a line they love, the melody, the story). If there is no run for that genre yet, `scout` + `listen` + `dna` it first.
2. **Bring it in:**
   ```bash
   python hit_maker.py author <run> <song> demo.m4a letra.txt --title "..."
   python hit_maker.py diagnose <run> <song>
   ```
   `author` copies the demo, transcribes its melody and chords with SheetSage2 into `author.abc`, profiles it with MOSS-Music and measures the lyrics. Lyric times are estimates spread over the demo's length. `diagnose` prints the gaps against `dna.json` (tempo, duration, time to first vocal, when the hook first lands, first repeated line, hook repeats, repetition, syllables per line, words per minute, key, mode and chord loop) and saves `diagnosis.json`.
3. **Hit report.** Read `author.abc`, the profile and the diagnosis, then give the user a short report in this order:
   - **What already works**, specific (a strong image, a melodic leap in the chorus, a progression that fits).
   - **The 3–5 changes with the biggest payoff**, each with the gap it closes, for example: hook first lands at 34 s vs 47 s in the genre, but the demo has an 11 s intro vs 2 s → start with the vocal or a hook teaser; hook sung twice vs 6 → open and close the chorus with it and add a post-chorus; 14 syllables per line vs 11 → trim.
   - **Production direction**: tempo, key (transpose if their range allows), drum pattern, instruments, vocal treatment.
   Ask which changes they accept. Do not rewrite everything at once.
4. **Edit, keeping their voice.** Write `song.json` from their lyrics with only the accepted changes; keep their words wherever the change does not need new ones, and mark new lines in your message so they can veto them. Edit `author.abc` the same way into `plan_author.abc`: keep their melody notes, change `Q:` (tempo), transpose `K:`, reharmonise chord symbols toward the genre's loop, move or repeat sections. When lines change, keep each line's syllable count close to the notes it sits on.
5. **Produce and compare.** `render` with `--abc plan_author.abc` (melody and chords from their song) or, if they want the melody untouched but a fresh arrangement, set `"mode": "melody"` in `song.json` and strip chords from the ABC. Then `critique`, and play them the before (`author_demo.*`) and after side by side. The Suno vertical works here too: Cover their demo with the new style and lyrics.

## Suno vertical

Use it when the user asks for Suno, or offer it when they want the most polished production. Suno has no local model and no official API: you write the prompt, the user generates by hand and gives the files back.

1. Add a `suno` block to `song.json` (every key optional; missing ones fall back to the song's own fields):

   ```json
   "suno": {"style": "...", "exclude": "...", "lyrics": "...", "vocal_gender": "male", "weirdness": 45, "style_influence": 65}
   ```

   - **style** (≤ 1000 chars): Suno reads it as a producer's brief, so be richer than the YuE2 line: genre and era, BPM and key, the DNA's progression family in words ("i–VI–III–VII minor loop"), drum pattern, bass, lead sounds, vocal timbre and processing, mix and energy curve.
   - **exclude**: what Suno tends to add that the DNA rules out (for example `rock guitars, trap hi-hat rolls, female vocals`).
   - **lyrics** (≤ 5000 chars): the same words, plus Suno metatags it understands: `[Intro]`, `[Pre-Chorus]`, `[Hook]`, `[Drop]`, `[Instrumental Break]`, `[Outro]`, short performance cues inside a tag (`[Chorus: full, crowd ad-libs]`) and ad-libs in parentheses. Keep the hook line identical in every chorus.
2. `python hit_maker.py suno <run> <song>` validates the lengths and writes `songs/<song>/suno_prompt.md` with one copy-paste block per Suno Custom-mode field. Show it to the user. If a YuE2 take was critiqued, the file also suggests uploading the best one to Suno's **Cover** so Suno keeps the planned melody and chords.
3. When the user sends the downloaded songs: `python hit_maker.py import <run> <song> <files...>`, then `critique`. Suno and YuE2 takes are ranked together; Suno's lyric match is measured against the `suno.lyrics`.

## 5. Deliver

Give the user the best one or two takes (`songs/<song>/take_s*.flac`), the final ABC (melody + chords), the lyrics, and a short arrangement note per section taken from the plan and the critique. Mention what you could not verify, such as pronunciation or whether the hook lands, and that YuE2 weights are CC BY-NC 4.0 (non-commercial).
