# Audio Transcription Project

Turn audio recordings into text on your own computer. Pick from OpenAI Whisper,
Mistral Voxtral, and Cohere Transcribe models, and optionally label who said
what (speaker diarization with pyannote). Everything runs locally: after the
models download once, no audio or transcript leaves your computer.

There are two ways to transcribe:

- **Web app (Streamlit)**: upload files in your browser and download the
  results. Good if you prefer not to use the command line.
- **Command line (CLI)**: transcribe a whole folder of recordings in one go and
  save the results as CSV, JSON, Parquet, or DuckDB.

There is also a helper that goes the other way: it reads a written, fictitious
transcript aloud and saves it as an MP3, so you can try everything without real
recordings.

**Contents**

1. [Get set up](#get-set-up)
2. [Generate demo audio](#generate-demo-audio-optional)
3. [Use the web app](#use-the-web-app)
4. [Use the command line](#use-the-command-line)
5. [Reference](#reference): models, Hugging Face access, speaker diarization,
   output formats, GPU
6. [Development](#development): tests, code quality, notebooks, contributing
7. [Troubleshooting](#troubleshooting)

## Get set up

You need about 15 minutes and a few GB of free disk space (models download on
first use and are cached).

### Step 1: Install the tools

You need [git](https://git-scm.com), [uv](https://docs.astral.sh/uv/) (manages
Python and the packages), and [just](https://just.systems) (runs the short
commands used below). You do **not** need to install Python yourself: uv
installs the right version (3.12).

```bash
# Windows (PowerShell)
winget install Git.Git astral-sh.uv Casey.Just

# macOS / Linux (Homebrew)
brew install git uv just
```

Close and reopen your terminal afterwards so the new commands are found.

### Step 2: Download the project

```bash
git clone https://github.com/PovertyAction/audio-transcription.git
cd audio-transcription
```

### Step 3: Install the Python packages

```bash
uv sync
```

This creates a `.venv/` folder with Python 3.12 and every dependency. It takes a
few minutes the first time (PyTorch is large). Re-run it whenever you pull new
changes.

> Contributors: run `just venv` instead. It does the same and also installs the
> pre-commit hooks.

### Step 4: Check that it works

```bash
uv run python src/transcribe_audio.py --help
```

You should see the list of options and models. Now transcribe the sample
recordings in `audio/`:

```bash
uv run python src/transcribe_audio.py --model whisper-tiny
```

The first run downloads the model (about 40 MB for `whisper-tiny`). The results
land in `output/transcribed_audio.csv`.

### Step 5 (only for some features): Hugging Face access

Whisper and Voxtral work right away. Three features use **gated** models, which
need a free [Hugging Face](https://huggingface.co) account and a one-time
approval:

  | Feature                 | Gated model                                                                                        |
  | ----------------------- | -------------------------------------------------------------------------------------------------- |
  | Cohere Transcribe model | [CohereLabs/cohere-transcribe-03-2026](https://hf.co/CohereLabs/cohere-transcribe-03-2026)         |
  | Speaker diarization     | [pyannote/speaker-diarization-community-1](https://hf.co/pyannote/speaker-diarization-community-1) |
  | Generating demo audio   | [gradium/phonon-40m-multilingual](https://hf.co/gradium/phonon-40m-multilingual)                   |

For each one you want to use:

1. Open the model page while logged in to Hugging Face and accept the terms.

2. Create a **classic token with the "Read" role** at
   <https://hf.co/settings/tokens> (fine-grained tokens need extra settings, see
   [Hugging Face access for gated
   models](#hugging-face-access-for-gated-models)).

3. Log in once on your machine and paste the token when asked:

   ```bash
   just hf-login
   ```

Check with `uv run hf auth whoami`. You only do this once per machine.

### Optional: GPU

On Windows and Linux with an NVIDIA GPU, transcription uses the GPU
automatically; without one it runs on the CPU (slower, but works). Check with:

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```

## Generate demo audio (optional)

No recordings to try? `src/generate_audio.py` reads a written transcript aloud
with computer voices and saves an MP3. It supports English (`en`), Spanish
(`es`, Mexican voices), and French (`fr`), and runs on a normal laptop CPU.

**Before the first run:** accept the terms for
[gradium/phonon-40m-multilingual](https://hf.co/gradium/phonon-40m-multilingual)
and log in (see [Step 5](#step-5-only-for-some-features-hugging-face-access)).
The voices are licensed CC BY-NC 4.0: **non-commercial use only**. Use them for
fictitious scripts, never to imitate a real person.

### Write a transcript

Use a plain text or Markdown file with one `SPEAKER: text` turn per line:

```text
# Lines starting with # are skipped
Interviewer: How long have you been in the savings group?
Respondent: About a year. My neighbor invited me.
Interviewer: And how has saving each week gone for you?
```

- Each line that starts with a label (`Interviewer:`, `SPEAKER_01:`, ...) starts
  a new turn. Lines without a label continue the previous turn.
- Labels can be in any language or script: `ENQUÊTRICE : Bonjour` (French space
  before the colon) and `采访者：你好` (full-width colon) both work. Times
  (`10:30`) and links (`https://`) are not mistaken for labels.
- Each speaker gets their own voice, in order of appearance, alternating female
  and male. To choose the country, gender, or a specific voice, see [Choose
  voices](#choose-voices-country-and-gender).
- **One speaker or several?** The number of different labels is the number of
  speakers. A file with no labels is read by a single voice. To read a file in
  one voice even though some lines look like labels (`Note: ...`), add
  `--single-speaker`: the labels are then read aloud as plain text.

A ready-made Spanish example lives in
[`examples/entrevista_ficticia.md`](examples/entrevista_ficticia.md).

### Make the MP3

```bash
just generate-audio examples/entrevista_ficticia.md --language es
```

This writes `audio/demo/entrevista_ficticia.mp3` (one MP3 per input file, named
after it). After each file, the script says how many speakers it heard and how
to transcribe the result:

```text
Wrote audio/demo/entrevista_ficticia.mp3 (... seconds)
2 speakers (ENTREVISTADORA, ENTREVISTADO): transcribe with --diarize --num-speakers 2.
```

A one-speaker file says `1 speaker: no diarization needed when transcribing.` If
a label looks accidental, it adds a `Check:` line. This happens when text starts
without a label but has labels later, or when a label is used only once while
the other speakers talk several times (like `Note: ...` in the middle of a
monologue). Reword that line, or rerun with `--single-speaker`. See [Speaker
diarization](#speaker-diarization) for what `--diarize` does.

Other options:

```bash
# Several transcripts at once, into a different folder
just generate-audio a.txt b.txt --language fr --output-path my-demo-audio

# Read everything in one voice, even lines that look like "Label: text"
just generate-audio monologue.txt --single-speaker

# Longer pause between turns (seconds; default 0.6)
just generate-audio script.md --pause 1.0
```

Without `just`, use `uv run python src/generate_audio.py ...` with the same
arguments.

### Choose voices (country and gender)

By default each language uses one country's accent: US English (`en`), Mexican
Spanish (`es`), and French from France (`fr`). Speakers get voices in order of
appearance, alternating female and male. To use another accent or gender, pick
the voices yourself in two steps.

**1. List the voices for your language:**

```bash
just generate-audio --list-voices --language es
```

```text
NAME        COUNTRY     GENDER  AGE
Alba        es-es       female  Young
Borja       es-es       male    Young Adult
...
Regina      es-mx       female  Adult
Lalo        es-mx       male    Young Adult
...
```

`COUNTRY` is the accent: `en-us` (United States), `en-gb` (United Kingdom),
`es-mx` (Mexico), `es-es` (Spain), and `fr-fr` (France). English also has
`characters` voices (stylized, no gender listed).

**2. Give each speaker a voice by name** with `--voices SPEAKER=VOICE`, using
the speaker labels from your transcript:

```bash
# Spanish from Spain: female interviewer, male respondent
just generate-audio examples/entrevista_ficticia.md --language es --voices "ENTREVISTADORA=Alba" "ENTREVISTADO=Borja"

# British English, two male speakers
just generate-audio script.md --voices "Interviewer=Hugh" "Respondent=Alfie"
```

Speakers you don't name get the default voices. Text without labels (or any file
read with `--single-speaker`) belongs to the speaker `SPEAKER`, so pick its
voice with `--voices "SPEAKER=Alba"`.

### Transcribe it

The CLI only reads the top level of its input folder, so point it at the demo
folder. Use the speaker summary from the step above: for the two-speaker
example, add `--diarize --num-speakers 2` (diarization needs [Hugging Face
access](#step-5-only-for-some-features-hugging-face-access)); for a one-speaker
file, leave both flags out.

```bash
uv run python src/transcribe_audio.py --input-path audio/demo --language es --diarize --num-speakers 2
```

`--num-speakers` applies to every file in the folder, so keep files with
different speaker counts in separate folders (or leave it out to let pyannote
guess). In the [web app](#use-the-web-app), upload the MP3 and switch on
**Multiple speakers (diarization)** with the same number of speakers.

## Use the web app

```bash
just app
```

(Without `just`: `uv run streamlit run src/app.py`.) Your browser opens at
<http://localhost:8501>. To stop the app, press `Ctrl+C` in the terminal.

**How to transcribe:**

1. **Upload** one or more audio files (mp3, wav, flac, m4a, ogg).
2. **Pick settings** in the left sidebar:
   - **Model**: `whisper-small` is a good default. See [Models](#models).
   - **Language**: Whisper can auto-detect; Voxtral and Cohere need an explicit
     language.
   - **Max tokens** (Whisper/Voxtral): raise it if transcripts get cut off.
   - **Fast chunked mode** (Whisper): faster on long recordings, slightly less
     accurate.
   - **Multiple speakers (diarization)**: labels who said what. Needs [Hugging
     Face access](#step-5-only-for-some-features-hugging-face-access). You can
     set the number of speakers, turn on LLM refinement (needs
     [Ollama](#optional---refine-fix-diarized-wording-with-a-local-llm-ollama)),
     or point to a local copy of the model for offline use.
   - **Compute device**: Auto, CPU, or GPU. Pick CPU if a large model runs out
     of GPU memory.
3. Click **Transcribe**. The first run of each model downloads it, so it takes
   longer.
4. **Read and play back** each result on the page, and **download** them as CSV
   or JSON.

Good to know:

- Results last only for the browser session. Download them before closing the
  tab. The app never writes to the `output/` folder used by the CLI.
- Files you already transcribed with the same settings are skipped. Tick
  **Re-transcribe processed files** to redo them.
- **Clear model cache** in the sidebar frees memory if you switch between big
  models.

## Use the command line

The CLI transcribes every audio file in a folder and appends the results to one
output file:

```bash
uv run python src/transcribe_audio.py [OPTIONS]
```

With no options it reads `audio/`, uses `whisper-small` in English, and writes
`output/transcribed_audio.csv`.

### Common tasks

```bash
# Your own recordings, results in your own folder
uv run python src/transcribe_audio.py --input-path ~/recordings --output-path ~/transcripts

# Another language (Whisper can also auto-detect with --language auto)
uv run python src/transcribe_audio.py --language es

# Fast test run vs. best quality
uv run python src/transcribe_audio.py --model whisper-tiny
uv run python src/transcribe_audio.py --model whisper-large-v3-turbo

# Save as JSON / Parquet / DuckDB instead of CSV
uv run python src/transcribe_audio.py --format json

# Label speakers (gated model, see Step 5)
uv run python src/transcribe_audio.py --diarize --num-speakers 2

# Cohere model (gated; always pass a language)
uv run python src/transcribe_audio.py --model cohere --language en

# Redo files that were already transcribed
uv run python src/transcribe_audio.py --all-audio
```

### How files are processed

- Every `.mp3`, `.wav`, `.flac`, `.m4a`, and `.ogg` file directly inside
  `--input-path` is transcribed, in alphabetical order. Subfolders are not
  searched.
- Results are **appended** to `<output-path>/transcribed_audio.<format>`.
  `--output-path` is a **folder**, not a file name.
- Files already in the output file are **skipped** (matched by file name and
  size), so you can re-run the same command as new recordings arrive. Use
  `--all-audio` to redo everything.
- Recordings of any length are transcribed in full.

### All options

  | Option                    | Default                  | What it does                                                                                                        |
  | ------------------------- | ------------------------ | ------------------------------------------------------------------------------------------------------------------- |
  | `--input-path PATH`       | `./audio`                | Folder with the audio files                                                                                         |
  | `--output-path PATH`      | `./output`               | Folder for `transcribed_audio.<format>`                                                                             |
  | `--format FORMAT`         | `csv`                    | `csv`, `json`, `parquet`, or `duckdb` (see [Output formats](#output-formats))                                       |
  | `--model MODEL`           | `whisper-small`          | Model to use (see [Models](#models))                                                                                |
  | `--language CODE`         | `en`                     | Language code such as `en`, `es`, `fr`. `auto` lets Whisper detect it. Voxtral and Cohere need an explicit language |
  | `--max-new-tokens N`      | `400`                    | Maximum length of the output. Whisper's limit is 448 per 30-second segment. Ignored by Cohere                       |
  | `--chunked`               | off                      | Whisper only: faster on long recordings, may lose a few words at chunk edges                                        |
  | `--all-audio`             | off                      | Re-transcribe files that are already in the output file                                                             |
  | `--diarize`               | off                      | Label speakers (see [Speaker diarization](#speaker-diarization))                                                    |
  | `--num-speakers N`        | auto                     | Exact number of speakers, if known (improves diarization)                                                           |
  | `--refine`                | off                      | Fix the wording of a diarized transcript with a local LLM (needs `--diarize` and Ollama)                            |
  | `--ollama-model MODEL`    | `qwen3.5:4b`             | Ollama model for `--refine`                                                                                         |
  | `--ollama-url URL`        | `http://localhost:11434` | Ollama server address                                                                                               |
  | `--hf-token TOKEN`        | none                     | Hugging Face token for gated models (otherwise uses `HF_TOKEN` or `hf auth login`)                                  |
  | `--diarization-path PATH` | none                     | Local copy of the diarization model, for fully offline use                                                          |

### What the output looks like

Each row is one audio file:

```csv
file_id,filename,file_size_bytes,transcription_time_seconds,transcription_text,model_id,started_at,processed_at
a1b2c3d4e5f6g7h8,sample.mp3,1048576,2.34,"Hello world, this is a test recording.",openai/whisper-small,2024-01-01T12:00:00Z,2024-01-01T12:00:02Z
```

  | Column                       | Meaning                                                      |
  | ---------------------------- | ------------------------------------------------------------ |
  | `file_id`                    | ID built from the file name and size (used to skip repeats)  |
  | `filename`                   | Audio file name                                              |
  | `file_size_bytes`            | File size                                                    |
  | `transcription_time_seconds` | How long the transcription took                              |
  | `transcription_text`         | The transcript (with speaker labels if `--diarize` was used) |
  | `model_id`                   | Model(s) used                                                |
  | `started_at`, `processed_at` | When processing started and finished                         |

To keep a log of a long run:
`uv run python src/transcribe_audio.py 2>&1 | tee transcription.log`

## Reference

### Models

  | Model                    | Family  | Size         | Languages | Best for                               |
  | ------------------------ | ------- | ------------ | --------- | -------------------------------------- |
  | `whisper-tiny`           | Whisper | \~39 MB      | 99        | Quick tests                            |
  | `whisper-small`          | Whisper | \~244 MB     | 99        | **Recommended default**                |
  | `whisper-medium`         | Whisper | \~769 MB     | 99        | Higher quality                         |
  | `whisper-large-v3`       | Whisper | \~3 GB       | 99        | Maximum quality, slowest               |
  | `whisper-large-v3-turbo` | Whisper | \~1.5 GB     | 99        | Near large-v3 quality, much faster     |
  | `voxtral-mini`           | Voxtral | \~3B params  | 8         | Multilingual transcription             |
  | `voxtral-small`          | Voxtral | \~24B params | 8         | Best multilingual accuracy (>8 GB GPU) |
  | `cohere`                 | Cohere  | \~2B params  | 14        | Long recordings, multilingual (gated)  |

**Languages** (use these codes with `--language`):

- **Whisper**: 99 languages, ISO 639-1 codes (`en`, `es`, `fr`, `de`, `zh`,
  `ja`, `ko`, `ar`, `hi`, ...), or `auto` to detect.
- **Voxtral**: `en`, `es`, `fr`, `pt`, `hi`, `de`, `nl`, `it`.
- **Cohere**: `en`, `de`, `fr`, `it`, `es`, `pt`, `el`, `nl`, `pl`, `ar`, `vi`,
  `zh`, `ja`, `ko`. No auto-detect.

Notes:

- **Whisper** recordings over 30 seconds are transcribed in full: sequentially
  by default (most accurate) or in parallel chunks with `--chunked` (faster).
- **Voxtral** works out of the box after `uv sync`. Check with
  `uv run python -c "from transformers import VoxtralForConditionalGeneration; print('Voxtral OK')"`.
  It needs more GPU memory than Whisper.
- **Cohere** chunks long recordings itself, so `--max-new-tokens` and
  `--chunked` are ignored. It is gated (see below).

**Adding a model:** models are listed in the `AVAILABLE_MODELS` dictionary in
`src/transcribe_audio.py`. Add an entry with the Hugging Face model ID, its
`type` (`whisper`, `voxtral`, or `cohere`), and a short description:

```python
"whisper-tiny": {
    "id": "openai/whisper-tiny",
    "type": "whisper",
    "description": "Fastest Whisper model, least accurate (~39 MB)",
},
```

### Hugging Face access for gated models

The Cohere, pyannote (diarization), and Phonon (demo audio) models are **gated**
on the Hugging Face Hub. Setup is once per machine; afterwards the models are
cached and run offline.

1. **Accept the terms** on each model page while logged in:
   - Cohere: <https://hf.co/CohereLabs/cohere-transcribe-03-2026>
   - pyannote: <https://hf.co/pyannote/speaker-diarization-community-1>
   - Phonon: <https://hf.co/gradium/phonon-40m-multilingual>

2. **Create a token** at <https://hf.co/settings/tokens>:
   - Simplest: a **classic token with the "Read" role**.
   - A **fine-grained token** must have **"Read access to contents of all public
     gated repos you can access"** enabled, or downloads fail with
     `403 Forbidden: Please enable access to public gated repositories in your fine-grained token settings`.

3. **Authenticate** (either way):

   ```bash
   # Option A: log in once; the token is saved for future runs
   just hf-login

   # Option B: set the token for the current terminal session only
   $env:HF_TOKEN = "hf_..."   # PowerShell
   export HF_TOKEN="hf_..."   # bash/zsh
   ```

Check your login with `uv run hf auth whoami`.

  | Error                                       | Cause                            | Fix                                          |
  | ------------------------------------------- | -------------------------------- | -------------------------------------------- |
  | `401 ... Please log in`                     | Not authenticated                | Step 3                                       |
  | `403 ... fine-grained token settings`       | Token lacks gated-repo access    | Step 2 (edit the token or use a classic one) |
  | `Cannot access gated repo` after logging in | Terms not accepted for the model | Step 1 (accept on each model's page)         |

### Speaker diarization

Add `--diarize` (CLI) or switch on **Multiple speakers (diarization)** (web app)
to label who said what.
[pyannote](https://hf.co/pyannote/speaker-diarization-community-1) finds the
speaker turns, which are paired with the transcription:

```text
SPEAKER_01: Thank you for joining. Let's begin with introductions.; SPEAKER_00: My name is ...; SPEAKER_01: Great, first question: ...
```

- Turns are separated by `"; "` and the whole transcript stays on one line, so
  the output columns don't change. Consecutive turns by the same speaker are
  merged.
- If you know the number of speakers, pass `--num-speakers N` (or set it in the
  app) for better accuracy.
- Files already transcribed without diarization are skipped. Add `--all-audio`
  to redo them with speaker labels.

**Which model to use with diarization:**

- **Whisper**: the recording is transcribed once with timestamps, and each piece
  is matched to a speaker. Wording is as good as a plain run.
- **Cohere and Voxtral**: these models have no timestamps, so each speaker turn
  is cut out and transcribed separately. Short pieces lose context, so wording
  is worse. `--refine` fixes this.

#### Optional: `--refine`: fix diarized wording with a local LLM (Ollama)

`--refine` also transcribes the full recording normally, then asks a local LLM
served by [Ollama](https://ollama.com) to correct the speaker-labeled transcript
so its wording matches. Everything runs on your machine.

One-time setup:

```bash
# 1. Install Ollama from https://ollama.com (Windows: winget install Ollama.Ollama)
# 2. Download the default model (any instruct model works)
ollama pull qwen3.5:4b
# 3. Make sure the server is running (the desktop app, or: ollama serve)
```

Usage:

```bash
uv run python src/transcribe_audio.py --model cohere --language en --diarize --refine
uv run python src/transcribe_audio.py --diarize --refine --ollama-model llama3.2:3b
```

- If Ollama is unreachable or misbehaves, a warning is printed and the unrefined
  transcript is saved. A run never fails because of refinement. `model_id`
  includes `+ ollama:<model>` only when refinement succeeded.
- Very long recordings are refined in chunks split at speaker turns; wording
  near chunk edges can be slightly off.
- Most useful with Cohere/Voxtral. With Whisper it usually changes little.

#### Fully offline diarization

Download the model once, then no token or internet is needed:

```bash
# One-time download (requires git-lfs and Hugging Face access)
git clone https://hf.co/pyannote/speaker-diarization-community-1 ~/models/pyannote-speaker-diarization-community-1

# Use the local copy
uv run python src/transcribe_audio.py --diarize --diarization-path ~/models/pyannote-speaker-diarization-community-1
```

In the web app, enter the same folder under **Local model directory (offline
use)**.

### Output formats

  | Format    | File                        | Best for                             |
  | --------- | --------------------------- | ------------------------------------ |
  | `csv`     | `transcribed_audio.csv`     | Excel, quick inspection              |
  | `json`    | `transcribed_audio.json`    | Web apps, APIs                       |
  | `parquet` | `transcribed_audio.parquet` | pandas / polars analysis             |
  | `duckdb`  | `transcribed_audio.duckdb`  | SQL queries (table `transcriptions`) |

Example DuckDB query:

```bash
echo "SELECT model_id, AVG(transcription_time_seconds) FROM transcriptions GROUP BY model_id;" | duckdb output/transcribed_audio.duckdb
```

To delete all CLI results in `output/`, run `just clean-outputs`.

### Performance and GPU

- On Windows and Linux the project installs CUDA 12.8 PyTorch, which uses an
  NVIDIA GPU when present and the CPU otherwise. macOS uses the CPU.
- On GPU, models run in float16; on CPU, in float32.
- Rough speeds: CPU takes 5 to 10 times the audio length; GPU is 20 to 50 times
  faster than that. Bigger models are slower.

## Development

### Commands

```bash
just venv             # Install packages + pre-commit hooks
just app              # Launch the web app
just lab              # Launch Jupyter Lab
just test             # Run the fast unit tests (mocked, no model downloads)
just test-integration # Run tests on real audio (downloads models)
just test-cov         # Tests with a coverage report
just lint-py          # Lint Python with ruff
just fmt-all          # Lint and format Python, SQL, and Markdown
just pre-commit-run   # Run the pre-commit hooks
just update-reqs      # Upgrade dependencies and hook versions
```

Run `just --list` to see every command.

### Tests

```text
tests/
├── assets/audio/      # Small real audio files for integration tests
├── integration/       # Tests with real audio and real models (slow)
├── unit/              # Fast tests with mocked models
└── conftest.py        # Shared fixtures
```

- `just test` is fast and downloads no models.
- Integration tests skip gracefully when audio files or gated models are
  unavailable. To add audio, put small files (<1 MB, 1 to 30 seconds) in
  `tests/assets/audio/` and document their source in `tests/assets/README.md`.

### Code quality

- `ruff` lints and formats Python (line length 88, Python 3.12).
- Pre-commit hooks validate YAML/JSON/TOML and `pyproject.toml`, check spelling
  (`codespell`), and lint Markdown and Python. `just venv` installs them; they
  run on every commit.
- Write docstrings for public functions.

### Demo notebooks

Run `just lab`, then open:

- `notebooks/demo_whisper_transcription.ipynb`: Whisper (`openai/whisper-small`)
  with audio players and accuracy comparison.
- `notebooks/demo_voxtral_transcription.ipynb`: Voxtral
  (`mistralai/Voxtral-Mini-3B-2507`) with timing metrics.

### Contributing

1. Create a branch: `git checkout -b feature/your-feature-name`
2. Make your changes, with tests for new behavior.
3. Run `just test`, `just fmt-all`, and `just pre-commit-run`.
4. Update `README.md` (users) and `CLAUDE.md` (developer notes) if behavior
   changes.
5. Commit, push, and open a pull request.

## Troubleshooting

  | Problem                                          | Fix                                                                                                    |
  | ------------------------------------------------ | ------------------------------------------------------------------------------------------------------ |
  | `just` or `uv` not found after installing        | Close and reopen the terminal                                                                          |
  | Errors mentioning Python 3.13 or `sentencepiece` | `uv python pin 3.12` then `uv sync`                                                                    |
  | `401`, `403`, or `Cannot access gated repo`      | See [Hugging Face access](#hugging-face-access-for-gated-models)                                       |
  | CLI says no audio files found                    | Files must sit directly in `--input-path` (subfolders are skipped) and be mp3/wav/flac/m4a/ogg         |
  | Files are skipped                                | They were already transcribed; add `--all-audio` (CLI) or tick **Re-transcribe processed files** (app) |
  | GPU runs out of memory                           | Use a smaller model, or pick **CPU** under **Compute device** in the app                               |
  | Voxtral model missing or failing                 | `uv sync` to match the lockfile, then run the Voxtral check in [Models](#models)                       |
  | `--refine` prints a warning                      | Make sure Ollama is running and the model is pulled (`ollama pull qwen3.5:4b`)                         |
