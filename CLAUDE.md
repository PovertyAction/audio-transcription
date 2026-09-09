# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## Project Overview

This is an audio transcription project built with Python that leverages machine
learning libraries for audio processing and transcription. The project uses a
modern Python toolchain with `uv` for dependency management and `just` for task
automation.

The workflow has two entry points:

- **CLI**: `src/transcribe_audio.py` --- batch transcription with configurable
  paths and formats
- **Web GUI**: `src/app.py` --- Streamlit app (`just app`) for uploading and
  transcribing files interactively. It reuses the functions in
  `transcribe_audio.py` (`load_model`, `transcribe_audio`,
  `create_transcription_record`, etc.) --- do not duplicate transcription logic
  in the app. GUI results are session-only downloads and are never appended to
  `output/transcribed_audio.*`.

The project demonstrates transcription capabilities using three model families:

- **Whisper** (OpenAI): `openai/whisper-small` for fast, accurate transcription
  (supports 99 languages). Recordings over 30 seconds are transcribed in full:
  sequential long-form generation by default, or a faster chunked pipeline via
  `--chunked` (CLI) / the "Fast chunked mode" toggle (GUI).
- **Voxtral** (Mistral): `mistralai/Voxtral-Mini-3B-2507` for multilingual
  speech recognition (supports 8 languages: English, Spanish, French,
  Portuguese, Hindi, German, Dutch, Italian)
- **Cohere** (Cohere Labs): `CohereLabs/cohere-transcribe-03-2026` (\~2B params,
  14 languages, no auto-detect). Uses native transformers support
  (`CohereAsrForConditionalGeneration`, requires transformers >=5.4): the
  processor chunks long recordings into overlapping windows and
  `processor.decode(..., audio_chunk_index=..., language=...)` reassembles them
  --- `--max-new-tokens` and `--chunked` are ignored. **Gated model**: accept
  the terms at hf.co/CohereLabs/cohere-transcribe-03-2026 and authenticate
  (HF_TOKEN or `huggingface-cli login`) before the first download.

**Speaker diarization** (`--diarize` CLI / "Multiple speakers" toggle GUI) uses
`pyannote/speaker-diarization-community-1` to find speaker turns, emitting a
one-line transcript of `SPEAKER_XX: text` turns separated by `"; "` (consecutive
same-speaker turns are concatenated) embedded in `transcription_text` (no
output-schema change; `model_id` records the chain). How the transcription is
paired with the turns depends on the model:

- **Whisper**: full-recording transcription with segment timestamps
  (`transcribe_whisper_with_timestamps`, via the ASR pipeline with
  `return_timestamps=True`), speakers assigned per segment by max temporal
  overlap (`assign_speakers_to_segments`) --- wording identical to plain
  whisper. Falls back to per-turn slicing if no usable timestamps.
- **Cohere/Voxtral** (no timestamps): per-turn slicing --- each pyannote turn is
  sliced to a temp WAV and transcribed separately (lower text quality;
  `--refine` fixes it).

**LLM refinement** (`--refine`, requires `--diarize`) rewrites the diarized
transcript with a local Ollama LLM (`refine_transcript_with_llm`, POST
`/api/chat`, `think: false` --- thinking models otherwise burn the whole token
budget on long inputs and return empty content) using a full-quality plain
transcript as reference; on the whisper path the plain text is free (same
segments), for cohere/voxtral one extra plain `transcribe_audio()` runs. Any
`OllamaError` (connection/timeout/404/bad format) prints a warning and keeps the
unrefined transcript; `model_id` gains `+ ollama:<model>` only when refinement
succeeded. `transcribe_with_diarization` returns a 4-tuple ending in
`refined: bool`. Defaults: `OLLAMA_DEFAULT_MODEL = "qwen3.5:4b"`,
`OLLAMA_DEFAULT_URL = "http://localhost:11434"`. `requests` is a direct
dependency.

The pyannote model is also gated --- one-time HF token setup, or fully offline
via `--diarization-path` pointing at a local download (see README "Speaker
Diarization"). Diarization functions live in `transcribe_audio.py`
(`load_diarization_pipeline`, `diarize_audio`, `transcribe_with_diarization`);
pyannote is imported behind a `PYANNOTE_AVAILABLE` guard and always receives
in-memory waveforms. torchcodec (installed with pyannote) has broken DLLs
without FFmpeg; `transcribe_audio.py` force-disables transformers' torchcodec
detection at import so the ASR pipeline skips it.

## Dependencies and Environment Setup

- Uses `uv` for Python environment management and dependency resolution
- Requires Python >=3.12 (NOT 3.13 due to dependency constraints)
- Key ML dependencies: `torch`, `torchaudio`, `transformers>=5.4` (needed for
  native Cohere ASR; do NOT downgrade below 5.4), `librosa`, `soundfile`
- Audio processing: `accelerate>=1.9.0`, `scipy>=1.16.0` (`moshi` was removed:
  unused, and it blocked the transformers 5.x upgrade)
- Diarization: `pyannote-audio>=4.0`. Note: `torchaudio` is pinned to `==2.9.*`
  because its native extension must match the installed torch version
  (mismatched pairs fail to import on Windows)
- Data processing: `pandas>=2.2.3`, `polars>=1.17.1`, `duckdb>=1.1.3`
- Notebooks: `jupyter>=1.1.1`, `jupytext>=1.17.2`, `ipykernel>=6.29.5`

## Repository Management Best Practices

### When Making Changes

1. **Always run code quality checks** before finalizing changes:

   ```bash
   just fmt-all              # Format all code
   just lint-py              # Check Python code
   just pre-commit-run       # Run all pre-commit hooks
   ```

2. **Test notebook functionality** after changes:

   ```bash
   just lab                  # Launch Jupyter Lab
   # Run both demo notebooks to ensure they work
   ```

3. **Update documentation** when adding features:
   - Update README.md for user-facing changes
   - Update CLAUDE.md for development guidance
   - Document new notebooks in README.md

### Special Dependencies

The project requires transformers >=5.4 (currently 5.13 in `uv.lock`): native
Cohere ASR support needs it, and Voxtral and Whisper are verified working there.
Note that transformers 5.x's built-in audio loading uses torchcodec (broken on
this Windows setup without FFmpeg DLLs), so all audio is loaded with librosa and
passed as numpy arrays --- never pass file paths to processors.

To confirm Voxtral models are available:

```bash
uv run python -c "from transformers import VoxtralForConditionalGeneration; print('Voxtral OK')"
```

## Essential Commands

Development uses `just` for command automation. Key commands:

```bash
# Environment setup
just get-started          # Install software and create venv
just venv                 # Create/sync virtual environment
just activate-venv        # Activate environment (uv shell)

# Transcription with custom paths
python src/transcribe_audio.py --input-path /custom/audio --output-path /custom/results
python src/transcribe_audio.py --input-path ~/recordings --output-path ~/transcriptions

# Development workflow
just app                  # Launch Streamlit transcription app (GUI)
just lab                  # Launch Jupyter Lab
just lint-py              # Lint Python code with ruff
just fmt-python           # Format Python code with ruff
just fmt-all              # Format all code (Python, SQL, Markdown)
just pre-commit-run       # Run pre-commit hooks

# Documentation
just preview-docs         # Preview Quarto documentation
just build-docs           # Build Quarto documentation

# Maintenance
just update-reqs          # Update dependencies and pre-commit
just clean                # Remove virtual environment
```

### Transcription Command Examples

```bash
# Basic transcription with defaults (English, 400 tokens)
uv run python src/transcribe_audio.py

# Specify language for better accuracy
uv run python src/transcribe_audio.py --language es  # Spanish
uv run python src/transcribe_audio.py --language fr  # French
uv run python src/transcribe_audio.py --language auto  # Whisper auto-detect

# Adjust max tokens for longer/shorter transcriptions
uv run python src/transcribe_audio.py --max-new-tokens 448  # Max for Whisper
uv run python src/transcribe_audio.py --max-new-tokens 200  # Shorter outputs

# Combined options (Voxtral supports more tokens)
uv run python src/transcribe_audio.py --model voxtral-mini --language de --max-new-tokens 600

# Cohere transcription (explicit language required; long-form handled internally)
uv run python src/transcribe_audio.py --model cohere --language en

# Speaker diarization (any model; gated pyannote model needs one-time HF setup)
uv run python src/transcribe_audio.py --model whisper-small --diarize --num-speakers 2
uv run python src/transcribe_audio.py --diarize --diarization-path ~/models/pyannote-speaker-diarization-community-1  # offline

# Diarization + local-LLM refinement (Ollama running, model pulled)
uv run python src/transcribe_audio.py --model cohere --language en --diarize --refine
uv run python src/transcribe_audio.py --diarize --refine --ollama-model llama3.2:3b

# Custom input and output directories
uv run python src/transcribe_audio.py --input-path /custom/audio --output-path /custom/results
uv run python src/transcribe_audio.py --input-path ~/my-recordings --output-path ~/transcription-output --model whisper-medium
```

## Code Quality Tools

- **Linting/Formatting**: `ruff` with line length 88, Python 3.12 target
- **Pre-commit hooks**: Configured for YAML/JSON/TOML validation, spell
  checking, markdown linting
- **Spell checking**: `codespell` with custom ignore list (jupyter, ipa)
- Uses `ruff` for both linting and formatting (replaces black/flake8/isort)
- **Markdown linting**: `markdownlint-cli` with auto-fixing enabled

### Pre-commit Configuration

The project uses comprehensive pre-commit hooks:

- File validation (YAML, JSON, TOML, merge conflicts)
- Python project validation (`validate-pyproject`)
- Spell checking (`codespell`)
- Markdown formatting (`markdownlint-fix`)
- Python linting and formatting (`ruff-check`, `ruff-format`)

## Project Structure

```markdown
audio-transcription/
├── audio/                     # Default audio files directory (configurable with --input-path)
│   ├── README.md             # Sources and descriptions
│   └── *.mp3                 # Sample audio files from HuggingFace datasets
├── output/                   # Default output directory (configurable with --output-path)
│   └── transcribed_audio.*   # Transcription results in various formats
├── notebooks/                # Demonstration notebooks
│   ├── demo_whisper_transcription.ipynb    # Whisper model demo
│   ├── demo_voxtral_transcription.ipynb    # Voxtral model demo
│   └── *.py                  # Python versions of notebooks (jupytext)
├── src/
│   ├── transcribe_audio.py   # CLI and core transcription functions
│   └── app.py                # Streamlit web GUI (just app)
├── pyproject.toml           # Project configuration and dependencies
├── Justfile                 # Command automation and task definitions
├── uv.lock                  # Locked dependency versions
├── .pre-commit-config.yaml  # Pre-commit hook configuration
└── README.md                # User-facing documentation
```

### Audio Files

Sample audio files are sourced from:

- `benjaminogbonna/nigerian_accented_english_dataset` (HuggingFace)
- `hf-internal-testing/dummy-audio-samples` (HuggingFace)

Files include diverse speech patterns for testing transcription accuracy.

## Platform Support

Supports Windows, macOS, and Linux with platform-specific installation commands:

- **Windows**: Uses `winget` for package installation
- **macOS/Linux**: Uses `brew` for package installation
- **Linux**: Additional `apt` packages for build tools

### Dependencies

- **Python version constraint**: Stick to Python 3.12 (avoid 3.13)
- **Special model requirements**: transformers >=5.4 (native Cohere ASR;
  satisfied by uv.lock)
- **GPU support**: torch/torchaudio come from the PyTorch cu128 index on
  Windows/Linux (see `[tool.uv.sources]` in pyproject.toml); CUDA wheels fall
  back to CPU on machines without an NVIDIA GPU
- **Platform considerations**: Some dependencies require build tools (cmake,
  build-essential)

### File Organization

- **Notebooks**: Keep demonstration notebooks in `notebooks/`
- **Audio samples**: Store in `audio/` with proper README documentation
- **Configuration**: Use pyproject.toml for all Python project configuration
- **Automation**: Use Justfile for cross-platform command automation
