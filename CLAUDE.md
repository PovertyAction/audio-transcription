# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an audio transcription project built with Python that leverages machine learning libraries for audio processing and transcription. The project uses a modern Python toolchain with `uv` for dependency management and `just` for task automation.

The project demonstrates transcription capabilities using two models:

- **Whisper** (OpenAI): `openai/whisper-small` for fast, accurate transcription (supports 99 languages)
- **Voxtral** (Mistral): `mistralai/Voxtral-Mini-3B-2507` for multilingual speech recognition (supports 8 languages: English, Spanish, French, Portuguese, Hindi, German, Dutch, Italian)

## Dependencies and Environment Setup

- Uses `uv` for Python environment management and dependency resolution
- Requires Python >=3.12 (NOT 3.13 due to dependency constraints)
- Key ML dependencies: `torch`, `torchaudio`, `transformers>=4.53.2`, `librosa`, `soundfile`
- Audio processing: `accelerate>=1.9.0`, `moshi>=0.2.11`, `scipy>=1.16.0`
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

The project requires special handling for Mistral's Voxtral model:

```bash
# Required for Voxtral model support (must be done in this order)
uv pip install git+https://github.com/huggingface/transformers
uv pip install --upgrade "mistral-common[audio]"
```

**Important**: After installing these dependencies, you must activate the virtual environment before testing Voxtral models:

```bash
source .venv/bin/activate
python src/transcribe_audio.py --help | grep -A 10 "Available models:"
```

This ensures that Voxtral models are properly recognized and available for use.

> **Note**: These extra installation steps may become obsolete once Voxtral models are available in a future stable release of HuggingFace transformers. The project will automatically use the standard dependencies when Voxtral support is included in the stable release.

## Essential Commands

Development uses `just` for command automation. Key commands:

```bash
# Environment setup
just get-started          # Install software and create venv
just venv                 # Create/sync virtual environment
just activate-venv        # Activate environment (uv shell)

# Development workflow
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

# Adjust max tokens for longer/shorter transcriptions
uv run python src/transcribe_audio.py --max-new-tokens 448  # Max for Whisper
uv run python src/transcribe_audio.py --max-new-tokens 200  # Shorter outputs

# Combined options (Voxtral supports more tokens)
uv run python src/transcribe_audio.py --model voxtral-mini --language de --max-new-tokens 600
```

## Code Quality Tools

- **Linting/Formatting**: `ruff` with line length 88, Python 3.12 target
- **Pre-commit hooks**: Configured for YAML/JSON/TOML validation, spell checking, markdown linting
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
├── audio/                     # Audio files for transcription demos
│   ├── README.md             # Sources and descriptions
│   └── *.mp3                 # Sample audio files from HuggingFace datasets
├── notebooks/                # Demonstration notebooks
│   ├── demo_whisper_transcription.ipynb    # Whisper model demo
│   ├── demo_voxtral_transcription.ipynb    # Voxtral model demo
│   └── *.py                  # Python versions of notebooks (jupytext)
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
- **Special model requirements**: Voxtral model needs git+transformers and mistral-common[audio]
- **Platform considerations**: Some dependencies require build tools (cmake, build-essential)

### File Organization

- **Notebooks**: Keep demonstration notebooks in `notebooks/`
- **Audio samples**: Store in `audio/` with proper README documentation
- **Configuration**: Use pyproject.toml for all Python project configuration
- **Automation**: Use Justfile for cross-platform command automation
