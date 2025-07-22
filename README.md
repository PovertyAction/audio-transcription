# Audio Transcription Project

Python project for audio transcription using machine learning models.

## Development set up

### Prerequisites

This project requires Python 3.11 or 3.12 (not 3.13 due to dependency constraints) and the following tools:

- **Python 3.11-3.12**: Required for compatibility with audio processing dependencies
- **uv**: Modern Python package manager for dependency management
- **just**: Command runner for common development tasks
- **cmake**: Required for building audio processing dependencies
- **git**: Version control

### Installation

#### 1. Install System Dependencies

**On macOS/Linux with Homebrew:**

```bash
brew install just uv cmake
```

**On Windows:**

```bash
winget install Casey.Just astral-sh.uv
# Install cmake via Visual Studio Build Tools or from cmake.org
```

**On Linux (Ubuntu/Debian):**

```bash
# Install via Homebrew
brew install just uv
# Install other tools via apt
sudo apt install cmake build-essential pkg-config
```

#### 2. Clone and Setup Project

```bash
git clone <repository-url>
cd audio-transcription
```

#### 3. Environment Setup

**Quick setup (recommended):**

```bash
just get-started
```

**Manual setup:**

```bash
# Create virtual environment with correct Python version
uv venv --python 3.12
uv sync
```

#### 4. Activate Environment

The project uses `uv` for environment management:

```bash
# Activate the environment
uv shell

# Or run commands directly with uv
uv run python your_script.py
uv run jupyter lab
```

### Development Workflow

Common development commands using `just`:

```bash
just lab              # Launch Jupyter Lab
just lint-py          # Lint Python code
just fmt-python       # Format Python code
just fmt-all          # Format all code (Python, SQL, Markdown)
just pre-commit-run   # Run pre-commit hooks
```

### IDE Setup

**VS Code (recommended):**

- Install Python extension
- Set Python interpreter to `.venv/bin/python` (or `.venv/Scripts/python.exe` on Windows)
- Install Jupyter extension for notebook support

## Demo Notebooks

The project includes demonstration notebooks showcasing different transcription models:

### Available Demos

- **`notebooks/demo_whisper_transcription.ipynb`**: Demonstrates audio transcription using OpenAI's Whisper model
  - Uses the `openai/whisper-small` model for faster inference
  - Includes interactive audio players for testing
  - Shows transcription accuracy comparison with original text

- **`notebooks/demo_voxtral_transcription.ipynb`**: Demonstrates audio transcription using Mistral's Voxtral model
  - Uses the `mistralai/Voxtral-Mini-3B-2507` model
  - Multilingual speech recognition capabilities
  - Interactive audio processing with timing metrics

### Running the Demos

```bash
# Launch Jupyter Lab
just lab

# Or run directly with uv
uv run jupyter lab
```

Both notebooks include sample audio files and demonstrate:

- Model loading and setup
- Audio file processing with interactive players
- Transcription accuracy comparison
- Performance timing metrics

### Troubleshooting

**Python 3.13 Issues:**
If you encounter errors with `sentencepiece` or other dependencies, ensure you're using Python 3.11 or 3.12:

```bash
uv python pin 3.12
uv sync
```

**Missing cmake:**
Audio dependencies require cmake for compilation. Install via your system package manager.

**Voxtral Model Issues:**
If you encounter issues with the Voxtral model, ensure you have the correct version of `uv` and that the model is downloaded correctly:

```bash
uv sync
uv pip install git+https://github.com/huggingface/transformers
uv pip install --upgrade "mistral-common[audio]"

# Or shortcut
just venv
source .venv/bin/activate  # On Linux/macOS
# .venv\Scripts\activate  # On Windows
```
