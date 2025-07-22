# Audio Transcription Project

Python project for audio transcription using OpenAI Whisper and Mistral Voxtral models.

## Usage

This project provides a command-line tool for transcribing audio files using OpenAI's Whisper models and Mistral's Voxtral models. You can transcribe individual files or batch process entire directories with support for multiple output formats.

### Quick Start

After setting up the environment (see [Development setup](#development-set-up) below):

```bash
# Activate the environment
uv sync

# Transcribe all audio files in the audio/ directory to CSV
uv run python src/transcribe_audio.py

# Or use specific options
uv run python src/transcribe_audio.py --model whisper-tiny --format json

uv run python src/transcribe_audio.py --model whisper-tiny --format duckdb
```

### Command Line Interface

The transcription script supports several command-line options:

```bash
uv run python src/transcribe_audio.py [OPTIONS]
```

**Available Options:**

- `--model MODEL`: Choose the transcription model - Whisper or Voxtral (default: `whisper-small`)
- `--format FORMAT`: Output format for results (default: `csv`)
- `--all-audio`: Re-process all files, including previously transcribed ones

### Available Models

#### Whisper Models (OpenAI)

Choose from different Whisper models based on your speed vs accuracy needs:

| Model | Description | Size | Use Case |
|-------|-------------|------|----------|
| `whisper-tiny` | Fastest model, least accurate | ~39 MB | Quick testing, real-time |
| `whisper-small` | Fast model, good accuracy | ~244 MB | **Recommended default** |
| `whisper-medium` | Balanced speed/accuracy | ~769 MB | High-quality transcription |
| `whisper-large-v3-turbo` | Best accuracy, slower | ~1550 MB | Maximum quality needed |

#### Voxtral Models (Mistral AI) - Optional

For multilingual speech recognition with advanced capabilities:

| Model | Description | Size | Use Case |
|-------|-------------|------|---------|
| `voxtral-mini` | Multilingual ASR model | ~3B params | Fast multilingual transcription |
| `voxtral-small` | High-quality multilingual ASR | ~24B params | Best multilingual accuracy |

> **Note**: Voxtral models require additional dependencies. See [Voxtral Setup](#voxtral-model-setup) below.

### Output Formats

Save your transcriptions in multiple formats:

| Format | Extension | Description | Best For |
|--------|-----------|-------------|----------|
| `csv` | `.csv` | Comma-separated values | Excel, data analysis |
| `json` | `.json` | JavaScript Object Notation | Web applications, APIs |
| `parquet` | `.parquet` | Apache Parquet columnar | Big data, analytics |
| `duckdb` | `.duckdb` | DuckDB database | SQL queries, complex analysis |

### Usage Examples

**Basic transcription:**

```bash
# Transcribe all audio files with default settings
uv run python src/transcribe_audio.py
```

**Choose model and format:**

```bash
# Use tiny Whisper model for fast processing, save as JSON
uv run python src/transcribe_audio.py --model whisper-tiny --format json

# Use large Whisper model for best quality, save to database
uv run python src/transcribe_audio.py --model whisper-large-v3-turbo --format duckdb

# Use Voxtral model for multilingual transcription (requires setup)
uv run python src/transcribe_audio.py --model voxtral-mini --format json
```

**Re-process all files:**

```bash
# Force re-transcription of all files (ignores existing results)
uv run python src/transcribe_audio.py --all-audio --format parquet
```

**Production batch processing:**

```bash
# Process large batches with balanced Whisper model
uv run python src/transcribe_audio.py --model whisper-medium --format duckdb

# Process multilingual content with Voxtral
uv run python src/transcribe_audio.py --model voxtral-small --format parquet --all-audio
```

### Input Requirements

**Supported Audio Formats:**

- MP3 (`.mp3`)
- WAV (`.wav`)
- FLAC (`.flac`)
- M4A (`.m4a`)
- OGG (`.ogg`)

**File Organization:**

- Place audio files in the `audio/` directory
- The script automatically discovers all supported audio files
- Files are processed in alphabetical order

### Output Structure

All transcription results include:

- **File ID**: Unique identifier based on filename and size
- **Filename**: Original audio file name
- **File Size**: Size in bytes
- **Transcription Time**: Processing duration in seconds
- **Transcription Text**: The actual transcribed text
- **Model ID**: Which Whisper model was used
- **Timestamps**: When processing started and completed

**Example CSV Output:**

```csv
file_id,filename,file_size_bytes,transcription_time_seconds,transcription_text,model_id,started_at,processed_at
a1b2c3d4e5f6g7h8,sample.mp3,1048576,2.34,"Hello world, this is a test recording.",openai/whisper-small,2024-01-01T12:00:00Z,2024-01-01T12:00:02Z
```

### Performance and GPU Support

**Automatic Device Detection:**

- Uses CUDA GPU if available for faster processing
- Falls back to CPU automatically
- Model precision adjusted based on device (float16 for GPU, float32 for CPU)

**Processing Speed Examples:**

- **CPU**: ~5-10x real-time (10 second audio = 50-100 seconds processing)
- **GPU**: ~20-50x real-time (10 second audio = 5-20 seconds processing)
- Actual speed varies by model size and hardware

### Incremental Processing

The script avoids re-processing files by:

1. **Generating unique file IDs** based on filename + file size
2. **Checking existing results** in the output file
3. **Skipping previously processed files** (unless `--all-audio` is used)
4. **Appending new results** to existing output files

This makes it efficient for processing large directories incrementally.

### Error Handling

The script handles various error conditions gracefully:

- **Missing audio directory**: Creates directory if needed
- **Unsupported file formats**: Skips with warning
- **Corrupted audio files**: Continues processing other files
- **Model loading errors**: Provides clear error messages
- **Disk space issues**: Fails gracefully with error details

### Integration Examples

**Using with other tools:**

```bash
# Process audio and analyze results with DuckDB
uv run python src/transcribe_audio.py --format duckdb
echo "SELECT model_id, AVG(transcription_time_seconds) FROM transcriptions GROUP BY model_id;" | duckdb output/transcribed_audio.duckdb

# Export to CSV for Excel analysis
uv run python src/transcribe_audio.py --format csv
# Open output/transcribed_audio.csv in Excel

# Process and convert to different format
uv run python src/transcribe_audio.py --format parquet
# Use pandas/polars to read parquet file in data science workflows
```

**Monitoring progress:**

```bash
# Watch processing in real-time
uv run python src/transcribe_audio.py --model whisper-small --format json 2>&1 | tee transcription.log
```

## Voxtral Model Setup

To use Mistral's Voxtral models for multilingual speech recognition, you need to install additional dependencies.

### Prerequisites for Voxtral

Voxtral models require the latest development version of the `transformers` library and additional audio processing dependencies.

### Installation Steps

1. **Install development transformers** (required for Voxtral support):

   ```bash
   uv pip install git+https://github.com/huggingface/transformers
   ```

2. **Install Mistral audio dependencies**:

   ```bash
   uv pip install --upgrade "mistral-common[audio]"
   ```

3. **Verify installation** by checking available models:

   ```bash
   uv run python src/transcribe_audio.py --help
   ```

   You should see both Whisper and Voxtral models listed if installation was successful.

### Voxtral vs Whisper Comparison

| Feature | Whisper | Voxtral |
|---------|---------|---------|
| **Languages** | 99+ languages | Optimized for multilingual |
| **Model Size** | 39MB - 1.5GB | 3B - 24B parameters |
| **Speed** | Fast to moderate | Moderate to slow |
| **Accuracy** | High for English | Very high for multilingual |
| **Dependencies** | Standard transformers | Development transformers + mistral-common |
| **Use Case** | General transcription | Advanced multilingual ASR |

### Troubleshooting Voxtral

**If Voxtral models don't appear:**

- Ensure you installed the development version of transformers
- Check that mistral-common[audio] is properly installed
- Restart your environment after installation

**If you get import errors:**

```bash
# Clean reinstall
uv pip uninstall transformers mistral-common
uv pip install git+https://github.com/huggingface/transformers
uv pip install --upgrade "mistral-common[audio]"
```

**Performance considerations:**

- Voxtral models require more GPU memory than Whisper
- Use `voxtral-mini` for faster processing
- Use `voxtral-small` only with sufficient GPU memory (>8GB recommended)

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
just test             # Run core test suite
just test-cov         # Run tests with coverage report
```

## Testing

This project includes a comprehensive test suite with 74 passing tests that validate all aspects of the audio transcription functionality.

### Test Structure

```markdown
tests/
├── assets/                    # Test audio files for integration tests
│   └── audio/                 # Real audio files (add your own)
├── integration/               # Integration tests with real files
│   └── test_real_audio.py     # Tests using actual audio files
├── unit/                      # Fast unit tests (mocked)
│   ├── test_file_operations.py    # ✅ File ID, audio discovery, sizes
│   ├── test_data_formats.py       # ✅ CSV, JSON, Parquet, DuckDB ops
│   ├── test_model_loading.py      # ✅ Whisper model loading & validation
│   ├── test_transcription.py      # ✅ Core transcription with mocks
│   ├── test_cli.py                # ✅ CLI argument parsing
│   └── test_error_handling.py     # ⚠️  Error handling (3 edge cases)
└── conftest.py                # Shared fixtures and test utilities
```

### Running Tests

**Quick Testing (Recommended):**

```bash
# Run core working tests (fast, ~6 seconds)
just test
# Or: uv run python -m pytest
```

**Comprehensive Testing:**

```bash
# Run all working unit tests
just test-unit

# Run integration tests with real audio files
just test-integration

# Run slow/comprehensive tests
just test-slow

# Run ALL tests (including broken ones for debugging)
just test-all

# Run only broken tests for debugging
just test-broken
```

**Coverage Reports:**

```bash
# Terminal coverage report
just test-cov

# HTML coverage report (opens in browser)
just test-cov-html

# XML coverage report (for CI)
just test-cov-xml
```

### Test Categories

**✅ Working Tests (74 tests):**

- **File Operations** (12 tests): File ID generation, audio discovery, file sizes
- **Data Formats** (22 tests): Save/load operations for CSV, JSON, Parquet, DuckDB
- **Model Loading** (13 tests): Whisper model validation, device handling
- **Transcription Core** (11 tests): Audio processing with mocked models
- **CLI Interface** (16 tests): Command-line argument parsing and workflow integration

**⚠️ Broken Tests (3 tests):**

- **Error Handling**: 3 filesystem permission/error detection edge cases

### Adding Real Audio Files for Integration Testing

1. **Place audio files** in `tests/assets/audio/`
2. **Keep files small** (< 1MB each, 1-30 seconds duration)
3. **Document sources** in `tests/assets/README.md`
4. **Run integration tests:**

   ```bash
   just test-integration
   ```

Integration tests will skip gracefully if no audio files are present.

### Test Design Principles

- **Fast by default**: Core tests run in ~6 seconds using mocks
- **No model downloads**: Uses mocked ML models to avoid heavy downloads
- **Graceful skipping**: Integration tests skip when real audio files unavailable
- **Comprehensive coverage**: Tests all output formats and error scenarios
- **CI-ready**: Provides detailed coverage reports for continuous integration

## Code Quality

The project enforces high code quality standards through automated tools:

### Code Formatting and Linting

```bash
# Format all code (Python, SQL, Markdown)
just fmt-all

# Format only Python code
just fmt-python

# Lint Python code with ruff
just lint-py

# Run all pre-commit hooks
just pre-commit-run
```

### Pre-commit Hooks

The project uses comprehensive pre-commit hooks that run automatically before each commit:

- **File validation**: YAML, JSON, TOML syntax checking
- **Python validation**: `validate-pyproject` for pyproject.toml
- **Spell checking**: `codespell` with custom ignore list
- **Markdown formatting**: `markdownlint-fix` with auto-fixing
- **Python formatting**: `ruff-format` for consistent code style
- **Python linting**: `ruff-check` with comprehensive rule set

**Setup pre-commit hooks:**

```bash
# Install pre-commit hooks (run once)
uv run pre-commit install

# Run hooks manually on all files
just pre-commit-run

# Update hook versions
just update-reqs
```

### Code Style Standards

- **Line length**: 88 characters (follows Black standard)
- **Python target**: 3.12 compatibility
- **Docstrings**: Required for public functions and classes
- **Type hints**: Encouraged but not enforced
- **Import sorting**: Automatic via ruff
- **Formatting**: Automatic via ruff (replaces Black)

### IDE Setup

**VS Code (recommended):**

- Install Python extension
- Set Python interpreter to `.venv/bin/python` (or `.venv/Scripts/python.exe` on Windows)
- Install Jupyter extension for notebook support
- Install Python Test Explorer for integrated test running

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

## Contributing

### Development Workflow

1. **Fork and clone** the repository

2. **Set up development environment**:

   ```bash
   just get-started
   uv shell
   ```

3. **Install pre-commit hooks**:

   ```bash
   uv run pre-commit install
   ```

4. **Create a feature branch**:

   ```bash
   git checkout -b feature/your-feature-name
   ```

5. **Make your changes** following the code style standards

6. **Run tests** to ensure everything works:

   ```bash
   just test
   just lint-py
   ```

7. **Commit your changes** (pre-commit hooks will run automatically):

   ```bash
   git add .
   git commit -m "Add your descriptive commit message"
   ```

8. **Push and create a pull request**

### Adding New Features

When adding new functionality:

1. **Write tests first** (TDD approach recommended)
2. **Update documentation** in README.md and CLAUDE.md
3. **Ensure code quality** by running `just fmt-all` and `just lint-py`
4. **Add integration tests** if working with real audio files
5. **Update notebooks** if the feature affects demo functionality

### Code Review Checklist

- [ ] Tests pass (`just test`)
- [ ] Code is formatted (`just fmt-all`)
- [ ] Code is linted (`just lint-py`)
- [ ] Documentation is updated
- [ ] Pre-commit hooks pass
- [ ] Integration tests work (if applicable)

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
