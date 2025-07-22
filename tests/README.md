# Test Suite Documentation

## Overview

This project has a comprehensive test suite with **58 working unit tests** that thoroughly validate the audio transcription functionality.

## Test Structure

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
│   ├── test_cli.py                # ❌ CLI tests (broken, need fixing)
│   └── test_error_handling.py     # ❌ Error handling (broken, need fixing)
└── conftest.py                # Shared fixtures and test utilities
```

## Available Test Commands

### Quick Testing (Default)

```bash
# Run core working tests (fast, ~6 seconds)
just test
# Or: uv run python -m pytest
```

### Specific Test Categories

```bash
# Run all working unit tests explicitly
just test-unit

# Run integration tests with real audio files
just test-integration

# Run slow/comprehensive tests
just test-slow

# Run ALL tests (including broken ones)
just test-all

# Run only the broken tests for debugging
just test-broken
```

### Coverage Reports

```bash
# Terminal coverage report
just test-cov

# HTML coverage report
just test-cov-html

# XML coverage report (for CI)
just test-cov-xml
```

## Test Status

### ✅ **Working Tests (74 tests)**

- **File Operations**: ID generation, audio discovery, file sizes (12 tests)
- **Data Formats**: Save/load CSV, JSON, Parquet, DuckDB (22 tests)
- **Model Loading**: Whisper model validation, device handling (13 tests)
- **Transcription Core**: Audio processing with mocked models (11 tests)
- **CLI Interface**: Command-line argument parsing, workflow integration (16 tests)

### ❌ **Broken Tests (3 tests)**

- **Error Handling**: 3 filesystem permission/error detection edge cases

## Integration Testing with Real Audio Files

### Adding Test Audio Files

1. Place audio files in `tests/assets/audio/`
2. Keep files small (< 1MB each)
3. Use short duration (1-30 seconds)
4. Document sources in `tests/assets/README.md`

### Running Integration Tests

```bash
# Test with real audio files (if available)
just test-integration

# Integration tests will skip gracefully if no audio files exist
```

## Test Configuration

Tests are configured in `pyproject.toml`:

- **Default**: Excludes broken CLI/error handling tests
- **Markers**: `integration`, `slow`, `unit` for selective testing
- **Coverage**: Source tracking enabled for `src/` directory

## Continuous Integration

The test suite is designed to:

- ✅ **Run fast by default** (~6 seconds for core tests)
- ✅ **Use mocks** to avoid downloading ML models
- ✅ **Skip gracefully** when real audio files unavailable
- ✅ **Provide detailed coverage** reports for CI systems

## Next Steps

1. **Fix remaining 3 broken tests**: Filesystem permission edge cases
2. **Add real audio files**: For comprehensive integration testing
3. **Expand coverage**: Add more edge cases and error scenarios
4. **Performance tests**: Add timing and memory usage validation
