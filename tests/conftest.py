import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def mock_audio_dir(temp_dir):
    """Create a mock audio directory with test files."""
    audio_dir = temp_dir / "audio"
    audio_dir.mkdir()

    # Create mock audio files
    (audio_dir / "test1.mp3").write_text("mock mp3 content")
    (audio_dir / "test2.wav").write_text("mock wav content")
    (audio_dir / "test3.flac").write_text("mock flac content")
    (audio_dir / "readme.txt").write_text("not an audio file")

    return audio_dir


@pytest.fixture
def mock_output_dir(temp_dir):
    """Create a mock output directory."""
    output_dir = temp_dir / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def test_assets_dir():
    """Get the test assets directory path."""
    return Path(__file__).parent / "assets"


@pytest.fixture
def test_audio_dir():
    """Get the test audio files directory."""
    return Path(__file__).parent / "assets" / "audio"


@pytest.fixture
def mock_processor():
    """Create a mock Whisper processor."""
    processor = Mock()
    processor.batch_decode.return_value = ["This is a test transcription."]
    return processor


@pytest.fixture
def mock_model():
    """Create a mock Whisper model."""
    model = Mock()
    model.generate.return_value = [[1, 2, 3, 4, 5]]  # Mock token IDs
    return model


@pytest.fixture
def sample_transcription_records():
    """Sample transcription records for testing."""
    return [
        {
            "file_id": "abc123",
            "filename": "test1.mp3",
            "file_size_bytes": 1024,
            "transcription_time_seconds": 2.5,
            "transcription_text": "Hello world",
            "model_id": "openai/whisper-small",
            "started_at": "2024-01-01T12:00:00Z",
            "processed_at": "2024-01-01T12:00:02Z",
        },
        {
            "file_id": "def456",
            "filename": "test2.wav",
            "file_size_bytes": 2048,
            "transcription_time_seconds": 3.0,
            "transcription_text": "This is a test",
            "model_id": "openai/whisper-small",
            "started_at": "2024-01-01T12:01:00Z",
            "processed_at": "2024-01-01T12:01:03Z",
        },
    ]
