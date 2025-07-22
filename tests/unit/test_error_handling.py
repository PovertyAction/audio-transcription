"""Tests for error handling in transcribe_audio module."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.transcribe_audio import (
    get_audio_files,
    get_file_size,
    load_existing_results,
    load_whisper_model,
    save_results,
    transcribe_audio,
)


class TestFileSystemErrors:
    """Test error handling for filesystem operations."""

    @patch("src.transcribe_audio.AUDIO_DIR")
    def test_get_audio_files_missing_directory(self, mock_audio_dir, temp_dir):
        """Test handling when audio directory doesn't exist."""
        missing_dir = temp_dir / "missing_audio"
        mock_audio_dir.return_value = missing_dir

        # Should raise FileNotFoundError when trying to iterate non-existent directory
        with pytest.raises(FileNotFoundError):
            get_audio_files()

    @patch("src.transcribe_audio.AUDIO_DIR")
    def test_get_audio_files_permission_denied(self, mock_audio_dir, temp_dir):
        """Test handling when audio directory has permission issues."""
        audio_dir = temp_dir / "restricted_audio"
        audio_dir.mkdir(mode=0o000)  # No permissions
        mock_audio_dir.return_value = audio_dir

        try:
            with pytest.raises(PermissionError):
                get_audio_files()
        finally:
            # Restore permissions for cleanup
            audio_dir.chmod(0o755)

    def test_get_file_size_missing_file(self, temp_dir):
        """Test file size calculation for non-existent file."""
        missing_file = temp_dir / "missing.txt"

        with pytest.raises(FileNotFoundError):
            get_file_size(missing_file)

    def test_get_file_size_permission_denied(self, temp_dir):
        """Test file size calculation with permission issues."""
        restricted_file = temp_dir / "restricted.txt"
        restricted_file.write_text("content")
        restricted_file.chmod(0o000)  # No permissions

        try:
            with pytest.raises(PermissionError):
                get_file_size(restricted_file)
        finally:
            # Restore permissions for cleanup
            restricted_file.chmod(0o644)


class TestDataFormatErrors:
    """Test error handling for data format operations."""

    def test_save_results_invalid_directory(self, sample_transcription_records):
        """Test saving to invalid directory path."""
        invalid_path = Path("/invalid/directory/that/does/not/exist/output.csv")

        # The actual error raised is OSError, not FileNotFoundError
        with pytest.raises(
            OSError, match="Cannot save file into a non-existent directory"
        ):
            save_results(sample_transcription_records, "csv", invalid_path)

    def test_save_results_permission_denied(
        self, temp_dir, sample_transcription_records
    ):
        """Test saving with permission denied on output directory."""
        restricted_dir = temp_dir / "restricted"
        restricted_dir.mkdir(mode=0o444)  # Read-only
        output_file = restricted_dir / "output.csv"

        try:
            with pytest.raises(PermissionError):
                save_results(sample_transcription_records, "csv", output_file)
        finally:
            # Restore permissions for cleanup
            restricted_dir.chmod(0o755)

    def test_load_existing_results_corrupted_duckdb(self, temp_dir):
        """Test loading from corrupted DuckDB file."""
        db_file = temp_dir / "corrupted.duckdb"
        db_file.write_text("This is not a valid DuckDB file")

        # Should handle corruption gracefully and return empty set
        file_ids = load_existing_results("duckdb", db_file)
        assert file_ids == set()

    def test_save_results_disk_space_error(
        self, temp_dir, sample_transcription_records
    ):
        """Test handling of disk space errors during save."""
        output_file = temp_dir / "output.csv"

        with patch("pandas.DataFrame.to_csv") as mock_to_csv:
            mock_to_csv.side_effect = OSError("No space left on device")

            with pytest.raises(OSError, match="No space left on device"):
                save_results(sample_transcription_records, "csv", output_file)


class TestModelLoadingErrors:
    """Test error handling for model loading operations."""

    @patch("src.transcribe_audio.WhisperProcessor")
    def test_load_whisper_model_network_error(self, mock_processor_cls):
        """Test handling of network errors during model loading."""
        mock_processor_cls.from_pretrained.side_effect = Exception("Connection timeout")

        with pytest.raises(Exception, match="Connection timeout"):
            load_whisper_model("whisper-small", "cpu")

    @patch("src.transcribe_audio.WhisperProcessor")
    @patch("src.transcribe_audio.WhisperForConditionalGeneration")
    def test_load_whisper_model_insufficient_memory(
        self, mock_model_cls, mock_processor_cls
    ):
        """Test handling of memory errors during model loading."""
        mock_processor_cls.from_pretrained.return_value = Mock()
        mock_model_cls.from_pretrained.side_effect = RuntimeError("CUDA out of memory")

        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            load_whisper_model("whisper-small", "cuda")

    @patch("src.transcribe_audio.WhisperProcessor")
    @patch("src.transcribe_audio.WhisperForConditionalGeneration")
    def test_load_whisper_model_invalid_device(
        self, mock_model_cls, mock_processor_cls
    ):
        """Test handling of invalid device specification."""
        mock_processor = Mock()
        mock_model = Mock()
        mock_processor_cls.from_pretrained.return_value = mock_processor
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_model.to.side_effect = RuntimeError("Invalid device specified")

        with pytest.raises(RuntimeError, match="Invalid device specified"):
            load_whisper_model("whisper-small", "invalid-device")


class TestTranscriptionErrors:
    """Test error handling for transcription operations."""

    @patch("src.transcribe_audio.librosa.load")
    def test_transcribe_audio_unsupported_format(
        self, mock_librosa_load, mock_processor, mock_model
    ):
        """Test handling of unsupported audio formats."""
        mock_librosa_load.side_effect = Exception("Unsupported audio format")

        audio_path = Path("/fake/path/unsupported.xyz")

        with pytest.raises(Exception, match="Unsupported audio format"):
            transcribe_audio(
                audio_path, mock_processor, mock_model, "cpu", "openai/whisper-small"
            )

    @patch("src.transcribe_audio.librosa.load")
    def test_transcribe_audio_corrupted_file(
        self, mock_librosa_load, mock_processor, mock_model
    ):
        """Test handling of corrupted audio files."""
        mock_librosa_load.side_effect = Exception("Audio file is corrupted")

        audio_path = Path("/fake/path/corrupted.mp3")

        with pytest.raises(Exception, match="Audio file is corrupted"):
            transcribe_audio(
                audio_path, mock_processor, mock_model, "cpu", "openai/whisper-small"
            )

    @patch("src.transcribe_audio.librosa.load")
    def test_transcribe_audio_empty_file(
        self, mock_librosa_load, mock_processor, mock_model
    ):
        """Test handling of empty audio files."""
        import numpy as np

        mock_librosa_load.return_value = (np.array([]), 16000)  # Empty audio

        mock_inputs = Mock()
        mock_inputs.input_features = Mock()
        mock_inputs.to.return_value = mock_inputs
        mock_processor.return_value = mock_inputs

        # This might cause issues during processing
        mock_processor.side_effect = Exception("Cannot process empty audio")

        audio_path = Path("/fake/path/empty.mp3")

        with pytest.raises(Exception, match="Cannot process empty audio"):
            transcribe_audio(
                audio_path, mock_processor, mock_model, "cpu", "openai/whisper-small"
            )


class TestInputValidationErrors:
    """Test error handling for invalid inputs."""

    def test_load_whisper_model_empty_model_name(self):
        """Test handling of empty model name."""
        with pytest.raises(ValueError):
            load_whisper_model("", "cpu")

    def test_load_whisper_model_none_model_name(self):
        """Test handling of None model name."""
        with pytest.raises((ValueError, TypeError)):
            load_whisper_model(None, "cpu")

    def test_save_results_invalid_format(self, temp_dir, sample_transcription_records):
        """Test saving with completely invalid format."""
        output_file = temp_dir / "output.invalid"

        with pytest.raises(ValueError, match="Unsupported output format"):
            save_results(sample_transcription_records, "invalid_format", output_file)

    def test_save_results_none_records(self, temp_dir):
        """Test saving None records."""
        output_file = temp_dir / "output.csv"

        # The function actually handles None by returning early, so no exception
        save_results(None, "csv", output_file)
        # File should not be created
        assert not output_file.exists()

    def test_load_existing_results_none_path(self):
        """Test loading with None path."""
        with pytest.raises((TypeError, AttributeError)):
            load_existing_results("csv", None)


class TestConcurrencyErrors:
    """Test error handling for concurrent access issues."""

    def test_save_results_file_locked(self, temp_dir, sample_transcription_records):
        """Test handling of file locking issues."""
        output_file = temp_dir / "locked.csv"

        with patch("pandas.DataFrame.to_csv") as mock_to_csv:
            mock_to_csv.side_effect = PermissionError(
                "File is locked by another process"
            )

            with pytest.raises(
                PermissionError, match="File is locked by another process"
            ):
                save_results(sample_transcription_records, "csv", output_file)

    def test_load_existing_results_file_being_written(self, temp_dir):
        """Test handling of files being written during read."""
        csv_file = temp_dir / "being_written.csv"

        with patch("pandas.read_csv") as mock_read_csv:
            mock_read_csv.side_effect = Exception("File is being modified")

            # Should handle gracefully and return empty set
            file_ids = load_existing_results("csv", csv_file)
            assert file_ids == set()


class TestResourceExhaustionErrors:
    """Test error handling for resource exhaustion."""

    @patch("src.transcribe_audio.librosa.load")
    def test_transcribe_audio_memory_exhaustion(
        self, mock_librosa_load, mock_processor, mock_model
    ):
        """Test handling of memory exhaustion during transcription."""
        import numpy as np

        # Simulate very large audio file
        mock_librosa_load.return_value = (np.zeros(1000000), 16000)

        mock_processor.side_effect = MemoryError("Insufficient memory for processing")

        audio_path = Path("/fake/path/huge_file.mp3")

        with pytest.raises(MemoryError, match="Insufficient memory for processing"):
            transcribe_audio(
                audio_path, mock_processor, mock_model, "cpu", "openai/whisper-small"
            )

    def test_save_results_large_dataset_memory_error(self, temp_dir):
        """Test handling of memory errors when saving large datasets."""
        # Create a very large dataset
        large_records = []
        for i in range(1000):
            large_records.append(
                {
                    "file_id": f"large_{i}",
                    "filename": f"large_{i}.mp3",
                    "file_size_bytes": 1000000,
                    "transcription_time_seconds": 10.0,
                    "transcription_text": "A" * 10000,  # Large text
                    "model_id": "openai/whisper-small",
                    "started_at": "2024-01-01T00:00:00Z",
                    "processed_at": "2024-01-01T00:00:10Z",
                }
            )

        output_file = temp_dir / "large.csv"

        with patch("pandas.DataFrame") as mock_df_cls:
            mock_df_cls.side_effect = MemoryError(
                "Cannot create DataFrame: insufficient memory"
            )

            with pytest.raises(
                MemoryError, match="Cannot create DataFrame: insufficient memory"
            ):
                save_results(large_records, "csv", output_file)
