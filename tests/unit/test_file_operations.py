"""Tests for file operations in transcribe_audio module."""

from pathlib import Path
from unittest.mock import patch

from src.transcribe_audio import (
    OUTPUT_DIR,
    OUTPUT_FORMATS,
    generate_file_id,
    get_audio_files,
    get_file_size,
    get_output_filename,
)


class TestFileIdGeneration:
    """Test file ID generation functionality."""

    def test_generate_file_id_consistent(self):
        """Test that file ID generation is consistent for same inputs."""
        filename = "test.mp3"
        file_size = 1024

        id1 = generate_file_id(filename, file_size)
        id2 = generate_file_id(filename, file_size)

        assert id1 == id2
        assert len(id1) == 16  # Should be 16 characters (truncated SHA256)

    def test_generate_file_id_different_inputs(self):
        """Test that different inputs produce different file IDs."""
        id1 = generate_file_id("test1.mp3", 1024)
        id2 = generate_file_id("test2.mp3", 1024)
        id3 = generate_file_id("test1.mp3", 2048)

        assert id1 != id2
        assert id1 != id3
        assert id2 != id3

    def test_generate_file_id_format(self):
        """Test that file ID has expected format."""
        file_id = generate_file_id("test.mp3", 1024)

        assert isinstance(file_id, str)
        assert len(file_id) == 16
        assert file_id.isalnum()


class TestAudioFileDiscovery:
    """Test audio file discovery functionality."""

    def test_get_audio_files_filters_correctly(self, mock_audio_dir):
        """Test that only audio files are returned."""
        with patch("src.transcribe_audio.AUDIO_DIR", mock_audio_dir):
            audio_files = get_audio_files()

        # Should only return audio files, sorted
        expected_files = ["test1.mp3", "test2.wav", "test3.flac"]
        actual_filenames = [f.name for f in audio_files]

        assert len(actual_filenames) == 3
        assert all(name in expected_files for name in actual_filenames)
        assert "readme.txt" not in actual_filenames

    def test_get_audio_files_empty_directory(self, temp_dir):
        """Test behavior with empty audio directory."""
        empty_dir = temp_dir / "empty_audio"
        empty_dir.mkdir()

        with patch("src.transcribe_audio.AUDIO_DIR", empty_dir):
            audio_files = get_audio_files()

        assert len(audio_files) == 0

    def test_get_audio_files_case_insensitive(self, temp_dir):
        """Test that audio file detection is case insensitive."""
        audio_dir = temp_dir / "audio"
        audio_dir.mkdir()

        # Create files with various case extensions
        (audio_dir / "test1.MP3").write_text("content")
        (audio_dir / "test2.WAV").write_text("content")
        (audio_dir / "test3.Flac").write_text("content")

        with patch("src.transcribe_audio.AUDIO_DIR", audio_dir):
            audio_files = get_audio_files()

        assert len(audio_files) == 3


class TestFileSizeOperations:
    """Test file size operations."""

    def test_get_file_size(self, temp_dir):
        """Test file size calculation."""
        test_file = temp_dir / "test.txt"
        content = "Hello World"
        test_file.write_text(content)

        size = get_file_size(test_file)

        assert size == len(content.encode())
        assert isinstance(size, int)

    def test_get_file_size_empty_file(self, temp_dir):
        """Test file size for empty file."""
        test_file = temp_dir / "empty.txt"
        test_file.write_text("")

        size = get_file_size(test_file)

        assert size == 0

    def test_get_file_size_large_content(self, temp_dir):
        """Test file size for larger content."""
        test_file = temp_dir / "large.txt"
        content = "A" * 1000
        test_file.write_text(content)

        size = get_file_size(test_file)

        assert size == 1000


class TestOutputFilenames:
    """Test output filename generation."""

    def test_get_output_filename_all_formats(self):
        """Test output filename generation for all supported formats."""
        for format_name in OUTPUT_FORMATS:
            filename = get_output_filename(format_name)

            assert isinstance(filename, Path)
            assert filename.parent == OUTPUT_DIR
            assert filename.name.startswith("transcribed_audio")
            assert filename.suffix == OUTPUT_FORMATS[format_name]["extension"]

    def test_get_output_filename_consistency(self):
        """Test that filename generation is consistent."""
        filename1 = get_output_filename("csv")
        filename2 = get_output_filename("csv")

        assert filename1 == filename2

    def test_get_output_filename_different_formats(self):
        """Test that different formats produce different filenames."""
        csv_file = get_output_filename("csv")
        json_file = get_output_filename("json")
        parquet_file = get_output_filename("parquet")
        duckdb_file = get_output_filename("duckdb")

        filenames = [csv_file, json_file, parquet_file, duckdb_file]
        assert len(set(filenames)) == 4  # All should be unique
