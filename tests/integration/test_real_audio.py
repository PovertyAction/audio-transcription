"""Integration tests using real audio files."""

from unittest.mock import patch

import pytest

from src.transcribe_audio import (
    get_audio_files,
    get_file_size,
    transcribe_audio,
)


@pytest.mark.integration
class TestRealAudioFileProcessing:
    """Test processing with real audio files."""

    def test_real_audio_file_detection(self, test_audio_dir):
        """Test that real audio files are detected correctly."""
        if not test_audio_dir.exists():
            pytest.skip("No test audio files available")

        with patch("src.transcribe_audio.AUDIO_DIR", test_audio_dir):
            audio_files = get_audio_files()

        # Should find actual audio files
        assert isinstance(audio_files, list)
        if audio_files:  # Only test if files exist
            for audio_file in audio_files:
                assert audio_file.exists()
                assert audio_file.suffix.lower() in {
                    ".mp3",
                    ".wav",
                    ".flac",
                    ".m4a",
                    ".ogg",
                }

    def test_real_audio_file_sizes(self, test_audio_dir):
        """Test file size calculation on real audio files."""
        if not test_audio_dir.exists():
            pytest.skip("No test audio files available")

        audio_files = list(test_audio_dir.glob("*"))
        audio_files = [
            f
            for f in audio_files
            if f.suffix.lower() in {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
        ]

        if not audio_files:
            pytest.skip("No audio files found in test assets")

        for audio_file in audio_files:
            file_size = get_file_size(audio_file)
            assert file_size > 0
            assert isinstance(file_size, int)
            # Test files should be reasonably sized (< 10MB)
            assert file_size < 10 * 1024 * 1024

    @pytest.mark.slow
    def test_real_audio_transcription_with_tiny_model(
        self, test_audio_dir, mock_processor, mock_model
    ):
        """Test transcription with real audio files using mocked model (to avoid downloading)."""
        if not test_audio_dir.exists():
            pytest.skip("No test audio files available")

        audio_files = list(test_audio_dir.glob("*"))
        audio_files = [
            f
            for f in audio_files
            if f.suffix.lower() in {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
        ]

        if not audio_files:
            pytest.skip("No audio files found in test assets")

        # Use first available audio file
        audio_file = audio_files[0]

        # Mock the model loading to avoid downloading
        with patch("src.transcribe_audio.librosa.load") as mock_librosa:
            import numpy as np

            # Mock librosa to return realistic audio data
            mock_librosa.return_value = (
                np.random.random(16000),
                16000,
            )  # 1 second of audio

            # Mock processor inputs
            from unittest.mock import Mock

            import torch

            mock_inputs = Mock()
            mock_inputs.input_features = torch.tensor(
                [[1, 2, 3, 4, 5]], dtype=torch.float32
            )
            mock_inputs.to.return_value = mock_inputs
            mock_processor.return_value = mock_inputs

            # Mock model generation
            mock_model.generate.return_value = torch.tensor([[10, 20, 30, 40]])
            mock_processor.batch_decode.return_value = [
                f"Transcription of {audio_file.name}"
            ]

            # Test transcription
            decoded_outputs, elapsed_time, started_at = transcribe_audio(
                audio_file, mock_processor, mock_model, "cpu", "openai/whisper-tiny"
            )

            assert len(decoded_outputs) > 0
            assert isinstance(decoded_outputs[0], str)
            assert elapsed_time > 0
            assert started_at is not None

    def test_audio_file_formats_supported(self, test_audio_dir):
        """Test that various audio formats are properly recognized."""
        if not test_audio_dir.exists():
            pytest.skip("No test audio files available")

        # Expected audio extensions
        supported_extensions = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}

        with patch("src.transcribe_audio.AUDIO_DIR", test_audio_dir):
            audio_files = get_audio_files()

        for audio_file in audio_files:
            assert audio_file.suffix.lower() in supported_extensions

    @pytest.mark.slow
    def test_audio_file_loading_with_librosa(self, test_audio_dir):
        """Test that real audio files can be loaded with librosa."""
        pytest.importorskip("librosa")  # Skip if librosa not available

        if not test_audio_dir.exists():
            pytest.skip("No test audio files available")

        audio_files = list(test_audio_dir.glob("*"))
        audio_files = [
            f for f in audio_files if f.suffix.lower() in {".mp3", ".wav", ".flac"}
        ]

        if not audio_files:
            pytest.skip("No audio files found in test assets")

        import librosa

        for audio_file in audio_files[
            :2
        ]:  # Test first 2 files to avoid long test times
            try:
                audio_data, sample_rate = librosa.load(str(audio_file), sr=16000)
                assert len(audio_data) > 0
                assert sample_rate == 16000
                assert audio_data.dtype in ["float32", "float64"]
            except Exception as e:
                pytest.fail(f"Failed to load {audio_file}: {e}")


@pytest.mark.integration
class TestRealAudioWorkflow:
    """Test complete workflow with real audio files."""

    def test_end_to_end_workflow_dry_run(self, test_audio_dir, temp_dir):
        """Test the complete workflow without actual model loading."""
        if not test_audio_dir.exists():
            pytest.skip("No test audio files available")

        # Test the workflow components individually
        with patch("src.transcribe_audio.AUDIO_DIR", test_audio_dir):
            audio_files = get_audio_files()

        if not audio_files:
            pytest.skip("No audio files found in test assets")

        # Test file processing components
        for audio_file in audio_files[:1]:  # Test with first file
            file_size = get_file_size(audio_file)
            assert file_size > 0

            from src.transcribe_audio import generate_file_id

            file_id = generate_file_id(audio_file.name, file_size)
            assert len(file_id) == 16
            assert file_id.isalnum()


@pytest.mark.integration
class TestAudioAssets:
    """Test audio asset management."""

    def test_test_assets_directory_structure(self, test_assets_dir, test_audio_dir):
        """Test that test assets directory has proper structure."""
        assert test_assets_dir.exists()
        assert test_audio_dir.exists() or not any(test_assets_dir.iterdir()), (
            "Assets directory exists but audio subdirectory is missing"
        )

    def test_audio_files_are_reasonable_size(self, test_audio_dir):
        """Test that test audio files are reasonably sized."""
        if not test_audio_dir.exists():
            pytest.skip("No test audio files available")

        audio_files = list(test_audio_dir.glob("*"))
        audio_files = [
            f
            for f in audio_files
            if f.suffix.lower() in {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
        ]

        for audio_file in audio_files:
            file_size = audio_file.stat().st_size
            # Test files should be small (< 5MB) to avoid repo bloat
            assert file_size < 5 * 1024 * 1024, (
                f"{audio_file.name} is too large: {file_size} bytes"
            )
            # But not empty
            assert file_size > 100, f"{audio_file.name} is too small: {file_size} bytes"

    def test_readme_exists_in_assets(self, test_assets_dir):
        """Test that README exists to document test assets."""
        readme_file = test_assets_dir / "README.md"
        assert readme_file.exists()

        readme_content = readme_file.read_text()
        assert "Test Assets" in readme_content
        assert len(readme_content) > 100  # Should have meaningful content
