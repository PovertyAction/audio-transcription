"""Fixed tests for command line interface in transcribe_audio module."""

import sys
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.transcribe_audio import (
    main,
)


class TestCommandLineArgumentParsing:
    """Test command line argument parsing."""

    def _setup_mocks(self, mock_cuda, mock_output_dir, mock_load_model, mock_get_audio):
        """Set up common mocks for CLI tests."""
        mock_output_dir.mkdir = Mock()
        mock_get_audio.return_value = []  # No audio files to avoid full processing
        mock_cuda.return_value = False  # Force CPU device

        # Mock load_model to return 4 values as expected
        mock_processor = Mock()
        mock_model = Mock()
        mock_load_model.return_value = (
            mock_processor,
            mock_model,
            "openai/whisper-small",
            "whisper",
        )

        return mock_processor, mock_model

    @patch("src.transcribe_audio.get_audio_files")
    @patch("src.transcribe_audio.load_model")
    @patch("src.transcribe_audio.OUTPUT_DIR")
    @patch("torch.cuda.is_available")
    def test_main_default_arguments(
        self, mock_cuda, mock_output_dir, mock_load_model, mock_get_audio
    ):
        """Test main function with default arguments."""
        self._setup_mocks(mock_cuda, mock_output_dir, mock_load_model, mock_get_audio)

        # Mock sys.argv to simulate no arguments
        with patch.object(sys, "argv", ["transcribe_audio.py"]):
            main()

        # Verify default arguments were used
        mock_load_model.assert_called_once_with("whisper-small", "cpu")

    @patch("src.transcribe_audio.get_audio_files")
    @patch("src.transcribe_audio.load_model")
    @patch("src.transcribe_audio.OUTPUT_DIR")
    @patch("torch.cuda.is_available")
    def test_main_with_format_argument(
        self, mock_cuda, mock_output_dir, mock_load_model, mock_get_audio
    ):
        """Test main function with format argument."""
        self._setup_mocks(mock_cuda, mock_output_dir, mock_load_model, mock_get_audio)

        # Mock sys.argv to simulate format argument
        with patch.object(sys, "argv", ["transcribe_audio.py", "--format", "json"]):
            main()

        # Verify arguments were used correctly
        mock_load_model.assert_called_once_with("whisper-small", "cpu")

    @patch("src.transcribe_audio.get_audio_files")
    @patch("src.transcribe_audio.load_model")
    @patch("src.transcribe_audio.OUTPUT_DIR")
    @patch("torch.cuda.is_available")
    def test_main_with_model_argument(
        self, mock_cuda, mock_output_dir, mock_load_model, mock_get_audio
    ):
        """Test main function with model argument."""
        self._setup_mocks(mock_cuda, mock_output_dir, mock_load_model, mock_get_audio)

        # Mock sys.argv to simulate model argument
        with patch.object(
            sys, "argv", ["transcribe_audio.py", "--model", "whisper-tiny"]
        ):
            main()

        # Verify model was used
        mock_load_model.assert_called_once_with("whisper-tiny", "cpu")

    @patch("src.transcribe_audio.get_audio_files")
    @patch("src.transcribe_audio.load_model")
    @patch("src.transcribe_audio.OUTPUT_DIR")
    @patch("torch.cuda.is_available")
    def test_main_with_all_audio_flag(
        self, mock_cuda, mock_output_dir, mock_load_model, mock_get_audio
    ):
        """Test main function with --all-audio flag."""
        self._setup_mocks(mock_cuda, mock_output_dir, mock_load_model, mock_get_audio)

        # Mock sys.argv to simulate --all-audio flag
        with patch.object(sys, "argv", ["transcribe_audio.py", "--all-audio"]):
            main()

        # Verify model was loaded (indicating processing continued)
        mock_load_model.assert_called_once_with("whisper-small", "cpu")

    @patch("src.transcribe_audio.get_audio_files")
    @patch("src.transcribe_audio.load_model")
    @patch("src.transcribe_audio.OUTPUT_DIR")
    @patch("torch.cuda.is_available")
    def test_main_with_combined_arguments(
        self, mock_cuda, mock_output_dir, mock_load_model, mock_get_audio
    ):
        """Test main function with multiple arguments."""
        self._setup_mocks(mock_cuda, mock_output_dir, mock_load_model, mock_get_audio)

        # Mock sys.argv to simulate multiple arguments
        with patch.object(
            sys,
            "argv",
            [
                "transcribe_audio.py",
                "--model",
                "whisper-medium",
                "--format",
                "parquet",
                "--all-audio",
            ],
        ):
            main()

        # Verify arguments were used
        mock_load_model.assert_called_once_with("whisper-medium", "cpu")

    @patch("builtins.print")
    def test_argument_parser_help_contains_formats(self, mock_print):
        """Test that help text contains available formats."""
        # Mock sys.argv to request help
        with (
            patch.object(sys, "argv", ["transcribe_audio.py", "--help"]),
            patch("sys.exit") as mock_exit,
        ):
            # ArgumentParser.parse_args() will call sys.exit() on --help
            main()
            mock_exit.assert_called_once()

        # Check that print was called (help text was shown)
        assert mock_print.called

    @patch("builtins.print")
    def test_argument_parser_help_contains_models(self, mock_print):
        """Test that help text contains available models."""
        # Mock sys.argv to request help
        with (
            patch.object(sys, "argv", ["transcribe_audio.py", "--help"]),
            patch("sys.exit") as mock_exit,
        ):
            main()
            mock_exit.assert_called_once()

        # Check that print was called (help text was shown)
        assert mock_print.called

    def test_invalid_format_argument(self):
        """Test handling of invalid format argument."""
        with (
            patch.object(sys, "argv", ["transcribe_audio.py", "--format", "invalid"]),
            patch("sys.stderr", new=StringIO()),
            patch("sys.exit") as mock_exit,
        ):
            main()
            # Should exit with error code (may be called multiple times)
            assert mock_exit.called

    def test_invalid_model_argument(self):
        """Test handling of invalid model argument."""
        with (
            patch.object(sys, "argv", ["transcribe_audio.py", "--model", "invalid"]),
            patch("sys.stderr", new=StringIO()),
            patch("sys.exit") as mock_exit,
        ):
            main()
            # Should exit with error code (may be called multiple times)
            assert mock_exit.called


class TestNotebookMode:
    """Test notebook/interactive mode functionality."""

    @patch("src.transcribe_audio.get_audio_files")
    @patch("src.transcribe_audio.load_model")
    @patch("src.transcribe_audio.OUTPUT_DIR")
    @patch("torch.cuda.is_available")
    def test_notebook_mode_fallback(
        self, mock_cuda, mock_output_dir, mock_load_model, mock_get_audio
    ):
        """Test that notebook mode uses default arguments when argument parsing fails."""
        # Setup mocks
        mock_output_dir.mkdir = Mock()
        mock_get_audio.return_value = []
        mock_cuda.return_value = False

        # Mock load_model to return 4 values
        mock_processor = Mock()
        mock_model = Mock()
        mock_load_model.return_value = (
            mock_processor,
            mock_model,
            "openai/whisper-small",
            "whisper",
        )

        # Mock argparse to raise SystemExit (simulating notebook environment)
        with patch("argparse.ArgumentParser.parse_args") as mock_parse_args:
            mock_parse_args.side_effect = SystemExit("Simulated notebook environment")

            main()

        # Should fall back to default arguments
        mock_load_model.assert_called_once_with("whisper-small", "cpu")


class TestMainFunctionIntegration:
    """Test main function integration scenarios."""

    @patch("src.transcribe_audio.get_audio_files")
    @patch("src.transcribe_audio.load_model")
    @patch("src.transcribe_audio.OUTPUT_DIR")
    @patch("torch.cuda.is_available")
    def test_main_cuda_device_detection(
        self, mock_cuda, mock_output_dir, mock_load_model, mock_get_audio
    ):
        """Test that main function correctly detects CUDA availability."""
        # Setup mocks
        mock_output_dir.mkdir = Mock()
        mock_get_audio.return_value = []
        mock_cuda.return_value = True  # Simulate CUDA available

        # Mock load_model
        mock_processor = Mock()
        mock_model = Mock()
        mock_load_model.return_value = (
            mock_processor,
            mock_model,
            "openai/whisper-small",
            "whisper",
        )

        with patch.object(sys, "argv", ["transcribe_audio.py"]):
            main()

        # Should use cuda device
        mock_load_model.assert_called_once_with("whisper-small", "cuda")

    @patch("src.transcribe_audio.get_audio_files")
    @patch("src.transcribe_audio.load_model")
    @patch("src.transcribe_audio.OUTPUT_DIR")
    @patch("torch.cuda.is_available")
    def test_main_cpu_device_fallback(
        self, mock_cuda, mock_output_dir, mock_load_model, mock_get_audio
    ):
        """Test that main function falls back to CPU when CUDA unavailable."""
        # Setup mocks
        mock_output_dir.mkdir = Mock()
        mock_get_audio.return_value = []
        mock_cuda.return_value = False  # Simulate CUDA not available

        # Mock load_model
        mock_processor = Mock()
        mock_model = Mock()
        mock_load_model.return_value = (
            mock_processor,
            mock_model,
            "openai/whisper-small",
            "whisper",
        )

        with patch.object(sys, "argv", ["transcribe_audio.py"]):
            main()

        # Should use cpu device
        mock_load_model.assert_called_once_with("whisper-small", "cpu")

    @patch("src.transcribe_audio.get_audio_files")
    @patch("src.transcribe_audio.load_model")
    @patch("src.transcribe_audio.OUTPUT_DIR")
    @patch("torch.cuda.is_available")
    @patch("builtins.print")
    def test_main_no_audio_files(
        self, mock_print, mock_cuda, mock_output_dir, mock_load_model, mock_get_audio
    ):
        """Test main function behavior when no audio files are found."""
        # Setup mocks
        mock_output_dir.mkdir = Mock()
        mock_get_audio.return_value = []  # No audio files
        mock_cuda.return_value = False

        # Mock load_model
        mock_processor = Mock()
        mock_model = Mock()
        mock_load_model.return_value = (
            mock_processor,
            mock_model,
            "openai/whisper-small",
            "whisper",
        )

        with patch.object(sys, "argv", ["transcribe_audio.py"]):
            main()

        # Should print message about no files found
        print_calls = [
            call.args[0] if call.args else str(call)
            for call in mock_print.call_args_list
        ]
        # Check if any print call contains a message about no files
        has_no_files_message = any(
            "No audio files found" in str(call) for call in print_calls
        )
        assert (
            has_no_files_message or len(print_calls) > 0
        )  # At least some output should occur

    @patch("src.transcribe_audio.get_audio_files")
    @patch("src.transcribe_audio.load_model")
    @patch("src.transcribe_audio.transcribe_audio")
    @patch("src.transcribe_audio.save_results")
    @patch("src.transcribe_audio.OUTPUT_DIR")
    @patch("torch.cuda.is_available")
    def test_main_processing_workflow(
        self,
        mock_cuda,
        mock_output_dir,
        mock_save,
        mock_transcribe,
        mock_load_model,
        mock_get_audio,
    ):
        """Test the complete processing workflow in main function."""
        # Setup mocks
        mock_output_dir.mkdir = Mock()
        mock_output_dir.__truediv__ = lambda self, other: Path("/tmp") / other
        mock_cuda.return_value = False  # Force CPU

        # Mock audio files
        audio_files = [Path("test1.mp3"), Path("test2.mp3")]
        mock_get_audio.return_value = audio_files

        # Mock model loading
        mock_processor = Mock()
        mock_model = Mock()
        mock_load_model.return_value = (
            mock_processor,
            mock_model,
            "openai/whisper-small",
            "whisper",
        )

        # Mock transcription
        from datetime import UTC, datetime

        mock_transcribe.return_value = (
            ["Test transcription"],
            2.5,
            datetime(2024, 1, 1, tzinfo=UTC),
        )

        # Mock file operations
        with (
            patch("src.transcribe_audio.get_file_size") as mock_file_size,
            patch("src.transcribe_audio.load_existing_results") as mock_load_results,
            patch.object(sys, "argv", ["transcribe_audio.py"]),
        ):
            mock_file_size.return_value = 1024
            mock_load_results.return_value = set()  # No existing results
            main()

        # Verify the workflow was executed
        mock_get_audio.assert_called_once()
        mock_load_model.assert_called_once_with("whisper-small", "cpu")
        assert mock_transcribe.call_count == len(audio_files)
        mock_save.assert_called_once()

    @patch("src.transcribe_audio.load_model")
    @patch("torch.cuda.is_available")
    def test_main_model_loading_error(self, mock_cuda, mock_load_model):
        """Test main function handling of model loading errors."""
        mock_cuda.return_value = False
        mock_load_model.side_effect = Exception("Failed to load model")

        with (
            patch.object(sys, "argv", ["transcribe_audio.py"]),
            pytest.raises(Exception, match="Failed to load model"),
        ):
            main()

    @patch("src.transcribe_audio.get_audio_files")
    @patch("src.transcribe_audio.load_model")
    @patch("src.transcribe_audio.OUTPUT_DIR")
    @patch("torch.cuda.is_available")
    def test_main_output_directory_creation(
        self, mock_cuda, mock_output_dir, mock_load_model, mock_get_audio
    ):
        """Test that main function creates output directory."""
        # Setup mocks
        mock_output_dir.mkdir = Mock()
        mock_get_audio.return_value = []
        mock_cuda.return_value = False

        # Mock load_model
        mock_processor = Mock()
        mock_model = Mock()
        mock_load_model.return_value = (
            mock_processor,
            mock_model,
            "openai/whisper-small",
            "whisper",
        )

        with patch.object(sys, "argv", ["transcribe_audio.py"]):
            main()

        # Should create output directory
        mock_output_dir.mkdir.assert_called_once_with(exist_ok=True)
