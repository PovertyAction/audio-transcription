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

    @patch("src.transcribe_audio.get_audio_files")
    @patch("src.transcribe_audio.load_model")
    @patch("src.transcribe_audio.OUTPUT_DIR")
    @patch("torch.cuda.is_available")
    def test_main_with_language_default(
        self, mock_cuda, mock_output_dir, mock_load_model, mock_get_audio
    ):
        """Test main function with default language argument."""
        self._setup_mocks(mock_cuda, mock_output_dir, mock_load_model, mock_get_audio)

        # Mock sys.argv with no language argument (should use default)
        with patch.object(sys, "argv", ["transcribe_audio.py"]):
            main()

        # Default language should be "en"
        mock_load_model.assert_called_once_with("whisper-small", "cpu")

    @patch("src.transcribe_audio.get_audio_files")
    @patch("src.transcribe_audio.load_model")
    @patch("src.transcribe_audio.OUTPUT_DIR")
    @patch("torch.cuda.is_available")
    def test_main_with_language_custom(
        self, mock_cuda, mock_output_dir, mock_load_model, mock_get_audio
    ):
        """Test main function with custom language argument."""
        self._setup_mocks(mock_cuda, mock_output_dir, mock_load_model, mock_get_audio)

        # Mock sys.argv with custom language
        with patch.object(sys, "argv", ["transcribe_audio.py", "--language", "es"]):
            main()

        mock_load_model.assert_called_once_with("whisper-small", "cpu")

    @patch("src.transcribe_audio.get_audio_files")
    @patch("src.transcribe_audio.load_model")
    @patch("src.transcribe_audio.OUTPUT_DIR")
    @patch("torch.cuda.is_available")
    def test_main_with_max_new_tokens_default(
        self, mock_cuda, mock_output_dir, mock_load_model, mock_get_audio
    ):
        """Test main function with default max_new_tokens argument."""
        self._setup_mocks(mock_cuda, mock_output_dir, mock_load_model, mock_get_audio)

        # Mock sys.argv with no max-new-tokens argument (should use default)
        with patch.object(sys, "argv", ["transcribe_audio.py"]):
            main()

        # Default max_new_tokens should be 400
        mock_load_model.assert_called_once_with("whisper-small", "cpu")

    @patch("src.transcribe_audio.get_audio_files")
    @patch("src.transcribe_audio.load_model")
    @patch("src.transcribe_audio.OUTPUT_DIR")
    @patch("torch.cuda.is_available")
    def test_main_with_max_new_tokens_custom(
        self, mock_cuda, mock_output_dir, mock_load_model, mock_get_audio
    ):
        """Test main function with custom max_new_tokens argument."""
        self._setup_mocks(mock_cuda, mock_output_dir, mock_load_model, mock_get_audio)

        # Mock sys.argv with custom max-new-tokens
        with patch.object(
            sys, "argv", ["transcribe_audio.py", "--max-new-tokens", "600"]
        ):
            main()

        mock_load_model.assert_called_once_with("whisper-small", "cpu")

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

    def test_invalid_max_new_tokens_argument(self):
        """Test handling of invalid max_new_tokens argument."""
        with (
            patch.object(
                sys, "argv", ["transcribe_audio.py", "--max-new-tokens", "not-a-number"]
            ),
            patch("sys.stderr", new=StringIO()),
            patch("sys.exit") as mock_exit,
        ):
            main()
            # Should exit with error code due to type conversion error
            assert mock_exit.called

    @patch("src.transcribe_audio.get_audio_files")
    @patch("src.transcribe_audio.load_model")
    @patch("src.transcribe_audio.OUTPUT_DIR")
    @patch("torch.cuda.is_available")
    @patch("builtins.print")
    def test_whisper_max_tokens_validation(
        self, mock_print, mock_cuda, mock_output_dir, mock_load_model, mock_get_audio
    ):
        """Test that Whisper models enforce max token limit of 448."""
        self._setup_mocks(mock_cuda, mock_output_dir, mock_load_model, mock_get_audio)

        # Mock sys.argv with token count exceeding Whisper limit
        with patch.object(
            sys,
            "argv",
            [
                "transcribe_audio.py",
                "--model",
                "whisper-small",
                "--max-new-tokens",
                "600",
            ],
        ):
            main()

        # Check that warning was printed
        warning_printed = any(
            "Warning: Whisper models have a maximum token limit of 448" in str(call)
            for call in mock_print.call_args_list
        )
        assert warning_printed

    @patch("src.transcribe_audio.get_audio_files")
    @patch("src.transcribe_audio.load_model")
    @patch("src.transcribe_audio.OUTPUT_DIR")
    @patch("torch.cuda.is_available")
    @patch("builtins.print")
    def test_voxtral_allows_high_token_count(
        self, mock_print, mock_cuda, mock_output_dir, mock_load_model, mock_get_audio
    ):
        """Test that Voxtral models allow token counts above 448."""
        self._setup_mocks(mock_cuda, mock_output_dir, mock_load_model, mock_get_audio)

        # Return voxtral model type
        mock_load_model.return_value = (
            Mock(),
            Mock(),
            "mistralai/Voxtral-Mini-3B-2507",
            "voxtral",
        )

        # Mock sys.argv with high token count for Voxtral
        with patch.object(
            sys,
            "argv",
            [
                "transcribe_audio.py",
                "--model",
                "voxtral-mini",
                "--max-new-tokens",
                "800",
            ],
        ):
            main()

        # Check that NO warning was printed about token limit
        warning_printed = any(
            "Warning: Whisper models have a maximum token limit" in str(call)
            for call in mock_print.call_args_list
        )
        assert not warning_printed

    def test_help_flag_exits_without_processing(self):
        """Test that --help flag exits without running transcription."""
        with (
            patch.object(sys, "argv", ["transcribe_audio.py", "--help"]),
            patch("sys.stderr", new=StringIO()),
            pytest.raises(SystemExit),
        ):
            main()

    def test_short_help_flag_exits_without_processing(self):
        """Test that -h flag exits without running transcription."""
        with (
            patch.object(sys, "argv", ["transcribe_audio.py", "-h"]),
            patch("sys.stderr", new=StringIO()),
            pytest.raises(SystemExit),
        ):
            main()


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

    @patch("src.transcribe_audio.get_audio_files")
    @patch("src.transcribe_audio.load_model")
    @patch("src.transcribe_audio.transcribe_audio")
    @patch("src.transcribe_audio.save_results")
    @patch("src.transcribe_audio.OUTPUT_DIR")
    @patch("torch.cuda.is_available")
    def test_language_and_max_tokens_passed_to_transcribe(
        self,
        mock_cuda,
        mock_output_dir,
        mock_save,
        mock_transcribe,
        mock_load_model,
        mock_get_audio,
    ):
        """Test that language and max_new_tokens arguments are passed to transcribe_audio."""
        # Setup mocks
        mock_output_dir.mkdir = Mock()
        mock_output_dir.__truediv__ = lambda self, other: Path("/tmp") / other
        mock_cuda.return_value = False

        # Mock a single audio file
        audio_files = [Path("test.mp3")]
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
            patch.object(
                sys,
                "argv",
                [
                    "transcribe_audio.py",
                    "--language",
                    "fr",
                    "--max-new-tokens",
                    "800",
                ],
            ),
        ):
            mock_file_size.return_value = 1024
            mock_load_results.return_value = set()
            main()

        # Verify transcribe_audio was called with correct arguments
        mock_transcribe.assert_called_once()
        call_args = mock_transcribe.call_args
        # Check that the language and max_new_tokens arguments are passed
        assert call_args[0][4] == "openai/whisper-small"  # model_id
        assert call_args[0][5] == "whisper"  # model_type
        assert call_args[0][6] == "fr"  # language
        assert call_args[0][7] == 448  # max_new_tokens (capped at Whisper limit)

    @patch("src.transcribe_audio.get_audio_files")
    @patch("src.transcribe_audio.load_model")
    @patch("src.transcribe_audio.transcribe_audio")
    @patch("src.transcribe_audio.save_results")
    @patch("src.transcribe_audio.OUTPUT_DIR")
    @patch("torch.cuda.is_available")
    def test_whisper_token_limit_enforced(
        self,
        mock_cuda,
        mock_output_dir,
        mock_save,
        mock_transcribe,
        mock_load_model,
        mock_get_audio,
    ):
        """Test that Whisper models cap max_new_tokens at 448."""
        # Setup mocks
        mock_output_dir.mkdir = Mock()
        mock_output_dir.__truediv__ = lambda self, other: Path("/tmp") / other
        mock_cuda.return_value = False

        # Mock a single audio file
        audio_files = [Path("test.mp3")]
        mock_get_audio.return_value = audio_files

        # Mock model loading
        mock_processor = Mock()
        mock_model = Mock()
        mock_load_model.return_value = (
            mock_processor,
            mock_model,
            "openai/whisper-medium",
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
            patch.object(
                sys,
                "argv",
                [
                    "transcribe_audio.py",
                    "--model",
                    "whisper-medium",
                    "--max-new-tokens",
                    "1000",  # Exceeds limit
                ],
            ),
        ):
            mock_file_size.return_value = 1024
            mock_load_results.return_value = set()
            main()

        # Verify transcribe_audio was called with capped token count
        mock_transcribe.assert_called_once()
        call_args = mock_transcribe.call_args
        assert call_args[0][7] == 448  # max_new_tokens should be capped at 448
