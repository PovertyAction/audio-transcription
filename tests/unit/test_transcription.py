"""Tests for transcription functionality in transcribe_audio module."""

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from src.transcribe_audio import transcribe_audio


class TestTranscriptionCore:
    """Test core transcription functionality."""

    @patch("src.transcribe_audio.librosa")
    def test_transcribe_audio_basic_functionality(
        self, mock_librosa, mock_processor, mock_model
    ):
        """Test basic transcription functionality with mocked components."""
        # Setup mocks
        mock_audio_data = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        mock_librosa.load.return_value = (mock_audio_data, 16000)

        # Mock processor inputs
        mock_inputs = Mock()
        mock_inputs.input_features = torch.tensor(
            [[1, 2, 3, 4, 5]], dtype=torch.float32
        )
        mock_inputs.to.return_value = mock_inputs
        mock_processor.return_value = mock_inputs

        # Mock model generation
        mock_model.generate.return_value = torch.tensor([[10, 20, 30, 40]])
        mock_processor.batch_decode.return_value = ["This is a test transcription."]

        audio_path = Path("/fake/path/test.mp3")

        # Call transcribe_audio
        decoded_outputs, elapsed_time, started_at = transcribe_audio(
            audio_path,
            mock_processor,
            mock_model,
            "cpu",
            "openai/whisper-small",
            "whisper",
        )

        # Verify results
        assert decoded_outputs == ["This is a test transcription."]
        assert isinstance(elapsed_time, float)
        assert elapsed_time > 0
        assert isinstance(started_at, datetime)
        assert started_at.tzinfo == UTC

        # Verify mocks were called correctly
        mock_librosa.load.assert_called_once_with(str(audio_path), sr=16000)
        mock_processor.assert_called_once_with(
            mock_audio_data, sampling_rate=16000, return_tensors="pt"
        )
        mock_model.generate.assert_called_once()
        # Just verify that batch_decode was called with correct arguments structure
        assert mock_processor.batch_decode.call_count == 1
        call_args = mock_processor.batch_decode.call_args
        assert call_args[1]["skip_special_tokens"] is True
        # Verify the tensor shape is correct
        assert call_args[0][0].shape == torch.Size([1, 4])

    @patch("src.transcribe_audio.librosa")
    def test_transcribe_audio_cuda_device(
        self, mock_librosa, mock_processor, mock_model
    ):
        """Test transcription with CUDA device handling."""
        mock_audio_data = np.array([0.1, 0.2, 0.3])
        mock_librosa.load.return_value = (mock_audio_data, 16000)

        # Mock processor inputs with proper tensor handling
        mock_inputs = Mock()
        mock_tensor = Mock(spec=torch.Tensor)
        mock_tensor.to.return_value = mock_tensor
        mock_inputs.input_features = mock_tensor
        mock_inputs.to.return_value = mock_inputs
        mock_processor.return_value = mock_inputs

        mock_model.generate.return_value = torch.tensor([[1, 2, 3]])
        mock_processor.batch_decode.return_value = ["CUDA transcription."]

        audio_path = Path("/fake/path/test.mp3")

        decoded_outputs, elapsed_time, started_at = transcribe_audio(
            audio_path,
            mock_processor,
            mock_model,
            "cuda",
            "openai/whisper-small",
            "whisper",
        )

        # Verify CUDA-specific handling
        mock_inputs.to.assert_called_with("cuda")
        mock_tensor.to.assert_called_with(
            torch.float16
        )  # Should convert to float16 for CUDA
        assert decoded_outputs == ["CUDA transcription."]

    @patch("src.transcribe_audio.librosa")
    def test_transcribe_audio_cpu_device(
        self, mock_librosa, mock_processor, mock_model
    ):
        """Test transcription with CPU device handling."""
        mock_audio_data = np.array([0.1, 0.2, 0.3])
        mock_librosa.load.return_value = (mock_audio_data, 16000)

        # Mock processor inputs
        mock_inputs = Mock()
        mock_inputs.input_features = torch.tensor([[1, 2, 3]], dtype=torch.float32)
        mock_inputs.to.return_value = mock_inputs
        mock_processor.return_value = mock_inputs

        mock_model.generate.return_value = torch.tensor([[1, 2, 3]])
        mock_processor.batch_decode.return_value = ["CPU transcription."]

        audio_path = Path("/fake/path/test.mp3")

        decoded_outputs, elapsed_time, started_at = transcribe_audio(
            audio_path,
            mock_processor,
            mock_model,
            "cpu",
            "openai/whisper-small",
            "whisper",
        )

        # Verify CPU handling (no dtype conversion)
        mock_inputs.to.assert_called_with("cpu")
        assert decoded_outputs == ["CPU transcription."]

    @patch("src.transcribe_audio.librosa")
    def test_transcribe_audio_librosa_error(
        self, mock_librosa, mock_processor, mock_model
    ):
        """Test handling of librosa loading errors."""
        mock_librosa.load.side_effect = Exception("Failed to load audio file")

        audio_path = Path("/fake/path/test.mp3")

        with pytest.raises(Exception, match="Failed to load audio file"):
            transcribe_audio(
                audio_path,
                mock_processor,
                mock_model,
                "cpu",
                "openai/whisper-small",
                "whisper",
            )

    @patch("src.transcribe_audio.librosa")
    def test_transcribe_audio_model_error(
        self, mock_librosa, mock_processor, mock_model
    ):
        """Test handling of model generation errors."""
        mock_audio_data = np.array([0.1, 0.2, 0.3])
        mock_librosa.load.return_value = (mock_audio_data, 16000)

        mock_inputs = Mock()
        mock_inputs.input_features = torch.tensor([[1, 2, 3]], dtype=torch.float32)
        mock_inputs.to.return_value = mock_inputs
        mock_processor.return_value = mock_inputs

        mock_model.generate.side_effect = RuntimeError("CUDA out of memory")

        audio_path = Path("/fake/path/test.mp3")

        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            transcribe_audio(
                audio_path,
                mock_processor,
                mock_model,
                "cuda",
                "openai/whisper-small",
                "whisper",
            )

    @patch("src.transcribe_audio.librosa")
    def test_transcribe_audio_processor_error(
        self, mock_librosa, mock_processor, mock_model
    ):
        """Test handling of processor errors."""
        mock_audio_data = np.array([0.1, 0.2, 0.3])
        mock_librosa.load.return_value = (mock_audio_data, 16000)

        mock_processor.side_effect = Exception("Processor failed")

        audio_path = Path("/fake/path/test.mp3")

        with pytest.raises(Exception, match="Processor failed"):
            transcribe_audio(
                audio_path,
                mock_processor,
                mock_model,
                "cpu",
                "openai/whisper-small",
                "whisper",
            )

    @patch("src.transcribe_audio.librosa")
    def test_transcribe_audio_decode_error(
        self, mock_librosa, mock_processor, mock_model
    ):
        """Test handling of decoding errors."""
        mock_audio_data = np.array([0.1, 0.2, 0.3])
        mock_librosa.load.return_value = (mock_audio_data, 16000)

        mock_inputs = Mock()
        mock_inputs.input_features = torch.tensor([[1, 2, 3]], dtype=torch.float32)
        mock_inputs.to.return_value = mock_inputs
        mock_processor.return_value = mock_inputs

        mock_model.generate.return_value = torch.tensor([[1, 2, 3]])
        mock_processor.batch_decode.side_effect = Exception("Decoding failed")

        audio_path = Path("/fake/path/test.mp3")

        with pytest.raises(Exception, match="Decoding failed"):
            transcribe_audio(
                audio_path,
                mock_processor,
                mock_model,
                "cpu",
                "openai/whisper-small",
                "whisper",
            )

    @patch("src.transcribe_audio.librosa")
    def test_transcribe_audio_timing_measurement(
        self, mock_librosa, mock_processor, mock_model
    ):
        """Test that timing measurement is reasonably accurate."""
        mock_audio_data = np.array([0.1, 0.2, 0.3])
        mock_librosa.load.return_value = (mock_audio_data, 16000)

        mock_inputs = Mock()
        mock_inputs.input_features = torch.tensor([[1, 2, 3]], dtype=torch.float32)
        mock_inputs.to.return_value = mock_inputs
        mock_processor.return_value = mock_inputs

        mock_model.generate.return_value = torch.tensor([[1, 2, 3]])
        mock_processor.batch_decode.return_value = ["Test transcription."]

        audio_path = Path("/fake/path/test.mp3")

        decoded_outputs, elapsed_time, started_at = transcribe_audio(
            audio_path,
            mock_processor,
            mock_model,
            "cpu",
            "openai/whisper-small",
            "whisper",
        )

        # Timing should be positive but reasonable (mocked execution should be fast)
        assert elapsed_time > 0
        assert elapsed_time < 1.0  # Should be well under a second for mocked execution

    @patch("src.transcribe_audio.librosa")
    @patch("src.transcribe_audio.datetime")
    def test_transcribe_audio_timestamp_generation(
        self, mock_datetime, mock_librosa, mock_processor, mock_model
    ):
        """Test that timestamp generation uses UTC and is consistent."""
        mock_start_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        mock_datetime.now.return_value = mock_start_time
        mock_datetime.UTC = UTC

        mock_audio_data = np.array([0.1, 0.2, 0.3])
        mock_librosa.load.return_value = (mock_audio_data, 16000)

        mock_inputs = Mock()
        mock_inputs.input_features = torch.tensor([[1, 2, 3]], dtype=torch.float32)
        mock_inputs.to.return_value = mock_inputs
        mock_processor.return_value = mock_inputs

        mock_model.generate.return_value = torch.tensor([[1, 2, 3]])
        mock_processor.batch_decode.return_value = ["Test transcription."]

        audio_path = Path("/fake/path/test.mp3")

        decoded_outputs, elapsed_time, started_at = transcribe_audio(
            audio_path,
            mock_processor,
            mock_model,
            "cpu",
            "openai/whisper-small",
            "whisper",
        )

        assert started_at == mock_start_time
        mock_datetime.now.assert_called_with(UTC)

    @patch("src.transcribe_audio.librosa")
    def test_transcribe_audio_multiple_outputs(
        self, mock_librosa, mock_processor, mock_model
    ):
        """Test handling of multiple transcription outputs."""
        mock_audio_data = np.array([0.1, 0.2, 0.3])
        mock_librosa.load.return_value = (mock_audio_data, 16000)

        mock_inputs = Mock()
        mock_inputs.input_features = torch.tensor([[1, 2, 3]], dtype=torch.float32)
        mock_inputs.to.return_value = mock_inputs
        mock_processor.return_value = mock_inputs

        mock_model.generate.return_value = torch.tensor([[1, 2, 3]])
        # Return multiple decoded outputs
        mock_processor.batch_decode.return_value = [
            "First part of transcription.",
            "Second part of transcription.",
        ]

        audio_path = Path("/fake/path/test.mp3")

        decoded_outputs, elapsed_time, started_at = transcribe_audio(
            audio_path,
            mock_processor,
            mock_model,
            "cpu",
            "openai/whisper-small",
            "whisper",
        )

        assert len(decoded_outputs) == 2
        assert decoded_outputs[0] == "First part of transcription."
        assert decoded_outputs[1] == "Second part of transcription."

    @patch("src.transcribe_audio.librosa")
    def test_transcribe_audio_empty_output(
        self, mock_librosa, mock_processor, mock_model
    ):
        """Test handling of empty transcription output."""
        mock_audio_data = np.array([0.1, 0.2, 0.3])
        mock_librosa.load.return_value = (mock_audio_data, 16000)

        mock_inputs = Mock()
        mock_inputs.input_features = torch.tensor([[1, 2, 3]], dtype=torch.float32)
        mock_inputs.to.return_value = mock_inputs
        mock_processor.return_value = mock_inputs

        mock_model.generate.return_value = torch.tensor([[1, 2, 3]])
        mock_processor.batch_decode.return_value = [""]  # Empty transcription

        audio_path = Path("/fake/path/test.mp3")

        decoded_outputs, elapsed_time, started_at = transcribe_audio(
            audio_path,
            mock_processor,
            mock_model,
            "cpu",
            "openai/whisper-small",
            "whisper",
        )

        assert decoded_outputs == [""]
        assert isinstance(elapsed_time, float)
        assert isinstance(started_at, datetime)
