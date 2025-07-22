"""Tests for model loading functionality in transcribe_audio module."""

from unittest.mock import Mock, patch

import pytest
import torch

from src.transcribe_audio import (
    WHISPER_MODELS,
    load_whisper_model,
)


class TestModelLoading:
    """Test Whisper model loading functionality."""

    @patch("src.transcribe_audio.WhisperProcessor")
    @patch("src.transcribe_audio.WhisperForConditionalGeneration")
    def test_load_whisper_model_valid_model(self, mock_model_cls, mock_processor_cls):
        """Test loading a valid Whisper model."""
        # Setup mocks
        mock_processor = Mock()
        mock_model = Mock()
        mock_processor_cls.from_pretrained.return_value = mock_processor
        mock_model_cls.from_pretrained.return_value = mock_model

        processor, model, model_id = load_whisper_model("whisper-small", "cpu")

        assert processor == mock_processor
        assert model == mock_model
        assert model_id == "openai/whisper-small"

        # Verify correct model ID was used
        mock_processor_cls.from_pretrained.assert_called_once_with(
            "openai/whisper-small"
        )
        mock_model_cls.from_pretrained.assert_called_once_with(
            "openai/whisper-small", torch_dtype=torch.float32
        )
        mock_model.to.assert_called_once_with("cpu")

    @patch("src.transcribe_audio.WhisperProcessor")
    @patch("src.transcribe_audio.WhisperForConditionalGeneration")
    def test_load_whisper_model_cuda_device(self, mock_model_cls, mock_processor_cls):
        """Test loading model with CUDA device."""
        mock_processor = Mock()
        mock_model = Mock()
        mock_processor_cls.from_pretrained.return_value = mock_processor
        mock_model_cls.from_pretrained.return_value = mock_model

        processor, model, model_id = load_whisper_model("whisper-tiny", "cuda")

        # Should use float16 for CUDA
        mock_model_cls.from_pretrained.assert_called_once_with(
            "openai/whisper-tiny", torch_dtype=torch.float16
        )
        mock_model.to.assert_called_once_with("cuda")

    @patch("src.transcribe_audio.WhisperProcessor")
    @patch("src.transcribe_audio.WhisperForConditionalGeneration")
    def test_load_whisper_model_all_models(self, mock_model_cls, mock_processor_cls):
        """Test loading all available Whisper models."""
        mock_processor = Mock()
        mock_model = Mock()
        mock_processor_cls.from_pretrained.return_value = mock_processor
        mock_model_cls.from_pretrained.return_value = mock_model

        for model_name in WHISPER_MODELS:
            processor, model, model_id = load_whisper_model(model_name, "cpu")

            assert processor == mock_processor
            assert model == mock_model
            assert model_id == WHISPER_MODELS[model_name]["id"]

    def test_load_whisper_model_invalid_model(self):
        """Test loading an invalid model raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model: invalid-model"):
            load_whisper_model("invalid-model", "cpu")

    def test_load_whisper_model_invalid_model_lists_available(self):
        """Test that error message lists available models."""
        try:
            load_whisper_model("invalid-model", "cpu")
        except ValueError as e:
            error_msg = str(e)
            # Should mention available models
            for model_name in WHISPER_MODELS:
                assert model_name in error_msg

    @patch("src.transcribe_audio.WhisperProcessor")
    @patch("src.transcribe_audio.WhisperForConditionalGeneration")
    def test_load_whisper_model_processor_error(
        self, mock_model_cls, mock_processor_cls
    ):
        """Test handling of processor loading errors."""
        mock_processor_cls.from_pretrained.side_effect = Exception("Network error")

        with pytest.raises(Exception, match="Network error"):
            load_whisper_model("whisper-small", "cpu")

    @patch("src.transcribe_audio.WhisperProcessor")
    @patch("src.transcribe_audio.WhisperForConditionalGeneration")
    def test_load_whisper_model_model_error(self, mock_model_cls, mock_processor_cls):
        """Test handling of model loading errors."""
        mock_processor_cls.from_pretrained.return_value = Mock()
        mock_model_cls.from_pretrained.side_effect = Exception("Model loading failed")

        with pytest.raises(Exception, match="Model loading failed"):
            load_whisper_model("whisper-small", "cpu")

    @patch("src.transcribe_audio.WhisperProcessor")
    @patch("src.transcribe_audio.WhisperForConditionalGeneration")
    def test_load_whisper_model_device_move_error(
        self, mock_model_cls, mock_processor_cls
    ):
        """Test handling of device move errors."""
        mock_processor_cls.from_pretrained.return_value = Mock()
        mock_model = Mock()
        mock_model.to.side_effect = RuntimeError("CUDA out of memory")
        mock_model_cls.from_pretrained.return_value = mock_model

        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            load_whisper_model("whisper-small", "cuda")


class TestWhisperModelsConfiguration:
    """Test the WHISPER_MODELS configuration."""

    def test_whisper_models_structure(self):
        """Test that WHISPER_MODELS has correct structure."""
        assert isinstance(WHISPER_MODELS, dict)
        assert len(WHISPER_MODELS) > 0

        for model_name, config in WHISPER_MODELS.items():
            assert isinstance(model_name, str)
            assert isinstance(config, dict)
            assert "id" in config
            assert "description" in config
            assert isinstance(config["id"], str)
            assert isinstance(config["description"], str)
            assert config["id"].startswith("openai/whisper-")

    def test_whisper_models_contains_expected_models(self):
        """Test that WHISPER_MODELS contains expected model variants."""
        expected_models = ["whisper-tiny", "whisper-small", "whisper-medium"]

        for model in expected_models:
            assert model in WHISPER_MODELS
            assert WHISPER_MODELS[model]["id"] == f"openai/{model}"

    def test_whisper_models_descriptions_not_empty(self):
        """Test that all models have non-empty descriptions."""
        for model_name, config in WHISPER_MODELS.items():
            assert len(config["description"].strip()) > 0
            # Should contain some indication of model characteristics
            description_lower = config["description"].lower()
            assert any(
                word in description_lower
                for word in [
                    "fast",
                    "accurate",
                    "small",
                    "large",
                    "medium",
                    "tiny",
                    "best",
                    "speed",
                ]
            )

    def test_whisper_models_unique_ids(self):
        """Test that all model IDs are unique."""
        model_ids = [config["id"] for config in WHISPER_MODELS.values()]
        assert len(model_ids) == len(set(model_ids))

    def test_whisper_models_valid_huggingface_format(self):
        """Test that model IDs follow HuggingFace format."""
        for config in WHISPER_MODELS.values():
            model_id = config["id"]
            # Should be in format "org/model-name"
            assert "/" in model_id
            org, model_name = model_id.split("/", 1)
            assert org == "openai"
            assert model_name.startswith("whisper-")
