"""Unit tests for diarization alignment and LLM refinement (all mocked)."""

from datetime import UTC, datetime
from unittest.mock import Mock, patch

import numpy as np
import pytest
import requests as requests_lib

from src.transcribe_audio import (
    MAX_REFINE_CHARS,
    OllamaError,
    _ollama_num_ctx,
    assign_speakers_to_segments,
    format_speaker_transcript,
    refine_transcript_with_llm,
    transcribe_whisper_with_timestamps,
    transcribe_with_diarization,
)


class TestSpeakerAssignment:
    """Tests for assign_speakers_to_segments and format_speaker_transcript."""

    def test_assign_speakers_by_max_overlap(self):
        """Each segment gets the speaker with maximal temporal overlap."""
        segments = [
            {"start": 0.0, "end": 4.0, "text": "hello there"},
            {"start": 4.0, "end": 8.0, "text": "hi back"},
            {"start": 8.0, "end": 12.0, "text": "how are you"},
        ]
        turns = [(0.0, 4.5, "SPEAKER_00"), (4.5, 12.0, "SPEAKER_01")]

        result = assign_speakers_to_segments(segments, turns)

        assert result == [
            ("SPEAKER_00", "hello there"),
            ("SPEAKER_01", "hi back how are you"),
        ]

    def test_assign_speakers_merges_consecutive_same_speaker(self):
        """Consecutive same-speaker segments are concatenated with spaces."""
        segments = [
            {"start": 0.0, "end": 2.0, "text": "one"},
            {"start": 2.0, "end": 4.0, "text": "two"},
            {"start": 4.0, "end": 6.0, "text": "three"},
        ]
        turns = [(0.0, 6.0, "SPEAKER_00")]

        result = assign_speakers_to_segments(segments, turns)

        assert result == [("SPEAKER_00", "one two three")]

    def test_assign_speakers_gap_segment_uses_nearest_turn(self):
        """A segment in a diarization gap goes to the nearest turn."""
        segments = [{"start": 10.0, "end": 11.0, "text": "orphan"}]
        turns = [(0.0, 2.0, "SPEAKER_00"), (11.5, 20.0, "SPEAKER_01")]

        result = assign_speakers_to_segments(segments, turns)

        assert result == [("SPEAKER_01", "orphan")]

    def test_assign_speakers_empty_turns(self):
        """With no turns at all, segments fall back to a default speaker."""
        segments = [{"start": 0.0, "end": 1.0, "text": "solo"}]

        result = assign_speakers_to_segments(segments, [])

        assert result == [("SPEAKER_00", "solo")]

    def test_format_speaker_transcript(self):
        """Formatter produces the single-line '; '-separated format."""
        turns = [("SPEAKER_00", "hi "), ("SPEAKER_01", "yo")]

        assert format_speaker_transcript(turns) == "SPEAKER_00: hi; SPEAKER_01: yo"


class TestWhisperTimestamps:
    """Tests for transcribe_whisper_with_timestamps (pipeline mocked)."""

    @patch("src.transcribe_audio.pipeline")
    @patch("src.transcribe_audio.librosa")
    def test_normalizes_chunks(self, mock_librosa, mock_pipeline):
        """None end -> audio duration; empty-text chunks dropped."""
        mock_librosa.load.return_value = (np.zeros(160000), 16000)  # 10s
        asr = Mock()
        asr.return_value = {
            "chunks": [
                {"timestamp": (0.0, 4.0), "text": " first "},
                {"timestamp": (4.0, 6.0), "text": "   "},
                {"timestamp": (6.0, None), "text": "last"},
            ]
        }
        mock_pipeline.return_value = asr

        segments = transcribe_whisper_with_timestamps(
            "fake.mp3", Mock(), Mock(), "cpu", "en"
        )

        assert segments == [
            {"start": 0.0, "end": 4.0, "text": "first"},
            {"start": 6.0, "end": 10.0, "text": "last"},
        ]
        assert mock_pipeline.call_args.kwargs["return_timestamps"] is True
        assert asr.call_args.kwargs["generate_kwargs"] == {
            "language": "en",
            "task": "transcribe",
        }

    @patch("src.transcribe_audio.pipeline")
    @patch("src.transcribe_audio.librosa")
    def test_auto_language_omits_generate_kwargs(self, mock_librosa, mock_pipeline):
        """'auto' language must not force a language on the model."""
        mock_librosa.load.return_value = (np.zeros(16000), 16000)
        asr = Mock()
        asr.return_value = {"chunks": []}
        mock_pipeline.return_value = asr

        segments = transcribe_whisper_with_timestamps(
            "fake.mp3", Mock(), Mock(), "cpu", "auto"
        )

        assert segments == []
        assert asr.call_args.kwargs["generate_kwargs"] is None


class TestTranscribeWithDiarization:
    """Routing tests for transcribe_with_diarization (everything mocked)."""

    @patch("src.transcribe_audio.transcribe_audio")
    @patch("src.transcribe_audio.transcribe_whisper_with_timestamps")
    @patch("src.transcribe_audio.diarize_audio")
    def test_whisper_uses_alignment(self, mock_diarize, mock_timestamps, mock_plain):
        """Whisper models align segments; the per-turn slicer is not used."""
        mock_diarize.return_value = [(0.0, 5.0, "SPEAKER_00")]
        mock_timestamps.return_value = [{"start": 0.0, "end": 5.0, "text": "hello"}]

        outputs, elapsed, started_at, refined = transcribe_with_diarization(
            "fake.mp3", Mock(), Mock(), Mock(), "cpu", "id", "whisper", "en", 400
        )

        assert outputs == ["SPEAKER_00: hello"]
        assert isinstance(started_at, datetime)
        assert started_at.tzinfo == UTC
        assert refined is False
        mock_plain.assert_not_called()

    @patch("soundfile.write")
    @patch("src.transcribe_audio.librosa")
    @patch("src.transcribe_audio.transcribe_audio")
    @patch("src.transcribe_audio.transcribe_whisper_with_timestamps")
    @patch("src.transcribe_audio.diarize_audio")
    def test_whisper_falls_back_when_no_segments(
        self, mock_diarize, mock_timestamps, mock_plain, mock_librosa, mock_sf_write
    ):
        """No usable timestamps -> the per-turn slicing path runs."""
        mock_diarize.return_value = [(0.0, 1.0, "SPEAKER_00")]
        mock_timestamps.return_value = []
        mock_librosa.load.return_value = (np.zeros(16000), 16000)
        mock_plain.return_value = (["sliced text"], 0.1, datetime.now(UTC))

        outputs, _, _, refined = transcribe_with_diarization(
            "fake.mp3", Mock(), Mock(), Mock(), "cpu", "id", "whisper", "en", 400
        )

        assert outputs == ["SPEAKER_00: sliced text"]
        assert refined is False
        mock_plain.assert_called_once()

    @patch("soundfile.write")
    @patch("src.transcribe_audio.librosa")
    @patch("src.transcribe_audio.transcribe_audio")
    @patch("src.transcribe_audio.diarize_audio")
    def test_cohere_still_slices(
        self, mock_diarize, mock_plain, mock_librosa, mock_sf_write
    ):
        """Cohere models keep the per-turn slicing path."""
        mock_diarize.return_value = [
            (0.0, 1.0, "SPEAKER_00"),
            (1.0, 2.0, "SPEAKER_01"),
        ]
        mock_librosa.load.return_value = (np.zeros(32000), 16000)
        mock_plain.side_effect = [
            (["first"], 0.1, datetime.now(UTC)),
            (["second"], 0.1, datetime.now(UTC)),
        ]

        outputs, _, _, refined = transcribe_with_diarization(
            "fake.mp3", Mock(), Mock(), Mock(), "cpu", "id", "cohere", "en", 400
        )

        assert outputs == ["SPEAKER_00: first; SPEAKER_01: second"]
        assert mock_plain.call_count == 2

    @patch("src.transcribe_audio.refine_transcript_with_llm")
    @patch("src.transcribe_audio.transcribe_audio")
    @patch("src.transcribe_audio.transcribe_whisper_with_timestamps")
    @patch("src.transcribe_audio.diarize_audio")
    def test_refine_fallback_on_ollama_error(
        self, mock_diarize, mock_timestamps, mock_plain, mock_refine
    ):
        """Refinement failure keeps the unrefined transcript, refined=False."""
        mock_diarize.return_value = [(0.0, 5.0, "SPEAKER_00")]
        mock_timestamps.return_value = [{"start": 0.0, "end": 5.0, "text": "hello"}]
        mock_refine.side_effect = OllamaError("server down")

        outputs, _, _, refined = transcribe_with_diarization(
            "fake.mp3",
            Mock(),
            Mock(),
            Mock(),
            "cpu",
            "id",
            "whisper",
            "en",
            400,
            refine=True,
        )

        assert outputs == ["SPEAKER_00: hello"]
        assert refined is False

    @patch("src.transcribe_audio.refine_transcript_with_llm")
    @patch("src.transcribe_audio.transcribe_audio")
    @patch("src.transcribe_audio.transcribe_whisper_with_timestamps")
    @patch("src.transcribe_audio.diarize_audio")
    def test_whisper_refine_needs_no_second_asr(
        self, mock_diarize, mock_timestamps, mock_plain, mock_refine
    ):
        """On the alignment path the plain transcript is free: no extra ASR."""
        mock_diarize.return_value = [(0.0, 5.0, "SPEAKER_00")]
        mock_timestamps.return_value = [{"start": 0.0, "end": 5.0, "text": "hello"}]
        mock_refine.return_value = "SPEAKER_00: hello refined"

        outputs, _, _, refined = transcribe_with_diarization(
            "fake.mp3",
            Mock(),
            Mock(),
            Mock(),
            "cpu",
            "id",
            "whisper",
            "en",
            400,
            refine=True,
        )

        assert outputs == ["SPEAKER_00: hello refined"]
        assert refined is True
        mock_plain.assert_not_called()
        assert mock_refine.call_args.args[0] == "hello"  # plain text from segments


class TestOllamaRefinement:
    """Tests for refine_transcript_with_llm (requests mocked)."""

    def _response(self, content, status=200):
        response = Mock()
        response.status_code = status
        response.json.return_value = {"message": {"content": content}}
        response.raise_for_status.return_value = None
        return response

    @patch("src.transcribe_audio.requests")
    def test_refine_success(self, mock_requests):
        """Valid response: POST to /api/chat with stream off and num_ctx set."""
        mock_requests.post.return_value = self._response(
            "SPEAKER_00: accurate text; SPEAKER_01: reply"
        )
        mock_requests.exceptions = requests_lib.exceptions

        result = refine_transcript_with_llm("accurate text reply", "SPEAKER_00: rough")

        assert result == "SPEAKER_00: accurate text; SPEAKER_01: reply"
        url = mock_requests.post.call_args.args[0]
        body = mock_requests.post.call_args.kwargs["json"]
        assert url.endswith("/api/chat")
        assert body["stream"] is False
        assert body["think"] is False  # thinking burns the budget on long inputs
        assert body["options"]["num_ctx"] >= 8192

    @patch("src.transcribe_audio.requests")
    def test_refine_strips_fences_and_preamble(self, mock_requests):
        """Code fences and preamble text are removed; newlines re-joined."""
        mock_requests.post.return_value = self._response(
            "Here is the transcript:\n```\nSPEAKER_00: hi\nSPEAKER_01: yo\n```"
        )
        mock_requests.exceptions = requests_lib.exceptions

        result = refine_transcript_with_llm("hi yo", "SPEAKER_00: h; SPEAKER_01: y")

        assert result == "SPEAKER_00: hi; SPEAKER_01: yo"

    @patch("src.transcribe_audio.requests")
    def test_refine_rejects_missing_speaker_labels(self, mock_requests):
        """A response without SPEAKER_ labels raises OllamaError."""
        mock_requests.post.return_value = self._response("I cannot help with that.")
        mock_requests.exceptions = requests_lib.exceptions

        with pytest.raises(OllamaError, match="speaker labels"):
            refine_transcript_with_llm("text", "SPEAKER_00: rough")

    @patch("src.transcribe_audio.requests")
    def test_refine_connection_error_message(self, mock_requests):
        """Connection failures produce an actionable install/pull message."""
        mock_requests.exceptions = requests_lib.exceptions
        mock_requests.post.side_effect = requests_lib.exceptions.ConnectionError()

        with pytest.raises(OllamaError, match="ollama.com"):
            refine_transcript_with_llm("text", "SPEAKER_00: rough")

    @patch("src.transcribe_audio.requests")
    def test_refine_missing_model_message(self, mock_requests):
        """HTTP 404 tells the user to pull the model."""
        mock_requests.exceptions = requests_lib.exceptions
        mock_requests.post.return_value = self._response("", status=404)

        with pytest.raises(OllamaError, match="ollama pull"):
            refine_transcript_with_llm("text", "SPEAKER_00: rough")

    @patch("src.transcribe_audio.requests")
    def test_refine_chunks_long_transcripts(self, mock_requests):
        """Inputs above MAX_REFINE_CHARS are refined in multiple POSTs."""
        mock_requests.exceptions = requests_lib.exceptions
        mock_requests.post.return_value = self._response("SPEAKER_00: chunk")
        turn = "SPEAKER_00: " + "word " * 800
        diarized = "; ".join([turn] * 20)  # far above MAX_REFINE_CHARS
        plain = "word " * 16000
        assert len(plain) + len(diarized) > MAX_REFINE_CHARS

        result = refine_transcript_with_llm(plain, diarized)

        assert mock_requests.post.call_count > 1
        assert result == "; ".join(
            ["SPEAKER_00: chunk"] * mock_requests.post.call_count
        )

    def test_ollama_num_ctx_tiers(self):
        """Character counts map to escalating context tiers."""
        assert _ollama_num_ctx(1_000) == 8192
        assert _ollama_num_ctx(15_000) == 16384
        assert _ollama_num_ctx(40_000) == 32768
        assert _ollama_num_ctx(500_000) == 32768
