"""Integration tests for real speaker diarization.

These exercise the pyannote diarization pipeline end to end against a small,
public two-speaker clip from the Hugging Face Hub
(``sanchit-gandhi/concatenated_librispeech`` -- two LibriSpeech speakers
concatenated into a single ~22 s recording with one clean speaker transition).

They are excluded from the default test run (``-m 'not integration and not
slow'``). Run explicitly with::

    uv run pytest tests/integration/test_diarization_real.py -m "integration and slow"

Requirements to actually run (otherwise the tests skip, never fail):

- network access to download the dataset once (~0.7 MB), and
- access to the gated ``pyannote/speaker-diarization-community-1`` model: accept
  its terms and authenticate via ``HF_TOKEN`` / ``huggingface-cli login``, or
  point ``DIARIZATION_MODEL_PATH`` at a local download for offline use.
"""

import io
import os
import re
import tempfile
from pathlib import Path

import pytest

from src.transcribe_audio import (
    PYANNOTE_AVAILABLE,
    diarize_audio,
    load_diarization_pipeline,
    load_model,
    transcribe_with_diarization,
)

DATASET_ID = "sanchit-gandhi/concatenated_librispeech"


def _speaker_labels(text: str) -> list[str]:
    """Return the distinct ``SPEAKER_XX`` labels appearing in ``text``."""
    return sorted(set(re.findall(r"SPEAKER_\d+", text)))


@pytest.fixture(scope="module")
def two_speaker_wav():
    """Download the two-speaker clip and write it to a temp WAV file.

    Decodes with soundfile rather than the ``datasets`` Audio feature: the
    latter uses torchcodec, whose DLLs are broken on this Windows setup without
    FFmpeg (see CLAUDE.md). Skips if the dataset can't be fetched.
    """
    datasets = pytest.importorskip("datasets", reason="datasets not installed")
    soundfile = pytest.importorskip("soundfile", reason="soundfile not installed")

    try:
        ds = datasets.load_dataset(DATASET_ID, split="train", streaming=True)
        # decode=False -> raw file bytes, bypassing torchcodec.
        ds = ds.cast_column("audio", datasets.Audio(decode=False))
        example = next(iter(ds))
    except Exception as exc:  # network / hub / dataset errors -> skip, don't fail
        pytest.skip(f"Could not fetch {DATASET_ID}: {exc}")

    audio = example["audio"]
    raw = audio.get("bytes")
    if raw is None and audio.get("path"):
        raw = Path(audio["path"]).read_bytes()
    if not raw:
        pytest.skip("Dataset example carried no audio bytes")

    array, sample_rate = soundfile.read(io.BytesIO(raw))
    if array.ndim > 1:  # collapse to mono to match the transcription pipeline
        array = array.mean(axis=1)

    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_path = Path(tmp_dir) / "two_speakers.wav"
        soundfile.write(wav_path, array, sample_rate)
        duration = len(array) / sample_rate
        yield wav_path, duration


@pytest.fixture(scope="module")
def diarization_pipeline():
    """Load the pyannote pipeline, skipping if it isn't available/authorized."""
    if not PYANNOTE_AVAILABLE:
        pytest.skip("pyannote.audio not installed")

    model_path = os.environ.get("DIARIZATION_MODEL_PATH") or None
    hf_token = os.environ.get("HF_TOKEN") or None
    try:
        return load_diarization_pipeline(
            "cpu", hf_token=hf_token, model_path=model_path
        )
    except Exception as exc:  # gated model / no token / offline -> skip
        pytest.skip(f"Diarization model unavailable: {exc}")


@pytest.mark.integration
@pytest.mark.slow
class TestRealDiarization:
    """Real pyannote diarization on the concatenated-LibriSpeech clip."""

    def test_finds_two_speakers_when_count_is_forced(
        self, two_speaker_wav, diarization_pipeline
    ):
        """Forcing num_speakers=2 recovers exactly two speakers with a turn."""
        wav_path, duration = two_speaker_wav

        turns = diarize_audio(wav_path, diarization_pipeline, num_speakers=2)

        assert turns, "diarization returned no speaker turns"
        # Turns are (start, end, speaker); each must be a valid, ordered span.
        for start, end, speaker in turns:
            assert 0.0 <= start < end <= duration + 0.5
            assert speaker.startswith("SPEAKER_")

        speakers = {speaker for _, _, speaker in turns}
        assert len(speakers) == 2, f"expected 2 speakers, got {sorted(speakers)}"

        # The clip is speaker A then speaker B, so the first and last turns
        # must belong to different speakers -- i.e. a real transition was found.
        assert turns[0][2] != turns[-1][2]

    def test_auto_detect_finds_multiple_speakers(
        self, two_speaker_wav, diarization_pipeline
    ):
        """Without a speaker count, auto-detection still finds >= 2 speakers."""
        wav_path, _ = two_speaker_wav

        turns = diarize_audio(wav_path, diarization_pipeline)

        speakers = {speaker for _, _, speaker in turns}
        assert len(speakers) >= 2, f"expected >=2 speakers, got {sorted(speakers)}"

    def test_end_to_end_diarized_transcript_with_whisper(
        self, two_speaker_wav, diarization_pipeline
    ):
        """Full pipeline: diarize + Whisper produces a labelled transcript."""
        wav_path, _ = two_speaker_wav

        processor, model, model_id, model_type = load_model("whisper-tiny", "cpu")

        outputs, elapsed, started_at, refined = transcribe_with_diarization(
            wav_path,
            diarization_pipeline,
            processor,
            model,
            "cpu",
            model_id,
            model_type,
            "en",
            400,
            num_speakers=2,
        )

        assert outputs and isinstance(outputs[0], str)
        transcript = outputs[0]
        assert len(_speaker_labels(transcript)) == 2
        # Some transcribed words survive alongside the labels.
        words = re.sub(r"SPEAKER_\d+:", "", transcript).strip()
        assert len(words) > 0
        assert elapsed > 0
        assert started_at is not None
        assert refined is False
