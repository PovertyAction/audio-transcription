"""Tests for parsing transcripts and assigning voices in generate_audio."""

import numpy as np
import pytest

from src.generate_audio import assign_voices, parse_transcript, to_mp3


def test_parse_speakers_and_continuation_lines():
    """Labels start turns, other lines continue them, headings are skipped."""
    text = (
        "# Fictitious interview\n\n"
        "ENTREVISTADORA: Buenos días.\n¿Desde cuándo participa?\n\n"
        "ENTREVISTADO: Desde hace un año.\n"
        "Dr. Ruiz: Gracias."
    )
    assert parse_transcript(text) == [
        ("ENTREVISTADORA", "Buenos días. ¿Desde cuándo participa?"),
        ("ENTREVISTADO", "Desde hace un año."),
        ("Dr. Ruiz", "Gracias."),
    ]


def test_parse_text_without_labels_is_one_speaker():
    """Text with no labels becomes one speaker."""
    assert parse_transcript("Hello there.\nSecond line.") == [
        ("SPEAKER", "Hello there. Second line.")
    ]


def test_assign_voices_in_order_with_overrides():
    """Overrides win; other speakers get unused defaults in order."""
    voices = assign_voices(["A", "B", "A", "C"], "es", {"B": "Regina"})
    assert voices == {"B": "Regina", "A": "Lalo", "C": "Jimena"}


def test_assign_voices_too_many_speakers():
    """Running out of default voices asks for --voices."""
    with pytest.raises(ValueError, match="--voices"):
        assign_voices([f"S{i}" for i in range(7)], "fr")


def test_to_mp3_returns_mp3_bytes():
    """The encoder returns an MP3 stream."""
    tone = 0.3 * np.sin(np.linspace(0, 2000, 24000)).astype(np.float32)
    mp3 = to_mp3(tone, 24000)
    assert len(mp3) > 1000 and (mp3[:3] == b"ID3" or mp3[0] == 0xFF)
