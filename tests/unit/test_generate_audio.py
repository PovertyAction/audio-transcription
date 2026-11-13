"""Tests for parsing transcripts and assigning voices in generate_audio."""

from pathlib import Path

import numpy as np
import pytest

from src.generate_audio import (
    assign_voices,
    describe_speakers,
    duplicate_stems,
    format_voices,
    parse_transcript,
    to_mp3,
)


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


def test_format_voices_filters_language_and_sorts_by_country():
    """Only the chosen language is listed, grouped by country then gender."""
    catalog = [
        {"name": "Lalo", "language": "es", "locale": "es-mx", "gender": "male"},
        {"name": "Lucía", "language": "es", "locale": "es-es", "gender": "female"},
        {"name": "Zoey", "language": "en", "locale": "en-us", "gender": "female"},
        {"name": "Bot", "language": "es", "locale": "characters"},
    ]
    lines = format_voices(catalog, "es").splitlines()
    assert lines[0].split() == ["NAME", "COUNTRY", "GENDER", "AGE"]
    assert [line.split()[0] for line in lines[1:]] == ["Bot", "Lucía", "Lalo"]
    assert lines[1].split()[2] == "-"


@pytest.mark.parametrize(
    "line",
    [
        "ENQUÊTRICE : Bonjour.",  # French space before colon
        "ENQUÊTRICE : Bonjour.",  # no-break space
        "ENQUÊTRICE : Bonjour.",  # narrow no-break space
        "L’ENQUÊTRICE: Bonjour.",  # typographic apostrophe
        "采访者：你好。",  # full-width colon, no space
    ],
)
def test_labels_in_other_languages(line):
    """Labels are recognized across scripts and colon conventions."""
    ((speaker, text),) = parse_transcript(line)
    assert speaker != "SPEAKER" and ":" not in speaker and "：" not in speaker
    assert text in {"Bonjour.", "你好。"}


def test_times_and_urls_are_not_labels():
    """Things like 10:30 or https:// do not start a turn."""
    turns = parse_transcript("Hola.\nA las 10:30 llegamos.\nVer https://ipa.org")
    assert len(turns) == 1


def test_single_speaker_keeps_labels_as_text():
    """--single-speaker reads everything, labels included, in one voice."""
    assert parse_transcript("Nota: traer recibos.\nGracias.", True) == [
        ("SPEAKER", "Nota: traer recibos. Gracias.")
    ]


def test_describe_one_speaker():
    """One speaker needs no diarization."""
    assert "no diarization" in describe_speakers([("SPEAKER", "Hola.")])


def test_describe_interview_suggests_diarize():
    """Several alternating speakers suggest --diarize with the count, no warnings."""
    turns = parse_transcript("A: Hola.\nB: Hola.\nA: ¿Cómo está?\nB: Bien.")
    summary = describe_speakers(turns)
    assert "--diarize --num-speakers 2" in summary and "Check" not in summary


def test_describe_flags_accidental_label_in_monologue():
    """A 'Nota:' line inside an unlabeled monologue is flagged (any language)."""
    turns = parse_transcript("Hoy hablamos del ahorro.\nNota: traer recibos.\nFin.")
    assert "--single-speaker" in describe_speakers(turns)


def test_describe_flags_label_used_once():
    """A label used once among recurring speakers is flagged."""
    turns = parse_transcript("A: Hola.\nB: Hola.\nRemarque: pause.\nA: Bien.")
    assert "Remarque speak(s) only once" in describe_speakers(turns)


def test_duplicate_stems_detects_clashing_outputs():
    """Different files with the same stem clash; the same file twice does not."""
    files = [Path("a/interview.txt"), Path("b/Interview.md"), Path("c/other.txt")]
    assert duplicate_stems(files) == [str(files[0]) + ", " + str(files[1])]
    assert duplicate_stems([Path("x.txt"), Path("x.txt")]) == []


def test_headings_only_transcript_has_no_turns():
    """Empty or heading-only files give no turns, so they are skipped."""
    assert parse_transcript("# Title\n\n# Notes\n") == []
