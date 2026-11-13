"""Generate speech audio from a written transcript, for demos and tests.

Each line that starts with a speaker label (`Interviewer: ...`,
`ENTREVISTADO: ...`, `SPEAKER_01: ...`) starts a new turn; lines without a
label continue the previous turn. A transcript with no labels is read by one
voice, and --single-speaker reads everything (labels included) in one voice.
Lines starting with `#` are skipped. Labels work in any language and script,
with a space before the colon (French typography) or a full-width colon.

After each MP3 the script says how many speakers it heard, which tells you
whether to transcribe the MP3 with --diarize, and warns about labels that look
accidental (for example `Note: ...` inside a monologue).

Speech comes from Phonon (gradium/phonon-40m-multilingual), which runs on a
laptop CPU. Each speaker gets a different voice, in order of first appearance,
alternating female and male. Defaults are US English, Mexican Spanish, and
French from France; run --list-voices to see every voice's country and gender,
then pick with --voices.

Before the first run, log in to Hugging Face (`hf auth login`) and accept the
terms at https://huggingface.co/gradium/phonon-40m-multilingual. The model and
voices are licensed CC BY-NC 4.0: non-commercial use only. Use it for
fictitious scripts, never to imitate a real person.

Example:
    uv run python src/generate_audio.py --list-voices --language es
    uv run python src/generate_audio.py script.md --language es
    uv run python src/generate_audio.py a.txt b.txt --output-path tmp/audio \
        --voices "Interviewer=Easton" "Respondent=Zoey"

"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import lameenc
import numpy as np

MODEL = "hf://gradium/phonon-40m-multilingual"
VOICES_REPO = "gradium/phonon-voices"
DEFAULT_OUTPUT = Path(__file__).parent.parent / "audio" / "demo"

# Alternating female and male voices from the Phonon catalog.
DEFAULT_VOICES = {
    "en": ["Harper", "Easton", "Zoey", "Micah", "Hadley", "Trevor"],
    "es": ["Regina", "Lalo", "Jimena", "Aldo", "Rebeca", "Ulises"],  # es-MX
    "fr": ["Noémie", "Jules", "Coralie", "Arnaud", "Solène", "Lucien"],
}

# A label starts with a letter (any script), then up to 40 letters, digits,
# spaces (including the no-break spaces French puts before ":"), dots,
# apostrophes, or hyphens, then ": " or a full-width "：".
TURN = re.compile(r"^\s*([^\W\d_][\w\s.'’-]{0,39}?)\s*(?::\s+|：\s*)(\S.*)$")
UNLABELED = "SPEAKER"  # speaker name for text before any label


def parse_transcript(text: str, single_speaker: bool = False) -> list[tuple[str, str]]:
    """Split a transcript into (speaker, text) turns.

    With single_speaker, labels are read as plain text and everything is one turn.
    """
    turns: list[list[str]] = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = None if single_speaker else TURN.match(line)
        if match:
            turns.append([match[1].strip(), match[2]])
        elif turns:
            turns[-1][1] += " " + line.strip()
        else:
            turns.append([UNLABELED, line.strip()])
    return [(speaker, " ".join(words.split())) for speaker, words in turns]


def describe_speakers(turns: list[tuple[str, str]]) -> str:
    """Say how many speakers a transcript has and whether to diarize it.

    Likely accidental labels are flagged from the turn structure alone (no word
    lists), so the checks work in any language.
    """
    counts = Counter(speaker for speaker, _ in turns)
    if len(counts) <= 1:
        return "1 speaker: no diarization needed when transcribing."
    lines = [
        f"{len(counts)} speakers ({', '.join(counts)}): transcribe with "
        f"--diarize --num-speakers {len(counts)}."
    ]
    if turns[0][0] == UNLABELED:
        lines.append(
            "Check: the text starts without a speaker label but has labels "
            "later. If it is one person, use --single-speaker."
        )
    once = [s for s, n in counts.items() if n == 1 and s != UNLABELED]
    if once and len(turns) > len(counts):
        lines.append(
            f"Check: {', '.join(once)} speak(s) only once. If a label is not a "
            "person (like 'Note: ...'), reword that line or use --single-speaker."
        )
    return "\n".join(lines)


def duplicate_stems(transcripts: list[Path]) -> list[str]:
    """Find inputs that would write the same MP3 (e.g. a/x.txt and b/X.md)."""
    by_stem: dict[str, list[str]] = {}
    for path in dict.fromkeys(transcripts):  # the same file twice is fine
        by_stem.setdefault(path.stem.lower(), []).append(str(path))
    return [", ".join(paths) for paths in by_stem.values() if len(paths) > 1]


def assign_voices(
    speakers: list[str], language: str, overrides: dict[str, str] | None = None
) -> dict[str, str]:
    """Give each speaker a voice: overrides first, then the language defaults."""
    voices = dict(overrides or {})
    free = [v for v in DEFAULT_VOICES[language] if v not in voices.values()]
    for speaker in dict.fromkeys(speakers):
        if speaker not in voices:
            if not free:
                raise ValueError(
                    f"More speakers than default voices for {language!r}; "
                    "name the rest with --voices."
                )
            voices[speaker] = free.pop(0)
    return voices


def format_voices(catalog: list[dict], language: str) -> str:
    """Table of a language's voices (name, country, gender, age), by country."""
    rows = sorted(
        (v for v in catalog if v["language"] == language),
        key=lambda v: (v["locale"], v.get("gender") or "", v["name"]),
    )
    lines = [f"{'NAME':<12}{'COUNTRY':<12}{'GENDER':<8}AGE"]
    for v in rows:
        gender = v.get("gender") or "-"
        lines.append(f"{v['name']:<12}{v['locale']:<12}{gender:<8}{v.get('age', '')}")
    return "\n".join(lines)


def list_voices(language: str) -> None:
    """Download the Phonon voice catalog and print the voices for a language."""
    import json

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(VOICES_REPO, "voices.jsonl")
    with open(path, encoding="utf-8") as f:
        catalog = [json.loads(line) for line in f if line.strip()]
    print(format_voices(catalog, language))


def to_mp3(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encode a mono float waveform in [-1, 1] as MP3."""
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(64)
    encoder.set_in_sample_rate(sample_rate)
    encoder.set_channels(1)
    encoder.set_quality(2)
    return encoder.encode(pcm.tobytes()) + encoder.flush()


def main() -> None:
    """Read each transcript and write one MP3 per transcript."""
    parser = argparse.ArgumentParser(
        description="Generate speech audio from written transcripts."
    )
    parser.add_argument("transcripts", nargs="*", type=Path, help="Text files")
    parser.add_argument(
        "--language", choices=sorted(DEFAULT_VOICES), default="en", help="Language"
    )
    parser.add_argument(
        "--voices",
        nargs="+",
        default=[],
        metavar="SPEAKER=VOICE",
        help="Voice for a speaker, by Phonon voice name",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Folder for the MP3 files (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="Print the voices for --language (country, gender, age) and exit",
    )
    parser.add_argument(
        "--single-speaker",
        action="store_true",
        help="Read the whole text in one voice, treating 'Label:' as plain text",
    )
    parser.add_argument(
        "--pause", type=float, default=0.6, help="Seconds of silence between turns"
    )
    args = parser.parse_args()
    if args.list_voices:
        list_voices(args.language)
        return
    if not args.transcripts:
        parser.error("give at least one transcript file (or use --list-voices)")
    if clashes := duplicate_stems(args.transcripts):
        parser.error(
            "these transcripts would write the same MP3, rename one or run them "
            "separately: " + "; ".join(clashes)
        )
    overrides = dict(v.split("=", 1) for v in args.voices)

    import ptts  # loaded here so --help and tests need no model

    tts = ptts.TTS(MODEL, lang=args.language, quant="q8")
    pause = np.zeros(int(args.pause * tts.sample_rate), dtype=np.float32)
    args.output_path.mkdir(parents=True, exist_ok=True)
    for transcript in args.transcripts:
        turns = parse_transcript(transcript.read_text("utf-8"), args.single_speaker)
        if not turns:
            print(f"Skipped {transcript}: no text to speak.\n")
            continue
        voices = assign_voices([s for s, _ in turns], args.language, overrides)
        parts = []
        for speaker, line in turns:
            print(f"{voices[speaker]:>8}: {line}")
            parts += [tts.synth(line, voice=voices[speaker]), pause]
        audio = np.concatenate(parts)
        out = args.output_path / f"{transcript.stem}.mp3"
        out.write_bytes(to_mp3(audio, tts.sample_rate))
        print(f"Wrote {out} ({len(audio) / tts.sample_rate:.0f} seconds)")
        print(describe_speakers(turns) + "\n")


if __name__ == "__main__":
    main()
