# %%
"""Reference-free quality checks for transcription output files.

Flags likely-bad transcriptions without needing ground-truth text: empty or
suspiciously short output, repetitive/looping decoding, transcription times
that are outliers for their model, and output that doesn't look like it's in
the expected language (e.g. Whisper drifting into French on English audio).
"""

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
from langdetect import DetectorFactory, LangDetectException, detect
from rich.console import Console
from rich.table import Table

# Make langdetect deterministic (it seeds from a global by default).
DetectorFactory.seed = 0

OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_EXTENSIONS = {
    "csv": ".csv",
    "json": ".json",
    "parquet": ".parquet",
    "duckdb": ".duckdb",
}

# Languages langdetect can recognize. Expected-language codes outside this set
# (e.g. "yo" for the Yoruba model) can't be checked and are skipped.
LANGDETECT_SUPPORTED_LANGUAGES = frozenset(
    [
        "af",
        "ar",
        "bg",
        "bn",
        "ca",
        "cs",
        "cy",
        "da",
        "de",
        "el",
        "en",
        "es",
        "et",
        "fa",
        "fi",
        "fr",
        "gu",
        "he",
        "hi",
        "hr",
        "hu",
        "id",
        "it",
        "ja",
        "kn",
        "ko",
        "lt",
        "lv",
        "mk",
        "ml",
        "mr",
        "ne",
        "nl",
        "no",
        "pa",
        "pl",
        "pt",
        "ro",
        "ru",
        "sk",
        "sl",
        "so",
        "sq",
        "sv",
        "sw",
        "ta",
        "te",
        "th",
        "tl",
        "tr",
        "uk",
        "ur",
        "vi",
        "zh-cn",
        "zh-tw",
    ]
)

MIN_CHARS_FOR_CHECK = 10
MIN_WORDS_FOR_CHECK = 3
REPETITION_NGRAM_SIZE = 3
REPETITION_MIN_REPEATS = 3
REPETITION_RATIO_THRESHOLD = 0.5
TIME_OUTLIER_Z_THRESHOLD = 3.5  # standard modified-z-score cutoff (Iglewicz & Hoaglin)
TIME_OUTLIER_MIN_GROUP_SIZE = 5


def get_output_filename(output_format: str, output_dir: Path) -> Path:
    """Get the output filename for the specified format."""
    return output_dir / f"transcribed_audio{OUTPUT_EXTENSIONS[output_format]}"


def load_output(output_format: str, output_file: Path) -> pd.DataFrame:
    """Load transcription records from an output file into a DataFrame."""
    if not output_file.exists():
        raise FileNotFoundError(f"Output file not found: {output_file}")

    if output_format == "csv":
        return pd.read_csv(output_file)
    if output_format == "json":
        return pd.read_json(output_file)
    if output_format == "parquet":
        return pd.read_parquet(output_file)
    if output_format == "duckdb":
        import duckdb

        conn = duckdb.connect(str(output_file))
        df = conn.execute("SELECT * FROM transcriptions").fetchdf()
        conn.close()
        return df

    raise ValueError(f"Unsupported output format: {output_format}")


def _clean_text(text: str) -> str:
    """Coerce a possibly-missing (NaN) transcription value to a stripped string."""
    if pd.isna(text):
        return ""
    return str(text).strip()


def check_empty_or_short(text: str) -> str | None:
    """Flag transcriptions that are empty or suspiciously short."""
    text = _clean_text(text)
    if not text:
        return "empty transcription"

    word_count = len(text.split())
    if len(text) < MIN_CHARS_FOR_CHECK or word_count < MIN_WORDS_FOR_CHECK:
        return f"very short transcription ({word_count} word(s))"

    return None


def check_repetition(text: str) -> str | None:
    """Flag transcriptions dominated by one repeated n-gram (a decoding loop)."""
    words = _clean_text(text).split()
    n = REPETITION_NGRAM_SIZE
    if len(words) < n * 2:
        return None

    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    ngram, count = Counter(ngrams).most_common(1)[0]
    ratio = (count * n) / len(words)

    if count >= REPETITION_MIN_REPEATS and ratio >= REPETITION_RATIO_THRESHOLD:
        phrase = " ".join(ngram)
        return (
            f'repetitive output ("{phrase}" x{count}, {min(ratio, 1.0):.0%} of words)'
        )

    return None


def check_language(text: str, expected_language: str) -> str | None:
    """Flag transcriptions whose detected language differs from expected."""
    if expected_language not in LANGDETECT_SUPPORTED_LANGUAGES:
        return None

    text = _clean_text(text)
    if len(text) < MIN_CHARS_FOR_CHECK:
        return None

    try:
        detected = detect(text)
    except LangDetectException:
        return None

    if detected != expected_language:
        return f"unexpected language '{detected}' (expected '{expected_language}')"

    return None


def check_time_outlier(
    transcription_time: float, group_median: float, group_mad: float
) -> str | None:
    """Flag a transcription time that's an outlier among same-model runs.

    Uses a modified z-score (median/MAD rather than mean/std) so that the
    outlier itself doesn't drag the mean/std enough to mask its own deviation.
    """
    if group_mad == 0 or pd.isna(group_mad):
        return None

    modified_z = 0.6745 * (transcription_time - group_median) / group_mad
    if abs(modified_z) >= TIME_OUTLIER_Z_THRESHOLD:
        direction = "slow" if modified_z > 0 else "fast"
        return (
            f"unusually {direction} transcription time "
            f"({transcription_time:.1f}s, z={modified_z:.1f})"
        )

    return None


def evaluate_records(df: pd.DataFrame, expected_language: str) -> pd.DataFrame:
    """Run all reference-free quality checks and return the flagged rows."""
    time_stats = {}
    if "transcription_time_seconds" in df.columns and "model_id" in df.columns:
        grouped = df.groupby("model_id")["transcription_time_seconds"]
        for model_id, group in grouped:
            if len(group) >= TIME_OUTLIER_MIN_GROUP_SIZE:
                median = group.median()
                mad = (group - median).abs().median()
                time_stats[model_id] = (median, mad)

    flagged = []
    for _, row in df.iterrows():
        text = row.get("transcription_text", "")
        issues = [
            check_empty_or_short(text),
            check_repetition(text),
            check_language(text, expected_language),
        ]

        stats = time_stats.get(row.get("model_id"))
        if stats is not None:
            issues.append(check_time_outlier(row["transcription_time_seconds"], *stats))

        issues = [issue for issue in issues if issue]
        if issues:
            flagged.append(
                {
                    "filename": row.get("filename", "unknown"),
                    "file_id": row.get("file_id", ""),
                    "model_id": row.get("model_id", ""),
                    "issues": "; ".join(issues),
                }
            )

    return pd.DataFrame(flagged)


def display_report(
    total_records: int, issues_df: pd.DataFrame, expected_language: str
) -> None:
    """Print a summary of the quality check results."""
    console = Console()
    console.print(
        f"\nEvaluated {total_records} transcription(s) "
        f"(expected language: '{expected_language}')"
    )

    if issues_df.empty:
        console.print("[green]No quality issues detected.[/green]\n")
        return

    table = Table(
        title="Quality Issues Detected", show_header=True, header_style="bold red"
    )
    table.add_column("Filename", style="cyan", no_wrap=True)
    table.add_column("Model", style="magenta")
    table.add_column("Issues", style="yellow")

    for _, row in issues_df.iterrows():
        table.add_row(row["filename"], row["model_id"], row["issues"])

    console.print(table)
    console.print(
        f"\n[yellow]{len(issues_df)} of {total_records} file(s) flagged[/yellow]\n"
    )


def main():
    """Run reference-free quality checks against a transcription output file."""
    parser = argparse.ArgumentParser(
        description="Run reference-free quality checks on transcription output "
        "(no ground-truth transcripts required)."
    )
    parser.add_argument(
        "--format",
        choices=list(OUTPUT_EXTENSIONS.keys()),
        default="csv",
        help="Output format to evaluate (default: csv)",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory containing the transcription output file (default: ./output)",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Expected language code for transcriptions (default: en). Note: "
        "the output file doesn't record per-run language, so this is applied "
        "uniformly to every row being evaluated.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Where to write the flagged-issues CSV (default: "
        "<output-path>/evaluation_report.csv)",
    )
    args = parser.parse_args()

    output_file = get_output_filename(args.format, args.output_path)
    df = load_output(args.format, output_file)

    if df.empty:
        print("No transcription records found.")
        return

    issues_df = evaluate_records(df, args.language)
    display_report(len(df), issues_df, args.language)

    if not issues_df.empty:
        report_file = args.report_path or (args.output_path / "evaluation_report.csv")
        issues_df.to_csv(report_file, index=False)
        print(f"Detailed report saved to: {report_file}")


# %%
if __name__ == "__main__":
    main()
