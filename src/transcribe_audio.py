# %%
import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import librosa
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Constants
MAX_NEW_TOKENS = 400  # Reduced for Whisper compatibility
LANGUAGE = "en"
AUDIO_DIR = Path(__file__).parent.parent / "audio"
OUTPUT_DIR = Path(__file__).parent.parent / "output"

# Output format configurations
OUTPUT_FORMATS = {
    "csv": {"extension": ".csv", "description": "CSV format"},
    "json": {"extension": ".json", "description": "JSON format"},
    "parquet": {"extension": ".parquet", "description": "Apache Parquet format"},
    "duckdb": {"extension": ".duckdb", "description": "DuckDB database format"},
}

# Available Whisper models
WHISPER_MODELS = {
    "whisper-tiny": {
        "id": "openai/whisper-tiny",
        "description": "Fastest model, least accurate (~39 MB)",
    },
    "whisper-small": {
        "id": "openai/whisper-small",
        "description": "Fast model, good accuracy (~244 MB)",
    },
    "whisper-medium": {
        "id": "openai/whisper-medium",
        "description": "Balanced speed/accuracy (~769 MB)",
    },
    "whisper-large-v3-turbo": {
        "id": "openai/whisper-large-v3-turbo",
        "description": "Best accuracy, slower (~1550 MB)",
    },
}


def generate_file_id(filename: str, file_size: int) -> str:
    """Generate a unique identifier hash based on filename and file size."""
    content = f"{filename}_{file_size}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def get_audio_files() -> list[Path]:
    """Get all audio files from the audio directory."""
    audio_extensions = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
    audio_files = []

    for file_path in AUDIO_DIR.iterdir():
        if file_path.suffix.lower() in audio_extensions:
            audio_files.append(file_path)

    return sorted(audio_files)


def get_file_size(file_path: Path) -> int:
    """Get file size in bytes."""
    return file_path.stat().st_size


def load_existing_results(output_format: str, output_file: Path) -> set[str]:
    """Load existing transcription results and return set of file IDs."""
    if not output_file.exists():
        return set()

    try:
        if output_format == "csv":
            df = pd.read_csv(output_file)
        elif output_format == "json":
            with open(output_file, encoding="utf-8") as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        elif output_format == "parquet":
            df = pd.read_parquet(output_file)
        elif output_format == "duckdb":
            conn = duckdb.connect(str(output_file))
            df = conn.execute("SELECT * FROM transcriptions").fetchdf()
            conn.close()
        else:
            return set()

        return set(df["file_id"].tolist()) if "file_id" in df.columns else set()

    except Exception as e:
        print(f"Warning: Could not read existing {output_format} file: {e}")
        return set()


def create_transcription_record(
    filename: str,
    file_id: str,
    file_size: int,
    transcription_time: float,
    transcription_text: str,
    model_id: str,
    started_at: datetime,
) -> dict:
    """Create a standardized transcription record."""
    return {
        "file_id": file_id,
        "filename": filename,
        "file_size_bytes": file_size,
        "transcription_time_seconds": round(transcription_time, 2),
        "transcription_text": transcription_text,
        "model_id": model_id,
        "started_at": started_at.isoformat(),
        "processed_at": datetime.now(UTC).isoformat(),
    }


def save_to_csv(records: list[dict], output_file: Path) -> None:
    """Save records to CSV format."""
    df = pd.DataFrame(records)
    if output_file.exists():
        existing_df = pd.read_csv(output_file)
        df = pd.concat([existing_df, df], ignore_index=True)
    df.to_csv(output_file, index=False)


def save_to_json(records: list[dict], output_file: Path) -> None:
    """Save records to JSON format."""
    all_records = records
    if output_file.exists():
        with open(output_file, encoding="utf-8") as f:
            existing_records = json.load(f)
        all_records = existing_records + records

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)


def save_to_parquet(records: list[dict], output_file: Path) -> None:
    """Save records to Parquet format."""
    df = pd.DataFrame(records)
    if output_file.exists():
        existing_df = pd.read_parquet(output_file)
        df = pd.concat([existing_df, df], ignore_index=True)

    # Convert to PyArrow table for better type handling
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_file)


def save_to_duckdb(records: list[dict], output_file: Path) -> None:
    """Save records to DuckDB database."""
    conn = duckdb.connect(str(output_file))

    # Create table if it doesn't exist
    conn.execute("""
        CREATE TABLE IF NOT EXISTS transcriptions (
            file_id VARCHAR PRIMARY KEY,
            filename VARCHAR,
            file_size_bytes BIGINT,
            transcription_time_seconds DOUBLE,
            transcription_text TEXT,
            model_id VARCHAR,
            started_at TIMESTAMP,
            processed_at TIMESTAMP
        )
    """)

    # Insert new records (ignore duplicates based on file_id)
    if records:
        df = pd.DataFrame(records)
        conn.register("temp_df", df)
        conn.execute("INSERT OR IGNORE INTO transcriptions SELECT * FROM temp_df")
    conn.close()


def save_results(records: list[dict], output_format: str, output_file: Path) -> None:
    """Save transcription results in the specified format."""
    if not records:
        return

    if output_format == "csv":
        save_to_csv(records, output_file)
    elif output_format == "json":
        save_to_json(records, output_file)
    elif output_format == "parquet":
        save_to_parquet(records, output_file)
    elif output_format == "duckdb":
        save_to_duckdb(records, output_file)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def get_output_filename(output_format: str) -> Path:
    """Get the output filename for the specified format."""
    extension = OUTPUT_FORMATS[output_format]["extension"]
    return OUTPUT_DIR / f"transcribed_audio{extension}"


def load_whisper_model(model_name: str, device: str):
    """Load the specified Whisper model."""
    if model_name not in WHISPER_MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. Available models: {list(WHISPER_MODELS.keys())}"
        )

    model_id = WHISPER_MODELS[model_name]["id"]
    print(f"Loading {model_name} model ({model_id})...")
    print(f"Description: {WHISPER_MODELS[model_name]['description']}")

    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    model.to(device)
    print("Model loaded successfully!")
    return processor, model, model_id


# %%
def transcribe_audio(audio_path: Path, processor, model, device, model_id: str):
    """Transcribe audio file and return decoded outputs, timing, and start timestamp."""
    start_time = time.time()
    started_at = datetime.now(UTC)

    # Load audio file
    audio, _ = librosa.load(str(audio_path), sr=16000)

    # Process audio with Whisper
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    inputs = inputs.to(device)

    # Ensure input features match model dtype
    if device == "cuda":
        inputs.input_features = inputs.input_features.to(torch.float16)

    # Generate transcription with smaller max length for Whisper
    with torch.no_grad():
        outputs = model.generate(inputs.input_features, max_new_tokens=200)

    end_time = time.time()
    elapsed_time = end_time - start_time

    decoded_outputs = processor.batch_decode(outputs, skip_special_tokens=True)
    return decoded_outputs, elapsed_time, started_at


# %%
def main():
    """Process all audio files for transcription."""
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using Whisper model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available output formats:
{chr(10).join(f"  {fmt}: {config['description']}" for fmt, config in OUTPUT_FORMATS.items())}

Available Whisper models:
{chr(10).join(f"  {model}: {config['description']}" for model, config in WHISPER_MODELS.items())}

Examples:
  python src/transcription_app.py --model whisper-tiny --format csv
  python src/transcription_app.py --model whisper-large-v3-turbo --format duckdb --all-audio
  python src/transcription_app.py --model whisper-medium --format parquet
        """,
    )

    parser.add_argument(
        "--all-audio",
        action="store_true",
        help="Re-run transcription on all audio files, including previously processed ones",
    )

    parser.add_argument(
        "--format",
        choices=list(OUTPUT_FORMATS.keys()),
        default="csv",
        help="Output format for transcription results (default: csv)",
    )

    parser.add_argument(
        "--model",
        choices=list(WHISPER_MODELS.keys()),
        default="whisper-small",
        help="Whisper model to use for transcription (default: whisper-small)",
    )

    # Handle both command line and notebook execution
    try:
        args = parser.parse_args()
    except SystemExit:
        # If we're in a notebook/interactive environment, create default args
        class Args:
            all_audio = False
            format = "csv"
            model = "whisper-small"

        args = Args()

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the specified model
    processor, model, model_id = load_whisper_model(args.model, device)

    # Get output file path
    output_file = get_output_filename(args.format)
    print(
        f"Output format: {args.format} ({OUTPUT_FORMATS[args.format]['description']})"
    )
    print(f"Output file: {output_file}")

    # Get all audio files
    audio_files = get_audio_files()
    if not audio_files:
        print(f"No audio files found in {AUDIO_DIR}")
        return

    print(f"Found {len(audio_files)} audio files in {AUDIO_DIR}")

    # Load existing results to avoid re-processing
    if not args.all_audio:
        existing_file_ids = load_existing_results(args.format, output_file)
        print(f"Found {len(existing_file_ids)} previously processed files")
    else:
        existing_file_ids = set()
        print("Re-processing all files due to --all-audio flag")

    # Process each audio file
    processed_count = 0
    skipped_count = 0
    new_records = []

    for audio_file in audio_files:
        filename = audio_file.name
        file_size = get_file_size(audio_file)
        file_id = generate_file_id(filename, file_size)

        # Skip if already processed (unless --all-audio flag is used)
        if file_id in existing_file_ids and not args.all_audio:
            print(f"Skipping {filename} (already processed, ID: {file_id})")
            skipped_count += 1
            continue

        print(f"\nProcessing {filename} (ID: {file_id})...")

        try:
            print(f"File size: {file_size} bytes ({file_size / 1024:.1f} KB)")

            # Transcribe audio
            decoded_outputs, transcription_time, started_at = transcribe_audio(
                audio_file, processor, model, device, model_id
            )

            # Combine all transcription outputs into a single text
            transcription_text = " ".join(decoded_outputs).strip()

            print(f"Transcription completed in {transcription_time:.2f} seconds")
            print(f"Started at: {started_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            preview = transcription_text[:100] + (
                "..." if len(transcription_text) > 100 else ""
            )
            print(f"Transcription preview: {preview}")

            # Create transcription record
            record = create_transcription_record(
                filename,
                file_id,
                file_size,
                transcription_time,
                transcription_text,
                model_id,
                started_at,
            )
            new_records.append(record)
            processed_count += 1

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    # Save all new records at once
    if new_records:
        save_results(new_records, args.format, output_file)

    print("\n=== Processing Complete ===")
    print(f"Processed: {processed_count} files")
    print(f"Skipped: {skipped_count} files")
    print(f"Results saved to: {output_file}")

    # Display results summary
    try:
        existing_file_ids = load_existing_results(args.format, output_file)
        total_files = len(existing_file_ids)
        print(f"\nTotal transcriptions in database: {total_files}")

        if args.format == "duckdb":
            conn = duckdb.connect(str(output_file))
            result = conn.execute(
                "SELECT AVG(transcription_time_seconds) as avg_time, SUM(file_size_bytes) as total_size FROM transcriptions"
            ).fetchone()
            conn.close()
            if result and result[0] is not None:
                print(f"Average transcription time: {result[0]:.2f} seconds")
                print(f"Total file size processed: {result[1] / 1024:.1f} KB")
        elif output_file.exists():
            if args.format == "csv":
                df = pd.read_csv(output_file)
            elif args.format == "json":
                with open(output_file) as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
            elif args.format == "parquet":
                df = pd.read_parquet(output_file)

            if not df.empty:
                print(
                    f"Average transcription time: {df['transcription_time_seconds'].mean():.2f} seconds"
                )
                print(
                    f"Total file size processed: {df['file_size_bytes'].sum() / 1024:.1f} KB"
                )

    except Exception as e:
        print(f"Warning: Could not display summary statistics: {e}")


# %%
# Run the main function
if __name__ == "__main__":
    main()

# %%
# Display results if any output files exist (only when run interactively)
if __name__ != "__main__":
    print("\n=== Available Output Files ===")
    for format_name in OUTPUT_FORMATS:
        output_file = get_output_filename(format_name)
        if output_file.exists():
            size_mb = output_file.stat().st_size / 1024 / 1024
            print(f"{format_name}: {output_file} ({size_mb:.2f} MB)")

# %%
