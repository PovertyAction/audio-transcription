# %%
import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import librosa
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

# Optional Voxtral support - only import if available
try:
    from transformers import AutoProcessor, VoxtralForConditionalGeneration

    VOXTRAL_AVAILABLE = True
except ImportError:
    VOXTRAL_AVAILABLE = False
    print(
        "Warning: Voxtral models not available. Install latest with: uv pip install git+https://github.com/huggingface/transformers"
    )

# Constants
AUDIO_DIR = Path(__file__).parent.parent / "audio"
OUTPUT_DIR = Path(__file__).parent.parent / "output"

# Output format configurations
OUTPUT_FORMATS = {
    "csv": {"extension": ".csv", "description": "CSV format"},
    "json": {"extension": ".json", "description": "JSON format"},
    "parquet": {"extension": ".parquet", "description": "Apache Parquet format"},
    "duckdb": {"extension": ".duckdb", "description": "DuckDB database format"},
}

# Available models (Whisper and optionally Voxtral)
AVAILABLE_MODELS = {
    # Whisper models
    "whisper-tiny": {
        "id": "openai/whisper-tiny",
        "type": "whisper",
        "description": "Fastest Whisper model, least accurate (~39 MB)",
    },
    "whisper-small": {
        "id": "openai/whisper-small",
        "type": "whisper",
        "description": "Fast Whisper model, good accuracy (~244 MB)",
    },
    "whisper-medium": {
        "id": "openai/whisper-medium",
        "type": "whisper",
        "description": "Balanced Whisper speed/accuracy (~769 MB)",
    },
    "whisper-large-v3-turbo": {
        "id": "openai/whisper-large-v3-turbo",
        "type": "whisper",
        "description": "Best Whisper accuracy, slower (~1550 MB)",
    },
    "whisper-large-v3": {
        "id": "openai/whisper-large-v3",
        "type": "whisper",
        "description": "Best Whisper accuracy, much slower",
    },
}

# Add Voxtral models if available
if VOXTRAL_AVAILABLE:
    AVAILABLE_MODELS.update(
        {
            "voxtral-mini": {
                "id": "mistralai/Voxtral-Mini-3B-2507",
                "type": "voxtral",
                "description": "Voxtral Mini model for multilingual ASR (~3B params)",
            },
            "voxtral-small": {
                "id": "mistralai/Voxtral-Small-24B-2507",
                "type": "voxtral",
                "description": "Voxtral Small model for high-quality multilingual ASR (~24B params)",
            },
        }
    )


def generate_file_id(filename: str, file_size: int) -> str:
    """Generate a unique identifier hash based on filename and file size."""
    content = f"{filename}_{file_size}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def get_audio_files(input_dir: Path = None) -> list[Path]:
    """Get all audio files from the specified directory."""
    if input_dir is None:
        input_dir = AUDIO_DIR

    audio_extensions = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
    audio_files = []

    if not input_dir.exists():
        return audio_files

    for file_path in input_dir.iterdir():
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


def get_output_filename(output_format: str, output_dir: Path = None) -> Path:
    """Get the output filename for the specified format."""
    if output_dir is None:
        output_dir = OUTPUT_DIR

    extension = OUTPUT_FORMATS[output_format]["extension"]
    return output_dir / f"transcribed_audio{extension}"


def load_model(model_name: str, device: str):
    """Load the specified model (Whisper or Voxtral)."""
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. Available models: {list(AVAILABLE_MODELS.keys())}"
        )

    model_config = AVAILABLE_MODELS[model_name]
    model_id = model_config["id"]
    model_type = model_config["type"]

    print(f"Loading {model_name} model ({model_id})...")
    print(f"Description: {model_config['description']}")
    print(f"Model type: {model_type.upper()}")

    if model_type == "whisper":
        processor = WhisperProcessor.from_pretrained(model_id)
        model = WhisperForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        model.to(device)
    elif model_type == "voxtral":
        if not VOXTRAL_AVAILABLE:
            raise ValueError(
                "Voxtral models are not available. Install with: uv pip install git+https://github.com/huggingface/transformers"
            )
        processor = AutoProcessor.from_pretrained(model_id)
        model = VoxtralForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map=device if device == "cuda" else None,
        )
        if device == "cpu":
            model.to(device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    print("Model loaded successfully!")
    return processor, model, model_id, model_type


def setup_logging(output_dir: Path) -> logging.Logger:
    """Set up logging to file in the output directory."""
    log_file = output_dir / "transcription.log"

    # Create logger
    logger = logging.getLogger("transcription")
    logger.setLevel(logging.INFO)

    # Clear any existing handlers
    logger.handlers.clear()

    # Create file handler
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(file_handler)

    return logger


def log_run_summary(
    logger: logging.Logger,
    command_args: str,
    start_time: datetime,
    end_time: datetime,
    total_files: int,
    processed_files: int,
    skipped_files: int,
    total_transcription_time: float,
) -> None:
    """Log a summary of the transcription run."""
    total_runtime = (end_time - start_time).total_seconds()

    logger.info("=== TRANSCRIPTION RUN SUMMARY ===")
    logger.info(f"Command: {command_args}")
    logger.info(f"Start time: {start_time.isoformat()}")
    logger.info(f"End time: {end_time.isoformat()}")
    logger.info(f"Total runtime: {total_runtime:.2f} seconds")
    logger.info(f"Total audio files found: {total_files}")
    logger.info(f"Files processed: {processed_files}")
    logger.info(f"Files skipped (already transcribed): {skipped_files}")
    logger.info(f"Total transcription time: {total_transcription_time:.2f} seconds")
    if processed_files > 0:
        logger.info(
            f"Average transcription time per file: {total_transcription_time / processed_files:.2f} seconds"
        )
    logger.info("=" * 40)


def display_rich_summary(
    command_args: str,
    start_time: datetime,
    end_time: datetime,
    total_files: int,
    processed_files: int,
    skipped_files: int,
    total_transcription_time: float,
) -> None:
    """Display a rich formatted summary of the transcription run."""
    console = Console()

    total_runtime = (end_time - start_time).total_seconds()

    # Create summary table
    table = Table(
        title="Transcription Run Summary", show_header=True, header_style="bold magenta"
    )
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    table.add_row("Command", command_args)
    table.add_row("Start Time", start_time.strftime("%Y-%m-%d %H:%M:%S UTC"))
    table.add_row("End Time", end_time.strftime("%Y-%m-%d %H:%M:%S UTC"))
    table.add_row("Total Runtime", f"{total_runtime:.2f} seconds")
    table.add_row("Audio Files Found", str(total_files))
    table.add_row("Files Processed", str(processed_files))
    table.add_row("Files Skipped", str(skipped_files))
    table.add_row("Total Transcription Time", f"{total_transcription_time:.2f} seconds")

    if processed_files > 0:
        avg_time = total_transcription_time / processed_files
        table.add_row("Avg Time per File", f"{avg_time:.2f} seconds")

    # Display in a panel
    panel = Panel(table, title="🎤 Audio Transcription Complete", border_style="blue")
    console.print("\n")
    console.print(panel)
    console.print("\n")


# %%
def transcribe_audio(
    audio_path: Path,
    processor,
    model,
    device,
    model_id: str,
    model_type: str,
    language: str,
    max_new_tokens: int,
):
    """Transcribe audio file and return decoded outputs, timing, and start timestamp."""
    start_time = time.time()
    started_at = datetime.now(UTC)

    if model_type == "whisper":
        # Load audio file for Whisper
        audio, _ = librosa.load(audio_path, sr=16000)

        # Process audio with Whisper
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        inputs = inputs.to(device)

        # Ensure input features match model dtype
        if device == "cuda":
            inputs.input_features = inputs.input_features.to(torch.float16)

        # Generate transcription with specified max length for Whisper
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_features, max_new_tokens=max_new_tokens
            )

        decoded_outputs = processor.batch_decode(outputs, skip_special_tokens=True)

    elif model_type == "voxtral":
        # For Voxtral, we need to load audio with librosa first, then save as temp WAV file
        # because Voxtral's processor expects a file path but can't handle M4A directly
        import tempfile

        # Load audio using librosa (handles M4A files)
        audio_data, sample_rate = librosa.load(audio_path, sr=16000)

        # Create temporary WAV file for Voxtral
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            import soundfile as sf

            sf.write(temp_file.name, audio_data, sample_rate)
            temp_audio_path = temp_file.name

        try:
            # Process audio with Voxtral using temporary WAV file
            inputs = processor.apply_transcrition_request(
                language=language, audio=temp_audio_path, model_id=model_id
            )
        finally:
            # Clean up temporary file
            import os

            os.unlink(temp_audio_path)
        inputs = inputs.to(device)

        # Ensure input tensors match model dtype
        if device == "cuda":
            inputs = inputs.to(dtype=torch.bfloat16)

        # Generate transcription for Voxtral
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

        # Decode Voxtral outputs (skip input tokens)
        decoded_outputs = processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    end_time = time.time()
    elapsed_time = end_time - start_time

    return decoded_outputs, elapsed_time, started_at


# %%
def main():
    """Process all audio files for transcription."""
    # Record start time for logging
    run_start_time = datetime.now(UTC)

    parser = argparse.ArgumentParser(
        description="Transcribe audio files using Whisper or Voxtral models, store results in local data formats.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available output formats:
{chr(10).join(f"  {fmt}: {config['description']}" for fmt, config in OUTPUT_FORMATS.items())}

Available models:
{chr(10).join(f"  {model}: {config['description']}" for model, config in AVAILABLE_MODELS.items())}

Examples:
  python src/transcribe_audio.py --model whisper-tiny --format csv
  python src/transcribe_audio.py --model voxtral-mini --format json
  python src/transcribe_audio.py --model whisper-large-v3-turbo --format duckdb --all-audio
  python src/transcribe_audio.py --model whisper-large-v3 --format duckdb --all-audio --language en
  python src/transcribe_audio.py --model voxtral-small --format parquet
  python src/transcribe_audio.py --input-path /path/to/audio --output-path /path/to/output
  python src/transcribe_audio.py --input-path ~/recordings --output-path ~/results --model whisper-medium
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
        choices=list(AVAILABLE_MODELS.keys()),
        default="whisper-small",
        help="Model to use for transcription: Whisper or Voxtral (default: whisper-small)",
    )

    parser.add_argument(
        "--language",
        default="en",
        help="Language code for transcription (default: en). Voxtral supports: en, es, fr, pt, hi, de, nl, it. Whisper supports 99 languages.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=400,
        help="Maximum number of new tokens to generate (default: 400). Whisper models have a maximum limit of 448 tokens.",
    )

    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path(__file__).parent.parent / "audio",
        help="Path to directory containing audio files (default: ./audio)",
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).parent.parent / "output",
        help="Path to directory for output files (default: ./output)",
    )

    # Handle both command line and notebook execution
    try:
        args = parser.parse_args()
    except SystemExit:
        # If --help was requested, re-raise to exit properly
        if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help"]:
            raise

        # Otherwise, we're in a notebook/interactive environment, create default args
        class Args:
            all_audio = False
            format = "csv"
            model = "whisper-small"
            language = "en"
            max_new_tokens = 400
            input_path = Path(__file__).parent.parent / "audio"
            output_path = Path(__file__).parent.parent / "output"

        args = Args()

    # Validate max_new_tokens for Whisper models
    if args.model.startswith("whisper") and args.max_new_tokens > 448:
        print(
            "Warning: Whisper models have a maximum token limit of 448. Setting max_new_tokens to 448."
        )
        args.max_new_tokens = 448

    # Ensure output directory exists
    args.output_path.mkdir(parents=True, exist_ok=True)

    # Set up logging
    logger = setup_logging(args.output_path)

    # Build command string for logging
    command_args = " ".join(sys.argv)

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the specified model
    processor, model, model_id, model_type = load_model(args.model, device)

    # Get output file path
    output_file = get_output_filename(args.format, args.output_path)
    print(
        f"Output format: {args.format} ({OUTPUT_FORMATS[args.format]['description']})"
    )
    print(f"Output file: {output_file}")

    # Get all audio files
    audio_files = get_audio_files(args.input_path)
    if not audio_files:
        print(f"No audio files found in {args.input_path}")
        return

    print(f"Found {len(audio_files)} audio files in {args.input_path}")

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
    total_transcription_time = 0.0

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
                audio_file,
                processor,
                model,
                device,
                model_id,
                model_type,
                args.language,
                args.max_new_tokens,
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
            total_transcription_time += transcription_time

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

    # Record end time and display/log summary
    run_end_time = datetime.now(UTC)

    # Display rich formatted summary
    display_rich_summary(
        command_args,
        run_start_time,
        run_end_time,
        len(audio_files),
        processed_count,
        skipped_count,
        total_transcription_time,
    )

    # Log summary to file
    log_run_summary(
        logger,
        command_args,
        run_start_time,
        run_end_time,
        len(audio_files),
        processed_count,
        skipped_count,
        total_transcription_time,
    )


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
