# %%
import argparse
import hashlib
import json
import logging
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import librosa
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from transformers import (
    AutoProcessor,
    CohereAsrForConditionalGeneration,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    pipeline,
)

# Optional Voxtral support - only import if available
try:
    from transformers import VoxtralForConditionalGeneration

    VOXTRAL_AVAILABLE = True
except ImportError:
    VOXTRAL_AVAILABLE = False
    print(
        "Warning: Voxtral models not available. Run 'uv sync' to install "
        "transformers >=4.54."
    )

# torchcodec ships with pyannote but its native DLLs need FFmpeg, which this
# project does not require. transformers' ASR pipeline imports torchcodec
# whenever the package looks installed and crashes if the DLLs are broken -
# disable the detection when the import fails so the pipeline skips it.
try:
    import torchcodec  # noqa: F401
except Exception:
    import transformers.pipelines.automatic_speech_recognition as _asr_pipeline_mod
    import transformers.utils as _tf_utils

    _tf_utils.is_torchcodec_available = lambda: False
    _asr_pipeline_mod.is_torchcodec_available = lambda: False

# Optional pyannote support for speaker diarization - only import if available.
# Suppress pyannote's torchcodec/FFmpeg warning: its built-in audio decoding
# is unused here because diarize_audio() always passes in-memory waveforms.
try:
    import warnings

    with warnings.catch_warnings():
        # (?s) lets .* cross newlines - the warning text starts with one
        warnings.filterwarnings("ignore", message="(?s).*torchcodec.*")
        from pyannote.audio import Pipeline as PyannotePipeline

    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False

# Constants
AUDIO_DIR = Path(__file__).parent.parent / "audio"
OUTPUT_DIR = Path(__file__).parent.parent / "output"

# Speaker diarization model (pyannote). Gated on the HF Hub: accept the terms
# at https://hf.co/pyannote/speaker-diarization-community-1 and authenticate
# with an HF token for the first download. For fully offline use, download the
# model once and point --diarization-path (CLI) / "Local model directory"
# (GUI) at the local copy - no token or internet needed afterwards.
DIARIZATION_MODEL_ID = "pyannote/speaker-diarization-community-1"

# Optional LLM refinement of diarized transcripts via a local Ollama server
# (--refine CLI / "Refine with local LLM" toggle GUI). Fully local: no
# transcript content leaves the machine.
OLLAMA_DEFAULT_MODEL = "qwen3.5:4b"
OLLAMA_DEFAULT_URL = "http://localhost:11434"
OLLAMA_TIMEOUT_S = 600  # rewriting a long interview on a small LLM is slow
MAX_REFINE_CHARS = 60_000  # combined transcripts above this get chunked

REFINE_SYSTEM_PROMPT = (
    "You are a meticulous transcript editor. You merge two versions of the same "
    "audio transcript: one has accurate text but no speaker labels, the other has "
    "speaker labels but lower-quality text. You never invent, summarize, or omit "
    "content, and you never add commentary."
)

REFINE_USER_PROMPT = """Below are two transcripts of the SAME recording.

ACCURATE TRANSCRIPT (correct wording, no speakers):
{plain_text}

SPEAKER-LABELED TRANSCRIPT (correct speakers, lower-quality wording):
{diarized_text}

Rewrite the speaker-labeled transcript so that its wording matches the accurate
transcript. Rules:
1. Keep the exact output format: `SPEAKER_XX: text; SPEAKER_YY: text` - speaker
   turns separated by "; ", everything on a single line.
2. Keep the speaker labels and the order of turns from the speaker-labeled
   transcript. Merge consecutive turns only if they have the same speaker.
3. Take the wording from the accurate transcript wherever the two versions
   describe the same speech. Do not paraphrase, translate, add, or drop content.
4. Output ONLY the rewritten transcript - no explanations, headers, or markdown.
"""

# Languages supported by the Cohere Transcribe model (no auto-detect)
COHERE_LANGUAGES = {
    "en",
    "de",
    "fr",
    "it",
    "es",
    "pt",
    "el",
    "nl",
    "pl",
    "ar",
    "vi",
    "zh",
    "ja",
    "ko",
}

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
    "whisper-large-v3": {
        "id": "openai/whisper-large-v3",
        "type": "whisper",
        "description": "Most accurate Whisper model, slowest (~3090 MB)",
    },
    "whisper-large-v3-turbo": {
        "id": "openai/whisper-large-v3-turbo",
        "type": "whisper",
        "description": "Near large-v3 accuracy, much faster (~1550 MB)",
    },
    # Cohere model
    "cohere": {
        "id": "CohereLabs/cohere-transcribe-03-2026",
        "type": "cohere",
        "description": "Cohere Transcribe multilingual STT (~2B params, 14 languages, built-in long-form chunking)",
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
            model_id, dtype=torch.float16 if device == "cuda" else torch.float32
        )
        model.to(device)
    elif model_type == "voxtral":
        if not VOXTRAL_AVAILABLE:
            raise ValueError(
                "Voxtral models are not available. Run 'uv sync' to install transformers >=4.54."
            )
        processor = AutoProcessor.from_pretrained(model_id)
        model = VoxtralForConditionalGeneration.from_pretrained(
            model_id,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map=device if device == "cuda" else None,
        )
        if device == "cpu":
            model.to(device)
    elif model_type == "cohere":
        # Native transformers support (>=5.4); the processor chunks long
        # recordings into overlapping windows and decode() reassembles them.
        try:
            processor = AutoProcessor.from_pretrained(model_id)
            model = CohereAsrForConditionalGeneration.from_pretrained(
                model_id,
                dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            )
        except OSError as e:
            # The gated-repo 401/403 is often buried in the exception chain
            # (transformers re-raises it as a generic connection error).
            chain, cause = [], e
            while cause is not None and len(chain) < 5:
                chain.append(str(cause))
                cause = cause.__cause__
            chain_text = " ".join(chain).lower()
            if "gated" in chain_text or "401" in chain_text:
                raise ValueError(
                    f"{model_id} is a gated model. Accept the terms at "
                    f"https://hf.co/{model_id} and authenticate with "
                    "'huggingface-cli login' or the HF_TOKEN environment "
                    "variable, then retry. The model is cached locally after "
                    "the first download."
                ) from e
            if "403" in chain_text or "permission" in chain_text:
                raise ValueError(
                    f"Access to {model_id} was denied (403). Your HF token "
                    "lacks permission for gated repositories: at "
                    "https://hf.co/settings/tokens either use a classic "
                    "'Read' token, or edit your fine-grained token and "
                    "enable 'Read access to contents of all public gated "
                    "repos you can access'. Also confirm you accepted the "
                    f"terms at https://hf.co/{model_id}."
                ) from e
            raise
        model.to(device)
        model.eval()
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
    try:
        console.print("\n")
        console.print(panel)
        console.print("\n")
    except UnicodeEncodeError:
        # Legacy Windows consoles (cp1252) can't render the emoji/box glyphs
        print("\n=== Audio Transcription Complete ===")
        print(f"Files processed: {processed_files}, skipped: {skipped_files}")


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
    chunked: bool = False,
):
    """Transcribe audio file and return decoded outputs, timing, and start timestamp.

    Whisper recordings longer than 30 seconds are transcribed in full using
    sequential long-form generation, or the faster chunked pipeline when
    ``chunked`` is True. ``chunked`` is ignored for Voxtral and Cohere models.
    Cohere models chunk long recordings internally and manage output length
    themselves, so ``max_new_tokens`` is also ignored for them.
    """
    start_time = time.perf_counter()
    started_at = datetime.now(UTC)

    if model_type == "whisper":
        # Load audio file for Whisper
        audio, _ = librosa.load(audio_path, sr=16000)
        duration = len(audio) / 16000

        if duration > 30 and chunked:
            # Fast mode: chunked pipeline with overlapping 30s windows,
            # batched for throughput. Slightly less accurate at boundaries.
            asr = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                chunk_length_s=30,
                batch_size=8 if device == "cuda" else 2,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device=device,
            )
            generate_kwargs = {}
            if language and language != "auto":
                generate_kwargs["language"] = language
                generate_kwargs["task"] = "transcribe"
            result = asr(audio, generate_kwargs=generate_kwargs or None)
            decoded_outputs = [result["text"]]
        else:
            # Process audio with Whisper; request the attention mask explicitly
            # (Whisper's pad token equals its eos token, so generate() cannot
            # infer the mask and warns without it). For audio over 30s,
            # truncation=False/padding="longest" switches generate() into
            # sequential long-form mode, which transcribes the full recording.
            long_form = duration > 30
            inputs = processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
                return_attention_mask=True,
                truncation=not long_form,
                padding="longest" if long_form else True,
            )
            inputs = inputs.to(device)

            # Ensure input features match model dtype
            if device == "cuda":
                inputs.input_features = inputs.input_features.to(torch.float16)

            # Force the target language unless "auto" is requested (auto-detect)
            generate_kwargs = {}
            if long_form:
                # Long-form generation needs timestamps; max_new_tokens is
                # managed per 30s segment internally, so don't pass it here
                generate_kwargs["return_timestamps"] = True
            else:
                generate_kwargs["max_new_tokens"] = max_new_tokens
            if getattr(inputs, "attention_mask", None) is not None:
                generate_kwargs["attention_mask"] = inputs.attention_mask
            if language and language != "auto":
                generate_kwargs["language"] = language
                generate_kwargs["task"] = "transcribe"
            with torch.no_grad():
                outputs = model.generate(inputs.input_features, **generate_kwargs)

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
            inputs = processor.apply_transcription_request(
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

    elif model_type == "cohere":
        if not language or language == "auto":
            raise ValueError(
                "Cohere models do not support language auto-detection. "
                f"Choose one of: {', '.join(sorted(COHERE_LANGUAGES))}"
            )
        if language not in COHERE_LANGUAGES:
            raise ValueError(
                f"Cohere models do not support language '{language}'. "
                f"Choose one of: {', '.join(sorted(COHERE_LANGUAGES))}"
            )

        # The processor splits recordings over ~35s into overlapping chunks
        # (audio_chunk_index maps chunks back to the sample) and decode()
        # reassembles the per-chunk texts into one transcript.
        audio_data, _ = librosa.load(audio_path, sr=16000)
        inputs = processor(
            audio_data, sampling_rate=16000, return_tensors="pt", language=language
        )
        audio_chunk_index = inputs.get("audio_chunk_index")
        inputs = inputs.to(device)
        if device == "cuda":
            inputs["input_features"] = inputs["input_features"].to(torch.bfloat16)

        # 448 comfortably covers a 35s chunk; the user-facing max_new_tokens
        # is ignored for Cohere (output length is per-chunk, not per-file)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=448)

        decoded = processor.decode(
            outputs,
            skip_special_tokens=True,
            audio_chunk_index=audio_chunk_index,
            language=language,
        )
        decoded_outputs = [decoded] if isinstance(decoded, str) else list(decoded)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    return decoded_outputs, elapsed_time, started_at


# %%
def load_diarization_pipeline(
    device: str,
    hf_token: str | None = None,
    model_path: Path | str | None = None,
):
    """Load the pyannote speaker diarization pipeline.

    ``model_path`` loads a locally downloaded copy of the model (fully
    offline, no token needed). Otherwise the model is fetched from the HF
    Hub, which requires accepting its gated terms and authenticating via
    ``hf_token``, the HF_TOKEN environment variable, or a cached
    ``huggingface-cli login``.
    """
    if not PYANNOTE_AVAILABLE:
        raise ValueError(
            "Speaker diarization requires pyannote.audio. Run 'uv sync' to install it."
        )

    source = str(model_path) if model_path else DIARIZATION_MODEL_ID
    try:
        diarization_pipeline = PyannotePipeline.from_pretrained(
            source, token=hf_token or None
        )
    except Exception as e:
        if model_path:
            raise ValueError(
                f"Could not load diarization model from '{model_path}'. "
                "Point --diarization-path at a local download of "
                f"{DIARIZATION_MODEL_ID} (see the model card's 'Offline use' "
                f"section). Original error: {e}"
            ) from e
        raise ValueError(
            f"Could not load {DIARIZATION_MODEL_ID} from the HF Hub. The model "
            "is gated: accept the terms at "
            f"https://hf.co/{DIARIZATION_MODEL_ID} and provide an HF token "
            "(--hf-token, the HF_TOKEN environment variable, or "
            "'huggingface-cli login'). For offline use, download the model "
            f"once and pass --diarization-path. Original error: {e}"
        ) from e

    if device == "cuda":
        diarization_pipeline.to(torch.device("cuda"))
    return diarization_pipeline


def diarize_audio(
    audio_path: Path,
    diarization_pipeline,
    num_speakers: int | None = None,
    min_turn_duration: float = 0.5,
    merge_gap: float = 0.5,
) -> list[tuple[float, float, str]]:
    """Run speaker diarization and return merged speaker turns.

    Returns a list of ``(start_seconds, end_seconds, speaker_label)`` tuples.
    Consecutive turns by the same speaker separated by less than
    ``merge_gap`` seconds are merged; turns shorter than
    ``min_turn_duration`` seconds are dropped.
    """
    # Hand pyannote an in-memory waveform rather than the file path: librosa
    # handles all our formats (incl. M4A) and this avoids pyannote's own
    # audio decoding, which is fragile on Windows.
    audio_data, sample_rate = librosa.load(audio_path, sr=16000)
    waveform = torch.from_numpy(audio_data).unsqueeze(0)
    kwargs = {}
    if num_speakers:
        kwargs["num_speakers"] = num_speakers
    output = diarization_pipeline(
        {"waveform": waveform, "sample_rate": sample_rate}, **kwargs
    )

    # exclusive_speaker_diarization (pyannote >= community-1) assigns each
    # instant to a single speaker, which suits transcription alignment.
    annotation = getattr(output, "exclusive_speaker_diarization", None)
    if annotation is None:
        annotation = getattr(output, "speaker_diarization", output)

    turns: list[tuple[float, float, str]] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        if turns and turns[-1][2] == speaker and turn.start - turns[-1][1] < merge_gap:
            turns[-1] = (turns[-1][0], turn.end, speaker)
        else:
            turns.append((turn.start, turn.end, speaker))

    return [
        (start, end, speaker)
        for start, end, speaker in turns
        if end - start >= min_turn_duration
    ]


def transcribe_whisper_with_timestamps(
    audio_path: Path,
    processor,
    model,
    device,
    language: str,
    chunked: bool = False,
) -> list[dict]:
    """Transcribe full audio with Whisper, returning timestamped segments.

    Returns ``[{"start": float, "end": float, "text": str}, ...]``. Because the
    whole recording is transcribed in one pass, the text quality matches plain
    (non-diarized) Whisper output. A ``None`` end timestamp on the final
    segment is replaced with the audio duration; segments without a usable
    start timestamp or with empty text are dropped. Returns ``[]`` when the
    pipeline yields no usable segments (the caller falls back to per-turn
    slicing).
    """
    audio, _ = librosa.load(audio_path, sr=16000)
    duration = len(audio) / 16000

    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30 if chunked else None,
        batch_size=(8 if device == "cuda" else 2) if chunked else 1,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device=device,
        return_timestamps=True,
    )
    generate_kwargs = {}
    if language and language != "auto":
        generate_kwargs["language"] = language
        generate_kwargs["task"] = "transcribe"
    result = asr(audio, generate_kwargs=generate_kwargs or None)

    segments = []
    for chunk in result.get("chunks", []):
        start, end = chunk.get("timestamp") or (None, None)
        text = (chunk.get("text") or "").strip()
        if start is None or not text:
            continue
        if end is None:
            end = duration
        segments.append({"start": float(start), "end": float(end), "text": text})
    return segments


def assign_speakers_to_segments(
    segments: list[dict],
    turns: list[tuple[float, float, str]],
) -> list[tuple[str, str]]:
    """Assign each timestamped segment a speaker by maximal temporal overlap.

    Segments that fall in a diarization gap are assigned to the nearest turn
    by midpoint distance. Returns merged ``(speaker, text)`` turns -
    consecutive segments with the same speaker are concatenated with a space.
    """
    speaker_turns: list[tuple[str, str]] = []
    last_speaker = turns[0][2] if turns else "SPEAKER_00"
    for seg in segments:
        best_speaker, best_overlap = None, 0.0
        for turn_start, turn_end, speaker in turns:
            overlap = max(
                0.0, min(seg["end"], turn_end) - max(seg["start"], turn_start)
            )
            if overlap > best_overlap:
                best_overlap, best_speaker = overlap, speaker
        if best_speaker is None:
            # Segment falls in a diarization gap: use the nearest turn
            mid = (seg["start"] + seg["end"]) / 2
            best_speaker = min(
                turns,
                key=lambda t: min(abs(mid - t[0]), abs(mid - t[1])),
                default=(0.0, 0.0, last_speaker),
            )[2]
        last_speaker = best_speaker
        if speaker_turns and speaker_turns[-1][0] == best_speaker:
            speaker_turns[-1] = (
                best_speaker,
                f"{speaker_turns[-1][1]} {seg['text']}",
            )
        else:
            speaker_turns.append((best_speaker, seg["text"]))
    return speaker_turns


def format_speaker_transcript(speaker_turns: list[tuple[str, str]]) -> str:
    """Format merged (speaker, text) turns as 'SPEAKER_XX: text; SPEAKER_YY: text'."""
    return "; ".join(f"{speaker}: {text.strip()}" for speaker, text in speaker_turns)


class OllamaError(Exception):
    """Raised when the Ollama refinement request fails or returns bad output."""


def _ollama_num_ctx(prompt_chars: int) -> int:
    """Pick an Ollama num_ctx that fits the prompt plus the rewritten output.

    ~3 chars/token (conservative for multilingual text); the output is roughly
    the diarized transcript again, so budget 2x prompt tokens plus headroom,
    rounded up to the next tier.
    """
    needed = (prompt_chars // 3) * 2 + 1024
    for tier in (8192, 16384, 32768):
        if needed <= tier:
            return tier
    return 32768


def _clean_llm_transcript(raw: str) -> str:
    """Normalize an LLM response back to the single-line SPEAKER_XX format."""
    text = raw.strip()
    # Strip Markdown code fences
    text = re.sub(r"^```[a-zA-Z]*\s*|```$", "", text).strip()
    # Drop any preamble before the first speaker label
    first = text.find("SPEAKER_")
    if first == -1:
        raise OllamaError("model response did not contain speaker labels")
    text = text[first:]
    # Newline-separated turns back to the "; " single-line format
    text = re.sub(r"\s*\n+\s*(?=SPEAKER_)", "; ", text)
    return re.sub(r"\s*\n+\s*", " ", text).strip()


def _split_diarized_for_refine(diarized_text: str, max_chars: int) -> list[str]:
    """Split a diarized transcript at turn boundaries into chunks under max_chars."""
    turns = diarized_text.split("; SPEAKER_")
    parts = [turns[0]] + [f"SPEAKER_{t}" for t in turns[1:]]
    chunks: list[str] = []
    current = ""
    for part in parts:
        candidate = f"{current}; {part}" if current else part
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = part
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def refine_transcript_with_llm(
    plain_text: str,
    diarized_text: str,
    ollama_model: str = OLLAMA_DEFAULT_MODEL,
    ollama_url: str = OLLAMA_DEFAULT_URL,
    timeout: float = OLLAMA_TIMEOUT_S,
) -> str:
    """Rewrite a rough speaker-labeled transcript using the accurate plain text.

    Sends both transcripts to a local Ollama server and asks the model to keep
    the speaker structure while adopting the accurate wording. Raises
    :class:`OllamaError` on connection failure, HTTP error, timeout, or a
    response without SPEAKER_ labels - the caller falls back to the unrefined
    transcript. Long inputs are split at turn boundaries and refined in
    sequential chunks paired with proportional slices of the plain text.
    """
    if len(plain_text) + len(diarized_text) > MAX_REFINE_CHARS:
        diarized_chunks = _split_diarized_for_refine(
            diarized_text, MAX_REFINE_CHARS // 2
        )
        plain_words = plain_text.split()
        refined_chunks = []
        consumed = 0.0
        for chunk in diarized_chunks:
            fraction = len(chunk) / len(diarized_text)
            start_word = int(consumed * len(plain_words))
            end_word = min(
                len(plain_words), int((consumed + fraction) * len(plain_words))
            )
            consumed += fraction
            plain_slice = " ".join(plain_words[start_word:end_word])
            refined_chunks.append(
                refine_transcript_with_llm(
                    plain_slice, chunk, ollama_model, ollama_url, timeout
                )
            )
        return "; ".join(refined_chunks)

    user_prompt = REFINE_USER_PROMPT.format(
        plain_text=plain_text, diarized_text=diarized_text
    )
    body = {
        "model": ollama_model,
        "messages": [
            {"role": "system", "content": REFINE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        # Thinking models (qwen3+, etc.) can spend their whole token budget
        # on chain-of-thought and return empty content for long transcripts -
        # this is a mechanical rewrite, so disable thinking.
        "think": False,
        "options": {
            "num_ctx": _ollama_num_ctx(len(REFINE_SYSTEM_PROMPT) + len(user_prompt)),
            "temperature": 0.1,
        },
    }
    try:
        response = requests.post(
            f"{ollama_url.rstrip('/')}/api/chat", json=body, timeout=timeout
        )
        if response.status_code == 400 and "think" in response.text.lower():
            # Older Ollama servers / non-thinking models may reject the field
            body.pop("think", None)
            response = requests.post(
                f"{ollama_url.rstrip('/')}/api/chat", json=body, timeout=timeout
            )
    except requests.exceptions.ConnectionError as e:
        raise OllamaError(
            f"Could not reach Ollama at {ollama_url}. Install it from "
            f"https://ollama.com, run 'ollama pull {ollama_model}', and make "
            "sure the server is running ('ollama serve' or the desktop app)."
        ) from e
    except requests.exceptions.Timeout as e:
        raise OllamaError(
            f"Ollama request timed out after {timeout:.0f}s - try a smaller "
            "--ollama-model or a shorter recording."
        ) from e

    if response.status_code == 404:
        raise OllamaError(
            f"Ollama model '{ollama_model}' is not installed - run "
            f"'ollama pull {ollama_model}'."
        )
    try:
        response.raise_for_status()
        content = response.json()["message"]["content"]
    except Exception as e:
        raise OllamaError(f"Unexpected Ollama response: {e}") from e

    return _clean_llm_transcript(content)


def transcribe_with_diarization(
    audio_path: Path,
    diarization_pipeline,
    processor,
    model,
    device,
    model_id: str,
    model_type: str,
    language: str,
    max_new_tokens: int,
    chunked: bool = False,
    num_speakers: int | None = None,
    refine: bool = False,
    ollama_model: str = OLLAMA_DEFAULT_MODEL,
    ollama_url: str = OLLAMA_DEFAULT_URL,
):
    """Transcribe audio with speaker labels ("who said what").

    Runs pyannote diarization to find speaker turns and returns
    ``(decoded_outputs, elapsed_time, started_at, refined)``, where the single
    decoded output is a one-line transcript of ``SPEAKER_XX: text`` turns
    separated by ``"; "`` (consecutive same-speaker turns concatenated).

    Whisper models transcribe the FULL recording once with segment timestamps
    and assign speakers by temporal overlap with the pyannote turns, so the
    text quality matches plain Whisper output. Cohere and Voxtral models emit
    no timestamps, so each speaker turn is sliced and transcribed separately.

    When ``refine`` is True, the diarized transcript is additionally rewritten
    by a local Ollama LLM using a full-quality plain transcript as reference;
    on any Ollama failure a warning is printed and the unrefined transcript is
    returned (``refined`` is False). Falls back to plain transcription if no
    speaker turns are found.
    """
    import os
    import tempfile

    import soundfile as sf

    start_time = time.perf_counter()
    started_at = datetime.now(UTC)

    turns = diarize_audio(audio_path, diarization_pipeline, num_speakers)
    if not turns:
        decoded_outputs, _, _ = transcribe_audio(
            audio_path,
            processor,
            model,
            device,
            model_id,
            model_type,
            language,
            max_new_tokens,
            chunked=chunked,
        )
        elapsed_time = time.perf_counter() - start_time
        return decoded_outputs, elapsed_time, started_at, False

    speaker_turns: list[tuple[str, str]] = []
    plain_text = None
    use_slicing = model_type != "whisper"

    if model_type == "whisper":
        segments = transcribe_whisper_with_timestamps(
            audio_path, processor, model, device, language, chunked=chunked
        )
        if segments:
            speaker_turns = assign_speakers_to_segments(segments, turns)
            # The full-context transcript comes free from the same segments
            plain_text = " ".join(seg["text"].strip() for seg in segments).strip()
        else:
            print(
                "Warning: Whisper returned no usable timestamps; falling back "
                "to per-turn transcription."
            )
            use_slicing = True

    if use_slicing:
        audio_data, sample_rate = librosa.load(audio_path, sr=16000)
        for turn_start, turn_end, speaker in turns:
            segment = audio_data[
                int(turn_start * sample_rate) : int(turn_end * sample_rate)
            ]
            if segment.size == 0:
                continue

            # Write the slice to a temp WAV (closed before use for Windows) so
            # every model type can transcribe it through transcribe_audio().
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                sf.write(temp_file.name, segment, sample_rate)
                temp_segment_path = Path(temp_file.name)
            try:
                decoded_outputs, _, _ = transcribe_audio(
                    temp_segment_path,
                    processor,
                    model,
                    device,
                    model_id,
                    model_type,
                    language,
                    max_new_tokens,
                    chunked=chunked,
                )
            finally:
                os.unlink(temp_segment_path)

            text = " ".join(part.strip() for part in decoded_outputs).strip()
            if not text:
                continue
            if speaker_turns and speaker_turns[-1][0] == speaker:
                speaker_turns[-1] = (speaker, f"{speaker_turns[-1][1]} {text}")
            else:
                speaker_turns.append((speaker, text))

    transcript = format_speaker_transcript(speaker_turns)

    refined = False
    if refine and transcript:
        if plain_text is None:
            # Cohere/Voxtral: run the full-quality plain transcription too
            decoded_outputs, _, _ = transcribe_audio(
                audio_path,
                processor,
                model,
                device,
                model_id,
                model_type,
                language,
                max_new_tokens,
                chunked=chunked,
            )
            plain_text = " ".join(part.strip() for part in decoded_outputs).strip()
        try:
            transcript = refine_transcript_with_llm(
                plain_text, transcript, ollama_model, ollama_url
            )
            refined = True
        except OllamaError as e:
            print(
                f"Warning: LLM refinement failed ({e}); using the unrefined "
                "diarized transcript."
            )

    elapsed_time = time.perf_counter() - start_time
    return [transcript], elapsed_time, started_at, refined


# %%
def main():
    """Process all audio files for transcription."""
    # Record start time for logging
    run_start_time = datetime.now(UTC)

    parser = argparse.ArgumentParser(
        description="Transcribe audio files using Whisper, Voxtral, or Cohere models, store results in local data formats.",
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
  python src/transcribe_audio.py --model cohere --language en --format csv
  python src/transcribe_audio.py --model whisper-small --diarize --num-speakers 2
  python src/transcribe_audio.py --model cohere --language en --diarize --refine
  python src/transcribe_audio.py --model cohere --diarize --diarization-path ~/models/pyannote-speaker-diarization-community-1
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
        help="Model to use for transcription: Whisper, Voxtral, or Cohere (default: whisper-small)",
    )

    parser.add_argument(
        "--language",
        default="en",
        help="Language code for transcription (default: en). Use 'auto' for Whisper language auto-detection. Voxtral supports: en, es, fr, pt, hi, de, nl, it. Cohere supports: "
        + ", ".join(sorted(COHERE_LANGUAGES))
        + " (no auto-detect). Whisper supports 99 languages.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=400,
        help="Maximum number of new tokens to generate (default: 400). Whisper models have a maximum limit of 448 tokens; for recordings over 30 seconds the limit applies per 30-second segment.",
    )

    parser.add_argument(
        "--chunked",
        action="store_true",
        help="Faster chunked transcription for long recordings (Whisper only); may lose accuracy at chunk boundaries. Default is sequential long-form processing. Ignored for Voxtral models.",
    )

    parser.add_argument(
        "--diarize",
        action="store_true",
        help="Label speakers ('who said what') using pyannote speaker diarization. "
        f"Uses {DIARIZATION_MODEL_ID}, a gated model: accept its terms on the HF Hub "
        "and authenticate with --hf-token / HF_TOKEN / 'huggingface-cli login' for "
        "the first download, or pass --diarization-path for fully offline use. "
        "Note: files already transcribed without diarization are skipped unless "
        "--all-audio is used.",
    )

    parser.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Exact number of speakers in the recordings, if known (improves diarization). Default: detect automatically.",
    )

    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face access token for downloading the gated diarization model. Falls back to the HF_TOKEN environment variable or a cached 'huggingface-cli login'.",
    )

    parser.add_argument(
        "--diarization-path",
        type=Path,
        default=None,
        help=f"Path to a local download of {DIARIZATION_MODEL_ID} for fully offline diarization (see the model card's 'Offline use' section). No token or internet needed.",
    )

    parser.add_argument(
        "--refine",
        action="store_true",
        help="Refine the diarized transcript with a local LLM served by Ollama "
        "(https://ollama.com): the speaker labels are combined with a "
        "full-quality transcript of the same recording. Requires --diarize and "
        "a running Ollama server; falls back to the unrefined transcript with "
        "a warning if Ollama is unavailable.",
    )

    parser.add_argument(
        "--ollama-model",
        default=OLLAMA_DEFAULT_MODEL,
        help=f"Ollama model for --refine (default: {OLLAMA_DEFAULT_MODEL}). Install with 'ollama pull <model>'.",
    )

    parser.add_argument(
        "--ollama-url",
        default=OLLAMA_DEFAULT_URL,
        help=f"Base URL of the Ollama server (default: {OLLAMA_DEFAULT_URL}).",
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
            chunked = False
            diarize = False
            num_speakers = None
            hf_token = None
            diarization_path = None
            refine = False
            ollama_model = OLLAMA_DEFAULT_MODEL
            ollama_url = OLLAMA_DEFAULT_URL
            input_path = Path(__file__).parent.parent / "audio"
            output_path = Path(__file__).parent.parent / "output"

        args = Args()

    # Validate max_new_tokens for Whisper models
    if args.model.startswith("whisper") and args.max_new_tokens > 448:
        print(
            "Warning: Whisper models have a maximum token limit of 448. Setting max_new_tokens to 448."
        )
        args.max_new_tokens = 448

    # --refine only makes sense on top of diarization
    if args.refine and not args.diarize:
        parser.error("--refine requires --diarize")

    # Validate language for Cohere models (no auto-detect, 14 languages)
    if (
        AVAILABLE_MODELS[args.model]["type"] == "cohere"
        and args.language not in COHERE_LANGUAGES
    ):
        parser.error(
            f"Cohere models do not support language '{args.language}'. "
            f"Choose one of: {', '.join(sorted(COHERE_LANGUAGES))}"
        )

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

    # Load the diarization pipeline once, if requested
    diarization_pipeline = None
    record_model_id = model_id
    if args.diarize:
        print("Loading speaker diarization pipeline...")
        diarization_pipeline = load_diarization_pipeline(
            device, hf_token=args.hf_token, model_path=args.diarization_path
        )
        record_model_id = f"{model_id} + {DIARIZATION_MODEL_ID}"
        print("Diarization pipeline loaded successfully!")

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

            # Transcribe audio (with speaker labels when --diarize is set)
            file_model_id = record_model_id
            if diarization_pipeline is not None:
                decoded_outputs, transcription_time, started_at, refined = (
                    transcribe_with_diarization(
                        audio_file,
                        diarization_pipeline,
                        processor,
                        model,
                        device,
                        model_id,
                        model_type,
                        args.language,
                        args.max_new_tokens,
                        chunked=args.chunked,
                        num_speakers=args.num_speakers,
                        refine=args.refine,
                        ollama_model=args.ollama_model,
                        ollama_url=args.ollama_url,
                    )
                )
                if refined:
                    file_model_id = f"{record_model_id} + ollama:{args.ollama_model}"
            else:
                decoded_outputs, transcription_time, started_at = transcribe_audio(
                    audio_file,
                    processor,
                    model,
                    device,
                    model_id,
                    model_type,
                    args.language,
                    args.max_new_tokens,
                    chunked=args.chunked,
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
                file_model_id,
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
