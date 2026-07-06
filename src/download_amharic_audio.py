"""Download a subset of Amharic audio samples for offline use.

Usage:
    uv run python src/download_amharic_audio.py
    uv run python src/download_amharic_audio.py --num-samples 10 --split validation
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import load_dataset

OUTPUT_DIR = Path(__file__).parent.parent / "audio" / "amharic"
METADATA_FILE = OUTPUT_DIR / "metadata.json"
DEFAULT_NUM_SAMPLES = 5
DEFAULT_SPLIT = "validation"


def download_samples(num_samples: int, split: str) -> None:
    """Download audio samples from the Amharic speech dataset."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {num_samples} samples from '{split}' split (streaming)...")
    ds = load_dataset("shunyalabs/amharic-speech-dataset", split=split, streaming=True)
    samples = list(ds.take(num_samples))

    metadata = []
    for idx, sample in enumerate(samples):
        audio_array = sample["audio"]["array"].astype(np.float32)
        sample_rate = sample["audio"]["sampling_rate"]
        transcript = sample["transcript"]

        filename = f"sample_{idx + 1:02d}.wav"
        output_path = OUTPUT_DIR / filename
        sf.write(output_path, audio_array, sample_rate)

        duration = len(audio_array) / sample_rate
        metadata.append(
            {
                "filename": filename,
                "transcript": transcript,
                "sample_rate": sample_rate,
                "duration_seconds": round(duration, 2),
                "split": split,
            }
        )
        print(f"  Saved {filename} ({duration:.1f}s): {transcript[:60]}...")

    METADATA_FILE.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"\nDownloaded {len(samples)} samples to {OUTPUT_DIR}")
    print(f"Metadata saved to {METADATA_FILE}")


def main() -> None:
    """Parse arguments and download samples."""
    parser = argparse.ArgumentParser(description="Download Amharic audio samples")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help=f"Number of samples to download (default: {DEFAULT_NUM_SAMPLES})",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        choices=["train", "validation", "test"],
        help=f"Dataset split to use (default: {DEFAULT_SPLIT})",
    )
    args = parser.parse_args()
    download_samples(args.num_samples, args.split)


if __name__ == "__main__":
    main()
