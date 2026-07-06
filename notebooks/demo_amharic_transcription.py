# %%
import json
import time
from pathlib import Path

import ipywidgets as widgets
import librosa
import torch
from IPython.display import Audio, Markdown, display
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Constants
LANGUAGE = "am"
MAX_NEW_TOKENS = 400
AUDIO_DIR = Path("../audio/amharic")
METADATA_FILE = AUDIO_DIR / "metadata.json"

# %%
# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# %%
# Model setup - using Whisper small model for faster inference
# Whisper supports Amharic (language code "am") among its 99 languages
# Documentation: https://huggingface.co/openai/whisper-small
model_id = "openai/whisper-small"

processor = WhisperProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
model.to(device)

# %%
# Load audio file metadata
# Run first: uv run python src/download_amharic_audio.py
if not METADATA_FILE.exists():
    raise FileNotFoundError(
        f"Audio files not found at {AUDIO_DIR}.\n"
        "Run: uv run python src/download_amharic_audio.py"
    )

with METADATA_FILE.open(encoding="utf-8") as f:
    audio_files = json.load(f)

print(f"Found {len(audio_files)} audio samples in {AUDIO_DIR}")

# %%
# Helper Functions


def create_audio_player(audio_path):
    """Create an interactive audio player widget for the given audio file."""
    play_button = widgets.Button(description="Play Audio")

    def on_play_clicked(b):
        display(Audio(audio_path))

    play_button.on_click(on_play_clicked)
    return play_button


def transcribe_audio(audio_path, processor, model, device):
    """Transcribe audio file and return decoded Amharic outputs."""
    start_time = time.time()

    # Load audio file
    audio, sr = librosa.load(audio_path, sr=16000)

    # Process audio with Whisper
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    inputs = inputs.to(device)

    # Ensure input features match model dtype
    if device == "cuda":
        inputs.input_features = inputs.input_features.to(torch.float16)

    # Force Amharic transcription (prevents Whisper from translating to English)
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=LANGUAGE, task="transcribe"
    )

    # Generate transcription
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_features,
            forced_decoder_ids=forced_decoder_ids,
            max_new_tokens=MAX_NEW_TOKENS,
        )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Transcription completed in {elapsed_time:.2f} seconds")

    return processor.batch_decode(outputs, skip_special_tokens=True)


def display_transcription(decoded_outputs):
    """Display transcription results with consistent formatting."""
    print("\nTranscribed responses:")
    print("=" * 80)
    for decoded_output in decoded_outputs:
        display(Markdown(decoded_output))
        print("=" * 80)


def process_audio_file(audio_config, idx, processor, model, device):
    """Process a single audio file: display info, player, and transcription."""
    audio_path = AUDIO_DIR / audio_config["filename"]

    # Display sample information
    display(Markdown(f"### Sample {idx + 1} ({audio_config['duration_seconds']}s)"))
    display(Markdown(f"**Original text:**\n> {audio_config['transcript']}"))

    # Create and display audio player
    player = create_audio_player(str(audio_path))
    display(player)

    # Transcribe and display results
    decoded_outputs = transcribe_audio(str(audio_path), processor, model, device)
    display_transcription(decoded_outputs)


# %%
# Process All Audio Files

for idx, audio_config in enumerate(audio_files):
    process_audio_file(audio_config, idx, processor, model, device)

# %%
