"""Streamlit GUI for the audio transcription workflow.

Upload audio files, choose a speech-to-text model and language, and
transcribe without touching the command line. Results are kept in the
browser session and offered as CSV/JSON downloads; nothing is written
to the shared ``output/`` files used by the CLI.

Launch with: ``just app`` or ``uv run streamlit run src/app.py``
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import torch

# Make the sibling module importable regardless of the working directory
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import transcribe_audio as ta  # noqa: E402

AUDIO_TYPES = ["mp3", "wav", "flac", "m4a", "ogg"]

WHISPER_MAX_TOKENS = 448
VOXTRAL_MAX_TOKENS = 2048

# Languages supported by Voxtral models
VOXTRAL_LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "pt": "Portuguese",
    "hi": "Hindi",
    "de": "German",
    "nl": "Dutch",
    "it": "Italian",
}

# Curated subset of Whisper's 99 supported languages
WHISPER_LANGUAGES = {
    "auto": "Auto-detect",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
    "it": "Italian",
    "nl": "Dutch",
    "hi": "Hindi",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "ru": "Russian",
    "tr": "Turkish",
    "pl": "Polish",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "th": "Thai",
    "sw": "Swahili",
    "am": "Amharic",
    "yo": "Yoruba",
    "ha": "Hausa",
    "tl": "Tagalog",
    "ur": "Urdu",
    "bn": "Bengali",
}

RESULT_COLUMNS = [
    "filename",
    "transcription_text",
    "model_id",
    "transcription_time_seconds",
    "processed_at",
]

# IPA brand styling (see .claude/skills/ipa-branding/): headings render in
# IPA Green per the typography guidelines; fonts and palette come from
# .streamlit/config.toml.
IPA_GREEN = "#49ac57"
IPA_CSS = f"""
<style>
h1, h2, h3 {{
    color: {IPA_GREEN} !important;
}}
</style>
"""


def get_device(choice: str = "auto") -> str:
    """Resolve a device choice (auto/cpu/gpu) to a torch device string."""
    if choice == "cpu":
        return "cpu"
    if choice == "gpu":
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource(show_spinner="Loading model (first load may take a while)...")
def load_model_cached(model_name: str, device: str):
    """Load and cache a model so switching settings doesn't reload it."""
    return ta.load_model(model_name, device)


def transcribe_upload(
    uploaded_file,
    processor,
    model,
    device: str,
    model_id: str,
    model_type: str,
    language: str,
    max_new_tokens: int,
    chunked: bool = False,
) -> dict:
    """Transcribe one uploaded file and return a transcription record."""
    data = uploaded_file.getvalue()
    suffix = Path(uploaded_file.name).suffix or ".wav"

    # Write to a temp file and close it before transcribing: transcribe_audio
    # needs a real path, and Windows can't reopen a file that is still open.
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    try:
        decoded_outputs, elapsed_time, started_at = ta.transcribe_audio(
            tmp_path,
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
        os.unlink(tmp_path)

    text = " ".join(part.strip() for part in decoded_outputs).strip()
    file_id = ta.generate_file_id(uploaded_file.name, len(data))
    return ta.create_transcription_record(
        uploaded_file.name,
        file_id,
        len(data),
        elapsed_time,
        text,
        model_id,
        started_at,
    )


def results_dataframe(records: list[dict]) -> pd.DataFrame:
    """Build a display dataframe from transcription records."""
    df = pd.DataFrame(records)
    return df[[col for col in RESULT_COLUMNS if col in df.columns]]


def render_sidebar() -> dict:
    """Render sidebar controls and return the selected settings."""
    st.sidebar.header("Settings")

    model_name = st.sidebar.selectbox(
        "Model",
        list(ta.AVAILABLE_MODELS),
        index=list(ta.AVAILABLE_MODELS).index("whisper-small"),
        format_func=lambda m: f"{m} — {ta.AVAILABLE_MODELS[m]['description']}",
        help="Whisper models support ~99 languages. Voxtral models support "
        "8 languages and may be more accurate for multilingual audio.",
    )
    model_type = ta.AVAILABLE_MODELS[model_name]["type"]

    if model_type == "voxtral":
        languages = VOXTRAL_LANGUAGES
        default_language = "en"
    else:
        languages = WHISPER_LANGUAGES
        default_language = "en"
    language = st.sidebar.selectbox(
        "Audio language",
        list(languages),
        index=list(languages).index(default_language),
        format_func=lambda code: f"{languages[code]} ({code})",
        help="Pick the language spoken in the recording. For Whisper models, "
        "'Auto-detect' lets the model guess the language.",
    )

    token_cap = WHISPER_MAX_TOKENS if model_type == "whisper" else VOXTRAL_MAX_TOKENS
    max_new_tokens = st.sidebar.slider(
        "Max new tokens",
        min_value=64,
        max_value=token_cap,
        value=min(400, token_cap),
        step=16,
        help="Upper limit on the length of the generated transcript. "
        f"Whisper models cap at {WHISPER_MAX_TOKENS} tokens; for recordings "
        "over 30 seconds the limit applies per 30-second segment.",
    )

    if model_type == "whisper":
        chunked = st.sidebar.toggle(
            "Fast chunked mode (long recordings)",
            value=False,
            key="chunked_mode",
            help="Transcribe recordings over 30 seconds in parallel 30-second "
            "chunks. Faster (especially on a GPU) but may lose accuracy at "
            "chunk boundaries. Off = sequential long-form processing, which "
            "is slower but most accurate.",
        )
    else:
        chunked = False

    st.sidebar.toggle(
        "Multiple speakers (diarization)",
        value=False,
        disabled=True,
        help="Coming soon — speaker diarization (who said what) is not yet "
        "supported. Transcripts currently treat all speech as one stream.",
    )
    st.sidebar.caption("Speaker diarization is coming in a future release.")

    st.sidebar.divider()
    cuda_available = torch.cuda.is_available()
    device_options = {"auto": "Auto", "cpu": "CPU"}
    if cuda_available:
        device_options["gpu"] = f"GPU ({torch.cuda.get_device_name(0)})"
    device_choice = st.sidebar.selectbox(
        "Compute device",
        list(device_options),
        format_func=lambda key: device_options[key],
        help="Auto uses the GPU when one is available. Choose CPU if a larger "
        "model runs out of GPU memory.",
    )
    device = get_device(device_choice)
    if cuda_available:
        st.sidebar.caption(f"Transcribing on: {device.upper()}")
    else:
        st.sidebar.caption("No CUDA GPU detected — transcription runs on the CPU.")

    if st.sidebar.button(
        "Clear model cache",
        icon=":material/delete_sweep:",
        help="Free memory used by loaded models",
    ):
        load_model_cached.clear()
        st.sidebar.success("Model cache cleared.", icon=":material/check_circle:")

    return {
        "model_name": model_name,
        "model_type": model_type,
        "language": language,
        "max_new_tokens": max_new_tokens,
        "chunked": chunked,
        "device": device,
    }


def run_transcriptions(uploads: list, settings: dict, retranscribe: bool) -> None:
    """Transcribe uploaded files and store records in session state."""
    device = settings["device"]
    try:
        processor, model, model_id, model_type = load_model_cached(
            settings["model_name"], device
        )
    except Exception as exc:  # noqa: BLE001 - surface any load failure in the UI
        st.error(
            f"Failed to load model '{settings['model_name']}': {exc}",
            icon=":material/error:",
        )
        return

    results = st.session_state.results
    audio_bytes = st.session_state.audio_bytes
    progress = st.progress(0.0, text="Starting transcription...")

    for i, uploaded_file in enumerate(uploads):
        file_id = ta.generate_file_id(uploaded_file.name, uploaded_file.size)
        result_key = f"{file_id}:{settings['model_name']}:{settings['language']}"
        progress.progress(
            i / len(uploads), text=f"Transcribing {uploaded_file.name}..."
        )

        if result_key in results and not retranscribe:
            st.info(
                f"Skipped **{uploaded_file.name}** — already transcribed with "
                "these settings. Check 'Re-transcribe processed files' to redo it.",
                icon=":material/skip_next:",
            )
            continue

        try:
            record = transcribe_upload(
                uploaded_file,
                processor,
                model,
                device,
                model_id,
                model_type,
                settings["language"],
                settings["max_new_tokens"],
                chunked=settings["chunked"],
            )
        except Exception as exc:  # noqa: BLE001 - keep the batch going
            st.error(
                f"Failed to transcribe {uploaded_file.name}: {exc}",
                icon=":material/error:",
            )
            continue

        results[result_key] = record
        audio_bytes[record["file_id"]] = uploaded_file.getvalue()

    progress.progress(1.0, text="Done")


def render_results() -> None:
    """Render the results table, per-file details, and download buttons."""
    records = list(st.session_state.results.values())
    if not records:
        return

    st.subheader("Results")
    df = results_dataframe(records)
    st.dataframe(df, width="stretch")

    for record in records:
        with st.expander(record["filename"], icon=":material/audio_file:"):
            audio = st.session_state.audio_bytes.get(record["file_id"])
            if audio:
                st.audio(audio)
            st.markdown(record["transcription_text"] or "*No speech detected.*")
            st.caption(
                f"Model: {record['model_id']} · "
                f"Time: {record['transcription_time_seconds']:.1f}s"
            )

    col_csv, col_json, col_clear = st.columns(3)
    col_csv.download_button(
        "Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="transcriptions.csv",
        mime="text/csv",
        icon=":material/download:",
    )
    col_json.download_button(
        "Download JSON",
        json.dumps(records, indent=2, ensure_ascii=False, default=str),
        file_name="transcriptions.json",
        mime="application/json",
        icon=":material/download:",
    )
    if col_clear.button("Clear results", icon=":material/delete:"):
        st.session_state.results = {}
        st.session_state.audio_bytes = {}
        st.rerun()


def main() -> None:
    """Run the Streamlit app."""
    st.set_page_config(
        page_title="Audio Transcription",
        page_icon=":material/mic:",
        layout="wide",
    )
    st.markdown(IPA_CSS, unsafe_allow_html=True)
    st.title("Audio transcription")
    st.caption(
        "Upload audio files and transcribe them with Whisper or Voxtral "
        "speech-to-text models. Results stay in this session — use the "
        "download buttons to save them."
    )

    st.session_state.setdefault("results", {})
    st.session_state.setdefault("audio_bytes", {})

    settings = render_sidebar()

    uploads = st.file_uploader(
        "Upload audio files",
        type=AUDIO_TYPES,
        accept_multiple_files=True,
        help="Supported formats: " + ", ".join(AUDIO_TYPES),
    )

    retranscribe = st.checkbox(
        "Re-transcribe processed files",
        value=False,
        help="Redo files already transcribed with the current settings.",
    )

    if st.button(
        "Transcribe",
        type="primary",
        icon=":material/graphic_eq:",
        disabled=not uploads,
        help=None if uploads else "Upload at least one audio file first.",
    ):
        run_transcriptions(uploads, settings, retranscribe)

    render_results()


if __name__ == "__main__":
    main()
