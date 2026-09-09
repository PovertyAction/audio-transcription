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

# Languages supported by the Cohere Transcribe model (no auto-detect)
COHERE_LANGUAGES = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "it": "Italian",
    "es": "Spanish",
    "pt": "Portuguese",
    "el": "Greek",
    "nl": "Dutch",
    "pl": "Polish",
    "ar": "Arabic",
    "vi": "Vietnamese",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
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


@st.cache_resource(show_spinner="Loading diarization pipeline...")
def load_diarization_cached(device: str, hf_token: str | None, model_path: str | None):
    """Load and cache the pyannote diarization pipeline."""
    return ta.load_diarization_pipeline(
        device, hf_token=hf_token or None, model_path=model_path or None
    )


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
    diarization_pipeline=None,
    num_speakers: int | None = None,
    refine: bool = False,
    ollama_model: str | None = None,
    ollama_url: str | None = None,
) -> dict:
    """Transcribe one uploaded file and return a transcription record."""
    data = uploaded_file.getvalue()
    suffix = Path(uploaded_file.name).suffix or ".wav"

    # Write to a temp file and close it before transcribing: transcribe_audio
    # needs a real path, and Windows can't reopen a file that is still open.
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    refined = False
    try:
        if diarization_pipeline is not None:
            decoded_outputs, elapsed_time, started_at, refined = (
                ta.transcribe_with_diarization(
                    tmp_path,
                    diarization_pipeline,
                    processor,
                    model,
                    device,
                    model_id,
                    model_type,
                    language,
                    max_new_tokens,
                    chunked=chunked,
                    num_speakers=num_speakers,
                    refine=refine,
                    ollama_model=ollama_model or ta.OLLAMA_DEFAULT_MODEL,
                    ollama_url=ollama_url or ta.OLLAMA_DEFAULT_URL,
                )
            )
        else:
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
    record_model_id = model_id
    if diarization_pipeline is not None:
        record_model_id = f"{model_id} + {ta.DIARIZATION_MODEL_ID}"
        if refined:
            record_model_id += f" + ollama:{ollama_model or ta.OLLAMA_DEFAULT_MODEL}"
    file_id = ta.generate_file_id(uploaded_file.name, len(data))
    return ta.create_transcription_record(
        uploaded_file.name,
        file_id,
        len(data),
        elapsed_time,
        text,
        record_model_id,
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
        "8 languages and may be more accurate for multilingual audio. "
        "Cohere supports 14 languages and handles long recordings "
        "automatically.",
    )
    model_type = ta.AVAILABLE_MODELS[model_name]["type"]

    if model_type == "voxtral":
        languages = VOXTRAL_LANGUAGES
        default_language = "en"
    elif model_type == "cohere":
        languages = COHERE_LANGUAGES
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

    if model_type == "cohere":
        # Cohere manages output length and long-form chunking internally;
        # max_new_tokens is ignored downstream.
        max_new_tokens = 400
        st.sidebar.caption(
            "Cohere manages output length and long-form chunking automatically."
        )
    else:
        token_cap = (
            WHISPER_MAX_TOKENS if model_type == "whisper" else VOXTRAL_MAX_TOKENS
        )
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

    num_speakers = None
    hf_token = ""
    diarization_path = ""
    refine = False
    ollama_model = ta.OLLAMA_DEFAULT_MODEL
    ollama_url = ta.OLLAMA_DEFAULT_URL
    diarize = st.sidebar.toggle(
        "Multiple speakers (diarization)",
        value=False,
        disabled=not ta.PYANNOTE_AVAILABLE,
        help="Label who said what using pyannote speaker diarization. "
        "Each speaker turn is transcribed with the selected model above.",
    )
    if not ta.PYANNOTE_AVAILABLE:
        st.sidebar.caption(
            "Speaker diarization requires pyannote.audio — run `uv sync` to install it."
        )
    elif diarize:
        num_speakers_input = st.sidebar.number_input(
            "Number of speakers (0 = detect automatically)",
            min_value=0,
            max_value=20,
            value=0,
            step=1,
            help="Set the exact speaker count if you know it — this improves "
            "diarization accuracy.",
        )
        num_speakers = int(num_speakers_input) or None
        refine = st.sidebar.toggle(
            "Refine with local LLM (Ollama)",
            value=False,
            help="Rewrite the diarized transcript with a local LLM so the "
            "wording matches a full-quality transcription. Requires Ollama "
            "(ollama.com) running locally with the model pulled. Falls back "
            "to the unrefined transcript if Ollama is unavailable.",
        )
        if refine:
            ollama_model = st.sidebar.text_input(
                "Ollama model",
                value=ta.OLLAMA_DEFAULT_MODEL,
                help="Any instruct model served by your local Ollama — "
                "install with 'ollama pull <model>'.",
            ).strip()
            ollama_url = st.sidebar.text_input(
                "Ollama URL",
                value=ta.OLLAMA_DEFAULT_URL,
                help="Base URL of the local Ollama server.",
            ).strip()
        diarization_path = st.sidebar.text_input(
            "Local model directory (offline use)",
            value="",
            help="Path to a local download of "
            f"{ta.DIARIZATION_MODEL_ID} — no token or internet needed. Leave "
            "blank to download from the HF Hub (requires accepting the "
            "model's terms and a token).",
        ).strip()
        if not diarization_path:
            hf_token = st.sidebar.text_input(
                "Hugging Face token",
                value=os.environ.get("HF_TOKEN", ""),
                type="password",
                help="Needed once to download the gated diarization model. "
                "Defaults to the HF_TOKEN environment variable; a cached "
                "'huggingface-cli login' also works.",
            ).strip()

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
        "diarize": diarize,
        "num_speakers": num_speakers,
        "hf_token": hf_token,
        "diarization_path": diarization_path,
        "refine": refine,
        "ollama_model": ollama_model,
        "ollama_url": ollama_url,
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

    diarization_pipeline = None
    if settings["diarize"]:
        try:
            diarization_pipeline = load_diarization_cached(
                device, settings["hf_token"], settings["diarization_path"]
            )
        except Exception as exc:  # noqa: BLE001 - surface any load failure in the UI
            st.error(
                f"Failed to load diarization pipeline: {exc}", icon=":material/error:"
            )
            return

    results = st.session_state.results
    audio_bytes = st.session_state.audio_bytes
    progress = st.progress(0.0, text="Starting transcription...")

    for i, uploaded_file in enumerate(uploads):
        file_id = ta.generate_file_id(uploaded_file.name, uploaded_file.size)
        if settings["diarize"]:
            diar_suffix = "diar-refined" if settings["refine"] else "diar"
        else:
            diar_suffix = "plain"
        result_key = (
            f"{file_id}:{settings['model_name']}:{settings['language']}:{diar_suffix}"
        )
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
                diarization_pipeline=diarization_pipeline,
                num_speakers=settings["num_speakers"],
                refine=settings["refine"],
                ollama_model=settings["ollama_model"],
                ollama_url=settings["ollama_url"],
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
            # Diarized transcripts are one line of "SPEAKER_XX: ..." turns
            # separated by "; " — break turns onto their own lines for
            # readability ("  \n" is a markdown line break).
            st.markdown(
                record["transcription_text"].replace("; SPEAKER_", ";  \nSPEAKER_")
                or "*No speech detected.*"
            )
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
        "Upload audio files and transcribe them with Whisper, Voxtral, or "
        "Cohere speech-to-text models — optionally with speaker labels. "
        "Results stay in this session — use the download buttons to save them."
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
