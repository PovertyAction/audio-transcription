# %%
import time

import ipywidgets as widgets
import torch
from IPython.display import Audio, Markdown, display
from transformers import AutoProcessor, VoxtralForConditionalGeneration

# Constants
MAX_NEW_TOKENS = 500
LANGUAGE = "en"

# %%
# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# %%
# Model setup - runtime approximately 6 minutes for model download
# Documentation: https://huggingface.co/mistralai/Voxtral-Mini-3B-2507
model_id = "mistralai/Voxtral-Mini-3B-2507"

processor = AutoProcessor.from_pretrained(model_id)
model = VoxtralForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map=device
)

# %%
# Helper Functions


def create_audio_player(audio_path):
    """Create an interactive audio player widget for the given audio file."""
    play_button = widgets.Button(description="Play Audio")

    def on_play_clicked(b):
        display(Audio(audio_path))

    play_button.on_click(on_play_clicked)
    return play_button


def transcribe_audio(audio_path, processor, model, device, model_id):
    """Transcribe audio file and return decoded outputs."""
    start_time = time.time()
    inputs = processor.apply_transcrition_request(
        language=LANGUAGE, audio=audio_path, model_id=model_id
    )
    inputs = inputs.to(device, dtype=torch.bfloat16)
    outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Transcription completed in {elapsed_time:.2f} seconds")
    return processor.batch_decode(
        outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
    )


def display_transcription(decoded_outputs):
    """Display transcription results with consistent formatting."""
    print("\nTranscribed responses:")
    print("=" * 80)
    for decoded_output in decoded_outputs:
        display(Markdown(decoded_output))
        print("=" * 80)


def process_audio_file(audio_config, processor, model, device, model_id):
    """Process a single audio file: display info, player, and transcription."""
    # Display audio file information
    display(Markdown(f"### {audio_config['title']}"))
    display(Markdown(f"**Recording length:** {audio_config['length']}"))
    display(Markdown(f"**Original text:**\n> {audio_config['original']}"))

    # Create and display audio player
    player = create_audio_player(audio_config["path"])
    display(player)

    # Transcribe and display results
    decoded_outputs = transcribe_audio(
        audio_config["path"], processor, model, device, model_id
    )
    display_transcription(decoded_outputs)


# %%
# Audio Files Configuration
audio_files = [
    # {
    #     "title": "Audio file #1: Frying Pan",
    #     "path": "../audio/frying-pan.mp3",
    #     "length": "39 seconds",
    #     "original": "Okay. Wow, I think what I'm seeing again is what I've described before. Ok. Are you try, are you going to try to describe it again? Frying pan , the first thing I described. Is it exactly the first one you described? Yeah, but let me describe again in case you got it wrong, I don't know. The the background is a green, like a green rug, or green material, then there's a frying pan on it. The frying pan is, the inside is black, the body is red. Then inside that frying pan, there are some fried things that are like chicken or donut Ok Inside it, you'll still see a small bowl, it has yellow something inside it Ok, its the same thing. I've",
    # },
    # {
    #     "title": "Audio file #2: Mary Had a Little Lamb",
    #     "path": "../audio/mary-had-a-little-lamb.mp3",
    #     "length": "16 seconds",
    #     "original": "The first words I spoke in the original phonograph. Eh, a little piece of practical poetry. Mary had a little lamb, its fleece was white as snow, and everywhere that Mary went, the lamb was sure to go.",
    # },
    # {
    #     "title": "Audio file #3: Barcelona Weather",
    #     "path": "../audio/barcelona-weather.mp3",
    #     "length": "11 seconds",
    #     "original": "Yesterday it was thirty-five degrees in Barcelona, but today the temperature will go down to minus twenty degrees.",
    # },
    # {
    #     "title": "Audio file #4: Phone Call",
    #     "path": "../audio/phone-call.mp3",
    #     "length": "25 seconds",
    #     "original": "This is the full reflection. Ok. Because the water is kind of boiling. Ok. So this is a bird the bird is black and blue. It's facing, its face is facing us but its full physique is facing left. Okay, yes.",
    # },
    # {
    #     "title": "Audio file #5: Brunei Gallery",
    #     "path": "../audio/brunei-gallery.mp3",
    #     "length": "25 seconds",
    #     "original": "The Brunei gallery hosts a programme of changing contemporary and historical exhibitions from Asia Africa and the Middle East",
    # },
    {
        "title": "Audio file #6: Jollof in the Oven",
        "path": "../audio/jollof.mp3",
        "length": "3 seconds",
        "original": "I like cooking my jollof rice in the oven",
    },
    {
        "title": "Audio file #7: Edikaikong Soup",
        "path": "../audio/edikaikong.mp3",
        "length": "6 seconds",
        "original": "Edikaikong is a vegetable soup which originated with the Annang Ibibio and Efik people",
    },
]

# %%
# Process All Audio Files
# This replaces all the individual audio processing cells with a clean loop

for audio_config in audio_files:
    process_audio_file(audio_config, processor, model, device, model_id)

# %%
