import gradio as gr
import soundfile as sf
import xxhash
import os
import spaces
from loguru import logger
from whisper_turbo import MLXWhisperTranscriber



# Transcribe audio using whisper model
def transcribe_audio(filename:str)-> str:
    """Transcribe audio file using whisper model.
    Args:
        filename (str): Path to audio file.
    Returns:
        str: Transcription text or error message."""
    
    transcriber = MLXWhisperTranscriber(model_name="turbo-v3", api_enabled=False)
    if filename is None:
        return None
    
    try:
        response, segments = transcriber.transcribe_file(filename)
        logger.info(f"Processed transcription: {response}")
        return response
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        return f"Error in transcription: {str(e)}"
    

@spaces.GPU(duration=40, progress=gr.Progress(track_tqdm=True))
def response(audio:tuple, filename:str) -> str:
    """Handle audio input, save to file, and transcribe.
    Args:
        audio (tuple): Tuple of (sample_rate, numpy array) from gr.Audio.
        filename (str): Desired filename for saving audio.
    Returns:
        str: Transcription text or error message."""
    if not audio:
        logger.warning("No audio input received.")
        return "No audio input received."
    sample_rate, arr = audio

    logger.info(f"Received audio with sample rate: {sample_rate}, array shape: {arr.shape}")
    folder_name = "voice_recordings"
    os.makedirs(folder_name, exist_ok=True)
    file_name = f"{folder_name}/{xxhash.xxh32(bytes(audio[1])).hexdigest()}_{filename}.wav"

    sf.write(file_name, audio[1], audio[0], format="wav")

    transcription = transcribe_audio(file_name)
    logger.info(f"Transcription result: {transcription}")
    if transcription:
        if transcription.startswith("Error"):
            transcription = "Error in audio transcription."
        else:
            transcription = transcription
            transc_folder_name = "transcripts"
            os.makedirs(transc_folder_name, exist_ok=True)
            with open(f"{transc_folder_name}/{filename}_transcription.txt", "w") as f:
                f.write(transcription)
            logger.info(f"Transcription saved to transcripts/{filename}_transcription.txt")
    return transcription

## set the gradio theme
theme = gr.themes.Soft(
    primary_hue=gr.themes.Color(
        c100="#82000019",
        c200="#82000033",
        c300="#8200004c",
        c400="#82000066",
        c50="#8200007f",
        c500="#8200007f",
        c600="#82000099",
        c700="#820000b2",
        c800="#820000cc",
        c900="#820000e5",
        c950="#820000f2",
    ),
    secondary_hue="rose",
    neutral_hue="stone",
)

with gr.Blocks() as demo:
    title = gr.Markdown("# Voice Recorder and Transcriber")
    description = gr.Markdown("Record audio or upload a voice recording for transcription.")
    filename_input = gr.Textbox(label=["Filename", "upload"], max_lines=1)
    input_mode = gr.Radio(["Microphone", "Upload"], label="Select Input Type", value="Microphone")
    mic_input = gr.Audio(sources=["microphone"], type="numpy", label="Speak now")
    upload_input = gr.Audio(sources="upload", type="numpy", label="Or upload a voice recording")
    output_txt = gr.Textbox(label="Transcription")

    # Toggle visibility depending on input_mode selection
    def toggle_inputs(mode):
        return gr.update(visible=mode=="Microphone"), gr.update(visible=mode=="Upload")
    input_mode.change(toggle_inputs, input_mode, outputs=[mic_input, upload_input])

    # Bind the response function to each input, so it runs when audio changes
    mic_input.change(response, [mic_input, filename_input], [output_txt])
    upload_input.change(response, [upload_input, filename_input], [output_txt])


if __name__ == "__main__":
    demo.launch(debug=True)