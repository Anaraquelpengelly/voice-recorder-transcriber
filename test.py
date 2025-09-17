import gradio as gr
import soundfile as sf
import xxhash
import os
import spaces
from loguru import logger
from whisper_turbo import MLXWhisperTranscriber


import torch
print(torch.backends.mps.is_available())



# Transcribe audio using whisper model
def transcribe_audio(filename:str)-> str:
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
def response(audio:tuple, filename:str):
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
    description = gr.Markdown("Record and transcribe here. Enter a filename and click the microphone to start recording.")
    filename_input = gr.Textbox(label="Filename", max_lines=1)
    audio_input = gr.Audio(sources=["microphone"], type="numpy", label="Speak now")
    output_txt = gr.Textbox(label="Transcription")

    # Automatically call response when audio input changes
    audio_input.change(fn=response, inputs=[audio_input, filename_input], outputs=[output_txt])



if __name__ == "__main__":
    demo.launch(debug=True)