import soundfile as sf
import xxhash
import os
from loguru import logger
from whisper_turbo import MLXWhisperTranscriber


# Transcribe audio using whisper model
def transcribe_audio(filename: str) -> str:
    """Transcribe audio file using whisper model.
    Args:
        filename (str): Path to audio file.
    Returns:
        str: Transcription text or error message.
    """

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


# Get transcript from audio file
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