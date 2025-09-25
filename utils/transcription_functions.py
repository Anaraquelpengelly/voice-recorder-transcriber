import soundfile as sf
import xxhash
import os
from loguru import logger
from whisper_turbo import MLXWhisperTranscriber
import asyncio

# Transcribe audio using whisper model (async version)
async def transcribe_audio(filename: str) -> str:
    transcriber = MLXWhisperTranscriber(model_name="turbo-v3", api_enabled=False)
    if filename is None:
        return None
    try:
        response, segments = await asyncio.to_thread(transcriber.transcribe_file, filename)
        logger.info(f"Processed transcription: {response}")
        return response
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        return f"Error in transcription: {str(e)}"

# Get transcript from audio file (async version)
async def response(audio: tuple, filename: str) -> str:
    if not audio:
        logger.warning("No audio input received.")
        return "No audio input received."
    sample_rate, arr = audio
    logger.info(f"Received audio with sample rate: {sample_rate}, array shape: {arr.shape}")
    folder_name = "voice_recordings"
    os.makedirs(folder_name, exist_ok=True)
    file_name = f"{folder_name}/{xxhash.xxh32(bytes(audio[1])).hexdigest()}_{filename}.wav"
    await asyncio.to_thread(sf.write, file_name, audio[1], audio[0], format="wav")
    transcription = await transcribe_audio(file_name)
    logger.info(f"Transcription result: {transcription}")
    if transcription:
        if transcription.startswith("Error"):
            transcription = "Error in audio transcription."
        else:
            transc_folder_name = "transcripts"
            os.makedirs(transc_folder_name, exist_ok=True)
            await asyncio.to_thread(
                lambda: open(f"{transc_folder_name}/{filename}_transcription.txt", "w").write(transcription)
            )
            logger.info(f"Transcription saved to transcripts/{filename}_transcription.txt")
    return transcription

