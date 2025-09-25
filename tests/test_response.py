from utils.transcription_functions import response
from unittest.mock import patch, MagicMock
import numpy as np

def test_response_with_valid_audio(tmp_path):
    # Simulate a valid audio input tuple (sample_rate, numpy array)
    sample_rate = 16000
    arr = np.ones(1000, dtype=np.float32)
    audio = (sample_rate, arr)
    filename = "testfile"

    # Mocks
    with patch("utils.transcription_functions.os.makedirs") as makedirs_mock, \
         patch("utils.transcription_functions.sf.write") as sf_write_mock, \
         patch("utils.transcription_functions.xxhash.xxh32") as xxh32_mock, \
         patch("utils.transcription_functions.transcribe_audio") as transcribe_mock, \
         patch("utils.transcription_functions.logger") as logger_mock, \
         patch("builtins.open", new_callable=MagicMock):

        fake_hash = MagicMock()
        fake_hash.hexdigest.return_value = "abc123"
        xxh32_mock.return_value = fake_hash
        transcribe_mock.return_value = "transcription text"

        result = response(audio, filename)
        assert result == "transcription text"
        makedirs_mock.assert_any_call("voice_recordings", exist_ok=True)
        makedirs_mock.assert_any_call("transcripts", exist_ok=True)
        sf_write_mock.assert_called_once()
        xxh32_mock.assert_called_once()
        transcribe_mock.assert_called_once()
        logger_mock.info.assert_any_call("Transcription result: transcription text")

def test_response_no_audio():
    with patch("utils.transcription_functions.logger") as logger_mock:
        result = response(None, "testfile")
        assert result == "No audio input received."
        logger_mock.warning.assert_called_once_with("No audio input received.")
