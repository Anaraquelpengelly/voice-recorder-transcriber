import pytest
from unittest.mock import patch, MagicMock
from utils.transcription_functions import transcribe_audio

def test_transcribe_audio_success():
    mock_response = "this is a transcription"
    mock_segments = ['segment1', 'segment2']
    filename = 'dummy.wav'

    with patch("utils.transcription_functions.MLXWhisperTranscriber") as mock_transcriber, \
        patch("utils.transcription_functions.logger") as mock_logger:
        mock_instance = mock_transcriber.return_value
        mock_instance.transcribe_file.return_value = (mock_response, mock_segments)
        #act:
        result = transcribe_audio(filename)
        #asser:
        assert result == mock_response
        mock_logger.info.assert_called_with(f'Processed transcription: {mock_response}')


def test_transcribe_audio_none_filename():
    #act:
    result = transcribe_audio(None)
    #assert:
    assert result is None


def test_transcribe_audio_exception():
    filename = "error.wav"

    with patch("utils.transcription_functions.MLXWhisperTranscriber") as MockTranscriber, \
         patch("utils.transcription_functions.logger") as mock_logger:
        mock_instance = MockTranscriber.return_value
        mock_instance.transcribe_file.side_effect = RuntimeError("Mocked error")

        # Act
        result = transcribe_audio(filename)

        # Assert
        assert result.startswith("Error in transcription: Mocked error")
        mock_logger.error.assert_called()


