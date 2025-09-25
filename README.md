# Voice-recorder-transcriber

This is a free, privacy preserving, homemade voice recorder-transcriber that I made using gradio and whisper-turbo. It has no limits to the length of the recording you can transcribe.

## Technical requirements
First, clone this repo.

```Bash
git clone git@github.com:Anaraquelpengelly/voice-recorder-transcriber.git
```

You need to download the ffmpeg library to use your machine's microphone. If you have a mac you can do it with homebrew:

```Bash
brew install ffmpeg
```

## How to run

You can run the app locally by first downloading uv, then, in the root folder of the repo do:

```bash
uv sync
```

Once all the dependencies are downloaded you run:

```bash
uv run main.py
```
Then go to the local url mentioned in the terminal. I would recommend running the app on Chrome, and you need to grant microphone permissions.
Your recordings and transcriptions will be saved in a `voice_recordings` and a `transcripts` folder respectively.

