# Speech to Text for Wayland

A lightweight local desktop speech-to-text tool for Linux Wayland sessions that records from your microphone, sends audio to Groq's hosted Whisper transcription API, and pastes the result into the active application.

It supports two workflows:

- `interactive`: start recording immediately, then press Enter to stop and transcribe
- `toggle`: use a keyboard shortcut to start recording on the first press and stop/transcribe on the second

## Features

- Records microphone input with `sounddevice`
- Transcribes speech using Groq-hosted Whisper models
- Pastes text into the focused app with `wl-copy` and `wtype`
- Falls back to clipboard copy if automatic typing is unavailable
- Shows desktop notifications during start, stop, and transcription
- Includes a `setup.sh` helper to create a virtual environment and install dependencies

## Requirements

- Linux desktop running Wayland
- Python 3
- A Groq API key
- System packages:
  - `libportaudio2`
  - `wl-clipboard`
  - `wtype`
  - `libnotify-bin`

On Ubuntu/Debian, install them with:

```bash
sudo apt-get install -y libportaudio2 wl-clipboard wtype libnotify-bin
```

## Python Dependencies

Defined in `requirements.txt`:

- `groq`
- `numpy`
- `python-dotenv`
- `sounddevice`

## Setup

1. Clone the project and move into it:

```bash
git clone <your-repo-url>
cd Speech_to_text
```

2. Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

3. Run the setup script:

```bash
bash setup.sh
```

This script will:

- create `.venv`
- install Python packages from `requirements.txt`
- try to install the required Ubuntu packages
- launch the app in interactive mode

## Usage

### Interactive Mode

Start recording immediately and press Enter to stop:

```bash
bash setup.sh
```

Or, after setup:

```bash
source .venv/bin/activate
python main.py interactive
```

### Toggle Mode

Start recording with one shortcut press, stop/transcribe with the next:

```bash
bash setup.sh toggle
```

You can use this command for a custom keyboard shortcut on Ubuntu:

```bash
bash -lc 'bash "/path/to/Speech_to_text/setup.sh" toggle'
```

## CLI Options

`main.py` supports:

```bash
python main.py [interactive|toggle] \
  --model whisper-large-v3-turbo \
  --language en \
  --prompt "optional vocabulary hint" \
  --sample-rate 16000 \
  --max-duration 60
```

## How It Works

1. Audio is recorded from the default microphone.
2. The recording is converted to WAV in memory.
3. The audio is sent to Groq's transcription API.
4. The transcribed text is cleaned and pasted into the active Wayland application.
5. If `wtype` is unavailable, the text is copied to the clipboard instead.

## Project Structure

- `main.py`: application entry point and speech-to-text logic
- `setup.sh`: environment setup and launch helper
- `requirements.txt`: Python dependencies

## Notes

- This project is designed for Wayland, not X11.
- `toggle` mode stores temporary state in `/tmp`.
- The default transcription model is `whisper-large-v3-turbo`.
- If `wl-copy` is missing, the app will print the text instead of pasting it automatically.

## License

Add your preferred license here before publishing the repository.
