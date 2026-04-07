# Fremantle Transcriber for ComfyUI

Custom ComfyUI nodes for:

- Batch media loading
- Whisper transcription (CUDA supported)
- Segment grouping
- Multi-language Google translation
- Hallucination filtering
- Subtitle export (SRT / TXT / JSON)

---

## Installation

1. Go to your ComfyUI folder
2. Open `custom_nodes`
3. Clone this repository:

git clone https://github.com/azaradio26/comfyui-fremantle-transcriber.git

4. Install Python dependencies using the same Python environment used by ComfyUI

python -m pip install -r custom_nodes/comfyui-fremantle-transcriber/requirements.txt

5. Make sure ffmpeg and ffprobe are installed on your system

6. Restart ComfyUI

Linux / RunPod

Install FFmpeg system binaries:

apt-get update && apt-get install -y ffmpeg

---

## Requirements

- Python 3.10+
- CUDA compatible GPU (recommended)
- ffmpeg installed on system

