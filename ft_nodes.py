from comfy.utils import ProgressBar

import os
import json
import time
import re
import math
import threading
import datetime
import traceback
import subprocess
import tempfile
import shutil
from typing import Dict, Any, List

import torch
import whisper
import numpy as np
import srt
from deep_translator import GoogleTranslator

PLUGIN_VERSION = "v1.0.0"

# ----------------------------
# Helpers / Constants
# ----------------------------

VALID_EXTS = (
    ".mp3", ".wav", ".flac", ".m4a", ".ogg",
    ".mp4", ".mkv", ".avi", ".mov", ".ts",
    ".aiff", ".aif", ".aac", ".wma",
    ".mxf",
)

VIDEO_EXTS = {".mp4", ".mov", ".mxf", ".mkv", ".avi", ".ts"}

ISO_TO_NAME = {
    "en": "English", "pt": "Portuguese", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "nl": "Dutch", "ru": "Russian", "zh": "Chinese", "ja": "Japanese",
    "ar": "Arabic", "he": "Hebrew"
}

LANG_MAP = {
    "Auto Detect": None,
    "Portuguese": "pt",
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "Hebrew": "he",    # Whisper uses 'he'
    "Arabic": "ar",
    "German": "de",
    "Dutch": "nl",
    "Italian": "it",
    "Chinese": "zh",
}

TRANSLATE_MAP = {
    "English": "en",
    "Portuguese": "pt",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Dutch": "nl",
    "Arabic": "ar",
    "Hebrew": "iw",     # Google uses 'iw'
    "Chinese": "zh-CN",
}

def ensure_ffmpeg_on_path():
    # macOS app bundles can have a limited PATH. Safe no-op elsewhere.
    if os.name != "posix":
        return
    cur = os.environ.get("PATH", "")
    extra = ["/opt/homebrew/bin", "/usr/local/bin"]
    parts = cur.split(":") if cur else []
    for p in reversed(extra):
        if p not in parts:
            parts.insert(0, p)
    os.environ["PATH"] = ":".join(parts)

ensure_ffmpeg_on_path()

def require_ffmpeg_tools():
    missing = []

    if shutil.which("ffmpeg") is None:
        missing.append("ffmpeg")
    if shutil.which("ffprobe") is None:
        missing.append("ffprobe")

    if missing:
        missing_txt = ", ".join(missing)
        raise RuntimeError(
            "Fremantle Transcriber: required system tool(s) not found: "
            f"{missing_txt}. "
            "Please install ffmpeg on the system and ensure both ffmpeg and ffprobe "
            "are available in PATH. "
            "Note: pip install -r requirements.txt does not install ffmpeg."
        )

def safe_json_load(s: str, fallback):
    try:
        return json.loads(s)
    except Exception:
        return fallback

def json_dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)

def extract_audio_to_wav(input_path: str) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()

    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        tmp.name
    ]

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({r.returncode}): {r.stderr[-500:]}")
    return tmp.name

def ffprobe_duration_seconds(path: str) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
        return float(out) if out else 0.0
    except Exception:
        return 0.0

def ffprobe_has_audio(path: str) -> bool:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=codec_name",
        "-of", "csv=p=0",
        path
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
        return bool(out)
    except Exception:
        # If ffprobe fails, do not block the pipeline.
        return True

def ffmpeg_is_silent(path: str, threshold_db: float = -58.0) -> bool:
    cmd = ["ffmpeg", "-i", path, "-af", "volumedetect", "-vn", "-sn", "-dn", "-f", "null", "-"]
    try:
        stderr = subprocess.run(cmd, capture_output=True, text=True).stderr
        m = re.search(r"max_volume: ([\-\d\.]+) dB", stderr)
        if m and float(m.group(1)) < threshold_db:
            return True
        return False
    except Exception:
        return False

def is_hallucination(text: str, duration: float, last_text: str) -> bool:
    clean = (text or "").strip().lower()
    if not clean or duration <= 0:
        return True
    if duration < 0.4:
        return True
    if len(clean) > 15 and len(set(clean.replace(" ", ""))) < 5:
        return True
    words = clean.split()
    if len(words) > 3:
        rep = max(words.count(w) for w in set(words)) / len(words)
        if rep > 0.6:
            return True
    if clean == (last_text or ""):
        return True
    return False

def group_segments(segments: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    blocks = []
    t_texts = []
    t_start = None

    for i, seg in enumerate(segments):
        if t_start is None:
            t_start = float(seg.get("start", 0.0))

        t_texts.append((seg.get("text") or "").strip())

        if (limit > 0 and len(t_texts) >= limit) or limit == 0 or i == len(segments) - 1:
            blocks.append({
                "start": t_start,
                "end": float(seg.get("end", 0.0)),
                "text": " ".join(t_texts).strip()
            })
            t_texts = []
            t_start = None

    return blocks

def split_for_srt(text: str, start: float, end: float, max_w: int = 42):
    import textwrap
    lines = textwrap.wrap(text, width=max_w, break_long_words=False)

    if len(lines) <= 2:
        return [(
            "\n".join(lines),
            datetime.timedelta(seconds=start),
            datetime.timedelta(seconds=end)
        )]

    mid = len(lines) // 2
    p1 = "\n".join(lines[:mid])
    p2 = "\n".join(lines[mid:])

    t_split = start + (len(p1) / max(1, len(text))) * (end - start)
    return [
        (p1, datetime.timedelta(seconds=start), datetime.timedelta(seconds=t_split)),
        (p2, datetime.timedelta(seconds=t_split), datetime.timedelta(seconds=end)),
    ]

_VERSION_RE = re.compile(r"^(?P<stem>.+?)_v(?P<v>\d+)$", re.IGNORECASE)

def safe_path(path: str) -> str:
    """
    Versioning rules:
    - If nothing exists -> use path as-is.
    - If 'base.ext' exists OR any 'base_vN.ext' exists -> use _v{max+1}
    - Base file counts as version 1 => first versioned file is _v2 (never _v1).
    """
    folder = os.path.dirname(path) or "."
    base_name = os.path.basename(path)
    stem, ext = os.path.splitext(base_name)
    ext_l = ext.lower()

    if not os.path.isdir(folder):
        return path

    max_v = 0

    try:
        for fn in os.listdir(folder):
            s, e = os.path.splitext(fn)
            if e.lower() != ext_l:
                continue

            if s == stem:
                max_v = max(max_v, 1)  # base exists => v1
                continue

            m = _VERSION_RE.match(s)
            if m and m.group("stem") == stem:
                try:
                    v = int(m.group("v"))
                    max_v = max(max_v, v)
                except Exception:
                    pass
    except Exception:
        # If listing fails for any reason, be conservative and return original path.
        return path

    if max_v == 0:
        return path

    next_v = max(2, max_v + 1)
    return os.path.join(folder, f"{stem}_v{next_v}{ext}")

# Simple in-memory model cache
_MODEL_CACHE: Dict[str, Any] = {}

def load_whisper_model(model_choice: str, custom_path: str, device: str):
    key = f"{model_choice}|{custom_path}|{device}"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    if model_choice == "custom .pt path":
        if not custom_path or not os.path.exists(custom_path):
            raise RuntimeError("Invalid custom model path or file does not exist.")
        model = whisper.load_model(custom_path, device=device)
    else:
        model = whisper.load_model(model_choice, device=device)

    _MODEL_CACHE[key] = model
    return model

def build_filtered_blocks(blocks, texts, apply_filter: bool):
    if not apply_filter:
        out = []
        for b, t in zip(blocks, texts):
            out.append({
                "start": float(b.get("start", 0.0)),
                "end": float(b.get("end", 0.0)),
                "text": t or ""
            })
        return out

    keep = []
    last_v = ""
    for b, t in zip(blocks, texts):
        start = float(b.get("start", 0.0))
        end = float(b.get("end", 0.0))
        dur = end - start
        if is_hallucination(t, dur, last_v):
            continue
        keep.append({"start": start, "end": end, "text": t})
        last_v = (t or "").strip().lower()
    return keep

# ----------------------------
# Node: Info
# ----------------------------
class FT_Info:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "show": ("BOOLEAN", {"default": True}),
                "notes": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": (
                            "Project Notes (editable):\n"
                            "- \n"
                            "- \n"
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "run"
    CATEGORY = "Fremantle/Transcriber"

    def run(self, show: bool, notes: str):
        if not show:
            return ("",)

        manual = (
            f"Fremantle Transcriber for ComfyUI\n"
            f"Version: {PLUGIN_VERSION}\n"
            f"=================================\n\n"
            "STANDARD WORKFLOW\n"
            "-----------------\n"
            "1) FT • Load Media Batch\n"
            "2) FT • Whisper Model (cached)\n"
            "3) FT • Transcribe Batch\n"
            "4) FT • Group Segments\n"
            "5) FT • Translate (Google) OR FT • Translate Multi (Google)\n"
            "6) FT • Hallucination Filter\n"
            "7) FT • Export (SRT / TXT / JSON) OR FT • Export Multi\n\n"
            "KEY SETTINGS\n"
            "------------\n"
            "- temperature: 0.0 recommended (deterministic)\n"
            "- word_timestamps: True = slower, word-level timing\n"
            "- batch_size (Translate): 10–20 recommended\n"
            "- Hebrew: Whisper uses 'he', Google uses 'iw'\n\n"
            "TROUBLESHOOTING\n"
            "---------------\n"
            "- If CUDA fails: switch device to CPU or use a smaller model.\n"
            "- If translation fails: lower batch_size.\n"
            "- If nodes appear missing: restart ComfyUI.\n\n"
        )

        notes_clean = (notes or "").strip()
        if notes_clean:
            combined = manual + "PROJECT NOTES\n-------------\n" + notes_clean + "\n"
        else:
            combined = manual + "PROJECT NOTES\n-------------\n(Empty)\n"

        return (combined,)

# ----------------------------
# Node 1: Load media batch
# ----------------------------
class FT_LoadMediaBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_path": ("STRING", {"default": "", "multiline": False}),
                "recursive": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("files",)
    FUNCTION = "run"
    CATEGORY = "Fremantle/Transcriber"

    def run(self, input_path: str, recursive: bool):
        p = (input_path or "").strip().strip('"').strip("'")
        if not p:
            raise RuntimeError("FT_LoadMediaBatch: Empty input_path.")

        files = []

        if os.path.isfile(p):
            if p.lower().endswith(VALID_EXTS):
                files.append(p)
        elif os.path.isdir(p):
            if recursive:
                for root, _, names in os.walk(p):
                    for n in names:
                        fp = os.path.join(root, n)
                        if fp.lower().endswith(VALID_EXTS):
                            files.append(fp)
            else:
                for n in os.listdir(p):
                    fp = os.path.join(p, n)
                    if os.path.isfile(fp) and fp.lower().endswith(VALID_EXTS):
                        files.append(fp)
        else:
            raise RuntimeError(f"FT_LoadMediaBatch: Path does not exist: {p}")

        files.sort()

        if not files:
            raise RuntimeError(
                "FT_LoadMediaBatch: No valid media files found.\n"
                f"input_path: {p}\n"
                f"valid_extensions: {', '.join(VALID_EXTS)}\n"
                f"recursive: {bool(recursive)}"
            )

        return (json_dumps({"files": files}),)

# ----------------------------
# Node 2: Whisper model loader (cached)
# ----------------------------
class FT_WhisperModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_choice": (["large-v3", "turbo", "medium", "small", "base", "tiny", "custom .pt path"],),
                "device": (["cpu", "cuda"],),
                "custom_pt_path": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_handle",)
    FUNCTION = "run"
    CATEGORY = "Fremantle/Transcriber"

    def run(self, model_choice: str, device: str, custom_pt_path: str):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        _ = load_whisper_model(model_choice, custom_pt_path, device)

        return (json_dumps({
            "model_choice": model_choice,
            "custom_pt_path": custom_pt_path,
            "device": device
        }),)

# ----------------------------
# Node 3: Transcribe batch
# ----------------------------
class FT_TranscribeBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "files": ("STRING", {"forceInput": True}),
                "model_handle": ("STRING", {"forceInput": True}),
                "extract_audio_for_video": ("BOOLEAN", {"default": True}),
                "source_language": (list(LANG_MAP.keys()),),
                "word_timestamps": ("BOOLEAN", {"default": True}),
                "temperature": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.1}),
                "no_speech_threshold": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05}),
                "logprob_threshold": ("FLOAT", {"default": -1.0, "min": -5.0, "max": 0.0, "step": 0.1}),
                "compression_ratio_threshold": ("FLOAT", {"default": 2.4, "min": 0.0, "max": 4.0, "step": 0.1}),
                "skip_if_no_audio": ("BOOLEAN", {"default": True}),
                "skip_if_silent": ("BOOLEAN", {"default": True}),
                "silent_threshold_db": ("FLOAT", {"default": -58.0, "min": -120.0, "max": 0.0}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("transcriptions",)
    FUNCTION = "run"
    CATEGORY = "Fremantle/Transcriber"

    def run(
        self,
        files: str,
        model_handle: str,
        extract_audio_for_video: bool,
        source_language: str,
        word_timestamps: bool,
        temperature: float,
        no_speech_threshold: float,
        logprob_threshold: float,
        compression_ratio_threshold: float,
        skip_if_no_audio: bool,
        skip_if_silent: bool,
        silent_threshold_db: float,
    ):
        require_ffmpeg_tools()
        
        STEP_SEC = 2.0  # smooth progress update interval

        files_obj = safe_json_load(files, {})
        file_list = files_obj.get("files", []) if isinstance(files_obj, dict) else []

        if not file_list:
            raise RuntimeError(
                "FT_TranscribeBatch: No files to process.\n"
                "Check FT_LoadMediaBatch input_path (path missing, empty, or no valid media extensions)."
            )

        missing = [p for p in file_list if not os.path.exists(p)]
        if missing:
            show = "\n".join(missing[:8])
            more = "" if len(missing) <= 8 else f"\n... (+{len(missing) - 8} more)"
            raise RuntimeError(
                "FT_TranscribeBatch: Some input files do not exist on disk:\n"
                f"{show}{more}\n"
                "Fix the input_path or re-run FT_LoadMediaBatch."
            )

        mh = safe_json_load(model_handle, {})
        model_choice = mh.get("model_choice", "large-v3")
        custom_pt_path = mh.get("custom_pt_path", "")
        device = mh.get("device", "cpu")

        model = load_whisper_model(model_choice, custom_pt_path, device)
        forced = LANG_MAP.get(source_language, None)

        # Progress estimation
        per_file_steps = []
        for p in file_list:
            dur = ffprobe_duration_seconds(p)
            steps = max(1, int(math.ceil(dur / STEP_SEC))) if dur > 0 else 1
            per_file_steps.append(steps)

        total_steps = sum(per_file_steps) if per_file_steps else 1
        pbar = ProgressBar(max(1, total_steps))

        results = []

        def smooth_updater(target_steps: int, stop_event: threading.Event, done_counter: dict):
            limit = max(0, target_steps - 1)
            while not stop_event.is_set() and done_counter["done"] < limit:
                pbar.update(1)
                done_counter["done"] += 1
                time.sleep(STEP_SEC)

        for path, file_steps in zip(file_list, per_file_steps):
            wav_path = None
            stop_event = threading.Event()
            done_counter = {"done": 0}
            t = None

            try:
                if skip_if_no_audio and not ffprobe_has_audio(path):
                    results.append({"path": path, "status": "EMPTY", "reason": "no_audio"})
                    pbar.update(file_steps)
                    continue

                if skip_if_silent and ffmpeg_is_silent(path, threshold_db=float(silent_threshold_db)):
                    results.append({"path": path, "status": "EMPTY", "reason": "silent"})
                    pbar.update(file_steps)
                    continue

                if file_steps > 1:
                    t = threading.Thread(
                        target=smooth_updater,
                        args=(file_steps, stop_event, done_counter),
                        daemon=True
                    )
                    t.start()

                ext = os.path.splitext(path)[1].lower()

                # For video/container formats (incl. MXF), extract WAV for reliability.
                if extract_audio_for_video and ext in VIDEO_EXTS:
                    wav_path = extract_audio_to_wav(path)
                    audio_src = wav_path
                else:
                    audio_src = path

                audio = whisper.load_audio(audio_src).astype(np.float32)

                detected = None
                if not forced:
                    audio_trimmed = whisper.pad_or_trim(audio)
                    mel = whisper.log_mel_spectrogram(
                        audio_trimmed,
                        n_mels=model.dims.n_mels
                    ).to(model.device)
                    _, probs = model.detect_language(mel)
                    detected = max(probs, key=probs.get)

                use_fp16 = (device == "cuda")

                res = model.transcribe(
                    audio,
                    language=forced,
                    fp16=use_fp16,
                    word_timestamps=bool(word_timestamps),
                    temperature=float(temperature),
                    verbose=False,
                    condition_on_previous_text=False,
                    no_speech_threshold=float(no_speech_threshold),
                    logprob_threshold=float(logprob_threshold),
                    compression_ratio_threshold=float(compression_ratio_threshold),
                )

                results.append({
                    "path": path,
                    "status": "OK",
                    "detected_language": detected,
                    "segments": res.get("segments", []),
                })

            except Exception as e:
                results.append({
                    "path": path,
                    "status": "ERROR",
                    "error": str(e),
                    "trace": traceback.format_exc()
                })

            finally:
                if t is not None:
                    stop_event.set()
                    t.join(timeout=1.0)

                remaining = max(0, file_steps - done_counter["done"])
                if remaining:
                    pbar.update(remaining)

                if wav_path and os.path.exists(wav_path):
                    try:
                        os.remove(wav_path)
                    except Exception:
                        pass

        return (json_dumps({"items": results}),)

# ----------------------------
# Node 4: Group segments
# ----------------------------
class FT_GroupSegments:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "transcriptions": ("STRING", {"forceInput": True}),
                "group_mode": (["Movie Style", "Group 10", "Group 20", "Custom"],),
                "custom_limit": ("INT", {"default": 10, "min": 0, "max": 200}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("grouped",)
    FUNCTION = "run"
    CATEGORY = "Fremantle/Transcriber"

    def run(self, transcriptions: str, group_mode: str, custom_limit: int):
        obj = safe_json_load(transcriptions, {})
        items = obj.get("items", []) if isinstance(obj, dict) else []

        if group_mode == "Movie Style":
            limit = 0
        elif group_mode == "Group 10":
            limit = 10
        elif group_mode == "Group 20":
            limit = 20
        else:
            limit = int(custom_limit)

        out = []
        for it in items:
            if it.get("status") != "OK":
                out.append(it)
                continue

            segs = it.get("segments", [])
            blocks = group_segments(segs, limit)
            out.append({
                **it,
                "group_limit": limit,
                "blocks": blocks,
            })

        return (json_dumps({"items": out}),)

# ----------------------------
# Node 5: Translate Google (single)
# ----------------------------
class FT_TranslateGoogle:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "grouped": ("STRING", {"forceInput": True}),
                "target_language": (list(TRANSLATE_MAP.keys()),),
                "batch_size": ("INT", {"default": 10, "min": 1, "max": 50}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("translated",)
    FUNCTION = "run"
    CATEGORY = "Fremantle/Transcriber"

    def run(self, grouped: str, target_language: str, batch_size: int):
        obj = safe_json_load(grouped, {})
        items = obj.get("items", []) if isinstance(obj, dict) else []

        tgt_code = TRANSLATE_MAP.get(target_language, "en")
        translator = GoogleTranslator(source="auto", target=tgt_code)

        # Progress counts batches for OK items.
        total_batches = 0
        for it in items:
            if it.get("status") == "OK":
                blocks = it.get("blocks", []) or []
                total_batches += len(range(0, len(blocks), int(batch_size))) or 1

        pbar = ProgressBar(max(1, total_batches))

        out = []
        for it in items:
            if it.get("status") != "OK":
                out.append(it)
                continue

            blocks = it.get("blocks", []) or []
            texts = [b.get("text", "") for b in blocks]

            translated = []
            total = len(texts)
            for i in range(0, total, int(batch_size)):
                chunk = texts[i:i + int(batch_size)]
                try:
                    got = translator.translate("\n".join(chunk))
                    got_lines = (got or "").split("\n") if got else []
                    if len(got_lines) == len(chunk):
                        translated.extend(got_lines)
                    else:
                        for x in chunk:
                            translated.append(translator.translate(x) if x.strip() else x)
                except Exception:
                    for x in chunk:
                        try:
                            translated.append(translator.translate(x) if x.strip() else x)
                        except Exception:
                            translated.append(x)

                time.sleep(0.1)
                pbar.update(1)

            out.append({
                **it,
                "translate": True,
                "target_language": target_language,
                "target_code": tgt_code,
                "translated_texts": translated
            })

        return (json_dumps({"items": out}),)

# ----------------------------
# Node 6: Translate Multi Google
# ----------------------------
class FT_TranslateMultiGoogle:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "grouped": ("STRING", {"forceInput": True}),
                "batch_size": ("INT", {"default": 10, "min": 1, "max": 50}),

                # Checkboxes (UI labels)
                "English": ("BOOLEAN", {"default": True}),
                "Portuguese": ("BOOLEAN", {"default": False}),
                "Spanish": ("BOOLEAN", {"default": False}),
                "French": ("BOOLEAN", {"default": False}),
                "German": ("BOOLEAN", {"default": False}),
                "Italian": ("BOOLEAN", {"default": False}),
                "Dutch": ("BOOLEAN", {"default": False}),
                "Arabic": ("BOOLEAN", {"default": False}),
                "Hebrew": ("BOOLEAN", {"default": False}),
                "Chinese": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("multi_translated",)
    FUNCTION = "run"
    CATEGORY = "Fremantle/Transcriber"

    def run(self, grouped: str, batch_size: int, **flags):
        obj = safe_json_load(grouped, {})
        items = obj.get("items", []) if isinstance(obj, dict) else []

        order = ["English", "Portuguese", "Spanish", "French", "German", "Italian", "Dutch", "Arabic", "Hebrew", "Chinese"]
        selected_langs = [l for l in order if bool(flags.get(l, False)) and l in TRANSLATE_MAP]

        if not selected_langs:
            raise RuntimeError(
                "FT_TranslateMultiGoogle: No target languages selected. Enable at least one checkbox."
            )

        total_batches = 0
        for it in items:
            if it.get("status") == "OK":
                blocks = it.get("blocks", []) or []
                total_batches += len(range(0, len(blocks), int(batch_size))) or 1

        total_steps = max(1, total_batches * len(selected_langs))
        pbar = ProgressBar(total_steps)

        out = []
        for it in items:
            if it.get("status") != "OK":
                out.append(it)
                continue

            blocks = it.get("blocks", []) or []
            original_texts = [b.get("text", "") for b in blocks]

            multi_translations = {}  # {lang_code: [translated lines...]}

            for lang in selected_langs:
                tgt_code = TRANSLATE_MAP[lang]
                translator = GoogleTranslator(source="auto", target=tgt_code)

                translated = []
                total = len(original_texts)

                for i in range(0, total, int(batch_size)):
                    chunk = original_texts[i:i + int(batch_size)]
                    try:
                        got = translator.translate("\n".join(chunk))
                        got_lines = (got or "").split("\n") if got else []
                        if len(got_lines) == len(chunk):
                            translated.extend(got_lines)
                        else:
                            for x in chunk:
                                translated.append(translator.translate(x) if x.strip() else x)
                    except Exception:
                        for x in chunk:
                            try:
                                translated.append(translator.translate(x) if x.strip() else x)
                            except Exception:
                                translated.append(x)

                    time.sleep(0.1)
                    pbar.update(1)

                multi_translations[tgt_code] = translated

            out.append({
                **it,
                "translate_multi": True,
                "target_languages": selected_langs,
                "translated_texts_by_lang_code": multi_translations,  # {code: texts}
            })

        return (json_dumps({"items": out}),)

# ----------------------------
# Node 7: Hallucination filter (single)
# ----------------------------
class FT_HallucinationFilter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "translated": ("STRING", {"forceInput": True}),
                "enabled": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filtered",)
    FUNCTION = "run"
    CATEGORY = "Fremantle/Transcriber"

    def run(self, translated: str, enabled: bool):
        obj = safe_json_load(translated, {})
        items = obj.get("items", []) if isinstance(obj, dict) else []

        if not enabled:
            return (json_dumps({"items": items}),)

        out = []
        for it in items:
            if it.get("status") != "OK":
                out.append(it)
                continue

            blocks = it.get("blocks", []) or []
            texts = it.get("translated_texts", None)
            if not texts:
                texts = [b.get("text", "") for b in blocks]

            filtered_blocks = build_filtered_blocks(blocks, texts, apply_filter=True)
            out.append({**it, "filtered_blocks": filtered_blocks})

        return (json_dumps({"items": out}),)

# ----------------------------
# Node 8: Export (SRT / TXT / JSON) - single
# ----------------------------
class FT_ExportSubtitles:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filtered": ("STRING", {"forceInput": True}),
                "output_dir": ("STRING", {"default": "", "multiline": False}),
                "format": (["SRT", "TXT", "JSON"],),
                "add_lang_suffix": ("BOOLEAN", {"default": True}),
            }
        }

    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("outputs",)
    FUNCTION = "run"
    CATEGORY = "Fremantle/Transcriber"

    def run(self, filtered: str, output_dir: str, format: str, add_lang_suffix: bool):
        obj = safe_json_load(filtered, {})
        items = obj.get("items", []) if isinstance(obj, dict) else []

        out_dir = (output_dir or "").strip().strip('"').strip("'")
        if not out_dir:
            raise RuntimeError("FT_ExportSubtitles: Empty output_dir.")

        os.makedirs(out_dir, exist_ok=True)

        outputs = []
        for it in items:
            path = it.get("path")
            if it.get("status") != "OK":
                outputs.append({"path": path, "status": it.get("status"), "skipped": True})
                continue

            base = os.path.splitext(os.path.basename(path))[0]
            suffix = ""
            if add_lang_suffix and it.get("translate") and it.get("target_code"):
                suffix = f"_{it['target_code']}"

            # Prefer filtered_blocks, fall back to blocks
            blocks_to_export = it.get("filtered_blocks") or it.get("blocks") or []

            if format == "JSON":
                fn = safe_path(os.path.join(out_dir, base + suffix + ".json"))
                payload = {
                    "path": path,
                    "detected_language": it.get("detected_language"),
                    "segments": it.get("segments", []),
                    "blocks": it.get("blocks", []),
                    "filtered_blocks": it.get("filtered_blocks", []),
                }
                with open(fn, "w", encoding="utf-8") as f:
                    f.write(json_dumps(payload))
                outputs.append({"path": path, "out": fn, "format": "JSON"})
                continue

            if format == "TXT":
                fn = safe_path(os.path.join(out_dir, base + suffix + ".txt"))
                with open(fn, "w", encoding="utf-8") as f:
                    for b in blocks_to_export:
                        txt = (b.get("text") or "").strip()
                        if txt:
                            f.write(txt + "\n")
                outputs.append({"path": path, "out": fn, "format": "TXT"})
                continue

            # SRT
            fn = safe_path(os.path.join(out_dir, base + suffix + ".srt"))
            subs = []
            idx = 1
            for b in blocks_to_export:
                txt = (b.get("text") or "").strip()
                if not txt:
                    continue
                for pt, ps, pe in split_for_srt(txt, float(b.get("start", 0.0)), float(b.get("end", 0.0))):
                    if not pt.strip():
                        continue
                    subs.append(srt.Subtitle(idx, ps, pe, pt))
                    idx += 1
            with open(fn, "w", encoding="utf-8") as f:
                f.write(srt.compose(subs))
            outputs.append({"path": path, "out": fn, "format": "SRT"})

        return (json_dumps({"outputs": outputs}),)

# ----------------------------
# Node 9: Export Multi (SRT/TXT/JSON) with hallucination filter + safe_path versioning
# ----------------------------
class FT_ExportMultiSubtitles:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "multi_translated": ("STRING", {"forceInput": True}),
                "output_dir": ("STRING", {"default": "", "multiline": False}),
                "format": (["SRT", "TXT", "JSON"],),
                "add_lang_suffix": ("BOOLEAN", {"default": True}),
                "apply_hallucination_filter": ("BOOLEAN", {"default": True}),
                "export_original_language": ("BOOLEAN", {"default": False}),
            }
        }

    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("outputs",)
    FUNCTION = "run"
    CATEGORY = "Fremantle/Transcriber"

    def run(
        self,
        multi_translated: str,
        output_dir: str,
        format: str,
        add_lang_suffix: bool,
        apply_hallucination_filter: bool,
        export_original_language: bool,
    ):
        obj = safe_json_load(multi_translated, {})
        items = obj.get("items", []) if isinstance(obj, dict) else []

        out_dir = (output_dir or "").strip().strip('"').strip("'")
        if not out_dir:
            raise RuntimeError("FT_ExportMultiSubtitles: Empty output_dir.")
        os.makedirs(out_dir, exist_ok=True)

        outputs = []

        for it in items:
            path = it.get("path")
            if it.get("status") != "OK":
                outputs.append({"path": path, "status": it.get("status"), "skipped": True})
                continue

            base = os.path.splitext(os.path.basename(path))[0]
            blocks = it.get("blocks", []) or []
            if not blocks:
                outputs.append({"path": path, "status": "ERROR", "error": "No blocks found to export."})
                continue

            # Multi translations (new format from FT_TranslateMultiGoogle):
            # translated_texts_by_lang_code: { "en": [...], "es": [...], ... }
            by_code = it.get("translated_texts_by_lang_code", {}) or {}
            target_langs = it.get("target_languages", None)
            if isinstance(target_langs, list) and target_langs:
                # Map language names -> codes for stable order if present
                ordered_codes = []
                for name in target_langs:
                    code = TRANSLATE_MAP.get(name)
                    if code and code in by_code:
                        ordered_codes.append(code)
                codes = ordered_codes if ordered_codes else list(by_code.keys())
            else:
                codes = list(by_code.keys())

            # Optionally export original language
            if export_original_language:
                suffix = "_original" if add_lang_suffix else ""
                if format == "SRT":
                    fn = safe_path(os.path.join(out_dir, base + suffix + ".srt"))
                    subs = []
                    idx = 1
                    for b in blocks:
                        txt = (b.get("text") or "").strip()
                        if not txt:
                            continue
                        for pt, ps, pe in split_for_srt(txt, float(b.get("start", 0.0)), float(b.get("end", 0.0))):
                            if not pt.strip():
                                continue
                            subs.append(srt.Subtitle(idx, ps, pe, pt))
                            idx += 1
                    with open(fn, "w", encoding="utf-8") as f:
                        f.write(srt.compose(subs))
                    outputs.append({"path": path, "out": fn, "format": "SRT", "target_code": "original"})
                elif format == "TXT":
                    fn = safe_path(os.path.join(out_dir, base + suffix + ".txt"))
                    with open(fn, "w", encoding="utf-8") as f:
                        for b in blocks:
                            txt = (b.get("text") or "").strip()
                            if txt:
                                f.write(txt + "\n")
                    outputs.append({"path": path, "out": fn, "format": "TXT", "target_code": "original"})
                else:  # JSON
                    fn = safe_path(os.path.join(out_dir, base + suffix + ".json"))
                    payload = {
                        "path": path,
                        "detected_language": it.get("detected_language"),
                        "segments": it.get("segments", []),
                        "blocks": blocks,
                    }
                    with open(fn, "w", encoding="utf-8") as f:
                        f.write(json_dumps(payload))
                    outputs.append({"path": path, "out": fn, "format": "JSON", "target_code": "original"})

            # Export each translation code
            for code in codes:
                texts = by_code.get(code, []) or []
                if not texts:
                    continue

                # Ensure same length pairing; if mismatch, pad with original
                if len(texts) < len(blocks):
                    for i in range(len(texts), len(blocks)):
                        texts.append(blocks[i].get("text", ""))
                elif len(texts) > len(blocks):
                    texts = texts[:len(blocks)]

                filtered_blocks = build_filtered_blocks(blocks, texts, apply_filter=bool(apply_hallucination_filter))

                suffix = ""
                if add_lang_suffix and code:
                    suffix = f"_{code}"

                if format == "JSON":
                    fn = safe_path(os.path.join(out_dir, base + suffix + ".json"))
                    payload = {
                        "path": path,
                        "target_code": code,
                        "blocks": blocks,
                        "filtered_blocks": filtered_blocks,
                    }
                    with open(fn, "w", encoding="utf-8") as f:
                        f.write(json_dumps(payload))
                    outputs.append({"path": path, "out": fn, "format": "JSON", "target_code": code})
                    continue

                if format == "TXT":
                    fn = safe_path(os.path.join(out_dir, base + suffix + ".txt"))
                    with open(fn, "w", encoding="utf-8") as f:
                        for b in filtered_blocks:
                            txt = (b.get("text") or "").strip()
                            if txt:
                                f.write(txt + "\n")
                    outputs.append({"path": path, "out": fn, "format": "TXT", "target_code": code})
                    continue

                # SRT
                fn = safe_path(os.path.join(out_dir, base + suffix + ".srt"))
                subs = []
                idx = 1
                for b in filtered_blocks:
                    txt = (b.get("text") or "").strip()
                    if not txt:
                        continue
                    for pt, ps, pe in split_for_srt(txt, float(b.get("start", 0.0)), float(b.get("end", 0.0))):
                        if not pt.strip():
                            continue
                        subs.append(srt.Subtitle(idx, ps, pe, pt))
                        idx += 1
                with open(fn, "w", encoding="utf-8") as f:
                    f.write(srt.compose(subs))
                outputs.append({"path": path, "out": fn, "format": "SRT", "target_code": code})

        return (json_dumps({"outputs": outputs}),)

# ----------------------------
# Node registration
# ----------------------------
NODE_CLASS_MAPPINGS = {
    "FT_Info": FT_Info,
    "FT_LoadMediaBatch": FT_LoadMediaBatch,
    "FT_WhisperModel": FT_WhisperModel,
    "FT_TranscribeBatch": FT_TranscribeBatch,
    "FT_GroupSegments": FT_GroupSegments,
    "FT_TranslateGoogle": FT_TranslateGoogle,
    "FT_TranslateMultiGoogle": FT_TranslateMultiGoogle,
    "FT_HallucinationFilter": FT_HallucinationFilter,
    "FT_ExportSubtitles": FT_ExportSubtitles,
    "FT_ExportMultiSubtitles": FT_ExportMultiSubtitles,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FT_Info": "FT • Info",
    "FT_LoadMediaBatch": "FT • Load Media Batch",
    "FT_WhisperModel": "FT • Whisper Model (cached)",
    "FT_TranscribeBatch": "FT • Transcribe Batch",
    "FT_GroupSegments": "FT • Group Segments",
    "FT_TranslateGoogle": "FT • Translate (Google)",
    "FT_TranslateMultiGoogle": "FT • Translate Multi (Google)",
    "FT_HallucinationFilter": "FT • Hallucination Filter",
    "FT_ExportSubtitles": "FT • Export (SRT/TXT/JSON)",
    "FT_ExportMultiSubtitles": "FT • Export Multi (SRT/TXT/JSON)",
}