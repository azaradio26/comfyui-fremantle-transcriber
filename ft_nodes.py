from comfy.utils import ProgressBar
import os
import json
import time
import re
import datetime
import traceback
import subprocess
from typing import Dict, Any, List

import torch
import whisper
import numpy as np
import srt
from deep_translator import GoogleTranslator

# ----------------------------
# Helpers
# ----------------------------

VALID_EXTS = (
    ".mp3", ".wav", ".flac", ".m4a", ".ogg",
    ".mp4", ".mkv", ".avi", ".mov", ".ts",
    ".aiff", ".aif", ".aac", ".wma"
)

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
    "Hebrew": "he",    # <-- Whisper Requires "iw"
    "Arabic": "ar",
    "German": "de",
    "Dutch": "nl",
    "Italian": "it",
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
    "Hebrew": "iw",   # <-- Whisper Requires "iw"
}

def ensure_ffmpeg_on_path():
    # On macOS, when running via an app bundle, PATH can be limited.
    # In ComfyUI you usually have a proper shell PATH, but this doesn't hurt.
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

def safe_json_load(s: str, fallback):
    try:
        return json.loads(s)
    except Exception:
        return fallback

def json_dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)

def ffprobe_has_audio(path: str) -> bool:
    # Lightweight & fast audio stream check
    cmd = ["ffprobe", "-v", "error", "-select_streams", "a",
           "-show_entries", "stream=codec_name", "-of", "csv=p=0", path]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
        return bool(out)
    except Exception:
        # If ffprobe fails, don't block the pipeline
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
            t_start = float(seg["start"])
        t_texts.append((seg.get("text") or "").strip())
        if (limit > 0 and len(t_texts) >= limit) or limit == 0 or i == len(segments) - 1:
            blocks.append({
                "start": t_start,
                "end": float(seg["end"]),
                "text": " ".join(t_texts).strip()
            })
            t_texts = []
            t_start = None
    return blocks

def split_for_srt(text: str, start: float, end: float, max_w: int = 42):
    import textwrap
    lines = textwrap.wrap(text, width=max_w, break_long_words=False)
    if len(lines) <= 2:
        return [("\n".join(lines),
                 datetime.timedelta(seconds=start),
                 datetime.timedelta(seconds=end))]
    mid = len(lines) // 2
    p1 = "\n".join(lines[:mid])
    p2 = "\n".join(lines[mid:])
    t_split = start + (len(p1) / max(1, len(text))) * (end - start)
    return [
        (p1, datetime.timedelta(seconds=start), datetime.timedelta(seconds=t_split)),
        (p2, datetime.timedelta(seconds=t_split), datetime.timedelta(seconds=end)),
    ]

def safe_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    b, e = os.path.splitext(path)
    c = 1
    while os.path.exists(f"{b}_v{c}{e}"):
        c += 1
    return f"{b}_v{c}{e}"

# Simple in-memory model cache (used for dropdown selection + reuse)
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
        # Use Whisper named models (if available in the environment)
        model = whisper.load_model(model_choice, device=device)

    _MODEL_CACHE[key] = model
    return model

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
        p = (input_path or "").strip()
        if not p:
            return (json_dumps({"files": [], "error": "Empty input path"}),)

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

        files.sort()
        return (json_dumps({"files": files}),)

# ----------------------------
# Node 2: Whisper model loader (cache)
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
    # Note: On Mac it's usually CPU; CUDA only on NVIDIA (Windows/Linux).
    # If CUDA is not available, whisper/torch may fail.
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        # Load once to validate and cache the model.
        model = load_whisper_model(model_choice, custom_pt_path, device)
        _ = model  # just to silence linters

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

    def run(self,
            files: str,
            model_handle: str,
            source_language: str,
            word_timestamps: bool,
            temperature: float,
            no_speech_threshold: float,
            logprob_threshold: float,
            compression_ratio_threshold: float,
            skip_if_no_audio: bool,
            skip_if_silent: bool,
            silent_threshold_db: float
            ):

        files_obj = safe_json_load(files, {})
        files = files_obj.get("files", []) if isinstance(files_obj, dict) else []
        mh = safe_json_load(model_handle, {})

        model_choice = mh.get("model_choice", "large-v3")
        custom_pt_path = mh.get("custom_pt_path", "")
        device = mh.get("device", "cpu")

        model = load_whisper_model(model_choice, custom_pt_path, device)

        forced = LANG_MAP.get(source_language, None)

        results = []
        for path in files:
            item = {"path": path}
            try:
                if skip_if_no_audio and not ffprobe_has_audio(path):
                    results.append({"path": path, "status": "EMPTY", "reason": "no_audio"})
                    continue
                if skip_if_silent and ffmpeg_is_silent(path, threshold_db=float(silent_threshold_db)):
                    results.append({"path": path, "status": "EMPTY", "reason": "silent"})
                    continue

                audio = whisper.load_audio(path).astype(np.float32)

                detected = None
                if not forced:
                    audio_trimmed = whisper.pad_or_trim(audio)
                    mel = whisper.log_mel_spectrogram(audio_trimmed, n_mels=model.dims.n_mels).to(model.device)
                    _, probs = model.detect_language(mel)
                    iso = max(probs, key=probs.get)
                    detected = iso

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

        return (json_dumps({"items": results}),)

# ----------------------------
# Node 4: Group segments (movie/group10/group20/custom limit)
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
# Node 5: Translate Google (batch blocks)
# ----------------------------
class FT_TranslateGoogle:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "grouped": ("STRING", {"forceInput": True}),
                "target_language": (["English", "Portuguese", "Spanish", "French", "German", "Italian", "Dutch", "Arabic", "Hebrew"],),
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

        out = []
        for it in items:
            if it.get("status") != "OK":
                out.append(it)
                continue

            blocks = it.get("blocks", [])
            texts = [b.get("text", "") for b in blocks]

            translated = []
            total = len(texts)
            for i in range(0, total, int(batch_size)):
                chunk = texts[i:i+int(batch_size)]
                try:
                    got = translator.translate("\n".join(chunk)).split("\n")
                    if len(got) == len(chunk):
                        translated.extend(got)
                    else:
                        # fallback line-by-line
                        for x in chunk:
                            translated.append(translator.translate(x))
                except Exception:
                    for x in chunk:
                        translated.append(translator.translate(x))

            out.append({
                **it,
                "translate": True,
                "target_language": target_language,
                "target_code": tgt_code,
                "translated_texts": translated
            })

        return (json_dumps({"items": out}),)
        
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
                "Spanish": ("BOOLEAN", {"default": True}),
                "French": ("BOOLEAN", {"default": True}),
                "German": ("BOOLEAN", {"default": False}),
                "Italian": ("BOOLEAN", {"default": False}),
                "Dutch": ("BOOLEAN", {"default": False}),
                "Arabic": ("BOOLEAN", {"default": False}),
                "Hebrew": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("multi_translated",)
    FUNCTION = "run"
    CATEGORY = "Fremantle/Transcriber"

    def run(self, grouped: str, batch_size: int, **flags):
        obj = safe_json_load(grouped, {})
        items = obj.get("items", []) if isinstance(obj, dict) else []

        # Deterministic order (nice for UI + exports)
        order = ["English", "Portuguese", "Spanish", "French", "German", "Italian", "Dutch", "Arabic", "Hebrew"]

        supported = set(TRANSLATE_MAP.keys())

        # flags will contain: {"English": True, "French": False, ...}
        langs = [l for l in order if bool(flags.get(l, False)) and l in supported]

        if not langs:
            return (json_dumps({
                "items": items,
                "error": "No target languages selected. Enable at least one language checkbox.",
                "supported_languages": sorted(list(supported)),
            }),)

        total_blocks = sum(len(it.get("blocks", [])) for it in items if it.get("status") == "OK")
        total_steps = max(1, total_blocks * len(langs))
        pbar = ProgressBar(total_steps)

        out = []
        for it in items:
            if it.get("status") != "OK":
                out.append(it)
                continue

            blocks = it.get("blocks", [])
            texts = [b.get("text", "") for b in blocks]

            translated_by_lang = {}
            target_codes_by_lang = {}

            for lang in langs:
                tgt_code = TRANSLATE_MAP.get(lang, "en")
                target_codes_by_lang[lang] = tgt_code

                translator = GoogleTranslator(source="auto", target=tgt_code)

                translated = []
                for i in range(0, len(texts), int(batch_size)):
                    chunk = texts[i:i + int(batch_size)]
                    try:
                        got = translator.translate("\n".join(chunk)).split("\n")
                        if len(got) == len(chunk):
                            translated.extend(got)
                        else:
                            for x in chunk:
                                translated.append(translator.translate(x))
                    except Exception:
                        for x in chunk:
                            translated.append(translator.translate(x))

                    pbar.update(len(chunk))

                translated_by_lang[lang] = translated

            out.append({
                **it,
                "translate_multi": True,
                "target_languages": langs,
                "target_codes_by_lang": target_codes_by_lang,
                "translated_texts_by_lang": translated_by_lang,
            })

        return (json_dumps({"items": out}),)
        
class FT_SelectTranslationLanguage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "multi_translated": ("STRING", {"forceInput": True}),
                "language": (list(TRANSLATE_MAP.keys()),),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("translated",)
    FUNCTION = "run"
    CATEGORY = "Fremantle/Transcriber"

    def run(self, multi_translated: str, language: str):
        obj = safe_json_load(multi_translated, {})
        items = obj.get("items", []) if isinstance(obj, dict) else []

        out = []
        for it in items:
            if it.get("status") != "OK":
                out.append(it)
                continue

            by_lang = it.get("translated_texts_by_lang", {}) or {}
            codes = it.get("target_codes_by_lang", {}) or {}

            translated_texts = by_lang.get(language, None)
            tgt_code = codes.get(language, TRANSLATE_MAP.get(language, "en"))

            # If not found, pass-through without translation (still works)
            if not translated_texts:
                blocks = it.get("blocks", [])
                translated_texts = [b.get("text", "") for b in blocks]

            out.append({
                **it,
                "translate": True,
                "target_language": language,
                "target_code": tgt_code,
                "translated_texts": translated_texts,
            })

        return (json_dumps({"items": out}),)

# ----------------------------
# Node 6: Hallucination filter (on translated_texts OR original blocks)
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

            blocks = it.get("blocks", [])
            texts = it.get("translated_texts", None)

            # if no translation is available, use the original text
            if not texts:
                texts = [b.get("text", "") for b in blocks]

            keep = []
            last_v = ""
            for b, t in zip(blocks, texts):
                dur = float(b.get("end", 0)) - float(b.get("start", 0))
                if is_hallucination(t, dur, last_v):
                    continue
                keep.append({"start": b["start"], "end": b["end"], "text": t})
                last_v = (t or "").strip().lower()

            out.append({**it, "filtered_blocks": keep})

        return (json_dumps({"items": out}),)

# ----------------------------
# Node 7: Export (SRT / TXT / JSON segments)
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

        out_dir = (output_dir or "").strip()
        if not out_dir:
            return (json_dumps({"outputs": [], "error": "Empty output directory"}),)

        os.makedirs(out_dir, exist_ok=True)

        outputs = []
        for it in items:
            path = it.get("path")
            status = it.get("status")
            if status != "OK":
                outputs.append({"path": path, "status": status, "skipped": True})
                continue

            base = os.path.splitext(os.path.basename(path))[0]

            suffix = ""
            if add_lang_suffix and it.get("translate") and it.get("target_code"):
                suffix = f"_{it['target_code']}"

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
                blocks = it.get("filtered_blocks", [])
                with open(fn, "w", encoding="utf-8") as f:
                    for b in blocks:
                        f.write(b.get("text", "").strip() + "\n")
                outputs.append({"path": path, "out": fn, "format": "TXT"})
                continue

            # SRT
            fn = safe_path(os.path.join(out_dir, base + suffix + ".srt"))
            blocks = it.get("filtered_blocks", [])
            subs = []
            idx = 1
            for b in blocks:
                txt = (b.get("text") or "").strip()
                for pt, ps, pe in split_for_srt(txt, float(b["start"]), float(b["end"])):
                    subs.append(srt.Subtitle(idx, ps, pe, pt))
                    idx += 1
            with open(fn, "w", encoding="utf-8") as f:
                f.write(srt.compose(subs))
            outputs.append({"path": path, "out": fn, "format": "SRT"})

        return (json_dumps({"outputs": outputs}),)
        
class FT_ExportMultiSubtitles:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # accepts output of FT_TranslateMultiGoogle
                "multi_translated": ("STRING", {"forceInput": True}),
                "output_dir": ("STRING", {"default": "", "multiline": False}),
                "format": (["SRT", "TXT", "JSON"],),
                "add_lang_suffix": ("BOOLEAN", {"default": True}),
                # optional: apply hallucination filter during export
                "apply_hallucination_filter": ("BOOLEAN", {"default": True}),
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
    ):
        obj = safe_json_load(multi_translated, {})
        items = obj.get("items", []) if isinstance(obj, dict) else []

        out_dir = (output_dir or "").strip()
        if not out_dir:
            return (json_dumps({"outputs": [], "error": "Empty output directory"}),)

        os.makedirs(out_dir, exist_ok=True)

        # compute total steps for progress:
        # sum over OK items: (num_blocks * num_languages)
        total_steps = 0
        for it in items:
            if it.get("status") != "OK":
                continue
            blocks = it.get("blocks", [])
            by_lang = it.get("translated_texts_by_lang", {}) or {}
            langs = list(by_lang.keys())
            total_steps += len(blocks) * max(1, len(langs))

        pbar = ProgressBar(max(1, total_steps))

        outputs = []

        for it in items:
            path = it.get("path")
            status = it.get("status")

            if status != "OK":
                outputs.append({"path": path, "status": status, "skipped": True})
                continue

            blocks = it.get("blocks", [])
            by_lang = it.get("translated_texts_by_lang", {}) or {}
            codes = it.get("target_codes_by_lang", {}) or {}

            # If for some reason it isn't multi, fall back to single-style export
            if not by_lang:
                # Try single translated_texts, else original
                texts = it.get("translated_texts", None)
                if not texts:
                    texts = [b.get("text", "") for b in blocks]

                filtered_blocks = self._build_filtered_blocks(
                    blocks=blocks,
                    texts=texts,
                    apply_filter=apply_hallucination_filter
                )

                out_files = self._export_one_language(
                    path=path,
                    detected_language=it.get("detected_language"),
                    segments=it.get("segments", []),
                    blocks=blocks,
                    filtered_blocks=filtered_blocks,
                    output_dir=out_dir,
                    format=format,
                    suffix=""  # no language suffix available
                )

                outputs.extend(out_files)
                # progress
                pbar.update(len(blocks))
                continue

            # Multi-language export
            # Ensure deterministic order (use target_languages if present)
            lang_order = it.get("target_languages", None)
            if isinstance(lang_order, list) and lang_order:
                langs = [l for l in lang_order if l in by_lang]
            else:
                langs = list(by_lang.keys())

            base = os.path.splitext(os.path.basename(path))[0]

            for lang in langs:
                texts = by_lang.get(lang, []) or []
                tgt_code = codes.get(lang, TRANSLATE_MAP.get(lang, "en"))

                filtered_blocks = self._build_filtered_blocks(
                    blocks=blocks,
                    texts=texts if texts else [b.get("text", "") for b in blocks],
                    apply_filter=apply_hallucination_filter
                )

                suffix = ""
                if add_lang_suffix and tgt_code:
                    suffix = f"_{tgt_code}"

                out_files = self._export_one_language(
                    path=path,
                    detected_language=it.get("detected_language"),
                    segments=it.get("segments", []),
                    blocks=blocks,
                    filtered_blocks=filtered_blocks,
                    output_dir=out_dir,
                    format=format,
                    suffix=suffix
                )

                # add metadata for clarity
                for o in out_files:
                    o["target_language"] = lang
                    o["target_code"] = tgt_code
                    o["base"] = base

                outputs.extend(out_files)

                # progress: 1 step per block for each language
                pbar.update(len(blocks))

        return (json_dumps({"outputs": outputs}),)

    def _build_filtered_blocks(self, blocks, texts, apply_filter: bool):
        if not apply_filter:
            # keep timings, just assign text
            out = []
            for b, t in zip(blocks, texts):
                out.append({"start": b["start"], "end": b["end"], "text": t})
            return out

        keep = []
        last_v = ""
        for b, t in zip(blocks, texts):
            dur = float(b.get("end", 0)) - float(b.get("start", 0))
            if is_hallucination(t, dur, last_v):
                continue
            keep.append({"start": b["start"], "end": b["end"], "text": t})
            last_v = (t or "").strip().lower()
        return keep

    def _export_one_language(
        self,
        path: str,
        detected_language,
        segments,
        blocks,
        filtered_blocks,
        output_dir: str,
        format: str,
        suffix: str
    ):
        base = os.path.splitext(os.path.basename(path))[0]
        out_files = []

        if format == "JSON":
            fn = safe_path(os.path.join(output_dir, base + suffix + ".json"))
            payload = {
                "path": path,
                "detected_language": detected_language,
                "segments": segments or [],
                "blocks": blocks or [],
                "filtered_blocks": filtered_blocks or [],
            }
            with open(fn, "w", encoding="utf-8") as f:
                f.write(json_dumps(payload))
            out_files.append({"path": path, "out": fn, "format": "JSON"})
            return out_files

        if format == "TXT":
            fn = safe_path(os.path.join(output_dir, base + suffix + ".txt"))
            with open(fn, "w", encoding="utf-8") as f:
                for b in (filtered_blocks or []):
                    f.write((b.get("text", "") or "").strip() + "\n")
            out_files.append({"path": path, "out": fn, "format": "TXT"})
            return out_files

        # SRT
        fn = safe_path(os.path.join(output_dir, base + suffix + ".srt"))
        subs = []
        idx = 1
        for b in (filtered_blocks or []):
            txt = (b.get("text") or "").strip()
            for pt, ps, pe in split_for_srt(txt, float(b["start"]), float(b["end"])):
                subs.append(srt.Subtitle(idx, ps, pe, pt))
                idx += 1
        with open(fn, "w", encoding="utf-8") as f:
            f.write(srt.compose(subs))
        out_files.append({"path": path, "out": fn, "format": "SRT"})
        return out_files

# ----------------------------
# Node registration
# ----------------------------
NODE_CLASS_MAPPINGS = {
    "FT_LoadMediaBatch": FT_LoadMediaBatch,
    "FT_WhisperModel": FT_WhisperModel,
    "FT_TranscribeBatch": FT_TranscribeBatch,
    "FT_GroupSegments": FT_GroupSegments,
    "FT_TranslateGoogle": FT_TranslateGoogle,
    "FT_HallucinationFilter": FT_HallucinationFilter,
    "FT_ExportSubtitles": FT_ExportSubtitles,
    "FT_TranslateMultiGoogle": FT_TranslateMultiGoogle,
    "FT_SelectTranslationLanguage": FT_SelectTranslationLanguage,
    "FT_ExportMultiSubtitles": FT_ExportMultiSubtitles,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FT_LoadMediaBatch": "FT • Load Media Batch",
    "FT_WhisperModel": "FT • Whisper Model (cached)",
    "FT_TranscribeBatch": "FT • Transcribe Batch",
    "FT_GroupSegments": "FT • Group Segments",
    "FT_TranslateGoogle": "FT • Translate (Google)",
    "FT_HallucinationFilter": "FT • Hallucination Filter",
    "FT_ExportSubtitles": "FT • Export (SRT/TXT/JSON)",
    "FT_TranslateMultiGoogle": "FT • Translate Multi (Google)",
    "FT_SelectTranslationLanguage": "FT • Select Translation Language",
    "FT_ExportMultiSubtitles": "FT • Export Multi (SRT/TXT/JSON)",
}