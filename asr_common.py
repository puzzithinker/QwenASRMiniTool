"""asr_common.py — Shared ASR utilities

Single source of truth for functions previously duplicated across:
  app.py, app-gpu.py, chatllm_engine.py, streamlit_vulkan.py

Contents:
  • Constants (sample rate, VAD, subtitle formatting)
  • Runtime config (vad_threshold, output_simplified, srt_dir)
  • detect_speech_groups()  — Silero VAD segmentation
  • split_to_lines()        — Punctuation-aware subtitle line splitting
  • assign_timestamps()     — Proportional timestamp assignment
  • enforce_chunk_limit()   — Split long audio chunks
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


# ══════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════

SAMPLE_RATE          = 16_000
VAD_CHUNK            = 512
MAX_GROUP_SEC        = 20
MAX_CHARS            = 20
MIN_SUB_SEC          = 0.6
GAP_SEC              = 0.08
RT_SILENCE_CHUNKS    = 25     # ~0.8s silence before triggering transcription
RT_MAX_BUFFER_CHUNKS = 600    # ~19s upper bound for forced transcription


# ══════════════════════════════════════════════════════
# Punctuation sets
# ══════════════════════════════════════════════════════

# Chinese clause-ending punctuation (hidden in subtitle output)
ZH_CLAUSE_END = frozenset('，。？！；：…—、·')
# English sentence-ending punctuation (including comma)
EN_SENT_END = frozenset('.,!?;:')
# Combined set — triggers line break and is removed from output
PUNCT_ALL = ZH_CLAUSE_END | EN_SENT_END


# ══════════════════════════════════════════════════════
# Runtime configuration (module-level, thread-safe for reads)
# ══════════════════════════════════════════════════════

_vad_threshold: float = 0.5
_output_simplified: bool = False
_srt_dir: Path = Path("subtitles")


def get_vad_threshold() -> float:
    return _vad_threshold


def set_vad_threshold(value: float) -> None:
    global _vad_threshold
    _vad_threshold = value


def get_output_simplified() -> bool:
    return _output_simplified


def set_output_simplified(value: bool) -> None:
    global _output_simplified
    _output_simplified = value


def get_srt_dir() -> Path:
    _srt_dir.mkdir(exist_ok=True)
    return _srt_dir


def set_srt_dir(path: Path) -> None:
    global _srt_dir
    _srt_dir = path
    _srt_dir.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════
# VAD speech detection
# ══════════════════════════════════════════════════════

def detect_speech_groups(
    audio: np.ndarray,
    vad_sess,
    max_group_sec: int = MAX_GROUP_SEC,
    vad_threshold: float | None = None,
) -> list[tuple[float, float, np.ndarray]]:
    """Silero VAD segmentation → [(start_sec, end_sec, audio_chunk), ...]

    Parameters
    ----------
    audio         : 16 kHz float32 mono waveform
    vad_sess      : ONNX Runtime InferenceSession for Silero VAD
    max_group_sec : Maximum seconds per merged segment
    vad_threshold : Speech probability threshold (None → module default)
    """
    if vad_threshold is None:
        vad_threshold = _vad_threshold

    h = np.zeros((2, 1, 64), dtype=np.float32)
    c = np.zeros((2, 1, 64), dtype=np.float32)
    sr = np.array(SAMPLE_RATE, dtype=np.int64)
    n = len(audio) // VAD_CHUNK
    probs = []
    for i in range(n):
        chunk = audio[i * VAD_CHUNK:(i + 1) * VAD_CHUNK].astype(np.float32)[np.newaxis, :]
        out, h, c = vad_sess.run(None, {"input": chunk, "h": h, "c": c, "sr": sr})
        probs.append(float(out[0, 0]))
    if not probs:
        return [(0.0, len(audio) / SAMPLE_RATE, audio)]

    # ── Detect speech regions ─────────────────────────────────────────
    MIN_CH = 16    # minimum chunks for a valid speech region
    PAD    = 5     # padding chunks around speech
    MERGE  = 16    # merge gap threshold (chunks)

    raw: list[tuple[int, int]] = []
    in_sp = False
    s0 = 0
    for i, p in enumerate(probs):
        if p >= vad_threshold and not in_sp:
            s0 = i; in_sp = True
        elif p < vad_threshold and in_sp:
            if i - s0 >= MIN_CH:
                raw.append((max(0, s0 - PAD), min(n, i + PAD)))
            in_sp = False
    if in_sp and n - s0 >= MIN_CH:
        raw.append((max(0, s0 - PAD), n))
    if not raw:
        return []

    # ── Merge adjacent regions ────────────────────────────────────────
    merged = [list(raw[0])]
    for s, e in raw[1:]:
        if s - merged[-1][1] <= MERGE:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    # ── Split into max_group_sec groups ───────────────────────────────
    mx_samp = max_group_sec * SAMPLE_RATE
    groups: list[tuple[int, int]] = []
    gs = merged[0][0] * VAD_CHUNK
    ge = merged[0][1] * VAD_CHUNK
    for seg in merged[1:]:
        s = seg[0] * VAD_CHUNK
        e = seg[1] * VAD_CHUNK
        if e - gs > mx_samp:
            groups.append((gs, ge))
            gs = s
        ge = e
    groups.append((gs, ge))

    # ── Convert to (start_sec, end_sec, audio_chunk) ──────────────────
    result = []
    for gs, ge in groups:
        ns = max(1, int((ge - gs) // SAMPLE_RATE))
        ch = audio[gs: gs + ns * SAMPLE_RATE].astype(np.float32)
        if len(ch) < SAMPLE_RATE:
            continue
        result.append((gs / SAMPLE_RATE, gs / SAMPLE_RATE + ns, ch))
    return result


# ══════════════════════════════════════════════════════
# Subtitle line splitting
# ══════════════════════════════════════════════════════

def split_to_lines(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Split transcribed text into subtitle lines.

    Rules (unified for CJK and Latin):
      1. All punctuation → immediate line break; punctuation hidden in output
      2. English words kept whole; word-boundary spaces preserved
      3. MAX_CHARS protection: force line break when exceeded
    """
    if "<asr_text>" in text:
        text = text.split("<asr_text>", 1)[1]
    text = text.strip()
    if not text:
        return []

    lines: list[str] = []
    buf = ""

    i = 0
    while i < len(text):
        ch = text[i]

        # ── Punctuation: line break, hide ─────────────────────────────
        if ch in PUNCT_ALL:
            if buf.strip():
                lines.append(buf.strip())
            buf = ""
            i += 1
            continue

        # ── Latin word: collect whole word ─────────────────────────────
        if ch.isalpha() and ord(ch) < 128:
            j = i
            while j < len(text) and text[j].isalpha() and ord(text[j]) < 128:
                j += 1
            word = text[i:j]
            prefix = " " if buf and not buf.endswith(" ") else ""
            if len(buf) + len(prefix) + len(word) > max_chars and buf.strip():
                lines.append(buf.strip())
                buf = word
            else:
                buf += prefix + word
            i = j
            continue

        # ── Space: preserve word separation ───────────────────────────
        if ch == " ":
            if buf and not buf.endswith(" "):
                buf += " "
            i += 1
            if len(buf.rstrip()) >= max_chars:
                lines.append(buf.strip())
                buf = ""
            continue

        # ── CJK / digits / other: accumulate ──────────────────────────
        buf += ch
        i += 1
        if len(buf) >= max_chars:
            lines.append(buf.strip())
            buf = ""

    if buf.strip():
        lines.append(buf.strip())
    return [line for line in lines if line.strip()]


# ══════════════════════════════════════════════════════
# Timestamp assignment
# ══════════════════════════════════════════════════════

def assign_timestamps(
    lines: list[str],
    g0: float,
    g1: float,
    min_sub_sec: float = MIN_SUB_SEC,
    gap_sec: float = GAP_SEC,
) -> list[tuple[float, float, str]]:
    """Assign proportional timestamps to subtitle lines within [g0, g1].

    Returns [(start_sec, end_sec, text), ...]
    """
    if not lines:
        return []
    total = sum(len(l) for l in lines)
    if total == 0:
        return []
    dur = g1 - g0
    res = []
    cur = g0
    for i, line in enumerate(lines):
        end = cur + max(min_sub_sec, dur * len(line) / total)
        if i == len(lines) - 1:
            end = max(end, g1)
        res.append((cur, end, line))
        cur = end + gap_sec
    return res


# ══════════════════════════════════════════════════════
# Chunk limit enforcement
# ══════════════════════════════════════════════════════

def enforce_chunk_limit(
    groups: list[tuple[float, float, np.ndarray, str | None]],
    max_chunk_secs: int,
) -> list[tuple[float, float, np.ndarray, str | None]]:
    """Split audio segments exceeding max_chunk_secs into equal sub-segments.

    Prevents silent truncation when audio exceeds the model's input length.
    """
    max_samples = max_chunk_secs * SAMPLE_RATE
    result = []
    for t0, t1, chunk, spk in groups:
        if len(chunk) <= max_samples:
            result.append((t0, t1, chunk, spk))
        else:
            pos = 0
            while pos < len(chunk):
                piece = chunk[pos: pos + max_samples]
                if len(piece) < SAMPLE_RATE:   # skip residual < 1s
                    break
                piece_t0 = t0 + pos / SAMPLE_RATE
                piece_t1 = min(t1, piece_t0 + len(piece) / SAMPLE_RATE)
                result.append((piece_t0, piece_t1, piece, spk))
                pos += max_samples
    return result
