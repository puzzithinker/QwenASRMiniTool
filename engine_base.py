"""engine_base.py — Abstract base class for ASR engines

Consolidates the shared process_file() logic previously duplicated across:
  ASREngine, ASREngine1p7B  (app.py)
  GPUASREngine              (app-gpu.py)
  ChatLLMASREngine          (chatllm_engine.py)

Subclasses only need to implement load() and transcribe().
The common pipeline (VAD → chunk → transcribe → format) lives here once.
"""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from asr_common import (
    SAMPLE_RATE,
    detect_speech_groups,
    split_to_lines,
    assign_timestamps,
    enforce_chunk_limit,
    get_srt_dir,
)
from subtitle_formatter import SubtitleFormat, write_subtitle_file, string_to_format


class ASREngineBase(ABC):
    """Abstract base for all ASR backends.

    Subclasses must implement:
      load()       — initialise models (called from background thread)
      transcribe() — single audio chunk → text string

    Provided by base class:
      process_file()         — full pipeline: audio file → subtitle file
      _process_chunk()       — per-chunk transcribe+format (overridable)
      _enforce_chunk_limit() — legacy alias for asr_common.enforce_chunk_limit

    Attributes (set by subclass in load()):
      ready        : bool — True when models are loaded and ready
      vad_sess     : onnxruntime.InferenceSession for Silero VAD
      diar_engine  : DiarizationEngine instance (optional)
      cc           : opencc.OpenCC instance (optional, for s2twp conversion)
    """

    max_chunk_secs: int = 30

    def __init__(self):
        self.ready       = False
        self._lock       = threading.Lock()
        self.vad_sess    = None
        self.diar_engine = None
        self.cc          = None

    # ── Abstract methods (subclass MUST implement) ────────────────────

    @abstractmethod
    def load(self, **kwargs) -> None:
        """Initialise models. Called from background thread.

        Implementations should accept at least:
          device, model_dir, cb (status callback)
        """
        ...

    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray,
        max_tokens: int = 300,
        language: str | None = None,
        context: str | None = None,
    ) -> str:
        """Transcribe a single 16 kHz float32 audio chunk to text."""
        ...

    # ── Overridable hook for per-chunk processing ─────────────────────

    def _process_chunk(
        self,
        chunk: np.ndarray,
        g0: float,
        g1: float,
        spk: str | None,
        language: str | None,
        context: str | None,
    ) -> list[tuple[float, float, str, str | None]]:
        """Process one audio chunk into subtitle entries.

        Default: transcribe → split_to_lines → assign_timestamps.
        Override in subclasses for ForcedAligner or custom logic.

        Returns: [(start_sec, end_sec, text, speaker), ...]
        """
        max_tok = 400 if language == "Japanese" else 300
        text = self.transcribe(chunk, max_tokens=max_tok, language=language, context=context)
        if not text:
            return []
        lines = split_to_lines(text)
        return [
            (s, e, line, spk)
            for s, e, line in assign_timestamps(lines, g0, g1)
        ]

    # ── Shared process_file pipeline ──────────────────────────────────

    def process_file(
        self,
        audio_path: Path,
        progress_cb=None,
        language: str | None = None,
        context: str | None = None,
        diarize: bool = False,
        n_speakers: int | None = None,
        output_format: str = "txt",
        output_dir: Path | None = None,
    ) -> Path | None:
        """Full pipeline: audio file → subtitle file.

        Parameters
        ----------
        audio_path    : Path to audio/video file
        progress_cb   : callback(done_idx, total, message)
        language      : Force language (e.g. "Chinese"), None = auto-detect
        context       : Recognition hint (lyrics, keywords)
        diarize       : Enable speaker diarization
        n_speakers    : Number of speakers (None = auto)
        output_format : "txt" or "srt"
        output_dir    : Output directory (None → module-level SRT_DIR)

        Returns
        -------
        Path to the generated subtitle file, or None if no speech detected.
        """
        import librosa

        audio, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)

        # ── Segmentation: diarization vs VAD ──────────────────────────
        use_diar = diarize and self.diar_engine is not None and self.diar_engine.ready
        if use_diar:
            diar_segs = self.diar_engine.diarize(audio, n_speakers=n_speakers)
            if not diar_segs:
                return None
            groups_spk = [
                (t0, t1,
                 audio[int(t0 * SAMPLE_RATE): int(t1 * SAMPLE_RATE)],
                 spk)
                for t0, t1, spk in diar_segs
            ]
        else:
            vad_groups = detect_speech_groups(audio, self.vad_sess, self.max_chunk_secs)
            if not vad_groups:
                return None
            groups_spk = [(g0, g1, chunk, None) for g0, g1, chunk in vad_groups]

        # Enforce chunk length limit (both paths may produce over-length segments)
        groups_spk = enforce_chunk_limit(groups_spk, self.max_chunk_secs)

        # ── ASR: transcribe each segment ──────────────────────────────
        all_subs: list[tuple[float, float, str, str | None]] = []
        total = len(groups_spk)
        for i, (g0, g1, chunk, spk) in enumerate(groups_spk):
            if progress_cb:
                spk_info = f" [{spk}]" if spk else ""
                progress_cb(i, total,
                            f"[{i+1}/{total}] {g0:.1f}s~{g1:.1f}s{spk_info}")
            subs = self._process_chunk(chunk, g0, g1, spk, language, context)
            all_subs.extend(subs)

        if not all_subs:
            return None

        # ── Write subtitle file ───────────────────────────────────────
        sub_format = string_to_format(output_format)
        if progress_cb:
            progress_cb(total, total, f"寫入 {sub_format.value.upper()}…")

        srt_dir = output_dir or get_srt_dir()
        srt_dir.mkdir(exist_ok=True)
        out = srt_dir / (audio_path.stem + ".srt")
        actual_path = write_subtitle_file(all_subs, out, sub_format)
        return actual_path

    # ── Legacy alias ──────────────────────────────────────────────────

    def _enforce_chunk_limit(
        self,
        groups: list[tuple[float, float, np.ndarray, str | None]],
    ) -> list[tuple[float, float, np.ndarray, str | None]]:
        """Legacy wrapper. Prefer asr_common.enforce_chunk_limit() directly."""
        return enforce_chunk_limit(groups, self.max_chunk_secs)
