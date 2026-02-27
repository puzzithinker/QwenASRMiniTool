"""
subtitle_formatter.py — 統一字幕格式化模組

支援兩種輸出格式：
- TXT: [00:00:00.000 --> 00:00:04.400]  Speaker: text (預設)
- SRT: 傳統字幕格式（含行號）

此模組為 逐字稿神器 提供統一的字幕輸出介面
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import List, Tuple, Optional


class SubtitleFormat(Enum):
    """字幕輸出格式列舉"""
    TXT = "txt"  # 新格式：[timestamp] text (預設)
    SRT = "srt"  # 傳統 SRT 格式


def format_timestamp(seconds: float, use_dot: bool = True) -> str:
    """
    將秒數格式化為時間戳字串
    
    Args:
        seconds: 時間（秒）
        use_dot: True 使用點號 (00:00:00.000), False 使用逗號 (00:00:00,000)
    
    Returns:
        格式化後的時間戳字串
    """
    ms = int(round(seconds * 1000))
    hh = ms // 3_600_000
    ms %= 3_600_000
    mm = ms // 60_000
    ms %= 60_000
    ss = ms // 1_000
    ms %= 1_000
    separator = "." if use_dot else ","
    return f"{hh:02d}:{mm:02d}:{ss:02d}{separator}{ms:03d}"


def parse_subtitle_file(file_path: Path) -> List[Tuple[float, float, str, Optional[str]]]:
    """
    解析字幕檔案（支援 TXT 和 SRT 格式）
    
    Args:
        file_path: 字幕檔案路徑
    
    Returns:
        條目列表，每個條目為 (start_time, end_time, text, speaker)
    """
    content = file_path.read_text(encoding="utf-8")
    
    # 嘗試檢測格式
    if " --> " in content and content.strip().startswith("["):
        # TXT 格式
        return _parse_txt_format(content)
    else:
        # SRT 格式
        return _parse_srt_format(content)


def _parse_txt_format(content: str) -> List[Tuple[float, float, str, Optional[str]]]:
    """解析 TXT 格式字幕"""
    import re
    entries = []
    
    # 匹配 [00:00:00.000 --> 00:00:04.400]  text
    pattern = r'\[(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})\]\s+(.+)'
    
    for match in re.finditer(pattern, content):
        start_str = match.group(1)
        end_str = match.group(2)
        text = match.group(3).strip()
        
        # 解析說話者
        speaker = None
        if ": " in text and not text.startswith("http"):
            parts = text.split(": ", 1)
            if len(parts) == 2 and parts[0].strip():
                speaker = parts[0].strip()
                text = parts[1].strip()
        
        start_sec = _timestamp_to_seconds(start_str)
        end_sec = _timestamp_to_seconds(end_str)
        
        entries.append((start_sec, end_sec, text, speaker))
    
    return entries


def _parse_srt_format(content: str) -> List[Tuple[float, float, str, Optional[str]]]:
    """解析 SRT 格式字幕"""
    import re
    entries = []
    
    # 分割條目
    blocks = re.split(r'\n\s*\n', content.strip())
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
        
        # 第二行應該是時間軸
        time_line = lines[1]
        match = re.match(r'(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})', time_line)
        
        if not match:
            continue
        
        start_str = match.group(1)
        end_str = match.group(2)
        
        # 第三行開始是文字
        text = ' '.join(lines[2:]).strip()
        
        # 解析說話者（檢查是否以 "SpeakerX：" 或 "說話者X：" 開頭）
        speaker = None
        speaker_match = re.match(r'^(Speaker\d+|說話者\d+)：(.+)$', text)
        if speaker_match:
            speaker = speaker_match.group(1)
            text = speaker_match.group(2).strip()
        
        start_sec = _timestamp_to_seconds(start_str.replace(',', '.'))
        end_sec = _timestamp_to_seconds(end_str.replace(',', '.'))
        
        entries.append((start_sec, end_sec, text, speaker))
    
    return entries


def _timestamp_to_seconds(ts: str) -> float:
    """將時間戳字串轉換為秒數"""
    try:
        parts = ts.split(':')
        hh = int(parts[0])
        mm = int(parts[1])
        ss_ms = parts[2].split('.')
        ss = int(ss_ms[0])
        ms = int(ss_ms[1]) if len(ss_ms) > 1 else 0
        return hh * 3600 + mm * 60 + ss + ms / 1000.0
    except (ValueError, IndexError):
        return 0.0


def write_subtitle_file(
    entries: List[Tuple[float, float, str, Optional[str]]],
    output_path: Path,
    format_type: SubtitleFormat = SubtitleFormat.TXT
) -> Path:
    """
    寫入字幕檔案
    
    Args:
        entries: 條目列表，每個條目為 (start_time, end_time, text, speaker)
        output_path: 輸出路徑（會根據格式調整副檔名）
        format_type: TXT 或 SRT 格式
    
    Returns:
        實際寫入的檔案路徑
    """
    # 根據格式調整副檔名
    actual_path = output_path.with_suffix(f".{format_type.value}")
    
    with open(actual_path, "w", encoding="utf-8") as f:
        if format_type == SubtitleFormat.TXT:
            # TXT 格式：[00:00:00.000 --> 00:00:04.400]  Speaker: text
            for s, e, line, spk in entries:
                prefix = f"{spk}: " if spk else ""
                start_ts = format_timestamp(s, use_dot=True)
                end_ts = format_timestamp(e, use_dot=True)
                f.write(f"[{start_ts} --> {end_ts}]  {prefix}{line}\n\n")
        else:
            # SRT 格式（傳統）
            for idx, (s, e, line, spk) in enumerate(entries, 1):
                prefix = f"{spk}：" if spk else ""
                ts_s = format_timestamp(s, use_dot=False)
                ts_e = format_timestamp(e, use_dot=False)
                f.write(f"{idx}\n{ts_s} --> {ts_e}\n{prefix}{line}\n\n")
    
    return actual_path


def format_to_string(format_type: SubtitleFormat) -> str:
    """將格式列舉轉換為顯示字串"""
    if format_type == SubtitleFormat.TXT:
        return "TXT ([timestamp] text)"
    return "SRT (傳統字幕格式)"


def string_to_format(format_str: str) -> SubtitleFormat:
    """將顯示字串轉換為格式列舉"""
    if "TXT" in format_str or "txt" in format_str:
        return SubtitleFormat.TXT
    return SubtitleFormat.SRT


# 為了向後相容，保留舊函式名稱
def _srt_ts(s: float) -> str:
    """舊版時間戳格式化函式（相容用）"""
    return format_timestamp(s, use_dot=True)
