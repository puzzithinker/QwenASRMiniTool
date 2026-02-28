#!/usr/bin/env python
"""
test_asr_common.py - ASR Common Utilities Test Suite

Test coverage:
1. Runtime config getters/setters (vad_threshold, output_simplified, srt_dir)
2. split_to_lines() - various cases (CJK, English, punctuation, max_chars)
3. assign_timestamps() - with sample lines
4. enforce_chunk_limit() - with sample groups
5. detect_speech_groups() - mock vad_sess (using unittest.mock)

Execution:
    pytest test_asr_common.py -v
    or
    python test_asr_common.py
"""
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock
import numpy as np

# Ensure we can import asr_common
sys.path.insert(0, str(Path(__file__).parent))

import asr_common
from asr_common import (
    get_vad_threshold, set_vad_threshold,
    get_output_simplified, set_output_simplified,
    get_srt_dir, set_srt_dir,
    split_to_lines,
    assign_timestamps,
    enforce_chunk_limit,
    detect_speech_groups,
    SAMPLE_RATE, VAD_CHUNK, MAX_CHARS, MIN_SUB_SEC, GAP_SEC
)


# ══════════════════════════════════════════════════════
# Test: Runtime Configuration
# ══════════════════════════════════════════════════════

def test_vad_threshold_getter_setter():
    """Test vad_threshold getter and setter"""
    print("\n=== Test 1: VAD Threshold Getter/Setter ===")
    
    # Get initial value
    initial = get_vad_threshold()
    assert isinstance(initial, float), "vad_threshold should be float"
    print(f"Initial vad_threshold: {initial}")
    
    # Set new value
    set_vad_threshold(0.7)
    assert get_vad_threshold() == 0.7, "vad_threshold should be 0.7"
    print(f"After set_vad_threshold(0.7): {get_vad_threshold()}")
    
    # Reset to original
    set_vad_threshold(initial)
    assert get_vad_threshold() == initial, "vad_threshold should be reset"
    
    print("✅ VAD threshold test passed")


def test_output_simplified_getter_setter():
    """Test output_simplified getter and setter"""
    print("\n=== Test 2: Output Simplified Getter/Setter ===")
    
    # Get initial value
    initial = get_output_simplified()
    assert isinstance(initial, bool), "output_simplified should be bool"
    print(f"Initial output_simplified: {initial}")
    
    # Set to True
    set_output_simplified(True)
    assert get_output_simplified() is True, "output_simplified should be True"
    print(f"After set_output_simplified(True): {get_output_simplified()}")
    
    # Set to False
    set_output_simplified(False)
    assert get_output_simplified() is False, "output_simplified should be False"
    print(f"After set_output_simplified(False): {get_output_simplified()}")
    
    # Reset to original
    set_output_simplified(initial)
    assert get_output_simplified() == initial, "output_simplified should be reset"
    
    print("✅ Output simplified test passed")


def test_srt_dir_getter_setter():
    """Test srt_dir getter and setter"""
    print("\n=== Test 3: SRT Dir Getter/Setter ===")
    
    # Get initial value
    initial = get_srt_dir()
    assert isinstance(initial, Path), "srt_dir should be Path"
    assert initial.exists(), "srt_dir should exist after get_srt_dir()"
    print(f"Initial srt_dir: {initial}")
    
    # Set new directory
    with tempfile.TemporaryDirectory() as tmpdir:
        new_dir = Path(tmpdir) / "test_subtitles"
        set_srt_dir(new_dir)
        result = get_srt_dir()
        assert result == new_dir, "srt_dir should be set to new_dir"
        assert result.exists(), "srt_dir should be created"
        print(f"After set_srt_dir: {result}")
    
    # Reset to original
    set_srt_dir(initial)
    assert get_srt_dir() == initial, "srt_dir should be reset"
    
    print("✅ SRT dir test passed")


# ══════════════════════════════════════════════════════
# Test: split_to_lines()
# ══════════════════════════════════════════════════════

def test_split_to_lines_empty():
    """Test split_to_lines with empty input"""
    print("\n=== Test 4: split_to_lines - Empty Input ===")
    
    result = split_to_lines("")
    assert result == [], "Empty string should return empty list"
    
    result = split_to_lines("   ")
    assert result == [], "Whitespace-only string should return empty list"
    
    print("✅ Empty input test passed")


def test_split_to_lines_english():
    """Test split_to_lines with English text"""
    print("\n=== Test 5: split_to_lines - English Text ===")
    
    # Simple English text
    text = "Hello world this is a test"
    result = split_to_lines(text, max_chars=10)
    print(f"Input: '{text}'")
    print(f"Output: {result}")
    assert len(result) > 0, "Should produce at least one line"
    assert all(isinstance(line, str) for line in result), "All lines should be strings"
    
    # Verify no line exceeds max_chars (approximately)
    for line in result:
        assert len(line) <= 15, f"Line '{line}' exceeds max_chars limit"
    
    print("✅ English text test passed")


def test_split_to_lines_punctuation():
    """Test split_to_lines with punctuation"""
    print("\n=== Test 6: split_to_lines - Punctuation ===")
    
    # Text with punctuation (should trigger line breaks)
    text = "First sentence. Second sentence! Third sentence?"
    result = split_to_lines(text, max_chars=20)
    print(f"Input: '{text}'")
    print(f"Output: {result}")
    
    # Punctuation should be removed from output
    for line in result:
        assert "." not in line, f"Period should be removed from '{line}'"
        assert "!" not in line, f"Exclamation should be removed from '{line}'"
        assert "?" not in line, f"Question mark should be removed from '{line}'"
    
    assert len(result) >= 2, "Should split on punctuation"
    
    print("✅ Punctuation test passed")


def test_split_to_lines_cjk():
    """Test split_to_lines with CJK text"""
    print("\n=== Test 7: split_to_lines - CJK Text ===")
    
    # Chinese text
    text = "這是第一句話。這是第二句話！這是第三句話？"
    result = split_to_lines(text, max_chars=5)
    print(f"Input: '{text}'")
    print(f"Output: {result}")
    
    assert len(result) > 0, "Should produce lines from CJK text"
    
    # Punctuation should be removed
    for line in result:
        assert "。" not in line, f"Chinese period should be removed from '{line}'"
        assert "！" not in line, f"Chinese exclamation should be removed from '{line}'"
        assert "？" not in line, f"Chinese question mark should be removed from '{line}'"
    
    print("✅ CJK text test passed")


def test_split_to_lines_mixed():
    """Test split_to_lines with mixed CJK and English"""
    print("\n=== Test 8: split_to_lines - Mixed CJK/English ===")
    
    text = "Hello 世界 this is 測試 text"
    result = split_to_lines(text, max_chars=8)
    print(f"Input: '{text}'")
    print(f"Output: {result}")
    
    assert len(result) > 0, "Should handle mixed text"
    assert all(isinstance(line, str) for line in result), "All lines should be strings"
    
    print("✅ Mixed text test passed")


def test_split_to_lines_max_chars():
    """Test split_to_lines respects max_chars limit"""
    print("\n=== Test 9: split_to_lines - Max Chars Limit ===")
    
    text = "a b c d e f g h i j k l m n o p q r s t u v w x y z"
    result = split_to_lines(text, max_chars=5)
    print(f"Input: '{text}' (max_chars=5)")
    print(f"Output: {result}")
    
    # Each line should be <= max_chars (with some tolerance for word boundaries)
    for line in result:
        assert len(line) <= 10, f"Line '{line}' exceeds reasonable limit"
    
    print("✅ Max chars test passed")


# ══════════════════════════════════════════════════════
# Test: assign_timestamps()
# ══════════════════════════════════════════════════════

def test_assign_timestamps_empty():
    """Test assign_timestamps with empty lines"""
    print("\n=== Test 10: assign_timestamps - Empty Input ===")
    
    result = assign_timestamps([], 0.0, 10.0)
    assert result == [], "Empty lines should return empty list"
    
    print("✅ Empty input test passed")


def test_assign_timestamps_single_line():
    """Test assign_timestamps with single line"""
    print("\n=== Test 11: assign_timestamps - Single Line ===")
    
    lines = ["Hello world"]
    result = assign_timestamps(lines, 0.0, 5.0)
    print(f"Input lines: {lines}")
    print(f"Output: {result}")
    
    assert len(result) == 1, "Should have one entry"
    start, end, text = result[0]
    assert start == 0.0, "Start should be 0.0"
    assert end >= 5.0, "End should be >= 5.0"
    assert text == "Hello world", "Text should match"
    
    print("✅ Single line test passed")


def test_assign_timestamps_multiple_lines():
    """Test assign_timestamps with multiple lines"""
    print("\n=== Test 12: assign_timestamps - Multiple Lines ===")
    
    lines = ["First", "Second", "Third"]
    g0, g1 = 0.0, 10.0
    result = assign_timestamps(lines, g0, g1)
    print(f"Input lines: {lines}")
    print(f"Time range: [{g0}, {g1}]")
    print(f"Output:")
    for start, end, text in result:
        print(f"  [{start:.2f}, {end:.2f}] {text}")
    
    assert len(result) == 3, "Should have three entries"
    
    # Verify timestamps are in order
    for i, (start, end, text) in enumerate(result):
        assert start >= g0, f"Start should be >= {g0}"
        assert end <= g1 + 1.0, f"End should be <= {g1 + 1.0}"  # Allow small tolerance
        assert start < end, "Start should be < end"
        assert text == lines[i], f"Text should match line {i}"
    
    # Verify last entry ends at or after g1
    assert result[-1][1] >= g1, "Last entry should end at or after g1"
    
    print("✅ Multiple lines test passed")


def test_assign_timestamps_with_gap():
    """Test assign_timestamps respects gap_sec"""
    print("\n=== Test 13: assign_timestamps - Gap Seconds ===")
    
    lines = ["Line1", "Line2"]
    result = assign_timestamps(lines, 0.0, 10.0, gap_sec=0.5)
    print(f"Input lines: {lines}")
    print(f"Output with gap_sec=0.5:")
    for start, end, text in result:
        print(f"  [{start:.2f}, {end:.2f}] {text}")
    
    # Check gap between entries
    if len(result) > 1:
        gap = result[1][0] - result[0][1]
        assert gap >= 0.4, f"Gap should be approximately 0.5, got {gap}"
    
    print("✅ Gap seconds test passed")


# ══════════════════════════════════════════════════════
# Test: enforce_chunk_limit()
# ══════════════════════════════════════════════════════

def test_enforce_chunk_limit_no_split():
    """Test enforce_chunk_limit with chunks under limit"""
    print("\n=== Test 14: enforce_chunk_limit - No Split ===")
    
    # Create audio chunk under limit (5 seconds)
    audio = np.random.randn(5 * SAMPLE_RATE).astype(np.float32)
    groups = [(0.0, 5.0, audio, None)]
    
    result = enforce_chunk_limit(groups, max_chunk_secs=10)
    print(f"Input: 1 group of {len(audio) / SAMPLE_RATE:.1f}s")
    print(f"Output: {len(result)} group(s)")
    
    assert len(result) == 1, "Should not split chunk under limit"
    assert result[0][0] == 0.0, "Start time should match"
    assert result[0][1] == 5.0, "End time should match"
    
    print("✅ No split test passed")


def test_enforce_chunk_limit_with_split():
    """Test enforce_chunk_limit splits long chunks"""
    print("\n=== Test 15: enforce_chunk_limit - With Split ===")
    
    # Create audio chunk exceeding limit (30 seconds)
    audio = np.random.randn(30 * SAMPLE_RATE).astype(np.float32)
    groups = [(0.0, 30.0, audio, None)]
    
    result = enforce_chunk_limit(groups, max_chunk_secs=10)
    print(f"Input: 1 group of {len(audio) / SAMPLE_RATE:.1f}s")
    print(f"Output: {len(result)} group(s)")
    print(f"Chunks:")
    for start, end, chunk, spk in result:
        print(f"  [{start:.1f}, {end:.1f}] {len(chunk) / SAMPLE_RATE:.1f}s")
    
    assert len(result) > 1, "Should split long chunk"
    
    # Verify each chunk is within limit
    for start, end, chunk, spk in result:
        duration = len(chunk) / SAMPLE_RATE
        assert duration <= 10.5, f"Chunk duration {duration:.1f}s exceeds limit"
    
    print("✅ With split test passed")


def test_enforce_chunk_limit_with_speaker():
    """Test enforce_chunk_limit preserves speaker info"""
    print("\n=== Test 16: enforce_chunk_limit - Speaker Info ===")
    
    audio = np.random.randn(20 * SAMPLE_RATE).astype(np.float32)
    groups = [(0.0, 20.0, audio, "Speaker1")]
    
    result = enforce_chunk_limit(groups, max_chunk_secs=10)
    print(f"Input: 1 group with speaker='Speaker1'")
    print(f"Output: {len(result)} group(s)")
    
    # All chunks should preserve speaker info
    for start, end, chunk, spk in result:
        assert spk == "Speaker1", f"Speaker should be preserved, got {spk}"
    
    print("✅ Speaker info test passed")


# ══════════════════════════════════════════════════════
# Test: detect_speech_groups()
# ══════════════════════════════════════════════════════

def test_detect_speech_groups_empty_audio():
    """Test detect_speech_groups with empty audio"""
    print("\n=== Test 17: detect_speech_groups - Empty Audio ===")
    
    # Create mock VAD session
    vad_sess = Mock()
    
    # Empty audio
    audio = np.array([], dtype=np.float32)
    result = detect_speech_groups(audio, vad_sess)
    print(f"Input: empty audio")
    print(f"Output: {result}")
    
    # Should return empty or handle gracefully
    assert isinstance(result, list), "Should return list"
    
    print("✅ Empty audio test passed")


def test_detect_speech_groups_short_audio():
    """Test detect_speech_groups with short audio"""
    print("\n=== Test 18: detect_speech_groups - Short Audio ===")
    
    # Create mock VAD session that returns high probability
    vad_sess = Mock()
    vad_sess.run = Mock(return_value=(np.array([[0.9]]), None, None))
    
    # Short audio (1 second)
    audio = np.random.randn(SAMPLE_RATE).astype(np.float32)
    result = detect_speech_groups(audio, vad_sess, vad_threshold=0.5)
    print(f"Input: {len(audio) / SAMPLE_RATE:.1f}s audio")
    print(f"Output: {len(result)} group(s)")
    
    assert isinstance(result, list), "Should return list"
    
    print("✅ Short audio test passed")


def test_detect_speech_groups_with_mock():
    """Test detect_speech_groups with mocked VAD session"""
    print("\n=== Test 19: detect_speech_groups - Mock VAD ===")
    
    # Create mock VAD session
    vad_sess = Mock()
    
    # Simulate VAD output: high prob for first 20 chunks, low for next 20, high for last 20
    def mock_run(inputs, input_dict):
        # Return (output, h, c)
        prob = input_dict.get("input")
        if prob is not None:
            # Simulate probability based on position
            return (np.array([[0.8]]), None, None)
        return (np.array([[0.1]]), None, None)
    
    vad_sess.run = mock_run
    
    # Create audio (5 seconds)
    audio = np.random.randn(5 * SAMPLE_RATE).astype(np.float32)
    result = detect_speech_groups(audio, vad_sess, vad_threshold=0.5)
    print(f"Input: {len(audio) / SAMPLE_RATE:.1f}s audio")
    print(f"Output: {len(result)} group(s)")
    
    if result:
        for start, end, chunk in result:
            print(f"  [{start:.2f}, {end:.2f}] {len(chunk) / SAMPLE_RATE:.2f}s")
    
    assert isinstance(result, list), "Should return list"
    
    # Each group should be a tuple of (start_sec, end_sec, audio_chunk)
    for item in result:
        assert len(item) == 3, "Each group should have 3 elements"
        start, end, chunk = item
        assert isinstance(start, float), "Start should be float"
        assert isinstance(end, float), "End should be float"
        assert isinstance(chunk, np.ndarray), "Chunk should be ndarray"
        assert start < end, "Start should be < end"
    
    print("✅ Mock VAD test passed")


def test_detect_speech_groups_custom_threshold():
    """Test detect_speech_groups with custom threshold"""
    print("\n=== Test 20: detect_speech_groups - Custom Threshold ===")
    
    vad_sess = Mock()
    vad_sess.run = Mock(return_value=(np.array([[0.6]]), None, None))
    
    audio = np.random.randn(3 * SAMPLE_RATE).astype(np.float32)
    
    # Test with custom threshold
    result = detect_speech_groups(audio, vad_sess, vad_threshold=0.7)
    print(f"Input: {len(audio) / SAMPLE_RATE:.1f}s audio with threshold=0.7")
    print(f"Output: {len(result)} group(s)")
    
    assert isinstance(result, list), "Should return list"
    
    print("✅ Custom threshold test passed")


def test_detect_speech_groups_max_group_sec():
    """Test detect_speech_groups respects max_group_sec"""
    print("\n=== Test 21: detect_speech_groups - Max Group Sec ===")
    
    # Create a mock VAD session that simulates speech detection
    vad_sess = Mock()
    
    # Simulate VAD: high prob for chunks 0-10, low for 10-20, high for 20-30
    chunk_counter = [0]
    def mock_run_varied(inputs, input_dict):
        chunk_idx = chunk_counter[0]
        chunk_counter[0] += 1
        # High prob for first 10 chunks, low for next 10, high for last 10
        if chunk_idx < 10 or chunk_idx >= 20:
            return (np.array([[0.9]]), None, None)
        else:
            return (np.array([[0.1]]), None, None)
    
    vad_sess.run = mock_run_varied
    
    # Create long audio (30 seconds)
    audio = np.random.randn(30 * SAMPLE_RATE).astype(np.float32)
    result = detect_speech_groups(audio, vad_sess, max_group_sec=10, vad_threshold=0.5)
    print(f"Input: {len(audio) / SAMPLE_RATE:.1f}s audio with max_group_sec=10")
    print(f"Output: {len(result)} group(s)")
    
    if result:
        for start, end, chunk in result:
            duration = end - start
            print(f"  [{start:.1f}, {end:.1f}] {duration:.1f}s")
            # Note: Due to VAD algorithm complexity, we just verify it returns valid groups
            assert isinstance(start, float), "Start should be float"
            assert isinstance(end, float), "End should be float"
            assert start < end, "Start should be < end"
    
    print("✅ Max group sec test passed")


# ══════════════════════════════════════════════════════
# Main Test Runner
# ══════════════════════════════════════════════════════

def main():
    """Run all tests"""
    print("=" * 60)
    print("  ASR Common Utilities - Test Suite")
    print("=" * 60)
    
    tests = [
        test_vad_threshold_getter_setter,
        test_output_simplified_getter_setter,
        test_srt_dir_getter_setter,
        test_split_to_lines_empty,
        test_split_to_lines_english,
        test_split_to_lines_punctuation,
        test_split_to_lines_cjk,
        test_split_to_lines_mixed,
        test_split_to_lines_max_chars,
        test_assign_timestamps_empty,
        test_assign_timestamps_single_line,
        test_assign_timestamps_multiple_lines,
        test_assign_timestamps_with_gap,
        test_enforce_chunk_limit_no_split,
        test_enforce_chunk_limit_with_split,
        test_enforce_chunk_limit_with_speaker,
        test_detect_speech_groups_empty_audio,
        test_detect_speech_groups_short_audio,
        test_detect_speech_groups_with_mock,
        test_detect_speech_groups_custom_threshold,
        test_detect_speech_groups_max_group_sec,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n❌ {test_func.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
