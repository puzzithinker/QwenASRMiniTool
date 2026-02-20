# runtime_hook_utf8.py
# PyInstaller runtime hook — runs before ANY user code or imports.
#
# Problem: Traditional Chinese Windows uses cp950 (Big5) as the system
# default encoding. Third-party libraries that call open() without an
# explicit encoding= parameter will use cp950, causing:
#   UnicodeDecodeError: 'utf-8' codec can't decode byte 0xa6 in position 0
# when reading UTF-8 files (e.g. opencc config files).
#
# Fix: Set PYTHONUTF8=1 which is equivalent to `python -X utf8`, making
# all text-mode file I/O default to UTF-8 regardless of system locale.
#
# Note: PYTHONUTF8 must be set before Python interpreter initialization
# to affect the C-level getpreferredencoding() result. PyInstaller runtime
# hooks run early enough for this to take effect.
import os
os.environ["PYTHONUTF8"] = "1"
