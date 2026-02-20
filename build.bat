@echo off
REM =======================================================
REM  Qwen3 ASR - PyInstaller Build Script (onedir mode)
REM
REM  OUTPUT STRUCTURE:
REM    dist\QwenASR\
REM      QwenASR.exe        <- launcher (~5 MB)
REM      prompt_template.json
REM      _internal\         <- Python runtime + packages
REM
REM  DISTRIBUTION:
REM    Run setup.iss with Inno Setup to produce
REM    QwenASR_Setup.exe (~400 MB installer).
REM    Models (~1.2 GB) are downloaded at first run.
REM
REM  STARTUP TIME:
REM    onedir  -> 3-5 s  (DLLs loaded directly)
REM    onefile -> 20-35 s (must extract to %%TEMP%% first)
REM =======================================================

REM Use build_venv (no torch) for smaller output.
REM Run build_venv.bat first if build_venv\ doesn't exist.
IF EXIST "F:\AIStudio\QwenASR\build_venv\Scripts\python.exe" (
    SET VENV=F:\AIStudio\QwenASR\build_venv
) ELSE (
    SET VENV=F:\AIStudio\QwenASR\venv
)
SET PYTHON=%VENV%\Scripts\python.exe
SET SRC=F:\AIStudio\QwenASR

echo === Step 1: Install PyInstaller ===
%PYTHON% -m pip install pyinstaller --quiet

echo.
echo === Step 2: Locate dependency paths ===

FOR /F "delims=" %%i IN ('%PYTHON% -c "import opencc, os; print(os.path.dirname(opencc.__file__))"') DO SET OPENCC_DIR=%%i
FOR /F "delims=" %%i IN ('%PYTHON% -c "import customtkinter, os; print(os.path.dirname(customtkinter.__file__))"') DO SET CTK_DIR=%%i
FOR /F "delims=" %%i IN ('%PYTHON% -c "import openvino, os; print(os.path.dirname(openvino.__file__))"') DO SET OV_PKG=%%i
FOR /F "delims=" %%i IN ('%PYTHON% -c "import kaldi_native_fbank, os; print(os.path.dirname(kaldi_native_fbank.__file__))"') DO SET KNF_DIR=%%i

echo opencc            : %OPENCC_DIR%
echo customtkinter     : %CTK_DIR%
echo openvino          : %OV_PKG%
echo kaldi_native_fbank: %KNF_DIR%

echo.
echo === Step 3: Build with PyInstaller (onedir) ===

REM --onedir is the DEFAULT (no --onefile flag).
REM _internal/ keeps the root folder tidy (PyInstaller >= 6.0).
REM
REM prompt_template.json and mel_filters.npy are bundled inside _internal/
REM so LightProcessor can find them via Path(__file__).parent fallback.
REM
REM runtime_hook_utf8.py: sets PYTHONUTF8=1 before any user code runs.
REM This prevents "utf-8 codec can't decode byte 0xa6" on Traditional
REM Chinese Windows (cp950 default encoding).

%PYTHON% -m PyInstaller ^
    --onedir ^
    --windowed ^
    --name "QwenASR" ^
    --icon NONE ^
    --add-data "%CTK_DIR%;customtkinter" ^
    --add-data "%OPENCC_DIR%;opencc" ^
    --add-data "%OV_PKG%;openvino" ^
    --add-data "%KNF_DIR%;kaldi_native_fbank" ^
    --add-data "%SRC%\prompt_template.json;." ^
    --add-data "%SRC%\ov_models\mel_filters.npy;ov_models" ^
    --runtime-hook "%SRC%\runtime_hook_utf8.py" ^
    --hidden-import openvino ^
    --hidden-import openvino.runtime ^
    --hidden-import onnxruntime ^
    --hidden-import opencc ^
    --hidden-import customtkinter ^
    --hidden-import sounddevice ^
    --hidden-import librosa ^
    --hidden-import kaldi_native_fbank ^
    --hidden-import scipy ^
    --hidden-import scipy.cluster ^
    --hidden-import scipy.cluster.hierarchy ^
    --hidden-import scipy.spatial ^
    --hidden-import scipy.spatial.distance ^
    --hidden-import scipy._lib.messagestream ^
    --exclude-module torch ^
    --exclude-module torchvision ^
    --exclude-module torchaudio ^
    --exclude-module transformers ^
    --exclude-module qwen_asr ^
    --exclude-module triton ^
    --exclude-module bitsandbytes ^
    --noconfirm ^
    %SRC%\app.py

echo.
IF EXIST "%SRC%\dist\QwenASR\QwenASR.exe" (
    echo ===================================================
    echo  Build SUCCESS
    echo  Launcher : dist\QwenASR\QwenASR.exe
    echo  Runtime  : dist\QwenASR\_internal\
    echo.
    echo  Next step: open setup.iss with Inno Setup
    echo  to produce QwenASR_Setup.exe for distribution.
    echo ===================================================
) ELSE (
    echo ===================================================
    echo  Build FAILED. Check output above.
    echo ===================================================
)
pause
