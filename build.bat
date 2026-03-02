@echo off
REM =======================================================
REM  逐字稿神器 - PyInstaller Build Script (onedir mode)
REM =======================================================
REM OUTPUT STRUCTURE:
REM   dist\逐字稿神器\
REM     逐字稿神器.exe     <- launcher (~5 MB)
REM     _internal\         <- Python runtime + packages

SET VENV=build_venv
SET PYTHON=%VENV%\Scripts\python.exe
SET SRC=.

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
echo === Step 2b: Ensure silero_vad_v4.onnx ===
IF NOT EXIST "%SRC%\ov_models\silero_vad_v4.onnx" (
    echo   Downloading VAD model...
    %PYTHON% -c "from downloader import _download_file, _VAD_URL; from pathlib import Path; p=Path(r'%SRC%\ov_models'); p.mkdir(exist_ok=True); _download_file(_VAD_URL, p/'silero_vad_v4.onnx')"
) ELSE (
    echo   VAD model already present.
)

echo.
echo === Step 2c: Check chatllm DLLs ===
SET CHATLLM_READY=0
IF EXIST "%SRC%\chatllm\libchatllm.dll" (
    IF EXIST "%SRC%\chatllm\main.exe" SET CHATLLM_READY=1
)
IF %CHATLLM_READY%==0 (
    echo   chatllm DLLs not found.
    echo   App will download automatically on first run.
    echo   Or manually from: https://github.com/foldl/chatllm.cpp/releases
) ELSE (
    echo   chatllm DLLs present - will be bundled.
)

echo.
echo === Step 3: Build with PyInstaller ===

%PYTHON% -m PyInstaller ^
    --onedir ^
    --windowed ^
    --name "逐字稿神器" ^
    --icon NONE ^
    --add-data "%CTK_DIR%;customtkinter" ^
    --add-data "%OPENCC_DIR%;opencc" ^
    --add-data "%OV_PKG%;openvino" ^
    --add-data "%KNF_DIR%;kaldi_native_fbank" ^
    --add-data "%SRC%\prompt_template.json;." ^
    --add-data "%SRC%\ov_models\mel_filters.npy;ov_models" ^
    --add-data "%SRC%\ov_models\silero_vad_v4.onnx;ov_models" ^
    --runtime-hook "%SRC%\runtime_hook_utf8.py" ^
    --collect-data certifi ^
    --hidden-import certifi ^
    --collect-all tokenizers ^
    --hidden-import openvino ^
    --hidden-import openvino.runtime ^
    --hidden-import onnxruntime ^
    --hidden-import opencc ^
    --hidden-import customtkinter ^
    --hidden-import sounddevice ^
    --hidden-import librosa ^
    --hidden-import soundfile ^
    --hidden-import kaldi_native_fbank ^
    --hidden-import scipy ^
    --hidden-import scipy.cluster ^
    --hidden-import scipy.cluster.hierarchy ^
    --hidden-import scipy.spatial ^
    --hidden-import scipy.spatial.distance ^
    --hidden-import scipy._lib.messagestream ^
    --hidden-import pycparser.lextab ^
    --hidden-import pycparser.yacctab ^
    --hidden-import scipy.special._cdflib ^
    --exclude-module torch ^
    --exclude-module torchvision ^
    --exclude-module torchaudio ^
    --exclude-module transformers ^
    --exclude-module qwen_asr ^
    --exclude-module triton ^
    --exclude-module bitsandbytes ^
    --noconfirm ^
    --add-data "%SRC%\batch_tab.py;." ^
    --add-data "%SRC%\ffmpeg_utils.py;." ^
    --add-data "%SRC%\subtitle_editor.py;." ^
    --add-data "%SRC%\setting.py;." ^
    --add-data "%SRC%\subtitle_formatter.py;." ^
    --add-data "%SRC%\asr_common.py;." ^
    --add-data "%SRC%\engine_base.py;." ^
    %SRC%\app.py

echo.
IF EXIST "%SRC%\dist\逐字稿神器\逐字稿神器.exe" (
    echo ==================================================
    echo  Build SUCCESS
    echo ==================================================

    IF EXIST "%SRC%\chatllm" (
        xcopy "%SRC%\chatllm\*" "%SRC%\dist\逐字稿神器\chatllm\" /Y /Q /E
        echo  chatllm/ DLLs copied.
    )

    IF EXIST "%SRC%\ffmpeg\ffmpeg.exe" (
        IF NOT EXIST "%SRC%\dist\逐字稿神器\ffmpeg\" mkdir "%SRC%\dist\逐字稿神器\ffmpeg\"
        xcopy "%SRC%\ffmpeg\ffmpeg.exe" "%SRC%\dist\逐字稿神器\ffmpeg\" /Y /Q
        echo  ffmpeg/ copied.
    )

    echo.
    echo  Output: dist\逐字稿神器\
    echo  Next: Run setup.iss with Inno Setup to create installer.
) ELSE (
    echo ==================================================
    echo  Build FAILED
    echo ==================================================
)
pause
