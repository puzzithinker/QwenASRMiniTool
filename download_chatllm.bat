@echo off
REM ============================================================================
REM chatllm DLL Download and Setup Script for Windows
REM ============================================================================
REM This script helps set up the required chatllm DLLs for GPU/Vulkan support
REM ============================================================================

setlocal enabledelayedexpansion

echo.
echo ============================================================================
echo  chatllm DLL Setup Helper
echo ============================================================================
echo.
echo This script will help you set up the required DLLs for chatllm.cpp
echo with GPU/Vulkan support.
echo.

REM Create chatllm directory
echo [1/3] Creating chatllm directory...
if not exist "chatllm" (
    mkdir chatllm
    echo Created: chatllm\
) else (
    echo Directory already exists: chatllm\
)

echo.
echo ============================================================================
echo [2/3] Required DLL Files
echo ============================================================================
echo.
echo The following files need to be placed in the 'chatllm\' folder:
echo.
echo CORE LIBRARIES:
echo   - libchatllm.dll          (Main chatllm library)
echo   - ggml.dll or ggml-base.dll (GGML base library)
echo.
echo GPU/VULKAN SUPPORT:
echo   - ggml-vulkan.dll         (Vulkan GPU acceleration)
echo   - vulkan-1.dll            (Vulkan runtime library)
echo.
echo CPU VARIANTS (optional, for CPU fallback):
echo   - ggml-cpu-avx.dll        (AVX support)
echo   - ggml-cpu-avx2.dll       (AVX2 support)
echo   - ggml-cpu-avx512.dll     (AVX512 support)
echo.
echo EXECUTABLE:
echo   - main.exe                (chatllm executable)
echo.

echo ============================================================================
echo [3/3] How to Obtain the DLLs
echo ============================================================================
echo.
echo OPTION A: Build from Source (Recommended for latest features)
echo ============================================================================
echo.
echo 1. Clone the repository:
echo    git clone https://github.com/foldl/chatllm.cpp.git
echo    cd chatllm.cpp
echo.
echo 2. Install dependencies:
echo    - CMake (https://cmake.org/download/)
echo    - Visual Studio Build Tools or MSVC compiler
echo    - Vulkan SDK (https://www.lunarg.com/vulkan-sdk/)
echo.
echo 3. Build with Vulkan support:
echo    mkdir build
echo    cd build
echo    cmake .. -G "Visual Studio 17 2022" -DGGML_VULKAN=ON
echo    cmake --build . --config Release
echo.
echo 4. Copy the built DLLs:
echo    - From: build\bin\Release\
echo    - To: chatllm\ folder in this project
echo.
echo OPTION B: Download Pre-built Binaries
echo ============================================================================
echo.
echo Check the chatllm.cpp releases page for pre-built binaries:
echo https://github.com/foldl/chatllm.cpp/releases
echo.
echo Look for releases that include:
echo   - Windows binaries
echo   - Vulkan support enabled
echo   - All required DLLs included
echo.
echo Extract the DLLs and place them in the chatllm\ folder.
echo.

echo ============================================================================
echo Setup Instructions Summary
echo ============================================================================
echo.
echo 1. Obtain the DLLs using Option A or B above
echo.
echo 2. Place all DLLs in: %cd%\chatllm\
echo.
echo 3. Verify the following files exist:
echo    - chatllm\libchatllm.dll
echo    - chatllm\ggml.dll (or ggml-base.dll)
echo    - chatllm\ggml-vulkan.dll
echo    - chatllm\vulkan-1.dll
echo    - chatllm\main.exe
echo.
echo 4. Once all files are in place, the application should be able to:
echo    - Load the chatllm library
echo    - Use GPU acceleration via Vulkan
echo    - Fall back to CPU if GPU is unavailable
echo.

echo ============================================================================
echo Useful Links
echo ============================================================================
echo.
echo chatllm.cpp Repository:
echo   https://github.com/foldl/chatllm.cpp
echo.
echo Vulkan SDK Download:
echo   https://www.lunarg.com/vulkan-sdk/
echo.
echo CMake Download:
echo   https://cmake.org/download/
echo.

echo ============================================================================
echo Current Status
echo ============================================================================
echo.
if exist "chatllm" (
    echo [OK] chatllm\ directory exists
    echo.
    echo Files currently in chatllm\ folder:
    if exist "chatllm\*" (
        dir /b chatllm\
    ) else (
        echo (empty - awaiting DLL files)
    )
) else (
    echo [ERROR] chatllm\ directory could not be created
)

echo.
echo ============================================================================
echo Setup Complete
echo ============================================================================
echo.
echo Next steps:
echo 1. Download or build the required DLLs
echo 2. Place them in the chatllm\ folder
echo 3. Run your application
echo.
echo For issues or questions, visit:
echo https://github.com/foldl/chatllm.cpp/issues
echo.

pause
