@echo off
setlocal
cd /d "%~dp0.." || goto fail

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found. Run scripts\setup.bat first.
    goto fail
)

".venv\Scripts\python.exe" -m pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*" || goto fail

set "EXIT_CODE=0"
goto finish

:fail
echo.
echo CUDA runtime installation failed.
set "EXIT_CODE=1"
goto finish

:finish
echo.
pause
endlocal & exit /b %EXIT_CODE%
