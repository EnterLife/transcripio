@echo off
setlocal
cd /d "%~dp0.." || goto fail

if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv || goto fail
)

".venv\Scripts\python.exe" -m pip install -U pip || goto fail
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto fail
".venv\Scripts\python.exe" scripts\install_ffmpeg.py || goto fail
".venv\Scripts\python.exe" -m pip install -e . --no-deps || goto fail
".venv\Scripts\python.exe" scripts\install_cuda_runtime_if_needed.py || goto fail

echo.
echo Done. To launch:
echo scripts\run_desktop.bat
echo or:
echo scripts\run.bat
set "EXIT_CODE=0"
goto finish

:fail
echo.
echo Setup failed.
set "EXIT_CODE=1"
goto finish

:finish
echo.
pause
endlocal & exit /b %EXIT_CODE%
