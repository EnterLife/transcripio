@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv || exit /b 1
)

".venv\Scripts\python.exe" -m pip install -U pip || exit /b 1
".venv\Scripts\python.exe" -m pip install -r requirements.txt || exit /b 1
".venv\Scripts\python.exe" -m pip install -e . --no-deps || exit /b 1

echo.
echo Done. To launch:
echo run.bat

endlocal
