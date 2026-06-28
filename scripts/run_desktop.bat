@echo off
setlocal
cd /d "%~dp0.." || goto fail

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found. Run scripts\setup.bat first.
    goto fail
)

set "PATH=%CD%\.venv\Scripts;%PATH%"
".venv\Scripts\python.exe" -m transcripio_desktop.app || goto fail

set "EXIT_CODE=0"
goto finish

:fail
echo.
echo Desktop launch failed.
set "EXIT_CODE=1"
goto finish

:finish
echo.
pause
endlocal & exit /b %EXIT_CODE%
