@echo off
setlocal
cd /d "%~dp0.." || goto fail

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found. Run scripts\setup.bat first.
    goto fail
)

set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
set STREAMLIT_SERVER_SHOW_EMAIL_PROMPT=false
set STREAMLIT_SERVER_MAX_UPLOAD_SIZE=102400
set STREAMLIT_SERVER_MAX_MESSAGE_SIZE=102400
set "PATH=%CD%\.venv\Scripts;%PATH%"
".venv\Scripts\python.exe" -m streamlit run app.py || goto fail

set "EXIT_CODE=0"
goto finish

:fail
echo.
echo Launch failed.
set "EXIT_CODE=1"
goto finish

:finish
echo.
pause
endlocal & exit /b %EXIT_CODE%
