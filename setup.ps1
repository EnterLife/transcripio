$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

& ".\.venv\Scripts\python.exe" -m pip install -U pip
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
& ".\.venv\Scripts\python.exe" -m pip install -e . --no-deps

Write-Host ""
Write-Host "Done. To launch:"
Write-Host ".\.venv\Scripts\Activate.ps1"
Write-Host "streamlit run app.py"
