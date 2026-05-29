@echo off
setlocal

set "PROJECT_ROOT=%~dp0"
set "VENV_PYTHON=%PROJECT_ROOT%.venv\Scripts\python.exe"

if exist "%VENV_PYTHON%" (
    "%VENV_PYTHON%" -m app.api.evaluation.pipeline.evaluate %*
) else (
    python -m app.api.evaluation.pipeline.evaluate %*
)
