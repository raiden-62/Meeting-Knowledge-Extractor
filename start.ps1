$ErrorActionPreference = "Stop"

$VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$PythonCommand = $null

if (Test-Path $VenvPython) {
    $PythonCommand = $VenvPython
} else {
    $PythonCommand = "python"
}

$Version = & $PythonCommand -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"

if ($Version -ne "3.12") {
    Write-Error "Нужен Python 3.12, сейчас используется Python $Version. Создайте .venv на Python 3.12 и запустите скрипт снова."
}

& $PythonCommand -m pip install -r "$PSScriptRoot\requirements.txt"
& $PythonCommand -m uvicorn app.main:app --reload
