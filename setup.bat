@echo off
REM setup.bat            full install (project + dev tools)
REM setup.bat runtime    runtime-only install
REM Requires Python 3.11.x (64-bit) on PATH to match the bundled cp311 wheels.

setlocal
cd /d "%~dp0"

set "EXTRAS=.[dev]"
set "MODE=full"
if /i "%~1"=="runtime" (
    set "EXTRAS=."
    set "MODE=runtime"
)
if /i "%~1"=="run" (
    set "EXTRAS=."
    set "MODE=runtime"
)

echo   AMPM Analyzer setup...

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] 'python' was not found on PATH.
    echo         Install Python 3.11.x 64-bit and ensure it is on PATH.
    goto :fail
)

python -c "import sys,struct; sys.exit(0 if (sys.version_info[:2]==(3,11) and struct.calcsize('P')*8==64) else 1)"
if errorlevel 1 (
    echo [ERROR] The bundled wheels are built for CPython 3.11.x, 64-bit.
    echo         The 'python' on PATH does not match. Check with:
    echo             python --version
    goto :fail
)

if not exist "wheels\windows" (
    echo [ERROR] Offline wheel folder 'wheels\windows' not found beside this script.
    echo         Copy it in before running setup.
    goto :fail
)

if exist ".venv" (
    echo Removing existing .venv ...
    rmdir /s /q ".venv"
)
echo Creating virtual environment .venv ...
python -m venv ".venv"
if errorlevel 1 (
    echo [ERROR] Failed to create the virtual environment.
    goto :fail
)

echo Installing dependencies from wheels\windows ^(offline^) ...
".venv\Scripts\python.exe" -m pip install "%EXTRAS%" --no-index --find-links "wheels\windows" --disable-pip-version-check
if errorlevel 1 (
    echo [ERROR] Offline install failed - see pip output above.
    goto :fail
)

echo Verifying core imports ...
".venv\Scripts\python.exe" -c "import ampm, trimesh, shapely, networkx, rtree, polars, numpy, scipy, sklearn, plotly, PyQt6; print('imports OK')"
if errorlevel 1 (
    echo [ERROR] Environment built but a core import failed - see above.
    goto :fail
)

echo.
echo   Setup complete ^(%MODE%^)
echo     Run the app:    .venv\Scripts\python.exe app.py
echo.
pause
endlocal
exit /b 0

:fail
echo.
echo Setup did not complete.
pause
endlocal
exit /b 1