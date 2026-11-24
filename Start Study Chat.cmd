@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo Put these files inside your existing rag-study folder, next to build_index.py.
    pause
    exit /b 1
)
if not exist "study_chat.html" (
    echo Extract all three Study Chat files together before launching.
    pause
    exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" "%~dp0study_chat.py"
