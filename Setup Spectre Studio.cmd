@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto wrong_folder
if not exist ".venv\Scripts\pythonw.exe" goto wrong_folder
if not exist "study_chat.py" goto wrong_folder
if not exist "study_chat.html" goto wrong_folder
if not exist "study_app.pyw" goto wrong_folder
if not exist "spectre-studio.ico" goto wrong_folder
echo Installing pywebview in the existing rag-study environment...
".venv\Scripts\python.exe" -m pip install pywebview
if errorlevel 1 goto failed
set "STUDY_CHAT_ROOT=%CD%"
powershell -NoProfile -Command "$ErrorActionPreference='Stop'; $root=$env:STUDY_CHAT_ROOT; $desktop=[Environment]::GetFolderPath('DesktopDirectory'); if (!$desktop) { throw 'Windows did not return a desktop path.' }; $shell=New-Object -ComObject WScript.Shell; $shortcut=$shell.CreateShortcut((Join-Path $desktop 'Spectre Studio.lnk')); $shortcut.TargetPath=(Join-Path $root '.venv\Scripts\pythonw.exe'); $shortcut.Arguments=[string][char]34+(Join-Path $root 'study_app.pyw')+[char]34; $shortcut.WorkingDirectory=$root; $shortcut.IconLocation=(Join-Path $root 'spectre-studio.ico')+',0'; $shortcut.Description='Open the local Spectre Studio app'; $shortcut.Save()"
if errorlevel 1 goto failed
echo.
echo Setup complete. Double-click Spectre Studio on your Windows desktop.
pause
exit /b 0

:wrong_folder
echo Copy the three desktop files into your existing rag-study folder
echo next to study_chat.py and study_chat.html, then try again.
pause
exit /b 1

:failed
echo.
echo Setup did not finish. Check the error above.
pause
exit /b 1
