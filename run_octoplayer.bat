@echo off
rem Launch Octoplayer without a console window.
cd /d "%~dp0"
start "" pythonw main.py %*
