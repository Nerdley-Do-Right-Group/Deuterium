@echo off
REM Convenience wrapper for Windows Command Prompt (cmd.exe) users.
REM Simply delegates to setup.ps1 so we don't maintain two identical Windows scripts.
powershell -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
exit /b %ERRORLEVEL%
