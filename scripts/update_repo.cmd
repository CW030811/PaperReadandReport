@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0update_repo.ps1" %*
