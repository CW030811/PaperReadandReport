@echo off
setlocal

if "%~1"=="" (
  echo Usage: bootstrap_ppt_workspace.cmd ^<paper_dir^>
  exit /b 1
)

set "PAPER_DIR=%~1"
set "SKILL_DIR=%~dp0.."
set "PPT_DIR=%PAPER_DIR%\05_slides\ppt"
set "SRC_DIR=%PPT_DIR%\src"
set "ASSET_DIR=%PPT_DIR%\assets"
set "OUTPUT_DIR=%PPT_DIR%\output"
set "RENDER_DIR=%PPT_DIR%\rendered"
set "TEMPLATE_FILE=%SKILL_DIR%\assets\deck-starter\research_deck_template.js"
set "TARGET_FILE=%SRC_DIR%\build_deck.js"

if not exist "%PAPER_DIR%" (
  echo Paper directory not found: %PAPER_DIR%
  exit /b 1
)

mkdir "%SRC_DIR%" 2>nul
mkdir "%ASSET_DIR%" 2>nul
mkdir "%OUTPUT_DIR%" 2>nul
mkdir "%RENDER_DIR%" 2>nul

if exist "%TARGET_FILE%" (
  echo Target already exists: %TARGET_FILE%
  exit /b 0
)

copy "%TEMPLATE_FILE%" "%TARGET_FILE%" >nul
if errorlevel 1 (
  echo Failed to copy template to %TARGET_FILE%
  exit /b 1
)

echo Bootstrapped PPT workspace at %PPT_DIR%
exit /b 0
