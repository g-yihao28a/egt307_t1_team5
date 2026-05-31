@echo off

:: Create the virtual environment if it does not exist
if not exist ".venv" (
    python -m venv venv
)

:: Activate virtual environment
call .venv\Scripts\activate

:: Run pipeline
python src\main.py

:: Keep window open
pause