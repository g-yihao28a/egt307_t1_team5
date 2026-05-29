@echo off
:: Activate virtual environment
call .venv\Scripts\activate

:: Run pipeline
python src\main.py

:: Keep window open
pause