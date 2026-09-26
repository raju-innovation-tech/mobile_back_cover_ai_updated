@echo off
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/check_environment.py
pause
