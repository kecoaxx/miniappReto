@echo off
cls

echo Starting the application setup...

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies from requirements.txt...
pip install -r requirements.txt

echo Starting the application...
python app.py

echo Application stopped. Press any key to close.
pause
