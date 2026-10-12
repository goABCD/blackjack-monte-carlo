@echo off
cd /d "%~dp0"
if not exist .venv (
  python -m venv .venv
  call .venv\Scripts\activate.bat
  python -m pip install -U pip
  pip install -r requirements.txt
) else (
  call .venv\Scripts\activate.bat
)
echo.
echo Opening Math IA Lab at http://localhost:8501
echo.
streamlit run app\streamlit_app.py
pause
