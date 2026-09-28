@echo off
REM Starts the demo using your existing conda environment and local weights.
REM Edit the two lines below if your paths differ.
set MODEL_PATH=D:\DS\my Jupyter Code for Learning\605 final\christchurch\mod_pt\optimised_config.pt
call conda activate building-demo
cd /d "%~dp0"
python app.py
pause
