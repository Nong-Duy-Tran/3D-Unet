@echo off

cd /d "d:\YEAR 3\Lab\3D-Unet"

set PYTHON=C:\Users\Lenovo\miniconda3\envs\alzheimer-baseline\python.exe
set TRAIN=d:\YEAR 3\Lab\3D-Unet\src\2d\train.py

for %%F in (0 1 2 3 4) do (
    echo ========== Training fold %%F ==========
    "%PYTHON%" "%TRAIN%" --fold %%F
    echo ========== Fold %%F done ==========
)

echo All folds completed.
pause
