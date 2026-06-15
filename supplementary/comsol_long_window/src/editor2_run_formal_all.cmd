@echo off
setlocal enabledelayedexpansion

if "%ROOT%"=="" set ROOT=%cd%
if "%PY%"=="" set PY=python
set SRC=%ROOT%\src\editor2_train_mipinn_pinn.py
set LOGDIR=%ROOT%\formal_logs
set OUTROOT=%ROOT%\formal_runs
set MASTER=%LOGDIR%\formal_master.log

if not exist "%LOGDIR%" mkdir "%LOGDIR%"
if not exist "%OUTROOT%" mkdir "%OUTROOT%"

echo START %date% %time% > "%MASTER%"
echo ROOT=%ROOT% >> "%MASTER%"
echo OUTROOT=%OUTROOT% >> "%MASTER%"
echo SETTINGS probe=2000 nobs=4000 nf=20000 nm=1140 pinn_epochs=20000 mamba_epochs=3000 >> "%MASTER%"

set COMMON=--probe-count 2000 --epochs-pinn 20000 --epochs-mamba 3000 --batch-data 4000 --batch-phys 20000 --batch-pseudo 1140 --mamba-batch 256 --report-every 200 --eval-batch 65536 --seed 20260611 --out-root "%OUTROOT%"

call :run_case single pinn
if errorlevel 1 exit /b %errorlevel%
call :run_case single mipinn
if errorlevel 1 exit /b %errorlevel%
call :run_case three pinn
if errorlevel 1 exit /b %errorlevel%
call :run_case three mipinn
if errorlevel 1 exit /b %errorlevel%

echo DONE %date% %time% >> "%MASTER%"
exit /b 0

:run_case
set CASE=%1
set VARIANT=%2
set LOG=%LOGDIR%\formal_%CASE%_%VARIANT%.log
set ERR=%LOGDIR%\formal_%CASE%_%VARIANT%.err
echo TASK_START %CASE% %VARIANT% %date% %time% >> "%MASTER%"
"%PY%" "%SRC%" --case %CASE% --variant %VARIANT% %COMMON% > "%LOG%" 2> "%ERR%"
set CODE=%errorlevel%
echo TASK_END %CASE% %VARIANT% exit=%CODE% %date% %time% >> "%MASTER%"
if not "%CODE%"=="0" (
  echo TASK_FAILED %CASE% %VARIANT% see %LOG% and %ERR% >> "%MASTER%"
  exit /b %CODE%
)
exit /b 0
