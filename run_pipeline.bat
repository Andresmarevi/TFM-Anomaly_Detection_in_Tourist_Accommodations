@echo off
setlocal
chcp 65001 >nul

set "PROJECT_DIR=%~dp0"
if "%PROJECT_DIR:~-1%"=="\" set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"
set "PYTHON_EXE=%PROJECT_DIR%\venv\Scripts\python.exe"
set "LOG_FILE=%PROJECT_DIR%\pipeline_cron.log"

if not exist "%PYTHON_EXE%" (
    echo [%DATE% %TIME%] ERROR: no existe el interprete del entorno virtual: %PYTHON_EXE% >> "%LOG_FILE%"
    exit /b 1
)

cd /d "%PROJECT_DIR%"
if errorlevel 1 (
    echo [%DATE% %TIME%] ERROR: no se pudo acceder a %PROJECT_DIR% >> "%LOG_FILE%"
    exit /b 1
)

echo ================================================== >> "%LOG_FILE%"
echo INICIO DE EJECUCION AUTOMATICA: %DATE% %TIME% >> "%LOG_FILE%"
echo DIRECTORIO: %PROJECT_DIR% >> "%LOG_FILE%"
echo ================================================== >> "%LOG_FILE%"

set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
set "EXPORT_REPORTS=0"
"%PYTHON_EXE%" "%PROJECT_DIR%\main.py" >> "%LOG_FILE%" 2>&1
set "PIPELINE_EXIT_CODE=%ERRORLEVEL%"

if "%PIPELINE_EXIT_CODE%"=="0" (
    echo EJECUCION FINALIZADA CON EXITO: %DATE% %TIME% >> "%LOG_FILE%"
) else (
    echo ERROR CRITICO EN LA EJECUCION: CODIGO %PIPELINE_EXIT_CODE% - %DATE% %TIME% >> "%LOG_FILE%"
)
echo ================================================== >> "%LOG_FILE%"

endlocal & exit /b %PIPELINE_EXIT_CODE%