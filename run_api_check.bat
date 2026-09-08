@echo off
chcp 65001 >nul
set "SCRIPT_DIR=%~dp0"
set /p "API_URL=请输入 API 地址（根地址、/v1 或完整接口，直接回车使用 https://api.openai.com/v1）："
if "%API_URL%"=="" set "API_URL=https://api.openai.com/v1"
set /p "MODEL=请输入模型名称（例如 gpt-4o）："
if "%MODEL%"=="" set "MODEL=gpt-4o"
python "%SCRIPT_DIR%model_route_consistency.py" --api-url "%API_URL%" --api-key "%API_KEY%" --model "%MODEL%" --mode standard --log-file "%USERPROFILE%\Documents\api-check.log"
pause
