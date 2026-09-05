@echo off
chcp 65001 >nul
set "SCRIPT_DIR=%~dp0"
python "%SCRIPT_DIR%model_route_consistency.py" --api-url "https://www.678api.ai/v1/chat/completions" --api-key "%API_KEY%" --model "gpt-5.6-sol,gpt-5.6-terra,gpt-5.6-luna,gpt-5.6-astra" --log-file "%USERPROFILE%\Documents\api-check.log"
pause
