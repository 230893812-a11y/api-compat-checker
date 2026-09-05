# API 模型路由一致性检查项目

## 功能

- 检查 API 返回的 `model` 是否与请求模型完全一致
- 支持批量模型和多轮测试
- 支持 HTTP/HTTPS 代理、SSL 验证开关和 30 秒超时
- 终端彩色输出，日志保存为 UTF-8
- 失败时返回退出码 `1`，成功时返回 `0`

## 安装

```cmd
python -m pip install -r requirements.txt
```

## 运行

建议使用环境变量保存 API Key：

```cmd
set API_KEY=你的真实API_KEY
python model_route_consistency.py --api-url "https://你的服务/v1/chat/completions" --model "gpt-5.6-sol,gpt-5.6-terra,gpt-5.6-luna,gpt-5.6-astra" --rounds 1 --log-file "%USERPROFILE%\Documents\api-check.log"
```

也可以双击 `run_api_check.bat`（运行前先设置 `API_KEY`）。

## 任务计划程序

将 `run_api_check.bat` 设置为“启动时”运行即可。日志建议写入用户的 Documents 目录，不要写入 `C:\` 根目录，以避免权限错误。

仅对你拥有或明确获授权的 API 进行检查；不要在日志、截图或问题报告中暴露 API Key。
