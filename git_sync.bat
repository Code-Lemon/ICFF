@echo off
chcp 65001 >nul

:: 设置项目目录
cd /d E:\Code\mmdetection-main

:: 检查网络连接
ping github.com -n 1 >nul
if errorlevel 1 (
    echo ❌ 无法连接 GitHub，请检查网络或开启代理！
    pause
    exit /b
)

:: 显示当前 Git 分支状态
git status

:: 添加变更
echo ✅ 正在添加文件...
git add .

:: 输入提交说明
set /p msg=请输入本次提交说明： 
git commit -m "%msg%"

:: 拉取远程更新
echo 🔄 正在拉取远程内容以避免冲突...
git pull origin main --allow-unrelated-histories

:: 推送到远程仓库
echo 🚀 正在推送到远程仓库...
git push origin main

echo ✅ 同步完成！
pause