#!/bin/bash
# 机器学习实验平台部署脚本
# Machine Learning Experiment Platform Deployment Script

echo "🚀 机器学习实验平台部署脚本"
echo "📊 Machine Learning Experiment Platform Deployment"
echo "=" * 50

# 检查Python版本
echo "🔍 检查Python版本..."
python3 --version

# 安装依赖
echo "📦 安装依赖包..."
pip3 install -r requirements.txt

# 创建必要目录
echo "📁 创建必要目录..."
mkdir -p assets/uploads
mkdir -p logs

# 设置权限
echo "🔐 设置文件权限..."
chmod +x run.py

# 启动服务
echo "🚀 启动服务..."
echo "访问地址: http://localhost:5050"
echo "按 Ctrl+C 停止服务"
echo "=" * 50

python3 run.py
