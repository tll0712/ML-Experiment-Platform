#!/usr/bin/env python3
"""
机器学习实验平台启动脚本
Machine Learning Experiment Platform Launcher
"""

import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# 导入并运行应用
from app import app, socketio

if __name__ == '__main__':
    print("🚀 启动机器学习实验平台...")
    print("📊 Machine Learning Experiment Platform")
    print("🌐 访问地址: http://localhost:5050")
    print("⚡ 按 Ctrl+C 停止服务")
    print("-" * 50)
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5050, allow_unsafe_werkzeug=True)
