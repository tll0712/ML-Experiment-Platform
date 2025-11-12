#!/bin/bash
#
# 机器学习实验平台部署脚本
# Machine Learning Experiment Platform Deployment Script
#
# 使用方法：
#   chmod +x deploy.sh
#   ./deploy.sh [环境] [端口]
#
# 示例：
#   ./deploy.sh development 5050
#   ./deploy.sh production 8080
#

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查命令是否存在
check_command() {
    if ! command -v $1 &> /dev/null; then
        print_error "$1 未安装，请先安装 $1"
        exit 1
    fi
}

# 检查Python版本
check_python() {
    print_info "检查Python版本..."
    if command -v python3 &> /dev/null; then
        PYTHON_CMD=python3
    elif command -v python &> /dev/null; then
        PYTHON_CMD=python
    else
        print_error "Python未安装，请先安装Python 3.8+"
        exit 1
    fi
    
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    print_success "Python版本: $PYTHON_VERSION"
    
    # 检查Python版本是否>=3.8
    REQUIRED_VERSION="3.8"
    if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
        print_error "Python版本需要 >= 3.8，当前版本: $PYTHON_VERSION"
        exit 1
    fi
}

# 检查依赖
check_dependencies() {
    print_info "检查依赖..."
    
    if [ ! -f "requirements.txt" ]; then
        print_error "requirements.txt 文件不存在"
        exit 1
    fi
    
    print_success "依赖文件检查通过"
}

# 创建虚拟环境
create_venv() {
    print_info "创建虚拟环境..."
    
    if [ -d "venv" ]; then
        print_warning "虚拟环境已存在，跳过创建"
    else
        $PYTHON_CMD -m venv venv
        print_success "虚拟环境创建成功"
    fi
}

# 激活虚拟环境
activate_venv() {
    print_info "激活虚拟环境..."
    
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        print_success "虚拟环境已激活"
    elif [ -f "venv/Scripts/activate" ]; then
        source venv/Scripts/activate
        print_success "虚拟环境已激活"
    else
        print_error "无法找到虚拟环境激活脚本"
        exit 1
    fi
}

# 安装依赖
install_dependencies() {
    print_info "安装依赖包..."
    
    # 升级pip
    pip install --upgrade pip > /dev/null 2>&1
    
    # 安装依赖
    if pip install -r requirements.txt; then
        print_success "依赖安装成功"
    else
        print_error "依赖安装失败"
        exit 1
    fi
}

# 创建必要的目录
create_directories() {
    print_info "创建必要的目录..."
    
    mkdir -p src/datasets
    mkdir -p logs
    mkdir -p assets/uploads
    
    print_success "目录创建成功"
}

# 设置环境变量
set_environment() {
    local ENV=$1
    local PORT=${2:-5050}
    
    print_info "设置环境变量..."
    
    export FLASK_ENV=$ENV
    export FLASK_APP=run.py
    export PORT=$PORT
    
    print_success "环境变量设置完成: ENV=$ENV, PORT=$PORT"
}

# 检查端口是否被占用
check_port() {
    local PORT=$1
    
    print_info "检查端口 $PORT 是否被占用..."
    
    if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        print_warning "端口 $PORT 已被占用"
        read -p "是否要终止占用该端口的进程? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            lsof -ti:$PORT | xargs kill -9
            print_success "已终止占用端口的进程"
        else
            print_error "请手动释放端口或使用其他端口"
            exit 1
        fi
    else
        print_success "端口 $PORT 可用"
    fi
}

# 运行测试
run_tests() {
    print_info "运行测试..."
    
    if [ -f "test_hyperparameter_tuning.py" ]; then
        if python test_hyperparameter_tuning.py; then
            print_success "测试通过"
        else
            print_warning "测试失败，但继续部署"
        fi
    else
        print_warning "测试文件不存在，跳过测试"
    fi
}

# 启动服务
start_server() {
    local ENV=$1
    local PORT=$2
    
    print_info "启动服务器..."
    print_info "环境: $ENV"
    print_info "端口: $PORT"
    print_info "访问地址: http://localhost:$PORT"
    
    if [ "$ENV" == "production" ]; then
        print_info "生产环境模式"
        # 生产环境可以使用gunicorn或uwsgi
        # gunicorn -w 4 -b 0.0.0.0:$PORT --worker-class eventlet -k eventlet "src.app:app"
        python run.py
    else
        print_info "开发环境模式"
        python run.py
    fi
}

# 主函数
main() {
    echo "=========================================="
    echo "  机器学习实验平台部署脚本"
    echo "  Machine Learning Experiment Platform"
    echo "=========================================="
    echo
    
    # 解析参数
    ENV=${1:-development}
    PORT=${2:-5050}
    
    # 验证环境参数
    if [ "$ENV" != "development" ] && [ "$ENV" != "production" ]; then
        print_error "无效的环境参数: $ENV (应为 development 或 production)"
        exit 1
    fi
    
    # 执行部署步骤
    check_python
    check_dependencies
    create_venv
    activate_venv
    install_dependencies
    create_directories
    set_environment $ENV $PORT
    check_port $PORT
    
    # 可选：运行测试
    read -p "是否运行测试? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        run_tests
    fi
    
    echo
    print_success "部署准备完成！"
    echo
    print_info "按 Ctrl+C 停止服务器"
    echo
    
    # 启动服务器
    start_server $ENV $PORT
}

# 运行主函数
main "$@"

