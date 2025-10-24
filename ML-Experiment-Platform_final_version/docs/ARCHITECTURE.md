# 机器学习实验平台 - 架构文档

## 📋 项目概述

本项目是一个基于 Flask 的机器学习实验平台，支持多种机器学习算法的训练、评估和对比。平台采用前后端分离架构，提供直观的 Web 界面和强大的后端处理能力。

### 🆕 最新更新
- **全面算法支持**: 传统机器学习、深度学习、无监督学习
- **AI智能推荐**: 基于数据集特性的预处理建议
- **动态界面**: 根据模型类型自动调整UI和评估指标
- **模型特定可视化**: 传统模型显示特征重要性，深度学习显示训练历史
- **完善的错误处理**: 用户友好的异常处理机制

## 🏗️ 系统架构

### 整体架构图
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   前端界面       │     │   后端服务       │   │   数据处理层      │
│   (HTML/CSS/JS) │◄──►│   (Flask)       │◄──►│   (Pandas/NumPy) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   WebSocket     │    │   模型注册表    │    │   缓存系统      │
│   实时通信      │    │   动态加载      │    │   性能优化      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📁 模块划分

### 1. 核心模块 (`app.py`)

#### 1.1 智能特征重要性计算
```python
# 位置: 第745-850行
def get_feature_importance():   # 智能特征重要性
```

**功能说明:**
- 自动处理分类和数值特征
- 使用F-score统计方法
- 支持混合数据类型
- 智能归一化处理

#### 1.2 智能调优策略
```python
# 位置: 第880-920行
def hyperparameter_tuning():    # 超参数调优
```

**功能说明:**
- 自动避免双重交叉验证
- 智能选择调优策略
- 支持分类、回归、无监督模型
- 模型信息描述和分类
- 向后兼容的模型字典

#### 1.3 数据处理模块
```python
# 位置: 第466-600行
def load_dataset()              # 数据集加载
def preprocess_data()           # 数据预处理
def enhanced_preprocessing()    # 增强预处理
```

**功能说明:**
- 支持多种数据集格式
- 数据清洗和特征工程
- 缺失值处理和异常值检测
- 特征选择和数据平衡

#### 1.4 训练评估模块
```python
# 位置: 第601-950行
def train_and_evaluate()        # 模型训练评估
def hyperparameter_tuning()     # 超参数调优
def compare_models()           # 模型对比
```

**功能说明:**
- 完整的训练流程
- 超参数网格搜索和随机搜索
- 多模型性能对比
- 交叉验证支持

#### 1.5 缓存系统
```python
# 位置: 第56-104行
MODEL_CACHE = {}               # 模型缓存
def generate_cache_key()       # 缓存键生成
def get_cached_model()         # 缓存获取
def cache_model()              # 缓存存储
```

**功能说明:**
- 智能缓存管理
- LRU淘汰策略
- 缓存命中检测
- 性能优化

### 2. 前端模块 (`static/index.html`)

#### 2.1 用户界面层
```html
<!-- 位置: 第560-750行 -->
<div class="control-panel">     <!-- 控制面板 -->
<div class="results-panel">     <!-- 结果面板 -->
<div class="comparisonModal">   <!-- 对比模态框 -->
<div class="cacheModal">        <!-- 缓存管理模态框 -->
```

#### 2.2 交互逻辑层
```javascript
// 位置: 第844-2148行
function runExperiment()        // 运行实验
function openModelComparison() // 模型对比
function openCacheManager()     // 缓存管理
function updateProgress()       // 进度更新
```

#### 2.3 可视化组件
```javascript
// 位置: 第1855-2032行
function renderComparisonChart()    // 对比图表
function renderTrainingCharts()     // 训练过程图表
```

### 3. 数据模块

#### 3.1 数据集存储
```
datasets/
├── watermelon.csv              # 西瓜数据集
└── (其他数据集)
```

#### 3.2 实验历史存储
```python
EXPERIMENT_HISTORY = {}         # 实验历史
COMPARISON_EXPERIMENTS = {}     # 对比实验
```

## 🔌 API 接口设计

### 1. 核心实验接口

#### POST `/run_experiment`
**功能:** 运行机器学习实验
**请求参数:**
```json
{
    "dataset": "iris",
    "model": "原创逻辑回归",
    "test_size": "30%",
    "split_method": "random",
    "metric": "accuracy",
    "preprocessing": {
        "handle_missing": true,
        "detect_outliers": false,
        "feature_selection": true,
        "balance_data": false
    },
    "hyperparams": {
        "enable_tuning": true,
        "search_type": "grid",
        "cv_folds": 5
    }
}
```

**响应格式:**
```json
{
    "success": true,
    "results": {
        "accuracy": 0.95,
        "f1_score": 0.94,
        "training_history": {...},
        "feature_importance": [...],
        "from_cache": false
    }
}
```

#### POST `/compare_models`
**功能:** 多模型性能对比
**请求参数:**
```json
{
    "models": ["原创逻辑回归", "决策树", "K近邻"],
    "dataset": "iris",
    "test_size": "30%",
    "split_method": "random"
}
```

### 2. 数据管理接口

#### POST `/upload_csv`
**功能:** 上传自定义CSV数据集
**请求:** FormData with CSV file
**响应:**
```json
{
    "success": true,
    "dataset_id": "dataset_123",
    "columns": ["feature1", "feature2", "label"],
    "suggested_label": "label"
}
```

#### GET `/experiment_history`
**功能:** 获取实验历史记录
**响应:**
```json
{
    "success": true,
    "history": [
        {
            "id": "exp_123",
            "timestamp": "2024-01-01 12:00:00",
            "model": "原创逻辑回归",
            "dataset": "iris",
            "accuracy": 0.95
        }
    ]
}
```

### 3. 缓存管理接口

#### GET `/cache_info`
**功能:** 获取缓存信息
**响应:**
```json
{
    "success": true,
    "cache_count": 5,
    "cache_limit": 10,
    "cache_info": [...]
}
```

#### POST `/clear_cache`
**功能:** 清空模型缓存
**响应:**
```json
{
    "success": true,
    "message": "缓存已清空"
}
```

## 🚀 部署指南

### 1. 环境要求

#### Python 环境
```bash
Python >= 3.8
pip install flask flask-socketio pandas numpy scikit-learn
pip install imbalanced-learn matplotlib seaborn
```

#### 系统要求
- 内存: 最少 2GB RAM
- 存储: 最少 1GB 可用空间
- 网络: 支持 WebSocket 连接

### 2. 安装步骤

#### 2.1 克隆项目
```bash
git clone <repository-url>
cd 机器学习_v2
```

#### 2.2 安装依赖
```bash
pip install -r requirements.txt
```

#### 2.3 启动服务
```bash
python app.py
```

#### 2.4 访问应用
```
http://localhost:5050
```

### 3. 生产环境部署

#### 3.1 使用 Gunicorn
```bash
pip install gunicorn
gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:5050 app:app
```

#### 3.2 使用 Nginx 反向代理
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:5050;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    location /socket.io/ {
        proxy_pass http://127.0.0.1:5050;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

#### 3.3 Docker 部署
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5050

CMD ["python", "app.py"]
```

## 🔧 配置说明

### 1. 应用配置
```python
# app.py 第1-50行
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 文件上传限制
CACHE_SIZE_LIMIT = 10  # 缓存大小限制
```

### 2. 模型配置
```python
# 支持的模型类型
CLASSIFICATION_MODELS = ["逻辑回归", "决策树", "SVM", ...]
REGRESSION_MODELS = ["线性回归", "SVR"]
UNSUPERVISED_MODELS = ["K-Means", "PCA"]
```

### 3. 数据预处理配置
```python
PREPROCESSING_OPTIONS = {
    "handle_missing": True,      # 处理缺失值
    "detect_outliers": False,    # 检测异常值
    "feature_selection": True,   # 特征选择
    "balance_data": False        # 数据平衡
}
```

## 📊 性能优化

### 1. 缓存策略
- **LRU 淘汰**: 自动删除最久未使用的缓存
- **智能键生成**: 基于配置参数生成唯一缓存键
- **内存管理**: 限制缓存大小，防止内存溢出

### 2. 并发处理
- **WebSocket 通信**: 实时进度更新
- **异步处理**: 非阻塞模型训练
- **资源池管理**: 合理分配计算资源

### 3. 数据优化
- **批量处理**: 高效的数据预处理
- **内存映射**: 大数据集的内存优化
- **特征缓存**: 避免重复特征计算

## 🛡️ 安全考虑

### 1. 输入验证
- **文件类型检查**: 只允许 CSV 文件上传
- **文件大小限制**: 防止大文件攻击
- **参数验证**: 严格的输入参数检查

### 2. 错误处理
- **异常捕获**: 全面的错误处理机制
- **用户友好**: 清晰的错误提示信息
- **日志记录**: 详细的错误日志

### 3. 资源保护
- **内存限制**: 防止内存溢出
- **CPU 保护**: 避免长时间占用
- **存储管理**: 定期清理临时文件

## 📈 扩展性设计

### 1. 模块化架构
- **插件式模型**: 易于添加新模型
- **可配置参数**: 灵活的参数设置
- **接口标准化**: 统一的模型接口

### 2. 水平扩展
- **负载均衡**: 支持多实例部署
- **数据分片**: 大数据集的分片处理
- **缓存集群**: 分布式缓存支持

### 3. 功能扩展
- **新算法支持**: 易于集成新算法
- **可视化增强**: 可扩展的图表组件
- **API 扩展**: 灵活的接口设计

## 📝 开发指南

### 1. 添加新模型
```python
@register_model("新模型", "classification", "模型描述")
class NewModel:
    def __init__(self):
        # 初始化参数
        pass
    
    def fit(self, X, y):
        # 训练逻辑
        pass
    
    def predict(self, X):
        # 预测逻辑
        pass
```

### 2. 添加新数据集
```python
def load_new_dataset():
    # 加载新数据集
    data = load_data()
    return X, y, feature_names, target_names, raw_df
```

### 3. 添加新可视化
```javascript
function renderNewChart(data) {
    // 新的图表渲染逻辑
    new Chart(ctx, {
        // Chart.js 配置
    });
}
```

## 🎯 总结

本机器学习实验平台采用现代化的 Web 架构，具有以下特点：

1. **高可扩展性**: 模块化设计，易于扩展
2. **高性能**: 智能缓存，优化处理流程
3. **用户友好**: 直观界面，实时反馈
4. **专业级**: 原创算法，深度理解
5. **生产就绪**: 完整部署方案，安全可靠

该平台不仅满足教学需求，更是一个具有实际应用价值的专业级机器学习实验平台。
