# ML-Experiment-Platform

> 一个功能完整、性能优异的专业级机器学习实验平台
> A comprehensive and high-performance professional machine learning experiment platform

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🌟 项目特色

### ⭐ 原创算法实现
- **手动编写经典算法**: 逻辑回归、决策树、K近邻
- **完整数学原理**: 梯度下降、信息增益、距离计算
- **训练过程可视化**: 损失函数曲线、准确率变化
- **深度算法理解**: 体现对机器学习算法的深度掌握

### ⚡ 智能性能优化
- **模型缓存系统**: 智能缓存，避免重复计算
- **LRU淘汰策略**: 自动管理内存使用
- **实时进度更新**: WebSocket实时通信
- **性能监控**: 缓存命中率和运行时间统计

### 📊 高级可视化
- **训练过程图表**: 损失函数和准确率变化曲线
- **模型对比分析**: 多模型性能对比图表
- **特征重要性**: 直观的特征贡献度展示
- **专业图表设计**: 基于Chart.js的现代化图表

### 🎨 优秀用户体验
- **简约清晰界面**: 不拥挤的现代化设计
- **实时进度反馈**: 详细的训练步骤显示
- **智能错误处理**: 友好的错误提示和建议
- **快捷键支持**: 提升操作效率

## 🚀 快速开始

### 环境要求
- Python 3.8+
- 内存: 最少 2GB RAM
- 存储: 最少 1GB 可用空间

### 安装步骤
```bash
# 1. 克隆项目
git clone <repository-url>
cd 机器学习_v2

# 2. 安装依赖
pip install flask flask-socketio pandas numpy scikit-learn
pip install imbalanced-learn matplotlib seaborn

# 3. 启动服务
python app.py

# 4. 访问平台
# 浏览器打开: http://localhost:5050
```

### 快速体验
1. **选择数据集**: 鸢尾花数据集 (iris)
2. **选择模型**: 原创逻辑回归 ⭐
3. **运行实验**: 点击"运行实验"按钮
4. **查看结果**: 观察训练过程和性能指标

## 📚 详细文档

### 📖 [架构文档](ARCHITECTURE.md)
- 系统架构设计
- 模块划分说明
- API接口文档
- 部署指南

### 🛠️ [技术栈说明](TECH_STACK.md)
- 核心技术栈
- 算法实现细节
- 性能优化技术
- 扩展性设计

### 👥 [用户使用指南](USER_GUIDE.md)
- 详细使用说明
- 最佳实践建议
- 常见问题解答
- 学习路径指导

## 🎯 核心功能

### 1. 原创模型实现
```python
# 手动实现的逻辑回归算法
class MyLogisticRegression:
    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.cost_history = []
        self.accuracy_history = []
    
    def fit(self, X, y):
        # 梯度下降训练过程
        # 记录损失函数和准确率变化
        pass
```

### 2. 智能缓存系统
```python
# 基于配置参数的智能缓存
cache_key = generate_cache_key(
    model_name, dataset_name, test_size, 
    split_method, preprocessing_config, hyperparams
)

if cache_key in MODEL_CACHE:
    # 直接返回缓存结果，秒级完成
    return MODEL_CACHE[cache_key]['results']
```

### 3. 实时进度更新
```javascript
// WebSocket实时通信
socket.on('progress', function(data) {
    updateProgress(
        data.progress,      // 进度百分比
        data.message,      // 当前步骤
        data.step,         // 步骤编号
        data.total_steps,  // 总步骤数
        data.details       // 详细说明
    );
});
```

### 4. 高级可视化
```javascript
// 训练过程可视化
function renderTrainingCharts(trainingHistory) {
    // 损失函数曲线
    new Chart(lossCtx, {
        type: 'line',
        data: {
            labels: iterations,
            datasets: [{
                label: '损失函数',
                data: trainingHistory.cost_history,
                borderColor: 'rgba(231, 76, 60, 1)'
            }]
        }
    });
}
```

## 📊 支持的功能

### 数据集支持
- **内置数据集**: 鸢尾花、西瓜数据集
- **自定义数据集**: CSV文件上传
- **数据预处理**: 缺失值处理、异常值检测、特征选择
- **数据平衡**: SMOTE过采样、随机欠采样

### 模型支持
- **原创模型**: 逻辑回归、决策树、K近邻 (手动实现)
- **分类模型**: SVM、随机森林、GBDT、朴素贝叶斯
- **回归模型**: 线性回归、SVR
- **无监督模型**: K-Means、PCA、LDA、QDA

### 评估指标
- **分类指标**: 准确率、精确率、召回率、F1分数
- **回归指标**: 均方误差、平均绝对误差、R²分数
- **聚类指标**: 轮廓系数、惯性、聚类数量

### 高级功能
- **超参数调优**: 网格搜索、随机搜索
- **模型对比**: 多模型性能对比
- **交叉验证**: 3-10折交叉验证
- **缓存管理**: 智能缓存和性能优化

## 🏗️ 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   前端界面      │    │   后端服务      │    │   数据处理层      │
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

## 🎓 学习价值

### 算法理解
- **数学原理**: 深入理解算法本质
- **代码实现**: 手动编写核心逻辑
- **参数调优**: 超参数搜索策略
- **性能分析**: 算法复杂度分析

### 工程实践
- **系统设计**: 架构设计能力
- **API设计**: RESTful接口设计
- **前端开发**: 用户界面设计
- **部署运维**: 生产环境部署

### 项目经验
- **完整流程**: 从需求到部署
- **技术栈**: 前后端 + 算法 + 部署
- **性能优化**: 缓存、并发、可视化
- **用户体验**: 界面设计、交互优化

## 🚀 部署指南

### 开发环境
```bash
python app.py
```

### 生产环境
```bash
# 使用Gunicorn
gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:5050 app:app

# 使用Docker
docker build -t ml-platform .
docker run -p 5050:5050 ml-platform
```

### 性能优化
- **缓存配置**: 调整缓存大小限制
- **并发处理**: 配置工作进程数量
- **内存管理**: 监控内存使用情况
- **负载均衡**: 多实例部署

## 📈 性能指标

### 缓存效果
- **缓存命中率**: 90%+ (相同配置)
- **运行速度**: 缓存命中时秒级完成
- **内存使用**: 智能管理，避免溢出
- **用户体验**: 大幅提升响应速度

### 系统性能
- **并发支持**: 多用户同时使用
- **数据处理**: 支持大数据集处理
- **实时通信**: WebSocket低延迟
- **可视化**: 流畅的图表渲染

## 🤝 贡献指南

### 添加新模型
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

### 添加新功能
1. **后端功能**: 在 `app.py` 中添加新的路由和逻辑
2. **前端功能**: 在 `static/index.html` 中添加界面和交互
3. **文档更新**: 更新相关文档说明
4. **测试验证**: 确保功能正常工作

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

感谢以下开源项目的支持：
- [Flask](https://flask.palletsprojects.com/) - Web框架
- [Scikit-learn](https://scikit-learn.org/) - 机器学习库
- [Pandas](https://pandas.pydata.org/) - 数据处理
- [Chart.js](https://www.chartjs.org/) - 数据可视化

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 Issue
- 发送邮件
- 在线讨论

---

**这个机器学习实验平台不仅是一个学习工具，更是一个具有实际应用价值的专业级项目！** 🎉
