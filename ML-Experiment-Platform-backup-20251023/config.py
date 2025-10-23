"""
机器学习实验平台配置文件
Machine Learning Experiment Platform Configuration
"""

import os

class Config:
    """应用配置类"""
    
    # 基础配置
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'ml-experiment-platform-2024'
    
    # 文件上传配置
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB
    UPLOAD_FOLDER = 'assets/uploads'
    ALLOWED_EXTENSIONS = {'csv'}
    
    # 缓存配置
    CACHE_SIZE_LIMIT = 10
    
    # 模型配置
    SUPPORTED_MODELS = [
        '原创逻辑回归', '原创决策树', '原创K近邻',
        '逻辑回归', '决策树', '支持向量机(SVM)', 'K近邻(KNN)',
        '随机森林', '梯度提升树(GBDT)', '朴素贝叶斯',
        '线性回归', '支持向量回归(SVR)',
        'K-Means', '主成分分析(PCA)', '线性判别分析(LDA)', '二次判别分析(QDA)'
    ]
    
    # 数据集配置
    BUILTIN_DATASETS = [
        'iris', 'wine', 'breast_cancer', 'digits', 
        'watermelon', 'synthetic_classification', 'synthetic_regression'
    ]
    
    # 评估指标
    CLASSIFICATION_METRICS = ['accuracy', 'precision', 'recall', 'f1']
    REGRESSION_METRICS = ['mse', 'mae', 'r2']
    
    # 预处理选项
    PREPROCESSING_OPTIONS = {
        'handle_missing': True,
        'detect_outliers': False,
        'feature_selection': True,
        'balance_data': False
    }
