from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits, make_classification, make_regression
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import os
import json

app = Flask(__name__)
# 限制上传大小为100MB（可按需调整）
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
# 收紧CORS到本地常见端口（可按需调整）
CORS(app, resources={r"/*": {"origins": [
    "http://localhost:5001", "http://127.0.0.1:5001",
    "http://localhost:5050", "http://127.0.0.1:5050",
    "http://localhost:3000", "http://127.0.0.1:3000"
]}})

# 初始化SocketIO
socketio = SocketIO(app, cors_allowed_origins=[
    "http://localhost:5001", "http://127.0.0.1:5001",
    "http://localhost:5050", "http://127.0.0.1:5050",
    "http://localhost:3000", "http://127.0.0.1:3000"
])

# 临时存储上传的数据集（仅开发模式内存保存）
UPLOADED_DATASETS = {}

# 历史实验存储
EXPERIMENT_HISTORY = {}

# 模型对比实验存储
COMPARISON_EXPERIMENTS = {}

# 模型缓存存储
MODEL_CACHE = {}
CACHE_SIZE_LIMIT = 10  # 最多缓存10个模型

# 缓存管理函数
def generate_cache_key(model_name, dataset_name, test_size, split_method, preprocessing_config, hyperparams):
    """生成缓存键"""
    import hashlib
    import json
    
    cache_data = {
        'model_name': model_name,
        'dataset_name': dataset_name,
        'test_size': test_size,
        'split_method': split_method,
        'preprocessing': preprocessing_config,
        'hyperparams': hyperparams
    }
    
    # 创建哈希键
    cache_str = json.dumps(cache_data, sort_keys=True)
    return hashlib.md5(cache_str.encode()).hexdigest()

def get_cached_model(cache_key):
    """获取缓存的模型"""
    if cache_key in MODEL_CACHE:
        MODEL_CACHE[cache_key]['last_used'] = pd.Timestamp.now()
        return MODEL_CACHE[cache_key]['model']
    return None

def cache_model(cache_key, model, results):
    """缓存模型"""
    # 如果缓存已满，删除最旧的条目
    if len(MODEL_CACHE) >= CACHE_SIZE_LIMIT:
        oldest_key = min(MODEL_CACHE.keys(), 
                        key=lambda k: MODEL_CACHE[k]['last_used'])
        del MODEL_CACHE[oldest_key]
    
    MODEL_CACHE[cache_key] = {
        'model': model,
        'results': results,
        'last_used': pd.Timestamp.now(),
        'created_at': pd.Timestamp.now()
    }

def clear_model_cache():
    """清空模型缓存"""
    global MODEL_CACHE
    MODEL_CACHE.clear()

# ==================== 模型注册 ====================

class MyLogisticRegression:
    """原创实现：逻辑回归算法"""
    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-4):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def sigmoid(self, z):
        """Sigmoid激活函数"""
        # 防止数值溢出
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        """训练逻辑回归模型"""
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0
        self.cost_history = []
        self.accuracy_history = []
        
        for i in range(self.max_iter):
            # 前向传播
            z = np.dot(X, self.weights) + self.bias
            h = self.sigmoid(z)
            
            # 计算损失（交叉熵）
            epsilon = 1e-15  # 防止log(0)
            h_clipped = np.clip(h, epsilon, 1 - epsilon)
            cost = -np.mean(y * np.log(h_clipped) + (1 - y) * np.log(1 - h_clipped))
            self.cost_history.append(cost)
            
            # 计算准确率
            predictions = (h >= 0.5).astype(int)
            accuracy = np.mean(predictions == y)
            self.accuracy_history.append(accuracy)
            
            # 反向传播
            dw = (1/m) * np.dot(X.T, (h - y))
            db = (1/m) * np.sum(h - y)
            
            # 更新参数
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # 早停条件
            if i > 0 and abs(self.cost_history[-1] - self.cost_history[-2]) < self.tol:
                break
    
    def predict_proba(self, X):
        """预测概率"""
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def predict(self, X):
        """预测类别"""
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)
    
    def get_training_history(self):
        """获取训练历史数据"""
        return {
            'cost_history': self.cost_history,
            'accuracy_history': self.accuracy_history,
            'iterations': len(self.cost_history)
        }

class MyDecisionTree:
    """原创实现：决策树算法（ID3）"""
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
    
    def entropy(self, y):
        """计算信息熵"""
        if len(y) == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-15))
        return entropy
    
    def information_gain(self, X, y, feature_idx, threshold):
        """计算信息增益"""
        # 分割数据
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return 0
        
        # 计算父节点熵
        parent_entropy = self.entropy(y)
        
        # 计算子节点熵
        left_entropy = self.entropy(y[left_mask])
        right_entropy = self.entropy(y[right_mask])
        
        # 计算加权平均熵
        left_weight = np.sum(left_mask) / len(y)
        right_weight = np.sum(right_mask) / len(y)
        weighted_entropy = left_weight * left_entropy + right_weight * right_entropy
        
        # 信息增益
        return parent_entropy - weighted_entropy
    
    def find_best_split(self, X, y):
        """找到最佳分割点"""
        best_gain = 0
        best_feature = None
        best_threshold = None
        
        for feature_idx in range(X.shape[1]):
            # 获取该特征的所有唯一值
            unique_values = np.unique(X[:, feature_idx])
            
            for threshold in unique_values:
                gain = self.information_gain(X, y, feature_idx, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def build_tree(self, X, y, depth=0):
        """递归构建决策树"""
        # 停止条件
        if (self.max_depth is not None and depth >= self.max_depth) or \
           len(y) < self.min_samples_split or \
           len(np.unique(y)) == 1:
            # 返回叶节点（多数类）
            unique, counts = np.unique(y, return_counts=True)
            return {'class': unique[np.argmax(counts)], 'is_leaf': True}
        
        # 找到最佳分割
        feature, threshold, gain = self.find_best_split(X, y)
        
        if gain == 0:  # 无法进一步分割
            unique, counts = np.unique(y, return_counts=True)
            return {'class': unique[np.argmax(counts)], 'is_leaf': True}
        
        # 分割数据
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        # 递归构建子树
        left_tree = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self.build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'feature': feature,
            'threshold': threshold,
            'left': left_tree,
            'right': right_tree,
            'is_leaf': False
        }
    
    def fit(self, X, y):
        """训练决策树"""
        self.tree = self.build_tree(X, y)
    
    def predict_single(self, x, tree):
        """预测单个样本"""
        if tree['is_leaf']:
            return tree['class']
        
        if x[tree['feature']] <= tree['threshold']:
            return self.predict_single(x, tree['left'])
        else:
            return self.predict_single(x, tree['right'])
    
    def predict(self, X):
        """预测"""
        predictions = []
        for x in X:
            pred = self.predict_single(x, self.tree)
            predictions.append(pred)
        return np.array(predictions)

class MyKNN:
    """原创实现：K近邻算法"""
    def __init__(self, k=3, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None
    
    def euclidean_distance(self, x1, x2):
        """欧几里得距离"""
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def manhattan_distance(self, x1, x2):
        """曼哈顿距离"""
        return np.sum(np.abs(x1 - x2))
    
    def calculate_distance(self, x1, x2):
        """计算距离"""
        if self.distance_metric == 'euclidean':
            return self.euclidean_distance(x1, x2)
        elif self.distance_metric == 'manhattan':
            return self.manhattan_distance(x1, x2)
        else:
            return self.euclidean_distance(x1, x2)
    
    def fit(self, X, y):
        """训练KNN模型（存储训练数据）"""
        self.X_train = X
        self.y_train = y
    
    def predict_single(self, x):
        """预测单个样本"""
        # 计算与所有训练样本的距离
        distances = []
        for i, train_x in enumerate(self.X_train):
            dist = self.calculate_distance(x, train_x)
            distances.append((dist, self.y_train[i]))
        
        # 按距离排序，取前k个
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:self.k]
        
        # 投票决定类别
        votes = {}
        for _, label in k_nearest:
            votes[label] = votes.get(label, 0) + 1
        
        # 返回得票最多的类别
        return max(votes, key=votes.get)
    
    def predict(self, X):
        """预测"""
        predictions = []
        for x in X:
            pred = self.predict_single(x)
            predictions.append(pred)
        return np.array(predictions)

# 模型注册框架
class ModelRegistry:
    def __init__(self):
        self._models = {}
        self._categories = {
            'classification': [],
            'regression': [],
            'unsupervised': []
        }
    
    def register(self, name, model_class, category='classification', description=''):
        """注册模型"""
        self._models[name] = {
            'class': model_class,
            'category': category,
            'description': description,
            'instance': None
        }
        if category in self._categories:
            self._categories[category].append(name)
    
    def get_model(self, name):
        """获取模型实例"""
        if name not in self._models:
            return None
        
        model_info = self._models[name]
        if model_info['instance'] is None:
            model_info['instance'] = model_info['class']()
        
        return model_info['instance']
    
    def get_models_by_category(self, category):
        """按类别获取模型列表"""
        return self._categories.get(category, [])
    
    def get_all_models(self):
        """获取所有模型"""
        return list(self._models.keys())
    
    def get_model_info(self, name):
        """获取模型信息"""
        return self._models.get(name, {})

# 创建全局模型注册表
model_registry = ModelRegistry()

# 模型注册装饰器
def register_model(name, category='classification', description=''):
    def decorator(model_class):
        model_registry.register(name, model_class, category, description)
        return model_class
    return decorator

# 使用装饰器注册模型

@register_model("线性回归", "regression", "线性回归模型")
class LinearRegressionModel(LinearRegression):
    pass

@register_model("逻辑回归", "classification", "逻辑回归分类器")
class LogisticRegressionModel(LogisticRegression):
    def __init__(self):
        super().__init__(random_state=42, max_iter=1000)

@register_model("决策树", "classification", "决策树分类器")
class DecisionTreeModel(DecisionTreeClassifier):
    def __init__(self):
        super().__init__(random_state=42)

@register_model("支持向量机(SVM)", "classification", "支持向量机分类器")
class SVMModel(SVC):
    def __init__(self):
        super().__init__(kernel='rbf', random_state=42, probability=True)

@register_model("K近邻(KNN)", "classification", "K近邻分类器")
class KNNModel(KNeighborsClassifier):
    def __init__(self):
        super().__init__(n_neighbors=3)

@register_model("随机森林", "classification", "随机森林分类器")
class RandomForestModel(RandomForestClassifier):
    def __init__(self):
        super().__init__(n_estimators=100, random_state=42)

@register_model("梯度提升树(GBDT)", "classification", "梯度提升树分类器")
class GBDTModel(GradientBoostingClassifier):
    def __init__(self):
        super().__init__(random_state=42)

@register_model("K 均值聚类(K-Means)", "unsupervised", "K均值聚类算法")
class KMeansModel(KMeans):
    def __init__(self):
        super().__init__(n_clusters=3, random_state=42)

@register_model("主成分分析(PCA)", "unsupervised", "主成分分析降维")
class PCAModel(PCA):
    def __init__(self):
        super().__init__(n_components=2, random_state=42)

@register_model("多层感知机(MLP)", "classification", "多层感知机神经网络")
class MLPModel(MLPClassifier):
    def __init__(self):
        super().__init__(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)

@register_model("朴素贝叶斯", "classification", "朴素贝叶斯分类器")
class NaiveBayesModel(GaussianNB):
    def __init__(self):
        super().__init__()

@register_model("线性判别分析(LDA)", "classification", "线性判别分析")
class LDAModel(LinearDiscriminantAnalysis):
    def __init__(self):
        super().__init__()

@register_model("二次判别分析(QDA)", "classification", "二次判别分析")
class QDAModel(QuadraticDiscriminantAnalysis):
    def __init__(self):
        super().__init__()

@register_model("岭回归", "regression", "岭回归模型")
class RidgeModel(Ridge):
    def __init__(self):
        super().__init__(alpha=1.0, random_state=42)

@register_model("Lasso回归", "regression", "Lasso回归模型")
class LassoModel(Lasso):
    def __init__(self):
        super().__init__(alpha=1.0, random_state=42)

@register_model("深度神经网络(DNN)", "classification", "深度神经网络")
class DNNModel(MLPClassifier):
    def __init__(self):
        super().__init__(hidden_layer_sizes=(100, 50, 25), max_iter=1000, random_state=42)

@register_model("卷积神经网络(CNN)", "classification", "卷积神经网络")
class CNNModel:
    def __init__(self):
        # 简化的CNN实现，用于表格数据
        from sklearn.neural_network import MLPClassifier
        self.model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    
    def fit(self, X, y):
        return self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)

@register_model("循环神经网络(RNN)", "classification", "循环神经网络")
class RNNModel:
    def __init__(self):
        # 简化的RNN实现，用于表格数据
        from sklearn.neural_network import MLPClassifier
        self.model = MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=1000, random_state=42)
    
    def fit(self, X, y):
        return self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)

@register_model("长短期记忆网络(LSTM)", "classification", "LSTM网络")
class LSTMModel:
    def __init__(self):
        # 简化的LSTM实现，用于表格数据
        from sklearn.neural_network import MLPClassifier
        self.model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
    
    def fit(self, X, y):
        return self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)

@register_model("支持向量回归(SVR)", "regression", "支持向量回归")
class SVRModel:
    def __init__(self):
        from sklearn.svm import SVR
        self.model = SVR(kernel='rbf')
    
    def fit(self, X, y):
        return self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)

# 保持向后兼容的MODELS字典
MODELS = {name: model_registry.get_model(name) for name in model_registry.get_all_models()}

# 修改 load_dataset 函数中的西瓜数据集部分
def load_dataset(dataset_name, test_size=0.3):
    if dataset_name == "iris":
        data = load_iris()
        X = data.data
        y = data.target
        feature_names = data.feature_names.tolist() if hasattr(data.feature_names, 'tolist') else list(data.feature_names)
        target_names = data.target_names.tolist()
        return X, y, feature_names, target_names, None
    
    elif dataset_name == "wine":
        data = load_wine()
        X = data.data
        y = data.target
        feature_names = data.feature_names.tolist() if hasattr(data.feature_names, 'tolist') else list(data.feature_names)
        target_names = data.target_names.tolist()
        return X, y, feature_names, target_names, None
    
    elif dataset_name == "breast_cancer":
        data = load_breast_cancer()
        X = data.data
        y = data.target
        feature_names = data.feature_names.tolist() if hasattr(data.feature_names, 'tolist') else list(data.feature_names)
        target_names = data.target_names.tolist()
        return X, y, feature_names, target_names, None
    
    elif dataset_name == "digits":
        data = load_digits()
        X = data.data
        y = data.target
        feature_names = [f'pixel_{i}' for i in range(X.shape[1])]
        target_names = [str(i) for i in range(10)]
        return X, y, feature_names, target_names, None
    
    elif dataset_name == "synthetic_classification":
        X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                                 n_redundant=5, n_classes=3, random_state=42)
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        target_names = ['class_0', 'class_1', 'class_2']
        return X, y, feature_names, target_names, None
    
    elif dataset_name == "synthetic_regression":
        X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        target_names = ['target']
        return X, y, feature_names, target_names, None
        
    elif dataset_name == "watermelon":
        # 加载完整的西瓜数据集
        try:
            # 尝试不同的文件路径
            possible_paths = [
                'datasets/watermelon.csv',
                'watermelon.csv',
                './datasets/watermelon.csv',
                './watermelon.csv'
            ]
            
            df = None
            for path in possible_paths:
                try:
                    df = pd.read_csv(path)
                    break
                except FileNotFoundError:
                    continue
            
            if df is None:
                print("未找到西瓜数据集文件，使用示例数据")
                return create_sample_watermelon_data()
            
            # 检查列名并处理
            if '编号' in df.columns:
                df = df.drop('编号', axis=1)
            
            # 分离特征和标签
            if '标签' in df.columns:
                X = df.drop('标签', axis=1)
                y = df['标签']
            else:
                # 如果没有标签列，使用最后一列作为标签
                X = df.iloc[:, :-1]
                y = df.iloc[:, -1]
            
            # 编码标签
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            
            feature_names = X.columns.tolist()
            target_names = le.classes_.tolist()
            
            return X, y_encoded, feature_names, target_names, df
            
        except Exception as e:
            print(f"加载西瓜数据集时出错: {e}")
            return create_sample_watermelon_data()
    
    return None, None, None, None, None

def load_custom_dataset(dataset_id, label_column):
    """从内存中读取用户上传的数据集，并根据标签列拆分X/y。"""
    if dataset_id not in UPLOADED_DATASETS:
        raise ValueError("未找到已上传的数据集，请重新上传")
    df = UPLOADED_DATASETS[dataset_id]
    if label_column not in df.columns:
        raise ValueError("标签列不存在于上传的数据集中")

    X = df.drop(columns=[label_column])
    y = df[label_column]
    # 标签编码
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    feature_names = X.columns.tolist()
    target_names = le.classes_.tolist()
    return X, y_encoded, feature_names, target_names, df
def create_sample_watermelon_data():
    """创建示例西瓜数据集"""
    X = np.array([
        [0.697, 0.460], [0.774, 0.376], [0.634, 0.264], [0.608, 0.318],
        [0.556, 0.215], [0.403, 0.237], [0.481, 0.149], [0.437, 0.211],
        [0.666, 0.091], [0.243, 0.267], [0.245, 0.057], [0.343, 0.099],
        [0.639, 0.161], [0.657, 0.198], [0.360, 0.370], [0.593, 0.042],
        [0.719, 0.103]
    ])
    y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    feature_names = ['密度', '含糖率']
    target_names = ['坏瓜', '好瓜']
    return X, y, feature_names, target_names, None

# 预处理数据
def preprocess_data(X, y):
    """预处理数据，处理分类特征"""
    if isinstance(X, pd.DataFrame):
        # 识别数值列和分类列
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        # 创建预处理管道
        # 兼容不同版本的sklearn: 优先使用 sparse_output，其次回退到 sparse
        try:
            ohe = OneHotEncoder(drop='first', sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(drop='first', sparse=False)

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', ohe, categorical_features)
            ])
        
        # 应用预处理
        X_processed = preprocessor.fit_transform(X)
        
        # 获取特征名称（对于独热编码后的分类特征）
        feature_names = numeric_features.copy()
        if categorical_features:
            ohe = preprocessor.named_transformers_['cat']
            for i, col in enumerate(categorical_features):
                categories = ohe.categories_[i][1:]  # 去掉第一个类别（作为基准）
                for cat in categories:
                    feature_names.append(f"{col}_{cat}")
        
        return X_processed, feature_names, preprocessor
    else:
        # 如果已经是numpy数组，直接标准化
        scaler = StandardScaler()
        X_processed = scaler.fit_transform(X)
        # 生成特征名称
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        return X_processed, feature_names, scaler

# 在 train_and_evaluate 函数中修复特征重要性处理
def get_feature_importance(model, n_features, feature_names=None, X=None, y=None):
    """获取特征重要性（如果模型支持）"""
    try:
        # 检查模型是否已经训练（有coef_或feature_importances_属性）
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            # 归一化
            if importance.sum() > 0:
                importance = importance / importance.sum()
            # 确保返回Python列表而不是NumPy数组
            return importance.tolist() if hasattr(importance, 'tolist') else list(importance)
        elif hasattr(model, 'coef_'):
            # 对于线性模型，使用系数的绝对值作为重要性
            if len(model.coef_.shape) > 1:
                importance = np.mean(np.abs(model.coef_), axis=0)
            else:
                importance = np.abs(model.coef_)
            # 归一化
            if importance.sum() > 0:
                importance = importance / importance.sum()
            # 确保返回Python列表而不是NumPy数组
            return importance.tolist() if hasattr(importance, 'tolist') else list(importance)
    except Exception as e:
        print(f"获取特征重要性时出错: {e}")
    
    # 如果不支持特征重要性，使用基于方差的重要性估计
    if X is not None and y is not None:
        try:
            from sklearn.feature_selection import f_classif, f_regression
            from sklearn.preprocessing import LabelEncoder
            
            # 判断是分类还是回归任务
            if len(np.unique(y)) <= 20:  # 分类任务
                f_scores, _ = f_classif(X, y)
            else:  # 回归任务
                f_scores, _ = f_regression(X, y)
            
            # 归一化F分数作为重要性
            if f_scores.sum() > 0:
                importance = f_scores / f_scores.sum()
            else:
                importance = np.ones(n_features) / n_features
            
            return importance.tolist() if hasattr(importance, 'tolist') else list(importance)
        except Exception as e:
            print(f"使用统计方法计算特征重要性时出错: {e}")
    
    # 最后的备选方案：返回基于特征方差的相对重要性
    if X is not None:
        try:
            feature_vars = np.var(X, axis=0)
            if feature_vars.sum() > 0:
                importance = feature_vars / feature_vars.sum()
            else:
                importance = np.ones(n_features) / n_features
            return importance.tolist() if hasattr(importance, 'tolist') else list(importance)
        except Exception as e:
            print(f"使用方差计算特征重要性时出错: {e}")
    
    # 如果所有方法都失败，返回平均分布
    return [1.0/n_features] * n_features if n_features > 0 else []

# 修改 train_and_evaluate 函数中的特征重要性调用
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, precision_score, recall_score

# WebSocket事件处理
@socketio.on('connect')
def handle_connect():
    print('客户端已连接')
    emit('status', {'message': '连接成功', 'type': 'success'})

@socketio.on('disconnect')
def handle_disconnect():
    print('客户端已断开连接')

# 发送进度更新的辅助函数
def send_progress(progress, message, step=None, total_steps=None, details=None):
    """发送训练进度更新"""
    progress_data = {
        'progress': progress,
        'message': message,
        'timestamp': pd.Timestamp.now().strftime('%H:%M:%S')
    }
    
    if step is not None and total_steps is not None:
        progress_data['step'] = step
        progress_data['total_steps'] = total_steps
        progress_data['step_message'] = f"步骤 {step}/{total_steps}"
    
    if details:
        progress_data['details'] = details
    
    socketio.emit('progress', progress_data)

# 超参数调优功能
def hyperparameter_tuning(model, model_name, X, y, param_grid, cv=5, search_type='grid'):
    """执行超参数调优"""
    try:
        if search_type == 'grid':
            search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
        else:  # random
            search = RandomizedSearchCV(model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, n_iter=20)
        
        search.fit(X, y)
        
        return {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'best_estimator': search.best_estimator_,
            'cv_results': search.cv_results_
        }
    except Exception as e:
        print(f"超参数调优失败: {e}")
        return None

# 数据预处理增强功能
def enhanced_preprocessing(X, y, preprocessing_config):
    """增强的数据预处理"""
    try:
        # 缺失值处理
        if preprocessing_config.get('handle_missing', False):
            imputer = SimpleImputer(strategy='mean')
            X = imputer.fit_transform(X)
        
        # 异常值检测
        if preprocessing_config.get('detect_outliers', False):
            # 使用IQR方法检测异常值
            Q1 = np.percentile(X, 25, axis=0)
            Q3 = np.percentile(X, 75, axis=0)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # 标记异常值但不删除，仅记录
            outlier_mask = np.any((X < lower_bound) | (X > upper_bound), axis=1)
            preprocessing_config['outlier_count'] = np.sum(outlier_mask)
        
        # 特征选择
        if preprocessing_config.get('feature_selection', False):
            k = preprocessing_config.get('n_features', min(10, X.shape[1]))
            selector = SelectKBest(score_func=f_classif, k=k)
            X = selector.fit_transform(X, y)
            preprocessing_config['selected_features'] = k
        
        # 数据平衡
        if preprocessing_config.get('balance_data', False):
            balance_method = preprocessing_config.get('balance_method', 'smote')
            if balance_method == 'smote':
                smote = SMOTE(random_state=42)
                X, y = smote.fit_resample(X, y)
            elif balance_method == 'undersample':
                undersampler = RandomUnderSampler(random_state=42)
                X, y = undersampler.fit_resample(X, y)
        
        return X, y, preprocessing_config
    except Exception as e:
        print(f"数据预处理失败: {e}")
        return X, y, preprocessing_config

# 模型对比功能
def compare_models(models_config, X, y, test_size, split_method, metric='accuracy', preprocessing_config=None, cv_folds='5'):
    """对比多个模型的性能"""
    comparison_results = {
        'models': {},
        'comparison_summary': {},
        'best_model': None,
        'comparison_metrics': []
    }
    
    try:
        # 数据预处理
        if preprocessing_config:
            X, y, _ = enhanced_preprocessing(X, y, preprocessing_config)
        
        X_processed, feature_names, preprocessor = preprocess_data(X, y)
        
        # 数据分割
        if split_method == "random":
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y, test_size=test_size, random_state=42
            )
        elif split_method == "stratified":
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y, test_size=test_size, random_state=42, stratify=y
            )
        else:
            # 交叉验证模式，使用全量数据
            X_train, X_test, y_train, y_test = X_processed, X_processed, y, y
        
        # 对每个模型进行训练和评估
        for model_name in models_config:
            try:
                send_progress(10, f'正在训练 {model_name}...')
                
                # 获取模型
                model = MODELS.get(model_name)
                if model is None:
                    continue
                
                # 训练模型
                model.fit(X_train, y_train)
                
                # 预测
                y_pred = model.predict(X_test)
                
                # 计算指标
                if get_task_type_extended(model_name) == 'classification':
                    accuracy = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    
                    model_result = {
                        'model_name': model_name,
                        'accuracy': accuracy,
                        'f1_score': f1,
                        'predictions': y_pred.tolist(),
                        'model_type': get_model_type(model_name)
                    }
                else:  # regression
                    r2 = r2_score(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    
                    model_result = {
                        'model_name': model_name,
                        'r2': r2,
                        'mse': mse,
                        'mae': mae,
                        'predictions': y_pred.tolist(),
                        'model_type': get_model_type(model_name)
                    }
                
                comparison_results['models'][model_name] = model_result
                
            except Exception as e:
                print(f"模型 {model_name} 训练失败: {e}")
                comparison_results['models'][model_name] = {
                    'error': str(e),
                    'model_name': model_name
                }
        
        # 生成对比摘要
        if comparison_results['models']:
            comparison_results['comparison_summary'] = generate_comparison_summary(comparison_results['models'], metric)
            comparison_results['best_model'] = find_best_model(comparison_results['models'], metric)
        
        return comparison_results
        
    except Exception as e:
        print(f"模型对比失败: {e}")
        return {'error': str(e)}

def generate_comparison_summary(models_results, metric):
    """生成对比摘要"""
    summary = {
        'total_models': len(models_results),
        'successful_models': 0,
        'failed_models': 0,
        'metrics_comparison': {}
    }
    
    successful_models = []
    for model_name, result in models_results.items():
        if 'error' not in result:
            summary['successful_models'] += 1
            successful_models.append((model_name, result))
        else:
            summary['failed_models'] += 1
    
    if successful_models:
        # 按主要指标排序
        if metric == 'accuracy' or 'accuracy' in successful_models[0][1]:
            sorted_models = sorted(successful_models, key=lambda x: x[1].get('accuracy', 0), reverse=True)
            summary['metrics_comparison']['accuracy_ranking'] = [
                {'model': name, 'score': result.get('accuracy', 0)} 
                for name, result in sorted_models
            ]
        elif metric == 'f1' or 'f1_score' in successful_models[0][1]:
            sorted_models = sorted(successful_models, key=lambda x: x[1].get('f1_score', 0), reverse=True)
            summary['metrics_comparison']['f1_ranking'] = [
                {'model': name, 'score': result.get('f1_score', 0)} 
                for name, result in sorted_models
            ]
        elif 'r2' in successful_models[0][1]:
            sorted_models = sorted(successful_models, key=lambda x: x[1].get('r2', 0), reverse=True)
            summary['metrics_comparison']['r2_ranking'] = [
                {'model': name, 'score': result.get('r2', 0)} 
                for name, result in sorted_models
            ]
    
    return summary

def find_best_model(models_results, metric):
    """找到最佳模型"""
    best_model = None
    best_score = -1
    
    for model_name, result in models_results.items():
        if 'error' not in result:
            if metric == 'accuracy' and 'accuracy' in result:
                if result['accuracy'] > best_score:
                    best_score = result['accuracy']
                    best_model = model_name
            elif metric == 'f1' and 'f1_score' in result:
                if result['f1_score'] > best_score:
                    best_score = result['f1_score']
                    best_model = model_name
            elif 'r2' in result:
                if result['r2'] > best_score:
                    best_score = result['r2']
                    best_model = model_name
    
    return {
        'model_name': best_model,
        'score': best_score,
        'metric': metric
    }

# 修改 train_and_evaluate 函数
def train_and_evaluate(model_name, X, y, test_size, split_method, metric='accuracy', hyperparams=None, preprocessing_config=None, dataset_name='unknown', cv_folds='5', original_feature_names=None):
    results = {}
    results['warnings'] = []
    
    # 发送开始训练消息
    send_progress(0, f'开始训练 {model_name} 模型...')
    
    # 增强数据预处理
    if preprocessing_config:
        send_progress(5, '正在执行增强数据预处理...')
        X, y, preprocessing_info = enhanced_preprocessing(X, y, preprocessing_config)
        results['preprocessing_info'] = preprocessing_info
    
    # 预处理数据
    send_progress(10, '正在预处理数据...', step=1, total_steps=6, details='数据清洗和特征工程')
    X_processed, feature_names, preprocessor = preprocess_data(X, y)
    
    # 检查缓存
    cache_key = generate_cache_key(model_name, dataset_name, test_size, split_method, preprocessing_config, hyperparams)
    cached_model = get_cached_model(cache_key)
    
    if cached_model:
        send_progress(20, '正在从缓存加载模型...', step=2, total_steps=6, details='使用缓存的训练结果')
        model = cached_model
        results['from_cache'] = True
        results['cache_info'] = {
            'cache_key': cache_key[:8] + '...',
            'cached_at': MODEL_CACHE[cache_key]['created_at'].strftime('%H:%M:%S')
        }
    else:
        # 获取模型
        send_progress(20, '正在初始化模型...', step=2, total_steps=6, details=f'加载 {model_name} 模型')
        model = MODELS.get(model_name)
        if model is None:
            raise ValueError(f"未知模型: {model_name}")
        results['from_cache'] = False

    # 超参数调优
    if hyperparams and hyperparams.get('enable_tuning', False):
        send_progress(25, '正在进行超参数调优...')
        param_grid = hyperparams.get('param_grid', {})
        search_type = hyperparams.get('search_type', 'grid')
        
        tuning_result = hyperparameter_tuning(model, model_name, X_processed, y, param_grid, search_type=search_type)
        if tuning_result:
            model = tuning_result['best_estimator']
            results['hyperparameter_tuning'] = {
                'best_params': tuning_result['best_params'],
                'best_score': tuning_result['best_score'],
                'search_type': search_type
            }
            send_progress(35, f'超参数调优完成，最佳参数: {tuning_result["best_params"]}')

    task_type = get_task_type_extended(model_name)

    if task_type == 'classification':
        # 统计各类别样本数，处理极端不平衡或极少样本类别
        try:
            # y 可能是 np.ndarray 或 pandas Series
            values, counts = np.unique(y, return_counts=True)
            min_count = int(counts.min()) if counts.size > 0 else 0
        except Exception:
            min_count = 0
        if split_method == "random":
            send_progress(30, '正在分割数据集...')
            # 如果任一类别样本过少，不使用 stratify，避免 "least populated class" 错误
            use_stratify = True
            if min_count < 2:
                use_stratify = False
                results['warnings'].append('部分类别样本过少(少于2)，已关闭分层抽样。')
            else:
                # 确保训练集与测试集中每个类别至少有1个样本
                if any((counts * (1 - test_size)) < 1) or any((counts * test_size) < 1):
                    use_stratify = False
                    results['warnings'].append('按当前测试集比例无法保证每类至少1个样本，已关闭分层抽样。')

            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y, test_size=test_size, random_state=42, stratify=(y if use_stratify else None)
            )
            send_progress(50, '正在训练模型...', step=4, total_steps=6, details=f'训练 {model_name} 模型')
            model.fit(X_train, y_train)
            
            
            send_progress(70, '正在预测测试集...', step=5, total_steps=6, details='模型预测和评估')
            y_pred = model.predict(X_test)
            send_progress(80, '正在计算评估指标...')
            # 计算所有指标
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            
            # 存储所有指标
            results['accuracy'] = round(accuracy, 4)
            results['f1_score'] = round(f1, 4)
            results['precision'] = round(precision, 4)
            results['recall'] = round(recall, 4)
            
            # 根据用户选择的指标设置主要指标
            if metric == 'f1':
                score = f1
            elif metric == 'precision':
                score = precision
            elif metric == 'recall':
                score = recall
            elif metric == 'f1_macro':
                score = f1_score(y_test, y_pred, average='macro')
                results['f1_score'] = round(score, 4)
            elif metric == 'f1_weighted':
                score = f1
            else:
                score = accuracy
            
            if split_method == 'random':
                results['split_method'] = '随机分割'
            elif split_method == 'stratified':
                results['split_method'] = '分层分割'
            else:
                results['split_method'] = '随机分割'
            results['test_size'] = test_size
            send_progress(90, '正在生成详细报告...')
            report = classification_report(y_test, y_pred, output_dict=True)
            results['classification_report'] = report
            cm = confusion_matrix(y_test, y_pred)
            results['confusion_matrix'] = cm.tolist()
        else:
            # 使用用户指定的折数
            if cv_folds == 'leave_one_out':
                from sklearn.model_selection import LeaveOneOut
                cv = LeaveOneOut()
                n_splits = len(y)
                send_progress(30, f'正在进行留一法交叉验证...')
                # 留一法使用准确率作为主要指标
                cv_scores = cross_val_score(model, X_processed, y, cv=cv, scoring='accuracy')
            else:
                desired_splits = int(cv_folds)
                n_splits = max(2, min(desired_splits, min_count)) if min_count > 0 else 2
                send_progress(30, f'正在进行{n_splits}折交叉验证...')
                scoring = 'f1_weighted' if metric == 'f1' else 'accuracy'
                try:
                    if split_method == 'stratified_kfold':
                        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                        send_progress(40, '正在执行分层交叉验证...')
                    else:
                        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                        send_progress(40, '正在执行K折交叉验证...')
                    cv_scores = cross_val_score(model, X_processed, y, cv=cv, scoring=scoring)
                    if n_splits < desired_splits:
                        results['warnings'].append(f'最小类别样本仅 {min_count}，交叉验证折数降为 {n_splits}。')
                except ValueError:
                    # 回退到普通 KFold
                    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                    send_progress(40, '正在执行普通交叉验证...')
                    cv_scores = cross_val_score(model, X_processed, y, cv=cv, scoring=scoring)
                    results['warnings'].append('分层交叉验证不可用，已回退至非分层 KFold。')
            send_progress(70, '正在计算交叉验证结果...')
            if metric == 'f1':
                results['f1_score'] = round(cv_scores.mean(), 4)
                accuracy_scores = cross_val_score(model, X_processed, y, cv=cv, scoring='accuracy')
                results['accuracy'] = round(accuracy_scores.mean(), 4)
                results['cv_f1_scores'] = {
                    'values': [round(score, 4) for score in cv_scores],
                    'mean': round(cv_scores.mean(), 4),
                    'std': round(cv_scores.std(), 4)
                }
                results['cv_scores'] = {
                    'values': [round(score, 4) for score in accuracy_scores],
                    'mean': round(accuracy_scores.mean(), 4),
                    'std': round(accuracy_scores.std(), 4)
                }
            elif metric == 'precision':
                precision_scores = cross_val_score(model, X_processed, y, cv=cv, scoring='precision_weighted')
                results['precision'] = round(precision_scores.mean(), 4)
                accuracy_scores = cross_val_score(model, X_processed, y, cv=cv, scoring='accuracy')
                results['accuracy'] = round(accuracy_scores.mean(), 4)
                results['cv_scores'] = {
                    'values': [round(score, 4) for score in accuracy_scores],
                    'mean': round(accuracy_scores.mean(), 4),
                    'std': round(accuracy_scores.std(), 4)
                }
            elif metric == 'recall':
                recall_scores = cross_val_score(model, X_processed, y, cv=cv, scoring='recall_weighted')
                results['recall'] = round(recall_scores.mean(), 4)
                accuracy_scores = cross_val_score(model, X_processed, y, cv=cv, scoring='accuracy')
                results['accuracy'] = round(accuracy_scores.mean(), 4)
                results['cv_scores'] = {
                    'values': [round(score, 4) for score in accuracy_scores],
                    'mean': round(accuracy_scores.mean(), 4),
                    'std': round(accuracy_scores.std(), 4)
                }
            elif metric == 'f1_macro':
                f1_scores = cross_val_score(model, X_processed, y, cv=cv, scoring='f1_macro')
                results['f1_score'] = round(f1_scores.mean(), 4)
                accuracy_scores = cross_val_score(model, X_processed, y, cv=cv, scoring='accuracy')
                results['accuracy'] = round(accuracy_scores.mean(), 4)
                results['cv_scores'] = {
                    'values': [round(score, 4) for score in accuracy_scores],
                    'mean': round(accuracy_scores.mean(), 4),
                    'std': round(accuracy_scores.std(), 4)
                }
            elif metric == 'f1_weighted':
                f1_scores = cross_val_score(model, X_processed, y, cv=cv, scoring='f1_weighted')
                results['f1_score'] = round(f1_scores.mean(), 4)
                accuracy_scores = cross_val_score(model, X_processed, y, cv=cv, scoring='accuracy')
                results['accuracy'] = round(accuracy_scores.mean(), 4)
                results['cv_scores'] = {
                    'values': [round(score, 4) for score in accuracy_scores],
                    'mean': round(accuracy_scores.mean(), 4),
                    'std': round(accuracy_scores.std(), 4)
                }
            else:
                results['accuracy'] = round(cv_scores.mean(), 4)
                f1_scores = cross_val_score(model, X_processed, y, cv=cv, scoring='f1_weighted')
                results['f1_score'] = round(f1_scores.mean(), 4)
                results['cv_scores'] = {
                    'values': [round(score, 4) for score in cv_scores],
                    'mean': round(cv_scores.mean(), 4),
                    'std': round(cv_scores.std(), 4)
                }
                results['cv_f1_scores'] = {
                    'values': [round(score, 4) for score in f1_scores],
                    'mean': round(f1_scores.mean(), 4),
                    'std': round(f1_scores.std(), 4)
                }
            if split_method == 'kfold':
                results['split_method'] = f'{n_splits}折交叉验证'
            elif split_method == 'stratified_kfold':
                results['split_method'] = f'{n_splits}折分层交叉验证'
            else:
                results['split_method'] = f'{n_splits}折交叉验证'
            results['test_size'] = 'N/A'
            
            # 为K折交叉验证生成混淆矩阵（使用交叉验证预测结果）
            send_progress(85, '正在生成混淆矩阵...')
            from sklearn.model_selection import cross_val_predict
            y_pred_cv = cross_val_predict(model, X_processed, y, cv=cv)
            cm = confusion_matrix(y, y_pred_cv)
            results['confusion_matrix'] = cm.tolist()

    elif task_type == 'regression':
        send_progress(30, '正在处理回归任务...')
        # y 必须为数值
        if not isinstance(y, (np.ndarray, list, pd.Series)):
            raise ValueError('回归任务需要数值型标签')
        if split_method == 'random':
            send_progress(40, '正在分割数据集...')
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y, test_size=test_size, random_state=42
            )
            send_progress(60, '正在训练回归模型...')
            model.fit(X_train, y_train)
            send_progress(80, '正在预测测试集...')
            y_pred = model.predict(X_test)
            send_progress(90, '正在计算回归指标...')
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            results['mse'] = round(float(mse), 6)
            results['mae'] = round(float(mae), 6)
            results['rmse'] = round(float(rmse), 6)
            results['r2'] = round(float(r2), 6)
            results['split_method'] = '随机分割'
            results['test_size'] = test_size
        else:
            # 使用用户指定的折数进行回归交叉验证
            if cv_folds == 'leave_one_out':
                from sklearn.model_selection import LeaveOneOut
                kf = LeaveOneOut()
                n_splits = len(y)
                send_progress(40, '正在进行留一法交叉验证...')
            else:
                n_splits = int(cv_folds)
                kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                send_progress(40, f'正在进行{n_splits}折交叉验证...')
            # 使用R2为主指标，同时返回MSE/MAE
            send_progress(50, '正在计算R²分数...')
            r2_scores = cross_val_score(model, X_processed, y, cv=kf, scoring='r2')
            send_progress(70, '正在计算MSE和MAE...')
            neg_mse_scores = cross_val_score(model, X_processed, y, cv=kf, scoring='neg_mean_squared_error')
            neg_mae_scores = cross_val_score(model, X_processed, y, cv=kf, scoring='neg_mean_absolute_error')
            mse_scores = -neg_mse_scores
            mae_scores = -neg_mae_scores
            results['r2'] = round(r2_scores.mean(), 6)
            results['cv_r2_scores'] = {
                'values': [round(float(s), 6) for s in r2_scores],
                'mean': round(float(r2_scores.mean()), 6),
                'std': round(float(r2_scores.std()), 6)
            }
            results['cv_mse_scores'] = {
                'values': [round(float(s), 6) for s in mse_scores],
                'mean': round(float(mse_scores.mean()), 6),
                'std': round(float(mse_scores.std()), 6)
            }
            results['cv_mae_scores'] = {
                'values': [round(float(s), 6) for s in mae_scores],
                'mean': round(float(mae_scores.mean()), 6),
                'std': round(float(mae_scores.std()), 6)
            }
            if cv_folds == 'leave_one_out':
                results['split_method'] = '留一法交叉验证'
            else:
                results['split_method'] = f'{n_splits}折交叉验证'
            results['test_size'] = 'N/A'

    else:  # unsupervised
        send_progress(30, '正在处理无监督任务...')
        # 无监督直接在全量数据上拟合
        if model_name.startswith('主成分分析'):
            send_progress(50, '正在执行PCA降维...')
            model.fit(X_processed)
            explained = getattr(model, 'explained_variance_ratio_', None)
            if explained is not None:
                results['explained_variance_ratio'] = [round(float(x), 6) for x in explained.tolist()]
            results['components'] = int(getattr(model, 'n_components_', getattr(model, 'n_components', 0)))
        elif model_name.startswith('K 均值聚类'):
            send_progress(50, '正在执行K-Means聚类...')
            model.fit(X_processed)
            send_progress(80, '正在分析聚类结果...')
            labels = model.labels_.tolist()
            counts = pd.Series(labels).value_counts().sort_index().to_dict()
            results['clusters'] = int(getattr(model, 'n_clusters', 0))
            results['inertia'] = round(float(model.inertia_), 6)
            results['cluster_counts'] = {str(k): int(v) for k, v in counts.items()}
        else:
            # 其他无监督算法（占位）
            send_progress(50, '正在执行无监督算法...')
            model.fit(X_processed)
        results['split_method'] = 'N/A'
        results['test_size'] = 'N/A'
    
    # 缓存模型（如果不是从缓存加载的）
    if not results.get('from_cache', False):
        cache_model(cache_key, model, results)
        results['cached'] = True
    else:
        results['cached'] = False

    # 添加模型信息
    send_progress(95, '正在生成最终结果...')
    results['model'] = model_name
    n_features = X_processed.shape[1] if hasattr(X_processed, 'shape') else len(X_processed[0])
    results['feature_importance'] = get_feature_importance(model, n_features, feature_names, X_processed, y)
    # 使用原始特征名称，如果预处理后特征数量匹配的话
    if original_feature_names is not None and len(original_feature_names) == n_features:
        results['processed_feature_names'] = original_feature_names if isinstance(original_feature_names, list) else list(original_feature_names)
    elif feature_names is not None:
        results['processed_feature_names'] = feature_names if isinstance(feature_names, list) else list(feature_names)
    else:
        results['processed_feature_names'] = [f'feature_{i}' for i in range(n_features)]
    
    # 模型类型信息
    results['model_type'] = get_model_type(model_name)
    results['task_type'] = get_task_type_extended(model_name)
    
    # 发送完成消息
    send_progress(100, '实验完成！')
    
    return results


def get_model_type(model_name):
    """获取模型类型"""
    tree_based = ["决策树", "随机森林", "梯度提升树(GBDT)"]
    linear_models = ["逻辑回归", "线性回归"]
    neural_models = ["多层感知机(MLP)"]
    distance_based = ["K近邻(KNN)"]
    probabilistic = []
    kernel_based = ["支持向量机(SVM)"]
    ensemble_models = []
    
    if model_name in tree_based:
        return "树模型"
    elif model_name in linear_models:
        return "线性模型"
    elif model_name in neural_models:
        return "神经网络"
    elif model_name in distance_based:
        return "距离模型"
    elif model_name in probabilistic:
        return "概率模型"
    elif model_name in kernel_based:
        return "核方法"
    elif model_name in ensemble_models:
        return "集成模型"
    else:
        return "其他"

def get_task_type_extended(model_name):
    """根据模型名称判断任务类型: classification / regression / unsupervised"""
    classification = {"逻辑回归", "支持向量机(SVM)", "K近邻(KNN)", "决策树", "随机森林", "梯度提升树(GBDT)", "多层感知机(MLP)"}
    regression = {"线性回归"}
    unsupervised = {"K 均值聚类(K-Means)", "主成分分析(PCA)"}
    if model_name in classification:
        return 'classification'
    if model_name in regression:
        return 'regression'
    if model_name in unsupervised:
        return 'unsupervised'
    return 'classification'

@app.route('/')
def serve_frontend():
    return send_from_directory('static', 'index.html')

@app.route('/favicon.ico')
def favicon():
    # 返回一个简单的空响应，避免404错误
    return '', 204

@app.route('/experiment_history', methods=['GET'])
def get_experiment_history():
    """获取历史实验列表"""
    try:
        history_list = []
        for exp_id, exp_data in EXPERIMENT_HISTORY.items():
            history_list.append({
                'id': exp_id,
                'timestamp': exp_data['timestamp'],
                'dataset': exp_data['config']['dataset'],
                'model': exp_data['config']['model'],
                'accuracy': exp_data['results'].get('accuracy', 'N/A'),
                'f1_score': exp_data['results'].get('f1_score', 'N/A'),
                'r2': exp_data['results'].get('r2', 'N/A')
            })
        
        # 按时间倒序排列
        history_list.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return jsonify({
            'success': True,
            'history': history_list
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/run_experiment', methods=['POST'])
# 修改 /run_experiment 路由
def run_experiment():
    try:
        data = request.json
        dataset_name = data.get('dataset', 'iris')
        test_size_str = data.get('test_size', '0.3')
        if isinstance(test_size_str, str):
            test_size = float(test_size_str.strip('%')) / 100
        else:
            test_size = float(test_size_str)
        split_method = data.get('split_method', 'random')
        cv_folds = data.get('cv_folds', '5')
        model_name = data.get('model', '逻辑回归')
        metric = data.get('metric', 'accuracy')  # 新增评估指标参数
        dataset_id = data.get('dataset_id')
        label_column = data.get('label_column')
        
        # 加载数据
        if dataset_name == 'custom':
            if not dataset_id or not label_column:
                raise ValueError('使用自定义数据集时必须提供 dataset_id 和 label_column')
            X, y, feature_names, target_names, raw_df = load_custom_dataset(dataset_id, label_column)
        else:
            X, y, feature_names, target_names, raw_df = load_dataset(dataset_name)
        
        # 获取预处理和超参数配置
        preprocessing_config = data.get('preprocessing', {})
        hyperparams = data.get('hyperparams', {})
        
        # 训练和评估模型，传入评估指标
        results = train_and_evaluate(model_name, X, y, test_size, split_method, metric, hyperparams, preprocessing_config, dataset_name, cv_folds, feature_names)
        
        # 生成实验ID和时间戳
        import time
        experiment_id = f"exp_{int(time.time())}"
        timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 保存实验配置和结果到历史记录
        EXPERIMENT_HISTORY[experiment_id] = {
            'timestamp': timestamp,
            'config': {
                'dataset': dataset_name,
                'test_size': test_size,
                'split_method': split_method,
                'model': model_name,
                'metric': metric,
                'dataset_id': dataset_id,
                'label_column': label_column
            },
            'results': results
        }
        
        # 添加数据集信息
        results['dataset'] = dataset_name
        results['experiment_id'] = experiment_id
        dataset_info = {
            'samples': len(y),
            'features': X.shape[1] if hasattr(X, 'shape') else len(X.columns),
            'feature_names': feature_names if isinstance(feature_names, list) else list(feature_names) if feature_names is not None else [],
            'target_names': target_names if isinstance(target_names, list) else list(target_names) if target_names is not None else []
        }
        
        # 如果是DataFrame，添加原始特征信息
        if isinstance(X, pd.DataFrame):
            dataset_info['original_features'] = X.columns.tolist()
            dataset_info['feature_types'] = {
                'numeric': X.select_dtypes(include=[np.number]).columns.tolist(),
                'categorical': X.select_dtypes(include=['object']).columns.tolist()
            }
        
        results['dataset_info'] = dataset_info
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        import traceback
        resp = {'success': False, 'error': str(e)}
        # 仅在调试模式下返回详细堆栈
        if app.debug:
            resp['traceback'] = traceback.format_exc()
        return jsonify(resp)

@app.route('/repeat_experiment/<experiment_id>', methods=['POST'])
def repeat_experiment(experiment_id):
    """重复运行历史实验"""
    try:
        if experiment_id not in EXPERIMENT_HISTORY:
            return jsonify({
                'success': False,
                'error': '实验记录不存在'
            })
        
        exp_data = EXPERIMENT_HISTORY[experiment_id]
        config = exp_data['config']
        
        # 重新运行实验
        if config['dataset'] == 'custom':
            if not config.get('dataset_id') or not config.get('label_column'):
                return jsonify({
                    'success': False,
                    'error': '自定义数据集信息不完整'
                })
            X, y, feature_names, target_names, raw_df = load_custom_dataset(
                config['dataset_id'], config['label_column']
            )
        else:
            X, y, feature_names, target_names, raw_df = load_dataset(config['dataset'])
        
        # 训练和评估模型
        results = train_and_evaluate(
            config['model'], X, y, config['test_size'], 
            config['split_method'], config['metric'], 
            None, None, config['dataset'], '5', feature_names
        )
        
        # 更新历史记录
        import time
        new_experiment_id = f"exp_{int(time.time())}"
        timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        
        EXPERIMENT_HISTORY[new_experiment_id] = {
            'timestamp': timestamp,
            'config': config,
            'results': results
        }
        
        # 添加数据集信息
        results['dataset'] = config['dataset']
        results['experiment_id'] = new_experiment_id
        dataset_info = {
            'samples': len(y),
            'features': X.shape[1] if hasattr(X, 'shape') else len(X.columns),
            'feature_names': feature_names if isinstance(feature_names, list) else list(feature_names) if feature_names is not None else [],
            'target_names': target_names if isinstance(target_names, list) else list(target_names) if target_names is not None else []
        }
        
        if isinstance(X, pd.DataFrame):
            dataset_info['original_features'] = X.columns.tolist()
            dataset_info['feature_types'] = {
                'numeric': X.select_dtypes(include=[np.number]).columns.tolist(),
                'categorical': X.select_dtypes(include=['object']).columns.tolist()
            }
        
        results['dataset_info'] = dataset_info
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        import traceback
        resp = {'success': False, 'error': str(e)}
        if app.debug:
            resp['traceback'] = traceback.format_exc()
        return jsonify(resp)

@app.route('/available_models')
def get_available_models():
    """获取所有可用的模型列表，按类别分组"""
    try:
        models_by_category = {}
        for category in ['classification', 'regression', 'unsupervised']:
            models = model_registry.get_models_by_category(category)
            models_by_category[category] = [
                {
                    'name': name,
                    'description': model_registry.get_model_info(name).get('description', '')
                }
                for name in models
            ]
        
        return jsonify({
            'success': True,
            'models': models_by_category
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/add_model', methods=['POST'])
def add_model():
    """动态添加新模型"""
    try:
        data = request.json
        model_name = data.get('name')
        model_category = data.get('category', 'classification')
        model_description = data.get('description', '')
        
        if not model_name:
            return jsonify({
                'success': False,
                'error': '模型名称不能为空'
            })
        
        # 这里可以添加更多模型类型的支持
        # 目前只支持基本的sklearn模型
        model_type = data.get('type', 'sklearn')
        
        if model_type == 'sklearn':
            # 可以根据需要扩展支持更多sklearn模型
            return jsonify({
                'success': False,
                'error': '暂不支持动态添加sklearn模型，请使用装饰器注册'
            })
        
        return jsonify({
            'success': True,
            'message': f'模型 {model_name} 添加成功'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    """上传CSV文件，解析后缓存于内存并返回列信息与dataset_id。"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': '未找到上传文件字段 file'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': '未选择文件'}), 400
    if not file.filename.lower().endswith('.csv'):
        return jsonify({'success': False, 'error': '仅支持CSV文件'}), 400

    try:
        df = pd.read_csv(file)
        # 生成简单的dataset_id
        dataset_id = f"ds_{len(UPLOADED_DATASETS) + 1}"
        UPLOADED_DATASETS[dataset_id] = df
        columns = df.columns.tolist()
        # 简单推断可能的标签列（最后一列）
        suggested_label = columns[-1] if columns else None
        return jsonify({
            'success': True,
            'dataset_id': dataset_id,
            'columns': columns,
            'suggested_label': suggested_label,
            'samples': int(df.shape[0]),
            'features': int(df.shape[1])
        })
    except Exception as e:
        import traceback
        resp = {'success': False, 'error': str(e)}
        if app.debug:
            resp['traceback'] = traceback.format_exc()
        return jsonify(resp), 400

@app.route('/compare_models', methods=['POST'])
def compare_models_endpoint():
    """模型对比API"""
    try:
        data = request.json
        models = data.get('models', [])
        dataset_name = data.get('dataset', 'iris')
        test_size_str = data.get('test_size', '0.3')
        if isinstance(test_size_str, str):
            test_size = float(test_size_str.strip('%')) / 100
        else:
            test_size = float(test_size_str)
        split_method = data.get('split_method', 'random')
        cv_folds = data.get('cv_folds', '5')
        metric = data.get('metric', 'accuracy')
        dataset_id = data.get('dataset_id')
        label_column = data.get('label_column')
        preprocessing_config = data.get('preprocessing', {})
        
        if not models:
            return jsonify({'success': False, 'error': '请选择至少一个模型进行对比'})
        
        # 加载数据
        if dataset_name == 'custom':
            if not dataset_id or not label_column:
                raise ValueError('使用自定义数据集时必须提供 dataset_id 和 label_column')
            X, y, feature_names, target_names, raw_df = load_custom_dataset(dataset_id, label_column)
        else:
            X, y, feature_names, target_names, raw_df = load_dataset(dataset_name)
        
        # 执行模型对比
        results = compare_models(models, X, y, test_size, split_method, metric, preprocessing_config, cv_folds)
        
        # 生成对比实验ID
        import time
        comparison_id = f"comp_{int(time.time())}"
        timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 保存对比结果
        COMPARISON_EXPERIMENTS[comparison_id] = {
            'timestamp': timestamp,
            'config': {
                'models': models,
                'dataset': dataset_name,
                'test_size': test_size,
                'split_method': split_method,
                'metric': metric,
                'dataset_id': dataset_id,
                'label_column': label_column,
                'preprocessing': preprocessing_config
            },
            'results': results
        }
        
        # 添加数据集信息
        results['dataset'] = dataset_name
        results['comparison_id'] = comparison_id
        results['dataset_info'] = {
            'samples': len(y),
            'features': X.shape[1] if hasattr(X, 'shape') else len(X.columns),
            'feature_names': feature_names if isinstance(feature_names, list) else list(feature_names) if feature_names is not None else [],
            'target_names': target_names if isinstance(target_names, list) else list(target_names) if target_names is not None else []
        }
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        import traceback
        resp = {'success': False, 'error': str(e)}
        if app.debug:
            resp['traceback'] = traceback.format_exc()
        return jsonify(resp)

@app.route('/comparison_history', methods=['GET'])
def get_comparison_history():
    """获取模型对比历史"""
    try:
        history_list = []
        for comp_id, comp_data in COMPARISON_EXPERIMENTS.items():
            history_list.append({
                'id': comp_id,
                'timestamp': comp_data['timestamp'],
                'models': comp_data['config']['models'],
                'dataset': comp_data['config']['dataset'],
                'best_model': comp_data['results'].get('best_model', {}).get('model_name', 'N/A'),
                'best_score': comp_data['results'].get('best_model', {}).get('score', 'N/A')
            })
        
        # 按时间倒序排列
        history_list.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return jsonify({
            'success': True,
            'history': history_list
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/cache_info', methods=['GET'])
def get_cache_info():
    """获取缓存信息"""
    try:
        cache_info = []
        for cache_key, cache_data in MODEL_CACHE.items():
            cache_info.append({
                'key': cache_key[:8] + '...',
                'model': cache_data['results'].get('model', 'Unknown'),
                'dataset': cache_data['results'].get('dataset', 'Unknown'),
                'created_at': cache_data['created_at'].strftime('%Y-%m-%d %H:%M:%S'),
                'last_used': cache_data['last_used'].strftime('%Y-%m-%d %H:%M:%S')
            })
        
        return jsonify({
            'success': True,
            'cache_count': len(MODEL_CACHE),
            'cache_limit': CACHE_SIZE_LIMIT,
            'cache_info': cache_info
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    """清空模型缓存"""
    try:
        clear_model_cache()
        return jsonify({
            'success': True,
            'message': '缓存已清空'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

# 全局错误处理
@app.errorhandler(400)
def handle_bad_request(e):
    return jsonify({
        'success': False,
        'error': '请求参数错误',
        'details': str(e)
    }), 400

@app.errorhandler(404)
def handle_not_found(e):
    return jsonify({
        'success': False,
        'error': '请求的资源不存在',
        'details': str(e)
    }), 404

@app.errorhandler(413)
def handle_file_too_large(e):
    return jsonify({
        'success': False,
        'error': '文件过大，超过服务器限制',
        'limit_mb': int(app.config.get('MAX_CONTENT_LENGTH', 0) / (1024*1024))
    }), 413

@app.errorhandler(500)
def handle_internal_error(e):
    return jsonify({
        'success': False,
        'error': '服务器内部错误',
        'details': str(e) if app.debug else '请联系管理员'
    }), 500

@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({
        'success': False,
        'error': '未知错误',
        'details': str(e) if app.debug else '请联系管理员'
    }), 500

if __name__ == '__main__':
    # 确保datasets目录存在
    os.makedirs('datasets', exist_ok=True)
    
    socketio.run(app, debug=True, port=5050, allow_unsafe_werkzeug=True)