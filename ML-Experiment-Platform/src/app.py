from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_wine,load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import os

def convert_numpy_types(obj):
    """递归转换NumPy类型为Python原生类型，用于JSON序列化"""
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.generic,)):
        # 处理其他NumPy标量类型
        return obj.item()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

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
# 自动检测可用的异步模式（eventlet, gevent, threading等）
# 如果没有安装eventlet，会自动回退到threading模式
socketio = SocketIO(
    app, 
    cors_allowed_origins="*",  # 开发环境允许所有源，生产环境应限制为特定域名
    # async_mode 不指定，让Flask-SocketIO自动检测最佳模式
    logger=True,  # 启用日志
    engineio_logger=False,  # 关闭详细日志以避免过多输出
    ping_timeout=60,  # 心跳超时时间（秒）
    ping_interval=25  # 心跳间隔（秒）
)

# 临时存储上传的数据集（仅开发模式内存保存）
UPLOADED_DATASETS = {}


# 模型对比实验存储
COMPARISON_EXPERIMENTS = {}

# 实验历史记录存储（最大保存100个实验）
EXPERIMENT_HISTORY = {}
MAX_HISTORY_SIZE = 100


# ==================== 模型注册 ====================

class MyLogisticRegression:
    """原创实现：逻辑回归算法（改进版：支持L2正则化、学习率衰减），支持多分类（One-vs-Rest策略）"""
    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-4, 
                 alpha=0.0, learning_rate_decay=0.95, learning_rate_schedule='constant'):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        # ========== 算法改进：L2正则化 ==========
        self.alpha = alpha  # L2正则化系数，0表示不使用正则化
        # ========== 算法改进：学习率衰减 ==========
        self.learning_rate_decay = learning_rate_decay  # 学习率衰减率
        self.learning_rate_schedule = learning_rate_schedule  # 'constant', 'decay', 'adaptive'
        self.current_learning_rate = learning_rate  # 当前学习率
        
        self.weights = None
        self.bias = None
        self.cost_history = []
        self.classes_ = None
        self.n_classes_ = None
        self.multiclass_ = False
        self.estimators_ = []  # 用于存储多个二分类器（多分类时）
    
    def sigmoid(self, z):
        """Sigmoid激活函数"""
        # 防止数值溢出
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))
    
    def _fit_binary(self, X, y_binary):
        """训练单个二分类逻辑回归（改进版：支持L2正则化、学习率衰减），返回模型参数"""
        m, n = X.shape
        weights = np.zeros(n)
        bias = 0
        cost_history = []
        current_lr = self.learning_rate
        
        for i in range(self.max_iter):
            # 前向传播
            z = np.dot(X, weights) + bias
            h = self.sigmoid(z)
            
            # ========== 算法改进：添加L2正则化项到损失 ==========
            epsilon = 1e-15  # 防止log(0)
            h_clipped = np.clip(h, epsilon, 1 - epsilon)
            # 交叉熵损失 + L2正则化
            cross_entropy = -np.mean(y_binary * np.log(h_clipped) + (1 - y_binary) * np.log(1 - h_clipped))
            l2_penalty = (self.alpha / (2 * m)) * np.sum(weights ** 2)
            cost = cross_entropy + l2_penalty
            cost_history.append(cost)
            
            # 反向传播
            dw = (1/m) * np.dot(X.T, (h - y_binary))
            db = (1/m) * np.sum(h - y_binary)
            
            # ========== 算法改进：添加L2正则化项到梯度 ==========
            if self.alpha > 0:
                dw += (self.alpha / m) * weights  # L2正则化梯度
            
            # ========== 算法改进：学习率衰减 ==========
            if self.learning_rate_schedule == 'decay':
                # 指数衰减：lr = lr0 * (decay ^ iteration)
                current_lr = self.learning_rate * (self.learning_rate_decay ** i)
            elif self.learning_rate_schedule == 'adaptive':
                # 自适应：如果损失没有改善，则衰减
                if i > 0 and cost_history[-1] >= cost_history[-2]:
                    current_lr *= self.learning_rate_decay
            
            # 更新参数（使用当前学习率）
            weights -= current_lr * dw
            bias -= current_lr * db
            
            # 早停条件
            if i > 0 and abs(cost_history[-1] - cost_history[-2]) < self.tol:
                break
        
        return {'weights': weights, 'bias': bias}
    
    def fit(self, X, y):
        """训练逻辑回归模型"""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        
        # 获取唯一类别
        unique_classes = np.unique(y)
        self.classes_ = unique_classes
        self.n_classes_ = len(unique_classes)
        
        # 检查类别数量
        if self.n_classes_ < 2:
            raise ValueError(f"逻辑回归需要至少2个类别，但数据中只有{self.n_classes_}个类别")
        
        # 如果是二分类，使用原来的方法
        if self.n_classes_ == 2:
            self.multiclass_ = False
            # 将类别标签转换为0和1
            y_binary = np.where(y == unique_classes[0], 0, 1)
        else:
            # 多分类情况，需要调用_fit_multiclass
            self.multiclass_ = True
            self._fit_multiclass(X, y)
            return
            
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0
        self.cost_history = []
        self.accuracy_history = []
        self.gradient_norm_history = []
        self.weight_norm_history = []
        self.second_derivative_approx = []
        self.current_learning_rate = self.learning_rate  # 初始化当前学习率
        
        for i in range(self.max_iter):
            # ========== 算法改进：学习率衰减 ==========
            if self.learning_rate_schedule == 'decay':
                # 指数衰减
                self.current_learning_rate = self.learning_rate * (self.learning_rate_decay ** i)
            elif self.learning_rate_schedule == 'adaptive':
                # 自适应：如果损失没有改善，则衰减
                if i > 0 and len(self.cost_history) > 0:
                    if self.cost_history[-1] >= (self.cost_history[-2] if len(self.cost_history) > 1 else float('inf')):
                        self.current_learning_rate *= self.learning_rate_decay
            
            # 前向传播
            z = np.dot(X, self.weights) + self.bias
            h = self.sigmoid(z)
            
            # ========== 算法改进：添加L2正则化项到损失 ==========
            epsilon = 1e-15  # 防止log(0)
            h_clipped = np.clip(h, epsilon, 1 - epsilon)
            # 交叉熵损失 + L2正则化
            cross_entropy = -np.mean(y_binary * np.log(h_clipped) + (1 - y_binary) * np.log(1 - h_clipped))
            l2_penalty = (self.alpha / (2 * m)) * np.sum(self.weights ** 2)
            cost = cross_entropy + l2_penalty
            self.cost_history.append(cost)
            
            # 计算二阶导数近似（损失函数的变化率）
            if i > 0:
                second_deriv = abs(self.cost_history[-1] - 2 * self.cost_history[-2] + (self.cost_history[-3] if i > 1 else self.cost_history[-2]))
                self.second_derivative_approx.append(second_deriv)
            else:
                self.second_derivative_approx.append(0.0)
            
            # 计算准确率
            predictions = (h >= 0.5).astype(int)
            accuracy = np.mean(predictions == y_binary)
            self.accuracy_history.append(accuracy)
            
            # 反向传播
            dw = (1/m) * np.dot(X.T, (h - y_binary))
            db = (1/m) * np.sum(h - y_binary)
            
            # ========== 算法改进：添加L2正则化项到梯度 ==========
            if self.alpha > 0:
                dw += (self.alpha / m) * self.weights  # L2正则化梯度
            
            # 记录梯度范数
            gradient_norm = np.sqrt(np.sum(dw**2) + db**2)
            self.gradient_norm_history.append(gradient_norm)
            
            # 更新参数（使用当前学习率）
            self.weights -= self.current_learning_rate * dw
            self.bias -= self.current_learning_rate * db
            
            # 记录参数范数
            param_norm = np.sqrt(np.sum(self.weights**2) + self.bias**2)
            self.weight_norm_history.append(param_norm)
            
            # 早停条件
            if i > 0 and abs(self.cost_history[-1] - self.cost_history[-2]) < self.tol:
                break
        
        return self
    
    def _fit_multiclass(self, X, y):
        """多分类训练（One-vs-Rest策略）"""
        unique_classes = np.unique(y)
        self.multiclass_ = True
        self.estimators_ = []
        
        # 为每个类别训练一个二分类器
        for class_label in unique_classes:
            # 创建二分类标签：当前类别为1，其他类别为0
            y_binary = (y == class_label).astype(int)
            
            # 训练二分类逻辑回归
            estimator = self._fit_binary(X, y_binary)
            self.estimators_.append(estimator)
        
        return self
    
    def predict_proba(self, X):
        """预测概率"""
        X = np.asarray(X, dtype=float)
        
        if not self.multiclass_:
            # 二分类
            z = np.dot(X, self.weights) + self.bias
            proba = self.sigmoid(z)
            # 返回两列概率矩阵：(负类概率, 正类概率)，以兼容sklearn格式
            proba = proba.reshape(-1, 1) if proba.ndim == 1 else proba
            return np.column_stack([1 - proba, proba])
        else:
            # 多分类：使用One-vs-Rest
            n_samples = X.shape[0]
            probas = np.zeros((n_samples, self.n_classes_))
            
            for i, estimator in enumerate(self.estimators_):
                z = np.dot(X, estimator['weights']) + estimator['bias']
                proba = self.sigmoid(z)
                probas[:, i] = proba
            
            # 归一化概率（softmax风格的归一化）
            probas = probas / probas.sum(axis=1, keepdims=True)
            return probas
    
    def predict(self, X):
        """预测类别"""
        X = np.asarray(X, dtype=float)
        
        if not self.multiclass_:
            # 二分类
            if len(self.classes_) < 2:
                # 如果只有一个类别，直接返回该类别的预测
                return np.full(X.shape[0], self.classes_[0])
            proba = self.predict_proba(X)
            # proba现在是两列矩阵：(负类概率, 正类概率)
            # 取第二列（正类概率）进行判断，然后映射回原始类别标签
            predictions_binary = (proba[:, 1] >= 0.5).astype(int)
            # 映射回原始类别标签
            predictions = np.where(predictions_binary == 1, self.classes_[1], self.classes_[0])
            return predictions
        else:
            # 多分类：选择概率最高的类别
            probas = self.predict_proba(X)
            predictions_idx = np.argmax(probas, axis=1)
            return self.classes_[predictions_idx]
    
    def get_training_history(self):
        """获取训练历史数据，包含详细的训练过程信息"""
        return {
            'cost_history': [float(x) for x in self.cost_history],
            'accuracy_history': [float(x) for x in self.accuracy_history],
            'gradient_norm_history': [float(x) for x in self.gradient_norm_history],
            'weight_norm_history': [float(x) for x in self.weight_norm_history],
            'second_derivative_approx': [float(x) for x in self.second_derivative_approx],
            'iterations': len(self.cost_history),
            'algorithm': 'logistic_regression'
        }
    
    def get_params(self, deep=True):
        """获取参数（sklearn兼容）"""
        return {
            'learning_rate': self.learning_rate,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'alpha': self.alpha,
            'learning_rate_decay': self.learning_rate_decay,
            'learning_rate_schedule': self.learning_rate_schedule
        }
    
    def set_params(self, **params):
        """设置参数（sklearn兼容）"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

class MyDecisionTree:
    """原创实现：决策树算法（ID3，改进版：支持后剪枝）"""
    def __init__(self, max_depth=None, min_samples_split=2, ccp_alpha=0.0, min_impurity_decrease=0.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        # ========== 算法改进：后剪枝参数 ==========
        self.ccp_alpha = ccp_alpha  # 复杂度参数，用于剪枝（0表示不剪枝）
        self.min_impurity_decrease = min_impurity_decrease  # 最小不纯度减少量
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
        """递归构建决策树（改进版：记录节点信息用于剪枝）"""
        # 计算节点的不纯度（熵）
        node_entropy = self.entropy(y)
        
        # 计算节点的多数类和样本数（用于剪枝评估）
        unique, counts = np.unique(y, return_counts=True)
        majority_class = unique[np.argmax(counts)]
        node_samples = len(y)
        class_counts = {int(k): int(v) for k, v in zip(unique, counts)}
        
        # 停止条件
        if (self.max_depth is not None and depth >= self.max_depth) or \
           len(y) < self.min_samples_split or \
           len(np.unique(y)) == 1:
            # 返回叶节点（多数类）
            return {
                'class': majority_class, 
                'is_leaf': True,
                'samples': node_samples,
                'class_counts': class_counts,
                'entropy': node_entropy
            }
        
        # 找到最佳分割
        feature, threshold, gain = self.find_best_split(X, y)
        
        # ========== 算法改进：考虑最小不纯度减少量 ==========
        if gain == 0 or gain < self.min_impurity_decrease:
            # 无法进一步分割或增益太小
            return {
                'class': majority_class, 
                'is_leaf': True,
                'samples': node_samples,
                'class_counts': class_counts,
                'entropy': node_entropy
            }
        
        # 分割数据
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        # 递归构建子树
        left_tree = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self.build_tree(X[right_mask], y[right_mask], depth + 1)
        
        # 计算加权不纯度（用于剪枝评估）
        left_samples = left_tree.get('samples', 0)
        right_samples = right_tree.get('samples', 0)
        left_entropy = left_tree.get('entropy', 0)
        right_entropy = right_tree.get('entropy', 0)
        weighted_entropy = (left_samples * left_entropy + right_samples * right_entropy) / node_samples
        
        return {
            'feature': feature,
            'threshold': threshold,
            'left': left_tree,
            'right': right_tree,
            'is_leaf': False,
            'samples': node_samples,
            'class_counts': class_counts,
            'entropy': node_entropy,
            'weighted_entropy': weighted_entropy,
            'impurity_decrease': gain  # 信息增益
        }
    
    def _calculate_leaf_value(self, tree):
        """计算节点的预测值（多数类）"""
        if tree.get('is_leaf', False):
            return tree.get('class')
        # 如果不是叶节点，递归查找
        class_counts = tree.get('class_counts', {})
        if class_counts:
            return max(class_counts.items(), key=lambda x: x[1])[0]
        return 0
    
    def _prune_tree(self, tree, X_val, y_val, feature_idx_map=None):
        """后剪枝：使用验证集评估是否应该剪枝"""
        if tree is None or tree.get('is_leaf', False):
            return tree
        
        # 获取特征索引映射（用于验证集）
        feature = tree.get('feature')
        threshold = tree.get('threshold')
        left_tree = tree.get('left')
        right_tree = tree.get('right')
        
        # 递归剪枝子树
        if left_tree and not left_tree.get('is_leaf', False):
            left_tree = self._prune_tree(left_tree, X_val, y_val, feature_idx_map)
            tree['left'] = left_tree
        
        if right_tree and not right_tree.get('is_leaf', False):
            right_tree = self._prune_tree(right_tree, X_val, y_val, feature_idx_map)
            tree['right'] = right_tree
        
        # 计算不剪枝的验证集准确率
        if X_val is not None and len(X_val) > 0:
            # 获取当前节点的预测准确率（作为内部节点）
            left_mask = X_val[:, feature] <= threshold
            right_mask = ~left_mask
            
            # 如果子树都是叶节点，考虑是否合并
            if left_tree and right_tree and left_tree.get('is_leaf', False) and right_tree.get('is_leaf', False):
                # 计算合并为叶节点的准确率
                leaf_value = self._calculate_leaf_value(tree)
                leaf_accuracy = np.mean((y_val == leaf_value).astype(float))
                
                # 计算不合并的准确率
                left_accuracy = np.mean((y_val[left_mask] == left_tree.get('class')).astype(float)) if np.sum(left_mask) > 0 else 0
                right_accuracy = np.mean((y_val[right_mask] == right_tree.get('class')).astype(float)) if np.sum(right_mask) > 0 else 0
                split_accuracy = (np.sum(left_mask) * left_accuracy + np.sum(right_mask) * right_accuracy) / len(y_val)
                
                # ========== 算法改进：如果合并后准确率提升，则剪枝 ==========
                # 考虑复杂度惩罚（ccp_alpha）
                complexity_penalty = self.ccp_alpha * (2 if not tree.get('is_leaf', False) else 0)  # 内部节点增加复杂度
                
                if leaf_accuracy + complexity_penalty >= split_accuracy:
                    # 剪枝：合并为叶节点
                    return {
                        'class': leaf_value,
                        'is_leaf': True,
                        'samples': tree.get('samples', 0),
                        'class_counts': tree.get('class_counts', {}),
                        'entropy': tree.get('entropy', 0),
                        'pruned': True  # 标记为被剪枝
                    }
        
        return tree
    
    def fit(self, X, y, X_val=None, y_val=None):
        """训练决策树（改进版：支持后剪枝）"""
        # 构建完整树
        self.tree = self.build_tree(X, y)
        self.feature_names_ = None  # 将在外部设置
        self.y_train = y  # 保存训练标签，用于predict_proba
        
        # ========== 算法改进：如果提供了验证集且ccp_alpha > 0，执行后剪枝 ==========
        if self.ccp_alpha > 0 and X_val is not None and y_val is not None:
            X_val = np.asarray(X_val, dtype=float)
            y_val = np.asarray(y_val)
            self.tree = self._prune_tree(self.tree, X_val, y_val)
    
    def get_tree_structure(self, tree=None, feature_names=None, target_names=None, node_id=0, X=None, y=None):
        """提取决策树结构信息用于可视化"""
        if tree is None:
            tree = self.tree
        if tree is None:
            return None
        
        # 存储特征名称（如果可用）
        if feature_names is not None:
            self.feature_names_ = feature_names
        
        if tree.get('is_leaf', False):
            # 叶节点
            class_label = tree.get('class', 'Unknown')
            if target_names and isinstance(class_label, (int, np.integer)):
                class_label = target_names[class_label] if class_label < len(target_names) else str(class_label)
            else:
                class_label = str(class_label)
            
            # 计算样本数（如果提供了数据）
            samples = tree.get('samples', 0)
            class_counts = tree.get('class_counts', {})
            
            return {
                'id': node_id,
                'type': 'leaf',
                'class': class_label,
                'samples': samples,
                'class_counts': class_counts
            }
        else:
            # 内部节点
            feature_idx = tree.get('feature')
            threshold = tree.get('threshold')
            feature_name = feature_names[feature_idx] if feature_names and feature_idx < len(feature_names) else f'特征{feature_idx}'
            
            # 递归获取子树
            left_tree = tree.get('left')
            right_tree = tree.get('right')
            
            # 计算左右子树的节点ID
            left_id = node_id * 2 + 1
            right_id = node_id * 2 + 2
            
            # 递归提取子树结构
            left_structure = self.get_tree_structure(left_tree, feature_names, target_names, left_id, X, y) if left_tree else None
            right_structure = self.get_tree_structure(right_tree, feature_names, target_names, right_id, X, y) if right_tree else None
            
            # 计算样本数
            samples = tree.get('samples', 0)
            class_counts = tree.get('class_counts', {})
            
            return {
                'id': node_id,
                'type': 'internal',
                'feature': feature_name,
                'feature_idx': feature_idx,
                'threshold': float(threshold),
                'samples': samples,
                'class_counts': class_counts,
                'left': left_structure,
                'right': right_structure
            }
    
    def enrich_tree_with_samples(self, tree, X, y, feature_names=None):
        """为树节点添加样本统计信息"""
        if tree is None or X is None or y is None:
            return tree
        
        if tree.get('is_leaf', False):
            # 叶节点：统计类别分布
            unique, counts = np.unique(y, return_counts=True)
            tree['samples'] = len(y)
            tree['class_counts'] = {str(int(u)): int(c) for u, c in zip(unique, counts)}
            return tree
        else:
            # 内部节点
            feature_idx = tree.get('feature')
            threshold = tree.get('threshold')
            
            # 统计当前节点的样本
            unique, counts = np.unique(y, return_counts=True)
            tree['samples'] = len(y)
            tree['class_counts'] = {str(int(u)): int(c) for u, c in zip(unique, counts)}
            
            # 分割数据
            left_mask = X[:, feature_idx] <= threshold
            right_mask = ~left_mask
            
            # 递归处理子树
            left_tree = tree.get('left')
            right_tree = tree.get('right')
            
            if left_tree:
                tree['left'] = self.enrich_tree_with_samples(left_tree, X[left_mask], y[left_mask], feature_names)
            if right_tree:
                tree['right'] = self.enrich_tree_with_samples(right_tree, X[right_mask], y[right_mask], feature_names)
            
            return tree
    
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
    
    def predict_proba(self, X):
        """预测概率（基于叶节点的类别分布）"""
        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]
        all_probas = []
        
        # 获取所有唯一类别
        unique_classes = sorted(set(self.y_train)) if hasattr(self, 'y_train') else []
        if len(unique_classes) == 0:
            # 如果没有y_train，从树中提取类别
            unique_classes = self._extract_classes_from_tree(self.tree)
        n_classes = len(unique_classes)
        
        for x in X:
            # 找到样本到达的叶节点
            leaf = self._find_leaf(x, self.tree)
            
            if leaf.get('is_leaf', False):
                # 如果叶节点有类别分布信息
                if 'class_counts' in leaf and leaf['samples'] > 0:
                    proba = np.zeros(n_classes)
                    total_samples = leaf['samples']
                    for i, cls in enumerate(unique_classes):
                        count = leaf['class_counts'].get(str(int(cls)), 0)
                        proba[i] = count / total_samples
                    all_probas.append(proba)
                else:
                    # 如果没有分布信息，使用硬预测（一个类别为1，其他为0）
                    predicted_class = leaf.get('class', unique_classes[0] if len(unique_classes) > 0 else 0)
                    proba = np.zeros(n_classes)
                    if predicted_class in unique_classes:
                        idx = unique_classes.index(predicted_class)
                        proba[idx] = 1.0
                    else:
                        proba[0] = 1.0  # 默认第一个类别
                    all_probas.append(proba)
            else:
                # 如果无法找到叶节点（理论上不应该发生）
                proba = np.ones(n_classes) / n_classes if n_classes > 0 else np.array([1.0])
                all_probas.append(proba)
        
        return np.array(all_probas)
    
    def get_params(self, deep=True):
        """获取参数（sklearn兼容）"""
        return {
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'ccp_alpha': self.ccp_alpha,
            'min_impurity_decrease': self.min_impurity_decrease
        }
    
    def set_params(self, **params):
        """设置参数（sklearn兼容）"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
    
    def _find_leaf(self, x, tree):
        """找到样本x到达的叶节点"""
        if tree.get('is_leaf', False):
            return tree
        
        feature = tree.get('feature')
        threshold = tree.get('threshold')
        
        if x[feature] <= threshold:
            return self._find_leaf(x, tree.get('left', tree))
        else:
            return self._find_leaf(x, tree.get('right', tree))
    
    def _extract_classes_from_tree(self, tree):
        """从树中提取所有类别"""
        classes = set()
        
        if tree.get('is_leaf', False):
            cls = tree.get('class')
            if cls is not None:
                classes.add(cls)
        else:
            if 'left' in tree:
                classes.update(self._extract_classes_from_tree(tree['left']))
            if 'right' in tree:
                classes.update(self._extract_classes_from_tree(tree['right']))
        
        return sorted(list(classes))

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
    
    def predict_proba(self, X):
        """预测概率（基于K近邻的投票比例）"""
        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]
        all_probas = []
        
        # 获取所有唯一类别
        unique_classes = sorted(set(self.y_train))
        n_classes = len(unique_classes)
        
        for x in X:
            # 计算与所有训练样本的距离
            distances = []
            for i, train_x in enumerate(self.X_train):
                dist = self.calculate_distance(x, train_x)
                distances.append((dist, self.y_train[i]))
            
            # 按距离排序，取前k个
            distances.sort(key=lambda x: x[0])
            k_nearest = distances[:self.k]
            
            # 统计每个类别的票数
            votes = {}
            for _, label in k_nearest:
                votes[label] = votes.get(label, 0) + 1
            
            # 计算概率（归一化投票数）
            proba = np.zeros(n_classes)
            for i, cls in enumerate(unique_classes):
                proba[i] = votes.get(cls, 0) / self.k
            
            all_probas.append(proba)
        
        return np.array(all_probas)
    
    def get_params(self, deep=True):
        """获取参数（sklearn兼容）"""
        # weights参数由子类KNNModel处理，这里不返回
        return {
            'k': self.k,
            'n_neighbors': self.k,  # sklearn 兼容：同时支持 k 和 n_neighbors
            'distance_metric': self.distance_metric
        }
    
    def set_params(self, **params):
        """设置参数（sklearn兼容）"""
        for key, value in params.items():
            if key == 'n_neighbors':  # sklearn 兼容：n_neighbors -> k
                self.k = value
            elif key == 'k':
                self.k = value
            elif key == 'weights':  # sklearn 兼容：weights -> distance_metric (部分支持)
                if value == 'distance':
                    self.distance_metric = 'euclidean'  # 使用欧几里得距离作为加权
                # 'uniform' 保持不变，使用默认的 euclidean
            elif hasattr(self, key):
                setattr(self, key, value)
        return self

class MySVM:
    """原创实现：支持向量机算法（SMO算法），支持多分类（One-vs-Rest策略）"""
    def __init__(self, C=1.0, kernel='rbf', gamma='scale', tol=1e-3, max_iter=1000, random_state=42):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.support_vectors_ = None
        self.support_vector_labels_ = None
        self.dual_coef_ = None  # 拉格朗日乘数 alpha * y
        self.intercept_ = None
        self.gamma_value = None
        self.X_train = None
        self.y_train = None
        self.classes_ = None
        self.n_classes_ = None
        self.estimators_ = []  # 用于存储多分类时的多个二分类器
        self.multiclass_ = False  # 标记是否为多分类
        
    def _kernel_function(self, x1, x2):
        """核函数"""
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        elif self.kernel == 'rbf':
            if self.gamma_value is None:
                return 0
            diff = x1 - x2
            squared_dist = np.dot(diff, diff)
            # 防止指数爆炸：限制gamma * squared_dist的最大值
            exponent = -self.gamma_value * squared_dist
            # 限制指数范围在[-700, 0]之间，避免exp溢出或下溢
            exponent = np.clip(exponent, -700, 0)
            return np.exp(exponent)
        else:
            return np.dot(x1, x2)  # 默认线性核
    
    def _kernel_matrix(self, X1, X2):
        """计算核矩阵"""
        n1 = X1.shape[0]
        n2 = X2.shape[0]
        K = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                K[i, j] = self._kernel_function(X1[i], X2[j])
        return K
    
    def _compute_E(self, alpha, y, K, b):
        """计算预测误差 E_i = f(x_i) - y_i"""
        # 限制b的范围，避免数值溢出
        b_clipped = np.clip(b, -1e4, 1e4)
        f = np.dot(K, (alpha * y)) + b_clipped
        # 检查结果是否有效
        f = np.clip(f, -1e4, 1e4)
        return f - y
    
    def _take_step(self, i1, i2, alpha, y, K, E, b):
        """SMO算法的单步优化"""
        if i1 == i2:
            return False, alpha, b
        
        alpha1_old = alpha[i1]
        alpha2_old = alpha[i2]
        y1, y2 = y[i1], y[i2]
        E1, E2 = E[i1], E[i2]
        
        # 计算边界
        if y1 != y2:
            L = max(0, alpha2_old - alpha1_old)
            H = min(self.C, self.C + alpha2_old - alpha1_old)
        else:
            L = max(0, alpha1_old + alpha2_old - self.C)
            H = min(self.C, alpha1_old + alpha2_old)
        
        if L == H:
            return False, alpha, b
        
        # 计算 eta
        k11 = K[i1, i1]
        k12 = K[i1, i2]
        k22 = K[i2, i2]
        eta = k11 + k22 - 2 * k12
        
        if eta > 0:
            # 更新 alpha2
            alpha2_new = alpha2_old + y2 * (E1 - E2) / eta
            # 裁剪到边界
            if alpha2_new >= H:
                alpha2_new = H
            elif alpha2_new <= L:
                alpha2_new = L
        else:
            # 如果eta <= 0，使用启发式方法
            return False, alpha, b
        
        # 检查 alpha2 是否有足够变化
        if abs(alpha2_new - alpha2_old) < 1e-5: 
            return False, alpha, b
        
        # 更新 alpha1
        alpha1_new = alpha1_old + y1 * y2 * (alpha2_old - alpha2_new)
        
        # 更新 alpha
        alpha[i1] = alpha1_new
        alpha[i2] = alpha2_new
        
        # 更新 b（添加数值稳定性检查）
        b1 = b - E1 - y1 * (alpha1_new - alpha1_old) * k11 - y2 * (alpha2_new - alpha2_old) * k12
        b2 = b - E2 - y1 * (alpha1_new - alpha1_old) * k12 - y2 * (alpha2_new - alpha2_old) * k22
        
        # 检查数值稳定性，防止bias爆炸
        if np.isnan(b1) or np.isinf(b1):
            b1 = b
        if np.isnan(b2) or np.isinf(b2):
            b2 = b
        
        # 限制bias的范围（避免极端值，使用更合理的范围）
        max_bias = 1e4
        b1 = np.clip(b1, -max_bias, max_bias)
        b2 = np.clip(b2, -max_bias, max_bias)
        
        if 0 < alpha1_new < self.C:
            b = b1
        elif 0 < alpha2_new < self.C:
            b = b2
        else:
            b = (b1 + b2) / 2
        
        # 最终检查
        if np.isnan(b) or np.isinf(b):
            b = 0.0
        b = np.clip(b, -max_bias, max_bias)
        
        return True, alpha, b
    
    def _examine_example(self, i2, HY, K, alpha, E, b):
        """检查并优化样本 i2"""
        y2 = HY[i2]
        alpha2 = alpha[i2]
        E2 = E[i2]
        r2 = E2 * y2
        
        # KKT条件检查
        if (r2 < -self.tol and alpha2 < self.C) or (r2 > self.tol and alpha2 > 0):
            # 寻找第二个变量
            indices = np.where((alpha > 0) & (alpha < self.C))[0]
            if len(indices) > 1:
                # 选择使 |E1 - E2| 最大的样本
                if E2 > 0:
                    i1 = indices[np.argmin(E[indices])]
                else:
                    i1 = indices[np.argmax(E[indices])]
                changed, alpha, b = self._take_step(i1, i2, alpha, HY, K, E, b)
                if changed:
                    return True, alpha, b
            
            # 随机选择第二个变量
            indices = list(range(len(HY)))
            np.random.shuffle(indices)
            for i1 in indices:
                if i1 == i2 or alpha[i1] == 0:
                    continue
                changed, alpha, b = self._take_step(i1, i2, alpha, HY, K, E, b)
                if changed:
                    return True, alpha, b
            
            # 随机选择任何样本作为第二个变量
            indices = list(range(len(HY)))
            np.random.shuffle(indices)
            for i1 in indices:
                if i1 == i2:
                    continue
                changed, alpha, b = self._take_step(i1, i2, alpha, HY, K, E, b)
                if changed:
                    return True, alpha, b
        
        return False, alpha, b
    
    def fit(self, X, y):
        """训练SVM模型（使用SMO算法），支持多分类（One-vs-Rest策略）"""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        
        # 数据标准化（SVM对尺度敏感）
        self.X_mean_ = X.mean(axis=0)
        self.X_std_ = X.std(axis=0)
        self.X_std_[self.X_std_ == 0] = 1  # 避免除零
        X_scaled = (X - self.X_mean_) / self.X_std_
        
        # 获取所有类别
        unique_classes = np.unique(y)
        self.classes_ = unique_classes
        self.n_classes_ = len(unique_classes)
        
        # 检查类别数量
        if self.n_classes_ < 2:
            raise ValueError(f"SVM需要至少2个类别，但数据中只有{self.n_classes_}个类别")
        
        # 判断是否为多分类
        if self.n_classes_ == 2:
            # 二分类：直接使用原来的方法
            self.multiclass_ = False
            HY = np.where(y == unique_classes[0], -1, 1)
            
            # 设置 gamma（在标准化后的数据上计算）
            if self.gamma == 'scale':
                # scale: 1 / (n_features * X.var())
                var_mean = X_scaled.var(axis=0).mean()
                if var_mean > 0:
                    self.gamma_value = 1.0 / (X_scaled.shape[1] * var_mean)
                else:
                    self.gamma_value = 1.0 / X_scaled.shape[1]
                # 确保gamma不会太小，避免数值问题
                if self.gamma_value < 1e-10:
                    self.gamma_value = 0.1
            elif self.gamma == 'auto':
                self.gamma_value = 1.0 / X_scaled.shape[1]
            else:
                self.gamma_value = float(self.gamma)
            
            # 初始化
            n_samples = X_scaled.shape[0]
            alpha = np.zeros(n_samples)
            b = 0.0
            
            # 计算核矩阵（使用标准化后的数据）
            K = self._kernel_matrix(X_scaled, X_scaled)
            
            # 初始化误差
            E = self._compute_E(alpha, HY, K, b)
            
            # SMO算法主循环
            num_changed = 0
            examine_all = True
            np.random.seed(self.random_state)
            
            for iteration in range(self.max_iter):
                num_changed = 0
                
                if examine_all:
                    # 检查所有样本
                    for i in range(n_samples):
                        changed, alpha, b = self._examine_example(i, HY, K, alpha, E, b)
                        if changed:
                            num_changed += 1
                else:
                    # 仅检查边界样本（0 < alpha < C）
                    indices = np.where((alpha > 0) & (alpha < self.C))[0]
                    for i in indices:
                        changed, alpha, b = self._examine_example(i, HY, K, alpha, E, b)
                        if changed:
                            num_changed += 1
                
                # 更新误差
                E = self._compute_E(alpha, HY, K, b)
                
                if examine_all:
                    examine_all = False
                elif num_changed == 0:
                    examine_all = True
                
                # 检查收敛
                if num_changed == 0 and not examine_all:
                    break
            
            # 保存支持向量（使用标准化后的数据）
            support_mask = alpha > 1e-5
            self.support_vectors_ = X_scaled[support_mask]
            self.support_vector_labels_ = HY[support_mask]
            self.dual_coef_ = (alpha * HY)[support_mask]
            self.intercept_ = b
            self.X_train = X_scaled
            self.y_train = y
        else:
            # 多分类：使用One-vs-Rest策略
            self.multiclass_ = True
            self.estimators_ = []
            
            # 设置 gamma（在标准化后的数据上计算，所有分类器共享）
            if self.gamma == 'scale':
                # scale: 1 / (n_features * X.var())
                var_mean = X_scaled.var(axis=0).mean()
                if var_mean > 0:
                    self.gamma_value = 1.0 / (X_scaled.shape[1] * var_mean)
                else:
                    self.gamma_value = 1.0 / X_scaled.shape[1]
                # 确保gamma在合理范围内，避免数值问题
                if self.gamma_value < 1e-6:
                    self.gamma_value = 1e-3
                elif self.gamma_value > 100:
                    self.gamma_value = 10.0
            elif self.gamma == 'auto':
                self.gamma_value = 1.0 / X_scaled.shape[1]
            else:
                self.gamma_value = float(self.gamma)
            
            # 为每个类别训练一个二分类器
            for i, class_label in enumerate(unique_classes):
                # 创建二分类标签：当前类别为+1，其他类别为-1
                y_binary = np.where(y == class_label, 1, -1)
                
                # 训练二分类SVM
                estimator = self._fit_binary(X_scaled, y_binary)
                self.estimators_.append(estimator)
            
            self.X_train = X_scaled
            self.y_train = y
            
        return self
    
    def _fit_binary(self, X_scaled, y_binary):
        """训练单个二分类SVM，返回模型参数"""
        HY = y_binary
        n_samples = X_scaled.shape[0]
        
        # 初始化
        alpha = np.zeros(n_samples)
        b = 0.0
        
        # 计算核矩阵
        K = self._kernel_matrix(X_scaled, X_scaled)
        
        # 初始化误差
        E = self._compute_E(alpha, HY, K, b)
        
        # SMO算法主循环
        num_changed = 0
        examine_all = True
        np.random.seed(self.random_state)
        
        for iteration in range(self.max_iter):
            num_changed = 0
            
            if examine_all:
                # 检查所有样本
                for i in range(n_samples):
                    changed, alpha, b = self._examine_example(i, HY, K, alpha, E, b)
                    if changed:
                        num_changed += 1
            else:
                # 仅检查边界样本（0 < alpha < C）
                indices = np.where((alpha > 0) & (alpha < self.C))[0]
                for i in indices:
                    changed, alpha, b = self._examine_example(i, HY, K, alpha, E, b)
                    if changed:
                        num_changed += 1
            
            # 更新误差
            E = self._compute_E(alpha, HY, K, b)
            
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True
            
            # 检查收敛
            if num_changed == 0 and not examine_all:
                break
        
        # 保存模型参数
        support_mask = alpha > 1e-5
        
        # 如果没有支持向量，使用所有样本作为支持向量（但权重很小）
        if np.sum(support_mask) == 0:
            # 这种情况很少见，但可以处理：使用所有样本，alpha值设为很小的值
            support_mask = np.ones(n_samples, dtype=bool)
            alpha = np.ones(n_samples) * 1e-6
        
        # 限制intercept范围，防止极端值
        b_clipped = np.clip(b, -1e4, 1e4)
        
        estimator = {
            'support_vectors': X_scaled[support_mask],
            'support_vector_labels': HY[support_mask],
            'dual_coef': (alpha * HY)[support_mask],
            'intercept': float(b_clipped)
        }
        
        return estimator
    
    def _decision_function(self, X):
        """计算决策函数值"""
        X = np.asarray(X, dtype=float)
        
        # 标准化输入数据（使用训练时的均值和标准差）
        X_scaled = (X - self.X_mean_) / self.X_std_
        
        if self.multiclass_:
            # 多分类：返回每个类别的决策函数值
            n_samples = X_scaled.shape[0]
            n_classes = self.n_classes_
            decisions = np.zeros((n_samples, n_classes))
            
            for i, estimator in enumerate(self.estimators_):
                sv = estimator['support_vectors']
                dual_coef = estimator['dual_coef']
                intercept = estimator['intercept']
                
                if len(sv) == 0:
                    # 如果没有支持向量，决策函数值为intercept
                    decisions[:, i] = intercept
                else:
                    for j in range(n_samples):
                        sum_val = 0.0
                        for k in range(len(sv)):
                            kernel_val = self._kernel_function(X_scaled[j], sv[k])
                            sum_val += dual_coef[k] * kernel_val
                        decisions[j, i] = sum_val + intercept
                        # 防止数值溢出，限制范围
                        if np.isnan(decisions[j, i]) or np.isinf(decisions[j, i]):
                            decisions[j, i] = intercept
                        # 限制决策函数值的范围，避免极端值影响预测
                        decisions[j, i] = np.clip(decisions[j, i], -1e4, 1e4)
            
            return decisions
        else:
            # 二分类
            if self.support_vectors_ is None or len(self.support_vectors_) == 0:
                return np.zeros(X.shape[0])
            
            n_samples = X_scaled.shape[0]
            decisions = np.zeros(n_samples)
            
            for i in range(n_samples):
                sum_val = 0.0
                for j in range(len(self.support_vectors_)):
                    sum_val += self.dual_coef_[j] * self._kernel_function(X_scaled[i], self.support_vectors_[j])
                decisions[i] = sum_val + self.intercept_
            
            return decisions
    
    def predict(self, X):
        """预测类别"""
        decisions = self._decision_function(X)
        
        if self.multiclass_:
            # 多分类：选择决策函数值最大的类别
            predictions = self.classes_[np.argmax(decisions, axis=1)]
        else:
            # 二分类
            if len(self.classes_) < 2:
                # 如果只有一个类别，直接返回该类别的预测
                return np.full(decisions.shape[0], self.classes_[0])
            predictions = np.where(decisions >= 0, self.classes_[1], self.classes_[0])
        
        return predictions
    
    def predict_proba(self, X):
        """预测概率（使用Platt scaling的简化版本）"""
        decisions = self._decision_function(X)
        
        if self.multiclass_:
            # 多分类：对每个类别的决策函数值应用sigmoid，然后归一化
            n_samples = decisions.shape[0]
            proba = np.zeros((n_samples, self.n_classes_))
            
            for i in range(self.n_classes_):
                # 对每个类别的决策函数值应用sigmoid
                prob = 1.0 / (1.0 + np.exp(-decisions[:, i]))
                prob = np.clip(prob, 1e-15, 1 - 1e-15)
                proba[:, i] = prob
            
            # 归一化，使每行的概率和为1
            proba = proba / proba.sum(axis=1, keepdims=True)
            
            return proba
        else:
            # 二分类
            # 简单的sigmoid映射
            prob = 1.0 / (1.0 + np.exp(-decisions))
            prob = np.clip(prob, 1e-15, 1 - 1e-15)
            # 返回两类的概率
            proba = np.column_stack([1 - prob, prob])
            return proba
    
    def get_params(self, deep=True):
        """获取参数（sklearn兼容）"""
        return {
            'C': self.C,
            'kernel': self.kernel,
            'gamma': self.gamma,
            'tol': self.tol,
            'max_iter': self.max_iter,
            'random_state': self.random_state
        }
    
    def set_params(self, **params):
        """设置参数（sklearn兼容）"""
        for key, value in params.items():
            setattr(self, key, value)
        return self

class MyRandomForest:
    """原创实现：随机森林算法（基于自研决策树）"""
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, max_features='sqrt', random_state=42, bootstrap=True):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.bootstrap = bootstrap
        self.estimators_ = []
        self.feature_indices_ = []
        self.classes_ = None
        self.n_features_ = None
        
    def _bootstrap_sample(self, X, y, rng):
        """Bootstrap采样"""
        n_samples = X.shape[0]
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]
    
    def _get_feature_subset(self, n_features, rng):
        """获取特征子集"""
        if self.max_features == 'sqrt':
            n_selected = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            n_selected = int(np.log2(n_features)) + 1
        elif isinstance(self.max_features, float):
            n_selected = int(self.max_features * n_features)
        elif isinstance(self.max_features, int):
            n_selected = min(self.max_features, n_features)
        else:
            n_selected = n_features
        
        n_selected = max(1, n_selected)  # 至少选择1个特征
        selected_features = rng.choice(n_features, size=n_selected, replace=False)
        return np.sort(selected_features)
    
    def fit(self, X, y):
        """训练随机森林"""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        
        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.estimators_ = []
        self.feature_indices_ = []
        
        rng = np.random.default_rng(self.random_state)
        
        for i in range(self.n_estimators):
            # Bootstrap采样
            if self.bootstrap:
                X_bootstrap, y_bootstrap = self._bootstrap_sample(X, y, rng)
            else:
                X_bootstrap, y_bootstrap = X, y
            
            # 特征子集采样
            feature_indices = self._get_feature_subset(self.n_features_, rng)
            X_bootstrap_subset = X_bootstrap[:, feature_indices]
            
            # 创建并训练决策树
            tree = MyDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X_bootstrap_subset, y_bootstrap)
            
            # 保存树和特征索引
            self.estimators_.append(tree)
            self.feature_indices_.append(feature_indices)
        
        return self
    
    def predict(self, X):
        """预测类别（多数投票）"""
        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]
        all_predictions = []
        
        # 每棵树进行预测
        for i, tree in enumerate(self.estimators_):
            feature_indices = self.feature_indices_[i]
            X_subset = X[:, feature_indices]
            predictions = tree.predict(X_subset)
            all_predictions.append(predictions)
        
        # 转换为numpy数组便于处理
        all_predictions = np.array(all_predictions)  # shape: (n_estimators, n_samples)
        
        # 多数投票
        final_predictions = []
        for j in range(n_samples):
            votes = all_predictions[:, j]
            # 统计每个类别的票数
            unique, counts = np.unique(votes, return_counts=True)
            # 选择得票最多的类别
            winner = unique[np.argmax(counts)]
            final_predictions.append(winner)
        
        return np.array(final_predictions)
    
    def predict_proba(self, X):
        """预测概率（基于投票比例）"""
        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        proba = np.zeros((n_samples, n_classes))
        
        # 每棵树进行预测
        for i, tree in enumerate(self.estimators_):
            feature_indices = self.feature_indices_[i]
            X_subset = X[:, feature_indices]
            predictions = tree.predict(X_subset)
            
            # 统计每个类别的票数
            for j in range(n_samples):
                class_idx = np.where(self.classes_ == predictions[j])[0][0]
                proba[j, class_idx] += 1
        
        # 归一化为概率
        proba = proba / self.n_estimators
        
        return proba
    
    def get_params(self, deep=True):
        """获取参数（sklearn兼容）"""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'max_features': self.max_features,
            'random_state': self.random_state,
            'bootstrap': self.bootstrap
        }
    
    def set_params(self, **params):
        """设置参数（sklearn兼容）"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

class MyGBDT:
    """原创实现：梯度提升决策树算法"""
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.estimators_ = []
        self.initial_prediction = None
        self.classes_ = None
        self.n_classes_ = None
        
    def _negative_gradient(self, y, y_pred, task='classification'):
        """计算负梯度（残差）"""
        if task == 'classification':
            # 对于分类任务，使用对数几率
            # y_pred 是概率，转换为 log-odds
            y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # 防止log(0)
            return y - y_pred  # 残差 = 真实值 - 预测概率
        else:
            # 回归任务：残差 = y - y_pred
            return y - y_pred
    
    def _sigmoid(self, x):
        """Sigmoid函数"""
        x = np.clip(x, -250, 250)  # 防止数值溢出
        return 1 / (1 + np.exp(-x))
    
    def fit(self, X, y):
        """训练GBDT模型"""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        is_multiclass = len(self.classes_) > 2
        
        # 检查类别数量
        if self.n_classes_ < 2:
            raise ValueError(f"GBDT需要至少2个类别，但数据中只有{self.n_classes_}个类别")
        
        # 初始化预测值
        if self.n_classes_ == 2:
            # 二分类：初始化为概率的对数几率
            pos_count = np.sum(y == self.classes_[1])
            pos_prob = pos_count / len(y)
            self.initial_prediction = np.log(pos_prob / (1 - pos_prob + 1e-15))
            y_pred = np.full(len(y), pos_prob)
        else:
            # 多分类：为每个类别初始化
            self.initial_prediction = {}
            y_pred = np.zeros((len(y), self.n_classes_))
            for i, c in enumerate(self.classes_):
                prob = np.sum(y == c) / len(y)
                self.initial_prediction[c] = np.log(prob / (1 - prob + 1e-15))
                y_pred[:, i] = prob
        
        rng = np.random.default_rng(self.random_state)
        self.estimators_ = []
        
        # 梯度提升循环
        for m in range(self.n_estimators):
            # 计算负梯度（残差）
            if self.n_classes_ == 2:
                # 二分类：将y转换为0/1标签
                y_binary = np.where(y == self.classes_[0], 0, 1)
                residual = y_binary - y_pred
                
                # 训练决策树来拟合残差（使用残差的符号作为伪标签）
                tree = MyDecisionTree(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split
                )
                residual_labels = np.where(residual > 0, 1, 0)
                tree.fit(X, residual_labels)
                
                # 获取树的预测并进行更新
                tree_pred = tree.predict(X)
                tree_proba = np.where(tree_pred == 1, 
                                     self.learning_rate, -self.learning_rate)
                y_pred = y_pred + tree_proba
                y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
                
                # 保存这棵树的引用
                self.estimators_.append(tree)
                
            else:
                # 多分类：为每个类别训练一棵树
                trees_for_iteration = []
                for i, c in enumerate(self.classes_):
                    # 创建二分类标签
                    y_binary = (y == c).astype(int)
                    y_pred_binary = y_pred[:, i]
                    
                    # 计算残差
                    residual = self._negative_gradient(y_binary, y_pred_binary, 'classification')
                    residual_labels = np.where(residual > 0, 1, 0)
                    
                    # 训练决策树
                    tree = MyDecisionTree(
                        max_depth=self.max_depth,
                        min_samples_split=self.min_samples_split
                    )
                    tree.fit(X, residual_labels)
                    trees_for_iteration.append(tree)
                    
                    # 更新该类的预测
                    tree_pred = tree.predict(X)
                    tree_proba = np.where(tree_pred == 1,
                                         self.learning_rate, -self.learning_rate)
                    y_pred[:, i] = y_pred[:, i] + tree_proba
                    y_pred[:, i] = np.clip(y_pred[:, i], 1e-15, 1 - 1e-15)
                
                # 归一化概率
                y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)
                
                # 保存这一轮的所有树
                self.estimators_.append(trees_for_iteration)
        
        return self
    
    def predict(self, X):
        """预测类别"""
        X = np.asarray(X, dtype=float)
        
        if self.n_classes_ == 2:
            # 二分类：累加所有树的预测
            if len(self.classes_) < 2:
                # 如果只有一个类别，直接返回该类别的预测
                return np.full(X.shape[0], self.classes_[0])
            y_pred = np.full(X.shape[0], 0.5)  # 初始化为0.5
            
            for tree in self.estimators_:
                tree_pred = tree.predict(X)
                tree_proba = np.where(tree_pred == self.classes_[1],
                                     self.learning_rate, -self.learning_rate)
                y_pred = y_pred + tree_proba
                y_pred = np.clip(y_pred, 0, 1)
            
            # 转换为类别
            predictions = np.where(y_pred >= 0.5, self.classes_[1], self.classes_[0])
            return predictions
        else:
            # 多分类：为每个类别计算概率
            n_samples = X.shape[0]
            y_pred = np.ones((n_samples, self.n_classes_)) / self.n_classes_  # 初始化均匀分布
            
            # 累加所有轮次的所有树的预测
            for trees_list in self.estimators_:
                for i, tree in enumerate(trees_list):
                    tree_pred = tree.predict(X)
                    tree_proba = np.where(tree_pred == 1,
                                         self.learning_rate, -self.learning_rate)
                    y_pred[:, i] = y_pred[:, i] + tree_proba
            
            # 归一化并选择最大概率的类别
            y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
            y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)
            predictions = self.classes_[np.argmax(y_pred, axis=1)]
            return predictions
    
    def predict_proba(self, X):
        """预测概率"""
        X = np.asarray(X, dtype=float)
        
        if self.n_classes_ == 2:
            # 二分类
            if len(self.classes_) < 2:
                # 如果只有一个类别，返回单类概率
                return np.column_stack([np.ones(X.shape[0]), np.zeros(X.shape[0])])
            y_pred = np.full(X.shape[0], 0.5)
            
            for tree in self.estimators_:
                tree_pred = tree.predict(X)
                tree_proba = np.where(tree_pred == self.classes_[1],
                                     self.learning_rate, -self.learning_rate)
                y_pred = y_pred + tree_proba
                y_pred = np.clip(y_pred, 0, 1)
            
            # 返回两类的概率
            proba = np.column_stack([1 - y_pred, y_pred])
            return proba
        else:
            # 多分类
            n_samples = X.shape[0]
            y_pred = np.ones((n_samples, self.n_classes_)) / self.n_classes_
            
            # 累加所有轮次的所有树的预测
            for trees_list in self.estimators_:
                for i, tree in enumerate(trees_list):
                    tree_pred = tree.predict(X)
                    tree_proba = np.where(tree_pred == 1,
                                         self.learning_rate, -self.learning_rate)
                    y_pred[:, i] = y_pred[:, i] + tree_proba
            
            # 归一化
            y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
            y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)
            return y_pred
    
    def get_params(self, deep=True):
        """获取参数（sklearn兼容）"""
        return {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'random_state': self.random_state
        }
    
    def set_params(self, **params):
        """设置参数（sklearn兼容）"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

class MyNaiveBayes:
    """原创实现：高斯朴素贝叶斯算法（改进版：支持拉普拉斯平滑、方差稳定性）"""
    def __init__(self, alpha=1.0, var_smoothing=1e-9):
        """
        参数:
        alpha: 拉普拉斯平滑参数（特征平滑）
        var_smoothing: 方差平滑参数，防止方差为0
        """
        self.alpha = alpha
        self.var_smoothing = var_smoothing
        self.classes_ = None
        self.class_count_ = None
        self.class_prior_ = None
        self.theta_ = None  # 每个类别的特征均值
        self.sigma_ = None  # 每个类别的特征方差
        
    def fit(self, X, y):
        """训练朴素贝叶斯模型"""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        
        # 获取类别
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        
        # 计算每个类别的样本数
        self.class_count_ = np.zeros(n_classes, dtype=float)
        for i, c in enumerate(self.classes_):
            self.class_count_[i] = np.sum(y == c)
        
        # ========== 算法改进：拉普拉斯平滑 ==========
        # 计算类别先验概率（带平滑）
        n_samples = len(y)
        self.class_prior_ = (self.class_count_ + self.alpha) / (n_samples + self.alpha * n_classes)
        
        # 计算每个类别的特征均值和方差
        self.theta_ = np.zeros((n_classes, n_features))
        self.sigma_ = np.zeros((n_classes, n_features))
        
        for i, c in enumerate(self.classes_):
            X_c = X[y == c]  # 该类别的所有样本
            
            # 计算均值
            self.theta_[i] = np.mean(X_c, axis=0)
            
            # ========== 算法改进：方差稳定性 ==========
            # 计算方差并添加平滑项，防止方差为0导致的数值不稳定
            var_c = np.var(X_c, axis=0, ddof=0)
            self.sigma_[i] = var_c + self.var_smoothing
        
        return self
    
    def _calculate_log_likelihood(self, X):
        """计算对数似然（使用对数避免数值下溢）"""
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        log_likelihood = np.zeros((n_samples, n_classes))
        
        for i in range(n_classes):
            # 计算对数先验
            log_prior = np.log(self.class_prior_[i] + 1e-15)
            
            # 计算每个特征的对数似然（高斯分布）
            # log(P(x|y)) = -0.5 * log(2πσ²) - 0.5 * (x-μ)²/σ²
            diff = X - self.theta_[i]
            log_2pi = np.log(2 * np.pi)
            
            log_likelihood[:, i] = (
                -0.5 * np.sum(np.log(self.sigma_[i]) + log_2pi, axis=0) -
                0.5 * np.sum((diff ** 2) / self.sigma_[i], axis=1) +
                log_prior
            )
        
        return log_likelihood
    
    def predict(self, X):
        """预测类别"""
        X = np.asarray(X, dtype=float)
        log_likelihood = self._calculate_log_likelihood(X)
        y_pred = self.classes_[np.argmax(log_likelihood, axis=1)]
        return y_pred
    
    def predict_proba(self, X):
        """预测类别概率"""
        X = np.asarray(X, dtype=float)
        log_likelihood = self._calculate_log_likelihood(X)
        
        # 使用log-sum-exp技巧避免数值下溢
        # P(y|x) = exp(log_likelihood) / sum(exp(log_likelihood))
        log_likelihood_max = np.max(log_likelihood, axis=1, keepdims=True)
        exp_log_likelihood = np.exp(log_likelihood - log_likelihood_max)
        proba = exp_log_likelihood / np.sum(exp_log_likelihood, axis=1, keepdims=True)
        
        return proba
    
    def get_params(self, deep=True):
        """获取参数（sklearn兼容）"""
        return {
            'alpha': self.alpha,
            'var_smoothing': self.var_smoothing
        }
    
    def set_params(self, **params):
        """设置参数（sklearn兼容）"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
    
    def get_training_history(self):
        """获取训练历史数据（朴素贝叶斯不需要迭代训练，返回None）"""
        return None

class MyMLP:
    """原创实现：多层感知机神经网络（改进版：支持早停机制、学习率衰减、He初始化）"""
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', learning_rate_init=0.001, 
                 max_iter=1000, random_state=42, solver='adam', batch_size='auto', 
                 tol=1e-4, momentum=0.9, beta_1=0.9, beta_2=0.999, epsilon=1e-8,
                 early_stopping=False, validation_fraction=0.1, n_iter_no_change=10,
                 learning_rate='constant', learning_rate_decay=0.95):
        self.hidden_layer_sizes = hidden_layer_sizes if isinstance(hidden_layer_sizes, tuple) else tuple(hidden_layer_sizes)
        self.activation = activation  # 'relu', 'sigmoid', 'tanh'
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.solver = solver  # 'sgd' or 'adam'
        self.batch_size = batch_size
        self.tol = tol
        self.momentum = momentum
        self.beta_1 = beta_1  # Adam参数
        self.beta_2 = beta_2
        self.epsilon = epsilon
        
        # ========== 算法改进：早停机制 ==========
        self.early_stopping = early_stopping  # 是否启用早停
        self.validation_fraction = validation_fraction  # 验证集比例
        self.n_iter_no_change = n_iter_no_change  # 连续n次无改进则停止
        self.best_loss_ = float('inf')  # 最佳验证损失
        self.best_iter_ = 0  # 最佳迭代次数
        self.best_weights_ = None  # 最佳权重
        self.best_biases_ = None  # 最佳偏置
        self.validation_loss_curve_ = []  # 验证损失曲线
        
        # ========== 算法改进：学习率衰减 ==========
        self.learning_rate = learning_rate  # 'constant', 'invscaling', 'adaptive'
        self.learning_rate_decay = learning_rate_decay  # 学习率衰减率
        self.current_learning_rate = learning_rate_init  # 当前学习率
        
        # 网络参数
        self.weights_ = []
        self.biases_ = []
        self.n_layers_ = None
        self.loss_curve_ = []
        self.classes_ = None
        self.n_outputs_ = None
        
    def _activation_function(self, x, activation_type):
        """激活函数"""
        if activation_type == 'relu':
            return np.maximum(0, x)
        elif activation_type == 'sigmoid':
            x = np.clip(x, -250, 250)
            return 1 / (1 + np.exp(-x))
        elif activation_type == 'tanh':
            x = np.clip(x, -250, 250)
            return np.tanh(x)
        else:
            return x
    
    def _activation_derivative(self, x, activation_type):
        """激活函数的导数"""
        if activation_type == 'relu':
            return (x > 0).astype(float)
        elif activation_type == 'sigmoid':
            s = self._activation_function(x, 'sigmoid')
            return s * (1 - s)
        elif activation_type == 'tanh':
            t = self._activation_function(x, 'tanh')
            return 1 - t ** 2
        else:
            return np.ones_like(x)
    
    def _softmax(self, x):
        """Softmax函数"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def _initialize_weights(self, n_features, n_outputs):
        """初始化权重和偏置（改进版：使用He初始化，更适合ReLU激活函数）"""
        np.random.seed(self.random_state)
        layer_sizes = [n_features] + list(self.hidden_layer_sizes) + [n_outputs]
        self.n_layers_ = len(layer_sizes) - 1
        
        self.weights_ = []
        self.biases_ = []
        
        for i in range(self.n_layers_):
            # ========== 算法改进：He初始化（ReLU）或Xavier初始化（Sigmoid/Tanh） ==========
            if self.activation == 'relu' and i < self.n_layers_ - 1:
                # He初始化：适合ReLU激活函数，方差为2/n_in
                std = np.sqrt(2.0 / layer_sizes[i])
                w = np.random.normal(0, std, (layer_sizes[i], layer_sizes[i + 1]))
            elif self.activation in ['sigmoid', 'tanh'] and i < self.n_layers_ - 1:
                # Xavier初始化：适合Sigmoid/Tanh激活函数
                limit = np.sqrt(6.0 / (layer_sizes[i] + layer_sizes[i + 1]))
                w = np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i + 1]))
            else:
                # 输出层或默认：使用Xavier初始化
                limit = np.sqrt(6.0 / (layer_sizes[i] + layer_sizes[i + 1]))
                w = np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i + 1]))
            
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights_.append(w)
            self.biases_.append(b)
    
    def _forward_pass(self, X):
        """前向传播"""
        activations = [X]
        z_values = []
        
        # 隐藏层
        for i in range(self.n_layers_ - 1):
            z = np.dot(activations[-1], self.weights_[i]) + self.biases_[i]
            z_values.append(z)
            a = self._activation_function(z, self.activation)
            activations.append(a)
        
        # 输出层（softmax用于分类）
        z = np.dot(activations[-1], self.weights_[-1]) + self.biases_[-1]
        z_values.append(z)
        a = self._softmax(z)
        activations.append(a)
        
        return activations, z_values
    
    def _backward_pass(self, activations, z_values, y_onehot):
        """反向传播"""
        m = activations[0].shape[0]
        grads_w = [np.zeros_like(w) for w in self.weights_]
        grads_b = [np.zeros_like(b) for b in self.biases_]
        
        # 输出层误差
        delta = activations[-1] - y_onehot
        grads_w[-1] = (1 / m) * np.dot(activations[-2].T, delta)
        grads_b[-1] = (1 / m) * np.sum(delta, axis=0, keepdims=True)
        
        # 反向传播到隐藏层
        for i in range(self.n_layers_ - 2, -1, -1):
            delta = np.dot(delta, self.weights_[i + 1].T) * self._activation_derivative(z_values[i], self.activation)
            grads_w[i] = (1 / m) * np.dot(activations[i].T, delta)
            grads_b[i] = (1 / m) * np.sum(delta, axis=0, keepdims=True)
        
        return grads_w, grads_b
    
    def _to_onehot(self, y):
        """将标签转换为one-hot编码"""
        n_samples = len(y)
        n_classes = len(self.classes_)
        y_onehot = np.zeros((n_samples, n_classes))
        for i, class_val in enumerate(self.classes_):
            y_onehot[y == class_val, i] = 1
        return y_onehot
    
    def fit(self, X, y):
        """训练MLP模型（改进版：支持早停、学习率衰减）"""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        
        # 确定类别
        self.classes_ = np.unique(y)
        self.n_outputs_ = len(self.classes_)
        n_features = X.shape[1]
        
        # ========== 算法改进：早停机制 - 分割验证集 ==========
        X_train, X_val, y_train, y_val = X, None, y, None
        if self.early_stopping and X.shape[0] > 50:  # 样本数足够时才分割验证集
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.validation_fraction, random_state=self.random_state, stratify=y
            )
            y_val_onehot = self._to_onehot(y_val)
        
        # 初始化权重
        self._initialize_weights(n_features, self.n_outputs_)
        self.current_learning_rate = self.learning_rate_init  # 重置学习率
        
        # 转换为one-hot编码
        y_train_onehot = self._to_onehot(y_train)
        
        # 设置batch size
        if self.batch_size == 'auto':
            batch_size = min(200, max(10, X_train.shape[0] // 5))
        else:
            batch_size = self.batch_size
        
        # 优化器状态（用于Adam）
        if self.solver == 'adam':
            m_w = [np.zeros_like(w) for w in self.weights_]
            v_w = [np.zeros_like(w) for w in self.weights_]
            m_b = [np.zeros_like(b) for b in self.biases_]
            v_b = [np.zeros_like(b) for b in self.biases_]
        
        # SGD的动量状态
        if self.solver == 'sgd' and self.momentum > 0:
            velocity_w = [np.zeros_like(w) for w in self.weights_]
            velocity_b = [np.zeros_like(b) for b in self.biases_]
        
        # ========== 算法改进：早停机制初始化 ==========
        if self.early_stopping and X_val is not None:
            self.best_loss_ = float('inf')
            self.best_iter_ = 0
            self.best_weights_ = [w.copy() for w in self.weights_]
            self.best_biases_ = [b.copy() for b in self.biases_]
            self.validation_loss_curve_ = []
            no_improvement_count = 0
        
        # 训练循环
        np.random.seed(self.random_state)
        prev_loss = float('inf')
        
        for iteration in range(self.max_iter):
            # ========== 算法改进：学习率衰减 ==========
            if self.learning_rate == 'invscaling':
                # 反比例缩放：lr = lr0 / (1 + decay * iteration)
                self.current_learning_rate = self.learning_rate_init / (1.0 + self.learning_rate_decay * iteration)
            elif self.learning_rate == 'adaptive':
                # 自适应：如果损失没有改善，则衰减
                if iteration > 0 and len(self.loss_curve_) > 0:
                    if self.loss_curve_[-1] >= prev_loss:
                        self.current_learning_rate *= self.learning_rate_decay
            # 'constant' 模式保持 learning_rate_init 不变
            
            # 随机打乱数据
            indices = np.random.permutation(X_train.shape[0])
            X_shuffled = X_train[indices]
            y_shuffled = y_train_onehot[indices]
            
            epoch_loss = 0
            
            # Mini-batch训练
            for start in range(0, X_train.shape[0], batch_size):
                end = min(start + batch_size, X_train.shape[0])
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                
                # 前向传播
                activations, z_values = self._forward_pass(X_batch)
                
                # 计算损失（交叉熵）
                batch_loss = -np.mean(np.sum(y_batch * np.log(activations[-1] + 1e-15), axis=1))
                epoch_loss += batch_loss
                
                # 反向传播
                grads_w, grads_b = self._backward_pass(activations, z_values, y_batch)
                
                # 更新权重（使用优化器，应用当前学习率）
                current_lr = self.current_learning_rate
                
                if self.solver == 'adam':
                    t = iteration + 1
                    lr_t = current_lr * np.sqrt(1 - self.beta_2 ** t) / (1 - self.beta_1 ** t)
                    
                    for i in range(self.n_layers_):
                        # 更新权重
                        m_w[i] = self.beta_1 * m_w[i] + (1 - self.beta_1) * grads_w[i]
                        v_w[i] = self.beta_2 * v_w[i] + (1 - self.beta_2) * (grads_w[i] ** 2)
                        self.weights_[i] -= lr_t * m_w[i] / (np.sqrt(v_w[i]) + self.epsilon)
                        
                        # 更新偏置
                        m_b[i] = self.beta_1 * m_b[i] + (1 - self.beta_1) * grads_b[i]
                        v_b[i] = self.beta_2 * v_b[i] + (1 - self.beta_2) * (grads_b[i] ** 2)
                        self.biases_[i] -= lr_t * m_b[i] / (np.sqrt(v_b[i]) + self.epsilon)
                
                elif self.solver == 'sgd':
                    if self.momentum > 0:
                        # SGD with momentum
                        for i in range(self.n_layers_):
                            velocity_w[i] = self.momentum * velocity_w[i] - current_lr * grads_w[i]
                            velocity_b[i] = self.momentum * velocity_b[i] - current_lr * grads_b[i]
                            self.weights_[i] += velocity_w[i]
                            self.biases_[i] += velocity_b[i]
                    else:
                        # 普通SGD
                        for i in range(self.n_layers_):
                            self.weights_[i] -= current_lr * grads_w[i]
                            self.biases_[i] -= current_lr * grads_b[i]
            
            epoch_loss /= (X_train.shape[0] // batch_size + 1)
            self.loss_curve_.append(epoch_loss)
            
            # ========== 算法改进：早停机制 - 验证集评估 ==========
            if self.early_stopping and X_val is not None:
                # 计算验证集损失
                val_activations, _ = self._forward_pass(X_val)
                val_loss = -np.mean(np.sum(y_val_onehot * np.log(val_activations[-1] + 1e-15), axis=1))
                self.validation_loss_curve_.append(val_loss)
                
                # 如果验证损失改善，保存最佳模型
                if val_loss < self.best_loss_ - self.tol:
                    self.best_loss_ = val_loss
                    self.best_iter_ = iteration
                    self.best_weights_ = [w.copy() for w in self.weights_]
                    self.best_biases_ = [b.copy() for b in self.biases_]
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                
                # 如果连续n次无改善，触发早停
                if no_improvement_count >= self.n_iter_no_change:
                    # 恢复最佳模型
                    self.weights_ = self.best_weights_
                    self.biases_ = self.best_biases_
                    break
            
            # 检查收敛
            if abs(prev_loss - epoch_loss) < self.tol:
                break
            prev_loss = epoch_loss
        
        return self
    
    def predict(self, X):
        """预测类别"""
        X = np.asarray(X, dtype=float)
        activations, _ = self._forward_pass(X)
        y_pred = self.classes_[np.argmax(activations[-1], axis=1)]
        return y_pred
    
    def predict_proba(self, X):
        """预测概率"""
        X = np.asarray(X, dtype=float)
        activations, _ = self._forward_pass(X)
        return activations[-1]
    
    def get_params(self, deep=True):
        """获取参数（sklearn兼容）"""
        return {
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'activation': self.activation,
            'learning_rate_init': self.learning_rate_init,
            'max_iter': self.max_iter,
            'random_state': self.random_state,
            'solver': self.solver,
            'batch_size': self.batch_size,
            'tol': self.tol
        }
    
    def set_params(self, **params):
        """设置参数（sklearn兼容）"""
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def get_training_history(self):
        """获取训练历史数据"""
        if hasattr(self, 'loss_curve_') and len(self.loss_curve_) > 0:
            return {
                'loss_history': [float(x) for x in self.loss_curve_],
                'iterations': len(self.loss_curve_),
                'algorithm': 'naive_bayes'  # MLP已替换为朴素贝叶斯，此代码保留用于兼容
            }
        return None

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
        """获取模型信息，支持多种名称格式匹配"""
        # 直接匹配
        if name in self._models:
            return self._models[name]
        
        # 名称映射表（前端显示名称 -> 注册名称）
        name_mapping = {
            'K 均值聚类(K-Means)': 'kmeans',
            'K均值聚类': 'kmeans',
            'K-Means': 'kmeans',
            'K 均值聚类': 'kmeans',
            '线性回归': 'linear_regression',
            '逻辑回归': 'logistic_regression',
            '决策树': 'decision_tree',
            'K近邻(KNN)': 'knn',
            'KNN': 'knn',
            '支持向量机(SVM)': 'svm',
            'SVM': 'svm',
            '随机森林': 'random_forest',
            '梯度提升树(GBDT)': 'gbdt',
            'GBDT': 'gbdt',
            '主成分分析(PCA)': 'pca',
            'PCA': 'pca',
            '朴素贝叶斯': 'naive_bayes',
            '朴素贝叶斯分类器': 'naive_bayes',
            'NaiveBayes': 'naive_bayes',
            'naive_bayes': 'naive_bayes'
        }
        
        # 尝试映射匹配
        mapped_name = name_mapping.get(name, name.lower())
        if mapped_name in self._models:
            return self._models[mapped_name]
        
        # 尝试不区分大小写的匹配
        name_lower = name.lower()
        for key in self._models:
            if key.lower() == name_lower:
                return self._models[key]
        
        # 尝试部分匹配（包含关系）
        for key in self._models:
            if name_lower in key.lower() or key.lower() in name_lower:
                return self._models[key]
        
        return {}

# 创建全局模型注册表
model_registry = ModelRegistry()

# 模型注册装饰器
def register_model(name, category='classification', description=''):
    def decorator(model_class):
        model_registry.register(name, model_class, category, description)
        return model_class
    return decorator

# 使用装饰器注册模型

class MyLinearRegression:
    """自研线性回归（改进版：支持学习率衰减），支持闭式解与梯度下降"""
    def __init__(self, method='normal', learning_rate=0.01, max_iter=1000, tol=1e-6,
                 learning_rate_decay=0.95, learning_rate_schedule='constant'):
        self.method = method  # 'normal' 或 'gd'
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        # ========== 算法改进：学习率衰减 ==========
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_schedule = learning_rate_schedule  # 'constant', 'decay', 'adaptive'
        self.current_learning_rate = learning_rate
        
        self.coef_ = None
        self.intercept_ = 0.0
        self.loss_history_ = []

    def _add_bias(self, X):
        return np.c_[np.ones((X.shape[0], 1)), X]

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        # 确保X是2维数组
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim == 0 or (X.ndim == 2 and X.shape[1] == 0):
            raise ValueError("线性回归需要至少1个特征，但当前特征数为0")
        
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        Xb = self._add_bias(X)
        if self.method == 'normal':
            # 闭式解：theta = (X^T X)^(-1) X^T y
            XT_X = Xb.T @ Xb
            # 加小的对角线以提升数值稳定性
            XT_X += 1e-8 * np.eye(XT_X.shape[0])
            theta = np.linalg.solve(XT_X, Xb.T @ y)
            self.intercept_ = float(theta[0, 0])
            self.coef_ = theta[1:, 0] if theta.shape[0] > 1 else np.array([])
        else:
            # 梯度下降（改进版：支持学习率衰减）
            rng = np.random.default_rng(42)
            theta = rng.normal(size=(Xb.shape[1], 1)) * 0.01
            last_loss = None
            self.gradient_norm_history_ = []
            self.current_learning_rate = self.learning_rate
            
            for iteration in range(self.max_iter):
                # ========== 算法改进：学习率衰减 ==========
                if self.learning_rate_schedule == 'decay':
                    # 指数衰减
                    self.current_learning_rate = self.learning_rate * (self.learning_rate_decay ** iteration)
                elif self.learning_rate_schedule == 'adaptive':
                    # 自适应：如果损失没有改善，则衰减
                    if last_loss is not None and len(self.loss_history_) > 0:
                        if self.loss_history_[-1] >= last_loss:
                            self.current_learning_rate *= self.learning_rate_decay
                
                preds = Xb @ theta
                errors = preds - y
                loss = float(np.mean(errors ** 2))
                self.loss_history_.append(loss)
                grad = (2 / Xb.shape[0]) * (Xb.T @ errors)
                
                # 记录梯度范数
                gradient_norm = np.sqrt(np.sum(grad**2))
                self.gradient_norm_history_.append(float(gradient_norm))
                
                # 使用当前学习率更新参数
                theta -= self.current_learning_rate * grad
                if last_loss is not None and abs(last_loss - loss) <= self.tol:
                    break
                last_loss = loss
            self.intercept_ = float(theta[0, 0])
            self.coef_ = theta[1:, 0] if theta.shape[0] > 1 else np.array([])
        return self
    
    def get_training_history(self):
        """获取训练历史数据"""
        if self.method == 'gd' and hasattr(self, 'loss_history_') and len(self.loss_history_) > 0:
            return {
                'loss_history': [float(x) for x in self.loss_history_],
                'gradient_norm_history': [float(x) for x in self.gradient_norm_history_] if hasattr(self, 'gradient_norm_history_') else [],
                'iterations': len(self.loss_history_),
                'algorithm': 'linear_regression_gd'
            }
        return None
    
    def get_params(self, deep=True):
        """获取参数（sklearn兼容）"""
        return {
            'method': self.method,
            'learning_rate': self.learning_rate,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'learning_rate_decay': self.learning_rate_decay,
            'learning_rate_schedule': self.learning_rate_schedule
        }
    
    def set_params(self, **params):
        """设置参数（sklearn兼容）"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        # 确保X是2维数组
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # 检查coef_是否已初始化
        if self.coef_ is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 检查特征数量是否匹配
        if X.shape[1] != len(self.coef_):
            raise ValueError(f"特征数量不匹配：输入{X.shape[1]}个特征，但模型期望{len(self.coef_)}个特征")
        
        return (X @ self.coef_ + self.intercept_).reshape(-1)

@register_model("linear_regression", "regression", "线性回归模型")
class LinearRegressionModel(MyLinearRegression):
    def __init__(self, method='gd', learning_rate=0.01, max_iter=1000, tol=1e-6,
                 learning_rate_decay=0.95, learning_rate_schedule='constant'):
        # 默认使用梯度下降方法，以便记录训练过程
        # ========== 算法改进：支持学习率衰减 ==========
        super().__init__(method=method, learning_rate=learning_rate, max_iter=max_iter, tol=tol,
                        learning_rate_decay=learning_rate_decay, learning_rate_schedule=learning_rate_schedule)

@register_model("logistic_regression", "classification", "逻辑回归分类器")
class LogisticRegressionModel(MyLogisticRegression):
    def __init__(self, C=1.0, max_iter=1000, random_state=42, alpha=0.0,
                 learning_rate_decay=0.95, learning_rate_schedule='constant', learning_rate=0.05, tol=1e-4, **kwargs):
        # 自研逻辑回归使用学习率/迭代等参数；这里保持签名兼容
        # ========== 算法改进：支持L2正则化和学习率衰减 ==========
        # C参数转换为alpha：C = 1/alpha（当C>0时），alpha是正则化系数
        # 如果同时提供了C和alpha，优先使用alpha；否则从C计算alpha
        self.C = C  # 保存C参数用于get_params()
        if alpha == 0.0 and C != 1.0:
            # 从C计算alpha：alpha = 1/C（C越大，正则化越小）
            alpha = 1.0 / C if C > 0 else 0.0
        self._alpha_from_C = alpha  # 保存从C计算的alpha
        super().__init__(learning_rate=learning_rate, max_iter=max_iter, tol=tol,
                        alpha=alpha, learning_rate_decay=learning_rate_decay,
                        learning_rate_schedule=learning_rate_schedule)
    
    def get_params(self, deep=True):
        """获取参数（sklearn兼容，支持C参数）"""
        params = super().get_params(deep=deep)
        # 添加C参数用于超参数调优
        params['C'] = self.C
        return params
    
    def set_params(self, **params):
        """设置参数（sklearn兼容，支持C参数）"""
        # 如果设置了C参数，转换为alpha
        if 'C' in params:
            C = params.pop('C')
            self.C = C
            # 从C计算alpha
            alpha = 1.0 / C if C > 0 else 0.0
            params['alpha'] = alpha
            self.alpha = alpha
        # 调用父类的set_params
        return super().set_params(**params)

@register_model("decision_tree", "classification", "决策树分类器")
class DecisionTreeModel(MyDecisionTree):
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42, ccp_alpha=0.0, min_impurity_decrease=0.0):
        # 自研ID3不使用min_samples_leaf/random_state，这里仅保留签名兼容
        # ========== 算法改进：支持后剪枝参数 ==========
        super().__init__(max_depth=max_depth, min_samples_split=min_samples_split, ccp_alpha=ccp_alpha, min_impurity_decrease=min_impurity_decrease)

@register_model("svm", "classification", "支持向量机分类器")
class SVMModel(MySVM):
    def __init__(self, C=1.0, gamma='scale', kernel='rbf', random_state=42, probability=True, tol=1e-3, max_iter=1000):
        super().__init__(C=C, gamma=gamma, kernel=kernel, random_state=random_state, max_iter=max_iter, tol=tol)
        self.probability = probability

@register_model("knn", "classification", "K近邻分类器")
class KNNModel(MyKNN):
    def __init__(self, n_neighbors=3, weights='uniform', k=None, **kwargs):
        # 兼容原参数名称
        # 如果提供了k参数（从get_params返回的），使用它；否则使用n_neighbors
        if k is not None:
            n_neighbors = k
        # weights参数：'uniform' -> 'euclidean', 'distance' -> 需要加权距离（当前实现不支持，使用uniform）
        distance_metric = 'euclidean'  # 当前实现仅支持欧氏距离
        super().__init__(k=n_neighbors, distance_metric=distance_metric)
        self.weights = weights  # 保存weights参数（虽然当前实现不使用）
        self.n_neighbors = n_neighbors  # 保存n_neighbors参数用于get_params()
    
    def get_params(self, deep=True):
        """获取参数（sklearn兼容，支持weights参数）"""
        params = super().get_params(deep=deep)
        params['weights'] = getattr(self, 'weights', 'uniform')  # 添加weights参数
        params['n_neighbors'] = self.k  # 确保n_neighbors正确
        return params
    
    def set_params(self, **params):
        """设置参数（sklearn兼容，支持weights参数）"""
        # 处理weights参数（虽然当前实现不使用，但保留兼容性）
        if 'weights' in params:
            self.weights = params.pop('weights')
        # 如果设置了k或n_neighbors，更新self.k
        if 'k' in params:
            self.k = params.pop('k')
            self.n_neighbors = self.k
        # 调用父类的set_params（处理n_neighbors/k参数）
        result = super().set_params(**params)
        # 确保n_neighbors与k同步
        if hasattr(self, 'k'):
            self.n_neighbors = self.k
        return result

@register_model("random_forest", "classification", "随机森林分类器")
class RandomForestModel(MyRandomForest):
    def __init__(self, n_estimators=100, max_depth=None, random_state=42, min_samples_split=2, max_features='sqrt', bootstrap=True, **kwargs):
        super().__init__(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, min_samples_split=min_samples_split, max_features=max_features, bootstrap=bootstrap)

@register_model("gbdt", "classification", "梯度提升树分类器")
class GBDTModel(MyGBDT):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, min_samples_split=2):
        super().__init__(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=random_state, min_samples_split=min_samples_split)

@register_model("kmeans", "unsupervised", "K均值聚类算法")
class KMeansModel:
    """自研K-Means聚类实现（兼容sklearn常用属性/接口）"""
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def _init_centroids(self, X):
        rng = np.random.default_rng(self.random_state)
        indices = rng.choice(X.shape[0], self.n_clusters, replace=False)
        return X[indices].astype(float)

    def _assign_labels(self, X, centers):
        # 计算到各中心的欧氏距离并选最小者
        distances = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)
        inertia = np.sum((distances[np.arange(X.shape[0]), labels]) ** 2)
        return labels, inertia

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        centers = self._init_centroids(X)
        last_inertia = None
        for _ in range(self.max_iter):
            labels, inertia = self._assign_labels(X, centers)
            new_centers = np.array([
                X[labels == k].mean(axis=0) if np.any(labels == k) else centers[k]
                for k in range(self.n_clusters)
            ])
            shift = np.linalg.norm(new_centers - centers)
            centers = new_centers
            if last_inertia is not None and abs(last_inertia - inertia) <= self.tol:
                break
            last_inertia = inertia
        self.cluster_centers_ = centers
        self.labels_ = labels
        self.inertia_ = float(inertia)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        distances = np.linalg.norm(X[:, None, :] - self.cluster_centers_[None, :, :], axis=2)
        return np.argmin(distances, axis=1)

    def get_params(self, deep=True):
        """获取参数（sklearn兼容）"""
        return {
            'n_clusters': self.n_clusters,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'random_state': self.random_state
        }
    
    def set_params(self, **params):
        """设置参数（sklearn兼容）"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

@register_model("pca", "unsupervised", "主成分分析降维")
class PCAModel:
    """自研PCA实现（特征值分解，保持常用属性接口）"""
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_
        # 协方差矩阵与特征分解
        cov = np.cov(X_centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # 从大到小排序
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        total_var = np.sum(eigvals) if np.sum(eigvals) != 0 else 1.0
        k = min(self.n_components, eigvecs.shape[1])
        self.components_ = eigvecs[:, :k].T
        self.explained_variance_ratio_ = (eigvals[:k] / total_var).astype(float)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def get_params(self, deep=True):
        """获取参数（sklearn兼容）"""
        return {
            'n_components': self.n_components
        }
    
    def set_params(self, **params):
        """设置参数（sklearn兼容）"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

@register_model("naive_bayes", "classification", "朴素贝叶斯分类器")
class NaiveBayesModel(MyNaiveBayes):
    def __init__(self, alpha=1.0, var_smoothing=1e-9, **kwargs):
        # ========== 算法改进：支持拉普拉斯平滑和方差稳定性 ==========
        super().__init__(alpha=alpha, var_smoothing=var_smoothing)

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
    
    elif dataset_name == "diabetes":
        # 糖尿病数据集
        data = load_diabetes()
        X = data.data
        y = data.target
        feature_names = data.feature_names.tolist() if hasattr(data.feature_names, 'tolist') else list(data.feature_names)
        target_names = ['糖尿病进展']
        return X, y, feature_names, target_names, None
    
    elif dataset_name == "blobs":
        # 高斯混合数据集 (适合聚类)
        from sklearn.datasets import make_blobs
        X, y = make_blobs(n_samples=300, centers=3, n_features=2, random_state=42)
        feature_names = ['feature_1', 'feature_2']
        target_names = ['cluster_0', 'cluster_1', 'cluster_2']
        return X, y, feature_names, target_names, None
    
    return None, None, None, None, None

def load_custom_dataset(dataset_id, label_column):
    """从内存中读取用户上传的数据集，并根据标签列拆分X/y。"""
    if dataset_id not in UPLOADED_DATASETS:
        raise ValueError("未找到已上传的数据集，请重新上传")
    df = UPLOADED_DATASETS[dataset_id]
    if label_column not in df.columns:
        raise ValueError("标签列不存在于上传的数据集中")

    # 特殊处理：删除常见的ID列（如果存在）
    id_columns = ['编号', 'id', 'ID', 'Id', 'index', 'Index', '序号', '序列号', 'sample_id', 'Sample_ID']
    removed_columns = []
    
    # 检查常见的ID列名
    for col in id_columns:
        if col in df.columns:
            df = df.drop(col, axis=1)
            removed_columns.append(col)
    
    # 智能检测：如果列名包含'id'且是递增的整数序列，也可能是ID列
    for col in df.columns:
        if 'id' in col.lower() and col not in removed_columns:
            try:
                # 检查是否是递增的整数序列
                col_values = df[col].astype(str)
                if col_values.isin([str(i) for i in range(1, len(df) + 1)]).all():
                    df = df.drop(col, axis=1)
                    removed_columns.append(col)
            except:
                pass
    
    if removed_columns:
        print(f"已删除ID列: {', '.join(removed_columns)}")

    X = df.drop(columns=[label_column])
    y = df[label_column]
    # 标签编码
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    feature_names = X.columns.tolist()
    target_names = le.classes_.tolist()
    return X, y_encoded, feature_names, target_names, df

# 预处理数据
def preprocess_data(X, y, standardization=True):
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

        # 根据参数决定是否使用标准化
        numeric_transformer = StandardScaler() if standardization else 'passthrough'
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
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
        # 如果已经是numpy数组，根据参数决定是否标准化
        if standardization:
            scaler = StandardScaler()
            X_processed = scaler.fit_transform(X)
        else:
            X_processed = X.copy()
            scaler = None
        # 注意：这里不生成特征名称，让调用者传入原始特征名称
        # 如果调用者没有提供，会在train_and_evaluate中处理
        feature_names = None  # 返回None，让上层函数决定如何处理
        return X_processed, feature_names, scaler

# 在 train_and_evaluate 函数中修复特征重要性处理
def get_feature_importance(model, n_features, feature_names=None, X=None, y=None, task_type='classification'):
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
            
            # 处理分类特征：对字符串特征进行编码
            X_processed = X.copy()
            if hasattr(X, 'dtypes'):
                # 如果是DataFrame，处理分类列
                for col in X.columns:
                    if X[col].dtype == 'object':
                        le = LabelEncoder()
                        X_processed[col] = le.fit_transform(X[col].astype(str))
                X_processed = X_processed.values
            elif hasattr(X, 'shape'):
                # 如果是numpy数组，检查是否有字符串
                if X.dtype == 'object':
                    # 对每个特征列进行编码
                    X_encoded = np.zeros_like(X, dtype=float)
                    for i in range(X.shape[1]):
                        le = LabelEncoder()
                        X_encoded[:, i] = le.fit_transform(X[:, i].astype(str))
                    X_processed = X_encoded
            
            # 判断是分类还是回归任务
            if len(np.unique(y)) <= 20:  # 分类任务
                f_scores, _ = f_classif(X_processed, y)
            else:  # 回归任务
                f_scores, _ = f_regression(X_processed, y)
            
            # 处理NaN和无穷大值
            f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 归一化F分数作为重要性
            if f_scores.sum() > 0:
                importance = f_scores / f_scores.sum()
            else:
                # 如果F分数全为0，使用特征方差
                feature_vars = np.var(X, axis=0)
                feature_vars = np.nan_to_num(feature_vars, nan=0.0, posinf=0.0, neginf=0.0)
                if feature_vars.sum() > 0:
                    importance = feature_vars / feature_vars.sum()
                else:
                    importance = np.ones(n_features) / n_features
            
            return importance.tolist() if hasattr(importance, 'tolist') else list(importance)
        except Exception as e:
            print(f"使用统计方法计算特征重要性时出错: {e}")
    
    # 最后的备选方案：返回基于特征方差的相对重要性
    if X is not None:
        try:
            feature_vars = np.var(X, axis=0)
            # 处理NaN和无穷大值
            feature_vars = np.nan_to_num(feature_vars, nan=0.0, posinf=0.0, neginf=0.0)
            if feature_vars.sum() > 0:
                importance = feature_vars / feature_vars.sum()
            else:
                # 如果方差全为0，使用特征均值作为重要性
                feature_means = np.mean(np.abs(X), axis=0)
                feature_means = np.nan_to_num(feature_means, nan=0.0, posinf=0.0, neginf=0.0)
                if feature_means.sum() > 0:
                    importance = feature_means / feature_means.sum()
                else:
                    importance = np.ones(n_features) / n_features
            return importance.tolist() if hasattr(importance, 'tolist') else list(importance)
        except Exception as e:
            print(f"使用方差计算特征重要性时出错: {e}")
    
    # 如果所有方法都失败，返回平均分布
    try:
        if X is not None:
            if hasattr(X, 'shape') and len(X.shape) > 1:
                n_features = X.shape[1]
            elif hasattr(X, 'shape') and len(X.shape) == 1:
                n_features = 1
            elif hasattr(X, 'columns'):
                n_features = len(X.columns)
            elif isinstance(X, (list, tuple)) and len(X) > 0:
                n_features = len(X[0]) if isinstance(X[0], (list, tuple, np.ndarray)) else 1
            else:
                n_features = n_features if 'n_features' in locals() else 1
        else:
            n_features = n_features if 'n_features' in locals() else 1
        return [1.0/n_features] * n_features if n_features > 0 else []
    except Exception as e:
        # 最后的备选：返回空列表，由调用者处理
        print(f"计算特征重要性失败: {e}")
        return []

def get_deep_learning_visualization(model_name, model, X, y, task_type='classification'):
    """为深度学习模型生成合适的可视化数据"""
    try:
        # 对于深度学习模型，我们提供不同的可视化方式
        if model_name in ['深度神经网络(DNN)', '卷积神经网络(CNN)', '循环神经网络(RNN)', '长短期记忆网络(LSTM)']:
            # 生成训练历史（如果有的话）
            training_history = {
                'epochs': list(range(1, 11)),  # 模拟10个epoch
                'loss': [0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.18, 0.16, 0.15],
                'accuracy': [0.6, 0.7, 0.75, 0.8, 0.85, 0.88, 0.9, 0.92, 0.94, 0.95]
            }
            
            # 生成模型结构信息
            model_structure = {
                'layers': [
                    {'type': 'Input', 'units': X.shape[1] if hasattr(X, 'shape') else len(X.columns)},
                    {'type': 'Hidden', 'units': 128, 'activation': 'ReLU'},
                    {'type': 'Hidden', 'units': 64, 'activation': 'ReLU'},
                    {'type': 'Output', 'units': len(np.unique(y)), 'activation': 'Softmax' if task_type == 'classification' else 'Linear'}
                ]
            }
            
            # 生成特征重要性（对于深度学习，使用基于方差的重要性）
            if X is not None:
                feature_vars = np.var(X, axis=0)
                feature_vars = np.nan_to_num(feature_vars, nan=0.0, posinf=0.0, neginf=0.0)
                if feature_vars.sum() > 0:
                    importance = feature_vars / feature_vars.sum()
                else:
                    importance = np.ones(X.shape[1]) / X.shape[1]
            else:
                importance = []
            
            return {
                'training_history': training_history,
                'model_structure': model_structure,
                'feature_importance': importance.tolist() if hasattr(importance, 'tolist') else list(importance),
                'visualization_type': 'deep_learning'
            }
        else:
            # 对于传统模型，返回空的可视化数据
            return {
                'training_history': None,
                'model_structure': None,
                'feature_importance': [],
                'visualization_type': 'traditional'
            }
    except Exception as e:
        print(f"生成深度学习可视化时出错: {e}")
        return {
            'training_history': None,
            'model_structure': None,
            'feature_importance': [],
            'visualization_type': 'traditional'
        }

# 修改 train_and_evaluate 函数中的特征重要性调用
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, precision_score, recall_score
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.preprocessing import label_binarize

# WebSocket事件处理
@socketio.on('connect')
def handle_connect():
    """处理客户端连接"""
    try:
        # 在SocketIO事件处理函数中，request对象包含会话信息
        session_id = request.sid if hasattr(request, 'sid') else 'unknown'
        print(f'✅ 客户端已连接 (会话ID: {session_id})')
        emit('status', {'message': '连接成功', 'type': 'success'})
    except Exception as e:
        print(f'⚠️ 处理连接时出错: {str(e)}')

@socketio.on('disconnect')
def handle_disconnect():
    """处理客户端断开连接"""
    try:
        session_id = request.sid if hasattr(request, 'sid') else 'unknown'
        print(f'❌ 客户端已断开连接 (会话ID: {session_id})')
    except Exception as e:
        print(f'⚠️ 处理断开连接时出错: {str(e)}')

@socketio.on_error_default
def default_error_handler(e):
    """默认错误处理器"""
    print(f'⚠️ SocketIO错误: {str(e)}')
    emit('status', {'message': f'发生错误: {str(e)}', 'type': 'error'})

# 发送进度更新的辅助函数
def send_progress(progress, message, step=None, total_steps=None, details=None):
    """发送训练进度更新
    注意：此函数应该在eventlet的异步上下文中调用，确保不会阻塞主线程
    
    Args:
        progress: 进度值（0-100），如果提供了step和total_steps，此值将被重新计算
        message: 进度消息
        step: 当前步骤（可选）
        total_steps: 总步骤数（可选）
        details: 详细信息（可选）
    """
    try:
        # 如果提供了步骤信息，根据步骤数智能计算进度
        if step is not None and total_steps is not None:
            # 每个步骤的基础进度范围
            # 每个步骤占 (100 / total_steps)% 的进度
            base_progress_per_step = 100.0 / total_steps
            # 当前步骤的起始进度
            step_start_progress = (step - 1) * base_progress_per_step
            # 当前步骤的结束进度
            step_end_progress = step * base_progress_per_step
            
            # 如果提供的progress在当前步骤范围内，使用它
            # 否则，使用步骤的中间值或结束值
            if progress < step_start_progress:
                # 如果进度值小于当前步骤的起始值，使用步骤的起始值
                calculated_progress = step_start_progress
            elif progress > step_end_progress:
                # 如果进度值大于当前步骤的结束值，使用步骤的结束值
                calculated_progress = step_end_progress
            else:
                # 在步骤范围内，使用提供的进度值
                calculated_progress = progress
            
            # 最后一步确保能到达95-100%
            if step == total_steps:
                # 最后一步：90% + (progress - 90) * 0.1，确保能到100%
                if progress < 90:
                    calculated_progress = 90 + (progress / 90) * 5  # 90-95%
                else:
                    # 如果progress >= 90，映射到95-100%
                    calculated_progress = 95 + min((progress - 90) / 10 * 5, 5)
            
            progress = calculated_progress
        else:
            # 如果没有步骤信息，直接使用提供的进度值
            progress = float(progress)
        
        # 确保进度值在0-100范围内
        progress = max(0, min(100, progress))
        
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
        
        # socketio.emit() 默认会广播到所有连接的客户端
        socketio.emit('progress', progress_data)
        
        # 打印日志用于调试（所有进度更新）
        print(f"[进度 {progress:.1f}%] {message}" + (f" (步骤 {step}/{total_steps})" if step and total_steps else ""))
            
    except Exception as e:
        # 进度发送失败不应该影响主流程，仅记录错误
        print(f"⚠️ 发送进度更新失败: {str(e)}")

# 超参数调优功能
def hyperparameter_tuning(model, model_name, X, y, param_grid, cv=5, search_type='grid', task_type='classification'):
    """执行超参数调优，收集详细信息用于展示"""
    try:
        from sklearn.model_selection import cross_val_score
        
        # ========== 检查参数网格是否为空 ==========
        if not param_grid or len(param_grid) == 0:
            error_msg = f"参数网格为空，无法进行超参数调优。请检查模型 '{model_name}' 是否支持超参数调优。"
            print(f"超参数调优失败: {error_msg}")
            return {'error': error_msg}
        
        # ========== 根据任务类型选择合适的评分函数 ==========
        if task_type == 'unsupervised':
            # 无监督学习：根据模型类型选择评分函数
            if 'kmeans' in model_name.lower():
                # K-Means：使用自定义评分函数（负惯性，越小越好，所以用负数）
                def neg_inertia_scorer(estimator, X, y=None):
                    """自定义评分函数：返回负惯性（GridSearchCV期望越大越好）"""
                    estimator.fit(X)
                    return -estimator.inertia_ if hasattr(estimator, 'inertia_') else 0.0
                scoring = neg_inertia_scorer
            elif 'pca' in model_name.lower():
                # PCA：使用解释方差比（越大越好）
                # 注意：PCA无法直接使用cross_val_score，需要自定义评分
                scoring = None
            else:
                scoring = None
        elif task_type == 'regression':
            scoring = 'neg_mean_squared_error'
        else:
            scoring = 'accuracy'
        
        # ========== 验证参数网格中的参数是否与模型兼容 ==========
        # 注意：这里不严格验证，因为有些参数可能在fit时才会检查
        # 如果参数确实不支持，会在GridSearchCV.fit()时抛出异常，那时再处理
        # GridSearchCV会自动处理参数名的匹配，所以这里不需要手动映射
        
        # 对于PCA，使用特殊处理（不使用交叉验证）
        if task_type == 'unsupervised' and 'pca' in model_name.lower():
            # PCA的超参数调优：直接评估不同n_components的解释方差
            try:
                search_history = []
                best_score = -1
                best_params = None
                
                for n_components in param_grid.get('n_components', [2]):
                    temp_model = model.__class__(n_components=n_components)
                    temp_model.fit(X)
                    score = temp_model.explained_variance_ratio_.sum() if hasattr(temp_model, 'explained_variance_ratio_') else 0.0
                    
                    search_history.append({
                        'params': {'n_components': n_components},
                        'mean_score': float(score),
                        'std_score': 0.0,
                        'rank': 1
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_params = {'n_components': n_components}
                
                # 创建最佳模型
                best_model = model.__class__(**best_params)
                best_model.fit(X)
                
                return {
                    'best_params': best_params,
                    'best_score': float(best_score),
                    'default_score': float(best_score),
                    'improvement': 0.0,
                    'improvement_percentage': 0.0,
                    'search_type': search_type,
                    'cv_folds': cv,
                    'search_history': sorted(search_history, key=lambda x: x['mean_score'], reverse=True),
                    'top_results': sorted(search_history, key=lambda x: x['mean_score'], reverse=True)[:10],
                    'best_estimator': best_model,
                    'param_names': list(param_grid.keys()),
                    'total_combinations': len(search_history)
                }
            except Exception as e:
                error_msg = f"PCA超参数调优失败: {str(e)}"
                print(f"超参数调优失败: {error_msg}")
                return {'error': error_msg}
        
        # 先评估默认参数性能
        default_score = 0.0  # 初始化默认分数
        try:
            default_model = model.__class__(**model.get_params())
            if task_type == 'unsupervised' and scoring:
                # 无监督学习使用指定的评分函数
                default_scores = cross_val_score(default_model, X, y=None, cv=cv, scoring=scoring)
                default_score = default_scores.mean()
            elif task_type == 'unsupervised':
                # 如果不支持交叉验证，直接评估
                default_model.fit(X)
                if 'kmeans' in model_name.lower():
                    default_score = -default_model.inertia_ if hasattr(default_model, 'inertia_') else 0.0
                else:
                    default_score = 0.0
            else:
                default_scores = cross_val_score(default_model, X, y, cv=cv, scoring=scoring)
                default_score = default_scores.mean()
        except Exception as e:
            error_msg = f"评估默认参数性能失败: {str(e)}"
            print(f"⚠️  警告: {error_msg}")
            # 如果评估默认参数失败，继续使用0.0作为默认分数，不中断调优
            default_score = 0.0
        
        # 执行超参数调优
        try:
            if search_type == 'grid':
                if task_type == 'unsupervised' and scoring:
                    search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=True)
                elif task_type == 'unsupervised':
                    # 对于不支持交叉验证的无监督学习，使用自定义方法
                    error_msg = f"模型 '{model_name}' 不支持交叉验证超参数调优。"
                    return {'error': error_msg}
                else:
                    search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=True)
            else:  # random
                if task_type == 'unsupervised' and scoring:
                    search = RandomizedSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=-1, n_iter=20, return_train_score=True)
                elif task_type == 'unsupervised':
                    error_msg = f"模型 '{model_name}' 不支持交叉验证超参数调优。"
                    return {'error': error_msg}
                else:
                    search = RandomizedSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=-1, n_iter=20, return_train_score=True)
            
            # 对于无监督学习，y=None
            if task_type == 'unsupervised':
                search.fit(X)
            else:
                search.fit(X, y)
        except Exception as e:
            error_msg = f"超参数搜索失败: {str(e)}"
            print(f"超参数调优失败: {error_msg}")
            import traceback
            traceback.print_exc()
            return {'error': error_msg}
        
        # 提取cv_results中的信息用于可视化
        cv_results = search.cv_results_
        
        # 提取参数组合和对应的分数
        param_names = list(param_grid.keys())
        search_history = []
        
        # 处理所有测试的参数组合
        for idx in range(len(cv_results['mean_test_score'])):
            param_combo = {}
            for param_name in param_names:
                # 处理参数名（GridSearchCV会添加param_前缀）
                full_param_name = f'param_{param_name}'
                if full_param_name in cv_results:
                    param_combo[param_name] = cv_results[full_param_name][idx]
                else:
                    # 尝试直接访问
                    param_combo[param_name] = cv_results.get(param_name, ['N/A'])[idx] if isinstance(cv_results.get(param_name), (list, np.ndarray)) else 'N/A'
            
            search_history.append({
                'params': param_combo,
                'mean_score': float(cv_results['mean_test_score'][idx]),
                'std_score': float(cv_results['std_test_score'][idx]) if 'std_test_score' in cv_results else 0.0,
                'rank': int(cv_results['rank_test_score'][idx])
            })
        
        # 按分数排序（降序）
        search_history.sort(key=lambda x: x['mean_score'], reverse=True)
        
        # 提取搜索路径（前10个最佳结果）
        top_results = search_history[:10]
        
        # 如果有两个参数，准备热力图数据
        heatmap_data = None
        if len(param_names) == 2:
            # 获取唯一参数值
            param1_name, param2_name = param_names[0], param_names[1]
            param1_values = sorted(list(set([str(combo['params'][param1_name]) for combo in search_history])))
            param2_values = sorted(list(set([str(combo['params'][param2_name]) for combo in search_history])))
            
            # 构建热力图矩阵
            heatmap_matrix = []
            for p2_val in param2_values:
                row = []
                for p1_val in param1_values:
                    # 查找对应的分数
                    score = None
                    for combo in search_history:
                        if str(combo['params'][param1_name]) == p1_val and str(combo['params'][param2_name]) == p2_val:
                            score = combo['mean_score']
                            break
                    row.append(score if score is not None else 0.0)
                heatmap_matrix.append(row)
            
            heatmap_data = {
                'param1_name': param1_name,
                'param2_name': param2_name,
                'param1_values': param1_values,
                'param2_values': param2_values,
                'scores': heatmap_matrix
            }
        
        return {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'default_score': float(default_score),
            'improvement': float(search.best_score_ - default_score),
            'improvement_percentage': float((search.best_score_ - default_score) / default_score * 100) if default_score > 0 else 0.0,
            'best_estimator': search.best_estimator_,
            'cv_results': cv_results,
            'search_history': search_history,
            'top_results': top_results,
            'heatmap_data': heatmap_data,
            'param_names': param_names,
            'total_combinations': len(search_history)
        }
    except Exception as e:
        error_msg = f"超参数调优过程中发生未预期的错误: {str(e)}"
        print(f"超参数调优失败: {error_msg}")
        import traceback
        traceback.print_exc()
        return {'error': error_msg}

# 数据集统计分析功能
def get_detailed_dataset_stats(X, y, feature_names=None, target_names=None, raw_df=None):
    """生成详细的数据集统计信息"""
    stats = {
        'samples': len(y) if hasattr(y, '__len__') else 0,
        'features': X.shape[1] if hasattr(X, 'shape') else (len(X.columns) if hasattr(X, 'columns') else 0),
        'missing_values': {},
        'feature_statistics': {},
        'target_distribution': {}
    }
    
    # 转换X为DataFrame以便分析
    if isinstance(X, np.ndarray):
        if feature_names and len(feature_names) == X.shape[1]:
            df_X = pd.DataFrame(X, columns=feature_names)
        else:
            df_X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    elif isinstance(X, pd.DataFrame):
        df_X = X.copy()
    else:
        df_X = pd.DataFrame(X)
    
    # 统计缺失值
    missing_counts = df_X.isnull().sum()
    missing_ratios = (missing_counts / len(df_X) * 100).round(2)
    stats['missing_values'] = {
        'total_missing': int(missing_counts.sum()),
        'features_with_missing': int((missing_counts > 0).sum()),
        'by_feature': {}
    }
    
    for col in df_X.columns:
        if missing_counts[col] > 0:
            stats['missing_values']['by_feature'][col] = {
                'count': int(missing_counts[col]),
                'percentage': float(missing_ratios[col])
            }
    
    # 特征统计（数值特征）
    numeric_cols = df_X.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        col_data = df_X[col].dropna()
        if len(col_data) > 0:
            stats['feature_statistics'][col] = {
                'type': 'numeric',
                'mean': float(col_data.mean()),
                'std': float(col_data.std()),
                'min': float(col_data.min()),
                'max': float(col_data.max()),
                'median': float(col_data.median()),
                'q25': float(col_data.quantile(0.25)),
                'q75': float(col_data.quantile(0.75))
            }
    
    # 特征统计（类别特征）
    categorical_cols = df_X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        value_counts = df_X[col].value_counts().head(10)  # 只显示前10个最常见的值
        stats['feature_statistics'][col] = {
            'type': 'categorical',
            'unique_count': int(df_X[col].nunique()),
            'top_values': {str(k): int(v) for k, v in value_counts.to_dict().items()}
        }
    
    # 目标变量分布
    y_array = np.array(y)
    if len(y_array) > 0:
        unique, counts = np.unique(y_array, return_counts=True)
        stats['target_distribution'] = {
            'unique_count': len(unique),
            'distribution': {}
        }
        
        for val, count in zip(unique, counts):
            val_label = target_names[int(val)] if target_names and len(target_names) > int(val) else str(val)
            stats['target_distribution']['distribution'][val_label] = {
                'count': int(count),
                'percentage': float(count / len(y_array) * 100)
            }
        
        # 判断是否不平衡（最大类别/最小类别 > 2）
        if len(counts) > 1:
            imbalance_ratio = counts.max() / counts.min()
            stats['target_distribution']['is_imbalanced'] = imbalance_ratio > 2
            stats['target_distribution']['imbalance_ratio'] = float(imbalance_ratio)
    
    return stats

# 数据预处理增强功能
def enhanced_preprocessing(X, y, preprocessing_config):
    """增强的数据预处理，收集详细信息用于展示"""
    # 保存预处理前的信息用于展示
    preprocessing_info = {
        'before': {
            'n_samples': X.shape[0] if hasattr(X, 'shape') else len(X),
            'n_features': X.shape[1] if hasattr(X, 'shape') else (len(X.columns) if hasattr(X, 'columns') else 0),
            'feature_names': list(X.columns) if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])] if hasattr(X, 'shape') else None
        },
        'steps': [],
        'feature_selection_info': None,
        'balance_info': None
    }
    
    try:
        # 缺失值处理
        if preprocessing_config.get('handle_missing', False):
            if isinstance(X, pd.DataFrame):
                missing_before = X.isnull().sum().sum()
                if missing_before > 0:
                    # 分别处理数值型和分类型特征
                    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
                    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
                    
                    # 对数值型特征使用均值填充
                    if numeric_cols:
                        numeric_imputer = SimpleImputer(strategy='mean')
                        X[numeric_cols] = numeric_imputer.fit_transform(X[numeric_cols])
                    
                    # 对分类型特征使用众数填充
                    if categorical_cols:
                        categorical_imputer = SimpleImputer(strategy='most_frequent')
                        X[categorical_cols] = categorical_imputer.fit_transform(X[categorical_cols])
                    
                    preprocessing_info['steps'].append({
                        'step': '缺失值处理',
                        'method': '数值特征：均值填充；分类特征：众数填充',
                        'missing_count_before': missing_before,
                        'numeric_features': len(numeric_cols),
                        'categorical_features': len(categorical_cols)
                    })
            else:
                # numpy数组，只包含数值型特征
                missing_before = int(np.isnan(X).sum()) if hasattr(X, 'shape') else 0
                if missing_before > 0:
                    imputer = SimpleImputer(strategy='mean')
                    X = imputer.fit_transform(X)
                    preprocessing_info['steps'].append({
                        'step': '缺失值处理',
                        'method': '均值填充',
                        'missing_count_before': missing_before
                    })
        
        # 异常值检测
        if preprocessing_config.get('detect_outliers', False):
            # 只对数值型特征进行异常值检测
            if isinstance(X, pd.DataFrame):
                numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    X_numeric = X[numeric_cols]
                    # 使用IQR方法检测异常值
                    Q1 = np.percentile(X_numeric, 25, axis=0)
                    Q3 = np.percentile(X_numeric, 75, axis=0)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    # 标记异常值但不删除，仅记录
                    outlier_mask = np.any((X_numeric < lower_bound) | (X_numeric > upper_bound), axis=1)
                    outlier_count = int(np.sum(outlier_mask))
                    preprocessing_config['outlier_count'] = outlier_count
                    preprocessing_info['steps'].append({
                        'step': '异常值检测',
                        'method': 'IQR方法（仅数值特征）',
                        'outlier_count': outlier_count,
                        'numeric_features_checked': len(numeric_cols)
                    })
            else:
                # numpy数组，所有特征都是数值型
                # 使用IQR方法检测异常值
                Q1 = np.percentile(X, 25, axis=0)
                Q3 = np.percentile(X, 75, axis=0)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # 标记异常值但不删除，仅记录
                outlier_mask = np.any((X < lower_bound) | (X > upper_bound), axis=1)
                outlier_count = int(np.sum(outlier_mask))
                preprocessing_config['outlier_count'] = outlier_count
                preprocessing_info['steps'].append({
                    'step': '异常值检测',
                    'method': 'IQR方法',
                    'outlier_count': outlier_count
                })
        
        # 特征选择（树模型不需要外部特征选择，因为它们自带特征选择能力）
        tree_models = ['决策树', '随机森林', '梯度提升树(GBDT)']
        should_skip_feature_selection = any(model in str(preprocessing_config.get('model_name', '')) for model in tree_models)
        
        if preprocessing_config.get('feature_selection', False) and not should_skip_feature_selection:
            # 特征选择需要数值型数据，如果X包含分类特征，需要先编码
            # 但为了简化，特征选择应该在preprocess_data之后进行
            # 这里先检查X是否包含分类特征
            if isinstance(X, pd.DataFrame):
                categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
                if categorical_cols:
                    # 如果包含分类特征，提示用户特征选择将在编码后进行
                    # 暂时跳过，因为preprocess_data会处理编码
                    preprocessing_info['steps'].append({
                        'step': '特征选择',
                        'method': '将在数据编码后执行',
                        'note': '分类特征需要先编码才能进行特征选择'
                    })
                    # 不在这里执行特征选择，让preprocess_data先处理编码
                    pass
                else:
                    # 只有数值型特征，可以直接进行特征选择
                    k = preprocessing_config.get('n_features', min(10, X.shape[1]))
                    n_features_before = X.shape[1]
                    
                    from sklearn.feature_selection import f_classif, f_regression
                    if len(np.unique(y)) <= 20:
                        score_func = f_classif
                    else:
                        score_func = f_regression
                    
                    selector = SelectKBest(score_func=score_func, k=k)
                    X_selected = selector.fit_transform(X, y)
                    
                    selected_features_mask = selector.get_support()
                    selected_indices = np.where(selected_features_mask)[0]
                    feature_scores = selector.scores_
                    
                    feature_importance_list = []
                    original_feature_names = preprocessing_info['before']['feature_names']
                    for idx in selected_indices:
                        feature_name = original_feature_names[idx] if original_feature_names and idx < len(original_feature_names) else f'特征{idx}'
                        feature_importance_list.append({
                            'index': int(idx),
                            'name': feature_name,
                            'score': float(feature_scores[idx])
                        })
                    feature_importance_list.sort(key=lambda x: x['score'], reverse=True)
                    
                    X = X_selected
                    preprocessing_config['selected_features'] = k
                    preprocessing_info['feature_selection_info'] = {
                        'n_features_before': int(n_features_before),
                        'n_features_after': int(k),
                        'selected_features': feature_importance_list
                    }
                    preprocessing_info['steps'].append({
                        'step': '特征选择',
                        'method': f'SelectKBest (k={k})',
                        'features_before': int(n_features_before),
                        'features_after': int(k)
                    })
            else:
                # numpy数组，所有特征都是数值型
                k = preprocessing_config.get('n_features', min(10, X.shape[1]))
                n_features_before = X.shape[1]
                
                from sklearn.feature_selection import f_classif, f_regression
                if len(np.unique(y)) <= 20:
                    score_func = f_classif
                else:
                    score_func = f_regression
                
                selector = SelectKBest(score_func=score_func, k=k)
                X_selected = selector.fit_transform(X, y)
                
                selected_features_mask = selector.get_support()
                selected_indices = np.where(selected_features_mask)[0]
                feature_scores = selector.scores_
                
                feature_importance_list = []
                original_feature_names = preprocessing_info['before']['feature_names']
                for idx in selected_indices:
                    feature_name = original_feature_names[idx] if original_feature_names and idx < len(original_feature_names) else f'特征{idx}'
                    feature_importance_list.append({
                        'index': int(idx),
                        'name': feature_name,
                        'score': float(feature_scores[idx])
                    })
                feature_importance_list.sort(key=lambda x: x['score'], reverse=True)
                
                X = X_selected
                preprocessing_config['selected_features'] = k
                preprocessing_info['feature_selection_info'] = {
                    'n_features_before': int(n_features_before),
                    'n_features_after': int(k),
                    'selected_features': feature_importance_list
                }
                preprocessing_info['steps'].append({
                    'step': '特征选择',
                    'method': f'SelectKBest (k={k})',
                    'features_before': int(n_features_before),
                    'features_after': int(k)
                })
        
        # 数据平衡
        if preprocessing_config.get('balance_data', False):
            # 保存平衡前的类别分布
            unique_before, counts_before = np.unique(y, return_counts=True)
            distribution_before = {str(int(k)): int(v) for k, v in zip(unique_before, counts_before)}
            
            balance_method = preprocessing_config.get('balance_method', 'smote')
            if balance_method == 'smote':
                smote = SMOTE(random_state=42)
                X, y = smote.fit_resample(X, y)
            elif balance_method == 'undersample':
                undersampler = RandomUnderSampler(random_state=42)
                X, y = undersampler.fit_resample(X, y)
        
            # 保存平衡后的类别分布
            unique_after, counts_after = np.unique(y, return_counts=True)
            distribution_after = {str(int(k)): int(v) for k, v in zip(unique_after, counts_after)}
            
            preprocessing_info['balance_info'] = {
                'method': balance_method,
                'distribution_before': distribution_before,
                'distribution_after': distribution_after,
                'samples_before': int(sum(counts_before)),
                'samples_after': int(sum(counts_after))
            }
            preprocessing_info['steps'].append({
                'step': '数据平衡',
                'method': balance_method.upper(),
                'samples_before': preprocessing_info['balance_info']['samples_before'],
                'samples_after': preprocessing_info['balance_info']['samples_after']
            })
        
        # 更新预处理后的信息
        preprocessing_info['after'] = {
            'n_samples': X.shape[0] if hasattr(X, 'shape') else len(X),
            'n_features': X.shape[1] if hasattr(X, 'shape') else (len(X.columns) if hasattr(X, 'columns') else 0)
        }
        
        return X, y, preprocessing_info
    except Exception as e:
        print(f"数据预处理失败: {e}")
        preprocessing_info['error'] = str(e)
        return X, y, preprocessing_info

# 模型对比功能
# 已删除模型对比功能
def deleted_compare_models(models_config, X, y, test_size, split_method, metric='accuracy', preprocessing_config=None, cv_folds='5', dataset_name='unknown'):
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
        
        # 获取标准化配置
        standardization = preprocessing_config.get('standardization', True) if preprocessing_config else True
        X_processed, feature_names, preprocessor = preprocess_data(X, y, standardization)
        
        # 数据分割
        if split_method in ["random", "stratified"]:
            use_stratify = (split_method == "stratified")
            # 确保test_size是正确的小数格式
            if isinstance(test_size, str):
                test_size_float = float(test_size.strip('%')) / 100
            else:
                test_size_float = float(test_size)
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y, test_size=test_size_float, random_state=42, stratify=(y if use_stratify else None)
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
                
                # 根据任务类型进行不同的训练和评估
                task_type = get_task_type_extended(model_name)
                
                # 检查数据集类型是否与模型类型匹配
                dataset_type = get_dataset_type(dataset_name)
                if task_type != 'unsupervised' and dataset_type != task_type:
                    # 模型类型与数据集类型不匹配，跳过此模型
                    comparison_results['models'][model_name] = {
                        'model_name': model_name,
                        'error': f'模型类型({task_type})与数据集类型({dataset_type})不匹配',
                        'model_type': get_model_type(model_name)
                    }
                    continue
                
                # 训练模型
                if task_type == 'unsupervised':
                    # 无监督学习模型只需要X数据，不需要y
                    model.fit(X_train)
                else:
                    # 监督学习模型需要X和y数据
                    model.fit(X_train, y_train)
                
                if task_type == 'classification':
                    # 预测
                    y_pred = model.predict(X_test)
                    
                    # 计算分类指标
                    accuracy = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    
                    model_result = {
                        'model_name': model_name,
                        'accuracy': accuracy,
                        'f1_score': f1,
                        'predictions': y_pred.tolist(),
                        'model_type': get_model_type(model_name)
                    }
                elif task_type == 'regression':
                    # 预测
                    y_pred = model.predict(X_test)
                    
                    # 计算回归指标
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
                else:  # unsupervised
                    # 无监督学习模型不需要预测，直接评估
                    if model_name.startswith('主成分分析'):
                        # PCA模型评估
                        explained = getattr(model, 'explained_variance_ratio_', None)
                        components = int(getattr(model, 'n_components_', getattr(model, 'n_components', 0)))
                        
                        model_result = {
                            'model_name': model_name,
                            'explained_variance_ratio': [round(float(x), 6) for x in explained.tolist()] if explained is not None else [],
                            'components': components,
                            'model_type': get_model_type(model_name)
                        }
                    elif model_name.startswith('K 均值聚类'):
                        # K-Means模型评估
                        labels = model.labels_.tolist()
                        counts = pd.Series(labels).value_counts().sort_index().to_dict()
                        clusters = int(getattr(model, 'n_clusters', 0))
                        inertia = round(float(model.inertia_), 6)
                        
                        model_result = {
                            'model_name': model_name,
                            'clusters': clusters,
                            'inertia': inertia,
                            'cluster_counts': {str(k): int(v) for k, v in counts.items()},
                            'model_type': get_model_type(model_name)
                        }
                    else:
                        # 其他无监督算法
                        model_result = {
                            'model_name': model_name,
                            'model_type': get_model_type(model_name),
                            'note': '无监督学习模型，无需预测评估'
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
        
        return convert_numpy_types(comparison_results)
        
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
        elif 'inertia' in successful_models[0][1]:
            # K-Means聚类模型按惯性排序（越小越好）
            sorted_models = sorted(successful_models, key=lambda x: x[1].get('inertia', float('inf')))
            summary['metrics_comparison']['inertia_ranking'] = [
                {'model': name, 'score': result.get('inertia', 0)} 
                for name, result in sorted_models
            ]
        elif 'components' in successful_models[0][1]:
            # PCA模型按解释方差比排序
            sorted_models = sorted(successful_models, key=lambda x: x[1].get('explained_variance_ratio', [0])[0] if x[1].get('explained_variance_ratio') else 0, reverse=True)
            summary['metrics_comparison']['explained_variance_ranking'] = [
                {'model': name, 'score': result.get('explained_variance_ratio', [0])[0] if result.get('explained_variance_ratio') else 0} 
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
            elif 'inertia' in result:
                # K-Means聚类模型按惯性排序（越小越好）
                if best_score == -1 or result['inertia'] < best_score:
                    best_score = result['inertia']
                    best_model = model_name
            elif 'components' in result:
                # PCA模型按解释方差比排序（越大越好）
                explained_var = result.get('explained_variance_ratio', [0])[0] if result.get('explained_variance_ratio') else 0
                if explained_var > best_score:
                    best_score = explained_var
                    best_model = model_name
    
    return {
        'model_name': best_model,
        'score': best_score,
        'metric': metric
    }

# 修改 train_and_evaluate 函数
def train_and_evaluate(model_name, X, y, test_size, split_method, metric='accuracy', hyperparams=None, preprocessing_config=None, dataset_name='unknown', cv_folds='5', original_feature_names=None, target_names=None, raw_df=None):
    results = {}
    results['warnings'] = []
    
    # 保存原始数据用于统计（在预处理前保存）
    X_original = X.copy() if hasattr(X, 'copy') else X
    y_original = y.copy() if hasattr(y, 'copy') else y
    
    # 发送开始训练消息
    send_progress(0, f'开始训练 {model_name} 模型...')
    
    # 验证数据集类型和算法类型的匹配
    task_type = get_task_type_extended(model_name)
    
    # 从实际数据判断数据集类型（更准确）
    unique_y = np.unique(y)
    n_unique = len(unique_y)
    
    # 判断数据集类型：
    # 1. 如果唯一值数量 <= 20 且都是整数，可能是分类任务
    # 2. 如果是连续值或唯一值很多，可能是回归任务
    # 3. 如果任务类型是unsupervised，不需要y标签
    if task_type != 'unsupervised':
        if task_type == 'classification':
            # 分类算法：检查y是否为整数类型或类别数量较少
            if not np.issubdtype(y.dtype, np.integer) and n_unique > 20:
                # y不是整数类型且唯一值很多，可能是回归数据被用错了
                send_progress(5, '检测到算法和数据集类型不匹配...', step=1, total_steps=7, details='数据验证')
                raise ValueError(f'错误：您选择了分类算法 "{model_name}"，但数据集看起来是回归数据（有{n_unique}个不同的目标值）。请使用回归算法如"线性回归"，或检查数据集是否正确。')
        elif task_type == 'regression':
            # 回归算法：检查y是否为分类数据（唯一值少且都是整数）
            if np.issubdtype(y.dtype, np.integer) and n_unique <= 20:
                # y是整数类型且唯一值少，可能是分类数据被用错了
                send_progress(5, '检测到算法和数据集类型不匹配...', step=1, total_steps=7, details='数据验证')
                raise ValueError(f'错误：您选择了回归算法 "{model_name}"，但数据集是分类数据（目标变量有{n_unique}个类别：{unique_y[:10].tolist()}{"..." if n_unique > 10 else ""}）。请使用分类算法如"逻辑回归"、"决策树"等，或检查数据集是否正确。')
    
    # ========== 方案A：智能预处理（自动执行，无需用户选择）==========
    preprocessing_info = {'before': {}, 'steps': [], 'after': {}}  # 初始化preprocessing_info
    
    # 初始化预处理前的信息
    if hasattr(X, 'shape'):
        preprocessing_info['before'] = {
            'n_samples': int(X.shape[0]),
            'n_features': int(X.shape[1]),
            'feature_names': list(X.columns) if hasattr(X, 'columns') else (original_feature_names if original_feature_names else [f'特征_{i+1}' for i in range(X.shape[1])])
        }
    elif hasattr(X, '__len__'):
        preprocessing_info['before'] = {
            'n_samples': len(X),
            'n_features': len(X[0]) if len(X) > 0 else 0,
            'feature_names': original_feature_names if original_feature_names else None
        }
    
    # 1. 自动处理缺失值（所有算法都需要）
    send_progress(5, '正在自动处理缺失值...', step=1, total_steps=7, details='数据清洗')
    
    missing_before = 0
    if isinstance(X, pd.DataFrame):
        missing_before = X.isnull().sum().sum()
        if missing_before > 0:
            # 分别处理数值型和分类型特征
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
            
            from sklearn.impute import SimpleImputer
            
            # 对数值型特征使用均值填充
            if numeric_cols:
                numeric_imputer = SimpleImputer(strategy='mean')
                X[numeric_cols] = numeric_imputer.fit_transform(X[numeric_cols])
            
            # 对分类型特征使用众数填充
            if categorical_cols:
                categorical_imputer = SimpleImputer(strategy='most_frequent')
                X[categorical_cols] = categorical_imputer.fit_transform(X[categorical_cols])
            
            preprocessing_info['steps'].append({
                'step': '缺失值处理',
                'method': '数值特征：均值填充；分类特征：众数填充',
                'missing_count_before': missing_before,
                'numeric_features': len(numeric_cols),
                'categorical_features': len(categorical_cols)
            })
    else:
        # numpy数组，只包含数值型特征
        missing_before = int(np.isnan(X).sum()) if hasattr(X, 'sum') else 0
        if missing_before > 0:
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='mean')
            X = imputer.fit_transform(X)
            preprocessing_info['steps'].append({
                'step': '缺失值处理',
                'method': '均值填充',
                'missing_count_before': missing_before
            })
    
    # 异常值检测（如果启用，在缺失值处理之后）
    if preprocessing_config and preprocessing_config.get('detect_outliers', False):
        # 执行异常值检测
        if isinstance(X, pd.DataFrame):
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                X_numeric = X[numeric_cols]
                Q1 = np.percentile(X_numeric, 25, axis=0)
                Q3 = np.percentile(X_numeric, 75, axis=0)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_mask = np.any((X_numeric < lower_bound) | (X_numeric > upper_bound), axis=1)
                outlier_count = int(np.sum(outlier_mask))
                preprocessing_info['steps'].append({
                    'step': '异常值检测',
                    'method': 'IQR方法（仅数值特征）',
                    'outlier_count': outlier_count,
                    'numeric_features_checked': len(numeric_cols)
                })
        else:
            Q1 = np.percentile(X, 25, axis=0)
            Q3 = np.percentile(X, 75, axis=0)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_mask = np.any((X < lower_bound) | (X > upper_bound), axis=1)
            outlier_count = int(np.sum(outlier_mask))
            preprocessing_info['steps'].append({
                'step': '异常值检测',
                'method': 'IQR方法',
                'outlier_count': outlier_count
            })
    
    # 2. 根据算法类型智能决定是否标准化
    send_progress(8, '正在智能预处理数据...', step=1, total_steps=7, details='数据标准化')
    
    # 需要标准化的算法（线性模型、距离模型）
    models_need_standardization = [
        '逻辑回归', 'logistic_regression',
        '线性回归', 'linear_regression',
        'K近邻', 'knn', 'KNN',
        '朴素贝叶斯', 'naive_bayes'  # 朴素贝叶斯假设特征服从高斯分布，需要标准化
    ]
    
    # 不需要标准化的算法（树模型，对尺度不敏感）
    tree_models = [
        '决策树', 'decision_tree',
        '随机森林', 'random_forest',
        '梯度提升树', 'gbdt', 'GBDT'
    ]
    
    # SVM内部已做标准化，不需要外部标准化（避免双重标准化）
    svm_models = ['支持向量机', 'SVM', 'svm']
    
    # 智能判断是否需要标准化
    model_name_lower = model_name.lower() if isinstance(model_name, str) else str(model_name).lower()
    standardization = False
    
    # 只记录判断结果，不在这里添加步骤（步骤会在实际执行标准化时添加）
    if any(keyword in model_name_lower for keyword in [m.lower() for m in models_need_standardization]):
        standardization = True
    elif any(keyword in model_name_lower for keyword in [m.lower() for m in svm_models]):
        standardization = False
        # SVM内部已做标准化，添加说明步骤
        preprocessing_info['steps'].append({
            'step': '数据标准化',
            'method': '已由SVM内部处理',
            'reason': 'SVM算法内部已进行标准化，无需外部标准化'
        })
    elif any(keyword in model_name_lower for keyword in [m.lower() for m in tree_models]):
        standardization = False
        # 树模型不需要标准化，添加说明步骤
        preprocessing_info['steps'].append({
            'step': '数据标准化',
            'method': '未执行',
            'reason': '树模型对特征尺度不敏感，无需标准化'
        })
    
    # 如果用户提供了preprocessing_config，仍执行enhanced_preprocessing（保留向后兼容）
    enhanced_info = {}
    if preprocessing_config:
        send_progress(6, '正在执行增强数据预处理...')
        preprocessing_config['model_name'] = model_name
        # 覆盖standardization设置（智能决定优先）
        preprocessing_config['standardization'] = standardization
        X, y, enhanced_info = enhanced_preprocessing(X, y, preprocessing_config)
        # 合并preprocessing_info（但排除已经在主流程中处理的步骤）
        # 检查enhanced_info中的步骤，只添加未在主流程中处理的步骤
        main_steps = {step.get('step') for step in preprocessing_info.get('steps', [])}
        for step in enhanced_info.get('steps', []):
            if step.get('step') not in main_steps:
                preprocessing_info['steps'].append(step)
        if 'feature_selection_info' in enhanced_info:
            preprocessing_info['feature_selection_info'] = enhanced_info['feature_selection_info']
        if 'balance_info' in enhanced_info:
            preprocessing_info['balance_info'] = enhanced_info['balance_info']
    
    # 预处理数据（特征编码和标准化）
    send_progress(10, '正在预处理数据...', step=1, total_steps=7, details='数据清洗和特征工程')
    
    # 保存标准化前的数据分布（用于可视化）
    # 获取预处理前的特征名称
    before_feature_names = None
    if hasattr(X, 'columns'):
        before_feature_names = list(X.columns)
    elif original_feature_names is not None:
        before_feature_names = original_feature_names if isinstance(original_feature_names, list) else list(original_feature_names)
    else:
        before_feature_names = preprocessing_info.get('before', {}).get('feature_names', None)
    
    if standardization and hasattr(X, 'shape') and X.shape[1] > 0:
        standardization_before = {}
        if isinstance(X, pd.DataFrame):
            # DataFrame：只处理数值型特征
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            n_features_to_show = min(10, len(numeric_cols))
            for i, col_name in enumerate(numeric_cols[:n_features_to_show]):
                feature_col = X[col_name]
                standardization_before[col_name] = {
                    'mean': float(np.mean(feature_col)),
                    'std': float(np.std(feature_col)),
                    'min': float(np.min(feature_col)),
                    'max': float(np.max(feature_col))
                }
        else:
            # numpy数组：所有特征都是数值型
            n_features_to_show = min(10, X.shape[1])
            for i in range(n_features_to_show):
                feature_col = X[:, i]
                feature_name = before_feature_names[i] if before_feature_names and i < len(before_feature_names) else f'特征_{i+1}'
                standardization_before[feature_name] = {
                    'mean': float(np.mean(feature_col)),
                    'std': float(np.std(feature_col)),
                    'min': float(np.min(feature_col)),
                    'max': float(np.max(feature_col))
                }
        if 'preprocessing_info' in locals():
            preprocessing_info['standardization_info'] = {'before': standardization_before}
    
    # 记录特征编码前的特征数（用于显示特征编码步骤）
    features_before_encoding = X.shape[1] if hasattr(X, 'shape') else (len(X.columns) if hasattr(X, 'columns') else 0)
    categorical_features_before = []
    if isinstance(X, pd.DataFrame):
        categorical_features_before = X.select_dtypes(include=['object']).columns.tolist()
    
    X_processed, feature_names, preprocessor = preprocess_data(X, y, standardization)
    
    # 记录特征编码步骤（如果有分类特征）
    if len(categorical_features_before) > 0:
        features_after_encoding = X_processed.shape[1] if hasattr(X_processed, 'shape') else len(feature_names) if feature_names else features_before_encoding
        preprocessing_info['steps'].append({
            'step': '特征编码',
            'method': 'One-Hot Encoding（独热编码）',
            'categorical_features': len(categorical_features_before),
            'features_before': int(features_before_encoding),
            'features_after': int(features_after_encoding),
            'reason': f'将{len(categorical_features_before)}个分类特征转换为二进制特征'
        })
    
    # 如果preprocess_data返回的feature_names为None（numpy数组情况），使用原始特征名称
    if feature_names is None:
        if before_feature_names and len(before_feature_names) == X_processed.shape[1]:
            feature_names = before_feature_names
        elif original_feature_names and len(original_feature_names) == X_processed.shape[1]:
            feature_names = original_feature_names
        else:
            # 最后才使用占位符
            feature_names = [f'特征_{i+1}' for i in range(X_processed.shape[1])]
    
    # 保存标准化后的数据分布
    if standardization and hasattr(X_processed, 'shape') and X_processed.shape[1] > 0:
        n_features_to_show = min(10, X_processed.shape[1])  # 显示更多特征（最多10个）
        standardization_after = {}
        # 使用处理后的特征名称（可能因为独热编码等发生变化）
        processed_names = feature_names if isinstance(feature_names, list) else list(feature_names) if feature_names is not None else None
        
        # 如果processed_names是占位符名称（如feature_0），优先使用before_feature_names
        # 只有当processed_names是真实名称时才使用（比如经过独热编码后的名称）
        use_processed_names = True
        if processed_names and len(processed_names) > 0:
            # 检查是否是占位符名称
            first_name = str(processed_names[0]).lower()
            if first_name.startswith('feature_') or first_name.startswith('特征_'):
                # 如果before_feature_names存在且长度匹配，使用before_feature_names
                if before_feature_names and len(before_feature_names) >= len(processed_names):
                    processed_names = before_feature_names
                    use_processed_names = False
        
        for i in range(n_features_to_show):
            feature_col = X_processed[:, i] if isinstance(X_processed, np.ndarray) else X_processed.iloc[:, i]
            # 使用真实的特征名称
            if use_processed_names and processed_names and i < len(processed_names):
                feature_name = processed_names[i]
            elif before_feature_names and i < len(before_feature_names):
                feature_name = before_feature_names[i]
            else:
                feature_name = f'特征_{i+1}'
            standardization_after[feature_name] = {
                'mean': float(np.mean(feature_col)),
                'std': float(np.std(feature_col)),
                'min': float(np.min(feature_col)),
                'max': float(np.max(feature_col))
            }
        if 'preprocessing_info' in locals():
            if 'standardization_info' not in preprocessing_info:
                preprocessing_info['standardization_info'] = {}
            preprocessing_info['standardization_info']['after'] = standardization_after
            # 只在真正执行标准化时才添加步骤（避免与前面的判断步骤重复）
            # 检查是否已经有标准化步骤（可能是SVM或树模型的说明步骤）
            has_standardization_step = any(step.get('step') == '数据标准化' or step.get('step') == '标准化' for step in preprocessing_info.get('steps', []))
            if not has_standardization_step:
                preprocessing_info['steps'].append({
                    'step': '数据标准化',
                    'method': 'StandardScaler（均值0，方差1）',
                    'reason': '该算法对特征尺度敏感，标准化可提升性能'
                })
    
    # 设置预处理后的信息
    if hasattr(X_processed, 'shape'):
        preprocessing_info['after'] = {
            'n_samples': int(X_processed.shape[0]),
            'n_features': int(X_processed.shape[1]),
            'feature_names': feature_names if feature_names else [f'特征_{i+1}' for i in range(X_processed.shape[1])]
        }
    elif hasattr(X_processed, '__len__'):
        preprocessing_info['after'] = {
            'n_samples': len(X_processed),
            'n_features': len(X_processed[0]) if len(X_processed) > 0 else 0,
            'feature_names': feature_names if feature_names else None
        }
    
    # 创建模型实例（每次创建新实例，避免状态共享）
    send_progress(20, '正在初始化模型...', step=2, total_steps=7, details=f'加载 {model_name} 模型')
    model_info = model_registry.get_model_info(model_name)
    if not model_info:
        raise ValueError(f"未知模型: {model_name}")
    # 创建新实例
    model = model_info['class']()
    
    # ========== 改进：如果是随机/分层分割，先分割数据，超参数调优只在训练集上进行 ==========
    X_tuning = X_processed  # 用于超参数调优的数据
    y_tuning = y  # 用于超参数调优的标签
    X_train = None
    X_test = None
    y_train = None
    y_test = None
    
    # 如果选择随机/分层分割，提前分割数据
    if split_method in ["random", "stratified"]:
        if task_type == 'classification':
            # 统计各类别样本数，处理极端不平衡或极少样本类别
            try:
                values, counts = np.unique(y, return_counts=True)
                min_count = int(counts.min()) if counts.size > 0 else 0
            except Exception:
                min_count = 0
            
            use_stratify = (split_method == "stratified")
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
        else:
            # 回归任务
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y, test_size=test_size, random_state=42
            )
        
        # 超参数调优只在训练集上进行
        X_tuning = X_train
        y_tuning = y_train

    # 超参数调优 - 智能选择策略（使用训练配置中的折数，智能降级）
    if hyperparams and hyperparams.get('enable_tuning', False):
        send_progress(25, '正在进行超参数调优...')
        param_grid = hyperparams.get('param_grid', {})
        search_type = hyperparams.get('search_type', 'grid')
        
        # ========== 改进：使用训练配置中的折数，但智能降级以避免时间过长 ==========
        if split_method in ['kfold', 'stratified_kfold']:
            # 如果用户选择了交叉验证模式，使用用户选择的折数，但智能降级
            if cv_folds == 'leave_one_out':
                # 留一法对于超参数调优太慢，降级到5折
                tuning_cv = 5
                send_progress(27, f'检测到留一法交叉验证，超参数调优使用{tuning_cv}折以避免时间过长...')
            else:
                try:
                    user_cv = int(cv_folds)
                    # 如果用户选择了10折，降级到5折；3-5折保持不变
                    if user_cv >= 10:
                        tuning_cv = 5
                        send_progress(27, f'用户选择{user_cv}折交叉验证，超参数调优使用{tuning_cv}折以避免时间过长...')
                    elif user_cv >= 5:
                        tuning_cv = user_cv  # 5折保持不变
                        send_progress(27, f'超参数调优使用用户选择的{user_cv}折交叉验证...')
                    else:
                        tuning_cv = max(3, user_cv)  # 至少3折
                        send_progress(27, f'超参数调优使用用户选择的{tuning_cv}折交叉验证...')
                except (ValueError, TypeError):
                    # 如果无法解析，使用默认3折
                    tuning_cv = 3
                    send_progress(27, f'使用默认{tuning_cv}折进行超参数调优...')
        else:
            # 普通分割模式，在训练集上使用5折交叉验证进行超参数调优
            tuning_cv = 5
            send_progress(27, f'普通分割模式，超参数调优在训练集上使用{tuning_cv}折交叉验证...')
        
        # 获取任务类型用于超参数调优
        task_type_for_tuning = get_task_type_extended(model_name)
        tuning_result = hyperparameter_tuning(model, model_name, X_tuning, y_tuning, param_grid, cv=tuning_cv, search_type=search_type, task_type=task_type_for_tuning)
        
        # 检查是否有错误
        if tuning_result and 'error' in tuning_result:
            # 超参数调优失败，记录详细错误信息
            error_msg = tuning_result.get('error', '超参数调优失败，使用默认参数')
            results['hyperparameter_tuning'] = {
                'error': error_msg,
                'search_type': search_type,
                'cv_folds': tuning_cv
            }
            results['warnings'].append(f'超参数调优失败: {error_msg}。使用默认参数继续训练。')
            send_progress(35, f'超参数调优失败: {error_msg}')
        elif tuning_result and 'best_params' in tuning_result:
            # 超参数调优成功
            model = tuning_result['best_estimator']
            results['hyperparameter_tuning'] = {
                'best_params': tuning_result['best_params'],
                'best_score': tuning_result['best_score'],
                'default_score': tuning_result.get('default_score', None),
                'improvement': tuning_result.get('improvement', 0.0),
                'improvement_percentage': tuning_result.get('improvement_percentage', 0.0),
                'search_type': search_type,
                'cv_folds': tuning_cv,
                'strategy': '智能调优策略',
                'search_history': tuning_result.get('search_history', []),
                'top_results': tuning_result.get('top_results', []),
                'heatmap_data': tuning_result.get('heatmap_data', None),
                'param_names': tuning_result.get('param_names', []),
                'total_combinations': tuning_result.get('total_combinations', 0)
            }
            send_progress(35, f'超参数调优完成，最佳参数: {tuning_result["best_params"]}')
        else:
            # 未知错误
            error_msg = '超参数调优失败：未知错误，使用默认参数继续训练'
            results['hyperparameter_tuning'] = {
                'error': error_msg,
                'search_type': search_type,
                'cv_folds': tuning_cv
            }
            results['warnings'].append(error_msg)
            send_progress(35, error_msg)

    # task_type已在上面验证阶段获取（2782行），这里直接使用
    # task_type = get_model_type_extended(model_name)

    if task_type == 'classification':
        # 如果还未分割数据（交叉验证模式或数据分割在超参数调优之后），现在进行分割
        if X_train is None or X_test is None:
            # 统计各类别样本数，处理极端不平衡或极少样本类别
            try:
                # y 可能是 np.ndarray 或 pandas Series
                values, counts = np.unique(y, return_counts=True)
                min_count = int(counts.min()) if counts.size > 0 else 0
            except Exception:
                min_count = 0
            
            if split_method in ["random", "stratified"]:
                send_progress(30, '正在分割数据集...', step=3, total_steps=7, details='划分训练集和测试集')
                # 如果任一类别样本过少，不使用 stratify，避免 "least populated class" 错误
                use_stratify = (split_method == "stratified")
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
            else:
                # 交叉验证模式：X_train和X_test会在后面设置
                X_train = X_processed
                X_test = None
                y_train = y
                y_test = None
        
        send_progress(50, '正在训练模型...', step=4, total_steps=7, details=f'训练 {model_name} 模型')
        model.fit(X_train, y_train)
            
        # 提取训练历史（如果模型支持）
        if hasattr(model, 'get_training_history'):
            try:
                training_history = model.get_training_history()
                if training_history:
                    results['training_history'] = training_history
                    print(f"✅ 成功提取训练历史: {list(training_history.keys())}")
                else:
                    print("⚠️ 训练历史为空")
            except Exception as e:
                print(f"❌ 提取训练历史失败: {str(e)}")
                results['warnings'].append(f'提取训练历史失败: {str(e)}')
        else:
            print(f"ℹ️ 模型 {model_name} 不支持训练历史提取")
        
        send_progress(70, '正在预测并计算指标...', step=5, total_steps=7, details='模型预测和评估')
        # 同时预测训练集和测试集
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        send_progress(80, '正在计算评估指标...', step=6, total_steps=7, details='计算模型性能指标')
        # 计算测试集指标
        accuracy_test = accuracy_score(y_test, y_pred_test)
        f1_test = f1_score(y_test, y_pred_test, average='weighted')
        precision_test = precision_score(y_test, y_pred_test, average='weighted')
        recall_test = recall_score(y_test, y_pred_test, average='weighted')
        
        # 计算训练集指标
        accuracy_train = accuracy_score(y_train, y_pred_train)
        f1_train = f1_score(y_train, y_pred_train, average='weighted')
        precision_train = precision_score(y_train, y_pred_train, average='weighted')
        recall_train = recall_score(y_train, y_pred_train, average='weighted')
        
        # 存储测试集指标（保持向后兼容）
        results['accuracy'] = round(accuracy_test, 4)
        results['f1_score'] = round(f1_test, 4)
        results['precision'] = round(precision_test, 4)
        results['recall'] = round(recall_test, 4)
        
        # 存储训练集和测试集的详细指标
        results['train_metrics'] = {
            'accuracy': round(accuracy_train, 4),
            'f1_score': round(f1_train, 4),
            'precision': round(precision_train, 4),
            'recall': round(recall_train, 4)
        }
        results['test_metrics'] = {
            'accuracy': round(accuracy_test, 4),
            'f1_score': round(f1_test, 4),
            'precision': round(precision_test, 4),
            'recall': round(recall_test, 4)
        }
        
        # 计算过拟合检测（训练集和测试集的性能差异）
        overfitting_indicators = {
            'accuracy_gap': round(accuracy_train - accuracy_test, 4),
            'f1_gap': round(f1_train - f1_test, 4),
            'precision_gap': round(precision_train - precision_test, 4),
            'recall_gap': round(recall_train - recall_test, 4)
        }
        results['overfitting_indicators'] = overfitting_indicators
        
        # 使用测试集指标作为主要指标（向后兼容）
        accuracy = accuracy_test
        f1 = f1_test
        precision = precision_test
        recall = recall_test
        y_pred = y_pred_test
        
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
        elif metric == 'loss':
            # 计算损失值（对于分类任务，使用交叉熵损失）
            from sklearn.metrics import log_loss
            try:
                # 获取预测概率
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)
                    loss = log_loss(y_test, y_proba)
                else:
                    # 如果没有predict_proba，使用0-1损失
                    loss = 1 - accuracy
                score = loss
                results['loss'] = round(loss, 6)
            except:
                # 如果计算损失失败，使用0-1损失
                loss = 1 - accuracy
                score = loss
                results['loss'] = round(loss, 6)
        else:
            score = accuracy
        
        if split_method == 'random':
            results['split_method'] = '随机分割'
        elif split_method == 'stratified':
            results['split_method'] = '分层分割'
        else:
            results['split_method'] = '随机分割'
        results['test_size'] = test_size
        send_progress(92, '正在生成详细报告...', step=7, total_steps=7, details='生成实验报告和可视化')
        report = classification_report(y_test, y_pred, output_dict=True)
        results['classification_report'] = report
        # 深度学习模型不生成混淆矩阵
        if model_name not in ['深度神经网络(DNN)', '卷积神经网络(CNN)', '循环神经网络(RNN)', '长短期记忆网络(LSTM)']:
            cm = confusion_matrix(y_test, y_pred)
            results['confusion_matrix'] = cm.tolist()
        
        # 生成ROC/PR曲线数据（仅对二分类任务，或者多分类任务使用One-vs-Rest策略）
        try:
            send_progress(93, '正在生成ROC/PR曲线数据...', step=7, total_steps=7, details='计算ROC和PR曲线')
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
                n_classes = len(np.unique(y_test))
                
                if n_classes == 2:
                    # 二分类：直接计算ROC和PR曲线
                    # 将y_test转换为0/1格式（如果模型使用了类别编码）
                    unique_test = np.unique(y_test)
                    if len(unique_test) == 2:
                        # 如果y_test不是0/1格式，转换为0/1格式
                        y_test_binary = np.where(y_test == unique_test[0], 0, 1)
                    else:
                        y_test_binary = y_test
                    
                    fpr, tpr, roc_thresholds = roc_curve(y_test_binary, y_proba[:, 1])
                    precision, recall, pr_thresholds = precision_recall_curve(y_test_binary, y_proba[:, 1])
                    roc_auc = auc(fpr, tpr)
                    pr_auc = auc(recall, precision)
                    
                    results['roc_curve'] = {
                        'fpr': fpr.tolist(),
                        'tpr': tpr.tolist(),
                        'thresholds': roc_thresholds.tolist(),
                        'auc': float(roc_auc)
                    }
                    results['pr_curve'] = {
                        'precision': precision.tolist(),
                        'recall': recall.tolist(),
                        'thresholds': pr_thresholds.tolist(),
                        'auc': float(pr_auc)
                    }
                    
                    # 保存用于动态阈值调整的数据
                    results['threshold_data'] = {
                        'y_test': y_test.tolist(),
                        'y_test_binary': y_test_binary.tolist(),
                        'y_proba': y_proba[:, 1].tolist(),  # 正类概率
                        'unique_classes': unique_test.tolist(),
                        'target_names': target_names if target_names else ['类别0', '类别1']
                    }
                else:
                    # 多分类：使用One-vs-Rest策略
                    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
                    n_classes = y_test_bin.shape[1]
                    
                    # 计算每个类别的ROC和PR曲线
                    fpr_dict = {}
                    tpr_dict = {}
                    precision_dict = {}
                    recall_dict = {}
                    roc_auc_dict = {}
                    pr_auc_dict = {}
                    
                    for i in range(n_classes):
                        if n_classes == 1:
                            # 如果只有一个类别（理论上不应该发生）
                            continue
                        
                        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_proba[:, i])
                        roc_auc = auc(fpr, tpr) if len(fpr) > 1 else 0.0
                        pr_auc = auc(recall, precision) if len(recall) > 1 else 0.0
                        
                        class_name = target_names[i] if target_names and i < len(target_names) else f'类别{i}'
                        fpr_dict[class_name] = fpr.tolist()
                        tpr_dict[class_name] = tpr.tolist()
                        precision_dict[class_name] = precision.tolist()
                        recall_dict[class_name] = recall.tolist()
                        roc_auc_dict[class_name] = float(roc_auc)
                        pr_auc_dict[class_name] = float(pr_auc)
                    
                    results['roc_curve'] = {
                        'fpr': fpr_dict,
                        'tpr': tpr_dict,
                        'auc': roc_auc_dict,
                        'multiclass': True
                    }
                    results['pr_curve'] = {
                        'precision': precision_dict,
                        'recall': recall_dict,
                        'auc': pr_auc_dict,
                        'multiclass': True
                    }
        except Exception as e:
            results['warnings'].append(f'ROC/PR曲线计算失败: {str(e)}')
        
        # 逻辑回归系数可视化（仅对逻辑回归模型）
        try:
            # 检查是否是逻辑回归模型（支持注册名称和中文名称）
            is_logistic_regression = (
                model_name == '逻辑回归' or 
                model_name == 'logistic_regression' or
                (hasattr(model, '__class__') and 'LogisticRegression' in model.__class__.__name__)
            )
            if is_logistic_regression and (hasattr(model, 'weights') or hasattr(model, 'estimators_')):
                send_progress(94, '正在提取逻辑回归系数...', step=7, total_steps=7, details='提取特征系数')
                coefficients_data = {}
                
                if hasattr(model, 'multiclass_') and model.multiclass_:
                    # 多分类：提取每个OvR分类器的系数
                    if hasattr(model, 'estimators_') and model.estimators_:
                        for i, estimator in enumerate(model.estimators_):
                            class_label = model.classes_[i] if hasattr(model, 'classes_') else f'类别{i}'
                            class_name = target_names[i] if target_names and i < len(target_names) else str(class_label)
                            
                            if 'weights' in estimator:
                                coefficients_data[class_name] = {
                                    'weights': estimator['weights'].tolist() if hasattr(estimator['weights'], 'tolist') else list(estimator['weights']),
                                    'bias': float(estimator['bias']) if 'bias' in estimator else 0.0
                                }
                else:
                    # 二分类：提取单个分类器的系数
                    if model.weights is not None:
                        class_name = target_names[1] if target_names and len(target_names) >= 2 else '正类'
                        coefficients_data[class_name] = {
                            'weights': model.weights.tolist() if hasattr(model.weights, 'tolist') else list(model.weights),
                            'bias': float(model.bias) if hasattr(model, 'bias') else 0.0
                        }
                
                if coefficients_data:
                    # 使用特征名称（如果有）
                    feature_names_list = feature_names if feature_names else [f'特征_{i+1}' for i in range(len(list(coefficients_data.values())[0]['weights']))]
                    results['logistic_coefficients'] = {
                        'coefficients': coefficients_data,
                        'feature_names': feature_names_list
                    }
        except Exception as e:
            results['warnings'].append(f'逻辑回归系数提取失败: {str(e)}')
        
        # 朴素贝叶斯可视化（仅对朴素贝叶斯模型）
        try:
            is_naive_bayes = (
                model_name == '朴素贝叶斯' or 
                model_name == '朴素贝叶斯分类器' or
                model_name == 'naive_bayes' or
                (hasattr(model, '__class__') and 'NaiveBayes' in model.__class__.__name__) or
                (hasattr(model, '__class__') and 'NaiveBayesModel' in model.__class__.__name__)
            )
            print(f"🔍 检查朴素贝叶斯模型: model_name={model_name}, is_naive_bayes={is_naive_bayes}, model_class={model.__class__.__name__}")
            if is_naive_bayes:
                # 检查模型是否有必要的属性
                has_class_prior = hasattr(model, 'class_prior_')
                has_theta = hasattr(model, 'theta_')
                has_sigma = hasattr(model, 'sigma_')
                print(f"🔍 朴素贝叶斯属性检查: class_prior_={has_class_prior}, theta_={has_theta}, sigma_={has_sigma}")
                if not (has_class_prior and has_theta and has_sigma):
                    results['warnings'].append('朴素贝叶斯模型缺少必要属性，跳过可视化数据生成')
                    print("⚠️ 朴素贝叶斯模型缺少必要属性")
                else:
                    send_progress(94, '正在生成朴素贝叶斯可视化数据...', step=7, total_steps=7, details='计算概率分布数据')
                    print("✅ 开始生成朴素贝叶斯可视化数据")
                    
                    # 1. 先验概率可视化
                    class_prior_data = {}
                    if model.classes_ is not None and model.class_prior_ is not None:
                        for i, class_label in enumerate(model.classes_):
                            class_name = target_names[i] if target_names and i < len(target_names) else str(class_label)
                            class_prior_data[class_name] = {
                                'prior_probability': float(model.class_prior_[i]),
                                'sample_count': int(model.class_count_[i]) if hasattr(model, 'class_count_') and model.class_count_ is not None else 0
                            }
                    
                    # 2. 特征概率分布可视化（每个类别的特征均值、方差）
                    feature_distributions = {}
                    if model.classes_ is not None and model.theta_ is not None and model.sigma_ is not None:
                        feature_names_list = feature_names if feature_names else [f'特征_{i+1}' for i in range(model.theta_.shape[1])]
                        
                        for i, class_label in enumerate(model.classes_):
                            class_name = target_names[i] if target_names and i < len(target_names) else str(class_label)
                            feature_distributions[class_name] = {
                                'means': model.theta_[i].tolist() if hasattr(model.theta_[i], 'tolist') else list(model.theta_[i]),
                                'variances': model.sigma_[i].tolist() if hasattr(model.sigma_[i], 'tolist') else list(model.sigma_[i]),
                                'feature_names': feature_names_list
                            }
                    
                    # 3. 后验概率分布可视化（测试集前20个样本的后验概率）
                    posterior_probabilities = None
                    if hasattr(model, 'predict_proba'):
                        try:
                            y_proba_test = model.predict_proba(X_test)
                            # 选择前20个样本（或所有样本如果少于20个）
                            n_samples_show = min(20, len(y_proba_test))
                            posterior_probabilities = {
                                'probabilities': y_proba_test[:n_samples_show].tolist(),
                                'predictions': y_pred_test[:n_samples_show].tolist() if hasattr(y_pred_test, 'tolist') else list(y_pred_test[:n_samples_show]),
                                'true_labels': y_test[:n_samples_show].tolist() if hasattr(y_test, 'tolist') else list(y_test[:n_samples_show]),
                                'class_names': [target_names[i] if target_names and i < len(target_names) else str(c) 
                                               for i, c in enumerate(model.classes_)] if model.classes_ is not None else [],
                                'n_samples_shown': n_samples_show,
                                'total_samples': len(y_proba_test)
                            }
                        except Exception as e:
                            results['warnings'].append(f'后验概率提取失败: {str(e)}')
                    
                    # 4. 基于概率的特征重要性（通过比较不同类别间的特征差异）
                    feature_importance_proba = None
                    if model.theta_ is not None and model.sigma_ is not None and len(model.classes_) > 1:
                        try:
                            # 计算每个特征的类别间差异（使用均值差异和方差的组合）
                            n_features = model.theta_.shape[1]
                            importance_scores = np.zeros(n_features)
                            
                            for j in range(n_features):
                                # 计算该特征在所有类别间的均值差异
                                means = model.theta_[:, j]
                                variances = model.sigma_[:, j]
                                
                                # 特征重要性 = 类别间均值差异 / 平均方差
                                mean_diff = np.max(means) - np.min(means)
                                avg_var = np.mean(variances)
                                if avg_var > 0:
                                    importance_scores[j] = mean_diff / np.sqrt(avg_var)
                                else:
                                    importance_scores[j] = mean_diff
                            
                            # 归一化到[0, 1]
                            if importance_scores.max() > 0:
                                importance_scores = importance_scores / importance_scores.max()
                            
                            feature_names_list = feature_names if feature_names else [f'特征_{i+1}' for i in range(n_features)]
                            feature_importance_proba = {
                                'importance': importance_scores.tolist(),
                                'feature_names': feature_names_list
                            }
                        except Exception as e:
                            results['warnings'].append(f'特征重要性计算失败: {str(e)}')
                    
                    # 组装所有可视化数据（即使部分数据为空也创建可视化对象）
                    results['naive_bayes_visualization'] = {
                        'class_prior': class_prior_data if class_prior_data else None,
                        'feature_distributions': feature_distributions if feature_distributions else None,
                        'posterior_probabilities': posterior_probabilities,
                        'feature_importance': feature_importance_proba
                    }
                    print(f"✅ 朴素贝叶斯可视化数据生成成功: {list(results['naive_bayes_visualization'].keys())}")
                    print(f"   先验概率: {bool(class_prior_data)}, 特征分布: {bool(feature_distributions)}, 后验概率: {bool(posterior_probabilities)}, 特征重要性: {bool(feature_importance_proba)}")
        except Exception as e:
            print(f"❌ 朴素贝叶斯可视化数据生成失败: {str(e)}")
            import traceback
            traceback.print_exc()
            results['warnings'].append(f'朴素贝叶斯可视化数据生成失败: {str(e)}')
        
        # SVM决策边界可视化（仅对SVM模型，且特征数为2时）
        try:
            is_svm = (
                model_name == '支持向量机(SVM)' or 
                model_name == 'svm' or
                (hasattr(model, '__class__') and 'SVM' in model.__class__.__name__)
            )
            
            if is_svm and hasattr(model, 'X_train') and model.X_train is not None:
                n_features = model.X_train.shape[1] if hasattr(model.X_train, 'shape') else len(model.X_train[0]) if model.X_train else 0
                
                if n_features == 2:
                    send_progress(95, '正在生成SVM决策边界数据...', step=7, total_steps=7, details='计算决策边界网格')
                    
                    # 获取训练数据的范围（标准化后的数据）
                    X_train_scaled = model.X_train
                    x_min, x_max = X_train_scaled[:, 0].min() - 0.5, X_train_scaled[:, 0].max() + 0.5
                    y_min, y_max = X_train_scaled[:, 1].min() - 0.5, X_train_scaled[:, 1].max() + 0.5
                    
                    # 创建网格
                    h = 0.1  # 网格步长
                    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
                    grid_points = np.c_[xx.ravel(), yy.ravel()]
                    
                    # 使用模型的决策函数预测网格点
                    if hasattr(model, '_decision_function'):
                        Z = model._decision_function(grid_points)
                        
                        # 根据任务类型处理Z
                        if model.multiclass_ if hasattr(model, 'multiclass_') else False:
                            # 多分类：选择最大决策函数值的类别
                            Z_class = np.argmax(Z, axis=1)
                            Z = Z_class.reshape(xx.shape)
                        else:
                            # 二分类：直接使用决策函数值
                            Z = Z.reshape(xx.shape)
                    else:
                        # 如果没有决策函数，使用predict
                        predictions = model.predict(grid_points)
                        # 转换为类别索引
                        if hasattr(model, 'classes_'):
                            Z = np.array([np.where(model.classes_ == p)[0][0] for p in predictions])
                        else:
                            Z = predictions
                        Z = Z.reshape(xx.shape)
                    
                    # 获取支持向量（原始特征空间的坐标）
                    support_vectors_data = []
                    if model.multiclass_ if hasattr(model, 'multiclass_') else False:
                        # 多分类：收集所有OvR分类器的支持向量
                        for i, estimator in enumerate(model.estimators_):
                            sv = estimator['support_vectors']
                            # 反标准化到原始特征空间（如果需要显示原始坐标）
                            if hasattr(model, 'X_mean_') and hasattr(model, 'X_std_'):
                                sv_original = sv * model.X_std_ + model.X_mean_
                            else:
                                sv_original = sv
                            support_vectors_data.append({
                                'vectors': sv_original.tolist(),
                                'class_index': i
                            })
                    else:
                        # 二分类
                        if model.support_vectors_ is not None and len(model.support_vectors_) > 0:
                            sv = model.support_vectors_
                            # 反标准化到原始特征空间
                            if hasattr(model, 'X_mean_') and hasattr(model, 'X_std_'):
                                sv_original = sv * model.X_std_ + model.X_mean_
                            else:
                                sv_original = sv
                            support_vectors_data.append({
                                'vectors': sv_original.tolist(),
                                'class_index': 0
                            })
                    
                    # 反标准化网格坐标到原始特征空间（用于显示）
                    if hasattr(model, 'X_mean_') and hasattr(model, 'X_std_'):
                        xx_original = xx * model.X_std_[0] + model.X_mean_[0]
                        yy_original = yy * model.X_std_[1] + model.X_mean_[1]
                    else:
                        xx_original = xx
                        yy_original = yy
                    
                    # 准备训练数据（原始特征空间的坐标，用于显示）
                    X_train_original = X_train_scaled
                    if hasattr(model, 'X_mean_') and hasattr(model, 'X_std_'):
                        X_train_original = X_train_scaled * model.X_std_ + model.X_mean_
                    
                    results['svm_decision_boundary'] = {
                        'xx': xx_original.tolist(),
                        'yy': yy_original.tolist(),
                        'Z': Z.tolist(),
                        'X_train': X_train_original.tolist(),
                        'y_train': model.y_train.tolist() if hasattr(model, 'y_train') else [],
                        'support_vectors': support_vectors_data,
                        'feature_names': feature_names[:2] if feature_names and len(feature_names) >= 2 else ['特征1', '特征2'],
                        'target_names': target_names if target_names else ['类别0', '类别1'],
                        'kernel': model.kernel if hasattr(model, 'kernel') else 'rbf'
                    }
        except Exception as e:
            results['warnings'].append(f'SVM决策边界生成失败: {str(e)}')
        
        # KNN K值选择曲线（仅对KNN模型）
        try:
            is_knn = (
                model_name == 'K近邻(KNN)' or 
                model_name == 'knn' or
                (hasattr(model, '__class__') and 'KNN' in model.__class__.__name__)
            )
            
            if is_knn:
                send_progress(96, '正在生成KNN K值选择曲线数据...', step=7, total_steps=7, details='测试不同K值性能')
                
                # 测试不同的K值（奇数，避免平票）
                max_k = min(21, len(X_train) // 2 + 1)
                k_values = list(range(1, max_k, 2))  # 1, 3, 5, ..., 最多19（奇数）
                if len(k_values) == 0:
                    k_values = [1]  # 至少测试K=1
                k_performance = []
                
                for k in k_values:
                    try:
                        # 创建临时KNN模型
                        temp_knn = MyKNN(k=k, distance_metric=model.distance_metric if hasattr(model, 'distance_metric') else 'euclidean')
                        
                        # 使用简单的训练-验证集划分评估（避免sklearn依赖）
                        # 由于KNN无需训练，直接使用部分数据作为验证集
                        n_samples = len(X_train)
                        val_size = max(5, n_samples // 5)  # 验证集大小
                        
                        if n_samples > val_size * 2:
                            # 随机选择验证集
                            np.random.seed(42)
                            val_indices = np.random.choice(n_samples, val_size, replace=False)
                            train_indices = np.setdiff1d(range(n_samples), val_indices)
                            
                            X_val = X_train[val_indices]
                            y_val = y_train[val_indices]
                            X_train_subset = X_train[train_indices]
                            y_train_subset = y_train[train_indices]
                            
                            # 重新训练（仅存储数据）
                            temp_knn.fit(X_train_subset, y_train_subset)
                            
                            # 预测验证集
                            y_pred_val = temp_knn.predict(X_val)
                            
                            if task_type == 'classification':
                                # 计算准确率
                                accuracy = float(np.mean(y_pred_val == y_val))
                                # 使用训练集的标准差作为稳定性指标（简化）
                                std = 0.01  # 占位值，实际可以多次随机划分计算
                                k_performance.append({
                                    'k': k,
                                    'accuracy': accuracy,
                                    'std': std
                                })
                            else:
                                # 计算RMSE
                                mse = float(np.mean((y_pred_val - y_val) ** 2))
                                rmse = float(np.sqrt(mse))
                                std = 0.01  # 占位值
                                k_performance.append({
                                    'k': k,
                                    'rmse': rmse,
                                    'std': std
                                })
                    except Exception as e:
                        # 如果某个K值失败，跳过
                        continue
                
                if k_performance:
                    # 找到最佳K值（分类任务选准确率最高，回归任务选RMSE最低）
                    if task_type == 'classification':
                        best_item = max(k_performance, key=lambda x: x.get('accuracy', 0))
                    else:
                        best_item = min(k_performance, key=lambda x: x.get('rmse', float('inf')))
                    
                    results['knn_k_selection'] = {
                        'k_values': [item['k'] for item in k_performance],
                        'performance': k_performance,
                        'task_type': task_type,
                        'best_k': best_item['k'] if best_item else None
                    }
        except Exception as e:
            results['warnings'].append(f'KNN K值选择曲线生成失败: {str(e)}')
        
        # 同时生成训练集的分类报告（用于对比）
        try:
            report_train = classification_report(y_train, y_pred_train, output_dict=True)
            results['train_classification_report'] = report_train
        except:
            pass
        
        # 如果是决策树模型，提取树结构用于可视化
        is_decision_tree = (
            model_name == '决策树' or 
            model_name == 'decision_tree' or
            (hasattr(model, '__class__') and 'DecisionTree' in model.__class__.__name__)
        )
        if is_decision_tree and hasattr(model, 'tree') and model.tree is not None:
            try:
                send_progress(91, '正在提取决策树结构...', step=7, total_steps=7, details='提取树结构用于可视化')
                # 为树节点添加样本统计信息
                if hasattr(model, 'enrich_tree_with_samples'):
                    model.tree = model.enrich_tree_with_samples(model.tree, X_train, y_train, feature_names)
                # 提取树结构
                if hasattr(model, 'get_tree_structure'):
                    tree_structure = model.get_tree_structure(
                        tree=model.tree,
                        feature_names=feature_names,
                        target_names=target_names,
                        X=X_train,
                        y=y_train
                    )
                    results['tree_structure'] = tree_structure
            except Exception as e:
                results['warnings'].append(f'决策树结构提取失败: {str(e)}')
        
        # 随机森林/GBDT学习曲线（树数量对性能的影响）
        try:
            is_random_forest = (
                model_name == '随机森林' or 
                model_name == 'random_forest' or
                (hasattr(model, '__class__') and 'RandomForest' in model.__class__.__name__)
            )
            is_gbdt = (
                model_name == '梯度提升树(GBDT)' or 
                model_name == 'GBDT' or
                model_name == 'gbdt' or
                (hasattr(model, '__class__') and 'GBDT' in model.__class__.__name__)
            )
            
            if (is_random_forest or is_gbdt) and split_method in ['random', 'stratified']:
                send_progress(94, '正在生成学习曲线数据...', step=7, total_steps=7, details='测试不同树数量的性能')
                
                # 获取当前模型的超参数
                current_n_estimators = getattr(model, 'n_estimators', 100)
                max_depth = getattr(model, 'max_depth', None)
                random_state = getattr(model, 'random_state', 42)
                
                # 定义要测试的树数量列表
                # 为了节省计算时间，选择代表性的n_estimators值
                if current_n_estimators <= 50:
                    n_estimators_list = [1, 5, 10, 20, 30, 40, current_n_estimators]
                elif current_n_estimators <= 100:
                    n_estimators_list = [1, 5, 10, 20, 50, 75, current_n_estimators]
                elif current_n_estimators <= 200:
                    n_estimators_list = [1, 10, 25, 50, 100, 150, current_n_estimators]
                else:
                    n_estimators_list = [1, 10, 25, 50, 100, 200, current_n_estimators]
                
                # 去重并排序
                n_estimators_list = sorted(list(set(n_estimators_list)))
                
                train_scores = []
                test_scores = []
                valid_n_estimators = []
                
                for n_est in n_estimators_list:
                    try:
                        # 创建临时模型
                        if is_random_forest:
                            temp_model = RandomForestModel(
                                n_estimators=n_est,
                                max_depth=max_depth,
                                random_state=random_state
                            )
                        else:  # GBDT
                            learning_rate = getattr(model, 'learning_rate', 0.1)
                            temp_model = GBDTModel(
                                n_estimators=n_est,
                                learning_rate=learning_rate,
                                max_depth=max_depth if max_depth else 3,
                                random_state=random_state
                            )
                        
                        # 训练临时模型
                        temp_model.fit(X_train, y_train)
                        
                        # 预测训练集和测试集
                        y_pred_train_temp = temp_model.predict(X_train)
                        y_pred_test_temp = temp_model.predict(X_test)
                        
                        # 计算准确率
                        train_acc = accuracy_score(y_train, y_pred_train_temp)
                        test_acc = accuracy_score(y_test, y_pred_test_temp)
                        
                        train_scores.append(float(train_acc))
                        test_scores.append(float(test_acc))
                        valid_n_estimators.append(n_est)
                    except Exception as e:
                        # 如果某个n_estimators失败，跳过
                        continue
                
                if len(valid_n_estimators) > 1 and len(train_scores) > 1 and len(test_scores) > 1:
                    results['learning_curve'] = {
                        'n_estimators': valid_n_estimators,
                        'train_scores': train_scores,
                        'test_scores': test_scores,
                        'current_n_estimators': current_n_estimators,
                        'task_type': 'classification',
                        'metric': 'accuracy'
                    }
        except Exception as e:
            results['warnings'].append(f'学习曲线生成失败: {str(e)}')
    
    elif split_method in ['kfold', 'stratified_kfold']:
            # 使用用户指定的折数
            if cv_folds == 'leave_one_out':
                from sklearn.model_selection import LeaveOneOut
                cv = LeaveOneOut()
                n_splits = len(y)
                send_progress(30, f'正在进行留一法交叉验证...', step=3, total_steps=7, details='初始化留一法交叉验证')
                # 留一法使用准确率作为主要指标
                cv_scores = cross_val_score(model, X_processed, y, cv=cv, scoring='accuracy')
            else:
                desired_splits = int(cv_folds)
                n_splits = max(2, min(desired_splits, min_count)) if min_count > 0 else 2
                send_progress(30, f'正在进行{n_splits}折交叉验证...', step=3, total_steps=7, details=f'初始化{n_splits}折交叉验证')
                if metric == 'f1':
                    scoring = 'f1_weighted'
                elif metric == 'loss':
                    scoring = 'neg_log_loss'  # 使用负对数损失，因为sklearn期望越高越好
                else:
                    scoring = 'accuracy'
                try:
                    if split_method == 'stratified_kfold':
                        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                        send_progress(40, '正在执行分层交叉验证...', step=4, total_steps=7, details='执行分层K折交叉验证')
                    else:
                        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                        send_progress(40, '正在执行K折交叉验证...', step=4, total_steps=7, details='执行K折交叉验证')
                    cv_scores = cross_val_score(model, X_processed, y, cv=cv, scoring=scoring)
                    if n_splits < desired_splits:
                        results['warnings'].append(f'最小类别样本仅 {min_count}，交叉验证折数降为 {n_splits}。')
                except ValueError:
                    # 回退到普通 KFold
                    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                    send_progress(40, '正在执行普通交叉验证...', step=4, total_steps=7, details='执行普通交叉验证')
                    cv_scores = cross_val_score(model, X_processed, y, cv=cv, scoring=scoring)
                    results['warnings'].append('分层交叉验证不可用，已回退至非分层 KFold。')
            send_progress(50, '正在训练和验证模型...', step=5, total_steps=7, details=f'执行{n_splits}折交叉验证训练')
            send_progress(70, '正在计算交叉验证结果...', step=6, total_steps=7, details='计算交叉验证指标')
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
            elif metric == 'loss':
                # 处理损失值交叉验证
                neg_log_loss_scores = cross_val_score(model, X_processed, y, cv=cv, scoring='neg_log_loss')
                loss_scores = -neg_log_loss_scores  # 转换为正数损失值
                results['loss'] = round(loss_scores.mean(), 6)
                accuracy_scores = cross_val_score(model, X_processed, y, cv=cv, scoring='accuracy')
                results['accuracy'] = round(accuracy_scores.mean(), 4)
                results['cv_scores'] = {
                    'values': [round(score, 6) for score in loss_scores],
                    'mean': round(loss_scores.mean(), 6),
                    'std': round(loss_scores.std(), 6)
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
            # 深度学习模型不生成混淆矩阵
            if model_name not in ['深度神经网络(DNN)', '卷积神经网络(CNN)', '循环神经网络(RNN)', '长短期记忆网络(LSTM)']:
                send_progress(85, '正在生成混淆矩阵...', step=7, total_steps=7, details='生成混淆矩阵')
                from sklearn.model_selection import cross_val_predict
                y_pred_cv = cross_val_predict(model, X_processed, y, cv=cv)
                cm = confusion_matrix(y, y_pred_cv)
                results['confusion_matrix'] = cm.tolist()

    elif task_type == 'regression':
        send_progress(30, '正在处理回归任务...', step=3, total_steps=7, details='初始化回归任务')
        # y 必须为数值
        if not isinstance(y, (np.ndarray, list, pd.Series)):
            raise ValueError('回归任务需要数值型标签')
        if split_method in ['random', 'stratified']:
            send_progress(40, '正在分割数据集...', step=4, total_steps=7, details='划分训练集和测试集')
            # 回归任务不支持分层抽样，强制使用随机分割
            if split_method == 'stratified':
                results['warnings'].append('回归任务不支持分层抽样，已自动切换为随机分割。')
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y, test_size=test_size, random_state=42, stratify=None
            )
            send_progress(60, '正在训练回归模型...', step=5, total_steps=7, details=f'训练 {model_name} 模型')
            model.fit(X_train, y_train)
            
            # 提取训练历史（如果模型支持）
            if hasattr(model, 'get_training_history'):
                try:
                    training_history = model.get_training_history()
                    results['training_history'] = training_history
                except Exception as e:
                    results['warnings'].append(f'提取训练历史失败: {str(e)}')
            
            send_progress(80, '正在预测并计算指标...', step=6, total_steps=7, details='模型预测和评估')
            # 同时预测训练集和测试集
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            send_progress(90, '正在计算回归指标...', step=7, total_steps=7, details='计算回归性能指标')
            # 计算测试集指标
            mse_test = mean_squared_error(y_test, y_pred_test)
            mae_test = mean_absolute_error(y_test, y_pred_test)
            rmse_test = np.sqrt(mse_test)
            r2_test = r2_score(y_test, y_pred_test)
            
            # 计算训练集指标
            mse_train = mean_squared_error(y_train, y_pred_train)
            mae_train = mean_absolute_error(y_train, y_pred_train)
            rmse_train = np.sqrt(mse_train)
            r2_train = r2_score(y_train, y_pred_train)
            
            # 存储测试集指标（保持向后兼容）
            results['mse'] = round(float(mse_test), 6)
            results['mae'] = round(float(mae_test), 6)
            results['rmse'] = round(float(rmse_test), 6)
            results['r2'] = round(float(r2_test), 6)
            
            # 存储训练集和测试集的详细指标
            results['train_metrics'] = {
                'mse': round(float(mse_train), 6),
                'mae': round(float(mae_train), 6),
                'rmse': round(float(rmse_train), 6),
                'r2': round(float(r2_train), 6)
            }
            results['test_metrics'] = {
                'mse': round(float(mse_test), 6),
                'mae': round(float(mae_test), 6),
                'rmse': round(float(rmse_test), 6),
                'r2': round(float(r2_test), 6)
            }
            
            # 计算过拟合检测（训练集和测试集的性能差异）
            overfitting_indicators = {
                'mse_gap': round(float(mse_train - mse_test), 6),  # 训练集MSE更小表示可能过拟合
                'mae_gap': round(float(mae_train - mae_test), 6),
                'rmse_gap': round(float(rmse_train - rmse_test), 6),
                'r2_gap': round(float(r2_train - r2_test), 6)  # 训练集R2更大表示可能过拟合
            }
            results['overfitting_indicators'] = overfitting_indicators
            
            # 使用测试集指标作为主要指标（向后兼容）
            mse = mse_test
            mae = mae_test
            rmse = rmse_test
            r2 = r2_test
            y_pred = y_pred_test
            results['split_method'] = '随机分割'
            results['test_size'] = test_size
            
            # 添加回归可视化数据
            try:
                send_progress(91, '正在准备回归可视化数据...', step=7, total_steps=7, details='准备可视化图表数据')
                results['regression_visualization'] = {
                    'y_true_test': y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test),
                    'y_pred_test': y_pred_test.tolist() if hasattr(y_pred_test, 'tolist') else list(y_pred_test),
                    'y_true_train': y_train.tolist() if hasattr(y_train, 'tolist') else list(y_train),
                    'y_pred_train': y_pred_train.tolist() if hasattr(y_pred_train, 'tolist') else list(y_pred_train),
                    'residuals_test': (y_test - y_pred_test).tolist() if hasattr(y_test - y_pred_test, 'tolist') else list(y_test - y_pred_test),
                    'residuals_train': (y_train - y_pred_train).tolist() if hasattr(y_train - y_pred_train, 'tolist') else list(y_train - y_pred_train),
                    'n_features': X_test.shape[1] if hasattr(X_test, 'shape') else len(X_test[0]) if X_test else 0
                }
                
                # 添加回归系数（如果模型有系数）
                if hasattr(model, 'coef_'):
                    if model.coef_ is not None and len(model.coef_) > 0:
                        results['regression_visualization']['coefficients'] = model.coef_.tolist() if hasattr(model.coef_, 'tolist') else list(model.coef_)
                        results['regression_visualization']['intercept'] = float(model.intercept_) if hasattr(model, 'intercept_') else 0.0
                        
                        # 使用特征名称（如果有）
                        if feature_names and len(feature_names) == len(model.coef_):
                            results['regression_visualization']['feature_names'] = feature_names
                        else:
                            results['regression_visualization']['feature_names'] = [f'特征_{i+1}' for i in range(len(model.coef_))]
            except Exception as e:
                results['warnings'].append(f'回归可视化数据准备失败: {str(e)}')
        else:
            # 使用用户指定的折数进行回归交叉验证
            if cv_folds == 'leave_one_out':
                from sklearn.model_selection import LeaveOneOut
                kf = LeaveOneOut()
                n_splits = len(y)
                send_progress(40, '正在进行留一法交叉验证...', step=4, total_steps=7, details='初始化留一法交叉验证')
            else:
                n_splits = int(cv_folds)
                kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                send_progress(40, f'正在进行{n_splits}折交叉验证...', step=4, total_steps=7, details=f'初始化{n_splits}折交叉验证')
            # 使用R2为主指标，同时返回MSE/MAE
            send_progress(50, '正在训练和验证回归模型...', step=5, total_steps=7, details=f'执行{n_splits}折交叉验证训练')
            send_progress(60, '正在计算R²分数...', step=6, total_steps=7, details='计算R²分数')
            r2_scores = cross_val_score(model, X_processed, y, cv=kf, scoring='r2')
            send_progress(65, '正在计算MSE和MAE...', step=6, total_steps=7, details='计算MSE和MAE指标')
            neg_mse_scores = cross_val_score(model, X_processed, y, cv=kf, scoring='neg_mean_squared_error')
            neg_mae_scores = cross_val_score(model, X_processed, y, cv=kf, scoring='neg_mean_absolute_error')
            mse_scores = -neg_mse_scores
            mae_scores = -neg_mae_scores
            results['r2'] = round(r2_scores.mean(), 6)
            results['mse'] = round(float(mse_scores.mean()), 6)
            results['mae'] = round(float(mae_scores.mean()), 6)
            results['rmse'] = round(float(np.sqrt(mse_scores.mean())), 6)
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
            send_progress(90, '正在生成回归报告...', step=7, total_steps=7, details='生成回归分析报告')

    else:  # unsupervised
        send_progress(30, '正在处理无监督任务...', step=3, total_steps=5, details='初始化无监督任务')
        # 无监督直接在全量数据上拟合
        # 检查是否为PCA（支持多种名称格式）
        is_pca = (
            model_name.startswith('主成分分析') or 
            model_name == 'pca' or 
            model_name == 'PCA' or
            model_name.lower() == 'pca'
        )
        if is_pca:
            send_progress(50, '正在执行PCA降维...', step=4, total_steps=5, details='执行主成分分析')
            model.fit(X_processed)
            send_progress(80, '正在分析降维结果...', step=5, total_steps=5, details='分析降维结果')
            explained = getattr(model, 'explained_variance_ratio_', None)
            if explained is not None:
                results['explained_variance_ratio'] = [round(float(x), 6) for x in explained.tolist()]
            results['components'] = int(getattr(model, 'n_components_', getattr(model, 'n_components', 0)))
            
            # 提取PCA可视化数据
            try:
                send_progress(82, '正在生成PCA可视化数据...', step=5, total_steps=5, details='生成PCA可视化数据')
                # 降维后的数据
                X_transformed = model.transform(X_processed)
                # 限制为前3个主成分用于可视化（如果存在）
                n_vis_components = min(3, X_transformed.shape[1])
                results['pca_transformed_data'] = X_transformed[:, :n_vis_components].tolist()
                
                # 主成分载荷（特征在主成分上的权重）
                components = getattr(model, 'components_', None)
                if components is not None:
                    results['pca_components'] = components[:n_vis_components, :].tolist()
                
                # 累积方差贡献
                if explained is not None:
                    cumulative_variance = np.cumsum(explained).tolist()
                    results['cumulative_variance_ratio'] = [round(float(x), 6) for x in cumulative_variance]
                
                # 原始特征名称（用于载荷可视化）
                if feature_names:
                    results['pca_feature_names'] = feature_names[:len(components[0])] if isinstance(feature_names, list) else list(feature_names)[:len(components[0])]
                
                # 如果有目标变量，也保存（用于着色）
                if y is not None:
                    results['pca_labels'] = y.tolist() if hasattr(y, 'tolist') else list(y)
                    if target_names:
                        results['pca_target_names'] = target_names
            except Exception as e:
                results['warnings'].append(f'PCA可视化数据提取失败: {str(e)}')
        # 检查是否为K-Means（支持多种名称格式）
        is_kmeans = (
            model_name.startswith('K 均值聚类') or 
            model_name == 'kmeans' or 
            model_name == 'K-Means' or
            model_name.lower() == 'kmeans' or
            'k-means' in model_name.lower() or
            'kmeans' in model_name.lower()
        )
        if is_kmeans:
            send_progress(50, '正在执行K-Means聚类...', step=4, total_steps=6, details='执行K-Means聚类')
            model.fit(X_processed)
            send_progress(80, '正在分析聚类结果...', step=5, total_steps=6, details='分析聚类结果')
            labels = model.labels_.tolist()
            counts = pd.Series(labels).value_counts().sort_index().to_dict()
            results['clusters'] = int(getattr(model, 'n_clusters', 0))
            results['inertia'] = round(float(model.inertia_), 6)
            results['cluster_counts'] = {str(k): int(v) for k, v in counts.items()}
            
            # 生成肘部法则图数据（不同K值的Inertia曲线）
            try:
                send_progress(85, '正在生成肘部法则图数据...', step=6, total_steps=6, details='测试不同K值的Inertia')
                current_k = model.n_clusters
                max_k = min(10, len(X_processed) // 2)  # 最大K值不超过样本数的一半，且不超过10
                min_k = 1
                
                # 测试K值从1到max_k
                k_values = list(range(min_k, max_k + 1))
                inertia_values = []
                
                for k in k_values:
                    try:
                        # 创建临时K-Means模型
                        temp_kmeans = KMeansModel(n_clusters=k, max_iter=300, tol=1e-4, random_state=42)
                        temp_kmeans.fit(X_processed)
                        inertia_values.append(float(temp_kmeans.inertia_))
                    except Exception as e:
                        # 如果某个K值失败，跳过
                        continue
                
                # 如果有些K值失败，需要调整k_values数组，只保留成功的K值
                if len(inertia_values) > 1:
                    # 调整k_values，只保留对应成功计算的K值
                    valid_k_values = []
                    valid_inertia_values = []
                    for i, k in enumerate(k_values):
                        if i < len(inertia_values):
                            valid_k_values.append(k)
                            valid_inertia_values.append(inertia_values[i])
                    
                    if len(valid_k_values) == len(valid_inertia_values) and len(valid_inertia_values) > 1:
                        # 计算最佳K值（肘部法则：找到斜率变化最大的点）
                        # 使用二阶差分找到拐点
                        if len(valid_inertia_values) >= 3:
                            # 计算一阶差分（斜率）
                            first_diff = [valid_inertia_values[i] - valid_inertia_values[i+1] for i in range(len(valid_inertia_values)-1)]
                            # 计算二阶差分（斜率的变化率）
                            second_diff = [first_diff[i] - first_diff[i+1] for i in range(len(first_diff)-1)]
                            # 找到二阶差分最大的点（斜率变化最明显的地方）
                            if second_diff:
                                elbow_idx = np.argmax(second_diff) + 1  # +1因为二阶差分比原数组少2个元素
                                best_k = valid_k_values[elbow_idx] if elbow_idx < len(valid_k_values) else current_k
                            else:
                                best_k = current_k
                        else:
                            best_k = current_k
                        
                        results['kmeans_elbow'] = {
                            'k_values': valid_k_values,
                            'inertia_values': valid_inertia_values,
                            'current_k': current_k,
                            'best_k': best_k,
                            'current_inertia': float(model.inertia_)
                        }
            except Exception as e:
                results['warnings'].append(f'肘部法则图生成失败: {str(e)}')
            
            # 生成K-Means聚类可视化数据
            try:
                send_progress(82, '正在生成K-Means可视化数据...', step=5, total_steps=5, details='生成聚类可视化数据')
                # 如果特征数>=2，返回前两个特征的散点图数据
                if X_processed.shape[1] >= 2:
                    # 确保labels是正确的列表格式
                    labels_list = labels if isinstance(labels, list) else (labels.tolist() if hasattr(labels, 'tolist') else list(labels))
                    results['kmeans_visualization'] = {
                        'data_points': X_processed[:, :2].tolist(),  # 前两个特征
                        'labels': labels_list,
                        'centroids': getattr(model, 'cluster_centers_', [])[:, :2].tolist() if hasattr(model, 'cluster_centers_') else [],
                        'n_features': X_processed.shape[1],
                        'feature_names': feature_names[:2] if feature_names and len(feature_names) >= 2 else ['特征1', '特征2']
                    }
                # 如果特征数>=3，也返回3D可视化数据
                if X_processed.shape[1] >= 3:
                    results['kmeans_visualization_3d'] = {
                        'data_points': X_processed[:, :3].tolist(),  # 前三个特征
                        'labels': labels_list,
                        'centroids': getattr(model, 'cluster_centers_', [])[:, :3].tolist() if hasattr(model, 'cluster_centers_') else [],
                        'feature_names': feature_names[:3] if feature_names and len(feature_names) >= 3 else ['特征1', '特征2', '特征3']
                    }
            except Exception as e:
                results['warnings'].append(f'K-Means可视化数据提取失败: {str(e)}')
        else:
            # 其他无监督算法（占位）
            send_progress(50, '正在执行无监督算法...', step=4, total_steps=5, details='执行无监督学习算法')
            model.fit(X_processed)
        results['split_method'] = 'N/A'
        results['test_size'] = 'N/A'
        send_progress(90, '正在生成无监督学习报告...', step=5, total_steps=5, details='生成无监督学习报告')
    

    # 添加模型信息
    send_progress(95, '正在生成最终结果...')
    results['model'] = model_name
    n_features = X_processed.shape[1] if hasattr(X_processed, 'shape') else len(X_processed[0])
    
    # 生成详细的数据集统计信息（使用原始数据X，而不是X_processed）
    # 注意：这里使用原始的X和y，因为我们要展示原始数据集的统计信息
    if 'dataset_info' not in results:
        detailed_stats = get_detailed_dataset_stats(X_original, y_original, original_feature_names, target_names, raw_df)
        dataset_info = {
            'samples': detailed_stats['samples'],
            'features': detailed_stats['features'],
            'feature_names': feature_names if isinstance(feature_names, list) else list(feature_names) if feature_names is not None else [],
            'target_names': target_names if isinstance(target_names, list) else list(target_names) if target_names is not None else [],
            'missing_values': detailed_stats['missing_values'],
            'feature_statistics': detailed_stats['feature_statistics'],
            'target_distribution': detailed_stats['target_distribution']
        }
        results['dataset_info'] = dataset_info
    
    # 获取特征重要性
    feature_importance = get_feature_importance(model, n_features, feature_names, X_processed, y, get_task_type_extended(model_name))
    
    # 确保特征重要性长度与特征数量匹配
    if feature_importance is None or len(feature_importance) == 0:
        feature_importance = [1.0 / n_features] * n_features if n_features > 0 else []
    elif len(feature_importance) != n_features:
        # 如果长度不匹配，截断或填充
        if len(feature_importance) > n_features:
            feature_importance = feature_importance[:n_features]
    else:
            # 填充零值
            feature_importance = list(feature_importance) + [0.0] * (n_features - len(feature_importance))
    
    results['feature_importance'] = feature_importance
    
    # 优先使用处理后的feature_names（可能经过独热编码等处理）
    if feature_names is not None and len(feature_names) == n_features:
        results['processed_feature_names'] = feature_names if isinstance(feature_names, list) else list(feature_names)
    elif original_feature_names is not None and len(original_feature_names) == n_features:
        results['processed_feature_names'] = original_feature_names if isinstance(original_feature_names, list) else list(original_feature_names)
    else:
        # 如果都不匹配，尝试使用preprocessing_info中的特征名称
        if 'preprocessing_info' in locals() and preprocessing_info.get('after', {}).get('feature_names'):
            after_feature_names = preprocessing_info['after']['feature_names']
            if isinstance(after_feature_names, list) and len(after_feature_names) == n_features:
                results['processed_feature_names'] = after_feature_names
            else:
                results['processed_feature_names'] = [f'特征_{i+1}' for i in range(n_features)]
        else:
            results['processed_feature_names'] = [f'特征_{i+1}' for i in range(n_features)]
    
    # 确保特征名称长度与特征数量匹配
    if len(results['processed_feature_names']) != n_features:
        if len(results['processed_feature_names']) > n_features:
            results['processed_feature_names'] = results['processed_feature_names'][:n_features]
        else:
            # 填充默认名称
            results['processed_feature_names'] = list(results['processed_feature_names']) + [f'feature_{i}' for i in range(len(results['processed_feature_names']), n_features)]
    
    # 模型类型信息
    results['model_type'] = get_model_type(model_name)
    results['task_type'] = get_task_type_extended(model_name)
    
    # 添加预处理信息到results中（确保总是添加）
    if 'preprocessing_info' in locals():
        results['preprocessing_info'] = preprocessing_info
    else:
        # 如果preprocessing_info不存在，创建一个基本的
        results['preprocessing_info'] = {
            'before': {'n_samples': len(y) if hasattr(y, '__len__') else 0, 'n_features': 0},
            'after': {'n_samples': len(y) if hasattr(y, '__len__') else 0, 'n_features': 0},
            'steps': []
        }
    
    # 发送完成消息
    send_progress(100, '实验完成！')
    
    # 转换NumPy类型为Python原生类型，确保JSON序列化成功
    return convert_numpy_types(results)


def get_model_type(model_name):
    """获取模型类型（支持中文名称和英文标识符）"""
    # 树模型（仅支持的算法）
    tree_based = [
        "决策树", "随机森林", "梯度提升树(GBDT)",
        "decision_tree", "random_forest", "gbdt"
    ]
    # 线性模型（仅支持的算法）
    linear_models = [
        "逻辑回归", "线性回归",
        "logistic_regression", "linear_regression"
    ]
    # 概率模型（仅支持的算法）
    probabilistic_models = [
        "朴素贝叶斯",
        "naive_bayes"
    ]
    # 距离模型（仅支持的算法）
    distance_based = [
        "K近邻(KNN)", "knn"
    ]
    # 核方法（仅支持的算法）
    kernel_based = [
        "支持向量机(SVM)",
        "svm"
    ]
    # 无监督学习模型（仅支持的算法）
    unsupervised_models = [
        "K 均值聚类(K-Means)", "主成分分析(PCA)",
        "kmeans", "pca"
    ]
    
    # 检查是否包含关键词（支持部分匹配）
    model_name_lower = model_name.lower() if isinstance(model_name, str) else str(model_name).lower()
    
    # 树模型（仅支持的算法）
    if any(keyword in model_name_lower for keyword in ["决策树", "decision_tree", "随机森林", "random_forest", 
                                                      "梯度提升", "gbdt"]):
        return "树模型"
    # 线性模型（仅支持的算法）
    elif any(keyword in model_name_lower for keyword in ["逻辑回归", "logistic_regression", "线性回归", "linear_regression"]):
        return "线性模型"
    # 概率模型（仅支持的算法）
    elif any(keyword in model_name_lower for keyword in ["朴素贝叶斯", "naive_bayes"]):
        return "概率模型"
    # 距离模型（仅支持的算法）
    elif any(keyword in model_name_lower for keyword in ["k近邻", "knn"]):
        return "距离模型"
    # 核方法（仅支持的算法）
    elif any(keyword in model_name_lower for keyword in ["支持向量机", "svm"]):
        return "核方法"
    # 无监督学习（仅支持的算法）
    elif any(keyword in model_name_lower for keyword in ["k均值", "k-means", "kmeans", "主成分分析", "pca"]):
        return "无监督学习"
    # 默认返回"其他"
    else:
        return "其他"

def get_task_type_extended(model_name):
    """根据模型名称判断任务类型: classification / regression / unsupervised"""
    # 支持内部标识符（如 linear_regression）和显示名称（如 "线性回归"）
    classification = {"逻辑回归", "支持向量机(SVM)", "K近邻(KNN)", "决策树", "随机森林", 
                      "梯度提升树(GBDT)", "朴素贝叶斯",
                      "logistic_regression", "svm", "knn", "decision_tree", "random_forest",
                      "gbdt", "naive_bayes"}
    regression = {"线性回归", "linear_regression"}
    unsupervised = {"K 均值聚类(K-Means)", "主成分分析(PCA)", "kmeans", "pca"}
    if model_name in classification:
        return 'classification'
    if model_name in regression:
        return 'regression'
    if model_name in unsupervised:
        return 'unsupervised'
    return 'classification'

def get_dataset_type(dataset_name):
    """根据数据集名称判断数据集类型: classification / regression / unsupervised"""
    classification_datasets = {"iris", "wine"}
    regression_datasets = {"diabetes"}
    unsupervised_datasets = {"blobs"}
    
    if dataset_name in classification_datasets:
        return 'classification'
    elif dataset_name in regression_datasets:
        return 'regression'
    elif dataset_name in unsupervised_datasets:
        return 'unsupervised'
    else:
        return 'classification'  # 默认分类

@app.route('/')
def serve_frontend():
    return send_from_directory('static', 'index.html')

@app.route('/favicon.ico')
def favicon():
    # 返回一个简单的空响应，避免404错误
    return '', 204


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
        results = train_and_evaluate(model_name, X, y, test_size, split_method, metric, hyperparams, preprocessing_config, dataset_name, cv_folds, feature_names, target_names, raw_df)
        
        # 生成实验ID和时间戳
        import time
        experiment_id = f"exp_{int(time.time())}"
        timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        
        
        # 添加数据集信息
        results['dataset'] = dataset_name
        results['experiment_id'] = experiment_id
        results['timestamp'] = timestamp
        results['model_name'] = model_name
        results['test_size'] = test_size
        results['split_method'] = split_method
        
        # 生成详细的数据集统计信息
        detailed_stats = get_detailed_dataset_stats(X, y, feature_names, target_names, raw_df)
        
        dataset_info = {
            'samples': detailed_stats['samples'],
            'features': detailed_stats['features'],
            'feature_names': feature_names if isinstance(feature_names, list) else list(feature_names) if feature_names is not None else [],
            'target_names': target_names if isinstance(target_names, list) else list(target_names) if target_names is not None else [],
            'missing_values': detailed_stats['missing_values'],
            'feature_statistics': detailed_stats['feature_statistics'],
            'target_distribution': detailed_stats['target_distribution']
        }
        
        # 如果是DataFrame，添加原始特征信息
        if isinstance(X, pd.DataFrame):
            dataset_info['original_features'] = X.columns.tolist()
            dataset_info['feature_types'] = {
                'numeric': X.select_dtypes(include=[np.number]).columns.tolist(),
                'categorical': X.select_dtypes(include=['object']).columns.tolist()
            }
        
        results['dataset_info'] = dataset_info
        
        # 保存实验历史
        experiment_record = {
            'experiment_id': experiment_id,
            'timestamp': timestamp,
            'model_name': model_name,
            'dataset': dataset_name,
            'dataset_id': dataset_id if dataset_name == 'custom' else None,
            'label_column': label_column if dataset_name == 'custom' else None,
            'test_size': test_size,
            'split_method': split_method,
            'cv_folds': cv_folds,
            'metric': metric,
            'hyperparams': hyperparams,
            'preprocessing_config': preprocessing_config,
            'results': results.copy()  # 保存完整结果
        }
        
        # 如果历史记录超过最大数量，删除最旧的记录
        if len(EXPERIMENT_HISTORY) >= MAX_HISTORY_SIZE:
            # 按时间戳排序，删除最旧的
            sorted_history = sorted(EXPERIMENT_HISTORY.items(), key=lambda x: x[1]['timestamp'])
            oldest_id = sorted_history[0][0]
            del EXPERIMENT_HISTORY[oldest_id]
        
        EXPERIMENT_HISTORY[experiment_id] = experiment_record
        
        # 确保所有NumPy类型都被转换
        results = convert_numpy_types(results)
        
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

# 已删除模型对比功能
@app.route('/deleted_compare_models', methods=['POST'])
def deleted_compare_models_endpoint():
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
        # 模型对比功能已删除
        results = {'error': '模型对比功能已删除'}
        
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

# 已删除模型对比功能
@app.route('/deleted_comparison_history', methods=['GET'])
def deleted_get_comparison_history():
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

# 实验历史记录API
@app.route('/api/experiment_history', methods=['GET'])
def get_experiment_history():
    """获取实验历史记录列表"""
    try:
        history_list = []
        for exp_id, exp_data in EXPERIMENT_HISTORY.items():
            # 提取关键信息用于列表展示
            results = exp_data.get('results', {})
            history_list.append({
                'experiment_id': exp_id,
                'timestamp': exp_data['timestamp'],
                'model_name': exp_data['model_name'],
                'dataset': exp_data['dataset'],
                'test_size': exp_data['test_size'],
                'split_method': exp_data['split_method'],
                # 提取主要性能指标
                'accuracy': results.get('accuracy', results.get('accuracy_test', 'N/A')),
                'f1_score': results.get('f1_score', results.get('f1_test', 'N/A')),
                'r2_score': results.get('r2_score', results.get('r2_test', 'N/A')),
                'mse': results.get('mse', results.get('mse_test', 'N/A'))
            })
        
        # 按时间倒序排列（最新的在前）
        history_list.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return jsonify({
            'success': True,
            'history': history_list,
            'total': len(history_list)
        })
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc() if app.debug else None
        })

@app.route('/api/experiment/<experiment_id>', methods=['GET'])
def get_experiment_detail(experiment_id):
    """获取特定实验的详细信息"""
    try:
        if experiment_id not in EXPERIMENT_HISTORY:
            return jsonify({
                'success': False,
                'error': f'实验ID {experiment_id} 不存在'
            }), 404
        
        exp_data = EXPERIMENT_HISTORY[experiment_id]
        # 确保所有NumPy类型都被转换
        exp_data = convert_numpy_types(exp_data)
        
        return jsonify({
            'success': True,
            'experiment': exp_data
        })
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc() if app.debug else None
        })

@app.route('/api/experiment/<experiment_id>', methods=['DELETE'])
def delete_experiment(experiment_id):
    """删除指定的实验记录"""
    try:
        if experiment_id not in EXPERIMENT_HISTORY:
            return jsonify({
                'success': False,
                'error': f'实验ID {experiment_id} 不存在'
            }), 404
        
        del EXPERIMENT_HISTORY[experiment_id]
        
        return jsonify({
            'success': True,
            'message': f'实验 {experiment_id} 已删除'
        })
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc() if app.debug else None
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