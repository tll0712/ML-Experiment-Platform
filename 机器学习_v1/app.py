from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
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
import os
import json

app = Flask(__name__)
# 限制上传大小为50MB（可按需调整）
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
# 收紧CORS到本地常见端口（可按需调整）
CORS(app, resources={r"/*": {"origins": [
    "http://localhost:5001", "http://127.0.0.1:5001",
    "http://localhost:3000", "http://127.0.0.1:3000"
]}})

# 临时存储上传的数据集（仅开发模式内存保存）
UPLOADED_DATASETS = {}

# 所有可用的模型
MODELS = {
    # 仅保留用户指定的10个经典模型
    # 回归
    "线性回归": LinearRegression(),
    # 分类
    "逻辑回归": LogisticRegression(random_state=42, max_iter=1000),
    "决策树": DecisionTreeClassifier(random_state=42),
    "支持向量机(SVM)": SVC(kernel='rbf', random_state=42, probability=True),
    "K近邻(KNN)": KNeighborsClassifier(n_neighbors=3),
    "随机森林": RandomForestClassifier(n_estimators=100, random_state=42),
    "梯度提升树(GBDT)": GradientBoostingClassifier(random_state=42),
    # 无监督
    "K 均值聚类(K-Means)": KMeans(n_clusters=3, random_state=42),
    "主成分分析(PCA)": PCA(n_components=2, random_state=42),
    # 神经网络
    "多层感知机(MLP)": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
}

# 修改 load_dataset 函数中的西瓜数据集部分
def load_dataset(dataset_name, test_size=0.3):
    if dataset_name == "iris":
        data = load_iris()
        X = data.data
        y = data.target
        feature_names = data.feature_names
        target_names = data.target_names.tolist()
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
        return X_processed, None, scaler

# 在 train_and_evaluate 函数中修复特征重要性处理
def get_feature_importance(model, n_features, feature_names=None):
    """获取特征重要性（如果模型支持）"""
    try:
        # 检查模型是否已经训练（有coef_或feature_importances_属性）
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            # 归一化
            if importance.sum() > 0:
                importance = importance / importance.sum()
            return importance.tolist()
        elif hasattr(model, 'coef_'):
            # 对于线性模型，使用系数的绝对值作为重要性
            if len(model.coef_.shape) > 1:
                importance = np.mean(np.abs(model.coef_), axis=0)
            else:
                importance = np.abs(model.coef_)
            # 归一化
            if importance.sum() > 0:
                importance = importance / importance.sum()
            return importance.tolist()
    except Exception as e:
        print(f"获取特征重要性时出错: {e}")
    
    # 如果不支持特征重要性，返回平均分布
    return [1.0/n_features] * n_features if n_features > 0 else []

# 修改 train_and_evaluate 函数中的特征重要性调用
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 修改 train_and_evaluate 函数
def train_and_evaluate(model_name, X, y, test_size, split_method, metric='accuracy'):
    results = {}
    results['warnings'] = []
    
    # 预处理数据
    X_processed, feature_names, preprocessor = preprocess_data(X, y)
    
    # 获取模型
    model = MODELS.get(model_name)
    if model is None:
        raise ValueError(f"未知模型: {model_name}")

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
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            if metric == 'f1':
                score = f1_score(y_test, y_pred, average='weighted')
                results['f1_score'] = round(score, 4)
                results['accuracy'] = round(accuracy_score(y_test, y_pred), 4)
            else:
                score = accuracy_score(y_test, y_pred)
                results['accuracy'] = round(score, 4)
                results['f1_score'] = round(f1_score(y_test, y_pred, average='weighted'), 4)
            results['split_method'] = '随机分割'
            results['test_size'] = test_size
            report = classification_report(y_test, y_pred, output_dict=True)
            results['classification_report'] = report
            cm = confusion_matrix(y_test, y_pred)
            results['confusion_matrix'] = cm.tolist()
        else:
            # 使用 StratifiedKFold 优先；若最小类别数不足，自动降低折数，直至>=2；再不行退回 KFold
            desired_splits = 5
            n_splits = max(2, min(desired_splits, min_count)) if min_count > 0 else 2
            scoring = 'f1_weighted' if metric == 'f1' else 'accuracy'
            try:
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                cv_scores = cross_val_score(model, X_processed, y, cv=cv, scoring=scoring)
                if n_splits < desired_splits:
                    results['warnings'].append(f'最小类别样本仅 {min_count}，交叉验证折数降为 {n_splits}。')
            except ValueError:
                # 回退到普通 KFold
                cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                cv_scores = cross_val_score(model, X_processed, y, cv=cv, scoring=scoring)
                results['warnings'].append('分层交叉验证不可用，已回退至非分层 KFold。')
            if metric == 'f1':
                results['f1_score'] = round(cv_scores.mean(), 4)
                accuracy_scores = cross_val_score(model, X_processed, y, cv=kf, scoring='accuracy')
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
            else:
                results['accuracy'] = round(cv_scores.mean(), 4)
                f1_scores = cross_val_score(model, X_processed, y, cv=kf, scoring='f1_weighted')
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
            results['split_method'] = '5折交叉验证'
            results['test_size'] = 'N/A'

    elif task_type == 'regression':
        # y 必须为数值
        if not isinstance(y, (np.ndarray, list, pd.Series)):
            raise ValueError('回归任务需要数值型标签')
        if split_method == 'random':
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y, test_size=test_size, random_state=42
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
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
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            # 使用R2为主指标，同时返回MSE/MAE
            r2_scores = cross_val_score(model, X_processed, y, cv=kf, scoring='r2')
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
            results['split_method'] = '5折交叉验证'
            results['test_size'] = 'N/A'

    else:  # unsupervised
        # 无监督直接在全量数据上拟合
        if model_name.startswith('主成分分析'):
            model.fit(X_processed)
            explained = getattr(model, 'explained_variance_ratio_', None)
            if explained is not None:
                results['explained_variance_ratio'] = [round(float(x), 6) for x in explained.tolist()]
            results['components'] = int(getattr(model, 'n_components_', getattr(model, 'n_components', 0)))
        elif model_name.startswith('K 均值聚类'):
            model.fit(X_processed)
            labels = model.labels_.tolist()
            counts = pd.Series(labels).value_counts().sort_index().to_dict()
            results['clusters'] = int(getattr(model, 'n_clusters', 0))
            results['inertia'] = round(float(model.inertia_), 6)
            results['cluster_counts'] = {str(k): int(v) for k, v in counts.items()}
        else:
            # 其他无监督算法（占位）
            model.fit(X_processed)
        results['split_method'] = 'N/A'
        results['test_size'] = 'N/A'
    
    # 添加模型信息
    results['model'] = model_name
    n_features = X_processed.shape[1] if hasattr(X_processed, 'shape') else len(X_processed[0])
    results['feature_importance'] = get_feature_importance(model, n_features, feature_names)
    results['processed_feature_names'] = feature_names
    
    # 模型类型信息
    results['model_type'] = get_model_type(model_name)
    results['task_type'] = get_task_type_extended(model_name)
    
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

@app.route('/run_experiment', methods=['POST'])
# 修改 /run_experiment 路由
def run_experiment():
    try:
        data = request.json
        dataset_name = data.get('dataset', 'iris')
        test_size = float(data.get('test_size', '0.3').strip('%')) / 100
        split_method = data.get('split_method', 'random')
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
        
        # 训练和评估模型，传入评估指标
        results = train_and_evaluate(model_name, X, y, test_size, split_method, metric)
        
        # 添加数据集信息
        results['dataset'] = dataset_name
        dataset_info = {
            'samples': len(y),
            'features': X.shape[1] if hasattr(X, 'shape') else len(X.columns),
            'feature_names': feature_names,
            'target_names': target_names
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

@app.route('/available_models')
def get_available_models():
    """获取所有可用的模型列表"""
    models_list = list(MODELS.keys())
    return jsonify({
        'success': True,
        'models': models_list
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

@app.errorhandler(413)
def handle_file_too_large(e):
    return jsonify({
        'success': False,
        'error': '文件过大，超过服务器限制',
        'limit_mb': int(app.config.get('MAX_CONTENT_LENGTH', 0) / (1024*1024))
    }), 413

if __name__ == '__main__':
    # 确保datasets目录存在
    os.makedirs('datasets', exist_ok=True)
    
    app.run(debug=True, port=5001)