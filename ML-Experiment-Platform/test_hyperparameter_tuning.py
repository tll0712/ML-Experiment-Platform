#!/usr/bin/env python3
"""
超参数调优功能测试脚本
Hyperparameter Tuning Test Script
"""

import sys
import os
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# 导入自定义模型
from app import (
    MyLogisticRegression, MySVM, MyDecisionTree, MyKNN,
    MyRandomForest, MyGBDT, MyNaiveBayes, MyLinearRegression,
    KMeansModel, PCAModel
)

def test_classification_hyperparameter_tuning():
    """测试分类任务的超参数调优"""
    print("=" * 60)
    print("测试分类任务超参数调优")
    print("=" * 60)
    
    # 生成测试数据
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=3,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 测试逻辑回归
    print("\n1. 测试逻辑回归超参数调优")
    model = MyLogisticRegression(max_iter=1000)
    param_grid = {
        'C': [0.1, 1, 10],
        'max_iter': [100, 500]
    }
    
    from sklearn.model_selection import GridSearchCV
    grid_search = GridSearchCV(
        model, param_grid, cv=3, scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"   最佳参数: {grid_search.best_params_}")
    print(f"   最佳得分: {grid_search.best_score_:.4f}")
    print(f"   测试集得分: {grid_search.score(X_test, y_test):.4f}")
    
    # 测试决策树
    print("\n2. 测试决策树超参数调优")
    model = MyDecisionTree()
    param_grid = {
        'max_depth': [3, 5, 10],
        'min_samples_split': [2, 5]
    }
    
    grid_search = GridSearchCV(
        model, param_grid, cv=3, scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"   最佳参数: {grid_search.best_params_}")
    print(f"   最佳得分: {grid_search.best_score_:.4f}")
    print(f"   测试集得分: {grid_search.score(X_test, y_test):.4f}")
    
    # 测试KNN
    print("\n3. 测试KNN超参数调优")
    model = MyKNN()
    param_grid = {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance']
    }
    
    grid_search = GridSearchCV(
        model, param_grid, cv=3, scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"   最佳参数: {grid_search.best_params_}")
    print(f"   最佳得分: {grid_search.best_score_:.4f}")
    print(f"   测试集得分: {grid_search.score(X_test, y_test):.4f}")
    
    # 测试随机森林
    print("\n4. 测试随机森林超参数调优")
    model = MyRandomForest()
    param_grid = {
        'n_estimators': [10, 20],
        'max_depth': [5, 10]
    }
    
    grid_search = GridSearchCV(
        model, param_grid, cv=3, scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"   最佳参数: {grid_search.best_params_}")
    print(f"   最佳得分: {grid_search.best_score_:.4f}")
    print(f"   测试集得分: {grid_search.score(X_test, y_test):.4f}")
    
    # 测试朴素贝叶斯
    print("\n5. 测试朴素贝叶斯超参数调优")
    model = MyNaiveBayes()
    param_grid = {
        'alpha': [0.1, 1.0, 10.0],
        'var_smoothing': [1e-9, 1e-8]
    }
    
    grid_search = GridSearchCV(
        model, param_grid, cv=3, scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"   最佳参数: {grid_search.best_params_}")
    print(f"   最佳得分: {grid_search.best_score_:.4f}")
    print(f"   测试集得分: {grid_search.score(X_test, y_test):.4f}")
    
    print("\n✅ 分类任务超参数调优测试完成")


def test_regression_hyperparameter_tuning():
    """测试回归任务的超参数调优"""
    print("\n" + "=" * 60)
    print("测试回归任务超参数调优")
    print("=" * 60)
    
    # 生成测试数据
    X, y = make_regression(
        n_samples=200,
        n_features=10,
        n_informative=5,
        noise=10,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 测试线性回归
    print("\n1. 测试线性回归超参数调优")
    model = MyLinearRegression()
    param_grid = {
        'learning_rate': [0.001, 0.01, 0.1],
        'max_iter': [500, 1000]
    }
    
    from sklearn.model_selection import GridSearchCV
    grid_search = GridSearchCV(
        model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"   最佳参数: {grid_search.best_params_}")
    print(f"   最佳得分: {grid_search.best_score_:.4f}")
    print(f"   测试集得分: {grid_search.score(X_test, y_test):.4f}")
    
    print("\n✅ 回归任务超参数调优测试完成")


def test_unsupervised_hyperparameter_tuning():
    """测试无监督学习的超参数调优"""
    print("\n" + "=" * 60)
    print("测试无监督学习超参数调优")
    print("=" * 60)
    
    # 生成测试数据
    X, _ = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=3,
        random_state=42
    )
    
    # 测试K-Means
    print("\n1. 测试K-Means超参数调优")
    model = KMeansModel()
    param_grid = {
        'n_clusters': [2, 3, 4, 5],
        'max_iter': [100, 300]
    }
    
    from sklearn.model_selection import GridSearchCV
    
    # 自定义评分函数：负惯性（越小越好，所以用负数）
    def neg_inertia_scorer(estimator, X, y=None):
        estimator.fit(X)
        return -estimator.inertia_ if hasattr(estimator, 'inertia_') else 0.0
    
    grid_search = GridSearchCV(
        model, param_grid, cv=3, scoring=neg_inertia_scorer, n_jobs=-1
    )
    grid_search.fit(X)
    
    print(f"   最佳参数: {grid_search.best_params_}")
    print(f"   最佳得分: {grid_search.best_score_:.4f}")
    
    # 测试PCA
    print("\n2. 测试PCA超参数调优")
    model = PCAModel()
    param_grid = {
        'n_components': [2, 3, 4, 5]
    }
    
    # PCA使用解释方差比作为评分
    def explained_variance_scorer(estimator, X, y=None):
        estimator.fit(X)
        return np.sum(estimator.explained_variance_ratio_) if hasattr(estimator, 'explained_variance_ratio_') else 0.0
    
    grid_search = GridSearchCV(
        model, param_grid, cv=3, scoring=explained_variance_scorer, n_jobs=-1
    )
    grid_search.fit(X)
    
    print(f"   最佳参数: {grid_search.best_params_}")
    print(f"   最佳得分: {grid_search.best_score_:.4f}")
    
    print("\n✅ 无监督学习超参数调优测试完成")


def test_randomized_search():
    """测试随机搜索超参数调优"""
    print("\n" + "=" * 60)
    print("测试随机搜索超参数调优")
    print("=" * 60)
    
    # 生成测试数据
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=3,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 测试随机搜索
    print("\n1. 测试随机搜索（决策树）")
    model = MyDecisionTree()
    param_grid = {
        'max_depth': [3, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10, 15]
    }
    
    from sklearn.model_selection import RandomizedSearchCV
    random_search = RandomizedSearchCV(
        model, param_grid, cv=3, scoring='accuracy', 
        n_iter=10, n_jobs=-1, random_state=42
    )
    random_search.fit(X_train, y_train)
    
    print(f"   最佳参数: {random_search.best_params_}")
    print(f"   最佳得分: {random_search.best_score_:.4f}")
    print(f"   测试集得分: {random_search.score(X_test, y_test):.4f}")
    print(f"   搜索的参数组合数: {len(random_search.cv_results_['params'])}")
    
    print("\n✅ 随机搜索超参数调优测试完成")


def main():
    """主测试函数"""
    print("🚀 开始超参数调优功能测试")
    print("=" * 60)
    
    try:
        # 测试分类任务
        test_classification_hyperparameter_tuning()
        
        # 测试回归任务
        test_regression_hyperparameter_tuning()
        
        # 测试无监督学习
        test_unsupervised_hyperparameter_tuning()
        
        # 测试随机搜索
        test_randomized_search()
        
        print("\n" + "=" * 60)
        print("✅ 所有测试通过！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

