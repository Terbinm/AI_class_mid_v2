"""
線性相關模型模組
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


def train_linear_regression(X_train, y_train):
    """
    訓練線性迴歸模型

    Args:
        X_train: 訓練特徵資料
        y_train: 訓練標籤資料

    Returns:
        dict: 訓練好的模型字典，鍵為目標變數名稱
    """
    models = {}

    # 對每個目標變數訓練單獨的模型
    for col in y_train.columns:
        model = LinearRegression()
        model.fit(X_train, y_train[col])
        models[col] = model
        print(f"線性迴歸模型 '{col}' 訓練完成")

    return models


def train_multiple_linear_regression(X_train, y_train):
    """
    訓練多變量線性迴歸模型 (與普通線性迴歸相同，但強調多個自變量)

    Args:
        X_train: 訓練特徵資料
        y_train: 訓練標籤資料

    Returns:
        dict: 訓練好的模型字典，鍵為目標變數名稱
    """
    # 在這個實現中，多變量線性迴歸與普通線性迴歸相同
    return train_linear_regression(X_train, y_train)


def train_lasso_regression(X_train, y_train, cv=5):
    """
    訓練Lasso迴歸模型 (L1正則化)

    Args:
        X_train: 訓練特徵資料
        y_train: 訓練標籤資料
        cv: 交叉驗證折數，預設為5

    Returns:
        dict: 訓練好的模型字典，鍵為目標變數名稱
    """
    models = {}

    # 對每個目標變數訓練單獨的模型
    for col in y_train.columns:
        # 定義參數網格
        param_grid = {
            'alpha': np.logspace(-4, 1, 20)
        }

        # 創建Lasso模型
        lasso = Lasso(max_iter=10000, random_state=42)

        # 使用網格搜索找到最佳參數
        grid = GridSearchCV(lasso, param_grid, cv=cv, scoring='neg_mean_squared_error')
        grid.fit(X_train, y_train[col])

        # 獲取最佳模型
        best_model = grid.best_estimator_
        models[col] = best_model

        print(f"Lasso迴歸模型 '{col}' 訓練完成，最佳alpha={best_model.alpha:.6f}")

    return models


def train_ridge_regression(X_train, y_train, cv=5):
    """
    訓練Ridge迴歸模型 (L2正則化)

    Args:
        X_train: 訓練特徵資料
        y_train: 訓練標籤資料
        cv: 交叉驗證折數，預設為5

    Returns:
        dict: 訓練好的模型字典，鍵為目標變數名稱
    """
    models = {}

    # 對每個目標變數訓練單獨的模型
    for col in y_train.columns:
        # 定義參數網格
        param_grid = {
            'alpha': np.logspace(-4, 1, 20)
        }

        # 創建Ridge模型
        ridge = Ridge(random_state=42)

        # 使用網格搜索找到最佳參數
        grid = GridSearchCV(ridge, param_grid, cv=cv, scoring='neg_mean_squared_error')
        grid.fit(X_train, y_train[col])

        # 獲取最佳模型
        best_model = grid.best_estimator_
        models[col] = best_model

        print(f"Ridge迴歸模型 '{col}' 訓練完成，最佳alpha={best_model.alpha:.6f}")

    return models


def train_polynomial_regression(X_train, y_train, degree=2, cv=5):
    """
    訓練多項式迴歸模型

    Args:
        X_train: 訓練特徵資料
        y_train: 訓練標籤資料
        degree: 多項式次數，預設為2
        cv: 交叉驗證折數，預設為5

    Returns:
        dict: 訓練好的模型字典，鍵為目標變數名稱
    """
    models = {}

    # 對每個目標變數訓練單獨的模型
    for col in y_train.columns:
        # 定義參數網格
        param_grid = {
            'polynomialfeatures__degree': range(1, degree + 1)
        }

        # 創建多項式迴歸管道
        poly_pipeline = Pipeline([
            ('polynomialfeatures', PolynomialFeatures()),
            ('linearregression', LinearRegression())
        ])

        # 使用網格搜索找到最佳參數
        grid = GridSearchCV(poly_pipeline, param_grid, cv=cv, scoring='neg_mean_squared_error')
        grid.fit(X_train, y_train[col])

        # 獲取最佳模型
        best_model = grid.best_estimator_
        models[col] = best_model

        best_degree = best_model.named_steps['polynomialfeatures'].degree
        print(f"多項式迴歸模型 '{col}' 訓練完成，最佳次數={best_degree}")

    return models


def predict_with_models(models, X_test):
    """
    使用訓練好的模型進行預測

    Args:
        models: 模型字典，鍵為目標變數名稱
        X_test: 測試特徵資料

    Returns:
        pandas.DataFrame: 預測結果
    """
    predictions = {}

    # 對每個目標變數使用對應的模型進行預測
    for target_name, model in models.items():
        predictions[target_name] = model.predict(X_test)

    # 將預測結果轉換為DataFrame
    predictions_df = pd.DataFrame(predictions)

    return predictions_df


def train_all_models(X_train, y_train, cv=5):
    """
    訓練所有類型的模型

    Args:
        X_train: 訓練特徵資料
        y_train: 訓練標籤資料
        cv: 交叉驗證折數，預設為5

    Returns:
        dict: 包含所有模型的字典
    """
    all_models = {}

    print("開始訓練線性迴歸模型...")
    all_models['linear'] = train_linear_regression(X_train, y_train)

    print("開始訓練多變量線性迴歸模型...")
    all_models['multiple_linear'] = train_multiple_linear_regression(X_train, y_train)

    print("開始訓練Lasso迴歸模型...")
    all_models['lasso'] = train_lasso_regression(X_train, y_train, cv)

    print("開始訓練Ridge迴歸模型...")
    all_models['ridge'] = train_ridge_regression(X_train, y_train, cv)

    print("開始訓練多項式迴歸模型...")
    all_models['polynomial'] = train_polynomial_regression(X_train, y_train, degree=3, cv=cv)

    return all_models