"""
模型評估工具模組
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold


def evaluate_model(model, X_test, y_test, target_name=None):
    """
    評估模型效能

    Args:
        model: 訓練好的模型
        X_test: 測試特徵資料
        y_test: 測試標籤資料
        target_name: 目標變數名稱，如果 y_test 是 DataFrame，則需要指定

    Returns:
        dict: 評估指標
    """
    if isinstance(y_test, pd.DataFrame):
        if target_name is None:
            raise ValueError("當 y_test 是 DataFrame 時，必須指定 target_name")
        y_true = y_test[target_name]
    else:
        y_true = y_test

    # 預測
    y_pred = model.predict(X_test)

    # 計算評估指標
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }


def evaluate_models_dict(models_dict, X_test, y_test):
    """
    評估多個模型的效能

    Args:
        models_dict: 模型字典，格式為 {目標變數名稱: 模型}
        X_test: 測試特徵資料
        y_test: 測試標籤資料

    Returns:
        dict: 評估指標字典，格式為 {目標變數名稱: 評估指標}
    """
    results = {}

    for target_name, model in models_dict.items():
        # 評估每個目標變數的模型
        eval_metrics = evaluate_model(model, X_test, y_test, target_name)
        results[target_name] = eval_metrics

    return results


def evaluate_all_models(all_models, X_test, y_test):
    """
    評估所有類型模型的效能

    Args:
        all_models: 所有模型的字典，格式為 {模型類型: {目標變數名稱: 模型}}
        X_test: 測試特徵資料
        y_test: 測試標籤資料

    Returns:
        dict: 評估指標字典，格式為 {模型類型: {目標變數名稱: 評估指標}}
    """
    results = {}

    for model_type, models_dict in all_models.items():
        # 評估每種類型的模型
        model_results = evaluate_models_dict(models_dict, X_test, y_test)
        results[model_type] = model_results

    return results


def compare_models(evaluation_results, metric='RMSE'):
    """
    比較不同模型的效能

    Args:
        evaluation_results: 評估結果字典，格式為 {模型類型: {目標變數名稱: 評估指標}}
        metric: 用於比較的指標，預設為'RMSE'

    Returns:
        pandas.DataFrame: 比較結果
    """
    # 創建比較結果資料框
    comparison = {}

    for model_type, model_results in evaluation_results.items():
        for target_name, metrics in model_results.items():
            if target_name not in comparison:
                comparison[target_name] = {}

            comparison[target_name][model_type] = metrics[metric]

    # 轉換為DataFrame
    comparison_df = pd.DataFrame(comparison)

    return comparison_df


def plot_comparison(comparison_df, title=None, is_lower_better=True):
    """
    繪製模型比較圖表

    Args:
        comparison_df: 比較結果資料框
        title: 圖表標題，預設為None
        is_lower_better: 指標是否為越低越好，預設為True
    """
    # 設置圖表大小
    plt.figure(figsize=(12, 8))

    # 繪製條形圖
    comparison_df.plot(kind='bar')

    if title:
        plt.title(title)

    plt.ylabel('指標值')
    plt.xlabel('目標變數')
    plt.xticks(rotation=0)
    plt.legend(title='模型類型')

    # 在每個條形上添加數值標籤
    for i, col in enumerate(comparison_df.columns):
        for j, value in enumerate(comparison_df[col]):
            plt.text(i, value, f'{value:.4f}', ha='center', va='bottom')

    # 添加注釋
    if is_lower_better:
        plt.annotate('* 較低的值表示更好的性能', xy=(0.02, 0.02), xycoords='figure fraction')
    else:
        plt.annotate('* 較高的值表示更好的性能', xy=(0.02, 0.02), xycoords='figure fraction')

    plt.tight_layout()

    return plt.gcf()


def plot_actual_vs_predicted(model, X_test, y_test, target_name=None, title=None):
    """
    繪製實際值與預測值對比圖

    Args:
        model: 訓練好的模型
        X_test: 測試特徵資料
        y_test: 測試標籤資料
        target_name: 目標變數名稱，如果 y_test 是 DataFrame，則需要指定
        title: 圖表標題，預設為None
    """
    if isinstance(y_test, pd.DataFrame):
        if target_name is None:
            raise ValueError("當 y_test 是 DataFrame 時，必須指定 target_name")
        y_true = y_test[target_name]
    else:
        y_true = y_test

    # 預測
    y_pred = model.predict(X_test)

    # 設置圖表大小
    plt.figure(figsize=(10, 6))

    # 繪製散點圖
    plt.scatter(y_true, y_pred, alpha=0.7)

    # 添加對角線
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')

    # 計算評估指標
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # 設置標題和標籤
    if title:
        plt.title(title)
    else:
        plt.title(f'實際值 vs 預測值 (目標: {target_name if target_name else "y"})')

    plt.xlabel('實際值')
    plt.ylabel('預測值')

    # 添加評估指標注釋
    plt.annotate(f'R² = {r2:.4f}\nRMSE = {rmse:.4f}',
                 xy=(0.05, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle='round', alpha=0.1))

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    return plt.gcf()


def cross_validate_model(model, X, y, target_name=None, cv=5, scoring='neg_mean_squared_error'):
    """
    使用交叉驗證評估模型

    Args:
        model: 模型
        X: 特徵資料
        y: 標籤資料
        target_name: 目標變數名稱，如果 y 是 DataFrame，則需要指定
        cv: 交叉驗證折數，預設為5
        scoring: 評分方法，預設為'neg_mean_squared_error'

    Returns:
        tuple: (平均分數, 標準差, 所有分數)
    """
    if isinstance(y, pd.DataFrame):
        if target_name is None:
            raise ValueError("當 y 是 DataFrame 時，必須指定 target_name")
        y_target = y[target_name]
    else:
        y_target = y

    # 執行交叉驗證
    scores = cross_val_score(model, X, y_target, cv=cv, scoring=scoring)

    # 如果是負分數（如負MSE），取絕對值
    if scoring.startswith('neg_'):
        scores = np.abs(scores)

    # 計算平均分數和標準差
    mean_score = np.mean(scores)
    std_score = np.std(scores)

    return mean_score, std_score, scores


def visualize_cross_validation(model, X, y, target_name=None, cv=5, scoring='neg_mean_squared_error'):
    """
    視覺化交叉驗證結果

    Args:
        model: 模型
        X: 特徵資料
        y: 標籤資料
        target_name: 目標變數名稱，如果 y 是 DataFrame，則需要指定
        cv: 交叉驗證折數，預設為5
        scoring: 評分方法，預設為'neg_mean_squared_error'
    """
    if isinstance(y, pd.DataFrame):
        if target_name is None:
            raise ValueError("當 y 是 DataFrame 時，必須指定 target_name")
        y_target = y[target_name]
    else:
        y_target = y

    # 設置交叉驗證
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    # 存儲每折的預測和實際值
    fold_predictions = []
    fold_actuals = []

    # 設置圖表
    plt.figure(figsize=(12, 6))

    # 執行交叉驗證並記錄每折的結果
    for i, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_fold_train, X_fold_test = X.iloc[train_idx], X.iloc[test_idx]
        y_fold_train, y_fold_test = y_target.iloc[train_idx], y_target.iloc[test_idx]

        # 訓練模型
        model.fit(X_fold_train, y_fold_train)

        # 預測
        y_fold_pred = model.predict(X_fold_test)

        # 存儲預測和實際值
        fold_predictions.append(y_fold_pred)
        fold_actuals.append(y_fold_test)

        # 計算這折的RMSE
        fold_rmse = np.sqrt(mean_squared_error(y_fold_test, y_fold_pred))

        # 繪製每折的預測和實際值
        plt.subplot(1, cv, i + 1)
        plt.scatter(y_fold_test, y_fold_pred, alpha=0.7)

        # 添加對角線
        min_val = min(y_fold_test.min(), y_fold_pred.min())
        max_val = max(y_fold_test.max(), y_fold_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')

        plt.title(f'折 {i + 1}\nRMSE: {fold_rmse:.4f}')
        plt.xlabel('實際值')
        if i == 0:
            plt.ylabel('預測值')

    plt.tight_layout()

    # 計算所有折的整體RMSE
    all_predictions = np.concatenate(fold_predictions)
    all_actuals = np.concatenate(fold_actuals)
    overall_rmse = np.sqrt(mean_squared_error(all_actuals, all_predictions))

    plt.suptitle(f'交叉驗證結果 - {cv}折 (整體RMSE: {overall_rmse:.4f})')

    return plt.gcf()


def plot_residuals(model, X_test, y_test, target_name=None, title=None):
    """
    繪製殘差圖

    Args:
        model: 訓練好的模型
        X_test: 測試特徵資料
        y_test: 測試標籤資料
        target_name: 目標變數名稱，如果 y_test 是 DataFrame，則需要指定
        title: 圖表標題，預設為None
    """
    if isinstance(y_test, pd.DataFrame):
        if target_name is None:
            raise ValueError("當 y_test 是 DataFrame 時，必須指定 target_name")
        y_true = y_test[target_name]
    else:
        y_true = y_test

    # 預測
    y_pred = model.predict(X_test)

    # 計算殘差
    residuals = y_true - y_pred

    # 設置圖表大小
    plt.figure(figsize=(12, 10))

    # 1. 殘差 vs 預測值
    plt.subplot(2, 2, 1)
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('殘差 vs 預測值')
    plt.xlabel('預測值')
    plt.ylabel('殘差')

    # 2. 殘差直方圖
    plt.subplot(2, 2, 2)
    plt.hist(residuals, bins=20, alpha=0.7, edgecolor='black')
    plt.title('殘差直方圖')
    plt.xlabel('殘差')
    plt.ylabel('頻率')

    # 3. Q-Q 圖
    plt.subplot(2, 2, 3)
    sorted_residuals = np.sort(residuals)
    norm_quantiles = np.linspace(0, 1, len(sorted_residuals))
    sorted_norm = np.sqrt(2) * np.erfinv(2 * norm_quantiles - 1)
    plt.scatter(sorted_norm, sorted_residuals, alpha=0.7)

    # 添加對角線
    min_val = min(sorted_norm.min(), sorted_residuals.min())
    max_val = max(sorted_norm.max(), sorted_residuals.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')

    plt.title('殘差 Q-Q 圖')
    plt.xlabel('理論分位數')
    plt.ylabel('樣本分位數')

    # 4. 殘差 vs 觀測值序號
    plt.subplot(2, 2, 4)
    plt.scatter(range(len(residuals)), residuals, alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('殘差 vs 觀測值序號')
    plt.xlabel('觀測值序號')
    plt.ylabel('殘差')

    # 設置整體標題
    if title:
        plt.suptitle(title, fontsize=16)
    else:
        plt.suptitle(f'殘差分析 (目標: {target_name if target_name else "y"})', fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return plt.gcf()


def plot_feature_importance_coefficients(model, feature_names, target_name=None, top_n=10):
    """
    繪製線性模型係數直方圖

    Args:
        model: 訓練好的線性模型
        feature_names: 特徵名稱列表
        target_name: 目標變數名稱，預設為None
        top_n: 顯示前幾個最重要的特徵，預設為10
    """
    # 獲取係數
    if hasattr(model, 'coef_'):
        coefs = model.coef_
    else:
        raise ValueError("模型沒有 'coef_' 屬性，不是線性模型")

    # 如果是多項式特徵，係數可能是二維數組
    if coefs.ndim > 1 and coefs.shape[0] == 1:
        coefs = coefs.ravel()

    # 配對特徵名稱和係數
    feature_importance = list(zip(feature_names, np.abs(coefs)))

    # 按重要性排序
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    # 取前N個重要特徵
    top_features = feature_importance[:top_n]

    # 提取名稱和係數
    names, values = zip(*top_features)

    # 設置圖表大小
    plt.figure(figsize=(10, 8))

    # 繪製水平條形圖
    bars = plt.barh(range(len(names)), values, align='center')

    # 根據係數正負設置顏色
    for i, (name, value) in enumerate(top_features):
        original_value = coefs[feature_names.index(name)]
        color = 'green' if original_value > 0 else 'red'
        bars[i].set_color(color)

    plt.yticks(range(len(names)), names)
    plt.xlabel('係數絕對值')

    target_str = f"目標: {target_name}" if target_name else ""
    plt.title(f'特徵重要性 ({target_str})')

    # 添加正負號示例
    plt.legend([plt.Rectangle((0, 0), 1, 1, color='green'),
                plt.Rectangle((0, 0), 1, 1, color='red')],
               ['正係數 (+)', '負係數 (-)'])

    plt.gca().invert_yaxis()  # 倒序顯示，最重要的在頂部
    plt.tight_layout()

    return plt.gcf()