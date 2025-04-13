"""
特徵選擇策略模組
"""
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

def all_in_selection(X, y, verbose=True):
    """
    All-in 策略：使用所有特徵

    Args:
        X: 特徵資料框
        y: 標籤資料或資料框
        verbose: 是否顯示詳細信息，預設為True

    Returns:
        pandas.DataFrame: 選取的特徵資料框
    """
    if verbose:
        print(f"使用 All-in 策略，保留所有 {X.shape[1]} 個特徵")

    return X

def backward_selection(X, y, target_name=None, cv=5, scoring='neg_mean_squared_error', verbose=True):
    """
    後向選擇策略：從所有特徵開始，逐一移除最不重要的特徵

    Args:
        X: 特徵資料框
        y: 標籤資料或資料框
        target_name: 目標變數名稱，如果 y 是 DataFrame，則需要指定
        cv: 交叉驗證折數，預設為5
        scoring: 評分方法，預設為'neg_mean_squared_error'
        verbose: 是否顯示詳細信息，預設為True

    Returns:
        tuple: (選取的特徵資料框, 選取的特徵名稱列表)
    """
    if isinstance(y, pd.DataFrame):
        if target_name is None:
            raise ValueError("當 y 是 DataFrame 時，必須指定 target_name")
        y_target = y[target_name]
    else:
        y_target = y

    features = list(X.columns)
    n_features = len(features)
    best_score = float('-inf')
    best_features = features.copy()

    if verbose:
        print(f"開始後向選擇，初始特徵數: {n_features}")

    # 逐一移除特徵
    for i in range(n_features):
        scores = []
        removed_features = []

        # 嘗試移除每個剩余特徵
        for j in range(len(best_features)):
            test_features = best_features.copy()
            removed_feature = test_features.pop(j)
            removed_features.append(removed_feature)

            # 評估移除特徵後的模型性能
            model = LinearRegression()
            score = np.mean(cross_val_score(model, X[test_features], y_target, cv=cv, scoring=scoring))
            scores.append(score)

        # 找到移除後性能最好的特徵組合
        max_score_idx = np.argmax(scores)

        # 如果移除某個特徵後性能提升，則移除該特徵
        if scores[max_score_idx] > best_score:
            feature_to_remove = removed_features[max_score_idx]
            best_features.remove(feature_to_remove)
            best_score = scores[max_score_idx]

            if verbose:
                print(f"移除特徵 '{feature_to_remove}'，當前分數: {best_score:.6f}，剩余特徵數: {len(best_features)}")
        else:
            if verbose:
                print(f"後向選擇完成，選取了 {len(best_features)} 個特徵，最佳分數: {best_score:.6f}")
            break

    return X[best_features], best_features

def forward_selection(X, y, target_name=None, cv=5, scoring='neg_mean_squared_error', verbose=True):
    """
    前向選擇策略：從空集開始，逐一添加最重要的特徵

    Args:
        X: 特徵資料框
        y: 標籤資料或資料框
        target_name: 目標變數名稱，如果 y 是 DataFrame，則需要指定
        cv: 交叉驗證折數，預設為5
        scoring: 評分方法，預設為'neg_mean_squared_error'
        verbose: 是否顯示詳細信息，預設為True

    Returns:
        tuple: (選取的特徵資料框, 選取的特徵名稱列表)
    """
    if isinstance(y, pd.DataFrame):
        if target_name is None:
            raise ValueError("當 y 是 DataFrame 時，必須指定 target_name")
        y_target = y[target_name]
    else:
        y_target = y

    features = list(X.columns)
    n_features = len(features)
    best_score = float('-inf')
    best_features = []

    if verbose:
        print(f"開始前向選擇，初始特徵數: 0")

    # 直到所有特徵都被考慮
    while len(best_features) < n_features:
        scores = []
        added_features = []

        # 嘗試添加每個尚未選取的特徵
        for feature in features:
            if feature not in best_features:
                test_features = best_features.copy()
                test_features.append(feature)
                added_features.append(feature)

                # 評估添加特徵後的模型性能
                model = LinearRegression()
                score = np.mean(cross_val_score(model, X[test_features], y_target, cv=cv, scoring=scoring))
                scores.append(score)

        # 如果沒有可以添加的特徵，退出循環
        if not scores:
            break

        # 找到添加後性能最好的特徵
        max_score_idx = np.argmax(scores)

        # 如果添加特徵後性能提升，則添加該特徵
        if scores[max_score_idx] > best_score:
            feature_to_add = added_features[max_score_idx]
            best_features.append(feature_to_add)
            best_score = scores[max_score_idx]

            if verbose:
                print(f"添加特徵 '{feature_to_add}'，當前分數: {best_score:.6f}，當前特徵數: {len(best_features)}")
        else:
            if verbose:
                print(f"前向選擇完成，選取了 {len(best_features)} 個特徵，最佳分數: {best_score:.6f}")
            break

    return X[best_features], best_features

def bidirectional_selection(X, y, target_name=None, cv=5, scoring='neg_mean_squared_error', verbose=True):
    """
    雙向選擇策略：結合前向和後向選擇

    Args:
        X: 特徵資料框
        y: 標籤資料或資料框
        target_name: 目標變數名稱，如果 y 是 DataFrame，則需要指定
        cv: 交叉驗證折數，預設為5
        scoring: 評分方法，預設為'neg_mean_squared_error'
        verbose: 是否顯示詳細信息，預設為True

    Returns:
        tuple: (選取的特徵資料框, 選取的特徵名稱列表)
    """
    if isinstance(y, pd.DataFrame):
        if target_name is None:
            raise ValueError("當 y 是 DataFrame 時，必須指定 target_name")
        y_target = y[target_name]
    else:
        y_target = y

    features = list(X.columns)
    n_features = len(features)
    best_score = float('-inf')
    best_features = []

    if verbose:
        print(f"開始雙向選擇，初始特徵數: 0")

    # 直到沒有改進為止
    improvement = True
    while improvement:
        improvement = False

        # 嘗試添加特徵（前向部分）
        forward_scores = []
        forward_added_features = []

        for feature in features:
            if feature not in best_features:
                test_features = best_features.copy()
                test_features.append(feature)
                forward_added_features.append(feature)

                model = LinearRegression()
                score = np.mean(cross_val_score(model, X[test_features], y_target, cv=cv, scoring=scoring))
                forward_scores.append(score)

        # 如果有可以添加的特徵
        if forward_scores:
            max_forward_score_idx = np.argmax(forward_scores)

            # 如果添加特徵後性能提升
            if forward_scores[max_forward_score_idx] > best_score:
                feature_to_add = forward_added_features[max_forward_score_idx]
                best_features.append(feature_to_add)
                best_score = forward_scores[max_forward_score_idx]
                improvement = True

                if verbose:
                    print(f"添加特徵 '{feature_to_add}'，當前分數: {best_score:.6f}，當前特徵數: {len(best_features)}")

        # 如果有至少兩個特徵，嘗試移除特徵（後向部分）
        if len(best_features) >= 2:
            backward_scores = []
            backward_removed_features = []

            for i, feature in enumerate(best_features):
                test_features = best_features.copy()
                test_features.pop(i)
                backward_removed_features.append(feature)

                model = LinearRegression()
                score = np.mean(cross_val_score(model, X[test_features], y_target, cv=cv, scoring=scoring))
                backward_scores.append(score)

            max_backward_score_idx = np.argmax(backward_scores)

            # 如果移除特徵後性能提升
            if backward_scores[max_backward_score_idx] > best_score:
                feature_to_remove = backward_removed_features[max_backward_score_idx]
                best_features.remove(feature_to_remove)
                best_score = backward_scores[max_backward_score_idx]
                improvement = True

                if verbose:
                    print(f"移除特徵 '{feature_to_remove}'，當前分數: {best_score:.6f}，當前特徵數: {len(best_features)}")

        # 如果沒有改進，結束循環
        if not improvement and verbose:
            print(f"雙向選擇完成，選取了 {len(best_features)} 個特徵，最佳分數: {best_score:.6f}")

    return X[best_features], best_features

def select_k_best(X, y, target_name=None, k=10, verbose=True):
    """
    使用 SelectKBest 選擇 k 個最佳特徵

    Args:
        X: 特徵資料框
        y: 標籤資料或資料框
        target_name: 目標變數名稱，如果 y 是 DataFrame，則需要指定
        k: 要選擇的特徵數量，預設為10
        verbose: 是否顯示詳細信息，預設為True

    Returns:
        tuple: (選取的特徵資料框, 選取的特徵名稱列表)
    """
    if isinstance(y, pd.DataFrame):
        if target_name is None:
            raise ValueError("當 y 是 DataFrame 時，必須指定 target_name")
        y_target = y[target_name]
    else:
        y_target = y

    # 調整 k 不超過特徵數量
    k = min(k, X.shape[1])

    # 使用 f_regression 統計測試
    selector = SelectKBest(f_regression, k=k)
    X_new = selector.fit_transform(X, y_target)

    # 獲取選取的特徵名稱
    mask = selector.get_support()
    selected_features = X.columns[mask].tolist()

    if verbose:
        # 獲取特徵分數
        scores = selector.scores_
        # 將特徵名稱和分數配對並排序
        feature_scores = list(zip(X.columns, scores))
        feature_scores.sort(key=lambda x: x[1], reverse=True)

        print(f"使用 SelectKBest 選取了 {k} 個特徵:")
        for feature, score in feature_scores[:k]:
            print(f"  - {feature}: {score:.6f}")

    return X[selected_features], selected_features

def recursive_feature_elimination(X, y, target_name=None, n_features=None, cv=5, verbose=True):
    """
    使用遞歸特徵消除(RFE)選擇特徵

    Args:
        X: 特徵資料框
        y: 標籤資料或資料框
        target_name: 目標變數名稱，如果 y 是 DataFrame，則需要指定
        n_features: 要選擇的特徵數量，預設為None (自動確定)
        cv: 交叉驗證折數，預設為5
        verbose: 是否顯示詳細信息，預設為True

    Returns:
        tuple: (選取的特徵資料框, 選取的特徵名稱列表)
    """
    if isinstance(y, pd.DataFrame):
        if target_name is None:
            raise ValueError("當 y 是 DataFrame 時，必須指定 target_name")
        y_target = y[target_name]
    else:
        y_target = y

    # 如果未指定特徵數量，預設為一半
    if n_features is None:
        n_features = max(1, X.shape[1] // 2)

    # 調整 n_features 不超過特徵數量
    n_features = min(n_features, X.shape[1])

    # 使用線性迴歸模型和RFE
    model = LinearRegression()
    rfe = RFE(model, n_features_to_select=n_features)
    rfe.fit(X, y_target)

    # 獲取選取的特徵名稱
    mask = rfe.support_
    selected_features = X.columns[mask].tolist()

    if verbose:
        # 獲取特徵排名
        ranks = rfe.ranking_
        # 將特徵名稱和排名配對
        feature_ranks = list(zip(X.columns, ranks))
        # 按排名排序
        feature_ranks.sort(key=lambda x: x[1])

        print(f"使用遞歸特徵消除選取了 {n_features} 個特徵:")
        for feature, rank in feature_ranks[:n_features]:
            print(f"  - {feature}: 排名 {rank}")

    return X[selected_features], selected_features

def plot_feature_importance(X, y, target_name=None, method='linear', top_n=10):
    """
    繪製特徵重要性圖表

    Args:
        X: 特徵資料框
        y: 標籤資料或資料框
        target_name: 目標變數名稱，如果 y 是 DataFrame，則需要指定
        method: 評估特徵重要性的方法，預設為'linear'
        top_n: 顯示前幾個重要特徵，預設為10
    """
    if isinstance(y, pd.DataFrame):
        if target_name is None:
            raise ValueError("當 y 是 DataFrame 時，必須指定 target_name")
        y_target = y[target_name]
    else:
        y_target = y

    plt.figure(figsize=(10, 6))

    if method == 'linear':
        # 使用線性迴歸係數的絕對值評估特徵重要性
        model = LinearRegression()
        model.fit(X, y_target)

        # 獲取係數並取絕對值
        importances = np.abs(model.coef_)

        # 將特徵名稱和重要性配對
        features_importance = list(zip(X.columns, importances))

        # 按重要性排序
        features_importance.sort(key=lambda x: x[1], reverse=True)

        # 取前N個重要特徵
        top_features = features_importance[:top_n]

        # 提取特徵名稱和重要性
        features = [f[0] for f in top_features]
        importance_values = [f[1] for f in top_features]

        # 繪製條形圖
        plt.barh(range(len(features)), importance_values, align='center')
        plt.yticks(range(len(features)), features)
        plt.xlabel('係數絕對值')
        plt.title(f'線性模型特徵重要性 (目標: {target_name if target_name else "y"})')

    elif method == 'f_regression':
        # 使用 f_regression 評估特徵重要性
        selector = SelectKBest(f_regression, k='all')
        selector.fit(X, y_target)

        # 獲取分數
        importances = selector.scores_

        # 將特徵名稱和重要性配對
        features_importance = list(zip(X.columns, importances))

        # 按重要性排序
        features_importance.sort(key=lambda x: x[1], reverse=True)

        # 取前N個重要特徵
        top_features = features_importance[:top_n]

        # 提取特徵名稱和重要性
        features = [f[0] for f in top_features]
        importance_values = [f[1] for f in top_features]

        # 繪製條形圖
        plt.barh(range(len(features)), importance_values, align='center')
        plt.yticks(range(len(features)), features)
        plt.xlabel('F-值')
        plt.title(f'F-統計量特徵重要性 (目標: {target_name if target_name else "y"})')

    plt.gca().invert_yaxis()  # 倒序顯示，最重要的在頂部
    plt.tight_layout()

    return plt.gcf()