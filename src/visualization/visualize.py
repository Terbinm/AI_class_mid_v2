"""
視覺化功能模組
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def set_plotting_style():
    """設置繪圖樣式"""
    sns.set(style="whitegrid")
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']  # 支援中文顯示
    plt.rcParams['axes.unicode_minus'] = False  # 正確顯示負號


def plot_correlation_matrix(df, target_cols=None, figsize=(12, 10)):
    """
    繪製相關矩陣熱圖

    Args:
        df: 資料框
        target_cols: 目標變數欄位名稱列表，這些欄位會在熱圖中標示
        figsize: 圖表大小，預設為(12, 10)
    """
    set_plotting_style()

    # 只選擇數值型欄位計算相關矩陣
    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    # 計算相關矩陣
    corr = numeric_df.corr()

    # 設置圖表大小
    plt.figure(figsize=figsize)

    # 繪製熱圖
    mask = np.triu(np.ones_like(corr, dtype=bool))  # 遮罩上三角形
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=0.5, annot=True, fmt=".2f", cbar_kws={"shrink": .5})

    # 如果指定了目標變數，標示它們
    if target_cols:
        for col in target_cols:
            if col in numeric_df.columns:
                plt.axhline(y=numeric_df.columns.get_loc(col) + 0.5, color='black', linestyle='--', linewidth=1)
                plt.axvline(x=numeric_df.columns.get_loc(col) + 0.5, color='black', linestyle='--', linewidth=1)

    plt.title('特徵相關矩陣')
    plt.tight_layout()

    return plt.gcf()

def plot_feature_distributions(df, categorical_cols=None, continuous_cols=None, figsize=(15, 10)):
    """
    繪製特徵分佈圖

    Args:
        df: 資料框
        categorical_cols: 類別型特徵列表
        continuous_cols: 連續型特徵列表
        figsize: 圖表大小，預設為(15, 10)
    """
    set_plotting_style()

    # 如果未指定類別型和連續型特徵，嘗試自動識別
    if categorical_cols is None and continuous_cols is None:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        continuous_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # 計算需要的列數和行數
    n_features = len(categorical_cols or []) + len(continuous_cols or [])
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    # 設置圖表大小
    plt.figure(figsize=figsize)

    plot_idx = 1

    # 繪製類別型特徵分佈
    if categorical_cols:
        for col in categorical_cols:
            plt.subplot(n_rows, n_cols, plot_idx)

            # 計算每個類別的頻率
            value_counts = df[col].value_counts()

            # 繪製條形圖
            sns.barplot(x=value_counts.index, y=value_counts.values)
            plt.title(f'{col} 分佈')
            plt.xticks(rotation=45)
            plt.tight_layout()

            plot_idx += 1

    # 繪製連續型特徵分佈
    if continuous_cols:
        for col in continuous_cols:
            plt.subplot(n_rows, n_cols, plot_idx)

            # 繪製直方圖和密度曲線
            sns.histplot(df[col], kde=True)
            plt.title(f'{col} 分佈')
            plt.tight_layout()

            plot_idx += 1

    return plt.gcf()


def plot_scatter_matrix(df, target_cols=None, features_to_plot=None, figsize=(15, 15)):
    """
    繪製散點矩陣

    Args:
        df: 資料框
        target_cols: 目標變數欄位名稱列表
        features_to_plot: 要繪製的特徵列表，如果未指定，會自動選擇與目標變數相關性最高的特徵
        figsize: 圖表大小，預設為(15, 15)
    """
    set_plotting_style()

    # 如果未指定要繪製的特徵，且指定了目標變數，選擇與目標變數相關性最高的特徵
    if features_to_plot is None and target_cols:
        # 計算所有特徵與目標變數的相關性
        correlations = {}
        for target in target_cols:
            if target in df.columns:
                numeric_cols = df.drop(target_cols, axis=1).select_dtypes(include=['int64', 'float64']).columns
                target_correlations = df[numeric_cols].corrwith(df[target]).abs().sort_values(ascending=False)
                for col, corr in target_correlations.items():
                    if col not in correlations:
                        correlations[col] = corr
                    else:
                        correlations[col] = max(correlations[col], corr)

        # 選擇相關性最高的前5個特徵
        features_to_plot = [col for col, _ in sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:5]]

    # 如果還是未指定要繪製的特徵，使用所有數值型特徵
    if features_to_plot is None:
        features_to_plot = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if target_cols:
            features_to_plot = [col for col in features_to_plot if col not in target_cols]

    # 限制特徵數量，避免圖表過大
    if len(features_to_plot) > 5:
        features_to_plot = features_to_plot[:5]

    # 創建用於繪圖的資料框
    plot_df = df[features_to_plot + (target_cols or [])]

    # 設置圖表
    plt.figure(figsize=figsize)

    # 繪製散點矩陣
    if target_cols and len(target_cols) == 1:
        # 單一目標變數，使用顏色標記
        scatter_matrix = sns.pairplot(plot_df, hue=target_cols[0], diag_kind='kde')
        plt.suptitle('特徵散點矩陣 (顏色表示目標變數)', y=1.02)
    else:
        # 多個或沒有目標變數
        scatter_matrix = sns.pairplot(plot_df, diag_kind='kde')
        plt.suptitle('特徵散點矩陣', y=1.02)

    plt.tight_layout()

    return scatter_matrix.fig


def plot_feature_target_relationships(df, feature_cols, target_cols, figsize=(15, 12)):
    """
    繪製特徵與目標變數的關係圖

    Args:
        df: 資料框
        feature_cols: 特徵欄位名稱列表
        target_cols: 目標變數欄位名稱列表
        figsize: 圖表大小，預設為(15, 12)
    """
    set_plotting_style()

    # 限制特徵數量，避免圖表過大
    if len(feature_cols) > 5:
        # 選擇與目標變數相關性最高的前5個特徵
        correlations = {}
        for target in target_cols:
            if target in df.columns:
                target_correlations = df[feature_cols].corrwith(df[target]).abs()
                for col, corr in target_correlations.items():
                    if col not in correlations:
                        correlations[col] = corr
                    else:
                        correlations[col] = max(correlations[col], corr)

        feature_cols = [col for col, _ in sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:5]]

    # 計算需要的列數和行數
    n_features = len(feature_cols)
    n_targets = len(target_cols)
    n_plots = n_features * n_targets
    n_cols = min(n_targets, 3)
    n_rows = (n_plots + n_cols - 1) // n_cols

    # 設置圖表大小
    plt.figure(figsize=figsize)

    plot_idx = 1

    # 繪製每個特徵與每個目標變數的關係
    for feature in feature_cols:
        for target in target_cols:
            plt.subplot(n_rows, n_cols, plot_idx)

            # 檢查特徵類型
            if df[feature].dtype in ['object', 'category']:
                # 類別型特徵，使用箱形圖
                sns.boxplot(x=df[feature], y=df[target])
                plt.title(f'{feature} vs {target}')
                plt.xticks(rotation=45)
            else:
                # 連續型特徵，使用散點圖和迴歸線
                sns.regplot(x=df[feature], y=df[target], scatter_kws={'alpha': 0.5})
                plt.title(f'{feature} vs {target}')

            plt.tight_layout()
            plot_idx += 1

    return plt.gcf()


def plot_pca_components(X, n_components=2, target=None, target_names=None, figsize=(10, 8)):
    """
    繪製PCA主成分分析結果

    Args:
        X: 特徵資料
        n_components: 主成分數量，預設為2
        target: 目標變數，用於著色，預設為None
        target_names: 目標變數類別名稱，預設為None
        figsize: 圖表大小，預設為(10, 8)
    """
    set_plotting_style()

    # 標準化資料
    X_std = StandardScaler().fit_transform(X)

    # 執行PCA
    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(X_std)

    # 創建主成分資料框
    column_names = [f'Principal Component {i + 1}' for i in range(n_components)]
    principalDf = pd.DataFrame(data=principalComponents, columns=column_names)

    # 如果提供了目標變數，添加到資料框
    if target is not None:
        finalDf = pd.concat([principalDf, target.reset_index(drop=True)], axis=1)
        target_col = target.name if hasattr(target, 'name') else 'Target'
    else:
        finalDf = principalDf
        target_col = None

    # 繪製結果
    plt.figure(figsize=figsize)

    if n_components == 2:
        # 2D圖
        if target_col:
            # 使用目標變數著色
            if target_names:
                # 使用提供的類別名稱
                for i, name in enumerate(target_names):
                    plt.scatter(finalDf.loc[finalDf[target_col] == i, 'Principal Component 1'],
                                finalDf.loc[finalDf[target_col] == i, 'Principal Component 2'],
                                label=name)
            else:
                # 使用目標變數值作為類別
                targets = finalDf[target_col].unique()
                for target_value in targets:
                    plt.scatter(finalDf.loc[finalDf[target_col] == target_value, 'Principal Component 1'],
                                finalDf.loc[finalDf[target_col] == target_value, 'Principal Component 2'],
                                label=target_value)

            plt.legend()
        else:
            # 不使用著色
            plt.scatter(finalDf['Principal Component 1'], finalDf['Principal Component 2'])

        plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%})')

    elif n_components == 3:
        # 3D圖
        from mpl_toolkits.mplot3d import Axes3D

        ax = plt.figure(figsize=figsize).add_subplot(111, projection='3d')

        if target_col:
            # 使用目標變數著色
            if target_names:
                # 使用提供的類別名稱
                for i, name in enumerate(target_names):
                    ax.scatter(finalDf.loc[finalDf[target_col] == i, 'Principal Component 1'],
                               finalDf.loc[finalDf[target_col] == i, 'Principal Component 2'],
                               finalDf.loc[finalDf[target_col] == i, 'Principal Component 3'],
                               label=name)
            else:
                # 使用目標變數值作為類別
                targets = finalDf[target_col].unique()
                for target_value in targets:
                    ax.scatter(finalDf.loc[finalDf[target_col] == target_value, 'Principal Component 1'],
                               finalDf.loc[finalDf[target_col] == target_value, 'Principal Component 2'],
                               finalDf.loc[finalDf[target_col] == target_value, 'Principal Component 3'],
                               label=target_value)

            plt.legend()
        else:
            # 不使用著色
            ax.scatter(finalDf['Principal Component 1'],
                       finalDf['Principal Component 2'],
                       finalDf['Principal Component 3'])

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%})')

    # 添加標題，顯示總解釋方差
    total_var = sum(pca.explained_variance_ratio_[:n_components])
    plt.title(f'PCA 主成分分析 (總解釋變異: {total_var:.2%})')
    plt.tight_layout()

    return plt.gcf(), pca


def plot_pca_explained_variance(X, n_components=None, figsize=(10, 6)):
    """
    繪製PCA解釋變異圖

    Args:
        X: 特徵資料
        n_components: 主成分數量，預設為None (自動確定)
        figsize: 圖表大小，預設為(10, 6)
    """
    set_plotting_style()

    # 標準化資料
    X_std = StandardScaler().fit_transform(X)

    # 如果未指定主成分數量，使用特徵數量
    if n_components is None:
        n_components = min(X.shape[0], X.shape[1])

    # 執行PCA
    pca = PCA(n_components=n_components)
    pca.fit(X_std)

    # 計算累積解釋變異
    cumsum = np.cumsum(pca.explained_variance_ratio_)

    # 繪製解釋變異
    plt.figure(figsize=figsize)

    # 繪製變異比率
    plt.bar(range(1, n_components + 1), pca.explained_variance_ratio_, alpha=0.5, label='個別變異')
    plt.step(range(1, n_components + 1), cumsum, where='mid', label='累積變異')

    # 添加註釋
    for i, var in enumerate(pca.explained_variance_ratio_):
        plt.text(i + 1, var + 0.01, f'{var:.2%}', ha='center')

    plt.axhline(y=0.95, color='r', linestyle='--', label='95% 閾值')

    # 找到達到95%變異的主成分數量
    n_components_95 = np.argmax(cumsum >= 0.95) + 1
    plt.text(n_components_95, 0.96, f'需要 {n_components_95} 個主成分\n達到 95% 變異',
             ha='center', va='bottom', bbox=dict(boxstyle='round', alpha=0.1))

    plt.xlabel('主成分數量')
    plt.ylabel('解釋變異比率')
    plt.title('PCA 解釋變異')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    return plt.gcf(), pca


def plot_pca_loadings(pca, feature_names, n_components=2, top_n=10, figsize=(12, 8)):
    """
    繪製PCA載荷圖

    Args:
        pca: 已擬合的PCA模型
        feature_names: 特徵名稱列表
        n_components: 要顯示的主成分數量，預設為2
        top_n: 每個主成分顯示的頂部特徵數量，預設為10
        figsize: 圖表大小，預設為(12, 8)
    """
    set_plotting_style()

    # 獲取載荷
    loadings = pca.components_

    # 限制主成分數量
    n_components = min(n_components, loadings.shape[0])

    # 設置圖表
    plt.figure(figsize=figsize)

    # 繪製每個主成分的載荷
    for i in range(n_components):
        plt.subplot(n_components, 1, i + 1)

        # 獲取此主成分的載荷
        component_loadings = loadings[i]

        # 配對特徵名稱和載荷
        feature_loading_pairs = list(zip(feature_names, component_loadings))

        # 按載荷絕對值排序
        feature_loading_pairs.sort(key=lambda x: abs(x[1]), reverse=True)

        # 取前N個特徵
        top_features = feature_loading_pairs[:top_n]

        # 提取特徵名稱和載荷
        names, values = zip(*top_features)

        # 繪製水平條形圖
        bars = plt.barh(range(len(names)), values, align='center')

        # 根據載荷正負設置顏色
        for j, value in enumerate(values):
            bars[j].set_color('green' if value > 0 else 'red')

        plt.yticks(range(len(names)), names)
        plt.title(f'主成分 {i + 1} 載荷 (解釋變異: {pca.explained_variance_ratio_[i]:.2%})')
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        plt.xlim(-1, 1)
        plt.xlabel('載荷值')
        plt.grid(True, linestyle='--', alpha=0.7)

        # 添加註釋
        plt.annotate('正相關', xy=(0.9, 0.9), xycoords='axes fraction', color='green')
        plt.annotate('負相關', xy=(0.1, 0.9), xycoords='axes fraction', color='red')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.suptitle('PCA 主成分載荷', fontsize=16)

    return plt.gcf()