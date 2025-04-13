"""
特徵建立功能模組
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import VarianceThreshold


def add_polynomial_features(X, degree=2, interaction_only=False):
    """
    添加多項式特徵

    Args:
        X: 特徵資料框
        degree: 多項式次數，預設為2
        interaction_only: 是否只包含交互項，預設為False

    Returns:
        tuple: (增強後的特徵資料框, 多項式轉換器)
    """
    # 創建多項式特徵轉換器
    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)

    # 轉換特徵
    X_poly = poly.fit_transform(X)

    # 生成特徵名稱
    feature_names = poly.get_feature_names_out(X.columns)

    # 轉換為DataFrame
    X_poly_df = pd.DataFrame(X_poly, columns=feature_names)

    return X_poly_df, poly


def create_material_dummies(X):
    """
    為材料創建虛擬變數

    Args:
        X: 特徵資料框

    Returns:
        pandas.DataFrame: 包含材料虛擬變數的資料框
    """
    # 檢查材料列是否存在
    if 'material' not in X.columns:
        return X

    # 創建虛擬變數
    material_dummies = pd.get_dummies(X['material'], prefix='material')

    # 移除原始材料列並添加虛擬變數
    X_new = X.drop('material', axis=1)
    X_new = pd.concat([X_new, material_dummies], axis=1)

    return X_new


def create_pattern_dummies(X):
    """
    為填充模式創建虛擬變數

    Args:
        X: 特徵資料框

    Returns:
        pandas.DataFrame: 包含填充模式虛擬變數的資料框
    """
    # 檢查填充模式列是否存在
    if 'infill_pattern' not in X.columns:
        return X

    # 創建虛擬變數
    pattern_dummies = pd.get_dummies(X['infill_pattern'], prefix='pattern')

    # 移除原始填充模式列並添加虛擬變數
    X_new = X.drop('infill_pattern', axis=1)
    X_new = pd.concat([X_new, pattern_dummies], axis=1)

    return X_new


def add_custom_features(X):
    """
    添加自定義特徵

    Args:
        X: 特徵資料框

    Returns:
        pandas.DataFrame: 包含自定義特徵的資料框
    """
    X_new = X.copy()

    # 檢查所需列是否存在
    required_cols = ['layer_height', 'wall_thickness', 'infill_density',
                     'nozzle_temperature', 'bed_temperature', 'fan_speed']

    available_cols = [col for col in required_cols if col in X.columns]

    # 溫度差異
    if 'nozzle_temperature' in X.columns and 'bed_temperature' in X.columns:
        X_new['temp_diff'] = X['nozzle_temperature'] - X['bed_temperature']

    # 體積填充率 (層高 * 壁厚 * 填充密度)
    if 'layer_height' in X.columns and 'wall_thickness' in X.columns and 'infill_density' in X.columns:
        X_new['volume_fill_rate'] = X['layer_height'] * X['wall_thickness'] * X['infill_density']

    # 溫度與風扇速度比
    if 'nozzle_temperature' in X.columns and 'fan_speed' in X.columns:
        # 避免除以零
        X_new['temp_fan_ratio'] = X['nozzle_temperature'] / (X['fan_speed'] + 1)

    # 壁厚與填充密度比
    if 'wall_thickness' in X.columns and 'infill_density' in X.columns:
        # 避免除以零
        X_new['wall_infill_ratio'] = X['wall_thickness'] / (X['infill_density'] + 1)

    return X_new


def remove_low_variance_features(X, threshold=0.01):
    """
    移除低方差特徵

    Args:
        X: 特徵資料框
        threshold: 方差閾值，預設為0.01

    Returns:
        tuple: (移除低方差特徵後的資料框, 方差閾值選擇器)
    """
    # 創建方差閾值選擇器
    selector = VarianceThreshold(threshold=threshold)

    # 轉換特徵
    X_var = selector.fit_transform(X)

    # 獲取選取的特徵索引
    selected_features = X.columns[selector.get_support()]

    # 轉換為DataFrame
    X_var_df = pd.DataFrame(X_var, columns=selected_features)

    return X_var_df, selector


def build_all_features(X_train, X_test):
    """
    構建所有特徵

    Args:
        X_train: 訓練特徵資料框
        X_test: 測試特徵資料框

    Returns:
        tuple: (增強後的訓練特徵, 增強後的測試特徵)
    """
    # 1. 創建類別變數的虛擬變數
    X_train_dummies = create_material_dummies(X_train)
    X_train_dummies = create_pattern_dummies(X_train_dummies)

    X_test_dummies = create_material_dummies(X_test)
    X_test_dummies = create_pattern_dummies(X_test_dummies)

    # 2. 添加自定義特徵
    X_train_custom = add_custom_features(X_train_dummies)
    X_test_custom = add_custom_features(X_test_dummies)

    # 3. 添加多項式特徵
    X_train_poly, poly_transformer = add_polynomial_features(X_train_custom, degree=2)
    X_test_poly = pd.DataFrame(
        poly_transformer.transform(X_test_custom),
        columns=poly_transformer.get_feature_names_out(X_test_custom.columns)
    )

    # 4. 移除低方差特徵
    X_train_final, variance_selector = remove_low_variance_features(X_train_poly)
    X_test_final = X_test_poly.loc[:, X_train_final.columns]

    return X_train_final, X_test_final