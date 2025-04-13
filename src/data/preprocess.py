"""
資料預處理功能模組
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def identify_column_types(df):
    """
    識別數值型和類別型特徵

    Args:
        df: 輸入資料框

    Returns:
        tuple: (數值型特徵列表, 類別型特徵列表)
    """
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    return numeric_cols, categorical_cols


def create_preprocessing_pipeline(numeric_cols, categorical_cols):
    """
    創建預處理管道

    Args:
        numeric_cols: 數值型特徵列表
        categorical_cols: 類別型特徵列表

    Returns:
        ColumnTransformer: 預處理管道
    """
    # 數值型特徵處理管道
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 類別型特徵處理管道
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # 合併為一個預處理管道
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough'
    )

    return preprocessor


def preprocess_data(X_train, X_test):
    """
    預處理訓練集和測試集資料

    Args:
        X_train: 訓練特徵資料
        X_test: 測試特徵資料

    Returns:
        tuple: (處理後的訓練特徵, 處理後的測試特徵, 預處理管道)
    """
    # 識別數值型和類別型特徵
    numeric_cols, categorical_cols = identify_column_types(X_train)

    # 創建預處理管道
    preprocessor = create_preprocessing_pipeline(numeric_cols, categorical_cols)

    # 轉換資料
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # 如果輸出是稀疏矩陣，轉換為密集矩陣
    if hasattr(X_train_processed, "toarray"):
        X_train_processed = X_train_processed.toarray()

    if hasattr(X_test_processed, "toarray"):
        X_test_processed = X_test_processed.toarray()

    # 獲取處理後的特徵名稱
    feature_names = get_feature_names(preprocessor, X_train.columns)

    # 轉換為DataFrame
    X_train_processed_df = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_processed_df = pd.DataFrame(X_test_processed, columns=feature_names)

    return X_train_processed_df, X_test_processed_df, preprocessor


def get_feature_names(preprocessor, input_features):
    """
    獲取預處理後的特徵名稱

    Args:
        preprocessor: 預處理管道
        input_features: 輸入特徵名稱

    Returns:
        list: 預處理後的特徵名稱列表
    """
    # 使用sklearn 1.0及以上版本的get_feature_names_out方法
    try:
        feature_names = preprocessor.get_feature_names_out(input_features)
        return list(feature_names)
    except AttributeError:
        # 舊版本的sklearn需要手動處理
        col_names = []
        for name, transformer, columns in preprocessor.transformers_:
            if name == 'num':
                col_names.extend(columns)
            elif name == 'cat':
                for col in columns:
                    col_names.extend([f"{col}_{val}" for val in
                                      transformer.named_steps['onehot'].categories_[
                                          list(columns).index(col)
                                      ]])
        return col_names


def normalize_target(y_train, y_test):
    """
    標準化目標變數

    Args:
        y_train: 訓練標籤
        y_test: 測試標籤

    Returns:
        tuple: (標準化後的訓練標籤, 標準化後的測試標籤, 標準化器)
    """
    scaler = StandardScaler()

    y_train_scaled = pd.DataFrame(
        scaler.fit_transform(y_train),
        columns=y_train.columns
    )

    y_test_scaled = pd.DataFrame(
        scaler.transform(y_test),
        columns=y_test.columns
    )

    return y_train_scaled, y_test_scaled, scaler