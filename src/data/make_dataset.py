"""
資料集生成與分割模組
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(filepath):
    """
    載入原始資料

    Args:
        filepath: 資料檔案路徑

    Returns:
        pandas.DataFrame: 載入的資料
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"找不到資料檔案: {filepath}")

    return pd.read_csv(filepath)


def split_data(df, target_cols, test_size=0.2, random_state=42):
    """
    將資料分割為訓練集和測試集

    Args:
        df: 原始資料框
        target_cols: 目標變數欄位名稱列表
        test_size: 測試集佔比，預設為0.2
        random_state: 隨機種子，預設為42

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # 分割特徵和目標變數
    X = df.drop(target_cols, axis=1)
    y = df[target_cols]

    # 分割訓練集和測試集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def save_datasets(X_train, X_test, y_train, y_test, output_dir):
    """
    保存分割後的資料集

    Args:
        X_train: 訓練特徵
        X_test: 測試特徵
        y_train: 訓練標籤
        y_test: 測試標籤
        output_dir: 輸出目錄
    """
    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)

    # 保存資料集
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)

    print(f"資料集已保存至 {output_dir}")


def main(input_filepath, output_dir):
    """
    主函數：載入、分割和保存資料集

    Args:
        input_filepath: 輸入檔案路徑
        output_dir: 輸出目錄路徑
    """
    # 載入資料
    print(f"載入資料從 {input_filepath}...")
    df = load_data(input_filepath)

    # 定義目標變數
    target_cols = ['roughness', 'tension_strenght', 'elongation']

    # 分割資料
    print("分割資料為訓練集和測試集...")
    X_train, X_test, y_train, y_test = split_data(df, target_cols)

    # 保存資料集
    print("保存處理後的資料集...")
    save_datasets(X_train, X_test, y_train, y_test, output_dir)

    print("資料處理完成！")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # 設定檔案路徑
    input_filepath = "../../data/raw/3d_printing_data.csv"
    output_dir = "../../data/processed/"

    # 執行主函數
    main(input_filepath, output_dir)