"""
3D列印品質預測模型主程式
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from datetime import datetime

# 導入自定義模組
from src.data.make_dataset import load_data, split_data, save_datasets
from src.data.preprocess import preprocess_data, normalize_target
from src.features.build_features import build_all_features
from src.models.linear_models import train_all_models
from src.models.feature_selection import (
    all_in_selection, backward_selection, forward_selection, bidirectional_selection,
    plot_feature_importance
)
from src.models.model_evaluation import (
    evaluate_all_models, compare_models, plot_comparison,
    plot_actual_vs_predicted, plot_residuals
)
from src.visualization.visualize import (
    plot_correlation_matrix, plot_feature_distributions,
    plot_feature_target_relationships, plot_pca_components, plot_pca_explained_variance
)


def main():
    """主程式"""
    print("=" * 80)
    print("3D列印品質預測模型")
    print("=" * 80)

    # 設定路徑
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    raw_data_dir = os.path.join(data_dir, "raw")
    processed_data_dir = os.path.join(data_dir, "processed")
    models_dir = os.path.join(base_dir, "models")
    results_dir = os.path.join(base_dir, "results")
    figures_dir = os.path.join(results_dir, "figures")
    tables_dir = os.path.join(results_dir, "tables")

    # 確保目錄存在
    for directory in [processed_data_dir, models_dir, figures_dir, tables_dir]:
        os.makedirs(directory, exist_ok=True)

    # 步驟1: 載入資料
    print("\n步驟1: 載入資料")
    input_filepath = os.path.join(raw_data_dir, "3d_printing_data.csv")

    try:
        df = load_data(input_filepath)
        print(f"成功載入資料，共 {df.shape[0]} 筆，{df.shape[1]} 個特徵")
        print(f"資料預覽:\n{df.head()}")
    except FileNotFoundError:
        print(f"錯誤: 找不到資料檔案 {input_filepath}")
        print("請確保資料檔案已放置在 data/raw/ 目錄下")
        return

    # 定義目標變數
    target_cols = ['roughness', 'tension_strenght', 'elongation']

    # 步驟2: 資料分析
    print("\n步驟2: 資料分析")

    # 2.1 資料基本統計
    print("\n2.1 資料基本統計")
    print(df.describe())

    # 2.2 相關矩陣
    print("\n2.2 相關矩陣分析")
    corr_matrix = plot_correlation_matrix(df, target_cols=target_cols)
    corr_matrix.savefig(os.path.join(figures_dir, "correlation_matrix.png"))

    # 2.3 特徵分佈
    print("\n2.3 特徵分佈分析")
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in target_cols]

    feature_dist = plot_feature_distributions(df, categorical_cols, numeric_cols)
    feature_dist.savefig(os.path.join(figures_dir, "feature_distributions.png"))

    # 2.4 特徵與目標關係
    print("\n2.4 特徵與目標關係分析")
    feature_target_rel = plot_feature_target_relationships(df, numeric_cols, target_cols)
    feature_target_rel.savefig(os.path.join(figures_dir, "feature_target_relationships.png"))

    # 步驟3: 資料分割
    print("\n步驟3: 資料分割")
    X_train, X_test, y_train, y_test = split_data(df, target_cols, test_size=0.2, random_state=42)
    print(f"訓練集: {X_train.shape[0]} 筆，測試集: {X_test.shape[0]} 筆")

    # 保存資料集
    save_datasets(X_train, X_test, y_train, y_test, processed_data_dir)

    # 步驟4: 資料預處理
    print("\n步驟4: 資料預處理")
    X_train_processed, X_test_processed, preprocessor = preprocess_data(X_train, X_test)
    print(f"預處理後特徵數: {X_train_processed.shape[1]}")

    # 標準化目標變數
    y_train_scaled, y_test_scaled, target_scaler = normalize_target(y_train, y_test)

    # 步驟5: 特徵工程
    print("\n步驟5: 特徵工程")
    X_train_featured, X_test_featured = build_all_features(X_train_processed, X_test_processed)
    print(f"特徵工程後特徵數: {X_train_featured.shape[1]}")

    # 步驟6: 特徵選擇
    print("\n步驟6: 特徵選擇")
    feature_selection_results = {}

    # 6.1 All-in選擇
    print("\n6.1 All-in特徵選擇")
    X_train_all_in = all_in_selection(X_train_featured, y_train_scaled)
    feature_selection_results['all_in'] = X_train_all_in

    # 對每個目標變數進行特徵選擇
    for target in y_train_scaled.columns:
        print(f"\n目標變數: {target}")

        # 6.2 Backward選擇
        print(f"\n6.2 Backward特徵選擇 - {target}")
        X_train_backward, backward_features = backward_selection(
            X_train_featured, y_train_scaled, target_name=target, verbose=True
        )
        feature_selection_results[f'backward_{target}'] = X_train_backward

        # 保存選擇的特徵
        with open(os.path.join(tables_dir, f"backward_features_{target}.txt"), 'w') as f:
            f.write('\n'.join(backward_features))

        # 6.3 Forward選擇
        print(f"\n6.3 Forward特徵選擇 - {target}")
        X_train_forward, forward_features = forward_selection(
            X_train_featured, y_train_scaled, target_name=target, verbose=True
        )
        feature_selection_results[f'forward_{target}'] = X_train_forward

        # 保存選擇的特徵
        with open(os.path.join(tables_dir, f"forward_features_{target}.txt"), 'w') as f:
            f.write('\n'.join(forward_features))

        # 6.4 Bidirectional選擇
        print(f"\n6.4 Bidirectional特徵選擇 - {target}")
        X_train_bidirectional, bidirectional_features = bidirectional_selection(
            X_train_featured, y_train_scaled, target_name=target, verbose=True
        )
        feature_selection_results[f'bidirectional_{target}'] = X_train_bidirectional

        # 保存選擇的特徵
        with open(os.path.join(tables_dir, f"bidirectional_features_{target}.txt"), 'w') as f:
            f.write('\n'.join(bidirectional_features))

        # 6.5 特徵重要性
        print(f"\n6.5 特徵重要性分析 - {target}")
        importance_plot = plot_feature_importance(X_train_featured, y_train_scaled, target_name=target, method='linear')
        importance_plot.savefig(os.path.join(figures_dir, f"feature_importance_{target}.png"))

    # 步驟7: 訓練模型
    print("\n步驟7: 訓練模型")

    # 使用All-in特徵集訓練所有模型
    print("\n使用All-in特徵集訓練模型")
    all_models = train_all_models(X_train_featured, y_train_scaled, cv=5)

    # 步驟8: 模型評估
    print("\n步驟8: 模型評估")
    evaluation_results = evaluate_all_models(all_models, X_test_featured, y_test_scaled)

    # 8.1 比較模型RMSE
    print("\n8.1 比較模型RMSE")
    rmse_comparison = compare_models(evaluation_results, metric='RMSE')
    print(rmse_comparison)
    rmse_comparison.to_csv(os.path.join(tables_dir, "rmse_comparison.csv"))

    # 繪製RMSE比較圖
    rmse_plot = plot_comparison(rmse_comparison, title='模型RMSE比較 (較低值表示更好的效能)', is_lower_better=True)
    rmse_plot.savefig(os.path.join(figures_dir, "rmse_comparison.png"))

    # 8.2 比較模型R2
    print("\n8.2 比較模型R2")
    r2_comparison = compare_models(evaluation_results, metric='R2')
    print(r2_comparison)
    r2_comparison.to_csv(os.path.join(tables_dir, "r2_comparison.csv"))

    # 繪製R2比較圖
    r2_plot = plot_comparison(r2_comparison, title='模型R²比較 (較高值表示更好的效能)', is_lower_better=False)
    r2_plot.savefig(os.path.join(figures_dir, "r2_comparison.png"))

    # 8.3 詳細評估最佳模型
    print("\n8.3 詳細評估最佳模型")

    # 找出每個目標變數效能最佳的模型類型
    best_models = {}
    for target in y_train_scaled.columns:
        # 根據R2找出最佳模型
        target_r2 = {model_type: results[target]['R2'] for model_type, results in evaluation_results.items()}
        best_model_type = max(target_r2, key=target_r2.get)
        best_models[target] = (best_model_type, all_models[best_model_type][target])

        print(f"\n目標變數: {target}")
        print(f"最佳模型: {best_model_type}, R² = {target_r2[best_model_type]:.4f}")

        # 繪製實際值與預測值對比圖
        actual_vs_pred_plot = plot_actual_vs_predicted(
            best_models[target][1], X_test_featured, y_test_scaled, target_name=target
        )
        actual_vs_pred_plot.savefig(os.path.join(figures_dir, f"actual_vs_predicted_{target}.png"))

        # 繪製殘差分析圖
        residuals_plot = plot_residuals(
            best_models[target][1], X_test_featured, y_test_scaled, target_name=target
        )
        residuals_plot.savefig(os.path.join(figures_dir, f"residuals_{target}.png"))

    # 步驟9: 保存模型
    print("\n步驟9: 保存模型")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for target, (model_type, model) in best_models.items():
        model_dir = os.path.join(models_dir, target.lower())
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{model_type}_{timestamp}.joblib")
        joblib.dump(model, model_path)
        print(f"模型已保存至: {model_path}")

    # 額外保存預處理器和目標標準化器
    preprocessor_path = os.path.join(models_dir, f"preprocessor_{timestamp}.joblib")
    joblib.dump(preprocessor, preprocessor_path)
    print(f"預處理器已保存至: {preprocessor_path}")

    target_scaler_path = os.path.join(models_dir, f"target_scaler_{timestamp}.joblib")
    joblib.dump(target_scaler, target_scaler_path)
    print(f"目標標準化器已保存至: {target_scaler_path}")

    print("\n專案執行完成!")
    print(f"結果已保存至: {results_dir}")
    print(f"模型已保存至: {models_dir}")


if __name__ == "__main__":
    main()