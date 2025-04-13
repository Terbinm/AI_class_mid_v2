import os
import pathlib


def create_project_structure(base_path="3d_printing_quality_prediction"):
    """
    創建 3D 列印品質預測專案的資料夾結構

    Args:
        base_path: 專案根目錄的路徑
    """
    # 定義要創建的資料夾結構
    directories = [
        "data/raw",
        "data/processed",
        "models/roughness",
        "models/tension_strength",
        "models/elongation",
        "notebooks",
        "src/data",
        "src/features",
        "src/models",
        "src/visualization",
        "results/figures",
        "results/tables"
    ]

    # 定義要創建的空檔案
    files = [
        "src/__init__.py",
        "src/data/__init__.py",
        "src/data/make_dataset.py",
        "src/data/preprocess.py",
        "src/features/__init__.py",
        "src/features/build_features.py",
        "src/models/__init__.py",
        "src/models/linear_models.py",
        "src/models/feature_selection.py",
        "src/models/model_evaluation.py",
        "src/visualization/__init__.py",
        "src/visualization/visualize.py",
        "requirements.txt",
        "setup.py",
        "README.md",
        "main.py",
        "notebooks/01_data_exploration.ipynb",
        "notebooks/02_feature_engineering.ipynb",
        "notebooks/03_model_comparison.ipynb"
    ]

    # 創建基本路徑
    pathlib.Path(base_path).mkdir(exist_ok=True)

    # 創建資料夾
    for directory in directories:
        full_path = os.path.join(base_path, directory)
        pathlib.Path(full_path).mkdir(parents=True, exist_ok=True)
        print(f"創建資料夾: {full_path}")

    # 創建空檔案
    for file in files:
        full_path = os.path.join(base_path, file)
        pathlib.Path(full_path).touch(exist_ok=True)
        print(f"創建檔案: {full_path}")

    # 創建原始資料檔案
    data_file = os.path.join(base_path, "data/raw/3d_printing_data.csv")
    with open(data_file, 'w', encoding='utf-8') as f:
        f.write(
            "layer_height,wall_thickness,infill_density,infill_pattern,nozzle_temperature,bed_temperature,print_speed,material,fan_speed,roughness,tension_strenght,elongation\n")
        f.write("0.02,8,90,grid,220,60,40,abs,0,25,18,1.2\n")
        f.write("0.02,7,90,honeycomb,225,65,40,abs,25,32,16,1.4\n")
        f.write("0.02,1,80,grid,230,70,40,abs,50,40,8,0.8\n")
        f.write("0.02,4,70,honeycomb,240,75,40,abs,75,68,10,0.5\n")
        f.write("0.02,6,90,grid,250,80,40,abs,100,92,5,0.7\n")
        f.write("0.02,10,40,honeycomb,200,60,40,pla,0,60,24,1.1\n")
    print(f"創建資料檔案: {data_file}")

    print("\n專案結構已成功創建！")
    print(f"專案路徑: {os.path.abspath(base_path)}")


if __name__ == "__main__":
    create_project_structure()