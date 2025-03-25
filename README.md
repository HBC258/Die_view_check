Below is an example README in English, Japanese, and Chinese that explains how to use the project.

---

# FoM Extraction and Wafer Visualization Project

This project is a Python-based tool designed to extract Figures-of-Merit (FoMs) from device measurement CSV files and to visualize the results as wafer-shot maps.

The project is divided into two parts:
- **Extraction Script (`extract_fom.py`)**: Processes CSV files, computes FoMs for each device, and outputs a summary CSV file (`FoM_summary.csv`).
- **Visualization Script (`plot_wafer.py`)**: Reads the summary CSV and displays wafer maps for selected FoM metrics.

---

## Requirements

- Python 3.x
- Packages: `pandas`, `numpy`, `matplotlib`
- CSV files should follow a specific format:
  - Column headers are in a known row (default is row 267, but can be changed).
  - Device coordinates (x, y) are included in the filename.

---

## How to Use

### English

1. **Prepare Data:**  
   Place your measurement CSV files into the designated folder (e.g., `D:\PL,\PL1`).

2. **Extract FoM Data:**  
   Run `extract_fom.py`. This script will:
   - Parse the CSV files.
   - Compute FoMs such as subthreshold slope, threshold voltage, hysteresis, and maximum transconductance.
   - Save the results into a CSV file named `FoM_summary.csv`.

3. **Visualize FoM Data:**  
   Run `plot_wafer.py`. This script will:
   - Load `FoM_summary.csv`.
   - Plot wafer maps for each FoM metric.
   - The wafer map automatically adjusts for negative coordinates (e.g., x ∈ [-5,5] and y ∈ [-4,4]) by shifting them to a zero-based grid.
   - Custom colorbar limits are applied for each metric:
     - Gm_max: 0 to 20  
     - VTH: -2 to 2  
     - Hysteresis: 0 to 2 V  
     - SS: 0.06 to 0.2 V/dec

4. **Customize Parameters:**  
   If needed, adjust parameters such as header row, device width, or grid dimensions by editing the corresponding sections in the scripts.

---

### 日本語 (Japanese)

1. **データの準備:**  
   測定結果のCSVファイルを指定フォルダ（例: `D:\PL,\PL1`）に配置してください。

2. **FoMデータの抽出:**  
   `extract_fom.py` を実行します。スクリプトは以下を行います:
   - CSVファイルのパース
   - サブスレッショルドスロープ、閾値電圧、ヒステリシス、最大トランスコンダクタンスなどのFoMを計算
   - 結果を `FoM_summary.csv` というCSVファイルに保存

3. **FoMデータの可視化:**  
   `plot_wafer.py` を実行します。スクリプトは以下を行います:
   - `FoM_summary.csv` を読み込み
   - 各FoM指標のウェハマップを表示
   - ウェハマップは負の座標（例：x ∈ [-5,5]、y ∈ [-4,4]）を自動的に0起点にシフトして表示します。
   - 各指標に対してカラーバーの範囲が設定されています:
     - Gm_max: 0～20  
     - VTH: -2～2  
     - ヒステリシス: 0～2 V  
     - SS: 0.06～0.2 V/dec

4. **パラメータのカスタマイズ:**  
   必要に応じて、ヘッダ行、デバイス幅、グリッドサイズなどのパラメータをスクリプト内で調整してください。

---

### 中文 (Chinese)

1. **准备数据:**  
   将您的测量 CSV 文件放置在指定的文件夹中（例如：`D:\PL,\PL1`）。

2. **提取 FoM 数据:**  
   运行 `extract_fom.py` 脚本。该脚本会：
   - 解析 CSV 文件
   - 计算设备的 FoM，如亚阈值斜率、阈值电压、滞后和最大跨导
   - 将结果输出到 `FoM_summary.csv` 文件中

3. **可视化 FoM 数据:**  
   运行 `plot_wafer.py` 脚本。该脚本会：
   - 读取 `FoM_summary.csv`
   - 绘制每个 FoM 指标的晶圆图
   - 晶圆图将自动处理负坐标（例如 x ∈ [-5,5] 和 y ∈ [-4,4]），通过平移将其转化为从0开始的网格
   - 每个指标都有自定义的色标范围：
     - Gm_max: 0到20  
     - VTH: -2到2  
     - 滞后: 0到2 V  
     - SS: 0.06到0.2 V/dec

4. **参数自定义:**  
   如有需要，可在脚本中调整参数，例如标题行、设备宽度或网格尺寸。

---

## Project Structure

- `extract_fom.py`: 提取 CSV 数据并计算 FoM，生成 `FoM_summary.csv`
- `plot_wafer.py`: 读取 `FoM_summary.csv` 并生成晶圆图
- `README.md`: 本文档

---

## License

