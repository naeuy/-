# Solar Energy Prediction

本專案使用 **LightGBM** 模型進行太陽能數據的訓練與預測。

## 專案結構

- **final_train.csv**: 訓練數據集，包含多個特徵欄位。
- **requirements.txt**: 專案所需的套件清單。
- **LightGBM_solar.py**: 主程式，負責數據處理、模型訓練及評估。
- **predict.py**: 主程式，負責數據處理、模型訓練及評估。
- **upload(no answer).csv**: 上傳 submit 檔案範本。
- **merged_gobalred.csv**: 外部資料全天空日射量。
- **merged_temperature.csv**: 外部資料氣溫。

## 環境設定

請確保已安裝 **Python 3.11.9** 的版本。

### 步驟 1: 安裝必要套件

執行以下指令安裝專案所需套件：

```bash
pip install -r requirements.txt
```

### 步驟 2: 準備數據

請將 final_train.csv 、merged_gobalred.csv、merged_temperature.csv 與 upload(no answer).csv 放置於專案根目錄

```bash
python LightGBM_solar.py
```

1. 加載訓練數據集 final_train.csv。
2. 進行數據清理及特徵工程。
3. 使用 LightGBM 訓練模型。
4. 輸出訓練結果。

### 步驟 3: 修改模型檔名

先將 84、85 修改加載縮放器的檔名，第 91 行的 model 名稱要修改。

### 步驟 4: 預測&生成 submit

```bash
python predict.py
```

### 步驟 5 : 上傳檔案

```bash
lightgbm_model-last.csv
```
