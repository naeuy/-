import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from datetime import datetime

# -----------------------
# 1. 數據讀取與處理
# -----------------------

def load_and_process_data(file_path):
    """
    讀取數據並將 "序號" 拆解為模型輸入欄位
    """
    data = pd.read_csv(file_path)
    
    # 確保 "序號" 是字符串格式
    data['序號'] = data['序號'].astype(str)
    
    # 拆解 "序號" 成各個欄位
    data['Year'] = data['序號'].str[:4].astype(int)
    data['Month'] = data['序號'].str[4:6].astype(int)
    data['Day'] = data['序號'].str[6:8].astype(int)
    data['Hours'] = data['序號'].str[8:10].astype(int)
    data['Minute'] = data['序號'].str[10:12].astype(int)
    data['LocationCode'] = data['序號'].str[12:14].astype(int)
    
    # 新增一天內的第幾分鐘欄位
    data['MinutesOfDay'] = data['Hours'] * 60 + data['Minute']
    
    # 確認必要欄位是否存在
    required_columns = ["Year", "Month", "Day", "Hours", "Minute"]
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")
    
    return data

# -----------------------
# 2. 溫度與全球輻照數據處理
# -----------------------

def load_time_based_data(file_path):
    """
    加載溫度或全球輻照數據，並解析日期時間
    """
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['Year'] = df['datetime'].dt.year
    df['Month'] = df['datetime'].dt.month
    df['Day'] = df['datetime'].dt.day
    df['Hour'] = df['datetime'].dt.hour
    return df

def get_time_based_value(df, year, month, day, hour, column_name):
    """
    根據年份、月份、日期和小時查找對應欄位值
    """
    result = df[(df['Year'] == year) & 
                (df['Month'] == month) & 
                (df['Day'] == day) & 
                (df['Hour'] == hour)]
    if not result.empty:
        return result.iloc[0][column_name]
    return None

def enrich_data_with_time_based_values(data, temperature_file, gobalred_file):
    """
    為數據添加對應的溫度和全球輻照數據
    """
    temperature_data = load_time_based_data(temperature_file)
    gobalred_data = load_time_based_data(gobalred_file)
    
    temp = []
    gobalred = []
    
    for i in range(len(data)):
        year, month, day, hour = data.loc[i, ['Year', 'Month', 'Day', 'Hours']]
        temp.append(get_time_based_value(temperature_data, year, month, day, hour, 'temperature'))
        gobalred.append(get_time_based_value(gobalred_data, year, month, day, hour, 'merged_gobalred'))
    
    data['Temperature'] = temp
    data['gobalred'] = gobalred
    return data

# -----------------------
# 3. 特徵縮放與模型加載
# -----------------------

def scale_features(data, scaler_path):
    """
    縮放特徵數據
    """
    scaler = joblib.load(scaler_path)
    scaled_data = scaler.transform(data)
    return scaled_data, scaler

def load_lightgbm_model(model_path):
    """
    加載 LightGBM 模型
    """
    return lgb.Booster(model_file=model_path)

# -----------------------
# 4. 預測與保存
# -----------------------

def predict_and_save(data, model, scaler_y, feature_columns, output_path):
    """
    使用模型進行預測，並保存結果到文件
    """
    # 特徵縮放後的預測
    input_scaled = data[feature_columns]
    y_pred_scaled = model.predict(input_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    
    # 將結果保存到原始數據
    data['答案'] = y_pred.round(2)
    data[['序號', '答案']].to_csv(output_path, index=False)
    print(f"預測結果已保存至: {output_path}")

# -----------------------
# 主程序執行流程
# -----------------------

if __name__ == "__main__":
    # 輸入文件路徑
    input_file = "upload.csv"
    temperature_file = "all_temperature.csv"
    gobalred_file = "all_gobalred.csv"
    model_dir = "models"
    model_file = os.path.join(model_dir, "lightgbm_model_20241209_232214.txt")
    scaler_X_path = os.path.join(model_dir, "scaler_X_20241209_232214.pkl")
    scaler_y_path = os.path.join(model_dir, "scaler_y_20241209_232214.pkl")
    output_file = "lightgbm_model.csv"
    
    # 1. 讀取並處理數據
    data = load_and_process_data(input_file)
    
    # 2. 添加溫度和全球輻照數據
    data = enrich_data_with_time_based_values(data, temperature_file, gobalred_file)
    
    # 3. 加載縮放器並縮放特徵
    feature_columns = ['LocationCode', 'Month', 'Day', 'Hours', 'Minute', 'Temperature', 'gobalred', 'MinutesOfDay']
    input_scaled, scaler_X = scale_features(data[feature_columns], scaler_X_path)
    
    # 4. 加載模型
    model = load_lightgbm_model(model_file)
    scaler_y = joblib.load(scaler_y_path)
    
    # 5. 預測並保存結果
    predict_and_save(data, model, scaler_y, input_scaled, output_file)
