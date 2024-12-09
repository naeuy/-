import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime

# 設置隨機種子，確保模型訓練的可重複性
def set_seed(seed=42):
    """設置隨機種子以確保結果可重現"""
    np.random.seed(seed)

set_seed(42)

# -----------------------
# 1. 數據讀取與預處理
# -----------------------
def load_and_clean_data(file_path):
    """
    讀取 CSV 數據，並進行清理和類型轉換
    """
    data = pd.read_csv(file_path)

    # 確保數值型欄位為正確格式
    data['Power(mW)'] = pd.to_numeric(data['Power(mW)'], errors='coerce')
    data['Temperature'] = pd.to_numeric(data['Temperature'], errors='coerce')
    data['gobalred'] = pd.to_numeric(data['gobalred'], errors='coerce')

    # 檢查是否存在缺失值
    if data.isnull().sum().sum() > 0:
        data = data.dropna()  # 丟棄含有缺失值的行
    return data

# 設定檔案路徑並讀取數據
file_path = "train.csv"
data = load_and_clean_data(file_path)

# -----------------------
# 2. 特徵選擇與縮放
# -----------------------
def scale_features_and_target(data, feature_columns, target_column):
    """
    縮放特徵和目標變量到 [0, 1] 範圍
    """
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X = data[feature_columns]
    y = data[target_column]

    # 特徵和目標縮放
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

    return X_scaled, y_scaled, scaler_X, scaler_y

# 選擇特徵和目標欄位
feature_columns = ['LocationCode', 'Month', 'Day', 'Hours', 'Minute', 
                   'Temperature', 'gobalred', 'MinutesOfDay']
target_column = 'Power(mW)'

X_scaled, y_scaled, scaler_X, scaler_y = scale_features_and_target(data, feature_columns, target_column)

# -----------------------
# 3. 模型訓練
# -----------------------
def train_lightgbm_model(X_scaled, y_scaled, params, num_round):
    """
    使用 LightGBM 訓練模型
    """
    train_data = lgb.Dataset(X_scaled, label=y_scaled)
    model = lgb.train(params, train_data, num_boost_round=num_round)
    return model

# LightGBM 模型參數
lgb_params = {
    'objective': 'tweedie', 
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42,
'learning_rate': 0.04687770911890441, 
'num_leaves': 255, 
'max_depth': 14,
'feature_fraction': 0.9167703585856889,
 'bagging_fraction': 0.8772961453227072, 
 'lambda_l1': 2.012803236786471,
'lambda_l2': 3.281644364763694
}

# 訓練模型
num_round = 3180
model = train_lightgbm_model(X_scaled, y_scaled, lgb_params, num_round)

# -----------------------
# 4. 模型評估
# -----------------------
def evaluate_model(model, X_scaled, y_scaled, scaler_y):
    """
    使用訓練好的模型進行預測並計算評估指標
    """
    y_pred_scaled = model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_true = scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).ravel()

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return y_true, y_pred, mse, mae, r2

# 模型評估
y_true, y_pred, mse, mae, r2 = evaluate_model(model, X_scaled, y_scaled, scaler_y)

# 輸出評估結果
print(f"均方誤差 (MSE): {mse:.4f}")
print(f"平均絕對誤差 (MAE): {mae:.4f}")
print(f"R-squared (R2): {r2:.4f}")

# -----------------------
# 5. 模型保存
# -----------------------
def save_model_and_scalers(model, scaler_X, scaler_y, model_dir="./models/"):
    """
    保存訓練好的模型和縮放器到本地文件
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(model_dir, exist_ok=True)

    # 保存模型
    model_path = os.path.join(model_dir, f"lightgbm_model_{timestamp}.txt")
    model.save_model(model_path)

    # 保存縮放器
    scaler_X_path = os.path.join(model_dir, f"scaler_X_{timestamp}.pkl")
    scaler_y_path = os.path.join(model_dir, f"scaler_y_{timestamp}.pkl")

    joblib.dump(scaler_X, scaler_X_path)
    joblib.dump(scaler_y, scaler_y_path)

    print(f"模型已保存到: {model_path}")
    print(f"縮放器已保存到: {scaler_X_path} 和 {scaler_y_path}")

save_model_and_scalers(model, scaler_X, scaler_y)

# -----------------------
# 6. 特徵重要性分析
# -----------------------
# def plot_feature_importance(model, feature_names, save_path):
#     """
#     繪製並保存特徵重要性圖
#     """
#     feature_importance = model.feature_importance()
#     importance_df = pd.DataFrame({
#         'feature': feature_names,
#         'importance': feature_importance
#     }).sort_values('importance', ascending=False)

#     plt.figure(figsize=(10, 6))
#     importance_df.plot(x='feature', y='importance', kind='bar', legend=False)
#     plt.title("LightGBM 模型特徵重要性")
#     plt.xlabel("特徵")
#     plt.ylabel("重要性")
#     plt.tight_layout()
#     plt.savefig(save_path)
#     print(f"特徵重要性圖已保存到: {save_path}")
#     return importance_df

# feature_importance_path = os.path.join("./models/", f"lightgbm_feature_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
# importance_df = plot_feature_importance(model, feature_columns, feature_importance_path)

# print("特徵重要性分析完成:")
# print(importance_df)
