import numpy as np
import joblib
from xgboost import XGBClassifier
import os

XGB_MODEL_PATH = "xgb_model.pkl"
TARGET_DIM = 10

# 加载或训练 XGBoost
def load_or_train_xgb():
    if os.path.exists(XGB_MODEL_PATH):
        print("Load an existing XGBoost model")
        xgb = joblib.load(XGB_MODEL_PATH)
    else:
        print("XGBoost file not found, start training XGBoost")
        X_train = np.random.rand(100, TARGET_DIM)
        y_train = np.random.randint(0, 2, 100)
        xgb = XGBClassifier()
        xgb.fit(X_train, y_train)
        joblib.dump(xgb, XGB_MODEL_PATH)
    return xgb

xgb_model = load_or_train_xgb()

# 预测函数
def predict(vector):
    prediction = xgb_model.predict(vector)
    return int(prediction[0])
