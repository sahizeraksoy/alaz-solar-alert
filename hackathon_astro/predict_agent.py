import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# pkll.py (veya predict_agent.py)
import joblib
import numpy as np

class PredictionAgent:
    def __init__(self, model_path='solar_flare_model.pkl'):
        try:
            self.model = joblib.load(model_path)
            print("AI Modeli Başarıyla Yüklendi.")
        except Exception as e:
            print(f"Model yükleme hatası: {e}")

    def predict_kp(self, flare_class, year, month, hour):
        # Flare skoru hesaplama (X, M, C, B)
        mapping = {'X': 100, 'M': 10, 'C': 1, 'B': 0.1}
        try:
            scale = flare_class[0].upper()
            value = float(flare_class[1:])
            flare_score = mapping.get(scale, 0) * value
        except:
            flare_score = 0
            
        # Zaman özelliklerini modele uygun hale getir (Sinüs dönüşümü)
        month_sin = np.sin(2 * np.pi * month / 12)
        hour_sin = np.sin(2 * np.pi * hour / 24)
        
        # Modelin beklediği formatta X oluştur
        # ÖNEMLİ: Jupyter'deki 'features' sırasıyla aynı olmalı!
        X_input = np.array([[flare_score, month_sin, hour_sin, year]])
        
        prediction = self.model.predict(X_input)
        return prediction[0]