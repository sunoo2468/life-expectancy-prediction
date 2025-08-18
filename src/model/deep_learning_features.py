import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class DeepLearningFeatures:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        
    def build_stress_mental_model(self, input_dim):
        """스트레스 및 정신건강 예측 모델"""
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(2, activation='linear')  # stress_level, mental_health_score
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def build_physical_activity_model(self, input_dim):
        """신체활동량 예측 모델"""
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.4),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(2, activation='linear')  # physical_activity, daily_steps
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def build_genetic_risk_model(self, input_dim):
        """유전적 위험도 예측 모델"""
        model = keras.Sequential([
            layers.Dense(32, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(8, activation='relu'),
            layers.Dense(1, activation='sigmoid')  # genetic_risk (0-1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_stress_mental_model(self, X_train, y_train, X_val, y_val):
        """스트레스 및 정신건강 모델 훈련"""
        print("스트레스 및 정신건강 예측 모델 훈련 중...")
        
        # 데이터 스케일링
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_val_scaled = scaler_X.transform(X_val)
        y_train_scaled = scaler_y.fit_transform(y_train)
        y_val_scaled = scaler_y.transform(y_val)
        
        # 모델 생성
        model = self.build_stress_mental_model(X_train.shape[1])
        
        # 조기 종료 콜백
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # 모델 훈련
        history = model.fit(
            X_train_scaled, y_train_scaled,
            validation_data=(X_val_scaled, y_val_scaled),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # 모델 저장
        self.models['stress_mental'] = model
        self.scalers['stress_mental_X'] = scaler_X
        self.scalers['stress_mental_y'] = scaler_y
        
        return history
    
    def train_physical_activity_model(self, X_train, y_train, X_val, y_val):
        """신체활동량 예측 모델 훈련"""
        print("신체활동량 예측 모델 훈련 중...")
        
        # 데이터 스케일링
        scaler_X = StandardScaler()
        scaler_y = MinMaxScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_val_scaled = scaler_X.transform(X_val)
        y_train_scaled = scaler_y.fit_transform(y_train)
        y_val_scaled = scaler_y.transform(y_val)
        
        # 모델 생성
        model = self.build_physical_activity_model(X_train.shape[1])
        
        # 조기 종료 콜백
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        # 모델 훈련
        history = model.fit(
            X_train_scaled, y_train_scaled,
            validation_data=(X_val_scaled, y_val_scaled),
            epochs=150,
            batch_size=64,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # 모델 저장
        self.models['physical_activity'] = model
        self.scalers['physical_activity_X'] = scaler_X
        self.scalers['physical_activity_y'] = scaler_y
        
        return history
    
    def train_genetic_risk_model(self, X_train, y_train, X_val, y_val):
        """유전적 위험도 예측 모델 훈련"""
        print("유전적 위험도 예측 모델 훈련 중...")
        
        # 데이터 스케일링
        scaler_X = StandardScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_val_scaled = scaler_X.transform(X_val)
        
        # 모델 생성
        model = self.build_genetic_risk_model(X_train.shape[1])
        
        # 조기 종료 콜백
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=20,
            restore_best_weights=True
        )
        
        # 모델 훈련
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=200,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # 모델 저장
        self.models['genetic_risk'] = model
        self.scalers['genetic_risk_X'] = scaler_X
        
        return history
    
    def predict_stress_mental(self, X):
        """스트레스 및 정신건강 예측"""
        if 'stress_mental' not in self.models:
            raise ValueError("스트레스 및 정신건강 모델이 훈련되지 않았습니다.")
        
        model = self.models['stress_mental']
        scaler_X = self.scalers['stress_mental_X']
        scaler_y = self.scalers['stress_mental_y']
        
        X_scaled = scaler_X.transform(X)
        y_pred_scaled = model.predict(X_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        
        return y_pred
    
    def predict_physical_activity(self, X):
        """신체활동량 예측"""
        if 'physical_activity' not in self.models:
            raise ValueError("신체활동량 모델이 훈련되지 않았습니다.")
        
        model = self.models['physical_activity']
        scaler_X = self.scalers['physical_activity_X']
        scaler_y = self.scalers['physical_activity_y']
        
        X_scaled = scaler_X.transform(X)
        y_pred_scaled = model.predict(X_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        
        return y_pred
    
    def predict_genetic_risk(self, X):
        """유전적 위험도 예측"""
        if 'genetic_risk' not in self.models:
            raise ValueError("유전적 위험도 모델이 훈련되지 않았습니다.")
        
        model = self.models['genetic_risk']
        scaler_X = self.scalers['genetic_risk_X']
        
        X_scaled = scaler_X.transform(X)
        y_pred = model.predict(X_scaled)
        
        return y_pred
    
    def save_models(self, filepath_prefix):
        """모든 모델 저장"""
        for name, model in self.models.items():
            model.save(f"{filepath_prefix}_{name}.h5")
        
        # 스케일러 저장
        import joblib
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, f"{filepath_prefix}_{name}_scaler.pkl")
        
        print("모든 딥러닝 모델이 저장되었습니다.")
    
    def load_models(self, filepath_prefix):
        """모든 모델 로드"""
        import joblib
        
        # 모델 로드
        model_names = ['stress_mental', 'physical_activity', 'genetic_risk']
        for name in model_names:
            try:
                self.models[name] = keras.models.load_model(f"{filepath_prefix}_{name}.h5")
            except:
                print(f"모델 {name}을 로드할 수 없습니다.")
        
        # 스케일러 로드
        scaler_names = ['stress_mental_X', 'stress_mental_y', 
                       'physical_activity_X', 'physical_activity_y',
                       'genetic_risk_X']
        for name in scaler_names:
            try:
                self.scalers[name] = joblib.load(f"{filepath_prefix}_{name}_scaler.pkl")
            except:
                print(f"스케일러 {name}을 로드할 수 없습니다.")
        
        print("딥러닝 모델들이 로드되었습니다.")

