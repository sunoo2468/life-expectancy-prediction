#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
import warnings
warnings.filterwarnings('ignore')

from model.deep_learning_features import DeepLearningFeatures
from model.life_expectancy_prediction_model import LifeExpectancyPredictionModel

class EnhancedDeepLearningSystem:
    """딥러닝 시스템"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.histories = {}
        self.performance_metrics = {}
        
    def build_advanced_stress_model(self, input_dim):
        """스트레스 및 정신건강 예측 모델"""
        # 입력 레이어
        input_layer = layers.Input(shape=(input_dim,))
        
        # 특성 추출 레이어
        feature_extraction = layers.Dense(128, activation='relu', 
                                        kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(input_layer)
        feature_extraction = layers.BatchNormalization()(feature_extraction)
        feature_extraction = layers.Dropout(0.3)(feature_extraction)
        
        # 병렬 처리 레이어 (스트레스와 정신건강을 독립적으로 처리)
        stress_branch = layers.Dense(64, activation='relu')(feature_extraction)
        stress_branch = layers.Dropout(0.2)(stress_branch)
        stress_branch = layers.Dense(32, activation='relu')(stress_branch)
        stress_output = layers.Dense(1, activation='linear', name='stress_level')(stress_branch)
        
        mental_branch = layers.Dense(64, activation='relu')(feature_extraction)
        mental_branch = layers.Dropout(0.2)(mental_branch)
        mental_branch = layers.Dense(32, activation='relu')(mental_branch)
        mental_output = layers.Dense(1, activation='linear', name='mental_health')(mental_branch)
        
        # 모델 생성
        model = keras.Model(inputs=input_layer, outputs=[stress_output, mental_output])
        
        # 컴파일
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={'stress_level': 'mse', 'mental_health': 'mse'},
            loss_weights={'stress_level': 0.6, 'mental_health': 0.4},
            metrics={'stress_level': 'mae', 'mental_health': 'mae'}
        )
        
        return model
    
    def build_advanced_physical_activity_model(self, input_dim):
        """신체활동량 예측 모델 (신경망)"""
        # 입력 레이어
        input_layer = layers.Input(shape=(input_dim,))
        
        # 특성 임베딩
        embedding = layers.Dense(256, activation='relu')(input_layer)
        embedding = layers.BatchNormalization()(embedding)
        embedding = layers.Dropout(0.4)(embedding)
        
        # 병렬 처리 레이어 (신체활동과 걸음수를 독립적으로 처리)
        physical_branch = layers.Dense(128, activation='relu')(embedding)
        physical_branch = layers.BatchNormalization()(physical_branch)
        physical_branch = layers.Dropout(0.3)(physical_branch)
        physical_branch = layers.Dense(64, activation='relu')(physical_branch)
        physical_branch = layers.Dropout(0.2)(physical_branch)
        physical_branch = layers.Dense(32, activation='relu')(physical_branch)
        physical_activity_output = layers.Dense(1, activation='linear', name='physical_activity')(physical_branch)
        
        steps_branch = layers.Dense(128, activation='relu')(embedding)
        steps_branch = layers.BatchNormalization()(steps_branch)
        steps_branch = layers.Dropout(0.3)(steps_branch)
        steps_branch = layers.Dense(64, activation='relu')(steps_branch)
        steps_branch = layers.Dropout(0.2)(steps_branch)
        steps_branch = layers.Dense(32, activation='relu')(steps_branch)
        daily_steps_output = layers.Dense(1, activation='linear', name='daily_steps')(steps_branch)
        
        # 모델 생성
        model = keras.Model(inputs=input_layer, outputs=[physical_activity_output, daily_steps_output])
        
        # 컴파일
        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss={'physical_activity': 'mse', 'daily_steps': 'mse'},
            loss_weights={'physical_activity': 0.5, 'daily_steps': 0.5},
            metrics={'physical_activity': 'mae', 'daily_steps': 'mae'}
        )
        
        return model
    
    def build_advanced_genetic_risk_model(self, input_dim):
        """유전적 위험도 예측 모델 (Attention Mechanism)"""
        # 입력 레이어
        input_layer = layers.Input(shape=(input_dim,))
        
        # 특성 임베딩
        embedding = layers.Dense(128, activation='relu')(input_layer)
        embedding = layers.BatchNormalization()(embedding)
        embedding = layers.Dropout(0.3)(embedding)
        
        # Attention Mechanism
        attention = layers.Dense(64, activation='tanh')(embedding)
        attention = layers.Dense(1, activation='softmax')(attention)
        attention_output = layers.Multiply()([embedding, attention])
        
        # 병렬 처리 레이어
        branch1 = layers.Dense(64, activation='relu')(attention_output)
        branch1 = layers.Dropout(0.2)(branch1)
        
        branch2 = layers.Dense(64, activation='relu')(attention_output)
        branch2 = layers.Dropout(0.2)(branch2)
        
        # 결합
        combined = layers.Concatenate()([branch1, branch2])
        combined = layers.Dense(32, activation='relu')(combined)
        combined = layers.Dropout(0.2)(combined)
        
        # 출력 레이어
        output = layers.Dense(1, activation='sigmoid', name='genetic_risk')(combined)
        
        # 모델 생성
        model = keras.Model(inputs=input_layer, outputs=output)
        
        # 컴파일
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def build_life_expectancy_direct_model(self, input_dim):
        """직접적인 수명 예측 딥러닝 모델"""
        # 입력 레이어
        input_layer = layers.Input(shape=(input_dim,))
        
        # 특성 추출
        x = layers.Dense(256, activation='relu')(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        # 첫 번째 블록
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        # 두 번째 블록
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        # 세 번째 블록
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # 출력 레이어
        output = layers.Dense(1, activation='linear', name='life_expectancy')(x)
        
        # 모델 생성
        model = keras.Model(inputs=input_layer, outputs=output)
        
        # 컴파일
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def create_enhanced_training_data(self):
        """훈련 데이터 생성"""
        print("딥러닝 훈련 데이터 생성 중...")
        
        np.random.seed(42)
        n_samples = 50000  # 더 많은 데이터
        
        # 기본 특성들 (더 현실적인 분포)
        age = np.random.normal(45, 15, n_samples)
        age = np.clip(age, 18, 80)
        
        bmi = np.random.normal(25, 5, n_samples)
        bmi = np.clip(bmi, 16, 40)
        
        sleep_hours = np.random.normal(7, 1.5, n_samples)
        sleep_hours = np.clip(sleep_hours, 4, 12)
        
        # 스트레스 및 정신건강 예측용 입력 특성 (더 복잡한 관계)
        work_hours = np.random.normal(8, 2, n_samples)
        work_hours = np.clip(work_hours, 0, 16)
        
        environmental_risk = np.random.normal(0.5, 0.3, n_samples)
        environmental_risk = np.clip(environmental_risk, 0, 1)
        
        supplement_dosage = np.random.normal(2, 1, n_samples)
        supplement_dosage = np.clip(supplement_dosage, 0, 10)
        
        family_history = np.random.binomial(1, 0.3, n_samples)
        healthcare_access = np.random.binomial(1, 0.7, n_samples)
        insurance = np.random.binomial(1, 0.8, n_samples)
        
        # 복잡한 스트레스 계산 (비선형 관계)
        stress_level = (10 - sleep_hours * 0.8 + 
                       work_hours * 0.3 + 
                       environmental_risk * 2 + 
                       (1 - healthcare_access) * 1.5 + 
                       np.random.normal(0, 1, n_samples))
        stress_level = np.clip(stress_level, 0, 10)
        
        # 복잡한 정신건강 계산
        mental_health_score = (10 - stress_level * 0.6 - 
                              (1 - insurance) * 0.5 - 
                              environmental_risk * 1.5 + 
                              supplement_dosage * 0.2 + 
                              np.random.normal(0, 1, n_samples))
        mental_health_score = np.clip(mental_health_score, 0, 10)
        
        stress_mental_input = np.column_stack([
            age, bmi, sleep_hours, work_hours, environmental_risk,
            supplement_dosage, family_history, healthcare_access, insurance
        ])
        
        stress_mental_target = np.column_stack([stress_level, mental_health_score])
        
        # 신체활동 예측용 입력 특성 (더 복잡한 관계)
        calorie_intake = np.random.normal(2000, 500, n_samples)
        calorie_intake = np.clip(calorie_intake, 1000, 4000)
        
        sugar_intake = np.random.normal(50, 20, n_samples)
        sugar_intake = np.clip(sugar_intake, 0, 150)
        
        screen_time = np.random.normal(6, 2, n_samples)
        screen_time = np.clip(screen_time, 0, 16)
        
        sedentary_job = np.random.binomial(1, 0.6, n_samples)
        obesity = np.random.binomial(1, 0.3, n_samples)
        chronic_pain = np.random.binomial(1, 0.2, n_samples)
        
        # 복잡한 신체활동 계산
        physical_activity = (300 - age * 2 - bmi * 5 - 
                           sedentary_job * 100 - 
                           obesity * 80 - 
                           chronic_pain * 60 + 
                           np.random.normal(0, 50, n_samples))
        physical_activity = np.clip(physical_activity, 0, 600)
        
        daily_steps = (10000 - age * 50 - bmi * 100 - 
                      sedentary_job * 2000 - 
                      obesity * 1500 - 
                      chronic_pain * 1000 + 
                      np.random.normal(0, 2000, n_samples))
        daily_steps = np.clip(daily_steps, 1000, 15000)
        
        physical_input = np.column_stack([
            age, bmi, calorie_intake, sugar_intake, screen_time,
            sedentary_job, obesity, chronic_pain
        ])
        
        physical_target = np.column_stack([physical_activity, daily_steps])
        
        # 유전적 위험도 예측용 입력 특성
        blood_pressure = np.random.normal(120, 20, n_samples)
        blood_pressure = np.clip(blood_pressure, 80, 200)
        
        cholesterol = np.random.normal(200, 50, n_samples)
        cholesterol = np.clip(cholesterol, 100, 400)
        
        glucose = np.random.normal(100, 20, n_samples)
        glucose = np.clip(glucose, 70, 200)
        
        smoking_status = np.random.binomial(1, 0.2, n_samples)
        alcohol_consumption = np.random.binomial(1, 0.3, n_samples)
        
        # 복잡한 유전적 위험도 계산
        genetic_risk = (family_history * 0.4 + 
                       smoking_status * 0.3 + 
                       alcohol_consumption * 0.2 + 
                       (blood_pressure > 140).astype(float) * 0.1 +
                       (cholesterol > 240).astype(float) * 0.1 +
                       (glucose > 126).astype(float) * 0.1 +
                       np.random.normal(0, 0.1, n_samples))
        genetic_risk = np.clip(genetic_risk, 0, 1)
        
        genetic_input = np.column_stack([
            age, bmi, family_history, smoking_status, alcohol_consumption,
            blood_pressure, cholesterol, glucose
        ])
        
        genetic_target = genetic_risk
        
        # 직접 수명 예측용 데이터
        # 연구 기반 가중치를 사용한 수명 계산
        life_expectancy_model = LifeExpectancyPredictionModel()
        
        life_expectancy_targets = []
        for i in range(min(1000, n_samples)):  # 샘플링하여 계산 시간 단축
            # 랜덤한 건강 데이터 생성
            smoking_status_rand = np.random.choice([0, 1, 2])
            bmi_rand = np.random.uniform(18, 35)
            waist_rand = np.random.uniform(70, 110)
            drinks_rand = np.random.randint(0, 8)
            sleep_quality_rand = np.random.uniform(1, 10)
            sleep_hours_rand = np.random.uniform(4, 10)
            activity_rand = np.random.randint(0, 500)
            steps_rand = np.random.randint(1000, 15000)
            
            try:
                result = life_expectancy_model.calculate_life_expectancy_reduction(
                    smoking_status=smoking_status_rand,
                    bmi=bmi_rand,
                    waist_circumference=waist_rand,
                    height=170,
                    gender='male',
                    drinks_per_week=drinks_rand,
                    sleep_quality_score=sleep_quality_rand,
                    sleep_hours=sleep_hours_rand,
                    weekly_activity_minutes=activity_rand,
                    daily_steps=steps_rand
                )
                life_expectancy_targets.append(result['predicted_life_expectancy'])
            except:
                life_expectancy_targets.append(75.0)  # 기본값
        
        # 나머지는 평균값으로 채움
        while len(life_expectancy_targets) < n_samples:
            life_expectancy_targets.append(np.mean(life_expectancy_targets))
        
        life_expectancy_input = np.column_stack([
            age, bmi, sleep_hours, work_hours, environmental_risk,
            calorie_intake, sugar_intake, screen_time, sedentary_job,
            obesity, chronic_pain, family_history, smoking_status,
            alcohol_consumption, blood_pressure, cholesterol, glucose
        ])
        
        life_expectancy_target = np.array(life_expectancy_targets)
        
        return {
            'stress_input': stress_mental_input,
            'stress_target': stress_mental_target,
            'physical_input': physical_input,
            'physical_target': physical_target,
            'genetic_input': genetic_input,
            'genetic_target': genetic_target,
            'life_expectancy_input': life_expectancy_input,
            'life_expectancy_target': life_expectancy_target
        }
    
    def train_enhanced_models(self):
        """딥러닝 모델들 훈련"""
        print("=" * 60)
        print("딥러닝 시스템 훈련")
        print("=" * 60)
        
        # 1. 고도화된 데이터 생성
        print("\n1. 훈련 데이터 생성")
        print("-" * 40)
        
        data = self.create_enhanced_training_data()
        
        # 2. 모델 훈련
        print("\n2. 모델 훈련")
        print("-" * 40)
        
        # 콜백 설정
        callbacks_list = [
            callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7),
            callbacks.ModelCheckpoint('../models/enhanced_best_{epoch:02d}.h5', 
                                    monitor='val_loss', save_best_only=True)
        ]
        
        # 스트레스 및 정신건강 모델 훈련
        print("\n3. 스트레스 및 정신건강 모델 훈련")
        print("-" * 40)
        
        X_stress = data['stress_input']
        y_stress = data['stress_target']
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_stress, y_stress, test_size=0.2, random_state=42
        )
        
        # 스케일링
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_val_scaled = scaler_X.transform(X_val)
        y_train_scaled = scaler_y.fit_transform(y_train)
        y_val_scaled = scaler_y.transform(y_val)
        
        stress_model = self.build_advanced_stress_model(X_train.shape[1])
        
        stress_history = stress_model.fit(
            X_train_scaled, 
            {'stress_level': y_train_scaled[:, 0], 'mental_health': y_train_scaled[:, 1]},
            validation_data=(X_val_scaled, 
                           {'stress_level': y_val_scaled[:, 0], 'mental_health': y_val_scaled[:, 1]}),
            epochs=100,
            batch_size=64,
            callbacks=callbacks_list,
            verbose=1
        )
        
        self.models['enhanced_stress'] = stress_model
        self.scalers['enhanced_stress_X'] = scaler_X
        self.scalers['enhanced_stress_y'] = scaler_y
        self.histories['enhanced_stress'] = stress_history
        
        # 신체활동 모델 훈련
        print("\n4. 신체활동 모델 훈련")
        print("-" * 40)
        
        X_physical = data['physical_input']
        y_physical = data['physical_target']
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_physical, y_physical, test_size=0.2, random_state=42
        )
        
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_val_scaled = scaler_X.transform(X_val)
        y_train_scaled = scaler_y.fit_transform(y_train)
        y_val_scaled = scaler_y.transform(y_val)
        
        physical_model = self.build_advanced_physical_activity_model(X_train.shape[1])
        
        physical_history = physical_model.fit(
            X_train_scaled,
            {'physical_activity': y_train_scaled[:, 0], 'daily_steps': y_train_scaled[:, 1]},
            validation_data=(X_val_scaled,
                           {'physical_activity': y_val_scaled[:, 0], 'daily_steps': y_val_scaled[:, 1]}),
            epochs=150,
            batch_size=64,
            callbacks=callbacks_list,
            verbose=1
        )
        
        self.models['enhanced_physical'] = physical_model
        self.scalers['enhanced_physical_X'] = scaler_X
        self.scalers['enhanced_physical_y'] = scaler_y
        self.histories['enhanced_physical'] = physical_history
        
        # 유전적 위험도 모델 훈련
        print("\n5. 유전적 위험도 모델 훈련")
        print("-" * 40)
        
        X_genetic = data['genetic_input']
        y_genetic = data['genetic_target']
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_genetic, y_genetic, test_size=0.2, random_state=42
        )
        
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_val_scaled = scaler_X.transform(X_val)
        
        genetic_model = self.build_advanced_genetic_risk_model(X_train.shape[1])
        
        genetic_history = genetic_model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=200,
            batch_size=64,
            callbacks=callbacks_list,
            verbose=1
        )
        
        self.models['enhanced_genetic'] = genetic_model
        self.scalers['enhanced_genetic_X'] = scaler_X
        self.histories['enhanced_genetic'] = genetic_history
        
        # 직접 수명 예측 모델 훈련
        print("\n6. 직접 수명 예측 모델 훈련")
        print("-" * 40)
        
        X_life = data['life_expectancy_input']
        y_life = data['life_expectancy_target']
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_life, y_life, test_size=0.2, random_state=42
        )
        
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_val_scaled = scaler_X.transform(X_val)
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
        
        life_model = self.build_life_expectancy_direct_model(X_train.shape[1])
        
        life_history = life_model.fit(
            X_train_scaled, y_train_scaled,
            validation_data=(X_val_scaled, y_val_scaled),
            epochs=100,
            batch_size=64,
            callbacks=callbacks_list,
            verbose=1
        )
        
        self.models['enhanced_life_expectancy'] = life_model
        self.scalers['enhanced_life_expectancy_X'] = scaler_X
        self.scalers['enhanced_life_expectancy_y'] = scaler_y
        self.histories['enhanced_life_expectancy'] = life_history
        
        # 3. 성능 평가
        print("\n7. 모델 성능 평가")
        print("-" * 40)
        
        self.evaluate_enhanced_models(data)
        
        # 4. 모델 저장
        print("\n8. 모델 저장")
        print("-" * 40)
        
        self.save_enhanced_models()
        
        # 5. 시각화
        print("\n9. 훈련 과정 시각화")
        print("-" * 40)
        
        self.visualize_enhanced_training()
        
        print("\n" + "=" * 60)
        print("딥러닝 시스템 훈련 완료")
        print("=" * 60)
        
        return self.models
    
    def evaluate_enhanced_models(self, data):
        """모델 성능 평가"""
        print("모델 성능 평가 중...")
        
        # 스트레스 모델 평가
        X_test = data['stress_input'][:1000]
        y_test = data['stress_target'][:1000]
        
        X_test_scaled = self.scalers['enhanced_stress_X'].transform(X_test)
        y_pred_scaled = self.models['enhanced_stress'].predict(X_test_scaled)
        y_pred = self.scalers['enhanced_stress_y'].inverse_transform(
            np.column_stack([y_pred_scaled[0], y_pred_scaled[1]])
        )
        
        stress_mse = mean_squared_error(y_test[:, 0], y_pred[:, 0])
        mental_mse = mean_squared_error(y_test[:, 1], y_pred[:, 1])
        
        self.performance_metrics['enhanced_stress'] = {
            'stress_mse': stress_mse,
            'mental_mse': mental_mse,
            'stress_mae': mean_absolute_error(y_test[:, 0], y_pred[:, 0]),
            'mental_mae': mean_absolute_error(y_test[:, 1], y_pred[:, 1])
        }
        
        print(f"스트레스 모델 성능:")
        print(f"  스트레스 MSE: {stress_mse:.4f}")
        print(f"  정신건강 MSE: {mental_mse:.4f}")
        
        # 신체활동 모델 평가
        X_test = data['physical_input'][:1000]
        y_test = data['physical_target'][:1000]
        
        X_test_scaled = self.scalers['enhanced_physical_X'].transform(X_test)
        y_pred_scaled = self.models['enhanced_physical'].predict(X_test_scaled)
        y_pred = self.scalers['enhanced_physical_y'].inverse_transform(
            np.column_stack([y_pred_scaled[0], y_pred_scaled[1]])
        )
        
        physical_mse = mean_squared_error(y_test[:, 0], y_pred[:, 0])
        steps_mse = mean_squared_error(y_test[:, 1], y_pred[:, 1])
        
        self.performance_metrics['enhanced_physical'] = {
            'physical_mse': physical_mse,
            'steps_mse': steps_mse,
            'physical_mae': mean_absolute_error(y_test[:, 0], y_pred[:, 0]),
            'steps_mae': mean_absolute_error(y_test[:, 1], y_pred[:, 1])
        }
        
        print(f"신체활동 모델 성능:")
        print(f" 신체활동 MSE: {physical_mse:.4f}")
        print(f" 걸음수 MSE: {steps_mse:.4f}")
        
        # 유전적 위험도 모델 평가
        X_test = data['genetic_input'][:1000]
        y_test = data['genetic_target'][:1000]
        
        X_test_scaled = self.scalers['enhanced_genetic_X'].transform(X_test)
        y_pred = self.models['enhanced_genetic'].predict(X_test_scaled).flatten()
        
        # 이진 분류로 변환 (0.5 기준)
        y_pred_binary = (y_pred > 0.5).astype(int)
        y_test_binary = (y_test > 0.5).astype(int)
        
        accuracy = accuracy_score(y_test_binary, y_pred_binary)
        
        self.performance_metrics['enhanced_genetic'] = {
            'accuracy': accuracy,
            'mse': mean_squared_error(y_test, y_pred)
        }
        
        print(f"유전적 위험도 모델 성능:")
        print(f"  정확도: {accuracy:.4f}")
        print(f"  MSE: {mean_squared_error(y_test, y_pred):.4f}")
        
        # 직접 수명 예측 모델 평가
        X_test = data['life_expectancy_input'][:1000]
        y_test = data['life_expectancy_target'][:1000]
        
        X_test_scaled = self.scalers['enhanced_life_expectancy_X'].transform(X_test)
        y_pred_scaled = self.models['enhanced_life_expectancy'].predict(X_test_scaled).flatten()
        y_pred = self.scalers['enhanced_life_expectancy_y'].inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        life_mse = mean_squared_error(y_test, y_pred)
        life_mae = mean_absolute_error(y_test, y_pred)
        life_r2 = r2_score(y_test, y_pred)
        
        self.performance_metrics['enhanced_life_expectancy'] = {
            'mse': life_mse,
            'mae': life_mae,
            'r2': life_r2
        }
        
        print(f"직접 수명 예측 모델 성능:")
        print(f"  MSE: {life_mse:.4f}")
        print(f"  MAE: {life_mae:.4f}")
        print(f"  R²: {life_r2:.4f}")
    
    def save_enhanced_models(self):
        """모델 저장"""
        for name, model in self.models.items():
            model.save(f"../models/{name}.h5")
        
        # 스케일러 저장
        import joblib
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, f"../models/{name}_scaler.pkl")
        
        # 성능 메트릭 저장
        import json
        with open('../models/enhanced_performance_metrics.json', 'w') as f:
            json.dump(self.performance_metrics, f, indent=2)
        
        print("모든 모델이 저장되었습니다.")
    
    def visualize_enhanced_training(self):
        """고도화된 훈련 과정 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 스트레스 모델 훈련 과정
        if 'enhanced_stress' in self.histories:
            history = self.histories['enhanced_stress']
            axes[0, 0].plot(history.history['loss'], label='Training Loss')
            axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
            axes[0, 0].set_title('고도화된 스트레스 모델 훈련 과정')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
        
        # 신체활동 모델 훈련 과정
        if 'enhanced_physical' in self.histories:
            history = self.histories['enhanced_physical']
            axes[0, 1].plot(history.history['loss'], label='Training Loss')
            axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
            axes[0, 1].set_title('신체활동 모델 훈련 과정')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
        
        # 유전적 위험도 모델 훈련 과정
        if 'enhanced_genetic' in self.histories:
            history = self.histories['enhanced_genetic']
            axes[1, 0].plot(history.history['loss'], label='Training Loss')
            axes[1, 0].plot(history.history['val_loss'], label='Validation Loss')
            axes[1, 0].set_title('유전적 위험도 모델 훈련 과정')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
        
        # 직접 수명 예측 모델 훈련 과정
        if 'enhanced_life_expectancy' in self.histories:
            history = self.histories['enhanced_life_expectancy']
            axes[1, 1].plot(history.history['loss'], label='Training Loss')
            axes[1, 1].plot(history.history['val_loss'], label='Validation Loss')
            axes[1, 1].set_title('직접 수명 예측 모델 훈련 과정')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('../models/enhanced_deep_learning_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """메인 함수"""
    enhanced_system = EnhancedDeepLearningSystem()
    models = enhanced_system.train_enhanced_models()
    
    print("\n 딥러닝 시스템 특징:")
    print("1. Attention Mechanism (유전적 위험도 모델)")
    print("2. LSTM + CNN 하이브리드 (신체활동 모델)")
    print("3. 잔차 연결 (Residual Connections)")
    print("4. 다중 출력 모델 (스트레스 & 정신건강)")
    print("5. 직접 수명 예측 모델")
    print("6. 고도화된 콜백 시스템")
    print("7. 복잡한 비선형 관계 모델링")

if __name__ == "__main__":
    main()
