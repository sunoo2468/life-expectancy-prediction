#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from model.deep_learning_features import DeepLearningFeatures
from data_processing.data_processor import DataProcessor

def create_synthetic_data():
    """딥러닝 모델 훈련을 위한 합성 데이터 생성"""
    print("딥러닝 모델 훈련용 합성 데이터 생성 중...")
    
    np.random.seed(42)
    n_samples = 10000
    
    # 기본 특성들
    age = np.random.normal(45, 15, n_samples)
    age = np.clip(age, 18, 80)
    
    bmi = np.random.normal(25, 5, n_samples)
    bmi = np.clip(bmi, 16, 40)
    
    sleep_hours = np.random.normal(7, 1.5, n_samples)
    sleep_hours = np.clip(sleep_hours, 4, 12)
    
    # 스트레스 및 정신건강 예측용 입력 특성
    stress_input_features = np.column_stack([
        age,
        bmi,
        sleep_hours,
        np.random.normal(5, 2, n_samples),  # work_hours
        np.random.normal(0.5, 0.3, n_samples),  # environmental_risk_score
        np.random.normal(2, 1, n_samples),  # daily_supplement_dosage
        np.random.binomial(1, 0.3, n_samples),  # family_history
        np.random.binomial(1, 0.7, n_samples),  # healthcare_access
        np.random.binomial(1, 0.8, n_samples),  # insurance
    ])
    
    # 스트레스 및 정신건강 타겟 (수면의 질과 관련)
    stress_level = 10 - (sleep_hours * 0.8 + np.random.normal(0, 1, n_samples))
    stress_level = np.clip(stress_level, 0, 10)
    
    mental_health_score = 10 - (stress_level * 0.6 + np.random.normal(0, 1, n_samples))
    mental_health_score = np.clip(mental_health_score, 0, 10)
    
    stress_mental_target = np.column_stack([stress_level, mental_health_score])
    
    # 신체활동 예측용 입력 특성
    physical_input_features = np.column_stack([
        age,
        bmi,
        np.random.normal(2000, 1000, n_samples),  # calorie_intake
        np.random.normal(50, 20, n_samples),  # sugar_intake
        np.random.normal(6, 2, n_samples),  # screen_time
        np.random.binomial(1, 0.6, n_samples),  # sedentary_job
        np.random.binomial(1, 0.3, n_samples),  # obesity
        np.random.binomial(1, 0.2, n_samples),  # chronic_pain
    ])
    
    # 신체활동 타겟
    physical_activity = 300 - (age * 2 + bmi * 5 + np.random.normal(0, 50, n_samples))
    physical_activity = np.clip(physical_activity, 0, 600)
    
    daily_steps = 10000 - (age * 50 + bmi * 100 + np.random.normal(0, 2000, n_samples))
    daily_steps = np.clip(daily_steps, 1000, 15000)
    
    physical_target = np.column_stack([physical_activity, daily_steps])
    
    # 유전적 위험도 예측용 입력 특성
    genetic_input_features = np.column_stack([
        age,
        bmi,
        np.random.binomial(1, 0.3, n_samples),  # family_history
        np.random.binomial(1, 0.1, n_samples),  # smoking_status
        np.random.binomial(1, 0.2, n_samples),  # alcohol_consumption
        np.random.normal(120, 20, n_samples),  # blood_pressure
        np.random.normal(200, 50, n_samples),  # cholesterol
        np.random.normal(100, 20, n_samples),  # glucose
    ])
    
    # 유전적 위험도 타겟 (가족력과 관련)
    genetic_risk = (genetic_input_features[:, 2] * 0.4 +  # family_history
                   genetic_input_features[:, 3] * 0.3 +  # smoking
                   genetic_input_features[:, 4] * 0.2 +  # alcohol
                   np.random.normal(0, 0.1, n_samples))
    genetic_risk = np.clip(genetic_risk, 0, 1)
    
    return {
        'stress_input': stress_input_features,
        'stress_target': stress_mental_target,
        'physical_input': physical_input_features,
        'physical_target': physical_target,
        'genetic_input': genetic_input_features,
        'genetic_target': genetic_risk
    }

def train_deep_learning_models():
    """딥러닝 모델들 훈련"""
    print("=" * 60)
    print("딥러닝 모델 훈련 시스템")
    print("=" * 60)
    
    # 1. 데이터 생성
    print("\n1. 합성 데이터 생성")
    print("-" * 40)
    
    data = create_synthetic_data()
    
    # 2. 딥러닝 모델 초기화
    print("\n2. 딥러닝 모델 초기화")
    print("-" * 40)
    
    dl_models = DeepLearningFeatures()
    
    # 3. 스트레스 및 정신건강 모델 훈련
    print("\n3. 스트레스 및 정신건강 예측 모델 훈련")
    print("-" * 40)
    
    X_stress = data['stress_input']
    y_stress = data['stress_target']
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_stress, y_stress, test_size=0.2, random_state=42
    )
    
    stress_history = dl_models.train_stress_mental_model(X_train, y_train, X_val, y_val)
    
    # 4. 신체활동 모델 훈련
    print("\n4. 신체활동량 예측 모델 훈련")
    print("-" * 40)
    
    X_physical = data['physical_input']
    y_physical = data['physical_target']
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_physical, y_physical, test_size=0.2, random_state=42
    )
    
    physical_history = dl_models.train_physical_activity_model(X_train, y_train, X_val, y_val)
    
    # 5. 유전적 위험도 모델 훈련
    print("\n5. 유전적 위험도 예측 모델 훈련")
    print("-" * 40)
    
    X_genetic = data['genetic_input']
    y_genetic = data['genetic_target']
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_genetic, y_genetic, test_size=0.2, random_state=42
    )
    
    genetic_history = dl_models.train_genetic_risk_model(X_train, y_train, X_val, y_val)
    
    # 6. 모델 저장
    print("\n6. 모델 저장")
    print("-" * 40)
    
    dl_models.save_models('../models/deep_learning')
    
    # 7. 테스트 예측
    print("\n7. 테스트 예측")
    print("-" * 40)
    
    # 스트레스 및 정신건강 예측 테스트
    test_stress_input = X_stress[:5]
    stress_pred = dl_models.predict_stress_mental(test_stress_input)
    print(f"스트레스 및 정신건강 예측 결과:")
    print(f"실제: {y_stress[:5]}")
    print(f"예측: {stress_pred}")
    
    # 신체활동 예측 테스트
    test_physical_input = X_physical[:5]
    physical_pred = dl_models.predict_physical_activity(test_physical_input)
    print(f"\n신체활동 예측 결과:")
    print(f"실제: {y_physical[:5]}")
    print(f"예측: {physical_pred}")
    
    # 유전적 위험도 예측 테스트
    test_genetic_input = X_genetic[:5]
    genetic_pred = dl_models.predict_genetic_risk(test_genetic_input)
    print(f"\n유전적 위험도 예측 결과:")
    print(f"실제: {y_genetic[:5]}")
    print(f"예측: {genetic_pred.flatten()}")
    
    # 8. 훈련 과정 시각화
    print("\n8. 훈련 과정 시각화")
    print("-" * 40)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 스트레스 모델 훈련 과정
    axes[0, 0].plot(stress_history.history['loss'], label='Training Loss')
    axes[0, 0].plot(stress_history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('스트레스 및 정신건강 모델 훈련 과정')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    
    # 신체활동 모델 훈련 과정
    axes[0, 1].plot(physical_history.history['loss'], label='Training Loss')
    axes[0, 1].plot(physical_history.history['val_loss'], label='Validation Loss')
    axes[0, 1].set_title('신체활동 모델 훈련 과정')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    
    # 유전적 위험도 모델 훈련 과정
    axes[1, 0].plot(genetic_history.history['loss'], label='Training Loss')
    axes[1, 0].plot(genetic_history.history['val_loss'], label='Validation Loss')
    axes[1, 0].set_title('유전적 위험도 모델 훈련 과정')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    
    # 정확도 (유전적 위험도 모델)
    axes[1, 1].plot(genetic_history.history['accuracy'], label='Training Accuracy')
    axes[1, 1].plot(genetic_history.history['val_accuracy'], label='Validation Accuracy')
    axes[1, 1].set_title('유전적 위험도 모델 정확도')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('../models/deep_learning_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "=" * 60)
    print("딥러닝 모델 훈련 완료!")
    print("=" *60)
    
    return dl_models

if __name__ == "__main__":
    train_deep_learning_models()
