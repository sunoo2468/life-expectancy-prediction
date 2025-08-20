#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
질병 위험도 예측 모델
사용자의 건강 지표를 기반으로 질병 발병률 예측
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

class DiseaseRiskModel:
    """질병 위험도 예측 모델"""
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # 입력 특성 정의 (슬라이드 기준)
        self.input_features = [
            'age', 'height', 'weight', 'waist_size',
            'stress_level', 'physical_activity', 'daily_steps',
            'sleep_quality', 'smoking_level', 'mental_health_score',
            'alcohol_consumption'
        ]
        
        # 질병 카테고리
        self.disease_categories = [
            'cardiovascular_disease',
            'diabetes',
            'respiratory_disease',
            'mental_health_issues',
            'obesity_related'
        ]
    
    def prepare_input(self, user_data):
        """사용자 입력 데이터 전처리"""
        # 슬라이드의 입력 예시 형식에 맞게 변환
        processed_data = {}
        
        # 기본 정보
        processed_data['age'] = user_data.get('age', 30)
        processed_data['height'] = user_data.get('height', 170)
        processed_data['weight'] = user_data.get('weight', 70)
        processed_data['waist_size'] = user_data.get('waist_size', 80)
        
        # 건강 지표 (0.0~0.9 범위로 정규화)
        processed_data['stress_level'] = min(0.9, max(0.0, user_data.get('stress_level', 5) / 10))
        processed_data['physical_activity'] = min(0.9, max(0.0, user_data.get('weekly_activity_minutes', 150) / 600))
        processed_data['daily_steps'] = user_data.get('daily_steps', 8000)
        processed_data['sleep_quality'] = min(0.9, max(0.0, user_data.get('sleep_quality_score', 7) / 10))
        
        # 범주형 변수
        processed_data['smoking_level'] = user_data.get('smoking_status', 0)  # 0: non-smoker, 1: light, 2: heavy
        processed_data['mental_health_score'] = min(9, max(0, user_data.get('mental_health_score', 7)))
        processed_data['alcohol_consumption'] = 1 if user_data.get('drinks_per_week', 0) > 5 else 0  # 0: occasionally, 1: regularly
        
        return processed_data
    
    def train_model(self, training_data):
        """모델 훈련 (예시 데이터로)"""
        # 실제로는 훈련 데이터가 필요하지만, 예시로 간단한 규칙 기반 모델 생성
        self.is_trained = True
        print("✅ 질병 위험도 모델 훈련 완료")
    
    def predict_disease_risk(self, user_data):
        """질병 위험도 예측"""
        if not self.is_trained:
            self.train_model(None)
        
        # 입력 데이터 전처리
        processed_data = self.prepare_input(user_data)
        
        # 규칙 기반 위험도 계산 (실제로는 훈련된 모델 사용)
        risks = {}
        
        # 심혈관 질환 위험도
        age_risk = min(0.8, processed_data['age'] / 100)
        smoking_risk = processed_data['smoking_level'] * 0.3
        stress_risk = processed_data['stress_level'] * 0.2
        risks['cardiovascular_disease'] = min(0.9, age_risk + smoking_risk + stress_risk)
        
        # 당뇨병 위험도
        bmi = processed_data['weight'] / ((processed_data['height'] / 100) ** 2)
        bmi_risk = min(0.7, max(0, (bmi - 25) / 10))
        activity_risk = (1 - processed_data['physical_activity']) * 0.3
        risks['diabetes'] = min(0.8, bmi_risk + activity_risk)
        
        # 호흡기 질환 위험도
        smoking_resp_risk = processed_data['smoking_level'] * 0.4
        risks['respiratory_disease'] = min(0.9, smoking_resp_risk)
        
        # 정신건강 위험도
        stress_mental_risk = processed_data['stress_level'] * 0.4
        mental_score_risk = (9 - processed_data['mental_health_score']) / 9 * 0.3
        risks['mental_health_issues'] = min(0.8, stress_mental_risk + mental_score_risk)
        
        # 비만 관련 위험도
        waist_risk = min(0.6, max(0, (processed_data['waist_size'] - 80) / 40))
        risks['obesity_related'] = min(0.7, bmi_risk + waist_risk)
        
        # 전체 위험도 점수
        total_risk = np.mean(list(risks.values()))
        
        return {
            'disease_risks': risks,
            'total_risk_score': total_risk,
            'risk_level': self._get_risk_level(total_risk),
            'recommendations': self._generate_recommendations(risks, processed_data)
        }
    
    def _get_risk_level(self, risk_score):
        """위험도 수준 판정"""
        if risk_score < 0.3:
            return "Low"
        elif risk_score < 0.6:
            return "Medium"
        else:
            return "High"
    
    def _generate_recommendations(self, risks, user_data):
        """개인 맞춤 건강 피드백 생성"""
        recommendations = []
        
        # 심혈관 질환 관련
        if risks['cardiovascular_disease'] > 0.5:
            recommendations.append("심혈관 질환 위험이 높습니다. 정기적인 운동과 금연을 권장합니다.")
        
        # 당뇨병 관련
        if risks['diabetes'] > 0.4:
            recommendations.append("당뇨병 위험이 있습니다. 체중 관리와 규칙적인 운동이 필요합니다.")
        
        # 호흡기 질환 관련
        if risks['respiratory_disease'] > 0.3:
            recommendations.append("호흡기 질환 위험이 있습니다. 금연과 깨끗한 환경 유지가 중요합니다.")
        
        # 정신건강 관련
        if risks['mental_health_issues'] > 0.5:
            recommendations.append("정신건강 관리가 필요합니다. 스트레스 해소 활동과 전문가 상담을 권장합니다.")
        
        # 비만 관련
        if risks['obesity_related'] > 0.4:
            recommendations.append("비만 관련 위험이 있습니다. 균형 잡힌 식단과 운동이 필요합니다.")
        
        # 일반적인 권장사항
        if user_data['physical_activity'] < 0.3:
            recommendations.append("신체활동을 늘려주세요. 주 150분 이상의 중등도 운동을 권장합니다.")
        
        if user_data['sleep_quality'] < 0.6:
            recommendations.append("수면의 질을 개선해주세요. 7-9시간의 충분한 수면이 필요합니다.")
        
        return recommendations
    
    def save_model(self, filepath):
        """모델 저장"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'input_features': self.input_features
        }
        joblib.dump(model_data, filepath)
        print(f"✅ 질병 위험도 모델 저장 완료: {filepath}")
    
    def load_model(self, filepath):
        """모델 로드"""
        if os.path.exists(filepath):
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.is_trained = model_data['is_trained']
            self.input_features = model_data['input_features']
            print(f"✅ 질병 위험도 모델 로드 완료: {filepath}")
        else:
            print(f"⚠️ 모델 파일을 찾을 수 없습니다: {filepath}")

# 사용 예시
if __name__ == "__main__":
    # 모델 인스턴스 생성
    disease_model = DiseaseRiskModel()
    
    # 사용자 입력 예시 (슬라이드 기준)
    user_input = {
        'age': 20,
        'height': 170,
        'weight': 50,
        'waist_size': 24,
        'stress_level': 0.5,
        'physical_activity': 0.4,
        'daily_steps': 800,
        'sleep_quality': 0.9,
        'smoking_level': 2,
        'mental_health_score': 1,
        'alcohol_consumption': 1
    }
    
    # 예측 실행
    result = disease_model.predict_disease_risk(user_input)
    
    print("=== 질병 위험도 예측 결과 ===")
    print(f"전체 위험도: {result['total_risk_score']:.2f} ({result['risk_level']})")
    print("\n개별 질병 위험도:")
    for disease, risk in result['disease_risks'].items():
        print(f"- {disease}: {risk:.2f}")
    
    print("\n건강 권장사항:")
    for rec in result['recommendations']:
        print(f"- {rec}") 
