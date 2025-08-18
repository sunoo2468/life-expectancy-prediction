#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model.life_expectancy_prediction_model import LifeExpectancyPredictionModel

def test_with_sample_data():
    """샘플 데이터로 수명 예측 테스트"""
    print("=" * 60)
    print("🧪 개인별 수명 예측 시스템 테스트")
    print("=" * 60)
    
    # 테스트 케이스 1: 건강한 생활습관
    print("\n📋 테스트 케이스 1: 건강한 생활습관")
    print("-" * 40)
    
    model = LifeExpectancyPredictionModel()
    
    # 건강한 생활습관 예시
    result1 = model.calculate_life_expectancy_reduction(
        smoking_status=0,  # 비흡연자
        bmi=22.0,  # 정상 BMI
        waist_circumference=75,  # 정상 허리둘레
        height=170,
        gender='male',
        age_group='middle',
        drinks_per_week=0,  # 비음주자
        sleep_quality_score=8.0,  # 좋은 수면
        sleep_hours=7.5,
        weekly_activity_minutes=300,  # 충분한 운동
        daily_steps=10000,
        intensity='moderate_intensity'
    )
    
    print(f"👤 30대 남성, 건강한 생활습관")
    print(f"📊 기준 수명: {result1['base_life_expectancy']:.1f}세")
    print(f"📉 예상 수명: {result1['predicted_life_expectancy']:.1f}세")
    print(f"⏰ 수명 단축: {result1['life_reduction']:.1f}년")
    print(f"⚠️ 위험 수준: {result1['risk_level']}")
    
    # 테스트 케이스 2: 위험한 생활습관
    print("\n📋 테스트 케이스 2: 위험한 생활습관")
    print("-" * 40)
    
    result2 = model.calculate_life_expectancy_reduction(
        smoking_status=2,  # 현재 흡연자
        cigarettes_per_day=20,
        bmi=32.0,  # 비만
        waist_circumference=95,
        height=170,
        gender='male',
        age_group='middle',
        drinks_per_week=5,  # 자주 마심
        binge_drinking=True,
        sleep_quality_score=3.0,  # 나쁜 수면
        sleep_hours=4,
        insomnia=True,
        weekly_activity_minutes=30,  # 부족한 운동
        daily_steps=3000,
        intensity='low_intensity',
        sedentary_job=True,
        obesity=True,
        poor_diet=True
    )
    
    print(f"👤 30대 남성, 위험한 생활습관")
    print(f"📊 기준 수명: {result2['base_life_expectancy']:.1f}세")
    print(f"📉 예상 수명: {result2['predicted_life_expectancy']:.1f}세")
    print(f"⏰ 수명 단축: {result2['life_reduction']:.1f}년")
    print(f"⚠️ 위험 수준: {result2['risk_level']}")
    
    # 테스트 케이스 3: 중간 생활습관
    print("\n📋 테스트 케이스 3: 중간 생활습관")
    print("-" * 40)
    
    result3 = model.calculate_life_expectancy_reduction(
        smoking_status=0,  # 비흡연자
        bmi=26.0,  # 과체중
        waist_circumference=85,
        height=170,
        gender='female',
        age_group='middle',
        drinks_per_week=2,  # 가끔 마심
        sleep_quality_score=6.0,  # 보통 수면
        sleep_hours=6.5,
        weekly_activity_minutes=150,  # 보통 운동
        daily_steps=7000,
        intensity='moderate_intensity'
    )
    
    print(f"👤 30대 여성, 중간 생활습관")
    print(f"📊 기준 수명: {result3['base_life_expectancy']:.1f}세")
    print(f"📉 예상 수명: {result3['predicted_life_expectancy']:.1f}세")
    print(f"⏰ 수명 단축: {result3['life_reduction']:.1f}년")
    print(f"⚠️ 위험 수준: {result3['risk_level']}")
    
    print("\n" + "=" * 60)
    print("✅ 테스트 완료!")
    print("=" * 60)
    
    print("\n💡 사용 방법:")
    print("1. python personal_input_predict.py 실행")
    print("2. 질문에 따라 개인 정보 입력")
    print("3. 개인별 맞춤 수명 예측 결과 확인")

if __name__ == "__main__":
    test_with_sample_data()
