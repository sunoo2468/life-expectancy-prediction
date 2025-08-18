#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from model.life_expectancy_prediction_model import LifeExpectancyPredictionModel
from model.integrated_weight_calculator import IntegratedWeightCalculator

def test_personal_prediction():
    """개인별 수명 예측 테스트"""
    
    print("=" * 80)
    print("개인별 수명 예측 AI 시스템")
    print("=" * 80)
    
    # 모델 초기화
    model = LifeExpectancyPredictionModel()
    calculator = IntegratedWeightCalculator()
    
    # 사용자 입력 데이터
    print("\n📋 입력된 생활습관 데이터:")
    print("-" * 50)
    
    # 흡연 상태 (0: 비흡연, 1: 과거 흡연, 2: 현재 흡연)
    smoking_status = 0  # 비흡연
    print(f"🚭 흡연 상태: 비흡연")
    
    # BMI
    bmi = 32.0
    print(f"📏 BMI: {bmi} (비만)")
    
    # 허리둘레 (BMI 32 기준으로 추정)
    height = 170  # cm (평균 키로 가정)
    waist_circumference = 95  # cm (BMI 32 기준 추정)
    print(f"📐 허리둘레: {waist_circumference}cm")
    
    # 알코올 섭취
    drinks_per_week = 1  # 1주일에 1병
    print(f"🍺 알코올 섭취: 주 1병")
    
    # 수면
    sleep_quality_score = 8.0  # 7시간 숙면이므로 양호
    sleep_hours = 7
    print(f"😴 수면: {sleep_hours}시간 (숙면)")
    
    # 신체활동
    weekly_activity_minutes = 60  # 런닝 1시간
    daily_steps = 8000  # 런닝 포함 추정
    print(f"🏃‍♂️ 신체활동: 런닝 1시간/주")
    
    print("\n🔬 수명 예측 분석 중...")
    print("-" * 50)
    
    # 통합 위험도 계산
    integrated_analysis = calculator.calculate_integrated_risk(
        smoking_status=smoking_status,
        years_since_quit=None,
        passive_smoking=False,
        cigarettes_per_day=0,
        smoking_type='traditional',
        bmi=bmi,
        waist_circumference=waist_circumference,
        height=height,
        gender='male',
        age_group='middle',
        drinks_per_week=drinks_per_week,
        drink_type='soju',
        binge_drinking=False,
        chronic_drinking=False,
        sleep_quality_score=sleep_quality_score,
        sleep_hours=sleep_hours,
        insomnia=False,
        sleep_apnea=False,
        irregular_schedule=False,
        stress_level='low',
        weekly_activity_minutes=weekly_activity_minutes,
        daily_steps=daily_steps,
        intensity='high_intensity',
        sedentary_job=False,
        no_exercise=False,
        poor_mobility=False,
        chronic_pain=False,
        obesity=True,  # BMI 32는 비만
        poor_diet=False
    )
    
    # 수명 예측
    life_expectancy_result = model.calculate_life_expectancy_reduction(
        integrated_risk=integrated_analysis['integrated_risk'],
        gender='male',
        age_group='middle'
    )
    
    print("\n📊 수명 예측 결과:")
    print("=" * 50)
    
    print(f"🎯 기준 수명: {life_expectancy_result['base_life_expectancy']:.1f}세")
    print(f"📉 예상 수명: {life_expectancy_result['predicted_life_expectancy']:.1f}세")
    print(f"⏰ 수명 단축: {life_expectancy_result['life_reduction']:.1f}년")
    print(f"📈 개선 잠재력: {life_expectancy_result['life_improvement_potential']:.1f}년")
    
    print(f"\n⚠️  위험 수준: {integrated_analysis['risk_level']}")
    print(f"📋 종합 위험도: {integrated_analysis['integrated_risk']:.3f}")
    
    print("\n🔍 피처별 기여도:")
    print("-" * 30)
    
    feature_contributions = integrated_analysis['feature_contributions']
    for feature, contribution in feature_contributions.items():
        if contribution > 0:
            print(f"  {feature}: {contribution:.1%}")
    
    print("\n💡 건강 권고사항:")
    print("-" * 30)
    
    recommendations = integrated_analysis['recommendations']
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
    
    print("\n📚 연구 근거:")
    print("-" * 30)
    print(f"  • 총 {integrated_analysis['research_stats']['total_papers']}개 연구 논문 기반")
    print(f"  • 최신 연구 (2020-2025): {integrated_analysis['research_stats']['recent_papers_2020_2025']}개")
    print(f"  • 한국인 대상 연구: {integrated_analysis['research_stats']['korean_studies']}개")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    test_personal_prediction()
