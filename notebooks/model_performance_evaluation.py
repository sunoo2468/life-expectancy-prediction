import pandas as pd
import numpy as np
import sys
import os
sys.path.append('..')

from src.model.life_expectancy_prediction_model import LifeExpectancyPredictionModel
from src.data_processing.data_processor import DataProcessor

def evaluate_model_performance():
    """모델 성능 평가"""
    print("=" * 80)
    print("수명 예측 모델 성능 평가")
    print("=" * 80)
    
    # 데이터 로드
    print("\n 데이터 로드 중...")
    data_processor = DataProcessor()
    
    # Life Expectancy Data 로드
    life_expectancy_data = data_processor.load_data('../data/Life Expectancy Data.csv')
    print(f"Life Expectancy Data 크기: {life_expectancy_data.shape}")
    
    # health_lifestyle_classification 데이터 로드
    lifestyle_data = data_processor.load_data('../data/health_lifestyle_classification.csv')
    print(f"Lifestyle Data 크기: {lifestyle_data.shape}")
    
    # 데이터 전처리
    print("\n 데이터 전처리 중...")
    
    # Life Expectancy Data 전처리
    life_expectancy_clean = data_processor.handle_missing_values(life_expectancy_data)
    
    # Lifestyle Data 전처리
    lifestyle_clean = data_processor.handle_missing_values(lifestyle_data)
    
    # 모델 초기화
    print("\n 모델 초기화 중...")
    model = LifeExpectancyPredictionModel()
    
    # 연구 기반 가중치 시스템 성능 평가
    print("\n 연구 기반 가중치 시스템 성능 평가:")
    
    # 테스트 케이스들
    test_cases = [
        {
            'name': '완전 건강한 생활습관',
            'params': {
                'bmi': 22.0, 'waist_size': 75, 'smoking_level': 0, 'alcohol_consumption': 0,
                'sleep_quality': 9.0, 'physical_activity': 7, 'age': 30, 'gender': 'male',
                'smoking_status': 0, 'height': 170, 'age_group': 'middle',
                'sleep_hours': 8, 'weekly_activity_minutes': 200, 'daily_steps': 10000, 'intensity': 'high_intensity'
            }
        },
        {
            'name': '보통 생활습관',
            'params': {
                'bmi': 25.0, 'waist_size': 85, 'smoking_level': 1, 'alcohol_consumption': 5,
                'sleep_quality': 6.0, 'physical_activity': 4, 'age': 40, 'gender': 'male',
                'smoking_status': 1, 'years_since_quit': 5, 'height': 170, 'age_group': 'middle',
                'sleep_hours': 7, 'weekly_activity_minutes': 120, 'daily_steps': 6000, 'intensity': 'moderate_intensity'
            }
        },
        {
            'name': '위험한 생활습관',
            'params': {
                'bmi': 32.0, 'waist_size': 105, 'smoking_level': 2, 'alcohol_consumption': 15,
                'sleep_quality': 3.0, 'physical_activity': 1, 'age': 50, 'gender': 'male',
                'smoking_status': 2, 'cigarettes_per_day': 20, 'height': 170, 'age_group': 'middle',
                'drinks_per_week': 15, 'binge_drinking': True, 'sleep_hours': 5, 'insomnia': True,
                'weekly_activity_minutes': 0, 'daily_steps': 2000, 'sedentary_job': True
            }
        }
    ]
    
    # 각 테스트 케이스 평가
    results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n 테스트 케이스 {i}: {test_case['name']}")
        
        result = model.predict_life_expectancy(**test_case['params'])
        
        print(f"  기준 수명: {result['base_life_expectancy']:.1f}세")
        print(f"  예상 수명: {result['final_life_expectancy']:.1f}세")
        print(f"  수명 단축: {result['final_life_reduction']:.1f}년")
        print(f"  위험 수준: {result['risk_level']}")
        print(f"  개선 잠재력: {result['life_improvement_potential']['improvement_potential']:.1f}년")
        
        results.append({
            'test_case': test_case['name'],
            'base_life': result['base_life_expectancy'],
            'predicted_life': result['final_life_expectancy'],
            'life_reduction': result['final_life_reduction'],
            'risk_level': result['risk_level'],
            'improvement_potential': result['life_improvement_potential']['improvement_potential']
        })
    
    # 성능 지표 계산
    print("\n 모델 성능 지표:")
    
    # 1. 논리적 일관성 검증
    print("\n 1. 논리적 일관성 검증:")
    healthy_reduction = results[0]['life_reduction']
    moderate_reduction = results[1]['life_reduction']
    risky_reduction = results[2]['life_reduction']
    
    print(f"  건강한 생활습관 수명 단축: {healthy_reduction:.1f}년")
    print(f"  보통 생활습관 수명 단축: {moderate_reduction:.1f}년")
    print(f"  위험한 생활습관 수명 단축: {risky_reduction:.1f}년")
    
    if healthy_reduction < moderate_reduction < risky_reduction:
        print(" 논리적 일관성: 위험도가 높을수록 수명 단축이 증가")
    else:
        print(" 논리적 일관성 문제 발견")
    
    # 2. 연구 기반 신뢰성
    print("\n 2. 연구 기반 신뢰성:")
    research_summary = model.get_research_summary()
    print(f"  총 연구 논문: {research_summary['total_papers']}개")
    print(f"  최신 연구 (2020-2025): {research_summary['recent_papers']}개")
    print(f"  한국인 대상 연구: {research_summary['korean_studies']}개")
    print(f"  메타분석/시스템 리뷰: {research_summary['meta_analyses']}개")
    print(f"  신뢰도 점수: {research_summary['reliability_score']:.0%}")
    
    # 3. 실제 데이터와의 비교
    print("\n3. 실제 데이터와의 비교:")
    
    # Life Expectancy Data에서 한국 데이터 추출
    korea_data = life_expectancy_clean[life_expectancy_clean['Country'] == 'Korea, Republic of']
    if not korea_data.empty:
        actual_life_expectancy = korea_data['Life expectancy'].mean()
        print(f"  실제 한국 평균 수명: {actual_life_expectancy:.1f}세")
        print(f"  모델 기준 수명 (남성): {research_summary['base_life_expectancy']['male']:.1f}세")
        print(f"  모델 기준 수명 (여성): {research_summary['base_life_expectancy']['female']:.1f}세")
        
        # 오차 계산
        male_error = abs(research_summary['base_life_expectancy']['male'] - actual_life_expectancy)
        female_error = abs(research_summary['base_life_expectancy']['female'] - actual_life_expectancy)
        print(f"  남성 기준 수명 오차: {male_error:.1f}세")
        print(f"  여성 기준 수명 오차: {female_error:.1f}세")
    
    # 4. 피처별 기여도 분석
    print("\n4. 피처별 기여도 분석:")
    for test_case in test_cases:
        result = model.predict_life_expectancy(**test_case['params'])
        print(f"\n  {test_case['name']}:")
        for feature, contribution in result['feature_contributions'].items():
            if contribution > 0:
                print(f"    {feature}: {contribution:.1%}")
    
    # 5. 종합 성능 평가
    print("\n5. 종합 성능 평가:")
    
    # 논리적 일관성 점수
    consistency_score = 1.0 if healthy_reduction < moderate_reduction < risky_reduction else 0.5
    
    # 연구 기반 신뢰성 점수
    research_score = research_summary['reliability_score']
    
    # 실제 데이터 일치도 점수
    data_accuracy_score = 0.9 if 'actual_life_expectancy' in locals() and male_error < 5 else 0.7
    
    # 종합 점수
    overall_score = (consistency_score + research_score + data_accuracy_score) / 3
    
    print(f"  논리적 일관성 점수: {consistency_score:.1%}")
    print(f"  연구 기반 신뢰성 점수: {research_score:.1%}")
    print(f"  실제 데이터 일치도 점수: {data_accuracy_score:.1%}")
    print(f"  종합 성능 점수: {overall_score:.1%}")
    
    if overall_score >= 0.9:
        print(" 우수한 성능")
    elif overall_score >= 0.8:
        print(" 양호한 성능")
    elif overall_score >= 0.7:
        print(" 보통 성능")
    else:
        print(" 개선 필요")
    
    return results, overall_score

if __name__ == "__main__":
    results, overall_score = evaluate_model_performance()
