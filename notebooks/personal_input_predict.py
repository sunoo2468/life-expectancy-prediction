#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from model.life_expectancy_prediction_model import LifeExpectancyPredictionModel

def get_user_input():
    """사용자로부터 건강 정보 입력받기"""
    print("=" * 60)
    print("🏥 개인별 수명 예측 시스템")
    print("=" * 60)
    
    # 기본 정보
    print("\n📋 기본 정보")
    print("-" * 30)
    
    while True:
        try:
            age = int(input("나이를 입력하세요 (18-100): "))
            if 18 <= age <= 100:
                break
            else:
                print("❌ 나이는 18-100 사이로 입력해주세요.")
        except ValueError:
            print("❌ 숫자로 입력해주세요.")
    
    while True:
        gender = input("성별을 입력하세요 (남성/여성): ").strip()
        if gender in ['남성', '여성']:
            break
        else:
            print("❌ '남성' 또는 '여성'으로 입력해주세요.")
    
    # 연령대 분류
    if age < 30:
        age_group = 'young'
    elif age < 60:
        age_group = 'middle'
    else:
        age_group = 'elderly'
    
    # 흡연 정보
    print("\n🚬 흡연 정보")
    print("-" * 30)
    while True:
        smoking_input = input("흡연 상태를 선택하세요:\n1. 비흡연자\n2. 과거 흡연자\n3. 현재 흡연자\n선택 (1-3): ")
        if smoking_input in ['1', '2', '3']:
            smoking_status = int(smoking_input) - 1
            break
        else:
            print("❌ 1, 2, 3 중에서 선택해주세요.")
    
    years_since_quit = None
    cigarettes_per_day = 0
    if smoking_status == 1:  # 과거 흡연자
        while True:
            try:
                years_since_quit = int(input("금연한 지 몇 년 되었나요? (0-50): "))
                if 0 <= years_since_quit <= 50:
                    break
                else:
                    print("❌ 0-50 사이로 입력해주세요.")
            except ValueError:
                print("❌ 숫자로 입력해주세요.")
    elif smoking_status == 2:  # 현재 흡연자
        while True:
            try:
                cigarettes_per_day = int(input("하루에 몇 개비를 피우나요? (1-50): "))
                if 1 <= cigarettes_per_day <= 50:
                    break
                else:
                    print("❌ 1-50 사이로 입력해주세요.")
            except ValueError:
                print("❌ 숫자로 입력해주세요.")
    
    # BMI & 허리둘레 정보
    print("\n📏 BMI & 허리둘레 정보")
    print("-" * 30)
    while True:
        try:
            height = float(input("키를 입력하세요 (cm): "))
            if 100 <= height <= 250:
                break
            else:
                print("❌ 100-250cm 사이로 입력해주세요.")
        except ValueError:
            print("❌ 숫자로 입력해주세요.")
    
    while True:
        try:
            weight = float(input("몸무게를 입력하세요 (kg): "))
            if 30 <= weight <= 200:
                break
            else:
                print("❌ 30-200kg 사이로 입력해주세요.")
        except ValueError:
            print("❌ 숫자로 입력해주세요.")
    
    while True:
        try:
            waist_circumference = float(input("허리둘레를 입력하세요 (cm): "))
            if 50 <= waist_circumference <= 200:
                break
            else:
                print("❌ 50-200cm 사이로 입력해주세요.")
        except ValueError:
            print("❌ 숫자로 입력해주세요.")
    
    bmi = weight / ((height / 100) ** 2)
    
    # 알코올 정보
    print("\n🍺 알코올 섭취 정보")
    print("-" * 30)
    while True:
        try:
            drinks_per_week = int(input("일주일에 몇 번 마시나요? (0-7): "))
            if 0 <= drinks_per_week <= 7:
                break
            else:
                print("❌ 0-7 사이로 입력해주세요.")
        except ValueError:
            print("❌ 숫자로 입력해주세요.")
    
    drink_type = 'soju'  # 기본값
    if drinks_per_week > 0:
        print("주로 마시는 술 종류:")
        print("1. 소주/맥주")
        print("2. 와인")
        print("3. 위스키/브랜디")
        while True:
            drink_choice = input("선택 (1-3): ")
            if drink_choice == '1':
                drink_type = 'soju'
                break
            elif drink_choice == '2':
                drink_type = 'wine'
                break
            elif drink_choice == '3':
                drink_type = 'spirit'
                break
            else:
                print("❌ 1, 2, 3 중에서 선택해주세요.")
    
    binge_drinking = False
    if drinks_per_week > 0:
        binge_input = input("한 번에 많이 마시는 편인가요? (예/아니오): ").strip()
        binge_drinking = binge_input == '예'
    
    # 수면 정보
    print("\n😴 수면 정보")
    print("-" * 30)
    while True:
        try:
            sleep_hours = float(input("하루 평균 수면 시간을 입력하세요 (시간): "))
            if 2 <= sleep_hours <= 12:
                break
            else:
                print("❌ 2-12시간 사이로 입력해주세요.")
        except ValueError:
            print("❌ 숫자로 입력해주세요.")
    
    while True:
        try:
            sleep_quality_score = float(input("수면의 질을 1-10점으로 평가하세요 (1=매우 나쁨, 10=매우 좋음): "))
            if 1 <= sleep_quality_score <= 10:
                break
            else:
                print("❌ 1-10 사이로 입력해주세요.")
        except ValueError:
            print("❌ 숫자로 입력해주세요.")
    
    insomnia = False
    sleep_apnea = False
    irregular_schedule = False
    
    if sleep_quality_score < 7:
        insomnia_input = input("불면증이 있나요? (예/아니오): ").strip()
        insomnia = insomnia_input == '예'
        
        apnea_input = input("수면무호흡증이 있나요? (예/아니오): ").strip()
        sleep_apnea = apnea_input == '예'
    
    schedule_input = input("수면 시간이 불규칙한가요? (예/아니오): ").strip()
    irregular_schedule = schedule_input == '예'
    
    # 신체활동 정보
    print("\n🏃‍♂️ 신체활동 정보")
    print("-" * 30)
    while True:
        try:
            weekly_activity_minutes = int(input("일주일 평균 운동 시간을 입력하세요 (분): "))
            if 0 <= weekly_activity_minutes <= 1000:
                break
            else:
                print("❌ 0-1000분 사이로 입력해주세요.")
        except ValueError:
            print("❌ 숫자로 입력해주세요.")
    
    while True:
        try:
            daily_steps = int(input("하루 평균 걸음 수를 입력하세요: "))
            if 0 <= daily_steps <= 20000:
                break
            else:
                print("❌ 0-20000보 사이로 입력해주세요.")
        except ValueError:
            print("❌ 숫자로 입력해주세요.")
    
    print("\n운동 강도:")
    print("1. 낮음 (걷기, 요가)")
    print("2. 중간 (조깅, 자전거)")
    print("3. 높음 (달리기, 수영)")
    print("4. 매우 높음 (격렬한 운동)")
    
    while True:
        intensity_choice = input("선택 (1-4): ")
        if intensity_choice == '1':
            intensity = 'low_intensity'
            break
        elif intensity_choice == '2':
            intensity = 'moderate_intensity'
            break
        elif intensity_choice == '3':
            intensity = 'high_intensity'
            break
        elif intensity_choice == '4':
            intensity = 'very_high_intensity'
            break
        else:
            print("❌ 1, 2, 3, 4 중에서 선택해주세요.")
    
    # 추가 위험 요인
    print("\n⚠️ 추가 위험 요인")
    print("-" * 30)
    sedentary_job = input("앉아서 하는 직업인가요? (예/아니오): ").strip() == '예'
    no_exercise = weekly_activity_minutes == 0
    poor_mobility = input("이동에 어려움이 있나요? (예/아니오): ").strip() == '예'
    chronic_pain = input("만성 통증이 있나요? (예/아니오): ").strip() == '예'
    obesity = bmi >= 30
    poor_diet = input("불규칙한 식습관인가요? (예/아니오): ").strip() == '예'
    
    # 스트레스 수준
    print("\n스트레스 수준:")
    print("1. 낮음")
    print("2. 중간")
    print("3. 높음")
    
    while True:
        stress_choice = input("선택 (1-3): ")
        if stress_choice == '1':
            stress_level = 'low'
            break
        elif stress_choice == '2':
            stress_level = 'medium'
            break
        elif stress_choice == '3':
            stress_level = 'high'
            break
        else:
            print("❌ 1, 2, 3 중에서 선택해주세요.")
    
    return {
        'age': age,
        'gender': gender,
        'age_group': age_group,
        'smoking_status': smoking_status,
        'years_since_quit': years_since_quit,
        'cigarettes_per_day': cigarettes_per_day,
        'height': height,
        'weight': weight,
        'bmi': bmi,
        'waist_circumference': waist_circumference,
        'drinks_per_week': drinks_per_week,
        'drink_type': drink_type,
        'binge_drinking': binge_drinking,
        'sleep_hours': sleep_hours,
        'sleep_quality_score': sleep_quality_score,
        'insomnia': insomnia,
        'sleep_apnea': sleep_apnea,
        'irregular_schedule': irregular_schedule,
        'stress_level': stress_level,
        'weekly_activity_minutes': weekly_activity_minutes,
        'daily_steps': daily_steps,
        'intensity': intensity,
        'sedentary_job': sedentary_job,
        'no_exercise': no_exercise,
        'poor_mobility': poor_mobility,
        'chronic_pain': chronic_pain,
        'obesity': obesity,
        'poor_diet': poor_diet
    }

def print_user_summary(user_data):
    """입력된 사용자 정보 요약 출력"""
    print("\n" + "=" * 60)
    print("📊 입력된 건강 정보 요약")
    print("=" * 60)
    
    print(f"👤 기본 정보: {user_data['age']}세 {user_data['gender']}")
    print(f"📏 신체 정보: 키 {user_data['height']}cm, 몸무게 {user_data['weight']}kg, BMI {user_data['bmi']:.1f}")
    print(f"📏 허리둘레: {user_data['waist_circumference']}cm")
    
    # 흡연 정보
    smoking_status_text = ['비흡연자', '과거 흡연자', '현재 흡연자'][user_data['smoking_status']]
    print(f"🚬 흡연: {smoking_status_text}")
    if user_data['smoking_status'] == 1 and user_data['years_since_quit']:
        print(f"   → 금연 {user_data['years_since_quit']}년")
    elif user_data['smoking_status'] == 2:
        print(f"   → 하루 {user_data['cigarettes_per_day']}개비")
    
    # 알코올 정보
    if user_data['drinks_per_week'] == 0:
        print(f"🍺 알코올: 비음주자")
    else:
        print(f"🍺 알코올: 주 {user_data['drinks_per_week']}회 ({user_data['drink_type']})")
        if user_data['binge_drinking']:
            print("   → 폭음 경향 있음")
    
    # 수면 정보
    print(f"😴 수면: {user_data['sleep_hours']}시간 (품질: {user_data['sleep_quality_score']}/10점)")
    if user_data['insomnia']:
        print("   → 불면증 있음")
    if user_data['sleep_apnea']:
        print("   → 수면무호흡증 있음")
    if user_data['irregular_schedule']:
        print("   → 불규칙한 수면")
    
    # 신체활동 정보
    print(f"🏃‍♂️ 신체활동: 주 {user_data['weekly_activity_minutes']}분, 하루 {user_data['daily_steps']}보")
    intensity_text = {
        'low_intensity': '낮음',
        'moderate_intensity': '중간',
        'high_intensity': '높음',
        'very_high_intensity': '매우 높음'
    }
    print(f"   → 운동 강도: {intensity_text[user_data['intensity']]}")
    
    # 추가 위험 요인
    risk_factors = []
    if user_data['sedentary_job']: risk_factors.append("앉아서 하는 직업")
    if user_data['poor_mobility']: risk_factors.append("이동 어려움")
    if user_data['chronic_pain']: risk_factors.append("만성 통증")
    if user_data['obesity']: risk_factors.append("비만")
    if user_data['poor_diet']: risk_factors.append("불규칙한 식습관")
    
    if risk_factors:
        print(f"⚠️ 추가 위험 요인: {', '.join(risk_factors)}")
    
    print(f"😰 스트레스 수준: {user_data['stress_level']}")

def main():
    """메인 함수"""
    try:
        # 사용자 입력 받기
        user_data = get_user_input()
        
        # 입력 정보 요약 출력
        print_user_summary(user_data)
        
        # 수명 예측 모델 초기화
        print("\n🔬 수명 예측 분석 중...")
        model = LifeExpectancyPredictionModel()
        
        # 수명 예측 실행
        result = model.calculate_life_expectancy_reduction(
            smoking_status=user_data['smoking_status'],
            years_since_quit=user_data['years_since_quit'],
            cigarettes_per_day=user_data['cigarettes_per_day'],
            bmi=user_data['bmi'],
            waist_circumference=user_data['waist_circumference'],
            height=user_data['height'],
            gender='male' if user_data['gender'] == '남성' else 'female',
            age_group=user_data['age_group'],
            drinks_per_week=user_data['drinks_per_week'],
            drink_type=user_data['drink_type'],
            binge_drinking=user_data['binge_drinking'],
            sleep_quality_score=user_data['sleep_quality_score'],
            sleep_hours=user_data['sleep_hours'],
            insomnia=user_data['insomnia'],
            sleep_apnea=user_data['sleep_apnea'],
            irregular_schedule=user_data['irregular_schedule'],
            stress_level=user_data['stress_level'],
            weekly_activity_minutes=user_data['weekly_activity_minutes'],
            daily_steps=user_data['daily_steps'],
            intensity=user_data['intensity'],
            sedentary_job=user_data['sedentary_job'],
            no_exercise=user_data['no_exercise'],
            poor_mobility=user_data['poor_mobility'],
            chronic_pain=user_data['chronic_pain'],
            obesity=user_data['obesity'],
            poor_diet=user_data['poor_diet']
        )
        
        # 결과 출력
        print("\n" + "=" * 60)
        print("🎯 수명 예측 결과")
        print("=" * 60)
        
        print(f"📊 기준 수명: {result['base_life_expectancy']:.1f}세")
        print(f"📉 예상 수명: {result['predicted_life_expectancy']:.1f}세")
        print(f"⏰ 수명 단축: {result['life_reduction']:.1f}년")
        print(f"📈 개선 잠재력: {result['life_improvement_potential']['improvement_potential']:.1f}년")
        
        print(f"\n⚠️ 위험 수준: {result['risk_level']}")
        print(f"📋 종합 위험도: {result['risk_score']:.3f}")
        
        print(f"\n🔍 피처별 기여도:")
        print("-" * 30)
        for feature, percentage in result['feature_contributions'].items():
            print(f"  {feature}: {percentage:.1f}%")
        
        print(f"\n💡 건강 권고사항:")
        print("-" * 30)
        for i, recommendation in enumerate(result['recommendations'], 1):
            print(f"  {i}. {recommendation}")
        
        print(f"\n📚 연구 근거:")
        print("-" * 30)
        print(f"  • 총 {result['research_credibility']['total_papers']}개 논문 기반")
        print(f"  • 최근 5년 내 {result['research_credibility']['recent_papers']}개 연구")
        print(f"  • 한국인 대상 연구 {result['research_credibility']['korean_studies']}개")
        print(f"  • 메타분석 {result['research_credibility']['meta_analyses']}개")
        print(f"  • 신뢰도 점수: {result['research_credibility']['reliability_score']:.2f}")
        
        print("\n" + "=" * 60)
        print("✅ 분석 완료!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n❌ 사용자가 중단했습니다.")
    except Exception as e:
        print(f"\n❌ 오류가 발생했습니다: {str(e)}")

if __name__ == "__main__":
    main()
