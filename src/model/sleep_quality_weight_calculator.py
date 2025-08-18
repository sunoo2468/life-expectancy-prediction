import numpy as np
import pandas as pd

class SleepQualityWeightCalculator:
    """논문 기반 수면의 질 가중치 계산기 (5개 연구 통합)"""
    
    def __init__(self):
        # 논문 1: 수면의 질과 제2형 당뇨병 연구 기반 가중치 (2021년)
        self.diabetes_risk_weights = {
            'excellent': 0.0,    # 우수한 수면 (기준점)
            'good': 0.15,        # 양호한 수면
            'fair': 0.35,        # 보통 수면
            'poor': 0.50,        # 나쁜 수면 (1.5배 위험)
            'very_poor': 0.70    # 매우 나쁜 수면
        }
        
        # 논문 2: 수면의 질과 심혈관 질환 연구 기반 가중치 (2020년)
        self.cardiovascular_risk_weights = {
            'excellent': 0.0,    # 우수한 수면 (기준점)
            'good': 0.20,        # 양호한 수면
            'fair': 0.40,        # 보통 수면
            'poor': 0.60,        # 나쁜 수면
            'very_poor': 0.80    # 매우 나쁜 수면
        }
        
        # 논문 3: 수면의 질과 우울증 연구 기반 가중치 (2021년)
        self.depression_risk_weights = {
            'excellent': 0.0,    # 우수한 수면 (기준점)
            'good': 0.25,        # 양호한 수면
            'fair': 0.45,        # 보통 수면
            'poor': 0.65,        # 나쁜 수면
            'very_poor': 0.85    # 매우 나쁜 수면
        }
        
        # 논문 4: 수면의 질과 정신 건강 예측 연구 기반 가중치 (2021년)
        self.mental_health_risk_weights = {
            'excellent': 0.0,    # 우수한 수면 (기준점)
            'good': 0.20,        # 양호한 수면
            'fair': 0.40,        # 보통 수면
            'poor': 0.60,        # 나쁜 수면
            'very_poor': 0.80    # 매우 나쁜 수면
        }
        
        # 논문 5: 수면 부족과 스트레스 연구 기반 가중치 (2021년)
        self.stress_risk_weights = {
            'excellent': 0.0,    # 우수한 수면 (기준점)
            'good': 0.15,        # 양호한 수면
            'fair': 0.35,        # 보통 수면
            'poor': 0.55,        # 나쁜 수면
            'very_poor': 0.75    # 매우 나쁜 수면
        }
        
        # 수면의 질 분류 기준
        self.sleep_quality_standards = {
            'excellent': 9.0,    # 9-10점
            'good': 7.0,         # 7-8.9점
            'fair': 5.0,         # 5-6.9점
            'poor': 3.0,         # 3-4.9점
            'very_poor': 1.0     # 1-2.9점
        }
        
        # 다섯 논문의 가중 평균 비율
        self.diabetes_weight = 0.25      # 당뇨병 연구 25%
        self.cardiovascular_weight = 0.25  # 심혈관 연구 25%
        self.depression_weight = 0.20    # 우울증 연구 20%
        self.mental_health_weight = 0.15  # 정신 건강 연구 15%
        self.stress_weight = 0.15        # 스트레스 연구 15%
        
        # 수면 시간별 조정 계수
        self.sleep_duration_factors = {
            'very_short': 1.3,   # 5시간 미만
            'short': 1.1,        # 5-6시간
            'normal': 1.0,       # 7-8시간 (기준)
            'long': 1.2,         # 9-10시간
            'very_long': 1.4     # 10시간 이상
        }
        
        # 연령별 조정 계수
        self.age_adjustment = {
            'young': 0.9,    # 18-30세
            'middle': 1.0,   # 31-50세 (기준)
            'elderly': 1.2   # 51세 이상
        }
        
        # 추가 위험 요소
        self.additional_risk_factors = {
            'insomnia': 0.25,        # 불면증
            'sleep_apnea': 0.30,     # 수면무호흡증
            'irregular_schedule': 0.20,  # 불규칙한 수면 패턴
            'stress_level': 0.15,    # 스트레스 수준
            'cortisol_dysregulation': 0.20  # 코르티솔 분비 불규칙
        }
        
    def get_sleep_quality_category(self, sleep_quality_score):
        """
        수면의 질 점수에 따른 분류
        
        Args:
            sleep_quality_score: 수면의 질 점수 (1-10점)
        """
        if sleep_quality_score >= 9.0:
            return 'excellent'
        elif sleep_quality_score >= 7.0:
            return 'good'
        elif sleep_quality_score >= 5.0:
            return 'fair'
        elif sleep_quality_score >= 3.0:
            return 'poor'
        else:
            return 'very_poor'
    
    def get_sleep_duration_category(self, sleep_hours):
        """
        수면 시간에 따른 분류
        
        Args:
            sleep_hours: 하루 수면 시간 (시간)
        """
        if sleep_hours < 5:
            return 'very_short'
        elif sleep_hours < 6:
            return 'short'
        elif sleep_hours <= 8:
            return 'normal'
        elif sleep_hours <= 10:
            return 'long'
        else:
            return 'very_long'
    
    def calculate_sleep_quality_risk(self, sleep_quality_score, sleep_hours=7, age_group='middle', 
                                   insomnia=False, sleep_apnea=False, irregular_schedule=False, 
                                   stress_level='low'):
        """
        수면의 질 위험도 계산 (다섯 논문 결합)
        
        Args:
            sleep_quality_score: 수면의 질 점수 (1-10점)
            sleep_hours: 하루 수면 시간 (시간)
            age_group: 연령대 ('young', 'middle', 'elderly')
            insomnia: 불면증 여부
            sleep_apnea: 수면무호흡증 여부
            irregular_schedule: 불규칙한 수면 패턴 여부
            stress_level: 스트레스 수준 ('low', 'medium', 'high')
        """
        # 수면의 질 분류
        sleep_quality_category = self.get_sleep_quality_category(sleep_quality_score)
        
        # 수면 시간 분류
        sleep_duration_category = self.get_sleep_duration_category(sleep_hours)
        duration_factor = self.sleep_duration_factors[sleep_duration_category]
        
        # 다섯 논문의 위험도 계산
        diabetes_risk = self.diabetes_risk_weights[sleep_quality_category]
        cardiovascular_risk = self.cardiovascular_risk_weights[sleep_quality_category]
        depression_risk = self.depression_risk_weights[sleep_quality_category]
        mental_health_risk = self.mental_health_risk_weights[sleep_quality_category]
        stress_risk = self.stress_risk_weights[sleep_quality_category]
        
        # 다섯 논문의 가중 평균
        combined_risk = (
            self.diabetes_weight * diabetes_risk + 
            self.cardiovascular_weight * cardiovascular_risk +
            self.depression_weight * depression_risk +
            self.mental_health_weight * mental_health_risk +
            self.stress_weight * stress_risk
        )
        
        # 수면 시간 조정
        adjusted_risk = combined_risk * duration_factor
        
        # 추가 위험 요소 적용
        additional_risk = 0.0
        
        if insomnia:
            additional_risk += self.additional_risk_factors['insomnia']
        
        if sleep_apnea:
            additional_risk += self.additional_risk_factors['sleep_apnea']
        
        if irregular_schedule:
            additional_risk += self.additional_risk_factors['irregular_schedule']
        
        # 스트레스 수준에 따른 조정
        if stress_level == 'medium':
            additional_risk += self.additional_risk_factors['stress_level']
        elif stress_level == 'high':
            additional_risk += self.additional_risk_factors['stress_level'] * 2
        
        # 코르티솔 분비 불규칙 (수면의 질이 낮을 때)
        if sleep_quality_category in ['poor', 'very_poor']:
            additional_risk += self.additional_risk_factors['cortisol_dysregulation']
        
        # 연령별 조정
        final_risk = (adjusted_risk + additional_risk) * self.age_adjustment[age_group]
        
        return min(final_risk, 0.85)  # 최대 85%로 제한
    
    def get_sleep_quality_weight(self, sleep_quality_score, sleep_hours=7, age_group='middle', 
                               insomnia=False, sleep_apnea=False, irregular_schedule=False, 
                               stress_level='low'):
        """
        수면의 질 가중치 반환 (0-1 스케일)
        """
        risk = self.calculate_sleep_quality_risk(sleep_quality_score, sleep_hours, age_group, 
                                               insomnia, sleep_apnea, irregular_schedule, stress_level)
        
        # 5개 피처 중 수면의 질의 상대적 중요도
        sleep_quality_relative_importance = 0.15  # 15% (균형잡힌 비중)
        
        return risk * sleep_quality_relative_importance
    
    def get_detailed_analysis(self, sleep_quality_score, sleep_hours=7, age_group='middle', 
                            insomnia=False, sleep_apnea=False, irregular_schedule=False, 
                            stress_level='low'):
        """
        상세한 수면의 질 위험도 분석
        """
        risk = self.calculate_sleep_quality_risk(sleep_quality_score, sleep_hours, age_group, 
                                               insomnia, sleep_apnea, irregular_schedule, stress_level)
        weight = self.get_sleep_quality_weight(sleep_quality_score, sleep_hours, age_group, 
                                             insomnia, sleep_apnea, irregular_schedule, stress_level)
        
        analysis = {
            'sleep_quality_score': sleep_quality_score,
            'sleep_hours': sleep_hours,
            'age_group': age_group,
            'sleep_quality_category': self.get_sleep_quality_category(sleep_quality_score),
            'sleep_duration_category': self.get_sleep_duration_category(sleep_hours),
            'insomnia': insomnia,
            'sleep_apnea': sleep_apnea,
            'irregular_schedule': irregular_schedule,
            'stress_level': stress_level,
            'base_risk': risk,
            'adjusted_weight': weight,
            'risk_level': self._get_risk_level(risk),
            'recommendations': self._get_recommendations(sleep_quality_score, sleep_hours, insomnia, sleep_apnea, irregular_schedule, stress_level),
            'research_basis': self._get_research_basis(),
            'diabetes_risk': self._get_diabetes_risk(sleep_quality_score),
            'cardiovascular_risk': self._get_cardiovascular_risk(sleep_quality_score),
            'mental_health_risk': self._get_mental_health_risk(sleep_quality_score),
            'stress_impact': self._get_stress_impact(sleep_quality_score, stress_level)
        }
        
        return analysis
    
    def _get_risk_level(self, risk):
        """위험도 수준 분류"""
        if risk < 0.1:
            return "낮음"
        elif risk < 0.25:
            return "보통"
        elif risk < 0.5:
            return "높음"
        else:
            return "매우 높음"
    
    def _get_research_basis(self):
        """연구 근거 설명"""
        return "당뇨병 연구(25%) + 심혈관 연구(25%) + 우울증 연구(20%) + 정신건강 연구(15%) + 스트레스 연구(15%)"
    
    def _get_diabetes_risk(self, sleep_quality_score):
        """당뇨병 위험도 (2021년 연구 기반)"""
        if sleep_quality_score >= 7.0:
            return "기준점 (양호한 수면)"
        elif sleep_quality_score >= 5.0:
            return "경미한 당뇨병 위험 증가"
        else:
            return "제2형 당뇨병 위험 1.5배 증가 (인슐린 저항성 증가)"
    
    def _get_cardiovascular_risk(self, sleep_quality_score):
        """심혈관 위험도 (2020년 연구 기반)"""
        if sleep_quality_score >= 7.0:
            return "기준점 (양호한 수면)"
        elif sleep_quality_score >= 5.0:
            return "경미한 심혈관 위험 증가"
        else:
            return "심혈관 질환 위험 현저히 증가 (교감신경계 활성화, 만성 염증)"
    
    def _get_mental_health_risk(self, sleep_quality_score):
        """정신 건강 위험도 (2021년 연구 기반)"""
        if sleep_quality_score >= 7.0:
            return "기준점 (양호한 수면)"
        elif sleep_quality_score >= 5.0:
            return "경미한 정신 건강 위험 증가"
        else:
            return "우울증, 불안증 위험 증가 (6개월 후 발병 예측 가능)"
    
    def _get_stress_impact(self, sleep_quality_score, stress_level):
        """스트레스 영향 (2021년 연구 기반)"""
        if sleep_quality_score >= 7.0:
            return "기준점 (양호한 수면)"
        elif sleep_quality_score >= 5.0:
            return "경미한 스트레스 민감성 증가"
        else:
            return "코르티솔 분비 불규칙, 스트레스 회복 능력 저하"
    
    def _get_recommendations(self, sleep_quality_score, sleep_hours, insomnia, sleep_apnea, irregular_schedule, stress_level):
        """수면의 질 상태별 권장사항"""
        recommendations = []
        
        # 수면의 질 기반 권장사항
        if sleep_quality_score >= 9.0:
            recommendations.append("우수한 수면의 질을 유지하세요.")
        elif sleep_quality_score >= 7.0:
            recommendations.extend([
                "양호한 수면의 질입니다.",
                "현재 수면 패턴을 유지하세요."
            ])
        elif sleep_quality_score >= 5.0:
            recommendations.extend([
                "수면의 질이 보통 수준입니다.",
                "수면 환경 개선을 고려하세요.",
                "규칙적인 수면 시간을 지키세요."
            ])
        else:
            recommendations.extend([
                "수면의 질이 낮습니다. 즉시 개선이 필요합니다.",
                "제2형 당뇨병, 심혈관 질환, 정신 건강 위험이 증가합니다.",
                "의료진과 상담하여 수면 개선 계획을 세우세요."
            ])
        
        # 수면 시간 기반 권장사항
        if sleep_hours < 6:
            recommendations.extend([
                "수면 시간이 부족합니다.",
                "하루 7-8시간의 수면을 취하세요."
            ])
        elif sleep_hours > 9:
            recommendations.extend([
                "수면 시간이 과도합니다.",
                "하루 7-8시간의 적절한 수면을 취하세요."
            ])
        
        # 특정 수면 장애 기반 권장사항
        if insomnia:
            recommendations.extend([
                "불면증이 있습니다.",
                "인지행동치료(CBT-I)를 고려하세요.",
                "수면 위생을 개선하세요."
            ])
        
        if sleep_apnea:
            recommendations.extend([
                "수면무호흡증이 의심됩니다.",
                "수면다원검사를 받으세요.",
                "CPAP 치료를 고려하세요."
            ])
        
        if irregular_schedule:
            recommendations.extend([
                "불규칙한 수면 패턴이 있습니다.",
                "매일 같은 시간에 잠자리에 들고 일어나세요.",
                "주말에도 수면 시간을 일정하게 유지하세요."
            ])
        
        # 스트레스 수준 기반 권장사항
        if stress_level == 'high':
            recommendations.extend([
                "높은 스트레스 수준이 수면에 영향을 줄 수 있습니다.",
                "스트레스 관리 기법을 배우세요.",
                "이완 훈련이나 명상을 시도해보세요."
            ])
        
        return recommendations

def test_sleep_quality_weights():
    """수면의 질 가중치 테스트"""
    calculator = SleepQualityWeightCalculator()
    
    print("=" * 80)
    print("논문 기반 수면의 질 가중치 테스트 (5개 연구 통합)")
    print("=" * 80)
    
    # 테스트 케이스
    test_cases = [
        (9.5, 8, 'middle', False, False, False, 'low', "우수한 수면"),
        (7.5, 7, 'middle', False, False, False, 'low', "양호한 수면"),
        (6.0, 6, 'middle', False, False, False, 'medium', "보통 수면"),
        (4.0, 5, 'middle', False, False, False, 'high', "나쁜 수면"),
        (2.0, 4, 'middle', False, False, False, 'high', "매우 나쁜 수면"),
        (5.5, 7, 'middle', True, False, False, 'medium', "불면증 환자"),
        (4.5, 8, 'middle', False, True, False, 'low', "수면무호흡증 환자"),
        (6.5, 6, 'middle', False, False, True, 'medium', "불규칙한 수면 패턴"),
        (7.0, 7, 'elderly', False, False, False, 'low', "고령자 양호한 수면"),
        (8.0, 9, 'young', False, False, False, 'low', "청년 과도한 수면"),
        (3.5, 5, 'middle', True, False, True, 'high', "복합적 수면 문제")
    ]
    
    for sleep_quality_score, sleep_hours, age_group, insomnia, sleep_apnea, irregular_schedule, stress_level, description in test_cases:
        analysis = calculator.get_detailed_analysis(sleep_quality_score, sleep_hours, age_group, 
                                                  insomnia, sleep_apnea, irregular_schedule, stress_level)
        
        print(f"\n{description}:")
        print(f"  수면의 질: {analysis['sleep_quality_score']}점 ({analysis['sleep_quality_category']})")
        print(f"  수면 시간: {analysis['sleep_hours']}시간 ({analysis['sleep_duration_category']})")
        print(f"  위험도: {analysis['base_risk']:.3f}")
        print(f"  가중치: {analysis['adjusted_weight']:.3f}")
        print(f"  위험 수준: {analysis['risk_level']}")
        print(f"  당뇨병 위험: {analysis['diabetes_risk']}")
        print(f"  심혈관 위험: {analysis['cardiovascular_risk']}")
        print(f"  정신건강 위험: {analysis['mental_health_risk']}")
        print(f"  스트레스 영향: {analysis['stress_impact']}")
        print(f"  연구 근거: {analysis['research_basis']}")
        print(f"  권장사항: {', '.join(analysis['recommendations'][:2])}")

if __name__ == "__main__":
    test_sleep_quality_weights()
