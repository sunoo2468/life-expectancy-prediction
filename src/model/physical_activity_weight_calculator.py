import numpy as np
import pandas as pd

class PhysicalActivityWeightCalculator:
    """논문 기반 신체활동 가중치 계산기 (3개 연구 통합)"""
    
    def __init__(self):
        # 연구 1: 신체활동과 모든 원인 사망률 연구 기반 가중치 (2025년)
        self.mortality_risk_weights = {
            'inactive': 1.0,      # 비활동적 (기준점)
            'low_activity': 0.85, # 낮은 활동 (사망 위험 15% 감소)
            'moderate_activity': 0.70,  # 중간 활동 (사망 위험 30% 감소)
            'high_activity': 0.60,      # 높은 활동 (사망 위험 40% 감소)
            'very_high_activity': 0.50  # 매우 높은 활동 (사망 위험 50% 감소)
        }
        
        # 연구 2: 일상 활동(걸음 수)과 심혈관 질환 연구 기반 가중치 (2024년)
        self.daily_activity_risk_weights = {
            'sedentary': 1.0,     # 앉아있는 생활 (기준점)
            'low_steps': 0.90,    # 낮은 걸음 수 (위험 10% 감소)
            'moderate_steps': 0.75,  # 중간 걸음 수 (위험 25% 감소)
            'high_steps': 0.60,      # 높은 걸음 수 (위험 40% 감소)
            'optimal_steps': 0.50     # 최적 걸음 수 (위험 50% 감소)
        }
        
        # 연구 3: WHO 가이드라인 기반 암 및 만성질환 위험 가중치
        self.chronic_disease_risk_weights = {
            'inactive': 1.0,      # 비활동적 (기준점)
            'low_activity': 0.80, # 낮은 활동 (만성질환 위험 20% 감소)
            'moderate_activity': 0.65,  # 중간 활동 (만성질환 위험 35% 감소)
            'high_activity': 0.50,      # 높은 활동 (만성질환 위험 50% 감소)
            'very_high_activity': 0.40  # 매우 높은 활동 (만성질환 위험 60% 감소)
        }
        
        # 신체활동 수준 분류 기준 (주당 분)
        self.activity_standards = {
            'inactive': 0,        # 0분
            'low_activity': 75,   # 75분 미만
            'moderate_activity': 150,  # 75-149분
            'high_activity': 300,      # 150-299분
            'very_high_activity': 450  # 300분 이상
        }
        
        # 걸음 수 분류 기준 (하루)
        self.step_standards = {
            'sedentary': 3000,    # 3,000보 미만
            'low_steps': 5000,    # 3,000-4,999보
            'moderate_steps': 7000,  # 5,000-6,999보
            'high_steps': 9000,      # 7,000-8,999보
            'optimal_steps': 11000    # 9,000보 이상
        }
        
        # 세 연구의 가중 평균 비율
        self.mortality_weight = 0.40      # 사망률 연구 40%
        self.daily_activity_weight = 0.35  # 일상 활동 연구 35%
        self.chronic_disease_weight = 0.25  # 만성질환 연구 25%
        
        # 연령별 조정 계수
        self.age_adjustment = {
            'young': 0.8,     # 18-30세 (활동적)
            'middle': 1.0,    # 31-50세 (기준)
            'elderly': 1.3    # 51세 이상 (활동 제한)
        }
        
        # 성별 조정 계수
        self.gender_adjustment = {
            'male': 1.0,      # 남성 (기준)
            'female': 0.9     # 여성 (일반적으로 더 활동적)
        }
        
        # 추가 위험 요소
        self.additional_risk_factors = {
            'sedentary_job': 0.20,     # 앉아있는 직업
            'no_exercise': 0.25,       # 운동 안함
            'poor_mobility': 0.30,     # 이동성 저하
            'chronic_pain': 0.15,      # 만성 통증
            'obesity': 0.20,           # 비만
            'smoking': 0.15,           # 흡연
            'poor_diet': 0.10          # 불량한 식습관
        }
        
        # 활동 강도별 보너스
        self.intensity_bonus = {
            'low_intensity': 0.0,      # 저강도 (걷기)
            'moderate_intensity': 0.1,  # 중강도 (빠른 걷기, 조깅)
            'high_intensity': 0.2,     # 고강도 (달리기, 헬스)
            'very_high_intensity': 0.3  # 매우 고강도 (격렬한 운동)
        }
        
    def get_activity_level(self, weekly_activity_minutes):
        """
        주간 신체활동 시간에 따른 분류
        
        Args:
            weekly_activity_minutes: 주간 신체활동 시간 (분)
        """
        if weekly_activity_minutes == 0:
            return 'inactive'
        elif weekly_activity_minutes < 75:
            return 'low_activity'
        elif weekly_activity_minutes < 150:
            return 'moderate_activity'
        elif weekly_activity_minutes < 300:
            return 'high_activity'
        else:
            return 'very_high_activity'
    
    def get_step_level(self, daily_steps):
        """
        하루 걸음 수에 따른 분류
        
        Args:
            daily_steps: 하루 걸음 수
        """
        if daily_steps < 3000:
            return 'sedentary'
        elif daily_steps < 5000:
            return 'low_steps'
        elif daily_steps < 7000:
            return 'moderate_steps'
        elif daily_steps < 9000:
            return 'high_steps'
        else:
            return 'optimal_steps'
    
    def calculate_physical_activity_risk(self, weekly_activity_minutes, daily_steps, 
                                       age_group='middle', gender='male', intensity='moderate_intensity',
                                       sedentary_job=False, no_exercise=False, poor_mobility=False,
                                       chronic_pain=False, obesity=False, smoking=False, poor_diet=False):
        """
        신체활동 위험도 계산 (세 연구 결합)
        
        Args:
            weekly_activity_minutes: 주간 신체활동 시간 (분)
            daily_steps: 하루 걸음 수
            age_group: 연령대 ('young', 'middle', 'elderly')
            gender: 성별 ('male', 'female')
            intensity: 활동 강도 ('low_intensity', 'moderate_intensity', 'high_intensity', 'very_high_intensity')
            sedentary_job: 앉아있는 직업 여부
            no_exercise: 운동 안함 여부
            poor_mobility: 이동성 저하 여부
            chronic_pain: 만성 통증 여부
            obesity: 비만 여부
            smoking: 흡연 여부
            poor_diet: 불량한 식습관 여부
        """
        # 활동 수준 분류
        activity_level = self.get_activity_level(weekly_activity_minutes)
        step_level = self.get_step_level(daily_steps)
        
        # 세 연구의 위험도 계산
        mortality_risk = self.mortality_risk_weights[activity_level]
        daily_activity_risk = self.daily_activity_risk_weights[step_level]
        chronic_disease_risk = self.chronic_disease_risk_weights[activity_level]
        
        # 세 연구의 가중 평균
        combined_risk = (
            self.mortality_weight * mortality_risk + 
            self.daily_activity_weight * daily_activity_risk +
            self.chronic_disease_weight * chronic_disease_risk
        )
        
        # 활동 강도 보너스 적용 (위험도 감소)
        intensity_bonus = self.intensity_bonus[intensity]
        adjusted_risk = combined_risk * (1 - intensity_bonus)
        
        # 추가 위험 요소 적용
        additional_risk = 0.0
        
        if sedentary_job:
            additional_risk += self.additional_risk_factors['sedentary_job']
        
        if no_exercise:
            additional_risk += self.additional_risk_factors['no_exercise']
        
        if poor_mobility:
            additional_risk += self.additional_risk_factors['poor_mobility']
        
        if chronic_pain:
            additional_risk += self.additional_risk_factors['chronic_pain']
        
        if obesity:
            additional_risk += self.additional_risk_factors['obesity']
        
        if smoking:
            additional_risk += self.additional_risk_factors['smoking']
        
        if poor_diet:
            additional_risk += self.additional_risk_factors['poor_diet']
        
        # 연령 및 성별 조정
        final_risk = (adjusted_risk + additional_risk) * self.age_adjustment[age_group] * self.gender_adjustment[gender]
        
        return min(final_risk, 0.90)  # 최대 90%로 제한
    
    def get_physical_activity_weight(self, weekly_activity_minutes, daily_steps, 
                                   age_group='middle', gender='male', intensity='moderate_intensity',
                                   sedentary_job=False, no_exercise=False, poor_mobility=False,
                                   chronic_pain=False, obesity=False, smoking=False, poor_diet=False):
        """
        신체활동 가중치 반환 (0-1 스케일)
        """
        risk = self.calculate_physical_activity_risk(weekly_activity_minutes, daily_steps, 
                                                   age_group, gender, intensity, sedentary_job, 
                                                   no_exercise, poor_mobility, chronic_pain, 
                                                   obesity, smoking, poor_diet)
        
        # 5개 피처 중 신체활동의 상대적 중요도
        physical_activity_relative_importance = 0.20  # 20% (균형잡힌 비중)
        
        return risk * physical_activity_relative_importance
    
    def get_detailed_analysis(self, weekly_activity_minutes, daily_steps, 
                            age_group='middle', gender='male', intensity='moderate_intensity',
                            sedentary_job=False, no_exercise=False, poor_mobility=False,
                            chronic_pain=False, obesity=False, smoking=False, poor_diet=False):
        """
        상세한 신체활동 위험도 분석
        """
        risk = self.calculate_physical_activity_risk(weekly_activity_minutes, daily_steps, 
                                                   age_group, gender, intensity, sedentary_job, 
                                                   no_exercise, poor_mobility, chronic_pain, 
                                                   obesity, smoking, poor_diet)
        weight = self.get_physical_activity_weight(weekly_activity_minutes, daily_steps, 
                                                 age_group, gender, intensity, sedentary_job, 
                                                 no_exercise, poor_mobility, chronic_pain, 
                                                 obesity, smoking, poor_diet)
        
        analysis = {
            'weekly_activity_minutes': weekly_activity_minutes,
            'daily_steps': daily_steps,
            'age_group': age_group,
            'gender': gender,
            'intensity': intensity,
            'activity_level': self.get_activity_level(weekly_activity_minutes),
            'step_level': self.get_step_level(daily_steps),
            'sedentary_job': sedentary_job,
            'no_exercise': no_exercise,
            'poor_mobility': poor_mobility,
            'chronic_pain': chronic_pain,
            'obesity': obesity,
            'smoking': smoking,
            'poor_diet': poor_diet,
            'base_risk': risk,
            'adjusted_weight': weight,
            'risk_level': self._get_risk_level(risk),
            'recommendations': self._get_recommendations(weekly_activity_minutes, daily_steps, intensity, sedentary_job, no_exercise, poor_mobility, chronic_pain, obesity, smoking, poor_diet),
            'research_basis': self._get_research_basis(),
            'mortality_risk': self._get_mortality_risk(weekly_activity_minutes),
            'daily_activity_risk': self._get_daily_activity_risk(daily_steps),
            'chronic_disease_risk': self._get_chronic_disease_risk(weekly_activity_minutes),
            'health_benefits': self._get_health_benefits(weekly_activity_minutes, daily_steps, intensity)
        }
        
        return analysis
    
    def _get_risk_level(self, risk):
        """위험도 수준 분류"""
        if risk < 0.3:
            return "매우 낮음"
        elif risk < 0.5:
            return "낮음"
        elif risk < 0.7:
            return "보통"
        else:
            return "높음"
    
    def _get_research_basis(self):
        """연구 근거 설명"""
        return "사망률 연구(40%) + 일상활동 연구(35%) + 만성질환 연구(25%)"
    
    def _get_mortality_risk(self, weekly_activity_minutes):
        """사망률 위험도 (2025년 연구 기반)"""
        if weekly_activity_minutes >= 150:
            return "모든 원인 사망 위험 30-40% 감소 (WHO 권장 기준)"
        elif weekly_activity_minutes >= 75:
            return "사망 위험 15% 감소"
        elif weekly_activity_minutes > 0:
            return "사망 위험 경미한 감소"
        else:
            return "비활동적 생활로 사망 위험 증가"
    
    def _get_daily_activity_risk(self, daily_steps):
        """일상 활동 위험도 (2024년 연구 기반)"""
        if daily_steps >= 9000:
            return "조기 사망 위험 현저히 감소 (8,000-10,000보)"
        elif daily_steps >= 7000:
            return "사망 위험 25% 감소"
        elif daily_steps >= 5000:
            return "사망 위험 10% 감소"
        else:
            return "앉아있는 생활로 질병 위험 증가"
    
    def _get_chronic_disease_risk(self, weekly_activity_minutes):
        """만성질환 위험도 (WHO 가이드라인 기반)"""
        if weekly_activity_minutes >= 300:
            return "심장병, 뇌졸중, 당뇨병, 암 위험 50% 감소"
        elif weekly_activity_minutes >= 150:
            return "만성질환 위험 35% 감소"
        elif weekly_activity_minutes >= 75:
            return "만성질환 위험 20% 감소"
        else:
            return "신체활동 부족으로 만성질환 위험 증가"
    
    def _get_health_benefits(self, weekly_activity_minutes, daily_steps, intensity):
        """건강상 이익"""
        benefits = []
        
        if weekly_activity_minutes >= 150:
            benefits.append("WHO 권장 기준 달성")
        
        if daily_steps >= 8000:
            benefits.append("치매, 뇌졸중, 불안, 수면 장애 위험 14-40% 감소")
        
        if intensity in ['high_intensity', 'very_high_intensity']:
            benefits.append("고강도 운동으로 추가 건강상 이익")
        
        if weekly_activity_minutes > 0:
            benefits.append("운동을 시작하기에 늦은 시점은 없다")
        
        return benefits
    
    def _get_recommendations(self, weekly_activity_minutes, daily_steps, intensity, 
                           sedentary_job, no_exercise, poor_mobility, chronic_pain, 
                           obesity, smoking, poor_diet):
        """신체활동 상태별 권장사항"""
        recommendations = []
        
        # 주간 활동량 기반 권장사항
        if weekly_activity_minutes == 0:
            recommendations.extend([
                "신체활동이 전혀 없습니다. 즉시 시작하세요.",
                "WHO 권장: 주당 150분 중강도 활동",
                "걷기부터 시작하여 점진적으로 늘리세요."
            ])
        elif weekly_activity_minutes < 75:
            recommendations.extend([
                "신체활동이 부족합니다.",
                "주당 75분 이상으로 늘리세요.",
                "일상생활에서 활동량을 늘리는 방법을 찾으세요."
            ])
        elif weekly_activity_minutes < 150:
            recommendations.extend([
                "WHO 권장 기준에 근접했습니다.",
                "주당 150분 이상으로 늘려 최적 건강상태를 달성하세요.",
                "현재 활동을 유지하면서 점진적으로 늘리세요."
            ])
        else:
            recommendations.extend([
                "우수한 신체활동 수준입니다.",
                "현재 활동을 유지하세요.",
                "다양한 운동을 통해 전신 건강을 증진하세요."
            ])
        
        # 걸음 수 기반 권장사항
        if daily_steps < 5000:
            recommendations.extend([
                "하루 걸음 수가 부족합니다.",
                "목표: 하루 8,000-10,000보",
                "계단 이용, 짧은 거리 걷기 등으로 늘리세요."
            ])
        elif daily_steps < 8000:
            recommendations.extend([
                "걸음 수가 보통 수준입니다.",
                "하루 8,000보 이상을 목표로 하세요.",
                "산책, 조깅 등으로 활동량을 늘리세요."
            ])
        else:
            recommendations.append("우수한 일상 활동 수준입니다.")
        
        # 특정 상황별 권장사항
        if sedentary_job:
            recommendations.extend([
                "앉아있는 직업입니다.",
                "1시간마다 5분씩 일어나서 움직이세요.",
                "스탠딩 데스크, 운동용 의자 등을 고려하세요."
            ])
        
        if no_exercise:
            recommendations.extend([
                "정기적인 운동이 없습니다.",
                "주 3-4회, 30분씩 운동을 시작하세요.",
                "걷기, 수영, 자전거 등 저부하 운동부터 시작하세요."
            ])
        
        if poor_mobility:
            recommendations.extend([
                "이동성에 제한이 있습니다.",
                "의료진과 상담하여 적절한 운동을 찾으세요.",
                "물리치료, 수중운동 등을 고려하세요."
            ])
        
        if chronic_pain:
            recommendations.extend([
                "만성 통증이 있습니다.",
                "통증을 악화시키지 않는 운동을 선택하세요.",
                "의료진과 상담하여 안전한 운동 계획을 세우세요."
            ])
        
        if obesity:
            recommendations.extend([
                "비만 상태입니다.",
                "체중 감량을 위한 운동과 식이요법을 병행하세요.",
                "저부하 유산소 운동부터 시작하세요."
            ])
        
        if smoking:
            recommendations.extend([
                "흡연 중입니다.",
                "금연과 함께 신체활동을 늘리면 건강상 이익이 배가됩니다.",
                "금연 후 운동 능력이 크게 향상됩니다."
            ])
        
        if poor_diet:
            recommendations.extend([
                "식습관이 불량합니다.",
                "운동과 함께 균형 잡힌 식단을 유지하세요.",
                "운동 전후 영양 섭취를 고려하세요."
            ])
        
        return recommendations

def test_physical_activity_weights():
    """신체활동 가중치 테스트"""
    calculator = PhysicalActivityWeightCalculator()
    
    print("=" * 80)
    print("논문 기반 신체활동 가중치 테스트 (3개 연구 통합)")
    print("=" * 80)
    
    # 테스트 케이스
    test_cases = [
        (0, 2000, 'middle', 'male', 'low_intensity', True, True, False, False, False, False, False, "완전 비활동적"),
        (50, 4000, 'middle', 'male', 'moderate_intensity', True, False, False, False, False, False, False, "낮은 활동"),
        (120, 6000, 'middle', 'male', 'moderate_intensity', False, False, False, False, False, False, False, "중간 활동"),
        (180, 8000, 'middle', 'male', 'high_intensity', False, False, False, False, False, False, False, "높은 활동"),
        (300, 10000, 'middle', 'male', 'high_intensity', False, False, False, False, False, False, False, "매우 높은 활동"),
        (150, 7000, 'elderly', 'female', 'moderate_intensity', False, False, True, False, False, False, False, "고령자 중간 활동"),
        (200, 9000, 'young', 'male', 'very_high_intensity', False, False, False, False, True, False, False, "청년 고강도 활동"),
        (100, 5000, 'middle', 'male', 'moderate_intensity', True, False, False, True, False, True, False, "복합적 위험 요소"),
        (0, 3000, 'middle', 'male', 'low_intensity', True, True, False, False, True, True, True, "최악의 상황"),
        (250, 12000, 'young', 'female', 'very_high_intensity', False, False, False, False, False, False, False, "최적의 활동")
    ]
    
    for weekly_activity_minutes, daily_steps, age_group, gender, intensity, sedentary_job, no_exercise, poor_mobility, chronic_pain, obesity, smoking, poor_diet, description in test_cases:
        analysis = calculator.get_detailed_analysis(weekly_activity_minutes, daily_steps, 
                                                  age_group, gender, intensity, sedentary_job, 
                                                  no_exercise, poor_mobility, chronic_pain, 
                                                  obesity, smoking, poor_diet)
        
        print(f"\n{description}:")
        print(f"  주간 활동: {analysis['weekly_activity_minutes']}분 ({analysis['activity_level']})")
        print(f"  일일 걸음: {analysis['daily_steps']}보 ({analysis['step_level']})")
        print(f"  활동 강도: {analysis['intensity']}")
        print(f"  위험도: {analysis['base_risk']:.3f}")
        print(f"  가중치: {analysis['adjusted_weight']:.3f}")
        print(f"  위험 수준: {analysis['risk_level']}")
        print(f"  사망률 위험: {analysis['mortality_risk']}")
        print(f"  일상활동 위험: {analysis['daily_activity_risk']}")
        print(f"  만성질환 위험: {analysis['chronic_disease_risk']}")
        print(f"  건강상 이익: {', '.join(analysis['health_benefits'][:2])}")
        print(f"  연구 근거: {analysis['research_basis']}")
        print(f"  권장사항: {', '.join(analysis['recommendations'][:2])}")

if __name__ == "__main__":
    test_physical_activity_weights()
