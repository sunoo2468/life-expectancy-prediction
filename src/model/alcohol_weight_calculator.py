import numpy as np
import pandas as pd

class AlcoholWeightCalculator:
    """논문 기반 알코올 가중치 계산기 (암 발병률 + J-곡선 반박 연구)"""
    
    def __init__(self):
        # 논문 1: 알코올과 암 발병률 연구 기반 가중치 (2023년)
        self.cancer_risk_weights = {
            'non_drinker': 0.0,      # 비음주자 (기준점)
            'light_drinker': 0.25,   # 경미한 음주자 (주 1-2회)
            'moderate_drinker': 0.62, # 중간 음주자 (주 3-4회, 소주 2병)
            'heavy_drinker': 0.85,   # 중증 음주자 (주 5회 이상)
            'binge_drinker': 0.95    # 폭음자
        }
        
        # 논문 1: 암별 위험도 (OR 값 기반)
        self.cancer_specific_risk = {
            'all_cancer': 1.62,      # 모든 암 위험 (OR 1.62)
            'oral_pharynx': 2.5,     # 구강, 인두암
            'esophagus': 3.2,        # 식도암
            'larynx': 2.8,           # 후두암
            'liver': 2.1,            # 간암
            'colorectal': 1.4,       # 대장암
            'breast': 1.3            # 유방암 (여성)
        }
        
        # 논문 2: J-곡선 반박 연구 기반 가중치 (2022년)
        self.cardiovascular_risk_weights = {
            'non_drinker': 0.0,      # 비음주자 (기준점)
            'light_drinker': 0.15,   # 경미한 음주자 (J-곡선 반박)
            'moderate_drinker': 0.35, # 중간 음주자
            'heavy_drinker': 0.65,   # 중증 음주자
            'binge_drinker': 0.80    # 폭음자
        }
        
        # 알코올 섭취량 기준 (g/주)
        self.alcohol_intake_standards = {
            'non_drinker': 0,        # 0g/주
            'light_drinker': 30,     # 1-30g/주
            'moderate_drinker': 90,  # 31-90g/주 (소주 2병)
            'heavy_drinker': 150,    # 91-150g/주
            'binge_drinker': 200     # 150g 이상/주
        }
        
        # 두 논문의 가중 평균 비율
        self.cancer_weight = 0.6     # 암 발병률 연구 60%
        self.cardiovascular_weight = 0.4  # 심혈관 연구 40%
        
        # 성별 조정 계수
        self.gender_adjustment = {
            'male': 1.0,     # 남성 (기준)
            'female': 1.2    # 여성 (더 높은 위험)
        }
        
        # 연령별 조정 계수
        self.age_adjustment = {
            'young': 0.8,    # 20-40세
            'middle': 1.0,   # 40-60세 (기준)
            'elderly': 1.3   # 60세 이상
        }
        
        # 추가 위험 요소
        self.additional_risk_factors = {
            'genetic_factor': 0.20,  # 유전적 요인
            'lifestyle_confounding': 0.15,  # 생활습관 교란요인
            'binge_drinking_risk': 0.30,  # 폭음 위험
            'chronic_drinking': 0.25  # 만성 음주 위험
        }
        
    def calculate_alcohol_intake(self, drinks_per_week, drink_type='soju'):
        """
        주간 알코올 섭취량 계산 (g/주)
        
        Args:
            drinks_per_week: 주간 음주 횟수
            drink_type: 음주 종류 ('soju', 'beer', 'wine', 'whiskey')
        """
        # 음주 종류별 알코올 함량 (g/잔)
        alcohol_content = {
            'soju': 13.5,    # 소주 1잔 (45ml, 20도)
            'beer': 12.8,    # 맥주 1잔 (355ml, 4.5도)
            'wine': 14.0,    # 와인 1잔 (150ml, 12도)
            'whiskey': 14.0  # 위스키 1잔 (30ml, 40도)
        }
        
        return drinks_per_week * alcohol_content.get(drink_type, 13.5)
    
    def get_drinking_category(self, alcohol_intake):
        """
        알코올 섭취량에 따른 음주 분류
        
        Args:
            alcohol_intake: 주간 알코올 섭취량 (g/주)
        """
        if alcohol_intake == 0:
            return 'non_drinker'
        elif alcohol_intake <= 30:
            return 'light_drinker'
        elif alcohol_intake <= 90:
            return 'moderate_drinker'
        elif alcohol_intake <= 150:
            return 'heavy_drinker'
        else:
            return 'binge_drinker'
    
    def calculate_alcohol_risk(self, drinks_per_week, drink_type='soju', gender='male', age_group='middle', binge_drinking=False, chronic_drinking=False):
        """
        알코올 위험도 계산 (두 논문 결합)
        
        Args:
            drinks_per_week: 주간 음주 횟수
            drink_type: 음주 종류
            gender: 성별 ('male' 또는 'female')
            age_group: 연령대 ('young', 'middle', 'elderly')
            binge_drinking: 폭음 여부
            chronic_drinking: 만성 음주 여부
        """
        # 알코올 섭취량 계산
        alcohol_intake = self.calculate_alcohol_intake(drinks_per_week, drink_type)
        
        # 음주 분류
        drinking_category = self.get_drinking_category(alcohol_intake)
        
        # 암 발병률 기반 위험도 (논문 1)
        cancer_risk = self.cancer_risk_weights[drinking_category]
        
        # 심혈관 질환 기반 위험도 (논문 2)
        cardiovascular_risk = self.cardiovascular_risk_weights[drinking_category]
        
        # 두 논문의 가중 평균
        combined_risk = (
            self.cancer_weight * cancer_risk + 
            self.cardiovascular_weight * cardiovascular_risk
        )
        
        # 추가 위험 요소 적용
        additional_risk = 0.0
        
        if binge_drinking:
            additional_risk += self.additional_risk_factors['binge_drinking_risk']
        
        if chronic_drinking:
            additional_risk += self.additional_risk_factors['chronic_drinking']
        
        # 유전적 요인 (J-곡선 반박 연구 반영)
        if drinking_category != 'non_drinker':
            additional_risk += self.additional_risk_factors['genetic_factor']
        
        # 생활습관 교란요인 (J-곡선 반박 연구 반영)
        if drinking_category == 'light_drinker':
            additional_risk += self.additional_risk_factors['lifestyle_confounding']
        
        # 성별 및 연령별 조정
        adjusted_risk = (combined_risk + additional_risk) * self.gender_adjustment[gender] * self.age_adjustment[age_group]
        
        return min(adjusted_risk, 0.90)  # 최대 90%로 제한
    
    def get_alcohol_weight(self, drinks_per_week, drink_type='soju', gender='male', age_group='middle', binge_drinking=False, chronic_drinking=False):
        """
        알코올 가중치 반환 (0-1 스케일)
        """
        risk = self.calculate_alcohol_risk(drinks_per_week, drink_type, gender, age_group, binge_drinking, chronic_drinking)
        
        # 5개 피처 중 알코올의 상대적 중요도
        alcohol_relative_importance = 0.20  # 20% (균형잡힌 비중)
        
        return risk * alcohol_relative_importance
    
    def get_detailed_analysis(self, drinks_per_week, drink_type='soju', gender='male', age_group='middle', binge_drinking=False, chronic_drinking=False):
        """
        상세한 알코올 위험도 분석
        """
        risk = self.calculate_alcohol_risk(drinks_per_week, drink_type, gender, age_group, binge_drinking, chronic_drinking)
        weight = self.get_alcohol_weight(drinks_per_week, drink_type, gender, age_group, binge_drinking, chronic_drinking)
        alcohol_intake = self.calculate_alcohol_intake(drinks_per_week, drink_type)
        
        analysis = {
            'drinks_per_week': drinks_per_week,
            'drink_type': drink_type,
            'gender': gender,
            'age_group': age_group,
            'alcohol_intake': alcohol_intake,
            'drinking_category': self.get_drinking_category(alcohol_intake),
            'binge_drinking': binge_drinking,
            'chronic_drinking': chronic_drinking,
            'base_risk': risk,
            'adjusted_weight': weight,
            'risk_level': self._get_risk_level(risk),
            'recommendations': self._get_recommendations(drinks_per_week, drink_type, gender, binge_drinking, chronic_drinking),
            'research_basis': self._get_research_basis(),
            'cancer_risks': self._get_cancer_risks(alcohol_intake),
            'cardiovascular_risks': self._get_cardiovascular_risks(alcohol_intake),
            'j_curve_myth': self._get_j_curve_myth(alcohol_intake)
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
        return "암 발병률 연구(60%) + J-곡선 반박 연구(40%)"
    
    def _get_cancer_risks(self, alcohol_intake):
        """암 위험도 (2023년 연구 기반)"""
        if alcohol_intake == 0:
            return "기준점 (비음주자)"
        elif alcohol_intake <= 30:
            return "경미한 암 위험 증가"
        elif alcohol_intake <= 90:
            return f"모든 암 위험 {self.cancer_specific_risk['all_cancer']:.1f}배 증가 (소주 2병 기준)"
        else:
            return "구강, 인두, 식도, 후두, 간암 위험 현저히 증가"
    
    def _get_cardiovascular_risks(self, alcohol_intake):
        """심혈관 위험도 (2022년 J-곡선 반박 연구 기반)"""
        if alcohol_intake == 0:
            return "기준점 (비음주자)"
        elif alcohol_intake <= 30:
            return "소량 음주도 심혈관 질환 위험 증가 (J-곡선 반박)"
        else:
            return "음주량 증가에 따른 심혈관 질환 위험 증가"
    
    def _get_j_curve_myth(self, alcohol_intake):
        """J-곡선 오해 해명"""
        if alcohol_intake <= 30:
            return "소량 음주의 보호 효과는 건강한 생활습관 때문 (J-곡선 반박)"
        else:
            return "모든 음주량에서 심혈관 질환 위험 증가"
    
    def _get_recommendations(self, drinks_per_week, drink_type, gender, binge_drinking, chronic_drinking):
        """알코올 상태별 권장사항"""
        recommendations = []
        
        if drinks_per_week == 0:
            recommendations.extend([
                "금주 상태를 유지하세요.",
                "정기적인 건강 검진을 받으세요."
            ])
        elif drinks_per_week <= 2:
            recommendations.extend([
                "소량 음주도 암 및 심혈관 질환 위험을 증가시킵니다.",
                "음주량을 점진적으로 줄여가세요.",
                "주 1회 이하로 제한하세요."
            ])
        elif drinks_per_week <= 4:
            recommendations.extend([
                "중간 음주는 모든 암 위험이 1.62배 증가합니다.",
                "특히 구강, 인두, 식도, 후두, 간암 위험이 높습니다.",
                "음주량을 줄이거나 금주를 고려하세요."
            ])
        else:
            recommendations.extend([
                "중증 음주는 모든 질환 위험을 크게 증가시킵니다.",
                "즉시 의료진과 상담하여 금주 계획을 세우세요.",
                "알코올 의존성 치료를 고려하세요."
            ])
        
        if binge_drinking:
            recommendations.extend([
                "폭음은 즉시 중단하세요.",
                "폭음은 모든 질환 위험을 크게 증가시킵니다."
            ])
        
        if chronic_drinking:
            recommendations.extend([
                "만성 음주는 장기적인 건강 위험을 초래합니다.",
                "체계적인 금주 프로그램에 참여하세요."
            ])
        
        if gender == 'female':
            recommendations.append("여성은 남성보다 더 적은 양의 알코올에도 민감합니다.")
        
        return recommendations

def test_alcohol_weights():
    """알코올 가중치 테스트"""
    calculator = AlcoholWeightCalculator()
    
    print("=" * 80)
    print("논문 기반 알코올 가중치 테스트 (암 발병률 + J-곡선 반박 연구)")
    print("=" * 80)
    
    # 테스트 케이스
    test_cases = [
        (0, 'soju', 'male', 'middle', False, False, "비음주자"),
        (1, 'soju', 'male', 'middle', False, False, "경미한 음주자 (주 1회)"),
        (2, 'soju', 'male', 'middle', False, False, "중간 음주자 (주 2회, 소주 2병)"),
        (4, 'soju', 'male', 'middle', False, False, "중간 음주자 (주 4회)"),
        (6, 'soju', 'male', 'middle', False, False, "중증 음주자 (주 6회)"),
        (8, 'soju', 'male', 'middle', True, False, "폭음자"),
        (3, 'soju', 'male', 'middle', False, True, "만성 음주자"),
        (2, 'soju', 'female', 'middle', False, False, "여성 중간 음주자"),
        (2, 'soju', 'male', 'elderly', False, False, "고령자 중간 음주자"),
        (1, 'beer', 'male', 'middle', False, False, "맥주 경미한 음주자"),
        (2, 'wine', 'female', 'middle', False, False, "여성 와인 중간 음주자")
    ]
    
    for drinks_per_week, drink_type, gender, age_group, binge_drinking, chronic_drinking, description in test_cases:
        analysis = calculator.get_detailed_analysis(drinks_per_week, drink_type, gender, age_group, binge_drinking, chronic_drinking)
        
        print(f"\n{description}:")
        print(f"  주간 음주: {analysis['drinks_per_week']}회 ({analysis['drink_type']})")
        print(f"  알코올 섭취: {analysis['alcohol_intake']:.1f}g/주")
        print(f"  음주 분류: {analysis['drinking_category']}")
        print(f"  위험도: {analysis['base_risk']:.3f}")
        print(f"  가중치: {analysis['adjusted_weight']:.3f}")
        print(f"  위험 수준: {analysis['risk_level']}")
        print(f"  암 위험: {analysis['cancer_risks']}")
        print(f"  심혈관 위험: {analysis['cardiovascular_risks']}")
        print(f"  J-곡선 오해: {analysis['j_curve_myth']}")
        print(f"  연구 근거: {analysis['research_basis']}")
        print(f"  권장사항: {', '.join(analysis['recommendations'][:2])}")

if __name__ == "__main__":
    test_alcohol_weights()
