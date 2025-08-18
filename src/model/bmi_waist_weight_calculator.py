import numpy as np
import pandas as pd

class BMIWaistWeightCalculator:
    """논문 기반 BMI & 허리둘레 가중치 계산기 (WHtR + 사망률 연구)"""
    
    def __init__(self):
        # 논문 1: WHtR 연구 기반 가중치 (2021년)
        self.whtr_weights = {
            'normal': 0.0,      # WHtR < 0.5 (정상)
            'abdominal_obese': 0.65  # WHtR >= 0.5 (복부비만)
        }
        
        # 논문 1: 대사증후군 위험도 (OR 값 기반)
        self.metabolic_syndrome_risk = {
            'metabolic_syndrome': 2.187,  # OR 2.187 (95% CI 1.727-2.770)
            'hypertension': 1.445,        # OR 1.445 (95% CI 1.091-1.914)
            'diabetes': 2.463             # OR 2.463 (95% CI 1.707-3.555)
        }
        
        # 논문 2: 허리둘레와 사망률 연구 기반 가중치 (2019년)
        self.waist_mortality_weights = {
            'male': {
                'level_1': 0.0,   # < 80cm (기준)
                'level_2': 0.15,  # 80-85cm
                'level_3': 0.0,   # 85-90cm (기준)
                'level_4': 0.156, # 90-95cm (HR 1.156)
                'level_5': 0.412, # 95-100cm (HR 1.412)
                'level_6': 0.614  # >= 100cm (HR 1.614)
            },
            'female': {
                'level_1': 0.0,   # < 75cm (기준)
                'level_2': 0.12,  # 75-80cm
                'level_3': 0.0,   # 80-85cm (기준)
                'level_4': 0.14,  # 85-90cm
                'level_5': 0.35,  # 90-95cm
                'level_6': 0.55   # >= 95cm
            }
        }
        
        # BMI 분류 기준 (WHO 기준)
        self.bmi_categories = {
            'underweight': 18.5,
            'normal': 25.0,
            'overweight': 30.0,
            'obese': 35.0
        }
        
        # BMI별 추가 위험도
        self.bmi_additional_risk = {
            'underweight': 0.1,   # 저체중 위험
            'normal': 0.0,        # 정상 (기준)
            'overweight': 0.2,    # 과체중 위험
            'obese': 0.4,         # 비만 위험
            'severe_obese': 0.6   # 중증비만 위험
        }
        
        # 두 논문의 가중 평균 비율
        self.whtr_weight = 0.6    # WHtR 연구 60%
        self.waist_mortality_weight = 0.4  # 사망률 연구 40%
        
        # 연령별 조정 계수
        self.age_adjustment = {
            'young': 0.8,     # 18-30세
            'middle': 1.0,    # 31-50세 (기준)
            'elderly': 1.2    # 51세 이상
        }
        
        # 성별 조정 계수 (남성이 더 높은 위험)
        self.gender_adjustment = {
            'male': 1.1,
            'female': 1.0
        }
        
    def calculate_whtr(self, waist_circumference, height):
        """
        허리둘레-신장 비(WHtR) 계산
        
        Args:
            waist_circumference: 허리둘레 (cm)
            height: 신장 (cm)
        """
        if height <= 0:
            return 0.0
        return waist_circumference / height
    
    def get_waist_level(self, waist_circumference, gender):
        """
        허리둘레 수준 분류
        
        Args:
            waist_circumference: 허리둘레 (cm)
            gender: 성별 ('male' 또는 'female')
        """
        if gender == 'male':
            if waist_circumference < 80:
                return 'level_1'
            elif waist_circumference < 85:
                return 'level_2'
            elif waist_circumference < 90:
                return 'level_3'
            elif waist_circumference < 95:
                return 'level_4'
            elif waist_circumference < 100:
                return 'level_5'
            else:
                return 'level_6'
        else:  # female
            if waist_circumference < 75:
                return 'level_1'
            elif waist_circumference < 80:
                return 'level_2'
            elif waist_circumference < 85:
                return 'level_3'
            elif waist_circumference < 90:
                return 'level_4'
            elif waist_circumference < 95:
                return 'level_5'
            else:
                return 'level_6'
    
    def get_bmi_category(self, bmi):
        """
        BMI 분류
        
        Args:
            bmi: 체질량지수
        """
        if bmi < self.bmi_categories['underweight']:
            return 'underweight'
        elif bmi < self.bmi_categories['normal']:
            return 'normal'
        elif bmi < self.bmi_categories['overweight']:
            return 'overweight'
        elif bmi < self.bmi_categories['obese']:
            return 'obese'
        else:
            return 'severe_obese'
    
    def calculate_bmi_waist_risk(self, bmi, waist_circumference, height, gender, age_group='young'):
        """
        BMI & 허리둘레 위험도 계산 (두 논문 결합)
        
        Args:
            bmi: 체질량지수
            waist_circumference: 허리둘레 (cm)
            height: 신장 (cm)
            gender: 성별 ('male' 또는 'female')
            age_group: 연령대 ('young' 또는 'elderly')
        """
        # WHtR 계산
        whtr = self.calculate_whtr(waist_circumference, height)
        
        # WHtR 기반 위험도 (논문 1)
        if whtr >= 0.5:
            whtr_risk = self.whtr_weights['abdominal_obese']
        else:
            whtr_risk = self.whtr_weights['normal']
        
        # 허리둘레 수준 분류
        waist_level = self.get_waist_level(waist_circumference, gender)
        
        # 허리둘레 기반 사망률 위험도 (논문 2)
        waist_mortality_risk = self.waist_mortality_weights[gender][waist_level]
        
        # BMI 분류 및 추가 위험도
        bmi_category = self.get_bmi_category(bmi)
        bmi_additional = self.bmi_additional_risk[bmi_category]
        
        # 두 논문의 가중 평균
        combined_risk = (
            self.whtr_weight * whtr_risk + 
            self.waist_mortality_weight * waist_mortality_risk
        )
        
        # BMI 추가 위험도 적용
        final_risk = combined_risk + bmi_additional
        
        # 연령 및 성별 조정
        adjusted_risk = final_risk * self.age_adjustment[age_group] * self.gender_adjustment[gender]
        
        return min(adjusted_risk, 0.85)  # 최대 85%로 제한
    
    def get_bmi_waist_weight(self, bmi, waist_circumference, height, gender, age_group='young'):
        """
        BMI & 허리둘레 가중치 반환 (0-1 스케일)
        """
        risk = self.calculate_bmi_waist_risk(bmi, waist_circumference, height, gender, age_group)
        
        # 5개 피처 중 BMI & 허리둘레의 상대적 중요도
        bmi_waist_relative_importance = 0.25  # 25% (균형잡힌 비중)
        
        return risk * bmi_waist_relative_importance
    
    def get_detailed_analysis(self, bmi, waist_circumference, height, gender, age_group='young'):
        """
        상세한 BMI & 허리둘레 위험도 분석
        """
        risk = self.calculate_bmi_waist_risk(bmi, waist_circumference, height, gender, age_group)
        weight = self.get_bmi_waist_weight(bmi, waist_circumference, height, gender, age_group)
        whtr = self.calculate_whtr(waist_circumference, height)
        
        analysis = {
            'bmi': bmi,
            'waist_circumference': waist_circumference,
            'height': height,
            'gender': gender,
            'age_group': age_group,
            'whtr': whtr,
            'whtr_category': '복부비만' if whtr >= 0.5 else '정상',
            'bmi_category': self.get_bmi_category(bmi),
            'waist_level': self.get_waist_level(waist_circumference, gender),
            'base_risk': risk,
            'adjusted_weight': weight,
            'risk_level': self._get_risk_level(risk),
            'recommendations': self._get_recommendations(bmi, whtr, waist_circumference, gender),
            'research_basis': self._get_research_basis(),
            'metabolic_risks': self._get_metabolic_risks(whtr),
            'mortality_risks': self._get_mortality_risks(waist_circumference, gender)
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
        return "WHtR 연구(60%) + 허리둘레 사망률 연구(40%)"
    
    def _get_metabolic_risks(self, whtr):
        """대사증후군 위험도 (WHtR 연구 기반)"""
        if whtr >= 0.5:
            return {
                'metabolic_syndrome': f"{self.metabolic_syndrome_risk['metabolic_syndrome']:.1f}배 위험",
                'hypertension': f"{self.metabolic_syndrome_risk['hypertension']:.1f}배 위험",
                'diabetes': f"{self.metabolic_syndrome_risk['diabetes']:.1f}배 위험"
            }
        else:
            return {
                'metabolic_syndrome': "기준점",
                'hypertension': "기준점",
                'diabetes': "기준점"
            }
    
    def _get_mortality_risks(self, waist_circumference, gender):
        """사망률 위험도 (허리둘레 연구 기반)"""
        waist_level = self.get_waist_level(waist_circumference, gender)
        risk_value = self.waist_mortality_weights[gender][waist_level]
        
        if risk_value == 0.0:
            return "기준점"
        else:
            return f"허리둘레 증가에 따른 사망률 증가 (HR: {1 + risk_value:.3f})"
    
    def _get_recommendations(self, bmi, whtr, waist_circumference, gender):
        """BMI & 허리둘레 상태별 권장사항"""
        recommendations = []
        
        # WHtR 기반 권장사항
        if whtr >= 0.5:
            recommendations.extend([
                "복부비만 상태입니다. 허리둘레를 줄이세요.",
                "대사증후군 위험이 높습니다. 정기 검진을 받으세요.",
                "식단 조절과 운동을 통해 복부 지방을 줄이세요."
            ])
        else:
            recommendations.append("허리둘레-신장 비가 정상 범위입니다.")
        
        # BMI 기반 권장사항
        bmi_category = self.get_bmi_category(bmi)
        if bmi_category == 'underweight':
            recommendations.append("저체중 상태입니다. 적절한 영양 섭취가 필요합니다.")
        elif bmi_category == 'overweight':
            recommendations.append("과체중 상태입니다. 체중 관리가 필요합니다.")
        elif bmi_category in ['obese', 'severe_obese']:
            recommendations.extend([
                "비만 상태입니다. 체계적인 체중 감량이 필요합니다.",
                "의료진과 상담하여 체중 관리 계획을 세우세요."
            ])
        
        # 허리둘레 기반 권장사항
        if gender == 'male' and waist_circumference >= 90:
            recommendations.append("남성 기준 허리둘레가 높습니다. 복부 지방 감소가 필요합니다.")
        elif gender == 'female' and waist_circumference >= 85:
            recommendations.append("여성 기준 허리둘레가 높습니다. 복부 지방 감소가 필요합니다.")
        
        return recommendations

def test_bmi_waist_weights():
    """BMI & 허리둘레 가중치 테스트"""
    calculator = BMIWaistWeightCalculator()
    
    print("=" * 80)
    print("논문 기반 BMI & 허리둘레 가중치 테스트 (WHtR + 사망률 연구)")
    print("=" * 80)
    
    # 테스트 케이스
    test_cases = [
        (22.0, 75, 170, 'male', 'young', "정상 BMI + 정상 허리둘레 (남성)"),
        (25.5, 85, 170, 'male', 'young', "과체중 + 경계 허리둘레 (남성)"),
        (28.0, 95, 170, 'male', 'young', "과체중 + 높은 허리둘레 (남성)"),
        (32.0, 105, 170, 'male', 'young', "비만 + 매우 높은 허리둘레 (남성)"),
        (21.0, 70, 160, 'female', 'young', "정상 BMI + 정상 허리둘레 (여성)"),
        (24.5, 80, 160, 'female', 'young', "과체중 + 경계 허리둘레 (여성)"),
        (27.0, 90, 160, 'female', 'young', "과체중 + 높은 허리둘레 (여성)"),
        (30.0, 100, 160, 'female', 'young', "비만 + 매우 높은 허리둘레 (여성)"),
        (26.0, 88, 175, 'male', 'elderly', "과체중 + 높은 허리둘레 (고령 남성)")
    ]
    
    for bmi, waist, height, gender, age_group, description in test_cases:
        analysis = calculator.get_detailed_analysis(bmi, waist, height, gender, age_group)
        
        print(f"\n{description}:")
        print(f"  BMI: {analysis['bmi']:.1f} ({analysis['bmi_category']})")
        print(f"  허리둘레: {analysis['waist_circumference']}cm ({analysis['waist_level']})")
        print(f"  WHtR: {analysis['whtr']:.3f} ({analysis['whtr_category']})")
        print(f"  위험도: {analysis['base_risk']:.3f}")
        print(f"  가중치: {analysis['adjusted_weight']:.3f}")
        print(f"  위험 수준: {analysis['risk_level']}")
        print(f"  대사증후군 위험: {analysis['metabolic_risks']['metabolic_syndrome']}")
        print(f"  사망률 위험: {analysis['mortality_risks']}")
        print(f"  연구 근거: {analysis['research_basis']}")
        print(f"  권장사항: {', '.join(analysis['recommendations'][:2])}")

if __name__ == "__main__":
    test_bmi_waist_weights()
