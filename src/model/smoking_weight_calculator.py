import numpy as np
import pandas as pd

class SmokingWeightCalculator:
    """논문 기반 흡연 가중치 계산기 (8개 논문 통합)"""
    
    def __init__(self):
        # 논문 1: 유전자 발현 연구 기반 가중치
        self.gene_expression_weights = {
            'current_smoker': 0.591,  # 591개 유전자 변화 (100% 기준)
            'former_smoker': 0.145,   # 145개 유전자 변화 (과거 흡연자)
            'never_smoker': 0.0       # 비흡연자 (기준점)
        }
        
        # 논문 2: 심혈관 질환 메타분석 기반 가중치
        self.cardiovascular_weights = {
            'current_smoker': 0.75,   # 심혈관 질환 위험도 증가 (메타분석 결과)
            'former_smoker': 0.25,    # 금연 후 위험도 감소
            'never_smoker': 0.0       # 기준점
        }
        
        # 논문 3: 2024년 최신 심혈관 연구 기반 가중치
        self.cardiovascular_2024_weights = {
            'current_smoker': 0.80,   # 2024년 연구: 심혈관 질환 위험도 증가
            'former_smoker': 0.20,    # 금연 후 위험도 감소 (더 빠른 회복)
            'never_smoker': 0.0       # 기준점
        }
        
        # 논문 4: 경미한 흡연 위험도 연구 기반 가중치
        self.light_smoking_weights = {
            'light_smoker': 0.60,     # 경미한 흡연자 (1-10개비/일)
            'moderate_smoker': 0.80,  # 중간 흡연자 (11-20개비/일)
            'heavy_smoker': 0.95,     # 중증 흡연자 (21개비 이상/일)
            'former_smoker': 0.20,    # 과거 흡연자
            'never_smoker': 0.0       # 비흡연자
        }
        
        # 논문 5: 암 질환 연구 기반 가중치 (2024년)
        self.cancer_weights = {
            'current_smoker': 0.85,   # 전 세계 암 부담의 주요 원인
            'former_smoker': 0.30,    # 금연 후 암 위험 감소
            'never_smoker': 0.0       # 기준점
        }
        
        # 논문 6: 심혈관/뇌혈관 질환 메타분석 기반 가중치 (2023년)
        self.cardiovascular_meta_weights = {
            'current_smoker': 0.82,   # 체계적 메타분석 결과
            'former_smoker': 0.22,    # 금연 후 위험 감소
            'never_smoker': 0.0       # 기준점
        }
        
        # 논문 7: 전자담배 연구 기반 가중치 (2022년)
        self.e_cigarette_weights = {
            'current_smoker': 0.70,   # 전통적 흡연
            'e_cigarette_user': 0.45, # 전자담배 사용자
            'dual_user': 0.90,        # 이중 사용자
            'former_smoker': 0.20,    # 과거 흡연자
            'never_smoker': 0.0       # 비흡연자
        }
        
        # 논문 8: 금연 효과 연구 기반 가중치 (2022년)
        self.cessation_benefit_weights = {
            'current_smoker': 0.85,   # 현재 흡연자
            'former_smoker_1y': 0.60, # 금연 1년
            'former_smoker_5y': 0.35, # 금연 5년
            'former_smoker_10y': 0.20, # 금연 10년
            'never_smoker': 0.0       # 비흡연자
        }
        
        # 핵심 유전자 영향도 (9개 중 7개가 발암물질 대사 관련)
        self.critical_gene_impact = 7/9  # 0.778
        
        # 시간 경과에 따른 위험도 감소 (금연 후) - 2024년 연구 반영
        self.risk_decay_rate = 0.03  # 연간 3% 위험도 감소
        
        # 여덟 논문의 가중 평균 비율
        self.gene_weight = 0.10      # 유전자 연구 10%
        self.cardiovascular_weight = 0.15  # 심혈관 연구 15%
        self.cardiovascular_2024_weight = 0.15  # 2024년 연구 15%
        self.light_smoking_weight = 0.10  # 경미한 흡연 연구 10%
        self.cancer_weight = 0.20    # 암 질환 연구 20%
        self.cardiovascular_meta_weight = 0.15  # 심혈관 메타분석 15%
        self.e_cigarette_weight = 0.10  # 전자담배 연구 10%
        self.cessation_benefit_weight = 0.05  # 금연 효과 연구 5%
        
        # 2024년 연구의 추가 위험 요소
        self.additional_risk_factors = {
            'passive_smoking': 0.15,  # 간접흡연 위험도
            'chemical_exposure': 0.10,  # 7,357개 화학물질 노출
            'rapid_recovery': 0.05,   # 금연 후 빠른 회복 효과
            'light_smoking_risk': 0.20,  # 경미한 흡연의 추가 위험
            'cancer_burden': 0.25,    # 전 세계 암 부담
            'cerebrovascular_risk': 0.15,  # 뇌혈관 질환 위험
            'e_cigarette_lung_risk': 0.20  # 전자담배 폐 위험
        }
        
        # 흡연량별 위험도 조정 계수
        self.smoking_intensity_factors = {
            'light': 0.6,      # 1-10개비/일
            'moderate': 0.8,   # 11-20개비/일
            'heavy': 1.0       # 21개비 이상/일
        }
        
        # 흡연 유형별 조정 계수
        self.smoking_type_factors = {
            'traditional': 1.0,    # 전통적 흡연
            'e_cigarette': 0.6,    # 전자담배
            'dual': 1.3,           # 이중 사용
            'former': 0.3          # 과거 흡연
        }
        
    def calculate_smoking_risk(self, smoking_status, years_since_quit=None, passive_smoking=False, cigarettes_per_day=0, smoking_type='traditional'):
        """
        흡연 상태별 위험도 계산 (여덟 논문 결합)
        
        Args:
            smoking_status: 0 (비흡연), 1 (과거 흡연), 2 (현재 흡연)
            years_since_quit: 금연 후 경과 년수 (과거 흡연자인 경우)
            passive_smoking: 간접흡연 노출 여부
            cigarettes_per_day: 하루 흡연량 (현재 흡연자인 경우)
            smoking_type: 흡연 유형 ('traditional', 'e_cigarette', 'dual')
        """
        
        if smoking_status == 0:  # 비흡연자
            base_risk = 0.0
            if passive_smoking:
                base_risk += self.additional_risk_factors['passive_smoking']
            return base_risk
            
        elif smoking_status == 1:  # 과거 흡연자
            if years_since_quit is None:
                years_since_quit = 5  # 기본값: 5년 전 금연
            
            # 금연 후 시간 경과에 따른 위험도 감소
            risk_decay = np.exp(-self.risk_decay_rate * years_since_quit)
            
            # 빠른 회복 효과 (2024년 연구)
            rapid_recovery_bonus = self.additional_risk_factors['rapid_recovery'] * (1 - risk_decay)
            
            # 유전자 발현 기반 위험도
            gene_risk = self.gene_expression_weights['former_smoker'] * self.critical_gene_impact * risk_decay
            
            # 심혈관 질환 기반 위험도
            cardio_risk = self.cardiovascular_weights['former_smoker'] * risk_decay
            
            # 2024년 심혈관 연구 기반 위험도
            cardio_2024_risk = self.cardiovascular_2024_weights['former_smoker'] * risk_decay
            
            # 경미한 흡연 연구 기반 위험도 (과거 흡연자)
            light_smoking_risk = self.light_smoking_weights['former_smoker'] * risk_decay
            
            # 암 질환 연구 기반 위험도
            cancer_risk = self.cancer_weights['former_smoker'] * risk_decay
            
            # 심혈관 메타분석 기반 위험도
            cardio_meta_risk = self.cardiovascular_meta_weights['former_smoker'] * risk_decay
            
            # 전자담배 연구 기반 위험도
            e_cig_risk = self.e_cigarette_weights['former_smoker'] * risk_decay
            
            # 금연 효과 연구 기반 위험도
            if years_since_quit <= 1:
                cessation_risk = self.cessation_benefit_weights['former_smoker_1y']
            elif years_since_quit <= 5:
                cessation_risk = self.cessation_benefit_weights['former_smoker_5y']
            elif years_since_quit <= 10:
                cessation_risk = self.cessation_benefit_weights['former_smoker_10y']
            else:
                cessation_risk = 0.15  # 10년 이상 금연
            
            # 여덟 논문의 가중 평균
            combined_risk = (
                self.gene_weight * gene_risk + 
                self.cardiovascular_weight * cardio_risk + 
                self.cardiovascular_2024_weight * cardio_2024_risk +
                self.light_smoking_weight * light_smoking_risk +
                self.cancer_weight * cancer_risk +
                self.cardiovascular_meta_weight * cardio_meta_risk +
                self.e_cigarette_weight * e_cig_risk +
                self.cessation_benefit_weight * cessation_risk
            )
            
            # 빠른 회복 효과 적용
            final_risk = combined_risk - rapid_recovery_bonus
            
            # 간접흡연 효과 추가
            if passive_smoking:
                final_risk += self.additional_risk_factors['passive_smoking'] * 0.5
            
            return max(0.0, min(final_risk, 0.25))  # 최대 25%로 제한
            
        elif smoking_status == 2:  # 현재 흡연자
            # 흡연량에 따른 위험도 조정
            smoking_intensity = self._get_smoking_intensity(cigarettes_per_day)
            intensity_factor = self.smoking_intensity_factors[smoking_intensity]
            
            # 흡연 유형에 따른 위험도 조정
            type_factor = self.smoking_type_factors[smoking_type]
            
            # 유전자 발현 기반 위험도
            gene_risk = self.gene_expression_weights['current_smoker'] * self.critical_gene_impact * intensity_factor * type_factor
            
            # 심혈관 질환 기반 위험도
            cardio_risk = self.cardiovascular_weights['current_smoker'] * intensity_factor * type_factor
            
            # 2024년 심혈관 연구 기반 위험도
            cardio_2024_risk = self.cardiovascular_2024_weights['current_smoker'] * intensity_factor * type_factor
            
            # 경미한 흡연 연구 기반 위험도
            if smoking_intensity == 'light':
                light_smoking_risk = self.light_smoking_weights['light_smoker'] * intensity_factor * type_factor
            elif smoking_intensity == 'moderate':
                light_smoking_risk = self.light_smoking_weights['moderate_smoker'] * intensity_factor * type_factor
            else:  # heavy
                light_smoking_risk = self.light_smoking_weights['heavy_smoker'] * intensity_factor * type_factor
            
            # 암 질환 연구 기반 위험도
            cancer_risk = self.cancer_weights['current_smoker'] * intensity_factor * type_factor
            
            # 심혈관 메타분석 기반 위험도
            cardio_meta_risk = self.cardiovascular_meta_weights['current_smoker'] * intensity_factor * type_factor
            
            # 전자담배 연구 기반 위험도
            if smoking_type == 'e_cigarette':
                e_cig_risk = self.e_cigarette_weights['e_cigarette_user'] * intensity_factor
            elif smoking_type == 'dual':
                e_cig_risk = self.e_cigarette_weights['dual_user'] * intensity_factor
            else:
                e_cig_risk = self.e_cigarette_weights['current_smoker'] * intensity_factor
            
            # 금연 효과 연구 기반 위험도 (현재 흡연자)
            cessation_risk = self.cessation_benefit_weights['current_smoker'] * intensity_factor * type_factor
            
            # 여덟 논문의 가중 평균
            combined_risk = (
                self.gene_weight * gene_risk + 
                self.cardiovascular_weight * cardio_risk + 
                self.cardiovascular_2024_weight * cardio_2024_risk +
                self.light_smoking_weight * light_smoking_risk +
                self.cancer_weight * cancer_risk +
                self.cardiovascular_meta_weight * cardio_meta_risk +
                self.e_cigarette_weight * e_cig_risk +
                self.cessation_benefit_weight * cessation_risk
            )
            
            # 화학물질 노출 위험 추가
            chemical_risk = self.additional_risk_factors['chemical_exposure'] * intensity_factor * type_factor
            
            # 경미한 흡연의 추가 위험 (용량-반응 관계)
            if smoking_intensity == 'light':
                additional_light_risk = self.additional_risk_factors['light_smoking_risk'] * 0.5
            else:
                additional_light_risk = 0.0
            
            # 암 부담 위험 추가
            cancer_burden_risk = self.additional_risk_factors['cancer_burden'] * intensity_factor * type_factor
            
            # 뇌혈관 질환 위험 추가
            cerebrovascular_risk = self.additional_risk_factors['cerebrovascular_risk'] * intensity_factor * type_factor
            
            # 전자담배 폐 위험 추가
            if smoking_type in ['e_cigarette', 'dual']:
                e_cig_lung_risk = self.additional_risk_factors['e_cigarette_lung_risk'] * intensity_factor
            else:
                e_cig_lung_risk = 0.0
            
            # 간접흡연 효과 추가
            if passive_smoking:
                combined_risk += self.additional_risk_factors['passive_smoking'] * intensity_factor * type_factor
            
            final_risk = (combined_risk + chemical_risk + additional_light_risk + 
                         cancer_burden_risk + cerebrovascular_risk + e_cig_lung_risk)
            
            return min(final_risk, 0.98)  # 최대 98%로 제한
        
        return 0.0
    
    def _get_smoking_intensity(self, cigarettes_per_day):
        """흡연량에 따른 강도 분류"""
        if cigarettes_per_day <= 0:
            return 'light'
        elif cigarettes_per_day <= 10:
            return 'light'
        elif cigarettes_per_day <= 20:
            return 'moderate'
        else:
            return 'heavy'
    
    def get_smoking_weight(self, smoking_status, years_since_quit=None, passive_smoking=False, cigarettes_per_day=0, smoking_type='traditional'):
        """
        흡연 가중치 반환 (0-1 스케일)
        """
        risk = self.calculate_smoking_risk(smoking_status, years_since_quit, passive_smoking, cigarettes_per_day, smoking_type)
        
        # 5개 피처 중 흡연의 상대적 중요도 (8개 논문 기반으로 균형잡힌 비중)
        smoking_relative_importance = 0.30  # 30% (균형잡힌 비중)
        
        return risk * smoking_relative_importance
    
    def get_detailed_analysis(self, smoking_status, years_since_quit=None, passive_smoking=False, cigarettes_per_day=0, smoking_type='traditional'):
        """
        상세한 흡연 위험도 분석 (8개 논문 반영)
        """
        risk = self.calculate_smoking_risk(smoking_status, years_since_quit, passive_smoking, cigarettes_per_day, smoking_type)
        weight = self.get_smoking_weight(smoking_status, years_since_quit, passive_smoking, cigarettes_per_day, smoking_type)
        
        analysis = {
            'smoking_status': smoking_status,
            'years_since_quit': years_since_quit,
            'passive_smoking': passive_smoking,
            'cigarettes_per_day': cigarettes_per_day,
            'smoking_type': smoking_type,
            'smoking_intensity': self._get_smoking_intensity(cigarettes_per_day),
            'base_risk': risk,
            'adjusted_weight': weight,
            'risk_level': self._get_risk_level(risk),
            'recommendations': self._get_recommendations(smoking_status, years_since_quit, passive_smoking, cigarettes_per_day, smoking_type),
            'research_basis': self._get_research_basis(smoking_status),
            'recovery_timeline': self._get_recovery_timeline(years_since_quit),
            'mortality_risk': self._get_mortality_risk(cigarettes_per_day),
            'cancer_risk': self._get_cancer_risk(smoking_type),
            'cardiovascular_risk': self._get_cardiovascular_risk(smoking_type)
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
    
    def _get_research_basis(self, smoking_status):
        """연구 근거 설명 (8개 논문)"""
        if smoking_status == 0:
            return "비흡연자: 기준점 (여덟 논문 모두 0% 위험도)"
        elif smoking_status == 1:
            return "과거 흡연자: 유전자(10%) + 심혈관(15%) + 2024년(15%) + 경미한 흡연(10%) + 암(20%) + 메타분석(15%) + 전자담배(10%) + 금연효과(5%)"
        elif smoking_status == 2:
            return "현재 흡연자: 유전자(10%) + 심혈관(15%) + 2024년(15%) + 경미한 흡연(10%) + 암(20%) + 메타분석(15%) + 전자담배(10%) + 금연효과(5%)"
        return ""
    
    def _get_cancer_risk(self, smoking_type):
        """암 위험도 (2024년 연구 기반)"""
        if smoking_type == 'traditional':
            return "전 세계 암 부담의 주요 원인 (2024년 연구)"
        elif smoking_type == 'e_cigarette':
            return "전자담배 관련 암 위험 (폐 건강 영향)"
        elif smoking_type == 'dual':
            return "이중 사용으로 인한 암 위험 증가"
        else:
            return "기준점"
    
    def _get_cardiovascular_risk(self, smoking_type):
        """심혈관 위험도 (2023년 메타분석 기반)"""
        if smoking_type == 'traditional':
            return "심혈관/뇌혈관 질환 위험 증가 (2023년 메타분석)"
        elif smoking_type == 'e_cigarette':
            return "전자담배 심혈관 위험 (체계적 메타분석)"
        elif smoking_type == 'dual':
            return "이중 사용으로 인한 심혈관 위험 증가"
        else:
            return "기준점"
    
    def _get_mortality_risk(self, cigarettes_per_day):
        """사망률 위험도 (경미한 흡연 연구 기반)"""
        if cigarettes_per_day == 0:
            return "기준점 (비흡연자)"
        elif cigarettes_per_day <= 10:
            return "경미한 흡연: 모든 원인 사망률 증가 (용량-반응 관계)"
        elif cigarettes_per_day <= 20:
            return "중간 흡연: 심혈관 질환 및 폐암 위험 증가"
        else:
            return "중증 흡연: 모든 원인 사망률 크게 증가"
    
    def _get_recovery_timeline(self, years_since_quit):
        """금연 후 회복 타임라인"""
        if years_since_quit is None:
            return "현재 흡연자"
        
        timeline = {
            'immediate': "금연 후 20분: 혈압과 맥박 정상화",
            'short_term': "금연 후 8시간: 혈중 일산화탄소 수치 정상화",
            'medium_term': "금연 후 1년: 관상동맥질환 위험 50% 감소",
            'long_term': "금연 후 15년: 심혈관질환 위험 비흡연자 수준"
        }
        
        if years_since_quit < 1:
            return f"{timeline['immediate']}, {timeline['short_term']}"
        elif years_since_quit < 5:
            return f"{timeline['medium_term']}"
        else:
            return f"{timeline['long_term']}"
    
    def _get_recommendations(self, smoking_status, years_since_quit, passive_smoking, cigarettes_per_day, smoking_type):
        """흡연 상태별 권장사항 (8개 논문 반영)"""
        if smoking_status == 0:  # 비흡연자
            recommendations = [
                "금연 상태를 유지하세요.",
                "정기적인 심혈관 건강 검진을 받으세요.",
                "심장 건강을 위한 운동을 하세요."
            ]
            if passive_smoking:
                recommendations.extend([
                    "간접흡연을 피하세요.",
                    "금연 구역을 이용하세요."
                ])
            return recommendations
            
        elif smoking_status == 1:  # 과거 흡연자
            recommendations = [
                "금연을 계속 유지하세요.",
                "정기적인 심혈관 검사를 받으세요.",
                "심장 건강을 위한 생활습관을 유지하세요.",
                "혈압과 콜레스테롤을 정기적으로 체크하세요."
            ]
            
            if years_since_quit and years_since_quit < 5:
                recommendations.append("금연 후 5년까지는 심혈관 질환 위험이 높습니다.")
            
            if passive_smoking:
                recommendations.append("간접흡연 노출을 최소화하세요.")
            
            return recommendations
            
        elif smoking_status == 2:  # 현재 흡연자
            intensity = self._get_smoking_intensity(cigarettes_per_day)
            
            recommendations = [
                "즉시 금연을 시작하세요 (모든 흡연량에서 위험 증가).",
                "금연 프로그램에 참여하세요.",
                "의료진과 상담하여 금연 계획을 세우세요.",
                "정기적인 심혈관 검사를 받으세요."
            ]
            
            if smoking_type == 'e_cigarette':
                recommendations.extend([
                    "전자담배도 폐 건강에 해롭습니다.",
                    "전자담배 사용을 중단하고 완전 금연을 고려하세요."
                ])
            elif smoking_type == 'dual':
                recommendations.extend([
                    "이중 사용은 더 큰 위험을 초래합니다.",
                    "모든 담배 제품 사용을 중단하세요."
                ])
            
            if intensity == 'light':
                recommendations.extend([
                    "경미한 흡연도 사망률을 증가시킵니다.",
                    "하루 흡연량을 점진적으로 줄여가세요."
                ])
            elif intensity == 'moderate':
                recommendations.extend([
                    "중간 흡연량은 심혈관 질환 위험을 크게 증가시킵니다.",
                    "금연을 위한 약물 치료를 고려하세요."
                ])
            else:  # heavy
                recommendations.extend([
                    "중증 흡연은 모든 원인 사망률을 크게 증가시킵니다.",
                    "즉시 의료진의 도움을 받아 금연하세요."
                ])
            
            if passive_smoking:
                recommendations.append("가족과 함께 금연하여 간접흡연을 방지하세요.")
            
            return recommendations
        
        return []

def test_smoking_weights():
    """흡연 가중치 테스트 (8개 논문 반영)"""
    calculator = SmokingWeightCalculator()
    
    print("=" * 80)
    print("논문 기반 흡연 가중치 테스트 (8개 논문 통합)")
    print("=" * 80)
    
    # 테스트 케이스
    test_cases = [
        (0, None, False, 0, 'traditional', "비흡연자"),
        (0, None, True, 0, 'traditional', "비흡연자 (간접흡연 노출)"),
        (1, 1, False, 0, 'traditional', "1년 전 금연"),
        (1, 5, False, 0, 'traditional', "5년 전 금연"),
        (1, 10, False, 0, 'traditional', "10년 전 금연"),
        (1, 20, False, 0, 'traditional', "20년 전 금연"),
        (2, None, False, 5, 'traditional', "경미한 흡연자 (5개비/일)"),
        (2, None, False, 15, 'traditional', "중간 흡연자 (15개비/일)"),
        (2, None, False, 30, 'traditional', "중증 흡연자 (30개비/일)"),
        (2, None, False, 10, 'e_cigarette', "전자담배 사용자"),
        (2, None, False, 20, 'dual', "이중 사용자"),
        (2, None, True, 5, 'traditional', "경미한 흡연자 + 간접흡연")
    ]
    
    for smoking_status, years_since_quit, passive_smoking, cigarettes_per_day, smoking_type, description in test_cases:
        analysis = calculator.get_detailed_analysis(smoking_status, years_since_quit, passive_smoking, cigarettes_per_day, smoking_type)
        
        print(f"\n{description}:")
        print(f"  흡연 유형: {analysis['smoking_type']}")
        print(f"  흡연 강도: {analysis['smoking_intensity']}")
        print(f"  위험도: {analysis['base_risk']:.3f}")
        print(f"  가중치: {analysis['adjusted_weight']:.3f}")
        print(f"  위험 수준: {analysis['risk_level']}")
        print(f"  암 위험: {analysis['cancer_risk']}")
        print(f"  심혈관 위험: {analysis['cardiovascular_risk']}")
        print(f"  사망률 위험: {analysis['mortality_risk']}")
        print(f"  연구 근거: {analysis['research_basis']}")
        if analysis['recovery_timeline']:
            print(f"  회복 타임라인: {analysis['recovery_timeline']}")
        print(f"  권장사항: {', '.join(analysis['recommendations'][:2])}")

if __name__ == "__main__":
    test_smoking_weights()
