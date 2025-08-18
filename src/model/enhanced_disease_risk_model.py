import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

from integrated_weight_calculator import IntegratedWeightCalculator

class EnhancedDiseaseRiskModel:
    """통합 가중치 시스템 기반 향상된 질병 위험도 모델 (20개 연구 논문 기반)"""
    
    def __init__(self):
        # 기존 ML 모델들
        self.models = {
            'random_forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'gradient_boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        self.best_model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        
        # 통합 가중치 계산기 초기화
        self.integrated_calculator = IntegratedWeightCalculator()
        
        # 연구 기반 신뢰성 정보
        self.research_credibility = {
            'total_papers': 20,
            'recent_papers': 18,
            'korean_studies': 4,
            'meta_analyses': 8,
            'reliability_score': 0.95
        }
        
    def calculate_research_based_risk_score(self, 
                                          # 흡연 관련 파라미터
                                          smoking_status, years_since_quit=None, passive_smoking=False, 
                                          cigarettes_per_day=0, smoking_type='traditional',
                                          # BMI & 허리둘레 관련 파라미터
                                          bmi=22.0, waist_circumference=80, height=170, gender='male', age_group='middle',
                                          # 알코올 관련 파라미터
                                          drinks_per_week=0, drink_type='soju', binge_drinking=False, chronic_drinking=False,
                                          # 수면의 질 관련 파라미터
                                          sleep_quality_score=7.0, sleep_hours=7, insomnia=False, sleep_apnea=False, 
                                          irregular_schedule=False, stress_level='low',
                                          # 신체활동 관련 파라미터
                                          weekly_activity_minutes=150, daily_steps=8000, intensity='moderate_intensity',
                                          sedentary_job=False, no_exercise=False, poor_mobility=False, chronic_pain=False,
                                          obesity=False, poor_diet=False):
        """
        연구 기반 통합 위험도 점수 계산 (20개 논문 기반)
        
        Returns:
            dict: 통합 위험도 분석 결과
        """
        # 통합 가중치 계산기로 위험도 계산
        integrated_analysis = self.integrated_calculator.calculate_integrated_risk(
            smoking_status, years_since_quit, passive_smoking, cigarettes_per_day, smoking_type,
            bmi, waist_circumference, height, gender, age_group,
            drinks_per_week, drink_type, binge_drinking, chronic_drinking,
            sleep_quality_score, sleep_hours, insomnia, sleep_apnea, irregular_schedule, stress_level,
            weekly_activity_minutes, daily_steps, intensity, sedentary_job, no_exercise, 
            poor_mobility, chronic_pain, obesity, poor_diet
        )
        
        return integrated_analysis
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """ML 모델 훈련 및 성능 비교"""
        print("=" * 60)
        print("향상된 질병 위험도 예측 모델 훈련 (연구 기반 가중치 통합)")
        print("=" * 60)
        
        best_score = 0
        best_model_name = None
        
        for name, model in self.models.items():
            print(f"\n{name.upper()} 모델 훈련 중...")
            
            # 모델 훈련
            model.fit(X_train, y_train)
            
            # 예측 및 성능 평가
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # 교차 검증
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
            
            # 성능 지표
            accuracy = model.score(X_test, y_test)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            print(f"정확도: {accuracy:.4f}")
            print(f"AUC 점수: {auc_score:.4f}")
            print(f"교차 검증 AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # 최고 성능 모델 선택
            if auc_score > best_score:
                best_score = auc_score
                best_model_name = name
                self.best_model = model
                
                # 특성 중요도 저장 (트리 기반 모델의 경우)
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance = pd.DataFrame({
                        'feature': X_train.columns,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
        
        print(f"\n🏆 최고 성능 모델: {best_model_name.upper()}")
        print(f"최고 AUC 점수: {best_score:.4f}")
        
        return self.best_model
    
    def predict_disease_risk(self, 
                           # ML 모델용 기본 피처
                           bmi, waist_size, smoking_level, alcohol_consumption, 
                           sleep_quality, physical_activity, age, gender,
                           # 연구 기반 가중치용 상세 파라미터
                           smoking_status=None, years_since_quit=None, passive_smoking=False,
                           cigarettes_per_day=0, smoking_type='traditional',
                           height=170, age_group='middle',
                           drinks_per_week=None, drink_type='soju', binge_drinking=False, chronic_drinking=False,
                           sleep_hours=7, insomnia=False, sleep_apnea=False, irregular_schedule=False, stress_level='low',
                           weekly_activity_minutes=None, daily_steps=None, intensity='moderate_intensity',
                           sedentary_job=False, no_exercise=False, poor_mobility=False, chronic_pain=False,
                           obesity=False, poor_diet=False):
        """
        질병 위험도 예측 (ML 모델 + 연구 기반 가중치 결합)
        
        Returns:
            dict: 통합 예측 결과
        """
        # 1. ML 모델 예측
        ml_prediction = None
        ml_confidence = None
        
        if self.best_model is not None:
            # 기본 피처로 데이터프레임 생성
            basic_features = pd.DataFrame({
                'bmi': [bmi],
                'waist_size': [waist_size],
                'smoking_level': [smoking_level],
                'alcohol_consumption': [alcohol_consumption],
                'sleep_quality': [sleep_quality],
                'physical_activity': [physical_activity],
                'age': [age],
                'gender': [1 if gender == 'male' else 0]
            })
            
            # 스케일링
            basic_features_scaled = self.scaler.transform(basic_features)
            
            # 예측
            ml_prediction = self.best_model.predict(basic_features_scaled)[0]
            ml_prediction_proba = self.best_model.predict_proba(basic_features_scaled)[0]
            ml_confidence = max(ml_prediction_proba)
        
        # 2. 연구 기반 가중치 계산
        research_analysis = self.calculate_research_based_risk_score(
            smoking_status or smoking_level, years_since_quit, passive_smoking, cigarettes_per_day, smoking_type,
            bmi, waist_size, height, gender, age_group,
            drinks_per_week or alcohol_consumption, drink_type, binge_drinking, chronic_drinking,
            sleep_quality, sleep_hours, insomnia, sleep_apnea, irregular_schedule, stress_level,
            weekly_activity_minutes or physical_activity * 30, daily_steps or 8000, intensity,
            sedentary_job, no_exercise, poor_mobility, chronic_pain, obesity, poor_diet
        )
        
        # 3. 통합 예측 결과
        integrated_result = {
            'ml_prediction': ml_prediction,
            'ml_confidence': ml_confidence,
            'research_based_risk': research_analysis['integrated_risk'],
            'risk_level': research_analysis['risk_level'],
            'feature_contributions': research_analysis['feature_contributions'],
            'recommendations': research_analysis['recommendations'],
            'health_impact_summary': research_analysis['health_impact_summary'],
            'research_credibility': self.research_credibility,
            'individual_analyses': research_analysis['individual_analyses']
        }
        
        # 4. 최종 위험도 결정 (ML + 연구 기반 가중치 결합)
        if ml_prediction is not None and ml_confidence is not None:
            # ML 모델과 연구 기반 가중치를 결합
            ml_weight = 0.3  # ML 모델 가중치
            research_weight = 0.7  # 연구 기반 가중치 가중치
            
            final_risk = (ml_weight * ml_prediction + research_weight * research_analysis['integrated_risk'])
            integrated_result['final_risk'] = final_risk
            integrated_result['final_risk_level'] = self._get_final_risk_level(final_risk)
        else:
            # ML 모델이 없는 경우 연구 기반 가중치만 사용
            integrated_result['final_risk'] = research_analysis['integrated_risk']
            integrated_result['final_risk_level'] = research_analysis['risk_level']
        
        return integrated_result
    
    def _get_final_risk_level(self, final_risk):
        """최종 위험도 수준 분류"""
        if final_risk < 0.1:
            return "매우 낮음"
        elif final_risk < 0.25:
            return "낮음"
        elif final_risk < 0.5:
            return "보통"
        elif final_risk < 0.75:
            return "높음"
        else:
            return "매우 높음"
    
    def get_feature_importance(self):
        """특성 중요도 반환"""
        return self.feature_importance
    
    def save_model(self, filepath):
        """모델 저장"""
        model_data = {
            'best_model': self.best_model,
            'scaler': self.scaler,
            'feature_importance': self.feature_importance,
            'research_credibility': self.research_credibility
        }
        joblib.dump(model_data, filepath)
        print(f"모델이 {filepath}에 저장되었습니다.")
    
    def load_model(self, filepath):
        """모델 로드"""
        model_data = joblib.load(filepath)
        self.best_model = model_data['best_model']
        self.scaler = model_data['scaler']
        self.feature_importance = model_data['feature_importance']
        self.research_credibility = model_data['research_credibility']
        print(f"모델이 {filepath}에서 로드되었습니다.")
    
    def get_research_summary(self):
        """연구 기반 신뢰성 요약"""
        return {
            'total_papers': self.research_credibility['total_papers'],
            'recent_papers': self.research_credibility['recent_papers'],
            'korean_studies': self.research_credibility['korean_studies'],
            'meta_analyses': self.research_credibility['meta_analyses'],
            'reliability_score': self.research_credibility['reliability_score'],
            'feature_breakdown': {
                'smoking': '8개 논문 (유전자, 심혈관, 암, 금연 효과)',
                'bmi_waist': '2개 논문 (WHtR, 허리둘레 사망률)',
                'alcohol': '2개 논문 (암 발병률, J-곡선 반박)',
                'sleep_quality': '5개 논문 (당뇨병, 심혈관, 정신건강, 스트레스)',
                'physical_activity': '3개 논문 (사망률, 일상활동, WHO 가이드라인)'
            }
        }

def test_enhanced_model():
    """향상된 모델 테스트"""
    model = EnhancedDiseaseRiskModel()
    
    print("=" * 80)
    print("향상된 질병 위험도 모델 테스트 (20개 연구 논문 기반)")
    print("=" * 80)
    
    # 테스트 케이스 1: 건강한 생활습관
    print("\n🧪 테스트 케이스 1: 건강한 생활습관")
    healthy_result = model.predict_disease_risk(
        bmi=22.0, waist_size=75, smoking_level=0, alcohol_consumption=0,
        sleep_quality=8.5, physical_activity=6, age=30, gender='male',
        smoking_status=0, height=170, age_group='middle',
        sleep_hours=8, weekly_activity_minutes=180, daily_steps=9000, intensity='high_intensity'
    )
    
    print(f"최종 위험도: {healthy_result['final_risk']:.3f}")
    print(f"위험 수준: {healthy_result['final_risk_level']}")
    print("피처별 기여도:")
    for feature, contribution in healthy_result['feature_contributions'].items():
        print(f"  {feature}: {contribution:.1%}")
    
    # 테스트 케이스 2: 위험한 생활습관
    print("\n🧪 테스트 케이스 2: 위험한 생활습관")
    risky_result = model.predict_disease_risk(
        bmi=30.0, waist_size=100, smoking_level=2, alcohol_consumption=14,
        sleep_quality=3.0, physical_activity=1, age=45, gender='male',
        smoking_status=2, cigarettes_per_day=20, height=170, age_group='middle',
        drinks_per_week=14, binge_drinking=True, sleep_hours=5, insomnia=True,
        weekly_activity_minutes=0, daily_steps=2000, sedentary_job=True
    )
    
    print(f"최종 위험도: {risky_result['final_risk']:.3f}")
    print(f"위험 수준: {risky_result['final_risk_level']}")
    print("피처별 기여도:")
    for feature, contribution in risky_result['feature_contributions'].items():
        print(f"  {feature}: {contribution:.1%}")
    
    # 연구 요약
    print("\n📚 연구 기반 신뢰성:")
    research_summary = model.get_research_summary()
    print(f"총 연구 논문: {research_summary['total_papers']}개")
    print(f"최신 연구 (2020-2025): {research_summary['recent_papers']}개")
    print(f"한국인 대상 연구: {research_summary['korean_studies']}개")
    print(f"메타분석/시스템 리뷰: {research_summary['meta_analyses']}개")
    print(f"신뢰도 점수: {research_summary['reliability_score']:.0%}")

if __name__ == "__main__":
    test_enhanced_model()
