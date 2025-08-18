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

class DiseaseRiskModel:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'logistic_regression': LogisticRegression(random_state=42)
        }
        self.best_model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.risk_weights = {
            'bmi': 0.25,
            'waist_size': 0.20,
            'smoking_level': 0.20,
            'alcohol_consumption': 0.15,
            'sleep_quality': 0.10,
            'physical_activity': 0.10
        }
        
    def calculate_weighted_risk_score(self, data):
        """가중치 기반 질병 위험도 점수 계산"""
        risk_score = 0
        
        # BMI 위험도 (18.5-24.9 정상)
        if 'bmi' in data.columns:
            bmi = data['bmi'].iloc[0]
            if bmi < 18.5:
                risk_score += self.risk_weights['bmi'] * 0.3  # 저체중
            elif bmi > 25:
                risk_score += self.risk_weights['bmi'] * (bmi - 25) / 10  # 과체중/비만
            else:
                risk_score += self.risk_weights['bmi'] * 0.1  # 정상
        
        # 허리둘레 위험도
        if 'waist_size' in data.columns:
            waist = data['waist_size'].iloc[0]
            if waist > 90:  # 남성 기준
                risk_score += self.risk_weights['waist_size'] * 0.8
            elif waist > 80:  # 여성 기준
                risk_score += self.risk_weights['waist_size'] * 0.6
            else:
                risk_score += self.risk_weights['waist_size'] * 0.2
        
        # 흡연 위험도
        if 'smoking_level' in data.columns:
            smoking = data['smoking_level'].iloc[0]
            if smoking == 2:  # 현재 흡연
                risk_score += self.risk_weights['smoking_level'] * 0.9
            elif smoking == 1:  # 과거 흡연
                risk_score += self.risk_weights['smoking_level'] * 0.5
            else:
                risk_score += self.risk_weights['smoking_level'] * 0.1
        
        # 알코올 위험도
        if 'alcohol_consumption' in data.columns:
            alcohol = data['alcohol_consumption'].iloc[0]
            if alcohol > 10:
                risk_score += self.risk_weights['alcohol_consumption'] * 0.8
            elif alcohol > 5:
                risk_score += self.risk_weights['alcohol_consumption'] * 0.5
            else:
                risk_score += self.risk_weights['alcohol_consumption'] * 0.2
        
        # 수면의 질 위험도
        if 'sleep_quality' in data.columns:
            sleep_quality = data['sleep_quality'].iloc[0]
            if sleep_quality < 3:  # 낮은 수면의 질
                risk_score += self.risk_weights['sleep_quality'] * 0.7
            elif sleep_quality < 5:
                risk_score += self.risk_weights['sleep_quality'] * 0.4
            else:
                risk_score += self.risk_weights['sleep_quality'] * 0.1
        
        # 신체활동 위험도
        if 'physical_activity' in data.columns:
            activity = data['physical_activity'].iloc[0]
            if activity < 2:  # 낮은 신체활동
                risk_score += self.risk_weights['physical_activity'] * 0.6
            elif activity < 4:
                risk_score += self.risk_weights['physical_activity'] * 0.3
            else:
                risk_score += self.risk_weights['physical_activity'] * 0.1
        
        return min(risk_score, 1.0)  # 최대 1.0으로 제한
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """모델 훈련 및 성능 비교"""
        print("=" * 60)
        print("질병 위험도 예측 모델 훈련")
        print("=" * 60)
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\n{name} 모델 훈련 중...")
            
            # 모델 훈련
            model.fit(X_train, y_train)
            
            # 예측
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # 성능 평가
            accuracy = model.score(X_test, y_test)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'auc': auc,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  AUC: {auc:.4f}")
        
        # 최고 성능 모델 선택
        best_model_name = max(results.keys(), key=lambda x: results[x]['auc'])
        self.best_model = results[best_model_name]['model']
        
        print(f"\n최고 성능 모델: {best_model_name}")
        print(f"AUC Score: {results[best_model_name]['auc']:.4f}")
        
        # 특성 중요도 계산
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        return results
    
    def predict_disease_risk(self, data):
        """질병 위험도 예측"""
        if self.best_model is None:
            raise ValueError("모델이 훈련되지 않았습니다.")
        
        # 가중치 기반 위험도 점수
        weighted_risk = self.calculate_weighted_risk_score(data)
        
        # 머신러닝 모델 예측
        ml_risk = self.best_model.predict_proba(data)[0, 1]
        
        # 두 점수를 결합 (가중 평균)
        combined_risk = 0.6 * weighted_risk + 0.4 * ml_risk
        
        return {
            'weighted_risk': weighted_risk,
            'ml_risk': ml_risk,
            'combined_risk': combined_risk,
            'risk_level': self._get_risk_level(combined_risk)
        }
    
    def _get_risk_level(self, risk_score):
        """위험도 수준 분류"""
        if risk_score < 0.3:
            return "낮음"
        elif risk_score < 0.6:
            return "보통"
        elif risk_score < 0.8:
            return "높음"
        else:
            return "매우 높음"
    
    def get_health_recommendations(self, data, risk_prediction):
        """건강 개선 권장사항 생성"""
        recommendations = []
        risk_score = risk_prediction['combined_risk']
        
        # BMI 권장사항
        if 'bmi' in data.columns:
            bmi = data['bmi'].iloc[0]
            if bmi > 25:
                recommendations.append("🔸 BMI가 높습니다. 체중을 줄이고 규칙적인 운동을 하세요.")
            elif bmi < 18.5:
                recommendations.append("🔸 BMI가 낮습니다. 균형 잡힌 식단과 근력 운동을 하세요.")
        
        # 흡연 권장사항
        if 'smoking_level' in data.columns:
            smoking = data['smoking_level'].iloc[0]
            if smoking > 0:
                recommendations.append("🔸 흡연을 중단하세요. 금연 프로그램에 참여하는 것을 권장합니다.")
        
        # 알코올 권장사항
        if 'alcohol_consumption' in data.columns:
            alcohol = data['alcohol_consumption'].iloc[0]
            if alcohol > 5:
                recommendations.append("🔸 알코올 섭취를 줄이세요. 하루 1-2잔 이하로 제한하세요.")
        
        # 수면 권장사항
        if 'sleep_quality' in data.columns:
            sleep_quality = data['sleep_quality'].iloc[0]
            if sleep_quality < 4:
                recommendations.append("🔸 수면의 질을 개선하세요. 규칙적인 수면 패턴과 편안한 환경을 만드세요.")
        
        # 신체활동 권장사항
        if 'physical_activity' in data.columns:
            activity = data['physical_activity'].iloc[0]
            if activity < 3:
                recommendations.append("🔸 신체활동을 늘리세요. 주 3-4회 30분 이상의 운동을 하세요.")
        
        # 위험도 수준별 추가 권장사항
        if risk_score > 0.7:
            recommendations.append("🔸 정기적인 건강 검진을 받으세요.")
            recommendations.append("🔸 전문의와 상담하여 개인화된 건강 관리 계획을 세우세요.")
        
        return recommendations
    
    def save_model(self, filepath):
        """모델 저장"""
        if self.best_model is None:
            raise ValueError("저장할 모델이 없습니다.")
        
        model_data = {
            'model': self.best_model,
            'feature_importance': self.feature_importance,
            'scaler': self.scaler,
            'risk_weights': self.risk_weights
        }
        
        joblib.dump(model_data, filepath)
        print(f"모델이 {filepath}에 저장되었습니다.")
    
    def load_model(self, filepath):
        """모델 로드"""
        model_data = joblib.load(filepath)
        
        self.best_model = model_data['model']
        self.feature_importance = model_data['feature_importance']
        self.scaler = model_data['scaler']
        self.risk_weights = model_data['risk_weights']
        
        print(f"모델이 {filepath}에서 로드되었습니다.")

