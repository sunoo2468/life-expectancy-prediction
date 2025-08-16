import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

class LifeExpectancyModel:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(random_state=42),
            'gradient_boosting': GradientBoostingRegressor(random_state=42),
            'linear_regression': LinearRegression(),
            'ridge': Ridge(random_state=42),
            'lasso': Lasso(random_state=42),
            'svr': SVR()
        }
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        
    def train_models(self, X_train, y_train, X_test, y_test):
        """여러 모델을 훈련하고 성능을 비교"""
        print("=" * 60)
        print("모델 훈련 및 성능 비교")
        print("=" * 60)
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\n{name} 모델 훈련 중...")
            
            # 모델 훈련
            model.fit(X_train, y_train)
            
            # 예측
            y_pred = model.predict(X_test)
            
            # 성능 평가
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'rmse': np.sqrt(mse)
            }
            
            print(f"  MSE: {mse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  R²: {r2:.4f}")
            print(f"  RMSE: {np.sqrt(mse):.4f}")
        
        # 최고 성능 모델 선택
        best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
        self.best_model = results[best_model_name]['model']
        self.best_model_name = best_model_name
        
        print(f"\n최고 성능 모델: {best_model_name}")
        print(f"R² Score: {results[best_model_name]['r2']:.4f}")
        
        # 특성 중요도 계산 (Random Forest 또는 Gradient Boosting인 경우)
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        return results
    
    def hyperparameter_tuning(self, X_train, y_train, model_name='random_forest'):
        """하이퍼파라미터 튜닝"""
        print(f"\n{model_name} 하이퍼파라미터 튜닝 중...")
        
        if model_name == 'random_forest':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestRegressor(random_state=42)
        
        elif model_name == 'gradient_boosting':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            }
            model = GradientBoostingRegressor(random_state=42)
        
        else:
            print("지원하지 않는 모델입니다.")
            return None
        
        # Grid Search
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        print(f"최적 파라미터: {grid_search.best_params_}")
        print(f"최적 점수: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def predict(self, X):
        """수명 예측"""
        if self.best_model is None:
            raise ValueError("모델이 훈련되지 않았습니다. 먼저 train_models()를 실행하세요.")
        
        return self.best_model.predict(X)
    
    def get_feature_importance(self, top_n=10):
        """특성 중요도 반환"""
        if self.feature_importance is None:
            return None
        
        return self.feature_importance.head(top_n)
    
    def save_model(self, filepath):
        """모델 저장"""
        if self.best_model is None:
            raise ValueError("저장할 모델이 없습니다.")
        
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'feature_importance': self.feature_importance,
            'scaler': self.scaler
        }
        
        joblib.dump(model_data, filepath)
        print(f"모델이 {filepath}에 저장되었습니다.")
    
    def load_model(self, filepath):
        """모델 로드"""
        model_data = joblib.load(filepath)
        
        self.best_model = model_data['model']
        self.best_model_name = model_data['model_name']
        self.feature_importance = model_data['feature_importance']
        self.scaler = model_data['scaler']
        
        print(f"모델이 {filepath}에서 로드되었습니다.")
    
    def get_health_recommendations(self, user_data, target_life_expectancy=80):
        """건강 습관 개선 권장사항 생성"""
        if self.feature_importance is None:
            return "특성 중요도 정보가 없습니다."
        
        recommendations = []
        current_prediction = self.predict(user_data)[0]
        
        # 상위 중요 특성들에 대한 권장사항
        top_features = self.feature_importance.head(5)
        
        for _, row in top_features.iterrows():
            feature = row['feature']
            importance = row['importance']
            
            if feature in user_data.columns:
                current_value = user_data[feature].iloc[0]
                
                # 특성별 권장사항 생성
                recommendation = self._generate_recommendation(feature, current_value, importance)
                if recommendation:
                    recommendations.append(recommendation)
        
        # 전체적인 권장사항
        if current_prediction < target_life_expectancy:
            recommendations.append(f"현재 예상 수명: {current_prediction:.1f}세")
            recommendations.append(f"목표 수명까지 {target_life_expectancy - current_prediction:.1f}년 더 필요합니다.")
        else:
            recommendations.append(f"현재 예상 수명: {current_prediction:.1f}세")
            recommendations.append("훌륭합니다! 건강한 생활을 유지하세요.")
        
        return recommendations
    
    def _generate_recommendation(self, feature, current_value, importance):
        """특성별 구체적인 권장사항 생성"""
        recommendations = {
            'bmi': {
                'high': 'BMI가 높습니다. 체중을 줄이고 규칙적인 운동을 하세요.',
                'low': 'BMI가 낮습니다. 균형 잡힌 식단과 근력 운동을 하세요.'
            },
            'blood_pressure': {
                'high': '혈압이 높습니다. 소금 섭취를 줄이고 정기적인 운동을 하세요.',
                'low': '혈압이 정상입니다. 현재 상태를 유지하세요.'
            },
            'physical_activity': {
                'low': '신체 활동이 부족합니다. 주 3-4회 30분 이상의 운동을 하세요.',
                'high': '적절한 신체 활동을 하고 있습니다. 계속 유지하세요.'
            },
            'sleep_hours': {
                'low': '수면 시간이 부족합니다. 하루 7-8시간의 충분한 수면을 취하세요.',
                'high': '적절한 수면 시간을 유지하고 있습니다.'
            },
            'stress_level': {
                'high': '스트레스 수준이 높습니다. 명상이나 요가 등 스트레스 해소 활동을 하세요.',
                'low': '스트레스 관리가 잘 되고 있습니다.'
            }
        }
        
        if feature in recommendations:
            # 임계값 기반 권장사항
            if feature == 'bmi':
                if current_value > 25:
                    return recommendations[feature]['high']
                elif current_value < 18.5:
                    return recommendations[feature]['low']
            elif feature == 'blood_pressure':
                if current_value > 140:
                    return recommendations[feature]['high']
                else:
                    return recommendations[feature]['low']
            elif feature == 'physical_activity':
                if current_value < 3:
                    return recommendations[feature]['low']
                else:
                    return recommendations[feature]['high']
            elif feature == 'sleep_hours':
                if current_value < 6:
                    return recommendations[feature]['low']
                else:
                    return recommendations[feature]['high']
            elif feature == 'stress_level':
                if current_value > 7:
                    return recommendations[feature]['high']
                else:
                    return recommendations[feature]['low']
        
        return None

