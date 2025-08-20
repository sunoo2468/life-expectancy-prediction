# PPT에 넣을 핵심 코드 스니펫들

## 1. 딥러닝 모델 구조 코드

### 4개 신경망 모델 아키텍처
```python
# 스트레스/정신건강 예측 모델
def create_stress_mental_model():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(9,)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(2, activation='tanh')  # 스트레스, 정신건강
    ])
    return model

# 신체활동 예측 모델
def create_physical_activity_model():
    model = Sequential([
        Dense(128, activation='relu', input_shape=(8,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(2, activation='linear')  # 신체활동, 걸음수
    ])
    return model

# 유전적 위험도 예측 모델
def create_genetic_risk_model():
    model = Sequential([
        Dense(96, activation='relu', input_shape=(8,)),
        Dropout(0.3),
        Dense(48, activation='relu'),
        Dense(24, activation='relu'),
        Dense(1, activation='sigmoid')  # 유전적 위험도
    ])
    return model

# 직접 수명 예측 모델
def create_life_expectancy_model():
    model = Sequential([
        Dense(256, activation='relu', input_shape=(17,)),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')  # 직접 수명 예측
    ])
    return model
```

## 2. 연구 기반 가중치 계산 코드

### 통합 위험도 계산
```python
class IntegratedWeightCalculator:
    def __init__(self):
        # 20개 논문 기반 가중치
        self.weights = {
            'smoking': 1.62,      # 8개 논문
            'bmi_waist': 1.45,    # 2개 논문
            'alcohol': 1.38,      # 2개 논문
            'sleep': 1.28,        # 5개 논문
            'physical_activity': 1.15  # 3개 논문
        }
    
    def calculate_integrated_risk(self, user_data):
        """통합 위험도 계산"""
        risk_score = 0
        
        # 흡연 위험도
        smoking_risk = self.calculate_smoking_risk(user_data['smoking_status'])
        risk_score += smoking_risk * self.weights['smoking']
        
        # BMI & 허리둘레 위험도
        bmi_risk = self.calculate_bmi_risk(user_data['bmi'], user_data['waist_size'])
        risk_score += bmi_risk * self.weights['bmi_waist']
        
        # 알코올 위험도
        alcohol_risk = self.calculate_alcohol_risk(user_data['drinks_per_week'])
        risk_score += alcohol_risk * self.weights['alcohol']
        
        # 수면 위험도
        sleep_risk = self.calculate_sleep_risk(user_data['sleep_quality_score'])
        risk_score += sleep_risk * self.weights['sleep']
        
        # 신체활동 위험도
        activity_risk = self.calculate_activity_risk(user_data['weekly_activity_minutes'])
        risk_score += activity_risk * self.weights['physical_activity']
        
        return risk_score
```

## 3. 하이브리드 예측 시스템 코드

### 메인 예측 시스템
```python
class LifeExpectancyPredictionModel:
    def __init__(self):
        self.deep_learning_models = {}
        self.weight_calculator = IntegratedWeightCalculator()
        self.auxiliary_models = {}
        self._load_models()
    
    def predict_life_expectancy(self, input_features):
        """하이브리드 예측 시스템"""
        try:
            # 1순위: 딥러닝 예측
            dl_predictions = self.predict_with_deep_learning(input_features)
            
            if dl_predictions and self._is_realistic_prediction(dl_predictions):
                return self._format_dl_result(dl_predictions)
            
            # 2순위: 연구 기반 가중치
            research_prediction = self.predict_with_research_weights(input_features)
            
            # 3순위: 보조 ML 모델
            ml_confidence = self.calculate_ml_confidence(input_features)
            
            return self._format_hybrid_result(research_prediction, ml_confidence)
            
        except Exception as e:
            print(f"예측 중 오류 발생: {e}")
            return self._fallback_prediction(input_features)
    
    def predict_with_deep_learning(self, input_features):
        """딥러닝 모델들을 사용한 예측"""
        predictions = {}
        
        # 4개 신경망 모델 예측
        for model_name, model in self.deep_learning_models.items():
            if model is not None:
                try:
                    prediction = model.predict(self._prepare_input(input_features))
                    predictions[model_name] = prediction
                except Exception as e:
                    print(f"{model_name} 예측 실패: {e}")
        
        return predictions
```

## 4. 사용자 인터페이스 코드

### CLI 인터페이스
```python
def run_cli_interface():
    """CLI 인터페이스 실행"""
    print("딥러닝 기반 수명 예측 AI 시스템")
    print("=" * 50)
    
    # 사용자 입력 수집
    user_data = {}
    
    user_data['age'] = int(input("나이를 입력하세요 (18-100): "))
    user_data['gender'] = input("성별을 선택하세요 (male/female): ")
    user_data['bmi'] = float(input("BMI를 입력하세요 (15-50): "))
    user_data['smoking_status'] = int(input("흡연 상태를 선택하세요 (0:비흡연, 1:과거흡연, 2:현재흡연): "))
    user_data['drinks_per_week'] = int(input("주간 알코올 섭취량을 입력하세요 (잔/주): "))
    user_data['weekly_activity_minutes'] = int(input("주간 신체활동 시간을 입력하세요 (분): "))
    user_data['sleep_quality_score'] = float(input("수면 품질 점수를 입력하세요 (1-10): "))
    
    # 예측 실행
    model = LifeExpectancyPredictionModel()
    result = model.predict_life_expectancy(user_data)
    
    # 결과 출력
    print("\n" + "=" * 20 + " 예측 결과 " + "=" * 20)
    print(f"예상 수명: {result['life_expectancy']:.1f}세")
    print(f"신뢰도: {result['confidence']:.1f}%")
    print(f"예측 방법: {result['prediction_method']}")
    
    # 권장사항 출력
    print("\n" + "=" * 20 + " 건강 권장사항 " + "=" * 20)
    for recommendation in result['recommendations']:
        print(f"💡 {recommendation}")
```

### Streamlit 웹앱
```python
import streamlit as st

def main():
    st.title("딥러닝 기반 수명 예측 AI 시스템")
    
    # 사이드바 - 입력 폼
    with st.sidebar:
        st.header("건강 정보 입력")
        
        age = st.slider("나이", 18, 100, 30)
        gender = st.selectbox("성별", ["male", "female"])
        bmi = st.slider("BMI", 15.0, 50.0, 22.0, 0.1)
        smoking_status = st.selectbox("흡연 상태", 
                                    ["비흡연", "과거흡연", "현재흡연"])
        drinks_per_week = st.slider("주간 알코올 섭취량 (잔)", 0, 20, 3)
        weekly_activity = st.slider("주간 신체활동 (분)", 0, 600, 150)
        sleep_quality = st.slider("수면 품질 (1-10)", 1, 10, 7)
        
        if st.button("수명 예측하기"):
            # 예측 실행
            user_data = {
                'age': age, 'gender': gender, 'bmi': bmi,
                'smoking_status': ["비흡연", "과거흡연", "현재흡연"].index(smoking_status),
                'drinks_per_week': drinks_per_week,
                'weekly_activity_minutes': weekly_activity,
                'sleep_quality_score': sleep_quality
            }
            
            model = LifeExpectancyPredictionModel()
            result = model.predict_life_expectancy(user_data)
            
            # 결과 표시
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("예상 수명", f"{result['life_expectancy']:.1f}세")
            with col2:
                st.metric("신뢰도", f"{result['confidence']:.1f}%")
            with col3:
                st.metric("예측 방법", result['prediction_method'])
            
            # 권장사항 표시
            st.subheader("건강 권장사항")
            for rec in result['recommendations']:
                st.write(f"💡 {rec}")
            
            # 차트 표시
            st.subheader("건강 위험도 분석")
            risk_data = {
                '흡연': result['risk_factors']['smoking'],
                'BMI': result['risk_factors']['bmi'],
                '알코올': result['risk_factors']['alcohol'],
                '신체활동': result['risk_factors']['activity'],
                '수면': result['risk_factors']['sleep']
            }
            st.bar_chart(risk_data)

if __name__ == "__main__":
    main()
```

## 5. 성능 평가 코드

### 모델 성능 평가
```python
def evaluate_model_performance():
    """모델 성능 평가"""
    # 테스트 데이터로 성능 측정
    test_data = generate_test_data(1000)
    
    model = LifeExpectancyPredictionModel()
    predictions = []
    actuals = []
    
    for data in test_data:
        pred = model.predict_life_expectancy(data['features'])
        predictions.append(pred['life_expectancy'])
        actuals.append(data['actual_life_expectancy'])
    
    # 성능 지표 계산
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    r2 = r2_score(actuals, predictions)
    correlation = np.corrcoef(actuals, predictions)[0, 1]
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'Correlation': correlation,
        'Accuracy': calculate_accuracy(actuals, predictions)
    }

def calculate_accuracy(actuals, predictions, tolerance=2.0):
    """정확도 계산 (허용 오차 내 예측 비율)"""
    correct = sum(1 for a, p in zip(actuals, predictions) 
                 if abs(a - p) <= tolerance)
    return (correct / len(actuals)) * 100
```

## 6. 시스템 통합 코드

### 메인 실행 파일
```python
#!/usr/bin/env python3
"""
딥러닝 기반 수명 예측 AI 시스템
메인 실행 파일
"""

import sys
import argparse
from src.model.life_expectancy_prediction_model import LifeExpectancyPredictionModel

def main():
    parser = argparse.ArgumentParser(description='수명 예측 AI 시스템')
    parser.add_argument('--mode', choices=['cli', 'web'], default='cli',
                       help='실행 모드 (cli 또는 web)')
    parser.add_argument('--port', type=int, default=8501,
                       help='웹앱 포트 (기본값: 8501)')
    
    args = parser.parse_args()
    
    if args.mode == 'cli':
        from notebooks.personal_input_predict import run_cli_interface
        run_cli_interface()
    elif args.mode == 'web':
        import subprocess
        subprocess.run(['streamlit', 'run', 'web_app/app.py', 
                       '--server.port', str(args.port)])

if __name__ == "__main__":
    main()
```

코드 스니펫: 기술적 깊이와 구현 내용을 효과적으로 제시
