import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processing.data_processor import DataProcessor
from model.life_expectancy_model import LifeExpectancyModel
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("=" * 60)
    print("수명 예측 AI 시스템 - 실제 수명 데이터 모델 훈련")
    print("=" * 60)
    
    # 데이터 프로세서 초기화
    processor = DataProcessor()
    
    # 1. 수명 예측 데이터 로드 및 전처리
    print("\n1. 수명 예측 데이터 처리")
    print("-" * 40)
    
    life_data = processor.load_data('../data/Life Expectancy Data.csv')
    if life_data is None:
        print("데이터 로드 실패!")
        return
    
    # 기본 정보 출력
    processor.basic_info(life_data)
    
    # 결측치 처리
    life_data = processor.handle_missing_values(life_data)
    
    # 범주형 변수 인코딩
    life_data = processor.encode_categorical(life_data)
    
    # 타겟 변수 확인
    print(f"\n타겟 변수 (Life expectancy) 통계:")
    print(life_data['Life expectancy '].describe())
    
    # 특성과 타겟 분리
    X, y = processor.prepare_features(life_data, 'Life expectancy ')
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = processor.split_data(X, y)
    
    # 특성 스케일링
    X_train_scaled, X_test_scaled = processor.scale_features(X_train, X_test)
    
    # 2. 모델 훈련
    print("\n2. 모델 훈련")
    print("-" * 40)
    
    model = LifeExpectancyModel()
    results = model.train_models(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # 3. 특성 중요도 시각화
    print("\n3. 특성 중요도 분석")
    print("-" * 40)
    
    feature_importance = model.get_feature_importance(15)
    if feature_importance is not None:
        print("\n상위 15개 중요 특성:")
        print(feature_importance)
        
        # 특성 중요도 시각화
        plt.figure(figsize=(12, 8))
        sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
        plt.title('상위 10개 특성 중요도', fontsize=16)
        plt.xlabel('중요도', fontsize=12)
        plt.ylabel('특성', fontsize=12)
        plt.tight_layout()
        plt.show()
    
    # 4. 모델 저장
    print("\n4. 모델 저장")
    print("-" * 40)
    
    model.save_model('../models/life_expectancy_model_final.pkl')
    
    # 5. 테스트 예측
    print("\n5. 테스트 예측")
    print("-" * 40)
    
    # 샘플 사용자 데이터로 예측 테스트
    sample_user = X_test_scaled.iloc[0:1]
    prediction = model.predict(sample_user)[0]
    actual = y_test.iloc[0]
    
    print(f"실제 수명: {actual:.1f}세")
    print(f"예측 수명: {prediction:.1f}세")
    print(f"오차: {abs(actual - prediction):.1f}세")
    
    # 건강 권장사항 생성
    recommendations = model.get_health_recommendations(sample_user)
    print("\n건강 권장사항:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    # 6. 성능 요약
    print("\n6. 모델 성능 요약")
    print("-" * 40)
    
    best_result = results[model.best_model_name]
    print(f"최고 성능 모델: {model.best_model_name}")
    print(f"R² Score: {best_result['r2']:.4f}")
    print(f"RMSE: {best_result['rmse']:.4f}")
    print(f"MAE: {best_result['mae']:.4f}")
    
    print("\n모델 훈련 완료!")

if __name__ == "__main__":
    main()
