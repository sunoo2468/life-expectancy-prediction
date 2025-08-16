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
    print("수명 예측 AI 시스템 - 모델 훈련")
    print("=" * 60)
    
    # 데이터 프로세서 초기화
    processor = DataProcessor()
    
    # 1. 건강 및 라이프스타일 데이터 로드 및 전처리
    print("\n1. 건강 및 라이프스타일 데이터 처리")
    print("-" * 40)
    
    health_data = processor.load_data('../data/health_lifestyle_classification.csv')
    if health_data is None:
        print("데이터 로드 실패!")
        return
    
    # 기본 정보 출력
    processor.basic_info(health_data)
    
    # 결측치 처리
    health_data = processor.handle_missing_values(health_data)
    
    # 범주형 변수 인코딩
    health_data = processor.encode_categorical(health_data)
    
    # 타겟 변수 확인
    print(f"\n타겟 변수 분포:")
    print(health_data['target'].value_counts())
    
    # 특성과 타겟 분리
    X, y = processor.prepare_features(health_data, 'target')
    
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
    
    model.save_model('../models/life_expectancy_model.pkl')
    
    # 5. 테스트 예측
    print("\n5. 테스트 예측")
    print("-" * 40)
    
    # 샘플 사용자 데이터로 예측 테스트
    sample_user = X_test_scaled.iloc[0:1]
    prediction = model.predict(sample_user)[0]
    print(f"샘플 사용자 예상 수명: {prediction:.1f}세")
    
    # 건강 권장사항 생성
    recommendations = model.get_health_recommendations(sample_user)
    print("\n건강 권장사항:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    print("\n모델 훈련 완료!")

if __name__ == "__main__":
    main()

