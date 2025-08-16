import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processing.data_processor import DataProcessor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def main():
    # 데이터 프로세서 초기화
    processor = DataProcessor()
    
    print("=" * 60)
    print("수명 예측 AI 시스템 - 데이터 분석")
    print("=" * 60)
    
    # 1. 건강 및 라이프스타일 데이터 분석
    print("\n1. 건강 및 라이프스타일 데이터 분석")
    print("-" * 40)
    
    health_data = processor.load_data('../data/health_lifestyle_classification.csv')
    if health_data is not None:
        processor.basic_info(health_data)
        
        # 결측치 처리
        health_data = processor.handle_missing_values(health_data)
        
        # 범주형 변수 인코딩
        health_data = processor.encode_categorical(health_data)
        
        # 주요 수치형 변수들의 분포 확인
        numeric_columns = health_data.select_dtypes(include=['int64', 'float64']).columns[:9]
        processor.create_distribution_plots(health_data, numeric_columns)
        
        # 상관관계 분석
        processor.create_correlation_plot(health_data)
    
    # 2. 수명 예측 데이터 분석
    print("\n2. 수명 예측 데이터 분석")
    print("-" * 40)
    
    life_data = processor.load_data('../data/Life Expectancy Data.csv')
    if life_data is not None:
        processor.basic_info(life_data)
        
        # 결측치 처리
        life_data = processor.handle_missing_values(life_data)
        
        # 범주형 변수 인코딩
        life_data = processor.encode_categorical(life_data)
        
        # 주요 수치형 변수들의 분포 확인
        numeric_columns = life_data.select_dtypes(include=['int64', 'float64']).columns[:9]
        processor.create_distribution_plots(life_data, numeric_columns)
        
        # 상관관계 분석
        processor.create_correlation_plot(life_data)
    
    print("\n데이터 분석 완료!")

if __name__ == "__main__":
    main()

