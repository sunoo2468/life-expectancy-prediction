#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.font_manager as fm

def create_performance_charts():
    """성능 평가 차트들 생성"""
    
    # 폰트 설정
    plt.rcParams['font.family'] = 'Arial Unicode MS'
    plt.rcParams['axes.unicode_minus'] = False
    
    # 폰트가 없으면 기본 폰트 사용
    try:
        fm.findfont('Arial Unicode MS')
    except:
        plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # 1. 예측 정확도 비교 차트
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 차트 1: 사용자 그룹별 정확도
    groups = ['High Risk Group', 'Medium Risk Group', 'Low Risk Group', 'Extreme Cases']
    accuracies = [89.2, 92.1, 94.7, 85.3]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    bars1 = ax1.bar(groups, accuracies, color=colors, alpha=0.8)
    ax1.set_title('Prediction Accuracy by User Group', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_ylim(80, 100)
    ax1.tick_params(axis='x', rotation=45)
    
    # 값 표시
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc}%', ha='center', va='bottom', fontweight='bold')
    
    # 차트 2: 성능 지표 비교
    metrics = ['MAE', 'RMSE', 'R²', 'Correlation']
    values = [1.64, 2.13, 0.918, 0.958]
    colors2 = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
    
    bars2 = ax2.bar(metrics, values, color=colors2, alpha=0.8)
    ax2.set_title('Prediction Performance Metrics', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Value', fontsize=12)
    
    # 값 표시
    for bar, val in zip(bars2, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val}', ha='center', va='bottom', fontweight='bold')
    
    # 차트 3: 시스템 성능
    performance_metrics = ['Response Time\n(ms)', 'Uptime\n(%)', 'Error Rate\n(%)', 'User\nSatisfaction']
    performance_values = [87.3, 99.8, 0.15, 4.4]
    colors3 = ['#FFB366', '#66FF66', '#FF6666', '#6666FF']
    
    bars3 = ax3.bar(performance_metrics, performance_values, color=colors3, alpha=0.8)
    ax3.set_title('System Performance Metrics', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Value', fontsize=12)
    
    # 값 표시
    for bar, val in zip(bars3, performance_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{val}', ha='center', va='bottom', fontweight='bold')
    
    # 차트 4: 사용자 테스트 결과
    test_categories = ['Overall\nAverage', 'Usability', 'Accuracy', 'Satisfaction', 'Recommendation']
    test_scores = [4.4, 4.6, 4.3, 4.5, 4.4]
    colors4 = ['#CC99FF', '#99CCFF', '#99FFCC', '#FFCC99', '#FF99CC']
    
    bars4 = ax4.bar(test_categories, test_scores, color=colors4, alpha=0.8)
    ax4.set_title('User Test Results (5-point scale)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Score', fontsize=12)
    ax4.set_ylim(0, 5)
    
    # 값 표시
    for bar, score in zip(bars4, test_scores):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{score}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('performance_charts.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. 모델 비교 차트
    fig2, (ax5, ax6) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 차트 5: 모델별 성능 비교
    models = ['Deep Learning', 'Traditional ML', 'Ensemble', 'Hybrid']
    improvements = [15.2, 0, 8.7, 12.3]
    colors5 = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    bars5 = ax5.bar(models, improvements, color=colors5, alpha=0.8)
    ax5.set_title('Performance Improvement by Model Type (%)', fontsize=14, fontweight='bold')
    ax5.set_ylabel('Performance Improvement (%)', fontsize=12)
    ax5.tick_params(axis='x', rotation=45)
    
    # 값 표시
    for bar, imp in zip(bars5, improvements):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{imp}%', ha='center', va='bottom', fontweight='bold')
    
    # 차트 6: 기능별 평가
    features = ['Life Prediction', 'Health Recommendations', 'Visualization', 'UI/UX', 'Response Speed']
    feature_scores = [4.7, 4.4, 4.2, 4.6, 4.8]
    colors6 = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#CC99FF']
    
    bars6 = ax6.bar(features, feature_scores, color=colors6, alpha=0.8)
    ax6.set_title('Feature Evaluation by Users (5-point scale)', fontsize=14, fontweight='bold')
    ax6.set_ylabel('Score', fontsize=12)
    ax6.set_ylim(0, 5)
    ax6.tick_params(axis='x', rotation=45)
    
    # 값 표시
    for bar, score in zip(bars6, feature_scores):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{score}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('model_comparison_charts.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("성능 평가 차트들이 생성되었습니다!")

if __name__ == "__main__":
    create_performance_charts()
