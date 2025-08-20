#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
import matplotlib.font_manager as fm

def create_system_architecture():
    """시스템 아키텍처 다이어그램 생성"""
    
    # 한글 폰트 설정
    plt.rcParams['font.family'] = 'Arial Unicode MS'
    plt.rcParams['axes.unicode_minus'] = False
    
    # 폰트가 없으면 기본 폰트 사용
    try:
        fm.findfont('Arial Unicode MS')
    except:
        plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # 프레젠테이션 생성
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # 색상 정의
    colors = {
        'input': '#E8F4FD',
        'dl': '#FFE6E6',
        'research': '#E6FFE6',
        'ml': '#FFF2E6',
        'output': '#F0E6FF',
        'ui': '#FFFFE6'
    }
    
    # 제목
    ax.text(6, 11.5, 'Deep Learning-based Life Expectancy Prediction AI System', 
            ha='center', va='center', fontsize=18, fontweight='bold')
    
    # 입력 데이터 (중앙 상단)
    input_box = FancyBboxPatch((4.5, 9.5), 3, 1, 
                              boxstyle="round,pad=0.1", 
                              facecolor=colors['input'], 
                              edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(6, 10, 'User Input\n(Age, Gender, BMI, etc.)', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    
    # 딥러닝 모델들 (상단 4개)
    dl_models = [
        ('Stress/Mental\nHealth Model', 1.5, 7.5),
        ('Physical Activity\nModel', 4.5, 7.5),
        ('Genetic Risk\nModel', 7.5, 7.5),
        ('Life Expectancy\nModel', 10.5, 7.5)
    ]
    
    for name, x, y in dl_models:
        dl_box = FancyBboxPatch((x-1, y-0.6), 2, 1.2, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['dl'], 
                               edgecolor='red', linewidth=2)
        ax.add_patch(dl_box)
        ax.text(x, y, name, ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 연구 기반 가중치 (중앙)
    research_box = FancyBboxPatch((3, 5.5), 6, 1, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor=colors['research'], 
                                 edgecolor='green', linewidth=2)
    ax.add_patch(research_box)
    ax.text(6, 6, 'Research-based Weight System\n(20 Papers)', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    # 보조 ML 모델들 (하단 4개)
    ml_models = [
        ('Random\nForest', 1.5, 3.5),
        ('Gradient\nBoosting', 4.5, 3.5),
        ('Linear\nRegression', 7.5, 3.5),
        ('SVR', 10.5, 3.5)
    ]
    
    for name, x, y in ml_models:
        ml_box = FancyBboxPatch((x-0.8, y-0.5), 1.6, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['ml'], 
                               edgecolor='orange', linewidth=2)
        ax.add_patch(ml_box)
        ax.text(x, y, name, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # 출력 결과 (중앙 하단)
    output_box = FancyBboxPatch((4.5, 1.5), 3, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['output'], 
                               edgecolor='purple', linewidth=2)
    ax.add_patch(output_box)
    ax.text(6, 2, 'Final Prediction\n(Life + Recommendations)', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    
    # 사용자 인터페이스 (최하단)
    ui_boxes = [
        ('CLI\nInterface', 2, 0.5),
        ('Streamlit\nWeb App', 10, 0.5)
    ]
    
    for name, x, y in ui_boxes:
        ui_box = FancyBboxPatch((x-1.2, y-0.4), 2.4, 0.8, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['ui'], 
                               edgecolor='blue', linewidth=2)
        ax.add_patch(ui_box)
        ax.text(x, y, name, ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 화살표 연결 (더 깔끔하게)
    arrows = [
        # 입력에서 딥러닝으로
        ((6, 9.5), (1.5, 8.1)),  # 입력 -> 스트레스
        ((6, 9.5), (4.5, 8.1)),  # 입력 -> 신체활동
        ((6, 9.5), (7.5, 8.1)),  # 입력 -> 유전적 위험도
        ((6, 9.5), (10.5, 8.1)), # 입력 -> 수명 예측
        
        # 딥러닝에서 연구 기반으로
        ((1.5, 6.9), (6, 6.5)),  # 스트레스 -> 연구 기반
        ((4.5, 6.9), (6, 6.5)),  # 신체활동 -> 연구 기반
        ((7.5, 6.9), (6, 6.5)),  # 유전적 위험도 -> 연구 기반
        ((10.5, 6.9), (6, 6.5)), # 수명 예측 -> 연구 기반
        
        # 연구 기반에서 ML로
        ((6, 5.5), (1.5, 4.0)),  # 연구 기반 -> Random Forest
        ((6, 5.5), (4.5, 4.0)),  # 연구 기반 -> Gradient Boosting
        ((6, 5.5), (7.5, 4.0)),  # 연구 기반 -> Linear Regression
        ((6, 5.5), (10.5, 4.0)), # 연구 기반 -> SVR
        
        # ML에서 출력으로
        ((1.5, 3.0), (6, 2.5)),  # Random Forest -> 출력
        ((4.5, 3.0), (6, 2.5)),  # Gradient Boosting -> 출력
        ((7.5, 3.0), (6, 2.5)),  # Linear Regression -> 출력
        ((10.5, 3.0), (6, 2.5)), # SVR -> 출력
        
        # 출력에서 UI로
        ((4.5, 1.5), (2, 0.9)),  # 출력 -> CLI
        ((7.5, 1.5), (10, 0.9))  # 출력 -> Streamlit
    ]
    
    for start, end in arrows:
        arrow = ConnectionPatch(start, end, "data", "data",
                               arrowstyle="->", shrinkA=5, shrinkB=5,
                               mutation_scale=20, fc="black", linewidth=2)
        ax.add_patch(arrow)
    
    # 범례 (우상단)
    legend_elements = [
        patches.Patch(color=colors['input'], label='Input Data'),
        patches.Patch(color=colors['dl'], label='Deep Learning Models'),
        patches.Patch(color=colors['research'], label='Research-based Weights'),
        patches.Patch(color=colors['ml'], label='Auxiliary ML Models'),
        patches.Patch(color=colors['output'], label='Output Results'),
        patches.Patch(color=colors['ui'], label='User Interfaces')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98),
              fontsize=10, frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig('system_architecture.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ 시스템 아키텍처 다이어그램이 생성되었습니다!")

if __name__ == "__main__":
    create_system_architecture()
