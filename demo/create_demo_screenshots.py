#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as patches
import matplotlib.font_manager as fm

def create_demo_screenshots():
    """시스템 데모 스크린샷 생성"""
    
    # 폰트 설정
    plt.rcParams['font.family'] = 'Arial Unicode MS'
    plt.rcParams['axes.unicode_minus'] = False
    
    # 폰트가 없으면 기본 폰트 사용
    try:
        fm.findfont('Arial Unicode MS')
    except:
        plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # 1. CLI 인터페이스 스크린샷
    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 8))
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    
    # CLI 배경
    cli_bg = Rectangle((0, 0), 10, 10, facecolor='black', alpha=0.9)
    ax1.add_patch(cli_bg)
    
    # CLI 텍스트
    cli_texts = [
        "Deep Learning-based Life Expectancy Prediction AI System",
        "==================================================",
        "",
        "Enter your age (18-100): 25",
        "Select your gender (male/female): male",
        "Enter your BMI (15-50): 24.5",
        "Select smoking status (0:Never, 1:Former, 2:Current): 0",
        "Enter weekly alcohol consumption (drinks/week): 3",
        "Enter weekly physical activity (minutes): 180",
        "Enter sleep quality score (1-10): 7",
        "",
        "[AI] Deep Learning Model Analysis...",
        "[DATA] Research-based Weight Calculation...",
        "",
        "=== Prediction Results ===",
        "Expected Life Expectancy: 82.3 years",
        "Confidence Level: 91.8%",
        "",
        "=== Health Recommendations ===",
        "[OK] Your current health status is good.",
        "[TIP] Suggestions for better health:",
        "   - Increase weekly physical activity to 300 minutes",
        "   - Improve sleep quality to 8 points",
        "",
        "Improved life expectancy: 84.1 years (+1.8 years)",
        "",
        "Press Enter to continue..."
    ]
    
    y_pos = 9.5
    for text in cli_texts:
        if text.startswith("[AI]") or text.startswith("[DATA]"):
            color = 'yellow'
        elif text.startswith("==="):
            color = 'cyan'
        elif text.startswith("Expected Life") or text.startswith("Confidence"):
            color = 'green'
        elif text.startswith("[OK]") or text.startswith("[TIP]"):
            color = 'lightgreen'
        elif text.startswith("Improved"):
            color = 'orange'
        else:
            color = 'white'
        
        ax1.text(0.5, y_pos, text, fontsize=10, color=color, 
                fontfamily='monospace', fontweight='bold')
        y_pos -= 0.4
    
    ax1.set_title('CLI Interface Demo', fontsize=16, fontweight='bold', color='white')
    plt.savefig('cli_demo_screenshot.png', dpi=300, bbox_inches='tight', 
                facecolor='black', edgecolor='none')
    plt.show()
    
    # 2. 웹앱 인터페이스 스크린샷
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8))
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    
    # 웹앱 배경
    web_bg = Rectangle((0, 0), 10, 10, facecolor='#f0f2f6', alpha=0.9)
    ax2.add_patch(web_bg)
    
    # 헤더
    header = FancyBboxPatch((0, 9), 10, 1, 
                           boxstyle="round,pad=0.1", 
                           facecolor='#1f77b4', 
                           edgecolor='#1f77b4', linewidth=2)
    ax2.add_patch(header)
    ax2.text(5, 9.5, 'Deep Learning-based Life Expectancy Prediction AI System', 
             ha='center', va='center', fontsize=16, fontweight='bold', color='white')
    
    # 입력 폼
    form_bg = FancyBboxPatch((0.5, 6), 4, 2.5, 
                            boxstyle="round,pad=0.1", 
                            facecolor='white', 
                            edgecolor='#ddd', linewidth=1)
    ax2.add_patch(form_bg)
    ax2.text(2.5, 8.2, 'Health Information Input', 
             ha='center', va='center', fontsize=14, fontweight='bold')
    
    # 입력 필드들
    fields = [
        ('Age:', '25'),
        ('Gender:', 'Male'),
        ('BMI:', '24.5'),
        ('Smoking Status:', 'Never'),
        ('Alcohol Consumption:', '3 drinks/week'),
        ('Physical Activity:', '180 min/week'),
        ('Sleep Quality:', '7/10')
    ]
    
    y_pos = 7.8
    for label, value in fields:
        ax2.text(1, y_pos, label, fontsize=10, fontweight='bold')
        ax2.text(3, y_pos, value, fontsize=10, color='#666')
        y_pos -= 0.25
    
    # 결과 영역
    result_bg = FancyBboxPatch((5.5, 6), 4, 2.5, 
                              boxstyle="round,pad=0.1", 
                              facecolor='#e8f5e8', 
                              edgecolor='#4caf50', linewidth=2)
    ax2.add_patch(result_bg)
    ax2.text(7.5, 8.2, 'Prediction Results', 
             ha='center', va='center', fontsize=14, fontweight='bold', color='#2e7d32')
    
    ax2.text(6, 7.5, 'Expected Life:', fontsize=12, fontweight='bold')
    ax2.text(8, 7.5, '82.3 years', fontsize=16, fontweight='bold', color='#2e7d32')
    
    ax2.text(6, 7.1, 'Confidence:', fontsize=12, fontweight='bold')
    ax2.text(8, 7.1, '91.8%', fontsize=12, color='#2e7d32')
    
    ax2.text(6, 6.7, 'Health Status:', fontsize=12, fontweight='bold')
    ax2.text(8, 6.7, 'Good', fontsize=12, color='#2e7d32')
    
    # 차트 영역
    chart_bg = FancyBboxPatch((0.5, 2), 9, 3.5, 
                             boxstyle="round,pad=0.1", 
                             facecolor='white', 
                             edgecolor='#ddd', linewidth=1)
    ax2.add_patch(chart_bg)
    ax2.text(5, 5.2, 'Health Risk Analysis', 
             ha='center', va='center', fontsize=14, fontweight='bold')
    
    # 간단한 차트
    categories = ['Smoking', 'BMI', 'Alcohol', 'Physical Activity', 'Sleep']
    values = [0, 15, 20, 85, 70]
    colors = ['#4caf50', '#ff9800', '#f44336', '#2196f3', '#9c27b0']
    
    for i, (cat, val, color) in enumerate(zip(categories, values, colors)):
        x = 1.5 + i * 1.5
        bar = Rectangle((x, 2.5), 0.8, val/20, facecolor=color, alpha=0.8)
        ax2.add_patch(bar)
        ax2.text(x + 0.4, 2.3, cat, ha='center', va='center', fontsize=10)
        ax2.text(x + 0.4, 2.5 + val/20 + 0.1, f'{val}%', 
                ha='center', va='center', fontsize=9, fontweight='bold')
    
    ax2.set_title('Streamlit Web App Interface Demo', fontsize=16, fontweight='bold')
    plt.savefig('webapp_demo_screenshot.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    create_demo_screenshots()
