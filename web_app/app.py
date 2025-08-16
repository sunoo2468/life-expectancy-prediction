import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# 프로젝트 루트 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model.life_expectancy_model import LifeExpectancyModel

# 페이지 설정
st.set_page_config(
    page_title="수명 예측 AI 시스템",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .recommendation-card {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

def load_model():
    """모델 로드"""
    try:
        model = LifeExpectancyModel()
        model.load_model('../models/life_expectancy_model_final.pkl')
        return model
    except Exception as e:
        st.error(f"모델 로드 실패: {e}")
        return None

def create_sample_data():
    """샘플 데이터 생성"""
    return {
        'Country': 0,  # 인코딩된 값
        'Year': 2015,
        'Status': 0,  # Developing
        'Adult Mortality': 150,
        'infant deaths': 30,
        'Alcohol': 5.0,
        'percentage expenditure': 5.0,
        'Hepatitis B': 80,
        'Measles ': 100,
        ' BMI ': 25.0,
        'under-five deaths ': 40,
        'Polio': 80,
        'Total expenditure': 5.0,
        'Diphtheria ': 80,
        ' HIV/AIDS': 0.1,
        'GDP': 2000,
        'Population': 1000000,
        ' thinness  1-19 years': 5.0,
        ' thinness 5-9 years': 5.0,
        'Income composition of resources': 0.7,
        'Schooling': 12.0
    }

def main():
    # 헤더
    st.markdown('<h1 class="main-header">🏥 수명 예측 AI 시스템</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">건강 습관을 입력하고 예상 수명을 알아보세요!</p>', unsafe_allow_html=True)
    
    # 모델 로드
    model = load_model()
    if model is None:
        st.error("모델을 로드할 수 없습니다. 관리자에게 문의하세요.")
        return
    
    # 사이드바
    st.sidebar.markdown("## 📊 데이터 입력")
    st.sidebar.markdown("아래 항목들을 입력해주세요:")
    
    # 입력 폼
    with st.sidebar.form("life_expectancy_form"):
        st.markdown("### 기본 정보")
        country = st.selectbox("국가", ["대한민국", "미국", "일본", "중국", "독일", "프랑스", "영국", "캐나다", "호주", "기타"])
        year = st.slider("연도", 2000, 2020, 2015)
        status = st.selectbox("개발 상태", ["개발도상국", "선진국"])
        
        st.markdown("### 건강 지표")
        adult_mortality = st.slider("성인 사망률 (1000명당)", 10, 500, 150)
        infant_deaths = st.slider("영아 사망률 (1000명당)", 1, 100, 30)
        alcohol = st.slider("알코올 소비량 (리터/인)", 0.0, 20.0, 5.0)
        bmi = st.slider("평균 BMI", 15.0, 35.0, 25.0)
        hiv_aids = st.slider("HIV/AIDS 비율 (%)", 0.0, 30.0, 0.1)
        
        st.markdown("### 의료 및 교육")
        health_expenditure = st.slider("의료비 지출 비율 (GDP 대비 %)", 1.0, 20.0, 5.0)
        schooling = st.slider("평균 교육 연수", 0.0, 20.0, 12.0)
        income_composition = st.slider("소득 구성 지수", 0.0, 1.0, 0.7)
        
        st.markdown("### 기타 지표")
        gdp = st.slider("GDP (달러)", 100, 100000, 20000)
        population = st.slider("인구", 100000, 1000000000, 50000000)
        
        submitted = st.form_submit_button("수명 예측하기", type="primary")
    
    # 메인 컨텐츠
    if submitted:
        # 입력 데이터 변환
        input_data = {
            'Country': 0,  # 간단화를 위해 고정값 사용
            'Year': year,
            'Status': 0 if status == "개발도상국" else 1,
            'Adult Mortality': adult_mortality,
            'infant deaths': infant_deaths,
            'Alcohol': alcohol,
            'percentage expenditure': health_expenditure,
            'Hepatitis B': 80,  # 기본값
            'Measles ': 100,  # 기본값
            ' BMI ': bmi,
            'under-five deaths ': infant_deaths * 1.3,  # 추정값
            'Polio': 80,  # 기본값
            'Total expenditure': health_expenditure,
            'Diphtheria ': 80,  # 기본값
            ' HIV/AIDS': hiv_aids,
            'GDP': gdp,
            'Population': population,
            ' thinness  1-19 years': 5.0,  # 기본값
            ' thinness 5-9 years': 5.0,  # 기본값
            'Income composition of resources': income_composition,
            'Schooling': schooling
        }
        
        # 데이터프레임 변환
        input_df = pd.DataFrame([input_data])
        
        # 예측
        try:
            prediction = model.predict(input_df)[0]
            
            # 결과 표시
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("예상 수명", f"{prediction:.1f}세")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                target_age = 80
                years_to_target = max(0, target_age - prediction)
                st.metric("목표 수명까지", f"{years_to_target:.1f}년")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                if prediction >= target_age:
                    status_text = "🟢 우수"
                elif prediction >= 70:
                    status_text = "🟡 양호"
                else:
                    status_text = "🔴 개선 필요"
                st.metric("건강 상태", status_text)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # 건강 권장사항
            st.markdown('<h2 class="sub-header">💡 건강 개선 권장사항</h2>', unsafe_allow_html=True)
            
            recommendations = []
            
            # BMI 기반 권장사항
            if bmi > 25:
                recommendations.append("🔸 BMI가 높습니다. 체중을 줄이고 규칙적인 운동을 하세요.")
            elif bmi < 18.5:
                recommendations.append("🔸 BMI가 낮습니다. 균형 잡힌 식단과 근력 운동을 하세요.")
            
            # 알코올 기반 권장사항
            if alcohol > 10:
                recommendations.append("🔸 알코올 섭취량이 높습니다. 음주를 줄이세요.")
            
            # 교육 기반 권장사항
            if schooling < 10:
                recommendations.append("🔸 교육 수준을 높이면 건강한 생활 습관을 기를 수 있습니다.")
            
            # 소득 기반 권장사항
            if income_composition < 0.5:
                recommendations.append("🔸 경제적 여유가 있으면 더 나은 의료 서비스를 받을 수 있습니다.")
            
            # 일반적인 권장사항
            if prediction < target_age:
                recommendations.append("🔸 정기적인 건강 검진을 받으세요.")
                recommendations.append("🔸 균형 잡힌 식단과 규칙적인 운동을 하세요.")
                recommendations.append("🔸 충분한 수면과 스트레스 관리를 하세요.")
            
            for rec in recommendations:
                st.markdown(f'<div class="recommendation-card">{rec}</div>', unsafe_allow_html=True)
            
            # 시각화
            st.markdown('<h2 class="sub-header">📈 수명 예측 분석</h2>', unsafe_allow_html=True)
            
            # 특성 중요도 시각화
            if model.feature_importance is not None:
                fig = px.bar(
                    model.feature_importance.head(10),
                    x='importance',
                    y='feature',
                    orientation='h',
                    title='수명에 영향을 미치는 주요 요인',
                    labels={'importance': '중요도', 'feature': '요인'}
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            # 수명 분포 시각화
            fig2 = go.Figure()
            fig2.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=prediction,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "예상 수명"},
                delta={'reference': 80},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 60], 'color': "lightgray"},
                        {'range': [60, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 80
                    }
                }
            ))
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
            
        except Exception as e:
            st.error(f"예측 중 오류가 발생했습니다: {e}")
    
    else:
        # 초기 화면
        st.markdown("""
        ## 🎯 프로젝트 개요
        
        이 시스템은 개인의 건강 습관과 사회경제적 요인을 분석하여 예상 수명을 예측하고, 
        건강한 생활을 위한 개인화된 권장사항을 제공합니다.
        
        ### 주요 기능:
        - 📊 **수명 예측**: 머신러닝 기반 정확한 수명 예측
        - 💡 **건강 권장사항**: 개인화된 건강 개선 방안 제시
        - 📈 **시각화**: 직관적인 결과 분석 및 시각화
        
        ### 사용 방법:
        1. 왼쪽 사이드바에서 건강 정보를 입력하세요
        2. "수명 예측하기" 버튼을 클릭하세요
        3. 예측 결과와 건강 권장사항을 확인하세요
        
        ---
        
        **모델 성능**: R² Score 96.88% | 평균 오차 ±1.64세
        """)
        
        # 샘플 데이터로 예시 보여주기
        st.markdown("### 📋 입력 예시")
        sample_data = create_sample_data()
        st.json(sample_data)

if __name__ == "__main__":
    main()
