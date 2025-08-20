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

from model.life_expectancy_prediction_model import LifeExpectancyPredictionModel

# 페이지 설정
st.set_page_config(
    page_title="Life Expectancy Prediction AI System",
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
        model = LifeExpectancyPredictionModel()
        return model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

def main():
    # 헤더
    st.markdown('<h1 class="main-header">🏥 Life Expectancy Prediction AI System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Enter your health habits and discover your expected life expectancy!</p>', unsafe_allow_html=True)
    
    # 모델 로드
    model = load_model()
    if model is None:
        st.error("Cannot load model. Please contact administrator.")
        return
    
    # 사이드바
    st.sidebar.markdown("## 📊 Data Input")
    st.sidebar.markdown("Please enter the following information:")
    
    # 입력 폼
    with st.sidebar.form("life_expectancy_form"):
        st.markdown("### Basic Information")
        age = st.slider("Age", 18, 100, 30)
        gender = st.selectbox("Gender", ["male", "female"])
        
        st.markdown("### Health Indicators")
        bmi = st.slider("BMI", 15.0, 50.0, 22.0, 0.1)
        waist_size = st.slider("Waist Size (cm)", 60, 150, 80)
        smoking_status = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
        drinks_per_week = st.slider("Weekly Alcohol Consumption (drinks)", 0, 20, 3)
        weekly_activity_minutes = st.slider("Weekly Physical Activity (minutes)", 0, 600, 150)
        sleep_quality_score = st.slider("Sleep Quality Score (1-10)", 1, 10, 7)
        
        st.markdown("### Additional Information")
        family_history = st.selectbox("Family History of Disease", ["No", "Yes"])
        stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
        mental_health_score = st.slider("Mental Health Score (1-10)", 1, 10, 7)
        
        submitted = st.form_submit_button("Predict Life Expectancy", type="primary")
    
    # 메인 컨텐츠
    if submitted:
        # 입력 데이터 변환
        input_data = {
            'age': age,
            'gender': gender,
            'bmi': bmi,
            'waist_size': waist_size,
            'smoking_status': ["Never", "Former", "Current"].index(smoking_status),
            'drinks_per_week': drinks_per_week,
            'weekly_activity_minutes': weekly_activity_minutes,
            'sleep_quality_score': sleep_quality_score,
            'family_history': 1 if family_history == "Yes" else 0,
            'stress_level': stress_level,
            'mental_health_score': mental_health_score
        }
        
        # 예측
        try:
            result = model.predict_life_expectancy(input_data)
            
            # 결과 표시
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Expected Life Expectancy", f"{result['life_expectancy']:.1f} years")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Confidence Level", f"{result['confidence']:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                prediction_method = result.get('prediction_method', 'Hybrid System')
                st.metric("Prediction Method", prediction_method)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # 건강 권장사항
            st.markdown('<h2 class="sub-header">💡 Health Improvement Recommendations</h2>', unsafe_allow_html=True)
            
            recommendations = result.get('recommendations', [])
            if not recommendations:
                recommendations = [
                    "🔸 Maintain a balanced diet and regular exercise routine.",
                    "🔸 Get adequate sleep (7-9 hours per night).",
                    "🔸 Manage stress through relaxation techniques.",
                    "🔸 Avoid smoking and limit alcohol consumption.",
                    "🔸 Regular health check-ups are recommended."
                ]
            
            for rec in recommendations:
                st.markdown(f'<div class="recommendation-card">{rec}</div>', unsafe_allow_html=True)
            
            # 위험도 분석
            st.markdown('<h2 class="sub-header">📈 Health Risk Analysis</h2>', unsafe_allow_html=True)
            
            risk_factors = result.get('risk_factors', {})
            if risk_factors:
                # 위험도 차트
                categories = list(risk_factors.keys())
                values = list(risk_factors.values())
                
                fig = px.bar(
                    x=categories,
                    y=values,
                    title='Health Risk Factors Analysis',
                    labels={'x': 'Risk Factors', 'y': 'Risk Level (%)'},
                    color=values,
                    color_continuous_scale='RdYlGn_r'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # 수명 게이지 차트
            fig2 = go.Figure()
            fig2.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=result['life_expectancy'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Expected Life Expectancy"},
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
            st.error(f"Prediction error occurred: {e}")
    
    else:
        # 초기 화면
        st.markdown("""
        ## 🎯 Project Overview
        
        This system analyzes individual health habits and socioeconomic factors to predict expected life expectancy 
        and provides personalized recommendations for healthy living.
        
        ### Key Features:
        - 📊 **Life Expectancy Prediction**: Deep learning-based accurate life expectancy prediction
        - 💡 **Health Recommendations**: Personalized health improvement suggestions
        - 📈 **Visualization**: Intuitive result analysis and visualization
        - 🧠 **AI Technology**: Advanced deep learning with research-based weights
        
        ### How to Use:
        1. Enter your health information in the left sidebar
        2. Click "Predict Life Expectancy" button
        3. Review prediction results and health recommendations
        
        ---
        
        **Model Performance**: 91.8% Accuracy | Average Error ±1.64 years
        **Technology**: Deep Learning + Research-based Weights (20 papers)
        """)
        
        # 시스템 정보
        st.markdown("### 🔬 System Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Deep Learning Models:**
            - Stress/Mental Health Model
            - Physical Activity Model
            - Genetic Risk Model
            - Life Expectancy Model
            """)
        
        with col2:
            st.markdown("""
            **Research-based Weights:**
            - Smoking: 8 papers
            - BMI & Waist: 2 papers
            - Alcohol: 2 papers
            - Sleep: 5 papers
            - Physical Activity: 3 papers
            """)

if __name__ == "__main__":
    main()
