# 수명 예측 AI 시스템 (Life Expectancy Prediction)

## 프로젝트 개요
SW창의 경진대회 AI응용 부문 참가 프로젝트

### 해결하고자 하는 문제
- 달고 짜고 맵고, 자극적인 음식에 빠진 현대인들
- 웰니스에 대한 관심 증가하지만 넘쳐나는 정보 속에서 올바른 정보 획득의 어려움
- 단기간보다는 장기간의 생활 습관의 중요성 인식 부족

### 해결 방안
- 개인의 건강 습관을 입력받아 예상 수명을 예측하는 AI 시스템 개발
- 예측 결과를 바탕으로 개인화된 건강 습관 개선 방안 제시
- 지속 가능한 건강 습관 형성을 위한 동기 부여

### 개발 범위
- 사용자의 건강 습관 데이터 입력 인터페이스 (CLI + 웹앱)
- 딥러닝 기반 수명 예측 모델 (4개 신경망)
- 연구 논문 기반 가중치 시스템 (20개 논문)
- 개인화된 건강 습관 개선 권장사항 제공
- 시각화를 통한 결과 분석 및 제시

## 프로젝트 구조
```
life_expectancy_prediction/
├── data/                   # 데이터 파일들
├── models/                 # 훈련된 모델들
├── src/                    # 소스 코드
│   ├── data_processing/    # 데이터 전처리
│   ├── model/             # 모델 정의 및 훈련
│   ├── evaluation/        # 모델 평가
│   └── visualization/     # 시각화
├── notebooks/             # Jupyter 노트북
├── web_app/               # 웹 애플리케이션
├── requirements.txt       # 필요한 패키지들
└── README.md
```

## 사용 데이터셋
1. **Disease Risk from Daily Habits Dataset**
   - 100,000명의 개인 라이프스타일 및 생체 인식 정보
   - 습관, 건강 지표, 인구 통계 및 심리적 지표 포함

2. **Life Expectancy Analysis Dataset**
   - 국가별 기대 수명 데이터
   - 다양한 건강 지표와 사회경제적 요인 포함

3. **Disease Risk Prediction Dataset**
   - 4,000명의 환자 건강 프로필 시뮬레이션
   - 만성 질환 발병 가능성 예측 데이터

## 기술 스택
- **Python**: 주요 프로그래밍 언어
- **TensorFlow/Keras**: 딥러닝 모델 (4개 신경망)
- **Scikit-learn**: 보조 머신러닝 모델 (앙상블용)
- **Pandas, NumPy**: 데이터 처리
- **Matplotlib, Seaborn**: 데이터 시각화
- **Streamlit**: 웹 애플리케이션
- **Jupyter**: 데이터 분석 및 모델 개발

## 설치 및 실행
```bash
# 필요한 패키지 설치
pip install -r requirements.txt

# 웹 애플리케이션 실행
cd web_app
streamlit run app.py
```

## 개발 일정

