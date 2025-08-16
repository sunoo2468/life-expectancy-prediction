import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='mean')
        
    def load_data(self, file_path):
        """데이터 로드"""
        try:
            data = pd.read_csv(file_path)
            print(f"데이터 로드 완료: {data.shape}")
            return data
        except Exception as e:
            print(f"데이터 로드 오류: {e}")
            return None
    
    def basic_info(self, data):
        """데이터 기본 정보 출력"""
        print("=" * 50)
        print("데이터 기본 정보")
        print("=" * 50)
        print(f"데이터 크기: {data.shape}")
        print(f"컬럼 수: {data.shape[1]}")
        print(f"행 수: {data.shape[0]}")
        print("\n컬럼 목록:")
        for i, col in enumerate(data.columns, 1):
            print(f"{i:2d}. {col}")
        
        print("\n데이터 타입:")
        print(data.dtypes.value_counts())
        
        print("\n결측치 정보:")
        missing_data = data.isnull().sum()
        missing_percent = (missing_data / len(data)) * 100
        missing_info = pd.DataFrame({
            '결측치 수': missing_data,
            '결측치 비율(%)': missing_percent
        })
        print(missing_info[missing_info['결측치 수'] > 0])
        
        print("\n수치형 데이터 통계:")
        print(data.describe())
    
    def handle_missing_values(self, data, strategy='mean'):
        """결측치 처리"""
        print(f"\n결측치 처리 중... (전략: {strategy})")
        
        # 수치형 컬럼과 범주형 컬럼 분리
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        categorical_columns = data.select_dtypes(include=['object']).columns
        
        # 수치형 컬럼 결측치 처리
        if strategy == 'mean':
            data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())
        elif strategy == 'median':
            data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())
        
        # 범주형 컬럼 결측치 처리
        data[categorical_columns] = data[categorical_columns].fillna(data[categorical_columns].mode().iloc[0])
        
        print("결측치 처리 완료!")
        return data
    
    def encode_categorical(self, data):
        """범주형 변수 인코딩"""
        print("\n범주형 변수 인코딩 중...")
        
        categorical_columns = data.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                data[col] = self.label_encoders[col].fit_transform(data[col])
            else:
                data[col] = self.label_encoders[col].transform(data[col])
        
        print("범주형 변수 인코딩 완료!")
        return data
    
    def remove_outliers(self, data, columns, method='iqr'):
        """이상치 제거"""
        print(f"\n이상치 제거 중... (방법: {method})")
        
        original_shape = data.shape
        
        if method == 'iqr':
            for col in columns:
                if col in data.columns:
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
        
        print(f"이상치 제거 완료: {original_shape} -> {data.shape}")
        return data
    
    def create_correlation_plot(self, data, save_path=None):
        """상관관계 히트맵 생성"""
        plt.figure(figsize=(15, 12))
        correlation_matrix = data.corr()
        
        # 상관관계가 높은 변수들만 선택 (절댓값 0.1 이상)
        high_corr = correlation_matrix[abs(correlation_matrix) > 0.1]
        
        sns.heatmap(high_corr, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('변수 간 상관관계 히트맵', fontsize=16, pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_distribution_plots(self, data, columns, save_path=None):
        """분포도 생성"""
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = [axes]
        if n_cols == 1:
            axes = [[ax] for ax in axes]
        
        for i, col in enumerate(columns):
            row = i // n_cols
            col_idx = i % n_cols
            
            if col in data.columns:
                sns.histplot(data[col], kde=True, ax=axes[row][col_idx])
                axes[row][col_idx].set_title(f'{col} 분포')
                axes[row][col_idx].tick_params(axis='x', rotation=45)
        
        # 빈 서브플롯 숨기기
        for i in range(len(columns), n_rows * n_cols):
            row = i // n_cols
            col_idx = i % n_cols
            axes[row][col_idx].set_visible(False)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def prepare_features(self, data, target_column=None):
        """특성 준비"""
        print("\n특성 준비 중...")
        
        # 타겟 변수 분리
        if target_column and target_column in data.columns:
            X = data.drop(columns=[target_column])
            y = data[target_column]
            print(f"특성 수: {X.shape[1]}, 타겟 변수: {target_column}")
            return X, y
        else:
            print(f"특성 수: {data.shape[1]}")
            return data, None
    
    def scale_features(self, X_train, X_test=None):
        """특성 스케일링"""
        print("\n특성 스케일링 중...")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """데이터 분할"""
        print(f"\n데이터 분할 중... (테스트 비율: {test_size})")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if len(y.unique()) < 10 else None
        )
        
        print(f"훈련 데이터: {X_train.shape}")
        print(f"테스트 데이터: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test

