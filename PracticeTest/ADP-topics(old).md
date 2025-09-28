# ADP 실기 시험 대비 학습 주제

## Part 1: 데이터 기초 및 전처리 (Data Basics & Preprocessing)

### 1.1. Python 기초 (Python Basics)
- **개발 환경**: Jupyter Notebook 사용법 및 단축키
- **기본 문법**: Python 기본 자료구조(List, Dict, Set, Tuple), 제어문, 함수, 클래스
- **핵심 라이브러리**:
    - **Numpy**: 배열 생성 및 조작, 벡터 연산, 브로드캐스팅
    - **Pandas**: Series, DataFrame 생성 및 조작, 인덱싱, 정렬, 그룹화(groupby)
    - **Scipy**: 기초 통계 함수 활용

### 1.2. 데이터 핸들링 (Data Handling)
- **데이터 입출력**: `pd.read_csv`, `pd.read_excel`, `df.to_csv` 등
- **데이터 구조 확인**: `df.head()`, `df.info()`, `df.describe()`, `df.shape`, `df.isnull().sum()`
- **데이터 조작**:
    - 행/열 선택, 추가, 삭제
    - 데이터 타입 변경 (`astype`)
    - 조건 기반 필터링
    - 함수 적용 (`apply`, `map`)
- **데이터 결합**:
    - `pd.concat` (행/열 기준 결합)
    - `pd.merge` (SQL-style join)

### 1.3. 데이터 정제 (Data Cleansing)
- **결측치 처리**:
    - 확인: `isnull()`, `isna()`
    - 제거: `dropna()`
    - 대치: `fillna()` (평균, 중앙값, 최빈값 등)
    - 보간: `interpolate()`
- **이상치 탐지 및 처리**:
    - Box Plot, Scatter Plot을 이용한 시각적 탐지
    - Z-score, IQR(Interquartile Range)을 이용한 탐지
    - 제거 또는 적절한 값으로 변환 (Capping)

### 1.4. 피처 엔지니어링 (Feature Engineering)
- **변수 변환**:
    - 로그 변환, 제곱근 변환 (데이터 분포 왜도 개선)
    - **스케일링 (Scaling)**:
        - `StandardScaler`: 표준화 (평균 0, 분산 1)
        - `MinMaxScaler`: 정규화 (0과 1 사이의 값으로 조정)
        - `RobustScaler`: 중앙값과 IQR 사용 (이상치에 강건)
- **범주형 변수 인코딩**:
    - `LabelEncoder`: 순서형 변수에 적용
    - `OneHotEncoder`: 명목형 변수에 적용
    - `pd.get_dummies`: Pandas를 이용한 원-핫 인코딩
- **파생 변수 생성**: 기존 변수를 조합하여 새로운 의미를 갖는 변수 생성
- **차원 축소**:
    - **주성분 분석 (PCA)**: 고차원 데이터를 저차원으로 변환 (설명력 유지)
    - t-SNE (t-Distributed Stochastic Neighbor Embedding): 시각화를 위한 차원 축소
- **클래스 불균형 처리**:
    - **언더샘플링 (Undersampling)**: 다수 클래스 데이터 제거
    - **오버샘플링 (Oversampling)**: 소수 클래스 데이터 증식 (e.g., `SMOTE`)
    - **Class Weight**: 모델 학습 시 소수 클래스에 가중치 부여

## Part 2: 탐색적 데이터 분석 및 시각화 (EDA & Visualization)

### 2.1. 탐색적 데이터 분석 (EDA)
- **단변량 분석**:
    - **수치형**: 기초 통계량(평균, 중앙값, 분산 등), 분포 시각화(히스토그램, 커널 밀도 추정)
    - **범주형**: 빈도 분석(value_counts), 막대 그래프
- **이변량/다변량 분석**:
    - **수치형 vs 수치형**: 산점도(Scatter Plot), 상관계수(Correlation)
    - **범주형 vs 수치형**: 그룹별 통계량, Box Plot, Violin Plot
    - **범주형 vs 범주형**: 교차표(Crosstab), 누적/그룹 막대 그래프

### 2.2. 데이터 시각화 (Data Visualization)
- **시각화 라이브러리**:
    - **Matplotlib**: 기본 시각화 도구, 상세한 커스터마이징 가능
    - **Seaborn**: 통계 기반의 미려한 시각화 제공
- **주요 차트 유형**:
    - Line Plot, Bar Plot, Scatter Plot, Histogram, Box Plot, Heatmap, Pair Plot
    - 분포 시각화, 변화 시각화, 잔차 시각화
- **시각화 옵션**:
    - 축 레이블, 제목, 범례 설정
    - 여러 그래프 겹쳐 그리기, 서브플롯(subplot) 활용

## Part 3: 통계 분석 (Statistical Analysis)

## 3.0. 확률 계산
1. 확률의 기본 원리
    - 표본 공간과 사건: 확률을 정의하는 기본 개념입니다.
    - 확률의 덧셈정리와 곱셈정리: 여러 사건의 확률을 계산하는 핵심 규칙입니다.
    - 순열과 조합: 경우의 수를 계산하여 확률을 구하는 데 필수적입니다.

2. 확률 변수(Random Variable)의 기초
    - 확률 변수의 정의: 이산형(Discrete)과 연속형(Continuous)으로 구분됩니다.
    - 확률 질량 함수(PMF) 및 확률 밀도 함수(PDF): 확률 분포를 수학적으로 표현하는 방법입니다.
    - 누적 분포 함수(CDF): 특정 값보다 작거나 같을 확률을 나타냅니다.
    - 기댓값(Expected Value)과 분산(Variance): 확률 변수의 중심 경향성과 변동성을 측정하는 지표입니다.

3. 기타 주요 개념
    - 베이즈 정리 및 추론: 사전 정보를 바탕으로 사후 확률을 추론하는 방법입니다.
    - 조건부 확률 연산: 특정 사건이 일어났다는 조건 하에 다른 사건이 일어날 확률을 계산합니다.
    - 표본 크기 산출 공식: 신뢰수준, 오차한계 등을 고려하여 적절한 표본의 크기를 결정합니다.
    - 평균 변화율 계산: 기하 평균, 로그 변화율 등을 사용하여 평균적인 변화의 비율을 계산합니다.

### 3.1. 기술 통계 및 확률 분포
- **기술 통계**: 중심 극한 정리, 표준 오차 등
- **확률 분포**: 이항 분포, 포아송 분포, 정규 분포 등
    - 이항 확률 분포에서 기댓값
    - 이산확률분포, 연속확률분포

### 3.2. 추론 통계 (가설 검정)
- **기본 개념**: 귀무가설/대립가설, 유의수준(α), p-value, 검정통계량
- **정규성 검정**: `Shapiro-Wilk` test, `Kolmogorov-Smirnov` test
- 등분산성 검정(`Levene` test)
- **모수 검정**:
    - **T-검정 (T-test)**:
        - `단일표본 T-검정` (One-sample t-test) + 정규성 검정 함께 확인
        - `독립표본 T-검정` (Independent two-sample t-test) - 등분산성 검정(`Levene` test) 선행
        - `대응표본 T-검정` (Paired t-test)
    - **Z-검정**: 모집단 분산을 알 때 사용
    - **F-Test**
    - **분산분석 (ANOVA)**: 셋 이상 집단의 평균 비교
        - `일원배치 분산분석` (One-way ANOVA)
        - `이원배치 분산분석` (Two-way ANOVA)
        - **사후분석 (Post-hoc)**: `Tukey's HSD`, `Bonferroni` 등
- **비모수 검정**:
    - `Wilcoxon signed-rank test`, `Mann-Whitney U test`, `Kruskal-Wallis H test`
- **범주형 자료 분석**:
    - **카이제곱 검정 (Chi-squared Test)**:
        - `적합도 검정` (Goodness of fit)
        - `독립성 검정` (Test of independence)
        - `동질성 검정` (Test of homogeneity)
    - **비율 검정**: 단일/두 집단 비율 비교
    - `McNemar test`, `Cochran's Q test`

### 3.3. 상관/회귀 분석
- 최소제곱법
- **상관 분석**:
    - `Pearson`, `Spearman`, `Kendall` 상관계수
    - 편상관 분석
    - 상관계수 결정/검정
- **회귀 분석**:
    - **선형 회귀**
        - `statsmodels.OLS`
        - `sklearn.LinearRegression`
        - 강건 회귀 (Robust Regression, RLM): 데이터에 이상치(outlier)가 있을 때, 이상치의 영향을 덜 받도록 만들어진 회귀 모델입니다.
    - **일반화 선형 모델**
        - 푸아송 회귀 (Poisson Regression): 종속 변수가 특정 기간 동안의 사건 발생 횟수와 같은 카운트(count) 데이터일 때 사용합니다.
        - 감마 회귀 (Gamma Regression): 종속 변수가 양수 값을 가지며 분포가 오른쪽으로 치우쳐져 있을 때 사용합니다.
        - 음이항 회귀 (Negative Binomial Regression): 푸아송 회귀와 유사하지만, 분산이 평균보다 큰 '과대산포' 현상이 나타나는 카운트 데이터에 사용합니다.
    - **다항 회귀**: `Polynomial Regression`
    - **고급 회귀**
        - 분위수 회귀 (Quantile Regression): 일반 회귀가 종속 변수의 '평균'을 예측하는 것과 달리, 특정 분위수(예: 중앙값, 25% 지점 등)를 예측하여 변수 간의 관계를 더 다각적으로 분석합니다.
        - 베이지안 회귀
    - **회귀 진단**:
        - 선형성 (잔차 플롯)
        - 잔차의 정규성 (Q-Q Plot, 정규성 검정)
        - 잔차의 등분산성
        - 잔차의 독립성 (Durbin-Watson 통계량)
    - **다중공선성**: 분산 팽창 요인(VIF) 확인 및 해결, Cor, VIF, PCA
    - **로지스틱 회귀**: 이진/다중 분류 문제에 사용

### 3.4. 시계열 데이터 분석
1. 시계열 데이터의 이해
    - 시계열 데이터의 정의 및 특징: 시간의 흐름에 따라 관측된 데이터
    - 시계열 데이터의 구성 요소:
        - 추세 (Trend): 데이터가 장기적으로 증가하거나 감소하는 경향
        - 계절성 (Seasonality): 특정 기간(요일, 월, 분기 등)마다 반복되는 패턴
        - 주기 (Cycle): 계절성보다 주기가 길고 불규칙한 반복 패턴
        - 불규칙 요소 (Irregular / Residual): 위 세 가지 요소로 설명되지 않는 임의의 변동

2. 정상성 (Stationarity)
    - 정상성의 개념: 시간의 추이와 관계없이 평균, 분산, 공분산이 일정한 시계열
    - 정상성의 중요성: 안정적인 시계열 모델링의 기본 가정
    - 정상성 확인 방법:
        - 시각적 확인: 시계열 그래프를 통해 추세나 분산의 변화 확인
        - 통계적 검정: 단위근 검정 (Unit Root Test), 예: ADF (Augmented Dickey-Fuller) 검정
    - 비정상 시계열을 정상 시계열로 변환:
        - 차분 (Differencing): 현 시점 데이터에서 이전 시점 데이터를 빼는 방법
        - 로그 변환 (Log Transformation): 분산이 일정하지 않을 때 사용

3. 자기상관성 (Autocorrelation)
    - 자기상관의 개념: 현재 시점의 데이터가 과거 시점의 데이터와 얼마나 관련이 있는지
    - 자기상관 함수 (ACF): 시차(lag)에 따른 자기상관계수를 나타내는 함수
    - 부분자기상관 함수 (PACF): 다른 시점의 데이터 영향을 제외한 순수한 자기상관계수를 나타내는 함수
    - ACF/PACF 플롯 해석: 시계열 모델(AR, MA)의 차수를 결정하는 데 활용

4. 기초 시계열 모델
    - 이동평균 (Moving Average): 특정 기간의 데이터 평균으로 미래를 예측
    - 지수평활법 (Exponential Smoothing): 최근 데이터에 더 큰 가중치를 주어 미래를 예측
        - 단순 지수평활 (Simple Exponential Smoothing)
        - 홀트 선형 추세 모형 (Holt's Linear Trend Model)
        - 홀트-윈터스 계절 모형 (Holt-Winters' Seasonal Model)

## Part 4: 머신러닝 모델링 (Machine Learning Modeling)

### 4.1. 모델링 준비
- **데이터 분할**: `train_test_split` (학습/검증/테스트 데이터)
- **교차 검증 (Cross-Validation)**: K-Fold, Stratified K-Fold
- **데이터셋 구축**
    - `pytorch`의 `dataset`, `dataloader` 사용한 구축 방법
    - `keras`를 사용한 `dataset`, `dataloader` 구축 방법

### 4.2. 지도 학습 (Supervised Learning)
- **회귀 모델**:
    - `LinearRegression` (선형 회귀)
    - `Ridge`, `Lasso`, `ElasticNet` (규제가 있는 선형 모델)
    - `DecisionTreeRegressor` (결정 트리)
        - Tree 시각화 (`export_graphviz`)
        - 이미지화 (`from IPython.display import Image`)
    - `SVR` (서포트 벡터 회귀)
- **분류 모델**:
    - `LogisticRegression` (로지스틱 회귀)
    - `KNeighborsClassifier` (K-최근접 이웃)
    - `SVC` (서포트 벡터 머신)
    - `DecisionTreeClassifier` (결정 트리)
        - Tree 시각화 (`export_graphviz`)
        - 이미지화 (`from IPython.display import Image`)
- **앙상블 모델**:
    - **보팅 (Voting)**: Hard/Soft Voting
    - **배깅 (Bagging)**: `RandomForest`
    - **부스팅 (Boosting)**: `AdaBoost`, `GradientBoosting`, `XGBoost`, `LightGBM`
    - **스태킹 (Stacking)**

### 4.3. 비지도 학습 (Unsupervised Learning)
- **군집 분석 (Clustering)**:
    - `K-Means` (분할 군집)
    - `Hierarchical Clustering` (계층적 군집)
    - `DBSCAN` (밀도 기반 군집)
- **연관 규칙 분석**: `Apriori` (지지도, 신뢰도, 향상도)
- **이상 탐지**: `Isolation Forest`, `Local Outlier Factor (LOF)`

### 4.4. 모델 성능 평가
- **회귀 평가지표**:
    - `MSE`, `RMSE`, `MAE`, `MAPE`
    - `R²` (결정계수), `Adjusted R²` (조정된 결정계수)
- **분류 평가지표**:
    - **혼동 행렬 (Confusion Matrix)**
    - `정확도 (Accuracy)`
    - `정밀도 (Precision)`
    - `재현율 (Recall)`
    - `F1-Score`
    - `ROC Curve`와 `AUC`
    - classification_report
- **군집 평가지표**: 실루엣 계수 (Silhouette score)

### 4.5. 모델 최적화
- **하이퍼파라미터 튜닝**:
    - `GridSearchCV`: 격자 탐색
    - `RandomizedSearchCV`: 임의 탐색
    - 베이지안 최적화
- **변수 중요도**: 모델 결과 해석 및 피처 선택에 활용
    - RandomForest 등을 활용한 변수 중요도 확인 방법 추가
- **변수 선택법**:후진 제거법, 전진 선택법 등

## Part 5: 고급 분석 (Advanced Analytics)

### 5.1. 시계열 분석 (Time Series Analysis)
- **시계열 데이터 전처리**
- **시계열 데이터 특성**: 정상성(Stationarity), 자기상관(Autocorrelation)
- **분해 및 진단**: ACF, PACF 플롯
- **시계열 모델**:
    - `AR`, `MA`, `ARMA`, `ARIMA`
    - `SARIMA` (계절성 포함)
    - `VAR` (Vector Autoregression)
    - 상태 공간 모델 (State Space Models)
    - `Prophet`, `LSTM` (딥러닝 기반)

### 5.2 이미지 데이터 (Image Data)
- **이미지 전처리**
- **이미지 데이터 분석**
- **이미지 분석 모델들 활용**: `pytorch`, `keras`에서 제공하는 모델들 활용 방법

### 5.3. 텍스트 마이닝 (Text Mining)
- **텍스트 전처리**: 토큰화, 불용어 제거, 형태소 분석, 표제어 추출/어간 추출
- **텍스트 벡터화**: `CountVectorizer`, `TfidfVectorizer`
- **분석 기법**: 감성 분석, 토픽 모델링(LDA)
- **텍스트 분석 모델들 활용**: `pytorch`, `keras`에서 제공하는 모델들 활용 방법

### 5.4. 생존 분석 (Survival Analysis)
- `Kaplan-Meier` 생존 곡선
- `Log-rank` test

### 5.5. 최적화 (Optimization)
- 선형 계획법 (Linear Programming)
- 정수 선형 계획법

### 5.6 딥러닝 모델링
- Simple DL
- `pytorch`를 활용한 딥러닝 모델 구축
    - 일반적인 딥러닝 모델 구축
    - CNN 모델 구축
    - RNN 모델 구축
    - LSTM 모델 구축
    - 각종 규제(L1, L2 등)나 dropout 추가
    - 마지막 레이어 처리 방식 (회귀, 이중 분류, 다중 분류)
    - 손실 함수 종류 및 활용 방법
    - 활성화 함수 종류 및 활용 방법
    - 레이어를 연결해서 구축하는 다양한 방법 추가 작성 (단일 구축 및 Sequential, Functional 활용 등)
    - 모델 구조 시각화
- `keras`를 활용한 딥러닝 모델 구축
    - 일반적인 딥러닝 모델 구축
    - CNN 모델 구축
    - RNN 모델 구축
    - LSTM 모델 구축
    - 각종 규제(L1, L2 등)나 dropout 추가
    - 마지막 레이어 처리 방식 (회귀, 이중 분류, 다중 분류)
    - 손실 함수 종류 및 활용 방법
    - 활성화 함수 종류 및 활용 방법
    - 레이어를 연결해서 구축하는 다양한 방법 추가 작성 (단일 구축 및 Sequential, Functional 활용 등)
    - 모델 구조 시각화

### 5.7 학습 및 평가 코드 작성
- `pytorch`의 예제 코드
    - 학습 진행하는 코드 작성
        - 가중치 업데이트 등
    - 성능 평가 코드 작성
    - 추론 코드 작성
    - 모델 저장 방법
    - 학습 과정 시각화
        - loss나 정확도 등
- `keras`의 예제 코드
    - 학습 진행하는 코드 작성
        - 가중치 업데이트 등
    - 성능 평가 코드 작성
    - 추론 코드 작성
    - 모델 저장 방법
    - 학습 과정 시각화
        - loss나 정확도 등