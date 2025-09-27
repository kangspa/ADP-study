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
- **시각화 옵션**:
    - 축 레이블, 제목, 범례 설정
    - 여러 그래프 겹쳐 그리기, 서브플롯(subplot) 활용

## Part 3: 통계 분석 (Statistical Analysis)

### 3.1. 기술 통계 및 확률 분포
- **기술 통계**: 중심 극한 정리, 표준 오차 등
- **확률 분포**: 이항 분포, 포아송 분포, 정규 분포 등

### 3.2. 추론 통계 (가설 검정)
- **기본 개념**: 귀무가설/대립가설, 유의수준(α), p-value, 검정통계량
- **정규성 검정**: `Shapiro-Wilk` test, `Kolmogorov-Smirnov` test
- **모수 검정**:
    - **T-검정 (T-test)**:
        - `단일표본 T-검정` (One-sample t-test)
        - `독립표본 T-검정` (Independent two-sample t-test) - 등분산성 검정(`Levene` test) 선행
        - `대응표본 T-검정` (Paired t-test)
    - **Z-검정**: 모집단 분산을 알 때 사용
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
- **상관 분석**:
    - `Pearson`, `Spearman`, `Kendall` 상관계수
    - 편상관 분석
- **회귀 분석**:
    - **선형 회귀**: `statsmodels.OLS`, `sklearn.LinearRegression`
    - **회귀 진단**:
        - 선형성 (잔차 플롯)
        - 잔차의 정규성 (Q-Q Plot, 정규성 검정)
        - 잔차의 등분산성
        - 잔차의 독립성 (Durbin-Watson 통계량)
    - **다중공선성**: 분산 팽창 요인(VIF) 확인 및 해결
    - **로지스틱 회귀**: 이진/다중 분류 문제에 사용
    - **고급 회귀**: 분위수 회귀, 베이지안 회귀

## Part 4: 머신러닝 모델링 (Machine Learning Modeling)

### 4.1. 모델링 준비
- **데이터 분할**: `train_test_split` (학습/검증/테스트 데이터)
- **교차 검증 (Cross-Validation)**: K-Fold, Stratified K-Fold

### 4.2. 지도 학습 (Supervised Learning)
- **회귀 모델**:
    - `LinearRegression` (선형 회귀)
    - `Ridge`, `Lasso`, `ElasticNet` (규제가 있는 선형 모델)
    - `DecisionTreeRegressor` (결정 트리)
    - `SVR` (서포트 벡터 회귀)
- **분류 모델**:
    - `LogisticRegression` (로지스틱 회귀)
    - `KNeighborsClassifier` (K-최근접 이웃)
    - `SVC` (서포트 벡터 머신)
    - `DecisionTreeClassifier` (결정 트리)
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
- **군집 평가지표**: 실루엣 계수 (Silhouette score)

### 4.5. 모델 최적화
- **하이퍼파라미터 튜닝**:
    - `GridSearchCV`: 격자 탐색
    - `RandomizedSearchCV`: 임의 탐색
    - 베이지안 최적화
- **변수 중요도**: 모델 결과 해석 및 피처 선택에 활용

## Part 5: 고급 분석 (Advanced Analytics)

### 5.1. 시계열 분석 (Time Series Analysis)
- **시계열 데이터 특성**: 정상성(Stationarity), 자기상관(Autocorrelation)
- **분해 및 진단**: ACF, PACF 플롯
- **시계열 모델**:
    - `AR`, `MA`, `ARMA`, `ARIMA`
    - `SARIMA` (계절성 포함)
    - `Prophet`, `LSTM` (딥러닝 기반)

### 5.2. 텍스트 마이닝 (Text Mining)
- **텍스트 전처리**: 토큰화, 불용어 제거, 형태소 분석, 표제어 추출/어간 추출
- **텍스트 벡터화**: `CountVectorizer`, `TfidfVectorizer`
- **분석 기법**: 감성 분석, 토픽 모델링(LDA)

### 5.3. 생존 분석 (Survival Analysis)
- `Kaplan-Meier` 생존 곡선
- `Log-rank` test

### 5.4. 최적화 (Optimization)
- 선형 계획법 (Linear Programming)