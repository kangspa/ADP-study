# ADP 27회 실기 문제 풀이 by Gemini

본 문서는 "제27회.md" 파일에 제시된 문제들에 대한 분석 방법론과 풀이 과정을 상세히 설명합니다. 각 문제에 대해 가능한 여러 분석 방법을 소개하고, 실제 문제에 적용하는 과정을 코드 예제와 함께 제시합니다.

---

## 1번 문제: EDA 및 전처리 필요성

### 1-1. EDA 데이터 탐색

- **분석 방법**
    - **데이터 기본 정보 확인**: `df.info()`, `df.describe()`를 통해 데이터 타입, 결측치 유무, 통계적 분포를 확인합니다.
    - **타겟 변수 분포 확인**: `df['Class'].value_counts()`를 통해 사기(1)와 정상(0) 거래의 비율을 확인합니다. 이를 통해 데이터의 불균형 정도를 파악합니다.
    - **주요 변수 분포 시각화**: `Time`과 `Amount` 변수의 분포를 히스토그램으로 시각화하여 데이터의 스케일과 분포 형태를 확인합니다.

- **현재 문제에 관한 풀이 방법**
    ```python
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # 데이터 로드 가정
    # df = pd.read_csv('creditcard.csv')

    # # 데이터 기본 정보
    # print(df.info())
    # print(df.describe())

    # # 클래스 불균형 확인
    # print("\nClass Distribution:")
    # print(df['Class'].value_counts(normalize=True))

    # # Time, Amount 분포 시각화
    # fig, ax = plt.subplots(1, 2, figsize=(18, 4))
    # sns.histplot(df['Amount'], ax=ax[0], color='r', bins=50)
    # ax[0].set_title('Distribution of Transaction Amount')
    # sns.histplot(df['Time'], ax=ax[1], color='b', bins=50)
    # ax[1].set_title('Distribution of Transaction Time')
    # plt.show()
    ```
    - **탐색 결과**: 데이터에는 결측치가 없으며, `Class` 변수는 0(정상)이 99.8% 이상을 차지하는 극심한 불균형 상태임을 알 수 있습니다. `Amount`와 `Time` 변수는 다른 V 변수들과 스케일 차이가 매우 큽니다.

### 1-2. 변수 간 상관관계 시각화 및 전처리 필요성 설명

- **상관관계 시각화**: `heatmap`을 사용하여 변수 간의 상관관계를 시각화합니다.
- **전처리 필요성 설명**
    1.  **피처 스케일링 (Feature Scaling)**: `Amount`와 `Time` 변수는 다른 V 변수들(PCA로 변환된 값)과 값의 범위(scale)가 크게 다릅니다. 거리 기반 알고리즘이나 경사 하강법을 사용하는 모델(e.g., Logistic Regression, SVM)은 변수 스케일에 민감하므로, 모델이 특정 변수에만 과도하게 영향을 받는 것을 방지하기 위해 `StandardScaler`나 `RobustScaler` 등으로 스케일링하여 모든 변수가 비슷한 범위를 갖도록 만들어야 합니다.
    2.  **데이터 불균형 처리 (Imbalanced Data Handling)**: 1-1에서 확인했듯이, 사기 거래(Class=1) 데이터가 극도로 적습니다. 이 상태로 모델을 학습하면 모델은 대부분의 예측을 정상(0)으로만 해도 높은 정확도를 얻게 되어, 정작 중요한 사기 거래를 탐지하지 못하는 문제가 발생합니다. 따라서, 오버샘플링(e.g., SMOTE)이나 언더샘플링 기법을 적용하여 학습 데이터의 클래스 비율을 맞춰주는 전처리가 필수적입니다.

    ```python
    # # 상관관계 히트맵
    # plt.figure(figsize=(12, 10))
    # sns.heatmap(df.corr(), cmap='coolwarm_r', annot=False)
    # plt.title('Correlation Matrix')
    # plt.show()
    ```

---

## 2번 문제: 차원 축소

### 2-1. 차원축소 방법 2가지 이상 비교

1.  **주성분 분석 (PCA, Principal Component Analysis)**
    - **설명**: 데이터의 분산을 가장 잘 설명하는 새로운 축(주성분)을 찾아, 기존의 고차원 데이터를 저차원의 새로운 데이터로 변환하는 선형 차원 축소 기법입니다. 변수 간의 상관관계를 제거하고 데이터의 노이즈를 줄이는 데 효과적입니다.
    - **장점**: 계산 속도가 빠르고, 변환된 각 주성분이 설명하는 분산량을 통해 기여도를 파악하기 용이합니다.
    - **단점**: 데이터의 비선형 구조를 잘 파악하지 못할 수 있습니다.

2.  **t-SNE (t-Distributed Stochastic Neighbor Embedding)**
    - **설명**: 고차원 공간에서 데이터 포인트 간의 유사도를 저차원 공간에서도 유지하도록 점을 배치하는 비선형 차원 축소 기법입니다. 주로 시각화 목적으로 사용됩니다.
    - **장점**: 복잡한 비선형 구조와 군집을 시각적으로 표현하는 데 매우 뛰어납니다.
    - **단점**: 계산 비용이 매우 높고, 학습 데이터에만 적용 가능하며 새로운 데이터를 변환하는 기능이 없습니다. 결과의 일관성이 부족하며, 주로 탐색적 분석 및 시각화에 한정적으로 사용됩니다.

### 2-2. 한 가지 방법 선택 및 수행

- **선택**: **PCA (주성분 분석)**
- **선택 이유**: 이 문제의 목표는 분류 모델의 성능을 높이기 위한 **전처리** 단계로서의 차원 축소입니다. t-SNE는 주로 시각화를 통한 데이터 탐색에 사용되며, 새로운 데이터에 적용하기 어렵다는 단점 때문에 전처리 단계에는 부적합합니다. 반면, PCA는 모델의 입력으로 사용할 저차원 데이터를 생성하는 데 효과적이며, 계산 효율성도 높아 대용량 데이터 처리에 적합합니다. V1~V17이 이미 PCA 결과물이지만, `Time`과 `Amount`를 포함한 전체 데이터셋에 PCA를 다시 적용하여 변수들을 완전히 새로운 축으로 재구성하고 차원을 축소할 수 있습니다.

- **수행**
    ```python
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    # # Time, Amount 스케일링
    # scaler = StandardScaler()
    # df[['Time', 'Amount']] = scaler.fit_transform(df[['Time', 'Amount']])

    # # PCA 수행 (설명된 분산이 95%가 되는 지점까지 차원 축소)
    # X = df.drop('Class', axis=1)
    # pca = PCA(n_components=0.95)
    # X_pca = pca.fit_transform(X)

    # print(f"Original number of features: {X.shape[1]}")
    # print(f"Reduced number of features: {X_pca.shape[1]}")
    ```

---

## 3번 문제: 불균형 데이터 처리 및 모델링

### 3-1. 오버샘플링과 언더샘플링 비교 및 선택

- **언더샘플링 (Undersampling)**
    - **장점**: 다수 클래스 데이터를 줄여 학습 데이터 크기를 감소시키므로, 학습 시간이 단축되고 저장 공간을 절약할 수 있습니다.
    - **단점**: 다수 클래스의 중요한 정보를 손실할 위험이 있어 모델의 전체적인 성능 저하를 유발할 수 있습니다.
- **오버샘플링 (Oversampling)**
    - **장점**: 정보 손실 없이 소수 클래스 데이터의 수를 늘려, 모델이 소수 클래스의 패턴을 더 잘 학습하도록 합니다. 일반적으로 언더샘플링보다 분류 성능이 더 좋습니다.
    - **단점**: 데이터 수가 늘어나 학습 시간이 길어지고, 단순 복제 시 과적합(overfitting)의 위험이 있습니다. (SMOTE는 이를 완화)

- **선택**: **오버샘플링 (SMOTE)**
    - **이유**: 사기 탐지에서는 정상 거래 데이터(다수 클래스)가 '정상'이 무엇인지 정의하는 중요한 정보 역할을 합니다. 언더샘플링으로 이 정보를 제거하면 모델의 일반화 성능이 저하될 수 있습니다. 반면, SMOTE와 같은 오버샘플링 기법은 정보 손실 없이 소수 클래스(사기)의 특징을 학습할 기회를 늘려주므로, 탐지 성능에 더 유리합니다.

### 3-2. 알고리즘 2가지 이상 비교 및 성능 측정

- **분석 방법**
    1.  데이터를 훈련/테스트 세트로 분리합니다.
    2.  **훈련 데이터에만** SMOTE를 적용하여 클래스 불균형을 해소합니다.
    3.  **로지스틱 회귀(Logistic Regression)**와 **랜덤 포레스트(Random Forest)** 두 모델을 학습시킵니다.
    4.  불균형 데이터에 적합한 평가지표인 **Precision, Recall, F1-score, AUPRC(Area Under Precision-Recall Curve)**를 사용하여 **원본 테스트 세트**에서 모델 성능을 비교합니다.

- **구현**
    ```python
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import SMOTE
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, average_precision_score

    # # y = df['Class'], X는 2-2에서 생성한 X_pca 또는 원본 X 사용
    # X_train, X_test, y_train, y_test = train_test_split(X, df['Class'], test_size=0.3, random_state=42, stratify=df['Class'])

    # # SMOTE 적용 (훈련 데이터에만)
    # smote = SMOTE(random_state=42)
    # X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    # # 모델 학습 및 평가
    # models = {
    #     'Logistic Regression': LogisticRegression(random_state=42),
    #     'Random Forest': RandomForestClassifier(random_state=42)
    # }

    # for name, model in models.items():
    #     model.fit(X_train_sm, y_train_sm)
    #     preds = model.predict(X_test)
    #     proba = model.predict_proba(X_test)[:, 1]
        
    #     print(f"--- {name} ---")
    #     print(classification_report(y_test, preds))
    #     print(f"AUPRC: {average_precision_score(y_test, proba):.4f}\n")
    ```

### 3-3. 모델 수행 결과 분석

- **결과 분석**: 일반적으로 랜덤 포레스트가 로지스틱 회귀보다 더 복잡한 패턴을 학습할 수 있으므로, 더 높은 Recall과 AUPRC 점수를 보일 가능성이 높습니다. 사기 탐지에서는 놓치는 사기 거래(FN)를 최소화하는 것이 중요하므로 **Recall**이 핵심 지표가 됩니다. 두 모델의 `classification_report`에서 Class 1의 Recall 점수를 비교하고, AUPRC 점수를 함께 고려하여 최종적으로 랜덤 포레스트가 더 우수한 모델이라고 결론 내릴 수 있습니다.

---

## 4번 문제: 이상탐지 모델

### 4-1. 이상탐지 모델 2가지 기술 및 장/단점 설명

1.  **Isolation Forest (고립 포레스트)**
    - **기술**: 데이터 포인트를 무작위로 분할하여 고립시키는 데 필요한 분할 횟수를 기반으로 이상 점수를 계산합니다. 이상치는 "소수이면서 특징이 뚜렷하기" 때문에 더 적은 분할로 쉽게 고립될 수 있다는 아이디어에 기반합니다.
    - **장점**: 대용량 데이터와 고차원 데이터에서 효율적으로 작동하며, 사전 스케일링이 필요 없습니다.
    - **단점**: 데이터셋 내 이상치의 비율(`contamination`)을 미리 지정해야 하는 점이 까다로울 수 있습니다.

2.  **Local Outlier Factor (LOF)**
    - **기술**: 각 데이터 포인트 주변의 지역적 밀도를 이웃과 비교하여 이상치를 탐지합니다. 주변보다 밀도가 현저히 낮은 데이터 포인트를 이상치로 판단합니다.
    - **장점**: 군집의 밀도가 다양한 데이터셋에서도 잘 작동합니다.
    - **단점**: 계산 복잡도가 높아 대용량 데이터에는 부적합하며, 전역적인 이상치(global outliers) 탐지에는 약할 수 있습니다.

### 4-2. 이상탐지 모델 구현 및 3번 모델과 비교

- **구현**: 비지도학습인 **Isolation Forest**를 2번에서 차원 축소한 데이터(`X_pca`)에 적용합니다. 이 모델은 `y` 라벨 없이 학습합니다.

    ```python
    from sklearn.ensemble import IsolationForest

    # # Isolation Forest 모델 학습 및 예측
    # # contamination은 원본 데이터의 사기 비율과 유사하게 설정
    # contamination_rate = df['Class'].value_counts(normalize=True)[1]
    # iso_forest = IsolationForest(contamination=contamination_rate, random_state=42)

    # # 비지도 학습이므로 전체 X 데이터로 학습 가능
    # preds_iso = iso_forest.fit_predict(X) 
    # # 예측 결과: 1은 정상, -1은 이상치. 비교를 위해 0과 1로 변환
    # preds_iso_mapped = [1 if p == -1 else 0 for p in preds_iso]

    # print("--- Isolation Forest ---")
    # # y 라벨과 비교하여 성능 평가
    # print(classification_report(df['Class'], preds_iso_mapped))
    # print(f"AUPRC: {average_precision_score(df['Class'], iso_forest.decision_function(X)):.4f}\n")
    ```
- **3번 모델(랜덤 포레스트)과 비교**: Isolation Forest는 비지도학습임에도 불구하고 어느 정도 사기를 탐지해낼 수 있습니다. 하지만, 실제 사기 거래 데이터를 학습한 3번의 랜덤 포레스트 모델(지도학습)이 일반적으로 훨씬 높은 Recall과 AUPRC 성능을 보입니다. 이는 지도학습이 '사기'라는 명확한 목표를 가지고 패턴을 학습하기 때문입니다.

### 4-3. 두 모델에 대한 데이터 분석가 관점 설명

- **3번 모델 (지도학습 - 랜덤 포레스트)**: 이 모델은 **"알려진 사기 유형을 탐지하는 모델"**입니다. 과거에 발생했던 사기 거래의 패턴을 명시적으로 학습하여, 이와 유사한 새로운 거래가 발생했을 때 이를 사기로 분류합니다. 이미 축적된 데이터가 충분하고 라벨링이 잘 되어 있을 때 매우 효과적입니다.

- **4-번 모델 (비지도학습 - Isolation Forest)**: 이 모델은 **"새롭고 알려지지 않은 이상 행위를 탐지하는 모델"**입니다. 정상 거래 데이터의 전반적인 분포를 학습한 후, 이 분포에서 크게 벗어나는 모든 거래를 '이상 신호'로 감지합니다. 이는 과거에 없었던 새로운 유형의 사기를 탐지할 잠재력을 가집니다. 따라서, 지도학습 모델을 보완하는 안전망 역할을 할 수 있습니다.

- **결론**: 실무에서는 두 접근법을 상호 보완적으로 사용합니다. 지도학습 모델로 대부분의 알려진 사기를 잡고, 비지도 이상탐지 모델로 알려지지 않은 패턴의 이상 거래를 모니터링하여 분석가에게 알려주는 시스템을 구축하는 것이 이상적입니다.

---

## 5번 문제: 연평균 상승률

- **분석 방법**: 평균 변화율을 계산할 때는 **기하평균**을 사용합니다.
- **풀이**: 2년간의 성장 인자는 각각 `150,000 / 100,000 = 1.5` 와 `250,000 / 150,000 ≈ 1.667` 입니다. 이 둘의 기하평균을 구한 후 1을 빼서 상승률로 변환합니다.
    - 연평균 상승 인자 = `(1.5 * (250/150))^(1/2) = sqrt(2.5) ≈ 1.5811`
    - 연평균 상승률 = `(1.5811 - 1) * 100 ≈ 58.11%`

    ```python
    from scipy.stats.mstats import gmean
    growth_factors = [150000/100000, 250000/150000]
    avg_growth_rate = (gmean(growth_factors) - 1) * 100
    print(f"연평균 상승률: {avg_growth_rate:.2f}%")
    ```

## 6번 문제: 신뢰구간 추정

- **분석 방법**: 모분산을 모르므로 **t-분포**를 사용하여 모평균의 신뢰구간을 추정합니다. 공식: $\bar{x} \pm t_{\alpha/2, n-1} \cdot \frac{s}{\sqrt{n}}$
- **풀이**:
    - $\bar{x}=15.5$, $s^2=3.2 \Rightarrow s=\sqrt{3.2}$, $n=12$, $df=11$
    - 신뢰수준 90% $\Rightarrow \alpha=0.1 \Rightarrow \alpha/2=0.05$. $t_{0.05, 11}$ 값을 구해야 합니다.

    ```python
    from scipy.stats import t
    import numpy as np
    n, x_bar, var = 12, 15.5, 3.2
    s = np.sqrt(var)
    df = n - 1
    alpha = 0.1

    t_critical = t.ppf(1 - alpha/2, df)
    margin_of_error = t_critical * (s / np.sqrt(n))
    
    ci_lower = x_bar - margin_of_error
    ci_upper = x_bar + margin_of_error
    print(f"90% 신뢰구간: [{ci_lower:.2f}, {ci_upper:.2f}]")
    ```

## 7번 문제: 대응표본 t-검정

- **7-1. 가설 설정**
    - **귀무가설(H0)**: 상류와 하류의 생물 다양성 점수 평균에 차이가 없다. ($\_D = 0$)
    - **대립가설(H1)**: 상류와 하류의 생물 다양성 점수 평균에 차이가 있다. ($\_D \neq 0$)
      (여기서 $\_D$는 (하류 점수 - 상류 점수)의 평균)

- **7-2. 가설 검증**
    - **분석 방법**: 동일한 강의 상류/하류 데이터는 서로 종속적이므로 **대응표본 t-검정(Paired t-test)**을 사용합니다. `scipy.stats.ttest_rel` 함수를 사용합니다.

    ```python
    from scipy.stats import ttest_rel
    # streams.csv 파일 로드 가정
    # streams_df = pd.read_csv('streams.csv') 
    # upstream = streams_df['upstream']
    # downstream = streams_df['downstream']

    # # 가상 데이터
    # upstream = [5.2, 4.8, 6.1, 5.5, 6.2, 7.1, 4.5, 5.6, 5.9, 6.3, 4.9, 5.8, 5.1, 6.0, 6.5, 5.4]
    # downstream = [6.1, 5.3, 6.6, 5.9, 6.9, 7.5, 4.6, 6.0, 6.2, 6.8, 5.2, 6.2, 5.5, 6.3, 7.0, 5.7]

    # t_stat, p_value = ttest_rel(downstream, upstream)

    # print(f"검정 통계량: {t_stat:.4f}")
    # print(f"유의확률(p-value): {p_value:.4f}")

    # if p_value < 0.05:
    #     print("연구가설 채택: 상류와 하류의 생물 다양성 점수 평균에 유의미한 차이가 있습니다.")
    # else:
    #     print("귀무가설 채택: 평균에 유의미한 차이가 없습니다.")
    ```

## 8번 문제: 분위수 회귀

- **8-1. 회귀 계수 계산**
    - **분석 방법**: **분위수 회귀(Quantile Regression)**는 최소제곱법(OLS)이 평균에 미치는 영향을 모델링하는 것과 달리, 설명 변수가 종속 변수의 특정 분위수(e.g., 중앙값)에 미치는 영향을 모델링합니다. `statsmodels` 라이브러리의 `quantreg`를 사용합니다.

    ```python
    import statsmodels.formula.api as smf
    # traffic.csv 파일 로드 가정
    # traffic_df = pd.read_csv('traffic.csv')
    # # 컬럼명 예시: traffic_volume, temp, rain_mm, wind_speed_ms

    # # 50백분위수(중앙값)에 대한 분위수 회귀 모델
    # model = smf.quantreg('traffic_volume ~ temp + rain_mm + wind_speed_ms', data=traffic_df)
    # result = model.fit(q=0.5)
    # print(result.summary())
    ```

- **8-2. 교통량 예측**
    - **풀이**: 학습된 모델의 `predict()` 메서드를 사용하여 새로운 데이터에 대한 교통량을 예측합니다.

    ```python
    # # 새로운 데이터 생성
    # new_data = pd.DataFrame({
    #     'temp': [15.5],
    #     'rain_mm': [16.5],
    #     'wind_speed_ms': [1.6]
    # })

    # # 교통량 예측
    # predicted_traffic = result.predict(new_data)
    # print(f"\n예측된 교통량: {predicted_traffic.iloc[0]:.2f}")
    ```

## 9번 문제: 이원배치 분산분석 (Two-Way ANOVA)

- **9-1. 가설 설정**
    - **귀무가설(H0)**: 호선과 월의 상호작용 효과는 없다. (즉, 월에 따른 승차 인원의 변화 패턴이 모든 호선에서 동일하다.)
    - **대립가설(H1)**: 호선과 월의 상호작용 효과가 있다.

- **9-2. 가설 검증**
    - **분석 방법**: 두 개의 범주형 변수(호선, 월)가 하나의 연속형 변수(승차 인원)에 미치는 영향을 분석하므로 **이원배치 분산분석(Two-Way ANOVA)**을 사용합니다. 상호작용 효과를 확인하는 것이 핵심입니다.

    ```python
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    # subway.csv 파일 로드 가정
    # subway_df = pd.read_csv('subway.csv')
    # # 컬럼명 예시: passengers, line, month

    # # 이원배치 분산분석 모델 (Type III 제곱합 사용)
    # formula = 'passengers ~ C(line) + C(month) + C(line):C(month)'
    # model = ols(formula, data=subway_df).fit()
    # anova_table = anova_lm(model, typ=3)

    # print(anova_table)

    # # 상호작용 효과의 p-value 확인
    # p_value_interaction = anova_table.loc['C(line):C(month)', 'PR(>F)']

    # if p_value_interaction < 0.05:
    #     print("\n연구가설 채택: 호선과 월의 상호작용 효과는 유의미합니다.")
    # else:
    #     print("\n귀무가설 채택: 상호작용 효과가 유의미하지 않습니다.")
    ```
