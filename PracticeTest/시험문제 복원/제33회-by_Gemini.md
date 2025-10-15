# ADP 33회 실기 문제 풀이 by Gemini

본 문서는 "제33회.md" 파일에 제시된 문제들에 대한 분석 방법론과 풀이 과정을 상세히 설명합니다. 각 문제에 대해 가능한 여러 분석 방법을 소개하고, 실제 문제에 적용하는 과정을 코드 예제와 함께 제시합니다.

---

## 1번 문제: 데이터 탐색 및 전처리 (간염 데이터)

### 1-1. 결측치 처리 방안 2가지 제시, 비교 및 선택

- **결측치 처리 방안**
    1.  **중앙값 대치 (Median Imputation)**: 결측치를 해당 변수의 중앙값으로 채우는 간단한 방법입니다. 이상치에 덜 민감하다는 장점이 있습니다.
    2.  **K-최근접 이웃 대치 (KNN Imputation)**: 결측치가 있는 샘플과 가장 유사한 K개의 다른 샘플(이웃)을 찾아, 그 이웃들의 평균값으로 결측치를 채우는 방법입니다. 변수 간의 복잡한 관계를 고려할 수 있어 더 정교한 대치가 가능합니다.

- **비교 및 선택**
    - **비교**: 두 방법으로 결측치를 처리한 후, 원본 데이터의 분포와 처리 후 데이터의 분포를 히스토그램 등으로 비교합니다. 중앙값 대치는 분포의 분산을 감소시켜 왜곡을 유발할 수 있는 반면, KNN 대치는 데이터의 원래 분포와 구조를 더 잘 보존하는 경향이 있습니다.
    - **선택**: **KNN Imputation**을 선택합니다.
    - **이유**: 의료 데이터에서 각 혈액 수치는 서로 유기적인 관계를 맺고 있을 가능성이 높습니다. KNN 대치는 이러한 다변수 관계를 고려하여 결측치를 예측하므로, 단순히 하나의 대표값으로 채우는 중앙값 대치보다 더 정확하고 데이터의 정보를 최대한 보존하는 방법입니다.

    ```python
    from sklearn.impute import KNNImputer
    import pandas as pd

    # 데이터 로드 및 범주형 변수 처리 가정
    df = pd.read_csv('hcv_data.csv')
    df['Sex'] = df['Sex'].map({'m': 0, 'f': 1})
    # 결측치가 있는 컬럼들 선택 (예시)
    cols_with_na = ['ALB', 'ALP', 'CHOL', ...]

    # KNN Imputer 적용
    imputer = KNNImputer(n_neighbors=5)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    ```

### 1-2. 이상치 확인 및 처리

- **이상치 확인**: `ALT`, `AST`, `BIL` 등 간 수치 관련 변수들은 질병 상태에 따라 매우 높은 값을 가질 수 있습니다. `sns.boxplot`을 사용하여 각 변수의 이상치를 시각적으로 확인합니다.
- **처리 여부 판단 및 작업**:
    - **판단**: 의료 데이터에서 이상치는 단순한 오류가 아니라, 질병의 심각성을 나타내는 중요한 임상적 정보일 수 있습니다. 예를 들어, `ALT` 수치가 매우 높은 것은 심각한 간 손상을 의미할 수 있으므로, 이를 무조건 제거하거나 왜곡해서는 안 됩니다.
    - **작업**: 이상치를 제거하는 대신, 이상치의 영향력을 줄이면서 정보를 보존하는 **`RobustScaler`**를 사용하여 데이터 스케일링을 진행합니다. `RobustScaler`는 중앙값과 사분위수를 사용하여 스케일링하므로, 극단적인 값(이상치)의 영향을 거의 받지 않습니다. 이는 이상치를 부드럽게 처리하는 효과를 가집니다.

    ```python
    from sklearn.preprocessing import RobustScaler

    # RobustScaler 적용
    scaler = RobustScaler()
    features = df_imputed.drop('Category', axis=1)
    scaled_features = scaler.fit_transform(features)
    df_processed = pd.DataFrame(scaled_features, columns=features.columns)
    df_processed['Category'] = df_imputed['Category']
    ```

### 1-3. EDA 및 그룹 간 차이 확인

- **EDA**: 1-2에서 전처리된 데이터를 사용하여, 종속변수 `Category`에 따라 각 독립변수의 분포가 어떻게 다른지 시각화합니다. `sns.boxplot`이나 `sns.violinplot`을 사용하면 그룹 간 분포 차이를 명확하게 확인할 수 있습니다.
- **통계적 확인**: **일원배치 분산분석(One-way ANOVA)**을 사용하여 각 독립변수의 평균이 `Category` 그룹 간에 통계적으로 유의미한 차이를 보이는지 검정합니다.

    ```python
    from scipy.stats import f_oneway

    # 시각화 예시 (ALT 변수)
    sns.boxplot(x='Category', y='ALT', data=df_processed)
    plt.title('ALT levels by Category')
    plt.show()

    # 통계 검정 (모든 변수에 대해 반복)
    categories = df_processed['Category'].unique()
    grouped_data = [df_processed['ALT'][df_processed['Category'] == cat] for cat in categories]
    f_stat, p_val = f_oneway(*grouped_data)

    if p_val < 0.05:
        print(f"ALT 변수는 Category 그룹 간에 통계적으로 유의미한 차이가 있습니다 (p-value: {p_val:.4f}).")
    ```

### 1-4. 주성분분석(PCA) 수행 가능성 검토

- **검토 방법**: PCA는 변수들 간에 상관관계가 존재할 때 의미가 있습니다. 이를 확인하기 위해 다음 두 가지 통계 검정을 수행합니다.
    1.  **KMO (Kaiser-Meyer-Olkin) 검정**: 변수들 간의 상관관계가 다른 변수에 의해 잘 설명되는지를 나타내는 지표입니다. 0.6 이상이면 PCA를 적용하기에 적합하다고 판단합니다.
    2.  **Bartlett의 구형성 검정**: 상관행렬이 단위행렬(변수들이 서로 독립)인지 검정합니다. 귀무가설(상관행렬=단위행렬)이 기각되어야(p-value < 0.05) PCA 적용이 의미 있습니다.

- **결과 제시**:
    ```python
    # !pip install factor_analyzer
    from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo

    # KMO 및 Bartlett 검정 수행
    kmo_all, kmo_model = calculate_kmo(features)
    chi_square_value, p_value = calculate_bartlett_sphericity(features)

    print(f"KMO Test: {kmo_model:.3f}")
    print(f"Bartlett Test p-value: {p_value:.4f}")
    ```
    - **판단**: KMO 값이 0.6 이상이고 Bartlett 검정의 p-value가 0.05 미만이라면, "변수들 간에 유의미한 상관관계가 존재하여 PCA를 통해 정보를 압축하고 새로운 설명변수를 도출하는 것이 가능하다"고 결론 내릴 수 있습니다.

---

## 2번 문제: 불균형 데이터 처리 및 분류

### 2-1. 데이터 불균형 문제 및 해결 방법

- **문제점**: 데이터 불균형이 심할 경우, 머신러닝 모델은 다수 클래스를 예측하도록 편향되어 학습됩니다. 이로 인해 모델의 정확도(Accuracy)는 높게 나타나지만, 정작 중요한 소수 클래스에 대한 예측 성능(e.g., Recall)은 매우 낮아지는 문제가 발생합니다.
- **해결 방법**:
    1.  **샘플링 기반 방법 (SMOTE)**: 소수 클래스의 데이터를 인위적으로 생성하여(오버샘플링) 클래스 간의 데이터 수를 맞춰주는 방법입니다. 정보 손실 없이 소수 클래스의 학습 기회를 늘릴 수 있습니다.
    2.  **비용 기반 방법 (Cost-Sensitive Learning)**: 알고리즘이 소수 클래스를 잘못 분류했을 때 더 큰 페널티(비용)를 부과하는 방법입니다. `class_weight='balanced'`와 같은 옵션을 통해 구현되며, 모델이 소수 클래스에 더 집중하도록 유도합니다.

### 2-2. 불균형 데이터 분류에 적합한 평가지표

1.  **재현율 (Recall)**: 실제 Positive인 것 중에서 모델이 Positive로 예측한 것의 비율. 질병 진단과 같이 실제 환자를 놓치지 않는 것(False Negative를 줄이는 것)이 중요할 때 핵심적인 지표입니다.
2.  **F1-Score**: 정밀도(Precision)와 재현율의 조화평균. 두 지표를 모두 고려해야 할 때 사용되는 균형 잡힌 지표입니다.
3.  **AUPRC (Area Under the Precision-Recall Curve)**: Precision-Recall 곡선의 아래 면적. 클래스 불균형이 매우 심할 때, 모델의 성능을 종합적으로 평가하는 데 AUC-ROC보다 더 신뢰성 높은 지표로 알려져 있습니다.

### 2-3. 이진 분류 및 불균형 처리 방법 비교

- **분석 방법**:
    1.  종속변수를 정상(0)과 비정상(1,2,3 -> 1)으로 이진화합니다.
    2.  `index % 5 == 0` 규칙에 따라 test/train 세트를 분리합니다.
    3.  **시나리오 1**: 훈련 데이터에 **SMOTE**를 적용한 후, 3가지 분류 모델(e.g., Logistic Regression, SVM, LightGBM)을 학습시키고 평가합니다.
    4.  **시나리오 2**: 원본 훈련 데이터에 `class_weight='balanced'` 옵션을 적용하여 3가지 모델을 학습시키고 평가합니다.
    5.  두 시나리오의 결과를 2-2에서 제시한 3가지 평가지표(Recall, F1-score, AUPRC)로 비교하여 어떤 불균형 처리 방식이 더 효과적이었는지 분석합니다.

---

## 3번 문제: 간염 심각도 분류

### 3-1. 다중 클래스 분류 모델 학습 및 평가

- **분석 방법**:
    1.  1번에서 전처리한 데이터에서 종속변수 `Category`가 1, 2, 3인 데이터만 필터링합니다.
    2.  `index % 5 == 0` 규칙에 따라 test/train 세트를 분리합니다.
    3.  3개의 분류 모델(e.g., Logistic Regression, Random Forest, LightGBM)을 학습시킵니다. 다중 클래스 분류이므로 Logistic Regression은 `multi_class='multinomial'`로 설정합니다.
    4.  테스트 데이터로 성능을 평가합니다. 평가지표는 `f1_score(average='weighted')`를 사용합니다.

### 3-2. 결과 기반 요인 논의

- **분석 방법**: 3-1에서 가장 성능이 좋았던 모델(e.g., LightGBM)의 **변수 중요도(`feature_importances_`)**를 추출하여 시각화합니다.
- **요인 논의**: 변수 중요도 그래프를 바탕으로 상위 3~5개 변수를 식별합니다. 예를 들어, `AST`, `ALT`, `GGT`, `BIL` 등이 높게 나타났다면, "간염의 심각도를 분류하는 데 있어 AST, ALT와 같은 간 손상 지표 효소 수치가 가장 결정적인 역할을 하는 것으로 나타났습니다. 또한, 담즙 관련 수치인 GGT와 빌리루빈(BIL) 수치 역시 간경화로 진행되는 심각한 상태를 판별하는 데 중요한 요인으로 작용함을 알 수 있습니다." 와 같이 의학적 의미와 연관 지어 설명합니다.

---

## 4번 문제: 지하철 이용객 예측

### 4-1. 데이터 전처리 및 기초통계량

- **분석 방법**: 문제에 제시된 6단계의 복잡한 전처리 과정을 순서대로 수행합니다.
    1.  승하차 인원이 모두 0인 행 제거
    2.  데이터를 long format으로 변환 (`pd.melt`) 및 `users` 컬럼 생성
    3.  날씨 데이터 결측치 처리 (강수량=0, 나머지는 ffill)
    4.  지하철 데이터와 날씨 데이터 병합 (`pd.merge`)
    5.  `weekday` 컬럼 추가
    6.  21-22년(훈련), 23년(테스트) 데이터 분리 후, 훈련 데이터의 기초통계량(`describe()`) 제시

### 4-2. 상관관계 분석 및 통계 검정

- **상관관계 분석**: 훈련 데이터에서 `users`와 날씨 관련 변수들(`기온`, `강수량` 등) 간의 상관계수 행렬을 구하고 `heatmap`으로 시각화합니다.
- **통계 검정**: `주말여부`에 따른 `users` 변수의 평균 차이가 있는지 확인하기 위해 **독립표본 t-검정(Independent t-test)**을 수행합니다. (`scipy.stats.ttest_ind`)

### 4-3. 회귀 모델 학습 및 평가

- **분석 방법**: 훈련 데이터를 사용하여 **선형 회귀(Linear Regression)**와 **LightGBM 회귀(LGBMRegressor)** 모델을 학습시키고, 테스트 데이터로 성능을 평가합니다. 평가지표는 **RMSE(Root Mean Squared Error)**를 사용합니다.

---

## 5번 문제: 대응표본 t-검정

### 5-1. EDA: `df.isnull().sum()`, `df.describe()`로 결측치와 기초통계량 확인.

### 5-2. 가설 설정

- **귀무가설(H0)**: A공장과 B공장의 평균 불량률에는 차이가 없다. ($\mu_{A_defect} = \mu_{B_defect}$)
- **대립가설(H1)**: A공장과 B공장의 평균 불량률에는 차이가 있다. ($\mu_{A_defect} \neq \mu_{B_defect}$)

### 5-3. 통계 검정

- **분석 방법**: 데이터가 동일한 날짜에 대해 A, B 공장의 수율을 기록한 것이므로, 두 샘플은 서로 종속적인 **대응표본**입니다. 따라서 **대응표본 t-검정(Paired t-test)**을 수행해야 합니다.
- **풀이**: `불량률 = 100 - 수율`로 불량률 컬럼을 계산한 후, `scipy.stats.ttest_rel(df['A_불량률'], df['B_불량률'])`을 사용하여 p-value를 구하고 가설 채택 여부를 결정합니다.

---

## 6번 문제: 분포 검정 및 카이제곱 검정

### 6-1. 푸아송 분포 적합도 검정 방법

1.  **카이제곱 적합도 검정 (Chi-Square Goodness-of-Fit Test)**: 일별 발생 빈도(0회, 1회, 2회...)의 관측도수와 푸아송 분포로부터 계산된 기대도수를 비교합니다. `scipy.stats.chisquare`를 사용하여 p-value를 계산하고, p-value가 크면 푸아송 분포를 따른다고 할 수 있습니다.
2.  **분산 대 평균 비율 확인**: 푸아송 분포는 평균과 분산이 같다는 특징이 있습니다. 데이터의 표본 평균과 표본 분산을 계산하여 그 비율이 1에 가까운지 확인합니다. 1에 가까울수록 푸아송 분포를 따른다고 볼 수 있습니다.

### 6-2. 교차표 생성: `pd.cut`으로 `지연정도` 범주를 만들고, `pd.crosstab`으로 `호선`과 `지연정도`의 교차표를 구합니다.

### 6-3. 통계 검정

- **가설**: H0: 호선과 지연정도는 서로 독립이다. H1: 호선과 지연정도는 서로 연관이 있다.
- **분석 방법**: 두 범주형 변수 간의 연관성을 검정하므로 **카이제곱 독립성 검정**을 사용합니다. 6-2의 교차표를 `scipy.stats.chi2_contingency`에 입력하여 p-value를 구하고 가설 채택 여부를 결정합니다.

---

## 7번 문제: 시계열 분석 (ARMA)

### 7-1. AR, MA, ARMA 설명

- **AR(p) (자기회귀 모델)**: 현재 시점의 데이터가 과거 `p`개 시점의 데이터에 선형적으로 의존하는 모델. ACF는 점차 감소하고, PACF는 `p`시차 이후에 급격히 0으로 절단되는 패턴을 보입니다.
- **MA(q) (이동평균 모델)**: 현재 시점의 데이터가 과거 `q`개 시점의 예측 오차(백색잡음)에 의존하는 모델. ACF는 `q`시차 이후에 급격히 0으로 절단되고, PACF는 점차 감소하는 패턴을 보입니다.
- **ARMA(p,q)**: AR(p)와 MA(q) 모델을 결합한 형태로, 과거 데이터와 과거 예측 오차 모두에 의존합니다. ACF와 PACF 모두 점차적으로 감소하는 패턴을 보입니다.

### 7-2. ACF, PACF 분석 및 모델 제시

- **분석 방법**: `statsmodels.graphics.tsaplots`의 `plot_acf`와 `plot_pacf` 함수를 사용하여 주어진 시계열 데이터의 ACF, PACF를 시각화합니다.
- **해석 및 모델 제시**: 두 그래프의 패턴을 분석합니다. 예를 들어, PACF가 2시차 이후에 절단되고 ACF가 점차 감소한다면 AR(2) 모델을, ACF가 1시차 이후에 절단되고 PACF가 점차 감소한다면 MA(1) 모델을, 둘 다 점차 감소한다면 ARMA(1,1) 등을 잠정적인 모델로 제시할 수 있습니다.
