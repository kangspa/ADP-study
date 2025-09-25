# ADP 24회 실기 문제 풀이 by Gemini

본 문서는 "제24회.md" 파일에 제시된 문제들에 대한 분석 방법론과 풀이 과정을 상세히 설명합니다. 각 문제에 대해 가능한 여러 분석 방법을 소개하고, 실제 문제에 적용하는 과정을 코드 예제와 함께 제시합니다.

---

## 1번 문제: 결석일수 예측

### 1-1. 데이터 EDA 및 시각화

데이터의 특징을 파악하고 변수 간의 관계를 시각적으로 탐색합니다.

- **사용 가능한 분석 방법**
    - **기술 통계 (Descriptive Statistics)**
        - **설명**: 데이터의 중심 경향(평균, 중앙값), 변동성(표준편차), 분포(사분위수) 등을 요약하여 데이터의 전반적인 특성을 파악합니다.
        - **코드 예제**:
        ```python
        import pandas as pd

        df = pd.read_csv('student_data.csv') # 데이터 로드 가정
        print(df.describe())
        ```
    - **상관관계 분석 (Correlation Analysis)**
        - **설명**: 숫자형 변수들 간의 선형 관계의 강도와 방향을 측정합니다. 히트맵으로 시각화하면 변수 간의 관계를 한눈에 파악하기 용이합니다.
        - **코드 예제**:
        ```python
        import seaborn as sns
        import matplotlib.pyplot as plt

        # 숫자형 데이터만 선택
        numeric_df = df.select_dtypes(include=['number'])
        corr_matrix = numeric_df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.show()
        ```
    - **분포 시각화 (Distribution Visualization)**
        - **설명**: 변수의 데이터 분포를 시각화합니다. 연속형 변수는 히스토그램이나 KDE 플롯, 범주형 변수는 카운트 플롯을 사용합니다.
        - **코드 예제**:
        ```python
        # 나이(age) 분포
        sns.histplot(df['age'], kde=True)
        plt.title('Age Distribution')
        plt.show()

        # 성별(sex) 분포
        sns.countplot(x='sex', data=df)
        plt.title('Gender Distribution')
        plt.show()
        ```

- **현재 문제에 관한 풀이 방법**
    - 위 세 가지 방법을 모두 사용하여 데이터의 특성을 탐색합니다.
    1. `df.describe()`를 통해 `age`, `absences` 등 주요 숫자 변수들의 통계치를 확인하여 데이터의 스케일과 이상치 존재 가능성을 확인합니다.
    2. `heatmap`을 그려 `absences`(결석일수)와 다른 숫자형 변수들(`studytime`, `failures` 등) 간의 상관관계를 파악하여, 어떤 변수가 예측에 중요할지 가늠합니다.
    3. `histplot`으로 `absences`의 분포를 확인하고, `countplot`으로 `sex`, `Pstatus` 등 범주형 변수들의 분포가 균일한지 확인합니다. `boxplot`을 사용하여 `studytime`에 따른 `absences`의 분포 차이를 시각적으로 확인해볼 수 있습니다.

### 1-2. 결측치 처리 및 변화 시각화

결측치를 적절한 값으로 대체하고, 처리 전후의 데이터 분포 변화를 시각화하여 타당성을 검증합니다.

- **사용 가능한 분석 방법**
    - **단순 대치법 (Simple Imputation)**
        - **설명**: 평균, 중앙값, 최빈값 등 대표값으로 결측치를 채웁니다. 간단하고 빠르지만, 데이터의 분산을 과소평가할 수 있습니다. `sklearn.impute.SimpleImputer`를 사용합니다.
        - **코드 예제**:
        ```python
        from sklearn.impute import SimpleImputer
        import numpy as np

        # 'age' 컬럼의 결측치를 중앙값으로 채우기
        imputer = SimpleImputer(strategy='median')
        df['age_imputed'] = imputer.fit_transform(df[['age']])
        ```
    - **K-최근접 이웃 (KNN) 대치법**
        - **설명**: 결측치가 있는 샘플과 가장 가까운 K개의 이웃 샘플들의 값을 사용하여 결측치를 예측하고 채웁니다. 데이터에 패턴이 존재할 때 유용합니다. `sklearn.impute.KNNImputer`를 사용합니다.
        - **코드 예제**:
        ```python
        from sklearn.impute import KNNImputer

        imputer = KNNImputer(n_neighbors=5)
        df_imputed = pd.DataFrame(imputer.fit_transform(numeric_df), columns=numeric_df.columns)
        ```

- **현재 문제에 관한 풀이 방법**
    - **결측치 처리**: `age` 컬럼에 결측치가 존재합니다. `age`는 특정 값에 몰려있을 가능성이 있으므로, 평균보다 이상치에 덜 민감한 **중앙값(median)**으로 대치하는 것이 안정적입니다.
    - **이유 및 기대효과**:
        - **이유**: 결측치가 있으면 대부분의 머신러닝 모델이 동작하지 않으므로, 모델 학습을 위해 결측치 처리는 필수적입니다. 중앙값 대치는 데이터의 원래 분포를 크게 왜곡하지 않으면서 간단하게 결측치를 제거하는 효과적인 방법입니다.
        - **기대효과**: 결측치를 처리함으로써 더 많은 데이터를 모델 학습에 사용할 수 있게 되어 모델의 성능과 안정성이 향상될 것으로 기대됩니다.
    - **변화 시각화**:
    ```python
    # 결측치 처리 전후 'age' 분포 시각화
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.histplot(df['age'], kde=True, color='blue')
    plt.title('Age Distribution (Before Imputation)')

    # 중앙값으로 대치
    median_age = df['age'].median()
    df['age_filled'] = df['age'].fillna(median_age)

    plt.subplot(1, 2, 2)
    sns.histplot(df['age_filled'], kde=True, color='green')
    plt.title('Age Distribution (After Imputation)')
    plt.show()
    ```

### 1-3. 결석일수 예측모델 2개 제시 및 선택 근거

`absences`를 예측하기 위한 두 가지 회귀 모델을 제시합니다.

1.  **선형 회귀 (Linear Regression)**
    - **선택 근거**: 모델이 단순하여 해석이 용이하고, 계산 비용이 적어 빠르게 결과를 확인할 수 있습니다. 변수 간의 선형 관계를 파악하는 데 적합하며, 다른 복잡한 모델의 성능을 평가하기 위한 좋은 **베이스라인 모델**이 됩니다.

2.  **랜덤 포레스트 회귀 (Random Forest Regressor)**
    - **선택 근거**: 여러 개의 의사결정나무를 결합한 앙상블 모델로, 변수 간의 복잡한 비선형 관계와 상호작용을 잘 잡아냅니다. 데이터의 스케일링에 영향을 받지 않고, 이상치에 비교적 강건하여 전반적으로 높은 예측 성능을 기대할 수 있습니다.

### 1-4. 모델 생성 및 평가

위에서 선정한 두 모델을 생성하고, 적절한 평가 기준을 통해 성능을 비교합니다.

- **사용 가능한 분석 방법 (모델 평가 기준)**
    - **RMSE (Root Mean Squared Error)**
        - **설명**: 예측 오차의 제곱 평균에 루트를 씌운 값입니다. 오류 값에 제곱을 하므로 큰 오류에 더 큰 패널티를 부여하며, 결과가 원래 타겟 변수와 동일한 단위를 가져 해석이 용이합니다.
    - **R-squared (결정 계수)**
        - **설명**: 모델이 데이터의 분산을 얼마나 잘 설명하는지를 나타내는 지표입니다. 1에 가까울수록 모델이 데이터를 잘 설명한다는 의미입니다.

- **선정 이유**
    - **RMSE**: 예측 모델이 평균적으로 얼마나 오차를 보이는지 직관적으로 파악할 수 있기 때문에 선정합니다. (예: RMSE가 2이면 평균적으로 2일 정도의 결석일수 차이를 보임)
    - **R-squared**: 모델의 설명력을 함께 확인함으로써, RMSE 값의 좋고 나쁨을 상대적으로 판단할 수 있는 기준을 제공하기 때문에 선정합니다.

- **현재 문제에 관한 풀이 방법**
    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    import numpy as np

    # 데이터 전처리 (예시: 원-핫 인코딩, 결측치 처리 등)
    # df_processed = pd.get_dummies(df_filled, drop_first=True)
    # X = df_processed.drop('absences', axis=1)
    # y = df_processed['absences']

    # 이 예제에서는 간소화를 위해 숫자형 데이터만 사용
    X = df_filled.select_dtypes(include=np.number).drop('absences', axis=1)
    y = df_filled['absences']
    X = X.fillna(X.median()) # 모든 숫자 컬럼 결측치 처리

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. 선형 회귀
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
    lr_r2 = r2_score(y_test, lr_preds)

    # 2. 랜덤 포레스트
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
    rf_r2 = r2_score(y_test, rf_preds)

    print(f"Linear Regression - RMSE: {lr_rmse:.4f}, R2: {lr_r2:.4f}")
    print(f"Random Forest - RMSE: {rf_rmse:.4f}, R2: {rf_r2:.4f}")
    ```

### 1-5. 모델의 일반화 가능성 설명 및 시각화

모델이 학습 데이터에만 과적합되지 않고, 새로운 데이터에도 잘 작동할 것임을 설명합니다.

- **사용 가능한 분석 방법**
    - **교차 검증 (Cross-Validation)**
        - **설명**: 데이터를 여러 개의 폴드(fold)로 나누어, 일부는 훈련에, 일부는 검증에 사용하는 과정을 반복합니다. 이를 통해 모델 성능을 보다 안정적으로 평가하고 일반화 가능성을 확인할 수 있습니다.
    - **잔차 플롯 (Residuals Plot)**
        - **설명**: 예측값과 실제값의 차이(잔차)를 시각화한 플롯입니다. 잔차가 특정 패턴 없이 0을 중심으로 무작위로 흩어져 있다면 모델이 데이터를 잘 적합시킨 것으로 판단할 수 있습니다.

- **현재 문제에 관한 풀이 방법**
    - **설명**: 모델의 일반화 성능을 평가하기 위해 **K-폴드 교차 검증(K-Fold Cross-Validation)**을 수행합니다. 데이터를 단 한 번만 나누는 것보다 여러 번 나누어 검증하므로, 특정 데이터 분할에 따른 성능 변동성을 줄이고 더 신뢰할 수 있는 평가 결과를 얻을 수 있습니다. 이는 모델이 다양한 데이터 조합에서도 일관된 성능을 보인다는 것을 의미하며, 일상적인 상황에서도 잘 동작할 것이라는 근거가 됩니다.
    - **시각화**: 랜덤 포레스트 모델의 **특성 중요도(Feature Importance)**를 시각화하여 모델이 어떤 변수를 기반으로 예측하는지 확인합니다. `failures`(학고 횟수), `age`(나이) 등 상식적으로 결석에 영향을 줄 만한 변수들이 높은 중요도를 보인다면, 모델이 데이터의 의미 있는 패턴을 학습했다고 볼 수 있으며, 이는 일반화 가능성에 대한 신뢰를 더해줍니다.

    ```python
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 랜덤 포레스트 모델의 특성 중요도 시각화
    importances = rf.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance_df)
    plt.title('Feature Importance of Random Forest Model')
    plt.show()
    ```

### 1-6. 모델 최적화 방안

모델의 예측 성능을 더욱 향상시키기 위한 구체적인 방법을 제시합니다.

- **사용 가능한 분석 방법**
    - **하이퍼파라미터 튜닝 (Hyperparameter Tuning)**
        - **설명**: 모델의 성능에 영향을 미치는 하이퍼파라미터(예: 랜덤 포레스트의 나무 개수, 깊이)를 조정하여 최적의 조합을 찾는 과정입니다. `GridSearchCV`나 `RandomizedSearchCV`를 사용합니다.
        - **코드 예제**:
        ```python
        from sklearn.model_selection import GridSearchCV

        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_leaf': [1, 2, 4]
        }
        # rf 모델은 이미 학습되었으므로 새로운 모델로 탐색
        rf_for_grid = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(estimator=rf_for_grid, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
        # grid_search.fit(X_train, y_train) # 실제 실행 시에는 이 코드를 활성화
        # print(f"Best parameters found: {grid_search.best_params_}")
        ```
    - **특성 공학 (Feature Engineering)**
        - **설명**: 기존 변수를 조합하거나 변형하여 새로운 예측 변수를 만듭니다. 예를 들어, 부모님의 학력(`Medu`, `Fedu`)을 합쳐 '부모학력총합'이라는 파생변수를 만들 수 있습니다. 이는 모델이 데이터의 패턴을 더 잘 학습하도록 도울 수 있습니다.

- **현재 문제에 관한 풀이 방법**
    1.  **하이퍼파라미터 튜닝**: `GridSearchCV`를 사용하여 랜덤 포레스트 모델의 `n_estimators`(나무의 개수), `max_depth`(트리의 최대 깊이), `min_samples_leaf`(리프 노드가 되기 위한 최소 샘플 수) 등 주요 하이퍼파라미터의 최적 조합을 찾습니다. 이를 통해 모델을 데이터에 맞게 미세 조정하여 성능을 극대화할 수 있습니다.
    2.  **특성 공학**: `Medu`와 `Fedu`를 더한 `parent_edu` 변수를 생성하거나, `studytime`과 `freetime`의 비율 같은 새로운 변수를 만들어 모델에 추가합니다. 이러한 새로운 특성들이 `absences`와 더 강한 상관관계를 가질 수 있으며, 모델의 예측력을 높일 수 있습니다.

---

## 2번 문제: 다중 선형 회귀

### 2-1. 가변수화 및 회귀계수 유의성 검정

- **사용 가능한 분석 방법**
    - **가변수(Dummy Variable) 생성**: `pandas.get_dummies()` 함수를 사용하여 범주형 변수인 '광고비'를 0과 1로 이루어진 숫자형 변수로 변환합니다.
    - **다중 선형 회귀분석**: `statsmodels.formula.api.ols`를 사용하여 회귀 모델을 적합합니다. `statsmodels`는 각 회귀계수의 t-통계량과 p-value를 포함한 상세한 통계 요약 정보를 제공하여 유의성 검정을 편리하게 해줍니다.

- **현재 문제에 관한 풀이 방법**
    1.  '광고비' 컬럼을 가변수화합니다. '낮음'이 0, '높음'이 1이 됩니다.
    2.  `ols`를 이용해 `판매액 ~ 광고비_높음 + 연구개발비` 형태의 회귀식을 구성하고 모델을 학습시킵니다.
    3.  `model.summary()`를 통해 출력된 결과표에서 각 회귀계수(coefficient)의 `P>|t|` (p-value)를 확인합니다.
    4.  p-value가 유의수준(예: 0.05)보다 작으면 해당 회귀계수는 통계적으로 유의하다고 결론 내립니다.

    ```python
    import pandas as pd
    import statsmodels.api as sm

    data = {'광고비': ['낮음', '낮음', '낮음', '낮음', '높음', '높음', '높음', '높음', '높음', '높음'],
            '연구개발비': [52, 63, 74, 81, 96, 112, 127, 135, 143, 153],
            '판매액': [1322.53, 824.10, 1492.06, 1566.05, 1422.84, 1887.65, 1221.44, 877.59, 1570.82, 1402.99]}
    df_ad = pd.DataFrame(data)

    # 가변수화
    df_ad_dummy = pd.get_dummies(df_ad, columns=['광고비'], drop_first=True)
    df_ad_dummy = df_ad_dummy.rename(columns={'광고비_높음': 'ad_high'})

    # OLS 모델 적합
    X = sm.add_constant(df_ad_dummy[['ad_high', '연구개발비']]) # 상수항 추가
    y = df_ad_dummy['판매액']
    model = sm.OLS(y, X).fit()

    # 결과 출력
    print(model.summary())
    ```
    결과 요약표의 `P>|t|` 열을 보고 `ad_high`와 `연구개발비`의 p-value가 0.05보다 작은지 확인하여 유의성을 판단합니다.

### 2-2. 회귀 모형 검정

- **사용 가능한 분석 방법**
    - **F-검정 (F-test)**: 모델 요약표의 `F-statistic`과 `Prob (F-statistic)` (p-value)를 확인합니다. 이 검정은 "모든 회귀계수가 0이다"라는 귀무가설을 검정하며, p-value가 유의수준보다 작으면 모델 전체가 통계적으로 유의미하다고 판단합니다.
    - **결정계수 (R-squared)**: 모델 요약표의 `R-squared` 또는 `Adj. R-squared` 값을 확인합니다. 이 값은 독립변수들이 종속변수의 분산을 얼마나 설명하는지를 나타냅니다.
    - **잔차 분석 (Residual Analysis)**: 잔차의 정규성(Jarque-Bera test), 등분산성(잔차-예측값 플롯), 독립성(Durbin-Watson test)을 검토하여 회귀분석의 기본 가정을 만족하는지 확인합니다.

- **현재 문제에 관한 풀이 방법**
    - 위 `model.summary()` 결과에서 다음을 확인합니다.
    1.  **F-statistic의 p-value**: 모델의 전반적인 유의성을 판단합니다.
    2.  **Adj. R-squared**: 변수의 개수를 조정한 결정계수 값으로 모델의 설명력을 평가합니다.
    3.  **Durbin-Watson**: 2에 가까운 값이면 잔차의 자기상관이 없다고 판단할 수 있습니다.
    4.  **Jarque-Bera의 p-value**: 잔차의 정규성 가정을 검정합니다.

---

## 3번 문제: 두 집단 평균 차이 검정 (Z-검정)

**주의**: 이 문제는 Z-검정을 수행하기 위한 필수 정보인 **샘플 크기(n)**가 주어지지 않아 수학적으로 완벽한 풀이가 불가능합니다. 아래 풀이는 분석 방법론을 설명하기 위해 샘플 크기를 임의로 가정(예: n=30)하여 진행합니다.

### 3-1. 귀무가설과 대립가설

- **귀무가설 (H0)**: A 생산라인과 B 생산라인의 제품 평균은 차이가 없다. (μ_A = μ_B)
- **대립가설 (H1)**: A 생산라인과 B 생산라인의 제품 평균은 차이가 있다. (μ_A ≠ μ_B)

### 3-2. 두 평균의 차이 검정

- **사용 가능한 분석 방법**
    - **2-표본 Z-검정 (Two-sample Z-test)**
        - **설명**: 두 독립적인 집단의 평균을 비교할 때, 두 집단의 분산(또는 표준편차)을 알고 있는 경우에 사용합니다.
        - **검정 통계량 공식**:
          $$ Z = \frac{(\bar{x}_A - \bar{x}_B) - (\mu_A - \mu_B)}{\sqrt{\frac{\sigma_A^2}{n_A} + \frac{\sigma_B^2}{n_B}}} $$

- **현재 문제에 관한 풀이 방법**
    1.  주어진 값들은 다음과 같습니다:
        - A라인:  $\bar{x}_A = 5.7$, $\sigma_A = 0.03$
        - B라인:  $\bar{x}_B = 5.6$, $\sigma_B = 0.04$
        - 유의수준 $\alpha = 0.05$
    2.  **샘플 크기 $n_A, n_B$가 없으므로, 각각 30이라고 가정합니다.**
    3.  Z-검정 통계량을 계산합니다.
    4.  계산된 Z-통계량의 절대값을 임계값과 비교합니다. 양측 검정이므로 유의수준 $\alpha=0.05$에 대한 임계값은 $Z_{\alpha/2} = Z_{0.025} = 1.96$ 입니다. (문제에 주어진 $Z_{0.05}=1.65$는 단측 검정 값으로, 이 문제에는 부적합합니다.)
    5.  만약 $|Z| > 1.96$ 이면 귀무가설을 기각하고, 두 제품의 평균에 유의미한 차이가 있다고 결론 내립니다.

    ```python
    import numpy as np
    from scipy.stats import norm

    # 주어진 값 (샘플 크기는 가정)
    mean_a, std_a, n_a = 5.7, 0.03, 30
    mean_b, std_b, n_b = 5.6, 0.04, 30
    alpha = 0.05

    # Z-statistic 계산
    z_stat = (mean_a - mean_b) / np.sqrt((std_a**2 / n_a) + (std_b**2 / n_b))

    # p-value 계산 (양측 검정)
    p_value = 2 * (1 - norm.cdf(abs(z_stat)))

    # 임계값
    critical_value = norm.ppf(1 - alpha / 2)

    print(f"Z-statistic: {z_stat:.4f}")
    print(f"Critical value: {critical_value:.4f}")
    print(f"P-value: {p_value:.4f}")

    if abs(z_stat) > critical_value:
        print("귀무가설을 기각합니다. 두 제품의 평균은 유의미한 차이가 있습니다.")
    else:
        print("귀무가설을 기각하지 못합니다. 두 제품의 평균은 차이가 없다고 볼 수 있습니다.")
    ```

---

## 4번 문제: 조건부 확률과 베이즈 정리

### 4-1. 키트의 민감도(Sensitivity) 계산

- **분석 방법**: 민감도는 실제 질병이 있는 사람을 검사가 양성으로 판정할 확률입니다.
  - $P(\text{Test Positive} | \text{Actual Positive})$
- **풀이**:
  - 실제 양성 환자 수 = 370 (양성/양성) + 15 (음성/양성) = 385명
  - 키트가 양성으로 판정한 실제 양성 환자 수 = 370명
  - 확률 = 370 / 385

  ```python
  sensitivity = 370 / (370 + 15)
  print(f"키트의 민감도 (양성으로 잡아낼 확률): {sensitivity:.4f}")
  ```

### 4-2. 주장의 오류 서술

- **주장**: "양성으로 나온 사람이 실제 코로나에 걸려있을 확률이 97%다."
- **오류**: 이 주장은 **민감도(Sensitivity)**와 **양성 예측도(Positive Predictive Value, PPV)**를 혼동하고 있습니다.
    - **민감도**: $P(\text{Test Positive} | \text{Actual Positive})$ (질병이 있을 때 양성일 확률)
    - **양성 예측도(PPV)**: $P(\text{Actual Positive} | \text{Test Positive})$ (양성이 나왔을 때 실제 질병이 있을 확률)
- **옳지 않은 이유**:
  1.  주장하는 "97%"라는 수치는 4-1에서 계산한 민감도(약 96.1%)를 반올림한 것으로 보입니다. 하지만 이는 '양성 예측도'가 아닙니다.
  2.  양성 예측도(PPV)는 키트의 성능(민감도, 특이도)뿐만 아니라, 검사 대상 집단의 **유병률(Prevalence)**에 큰 영향을 받습니다.
  3.  문제에 제시된 표는 '코로나 의심 환자' 1,085명을 대상으로 한 결과입니다. 이 집단은 일반 인구보다 유병률이 훨씬 높은 특수 집단이므로, 이 표에서 계산한 PPV(370 / (370+10) ≈ 97.4%)를 일반적인 상황의 키트 우수성 근거로 사용하는 것은 **'기저율 오류(Base Rate Fallacy)'**에 해당합니다. 즉, 매우 편향된 샘플을 가지고 일반화하는 오류를 범하고 있습니다.

### 4-3. 실제 양성 예측도(PPV) 계산

- **분석 방법**: **베이즈 정리(Bayes' Theorem)**를 사용하여 유병률 1%를 적용했을 때의 실제 PPV를 계산합니다.
  - $P(D|T+) = \frac{P(T+|D)P(D)}{P(T+)}$
  - $P(T+) = P(T+|D)P(D) + P(T+|D')P(D')$
- **필요한 값**:
  - $P(D)$ (유병률) = 0.01
  - $P(D')$ (정상일 확률) = 1 - 0.01 = 0.99
  - $P(T+|D)$ (민감도) = 370 / 385
  - $P(T+|D')$ (위양성률, False Positive Rate) = 1 - 특이도
    - 특이도 $P(T-|D')$ = 690 / (10 + 690) = 690 / 700
    - 위양성률 = 1 - (690/700) = 10 / 700

- **풀이**:
  ```python
  # 값 정의
  p_D = 0.01 # 유병률
  p_notD = 1 - p_D
  p_Tpos_given_D = 370 / 385 # 민감도
  p_Tpos_given_notD = 10 / 700 # 위양성률

  # P(T+) 계산
  p_Tpos = (p_Tpos_given_D * p_D) + (p_Tpos_given_notD * p_notD)

  # PPV 계산 (베이즈 정리)
  ppv = (p_Tpos_given_D * p_D) / p_Tpos

  print(f"유병률 1% 적용 시, 실제 양성 예측도(PPV): {ppv:.4f}")
  ```
  계산된 PPV 값은 주장된 97%보다 훨씬 낮게 나타날 것입니다.

---

## 5번 문제: 신뢰구간 추정

### 5-1. 모평균에 대한 95% 신뢰구간 (모표준편차를 모를 때)

- **분석 방법**: 모표준편차를 모르고 표본 크기가 작으므로(n=16), **t-분포**를 사용하여 신뢰구간을 추정합니다.
  - 신뢰구간 공식: $\bar{x} \pm t_{\alpha/2, n-1} \times \frac{s}{\sqrt{n}}$
- **풀이**:
  1.  표본 평균($\bar{x}$)과 표본 표준편차($s$)를 계산합니다.
  2.  신뢰수준 95%($\alpha=0.05$)이고 자유도 $df = n-1 = 15$일 때의 t-임계값을 찾습니다. 문제에서 $t_{0.025, 15} = 2.131$로 주어졌습니다. (문제의 $t_{0.025,16}$은 오타로 보이며, 자유도 15가 맞습니다.)
  3.  공식에 값을 대입하여 신뢰구간을 계산합니다.

  ```python
  import numpy as np
  from scipy import stats

  data = np.array([71.2, 62.2, 53.2, 70.1, 65.7, 82.9, 62.9, 82, 68, 67.3, 75.3, 67.9, 77.6, 78.6, 66, 79])

  # 1. 표본 통계량 계산
  n = len(data)
  x_bar = np.mean(data)
  s = np.std(data, ddof=1) # 표본 표준편차 (ddof=1)

  # 2. t-임계값
  df = n - 1
  t_critical = 2.131 # t(0.025, 15)

  # 3. 신뢰구간 계산
  margin_of_error = t_critical * (s / np.sqrt(n))
  ci_lower = x_bar - margin_of_error
  ci_upper = x_bar + margin_of_error

  print(f"표본 평균: {x_bar:.4f}")
  print(f"표본 표준편차: {s:.4f}")
  print(f"95% 신뢰구간 (t-분포): [{ci_lower:.4f}, {ci_upper:.4f}]")
  ```

### 5-2. 모평균에 대한 95% 신뢰구간 (모표준편차를 알 때)

- **분석 방법**: 모표준편차($\sigma$)를 알고 있으므로, 표본 크기와 상관없이 **Z-분포(정규분포)**를 사용하여 신뢰구간을 추정합니다.
  - 신뢰구간 공식: $\bar{x} \pm Z_{\alpha/2} \times \frac{\sigma}{\sqrt{n}}$
- **풀이**:
  1.  표본 평균($\bar{x}$)은 5-1과 동일합니다.
  2.  모표준편차 $\sigma = 6$ kg을 사용합니다.
  3.  신뢰수준 95%($\alpha=0.05$)일 때의 Z-임계값을 찾습니다. 문제에서 $Z_{0.025} = 1.96$으로 주어졌습니다.
  4.  공식에 값을 대입하여 신뢰구간을 계산합니다.

  ```python
  # 2. 모표준편차, Z-임계값
  sigma = 6
  z_critical = 1.96 # Z(0.025)

  # 3. 신뢰구간 계산
  margin_of_error_z = z_critical * (sigma / np.sqrt(n))
  ci_lower_z = x_bar - margin_of_error_z
  ci_upper_z = x_bar + margin_of_error_z

  print(f"95% 신뢰구간 (Z-분포): [{ci_lower_z:.4f}, {ci_upper_z:.4f}]")
  ```
  ```