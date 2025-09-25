# ADP 30회 실기 문제 풀이 by Gemini

본 문서는 "제30회.md" 파일에 제시된 문제들에 대한 분석 방법론과 풀이 과정을 상세히 설명합니다. 각 문제에 대해 가능한 여러 분석 방법을 소개하고, 실제 문제에 적용하는 과정을 코드 예제와 함께 제시합니다.

---

## 1번 문제: 데이터 탐색 및 전처리 (혈압 데이터)

### 1-1. EDA를 시행하라.

- **분석 방법**:
    1.  **데이터 기본 정보 확인**: `df.info()`, `df.describe()`로 데이터의 구조, 타입, 결측치 유무, 변수별 통계치를 파악합니다.
    2.  **종속변수(DBP) 분포 확인**: `sns.histplot`과 `sns.boxplot`을 사용하여 혈압(DBP)의 분포가 정규성을 따르는지, 이상치는 없는지 시각적으로 확인합니다.
    3.  **독립변수와 종속변수 관계 확인**: `sns.scatterplot`을 사용하여 `Age`, `BMI` 등 주요 독립변수와 `DBP` 간의 선형 관계가 있는지 탐색합니다.
    4.  **변수 간 상관관계 분석**: `sns.heatmap`을 사용하여 모든 숫자형 변수 간의 상관관계를 시각화합니다. 이를 통해 다중공선성 문제의 가능성을 미리 확인하고, DBP와 강한 상관관계를 갖는 변수들을 파악합니다.

    ```python
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # 데이터 로드 가정
    # df = pd.read_csv('blood_pressure_data.csv')

    # # DBP 분포 확인
    # sns.histplot(df['DBP'], kde=True)
    # plt.title('Distribution of DBP')
    # plt.show()

    # # 상관관계 히트맵
    # plt.figure(figsize=(12, 10))
    # sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    # plt.title('Correlation Matrix')
    # plt.show()
    ```

### 1-2. 데이터 전처리가 필요하다면 수행하고 이유를 작성하라.

- **전처리 필요성 및 수행**:
    1.  **이상치 처리**: EDA 과정에서 Boxplot 등을 통해 발견된 이상치는 회귀 모델의 성능을 저하시킬 수 있습니다. 예를 들어, `Tri`(트리글리세리드)나 `ALT` 같은 변수에서 비정상적으로 높은 값이 발견될 수 있습니다. **이유**: 이상치는 회귀선을 왜곡시켜 모델의 예측력을 떨어뜨리고, 잔차 가정(정규성, 등분산성)을 위반하게 만들 수 있습니다. **처리**: IQR Rule을 사용하여 이상치를 식별하고, 제거하거나 상/하한값으로 대체(Capping)합니다.
    2.  **피처 스케일링**: `Age`, `BMI`, `Chol` 등 각 변수들은 서로 다른 단위와 값의 범위(scale)를 가집니다. **이유**: 규제가 있는 회귀 모델(Ridge, Lasso)이나 경사하강법 기반의 최적화를 사용하는 모델들은 변수 스케일에 영향을 받습니다. 스케일링을 통해 모든 변수가 모델에 동등한 영향력을 갖도록 조정하고, 모델의 수렴 속도를 높일 수 있습니다. **처리**: `StandardScaler`를 사용하여 모든 독립변수를 표준화합니다.

    ```python
    from sklearn.preprocessing import StandardScaler

    # # 이상치 처리 (예시: BMI)
    # Q1 = df['BMI'].quantile(0.25)
    # Q3 = df['BMI'].quantile(0.75)
    # IQR = Q3 - Q1
    # upper_bound = Q3 + 1.5 * IQR
    # df['BMI'] = df['BMI'].clip(upper=upper_bound)

    # # 피처 스케일링
    # scaler = StandardScaler()
    # X = df.drop('DBP', axis=1)
    # y = df['DBP']
    # X_scaled = scaler.fit_transform(X)
    # X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    ```

### 1-3. train test set 분리 및 통계적 확인

- **분석 방법**: `train_test_split`을 사용하여 데이터를 7:3으로 분리하고, 두 집단의 종속변수(`DBP`) 분포가 통계적으로 동일한지 t-test를 통해 확인합니다.

    ```python
    from sklearn.model_selection import train_test_split
    from scipy.stats import ttest_ind

    # # 데이터 분리
    # X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.3, random_state=42)

    # # 통계적 확인
    # print("Train set DBP mean:", y_train.mean())
    # print("Test set DBP mean:", y_test.mean())

    # t_stat, p_val = ttest_ind(y_train, y_test)
    # print(f"\nT-test for DBP distribution: t-statistic={t_stat:.3f}, p-value={p_val:.3f}")

    # if p_val > 0.05:
    #     print("p-value가 0.05보다 크므로, train/test set의 DBP 분포는 통계적으로 유의미한 차이가 없습니다. (잘 나뉨)")
    # else:
    #     print("p-value가 0.05보다 작으므로, 분포에 유의미한 차이가 있습니다. (잘못 나뉨)")
    ```

---

## 2번 문제: 차원축소 및 회귀분석 가정 검토

### 2-1. 독립변수의 차원축소 필요성 논의

- **결론**: **차원축소는 불필요합니다.**
- **근거**:
    1.  **낮은 차원**: 독립변수의 개수가 약 12개로, 모델이 처리하기에 충분히 적은 수입니다. 차원축소로 인한 계산 효율성 증가는 미미합니다.
    2.  **해석력 유지**: 혈압 예측 모델에서 각 변수(`Age`, `BMI`, `Chol` 등)가 혈압에 미치는 영향을 개별적으로 파악하는 것은 매우 중요합니다. 차원축소를 수행하면 변수들이 결합되어 해석이 불가능해지므로, 분석의 중요한 목적 중 하나를 잃게 됩니다.
    3.  **다중공선성 문제**: 상관관계가 높은 변수들이 일부 있지만(e.g., `FPG`와 `FFPG`), 이는 규제가 있는 회귀 모델(Ridge, Lasso)이나 트리 기반 모델로 충분히 대응 가능합니다. 해석력을 희생하면서까지 차원축소를 할 만큼 심각한 문제는 아닙니다.

### 2-2. 회귀분석의 기본가정 충족 여부 설명

- **분석 방법**: `statsmodels` 라이브러리를 사용하여 OLS 회귀분석을 수행한 후, 그 결과(특히 잔차)를 통해 기본 가정을 검토합니다.

    ```python
    import statsmodels.api as sm

    # # statsmodels를 위한 상수항 추가 및 모델 학습
    # X_train_sm = sm.add_constant(X_train)
    # model = sm.OLS(y_train, X_train_sm).fit()
    # predictions = model.predict(X_train_sm)
    # residuals = model.resid

    # print(model.summary())
    ```
- **가정 검토**:
    1.  **선형성**: **잔차 대 예측값 산점도**를 그려, 잔차들이 0을 중심으로 무작위로 흩어져 있으면 선형성 가정을 만족합니다. `sns.residplot(x=predictions, y=residuals)`로 확인합니다.
    2.  **잔차의 독립성**: `model.summary()` 결과의 **Durbin-Watson 통계량**을 확인합니다. 2에 가까운 값이면 자기상관이 없어 독립성 가정을 만족합니다.
    3.  **잔차의 정규성**: `model.summary()` 결과의 **Jarque-Bera p-value(Prob(JB))**가 0.05보다 크거나, **Q-Q Plot**(`sm.qqplot(residuals, line='s')`)에서 잔차들이 직선에 가깝게 분포하면 정규성 가정을 만족합니다.
    4.  **등분산성**: 선형성 검토에 사용한 **잔차 대 예측값 산점도**에서, 예측값의 크기와 상관없이 잔차의 흩어진 정도(분산)가 일정하면 등분산성 가정을 만족합니다. 깔때기 형태를 보이면 등분산성 가정을 위반한 것입니다.

---

## 3번 문제: 회귀 모델링 및 평가

### 3-1. 회귀분석 알고리즘 3개 선택 및 비교

1.  **Ridge Regression**
    - **선정 이유**: 전통적인 선형 회귀에 L2 규제를 추가하여 다중공선성 문제를 완화하고 모델의 과적합을 방지하는 안정적인 모델입니다.
    - **장점**: 모델이 안정적이고 해석이 용이하며, 다중공선성에 강합니다.
    - **단점**: 변수 자체를 제거하지는 못하며, 비선형 관계를 모델링하기 어렵습니다.
2.  **Random Forest Regressor**
    - **선정 이유**: 여러 개의 의사결정나무를 사용하는 앙상블 기법으로, 비선형적이고 복잡한 관계를 잘 학습합니다.
    - **장점**: 성능이 강력하고, 이상치에 강건하며, 변수 중요도를 제공합니다.
    - **단점**: 모델 내부가 복잡하여 해석이 어렵고(블랙박스), 선형 모델보다 학습/예측 속도가 느립니다.
3.  **LightGBM Regressor**
    - **선정 이유**: Gradient Boosting 기반의 고성능 앙상블 모델로, 속도와 정확도 면에서 뛰어납니다.
    - **장점**: 학습 속도가 매우 빠르고 메모리 사용량이 적으며, 일반적으로 매우 높은 예측 정확도를 보입니다.
    - **단점**: 하이퍼파라미터에 민감하여 튜닝이 중요하고, 데이터가 적을 경우 과적합되기 쉽습니다.

### 3-2. 3개 모델링 및 최적 알고리즘 선정

- **분석 방법**: 1-3에서 분리한 데이터를 사용하여 세 모델을 학습시키고, 테스트 데이터에 대한 **RMSE(Root Mean Squared Error)**를 계산하여 성능을 비교합니다. RMSE가 가장 낮은 모델을 최적 알고리즘으로 선정합니다.

    ```python
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from lightgbm import LGBMRegressor
    from sklearn.metrics import mean_squared_error
    import numpy as np

    # # 모델 학습 및 평가
    # models = {
    #     'Ridge': Ridge(),
    #     'RandomForest': RandomForestRegressor(random_state=42),
    #     'LightGBM': LGBMRegressor(random_state=42)
    # }

    # for name, model in models.items():
    #     model.fit(X_train, y_train)
    #     preds = model.predict(X_test)
    #     rmse = np.sqrt(mean_squared_error(y_test, preds))
    #     print(f"{name} RMSE: {rmse:.4f}")
    ```
    - **선정**: RMSE 값을 비교하여 가장 낮은 값을 기록한 모델(e.g., LightGBM)을 최종 선택합니다.

### 3-3. K-Fold 교차검증 수행

- **분석 방법**: 3-2에서 선정한 최적 모델(예: LightGBM)의 일반화 성능을 더 신뢰성 있게 평가하기 위해 K-Fold 교차검증을 수행합니다. 전체 데이터를 사용하여 5-Fold 교차검증을 진행하고, 5개 폴드에서 나온 RMSE 값들의 평균과 표준편차를 확인합니다.

    ```python
    from sklearn.model_selection import cross_val_score

    # # 최적 모델로 LightGBM 선정 가정
    # final_model = LGBMRegressor(random_state=42)

    # # K-Fold 교차검증 (k=5)
    # scores = cross_val_score(final_model, X_scaled_df, y, cv=5, scoring='neg_root_mean_squared_error')
    # rmse_scores = -scores # neg_... 이므로 부호 변경

    # print(f"K-Fold CV RMSE scores: {rmse_scores}")
    # print(f"K-Fold CV Mean RMSE: {rmse_scores.mean():.4f}")
    # print(f"K-Fold CV Std Dev: {rmse_scores.std():.4f}")
    ```

---

## 4번 문제: 자전거 사고 데이터 분석

### 4-1. ‘주말여부’ 변수 추가

- **분석 방법**: `시각` 컬럼에서 요일 정보를 추출하여 ‘주말여부’ 변수를 생성합니다.

    ```python
    # # 데이터 로드 가정
    # # df_bike = pd.read_csv('bike_accident.csv')
    # df_bike['datetime'] = pd.to_datetime(df_bike['시각'].str.replace('_', ' '), format='%Y-%m-%d %H시')
    # df_bike['weekday'] = df_bike['datetime'].dt.dayofweek # 월요일=0, 일요일=6
    # df_bike['주말여부'] = df_bike['weekday'].apply(lambda x: '주말' if x >= 5 else '평일')
    # print(df_bike['주말여부'].value_counts())
    ```

### 4-2. 독립변수 유의성 검정

- **분석 방법**: 종속변수 `피해자신체상해정도`와 각 독립변수 간의 연관성을 통계적으로 검정합니다.
    - **범주형 vs 범주형** (e.g., `가해자성별` vs. `피해자신체상해정도`): **카이제곱 검정** (`scipy.stats.chi2_contingency`)
    - **연속형 vs 범주형** (e.g., `가해자연령` vs. `피해자신체상해정도`): **ANOVA** (`scipy.stats.f_oneway`)
- **결과**: 각 검정의 p-value가 유의수준(0.05)보다 작은 변수들을 유의한 변수로 선택합니다.

### 4-3. SMOTE 적용 및 데이터 확인

- **분석 방법**: 4-2에서 선택된 유의한 변수들만 사용하여 훈련 데이터에 SMOTE를 적용합니다. 그 후, 원본 데이터와 SMOTE로 생성된 데이터를 합치고, 범주형 변수는 빈도, 연속형 변수는 평균을 계산하여 분포 변화를 확인합니다.

### 4-4. 모델링 및 성능 비교

- **분석 방법**: 4-3 데이터를 사용하여 로지스틱 회귀와 XGBoost 분류 모델을 학습시키고, 테스트 데이터로 성능을 비교합니다. 평가지표는 F1-score(weighted)나 AUPRC를 사용합니다. XGBoost 모델의 `feature_importances_`를 통해 영향력 있는 변수를 확인합니다.

---

## 5번 문제: 운송 최적화

- **분석 방법**: 총 운송비를 최소화하는 문제로, **선형 계획법(Linear Programming)**의 한 종류인 **수송 문제(Transportation Problem)**에 해당합니다. `scipy.optimize.linprog`를 사용하여 해결할 수 있습니다.
- **정식화**:
    - **결정변수**: $x_{ij}$ (공장 $i$에서 지역 $j$로 운송하는 제품의 수량)
    - **목적함수**: $Minimize \ Cost = 12x_{A1} + 5x_{A2} + ... + 15x_{C3}$
    - **제약조건**:
        - **공급 제약**: 각 공장의 총생산량을 초과할 수 없음 (e.g., $x_{A1}+x_{A2}+x_{A3} \le 70$)
        - **수요 제약**: 각 지역의 총수요량을 충족해야 함 (e.g., $x_{A1}+x_{B1}+x_{C1} \ge 30$)
        - **음이 아닌 조건**: $x_{ij} \ge 0$
- **결과**: 최적의 운송 계획(각 $x_{ij}$ 값)과 그때의 최소 총 운송비를 도출합니다.

---

## 6번 문제: 카이제곱 독립성 검정

### 6-1. 가설 설정

- **귀무가설(H0)**: 연령대와 헤드셋 선호도는 서로 관련이 없다 (독립이다).
- **연구가설(H1)**: 연령대와 헤드셋 선호도는 서로 관련이 있다 (독립이 아니다).

### 6-2. 가설 검증

- **분석 방법**: 두 범주형 변수 간의 연관성을 검정하므로 **카이제곱 독립성 검정**을 사용합니다.
- **풀이**: `pd.crosstab`으로 교차표를 생성하고, `scipy.stats.chi2_contingency`를 적용하여 p-value를 계산합니다. p-value가 유의수준 0.05보다 작으면 귀무가설을 기각하고, 연령대별 선호도에 차이가 있다고 결론 내립니다.

---

## 7번 문제: 이항분포 확률 및 기댓값

### 7-1. 복합 이항 확률

- **분석 방법**: 두 단계의 이항분포 문제입니다.
    1.  **1단계 (한 가족)**: 한 가족(n=6, p=0.5)에서 딸이 4명 이상(k=4, 5, 6)일 확률 $P(D \ge 4)$를 계산합니다. 이는 $P(D=4) + P(D=5) + P(D=6)$ 입니다.
    2.  **2단계 (다섯 가족)**: 1단계에서 구한 확률 $p_{new} = P(D \ge 4)$를 성공 확률로 하는 새로운 이항분포(n=5, $p=p_{new}$)를 생각합니다. 여기서 세 가족 이상(k=3, 4, 5)이 성공할 확률을 계산합니다.

### 7-2. 기댓값 계산

- **분석 방법**: 이항분포의 기댓값 공식 $E(X) = n \times p$를 사용합니다.
- **풀이**: n=5(다섯 가족), p는 7-1의 1단계에서 구한 한 가족이 4명 이상의 딸을 가질 확률($p_{new}$)입니다. $E(X) = 5 \times p_{new}$를 계산합니다.
