# ADP 29회 실기 문제 풀이 by Gemini

본 문서는 "제29회.md" 파일에 제시된 문제들에 대한 분석 방법론과 풀이 과정을 상세히 설명합니다. 각 문제에 대해 가능한 여러 분석 방법을 소개하고, 실제 문제에 적용하는 과정을 코드 예제와 함께 제시합니다.

---

## 1번 문제: 데이터 전처리

### 1-1. 각 계약자고유번호에 대해 가장 최신의 거주연도 행만 남기기

- **분석 방법**: 각 계약자의 연도별 데이터가 쌓여 있으므로, 분석의 기준 시점을 통일하기 위해 각 계약자의 가장 마지막 데이터만 남깁니다. `groupby()`와 `idxmax()` 또는 `tail(1)`을 사용합니다.

- **구현**:
    ```python
    import pandas as pd

    # 데이터 로드 가정
    # df = pd.read_csv('tenant_data.csv')

    # 방법 1: idxmax() 사용 (일반적으로 더 효율적)
    latest_df = df.loc[df.groupby('계약자고유번호')['거주연도'].idxmax()]

    # 방법 2: sort_values() + tail(1) 사용
    # latest_df = df.sort_values(by='거주연도').groupby('계약자고유번호').tail(1)

    # print(latest_df.head())
    ```

### 1-2. EDA 및 결측치 처리

- **EDA (탐색적 데이터 분석)**
    - `latest_df.info()`와 `latest_df.isnull().sum()`으로 데이터 타입과 결측치를 확인합니다. `퇴거연도`는 미퇴거 세대의 경우 모두 결측치이므로, 예측 변수로는 부적합합니다.
    - `latest_df.describe()`로 숫자형 변수의 분포를 확인하고, `sns.histplot` 등으로 시각화하여 이상치나 치우침을 확인합니다.
    - `sns.countplot`으로 `성별`, `결혼여부` 등 범주형 변수의 분포를 확인합니다.

- **결측치 처리**
    - **퇴거연도**: 예측에 사용할 변수가 아니므로 제거합니다.
    - **기타 변수**: 만약 다른 변수에 결측치가 있다면, 해당 변수의 특성에 따라 처리합니다.
        - **숫자형 변수**: 데이터 분포에 따라 평균(mean) 또는 중앙값(median)으로 대치합니다.
        - **범주형 변수**: 최빈값(mode)으로 대치합니다.

    ```python
    # # 결측치 확인
    # print(latest_df.isnull().sum())

    # # '퇴거연도' 컬럼 제거
    # latest_df.drop('퇴거연도', axis=1, inplace=True)

    # # 다른 컬럼(예: '아파트 평점')의 결측치를 중앙값으로 대치
    # if '아파트 평점' in latest_df.columns:
    #     median_score = latest_df['아파트 평점'].median()
    #     latest_df['아파트 평점'].fillna(median_score, inplace=True)
    ```

### 1-3. 이상치 처리

- **분석 방법**: `월세(원)`, `보증금(원)`과 같은 금액 변수나 `나이` 변수에서 비정상적으로 크거나 작은 값을 찾습니다. `sns.boxplot`으로 시각화하면 이상치를 쉽게 확인할 수 있습니다. 통계적으로는 **IQR Rule**을 적용하여 이상치를 식별하고 처리합니다.

- **구현 (Capping)**: 이상치를 제거하는 대신, 이상치의 영향력을 줄이기 위해 정상 범위의 최댓값/최솟값으로 값을 조정(Capping)하는 방법을 사용합니다.

    ```python
    def cap_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
        return df

    # # '월세(원)'과 '보증금(원)'에 대해 이상치 처리
    # for col in ['월세(원)', '보증금(원)']:
    #     latest_df = cap_outliers(latest_df, col)

    # # 처리 후 boxplot으로 확인
    # sns.boxplot(data=latest_df[['월세(원)', '보증금(원)']])
    # plt.show()
    ```

---

## 2번 문제: 파생변수 생성 및 차원축소

### 2-1. 재계약 횟수 이분 변수 구성

- **분석 방법**: `재계약횟수`의 중앙값을 기준으로 '높음'과 '낮음' 두 그룹으로 나눕니다.

    ```python
    # # 중앙값 계산
    # median_renewal = latest_df['재계약횟수'].median()

    # # 이분 변수 생성
    # latest_df['재계약횟수_이분'] = latest_df['재계약횟수'].apply(lambda x: '높음' if x >= median_renewal else '낮음')

    # print(latest_df['재계약횟수_이분'].value_counts())
    ```

### 2-2. 차원축소의 필요성 논의

- **결론**: **차원축소는 불필요합니다.**
- **근거**:
    1.  **낮은 차원**: 데이터의 변수(피처) 개수가 수십 개 수준으로, 현대적인 머신러닝 알고리즘이 처리하기에 충분히 적은 수입니다. 차원축소로 얻는 계산 효율성의 이득이 거의 없습니다.
    2.  **해석력 유지**: 각 변수(`나이`, `월세`, `거주개월` 등)는 그 자체로 중요한 의미를 가집니다. 차원축소를 수행하면 여러 변수가 결합된 새로운 변수가 생성되어, 어떤 요인이 재계약에 영향을 미치는지 직관적으로 해석하기 매우 어려워집니다. 분석의 목표가 예측뿐만 아니라 요인 파악에 있다면, 변수의 해석력을 유지하는 것이 중요합니다.
    3.  **정보 손실**: 차원축소는 필연적으로 원본 데이터의 정보 손실을 수반합니다. 현재 데이터셋에서는 정보 손실의 위험을 감수할 만큼 차원축소의 필요성이 크지 않습니다.

---

## 3번 문제: 모델링 및 결과 분석

### 3-1. 세그먼트 구분 및 특징 분석

- **분석 방법**: 2-1에서 생성한 `재계약횟수_이분` 변수를 기준으로 데이터를 '높음' 그룹과 '낮음' 그룹으로 나누고, 각 그룹의 인구통계학적, 계약 관련 변수들의 평균 또는 분포를 비교합니다.

    ```python
    # # 그룹별 특징 분석
    # segment_analysis = latest_df.groupby('재계약횟수_이분').agg({
    #     '나이': 'mean',
    #     '거주개월': 'mean',
    #     '월세(원)': 'mean',
    #     '결혼여부': lambda x: x.value_counts().index[0], # 최빈값
    #     '거주자 수': 'mean'
    # })
    # print(segment_analysis)
    ```
- **특징 분석 예시**: "재계약 횟수가 '높음'인 세그먼트는 '낮음'인 세그먼트에 비해 평균 연령과 평균 거주 개월이 더 높고, 기혼자 비율이 높은 특징을 보인다. 이는 안정적인 주거 환경을 선호하는 고연령층 및 기혼 가구가 장기 거주하는 경향이 있음을 시사한다."

### 3-2. 회귀 및 분류 분석

- **데이터 준비**: 모델링을 위해 범주형 변수를 원-핫 인코딩하고, 데이터를 훈련/테스트 세트로 분리합니다.

- **회귀 분석 (종속변수: `재계약횟수`)**
    - **방법론**: **Linear Regression** (선형 관계 확인을 위한 기준 모델), **Random Forest Regressor** (비선형 관계 학습에 강한 앙상블 모델)
    - **최종 모델 결정**: 두 모델의 성능을 R-squared, RMSE로 비교합니다. 일반적으로 비선형 패턴을 잘 학습하는 Random Forest가 더 높은 성능을 보여 최종 모델로 채택될 가능성이 높습니다.

- **분류 분석 (종속변수: `재계약횟수_이분`)**
    - **방법론**: **Logistic Regression** (결정 경계가 선형적인 기준 모델), **Random Forest Classifier** (강력한 성능의 앙상블 모델)
    - **최종 모델 결정**: 두 모델의 성능을 Accuracy, F1-Score, AUC로 비교합니다. 마찬가지로 Random Forest가 더 높은 성능을 보일 것으로 예상되어 최종 모델로 채택될 가능성이 높습니다.

### 3-3. 최종 모델의 유의 변수 확인

- **분석 방법**: 3-2에서 최종 채택된 Random Forest 모델의 `feature_importances_` 속성을 활용하여, 어떤 변수가 예측에 중요하게 작용했는지 확인하고 시각화합니다.

    ```python
    # # 분류 모델(RandomForestClassifier) 학습 후 가정
    # rf_clf.fit(X_train, y_train)
    # importances = rf_clf.feature_importances_
    # feature_df = pd.DataFrame({'feature': X_train.columns, 'importance': importances})
    # feature_df = feature_df.sort_values(by='importance', ascending=False)

    # # 중요도 시각화
    # plt.figure(figsize=(10, 8))
    # sns.barplot(x='importance', y='feature', data=feature_df.head(10))
    # plt.title('Top 10 Feature Importances')
    # plt.show()
    ```
- **설명**: "분석 결과, `거주개월`, `대표나이`, `보증금(원)` 등이 재계약 횟수 분류에 가장 중요한 변수로 나타났습니다. 이는 입주자의 나이가 많고, 거주 기간이 길수록, 그리고 상대적으로 높은 보증금을 내는 가구일수록 장기 재계약 경향이 강함을 의미합니다."

### 3-4. 분석 결과로 얻을 수 있는 점

1.  **장기 거주자 특성 파악**: 분석을 통해 어떤 특성을 가진 입주자가 장기 거주하는 경향이 있는지(e.g., 고연령, 기혼, 특정 평형대 선호)를 정량적으로 파악할 수 있습니다.
2.  **이탈 위험군 예측**: 재계약 횟수가 낮을 것으로 예측되는 입주자 그룹을 '이탈 위험군'으로 정의하고, 이들의 특성을 파악하여 주거 만족도를 높이기 위한 정책을 선제적으로 제안할 수 있습니다. (e.g., 청년층을 위한 커뮤니티 시설 확충, 특정 평형대의 시설 개선)
3.  **주거 안정 정책 수립**: 장기 거주를 유도하기 위한 새로운 인센티브 제도(e.g., 장기 계약자 월세 할인)를 기획하거나, 공실률을 최소화하기 위한 입주자 관리 전략을 수립하는 데 데이터 기반의 근거로 활용할 수 있습니다.

---

## 4번 문제: 야구 데이터 분석

**참고**: 데이터셋이 복잡하고 전처리 과정이 중요합니다. 문제의 의도에 맞게 논리적으로 데이터를 가공하는 것이 핵심입니다.

### 4-1. 데이터 전처리

- **분석 방법**:
    1.  **홈런 데이터 제외**: 1번 혹은 2번 타자가 홈런(value=4)을 친 회차(game_id, inning 조합)는 분석에서 제외합니다.
    2.  **1번 타자 출루 여부 정의**: 1번 타자의 이벤트(value)가 1루타(1), 2루타(2), 3루타(3), 볼넷/사사구(14, 15, 16) 중 하나에 해당하면 '출루(1)', 아니면 '아웃(0)'으로 이진 변수(`first_batter_on_base`)를 생성합니다.
    3.  **득점 발생 여부 정의**: 각 회차별로 최종 득점(`runs_scored`)이 0보다 크면 '득점(1)', 아니면 '무득점(0)'으로 종속 변수(`run_scored`)를 생성합니다.
    4.  **희생 번트 여부 정의**: 2번 타자의 이벤트가 희생 번트(value=6)이면 '번트(1)', 아니면 '번트아님(0)'으로 이진 변수(`sac_bunt`)를 생성합니다.

### 4-2. 로지스틱 회귀 및 계수 검정

- **분석 방법**: `statsmodels`의 `Logit`을 사용하여 로지스틱 회귀분석을 수행합니다. `statsmodels`는 각 회귀계수의 유의확률(p-value)을 제공하므로 계수 검정에 용이합니다.
- **모델**: `run_scored ~ first_batter_on_base + sac_bunt`
- **계수 검정**: `summary()` 결과 테이블에서 `sac_bunt` 변수의 `P>|z|` 값을 확인합니다. 이 값이 유의수준(e.g., 0.05)보다 작으면, 2번 타자의 희생번트 여부는 득점 발생에 유의미한 영향을 미친다고 해석할 수 있습니다.

### 4-3. SMOTE 적용

- **분석 방법**: 득점이 발생한 회차(1)는 발생하지 않은 회차(0)보다 훨씬 적어 클래스 불균형이 심각할 것입니다. `imblearn`의 `SMOTE`를 훈련 데이터에 적용하여 득점 발생 케이스를 오버샘플링합니다.

### 4-4. SMOTE 적용 후 로지스틱 회귀 분석

- **분석 방법**: SMOTE로 처리된 훈련 데이터로 로지스틱 회귀 모델을 다시 학습시키고 `summary()` 결과를 확인합니다.
- **결과 분석**: SMOTE 적용 후, 모델은 소수 클래스(득점 발생)의 특징을 더 잘 학습하게 됩니다. 이로 인해 이전 모델에서는 유의하지 않았던 `sac_bunt`와 같은 변수가 유의하게 나타나거나, 회귀계수의 크기나 부호가 바뀔 수 있습니다. 이는 불균형 상태에서는 제대로 파악되지 않았던 소수 클래스의 패턴이 드러났기 때문이라고 해석할 수 있습니다.

---

## 5번 문제: 이항 확률

- **분석 방법**: 정해진 횟수(n)의 독립적인 시도에서 특정 성공 확률(p)을 가질 때, 특정 성공 횟수(k)가 나타날 확률을 계산하는 **이항분포** 문제입니다.
- **풀이**: n=25, p=0.03, k=3. `scipy.stats.binom.pmf(k, n, p)`를 사용합니다.

    ```python
    from scipy.stats import binom
    n, p, k = 25, 0.03, 3
    prob = binom.pmf(k, n, p)
    print(f"3개가 불량일 확률: {prob:.4f}")
    ```

## 6번 문제: 비율 검정

- **분석 방법**: 두 집단의 비율(양품률)에 차이가 있는지 검정하므로 **2-비율 Z-검정(Two-proportion Z-test)**을 사용합니다.
- **가설**: H0: $p_C = p_D$, H1: $p_C \neq p_D$
- **풀이**: `statsmodels.stats.proportion.proportions_ztest`를 사용합니다.

    ```python
    from statsmodels.stats.proportion import proportions_ztest
    count = [600, 200]  # 양품 개수
    nobs = [1000, 500] # 전체 제품 수
    stat, pval = proportions_ztest(count, nobs)
    print(f"검정 통계량: {stat:.3f}, p-value: {pval:.4f}")
    if pval < 0.05:
        print("귀무가설 기각: 두 회사의 양품률에 유의미한 차이가 있습니다.")
    else:
        print("귀무가설 채택: 양품률에 차이가 있다고 보기 어렵습니다.")
    ```

## 7번 문제: 분산분석 (ANOVA)

### 7-1. 차종 별 차이 검정

- **분석 방법**: 3개 이상의 그룹(차종 A,B,C,D)의 평균(범퍼 파손 정도)에 차이가 있는지 비교하므로 **일원배치 분산분석(One-way ANOVA)**을 사용합니다.
- **가설**: H0: $\mu_A = \mu_B = \mu_C = \mu_D$, H1: 모든 평균이 같지는 않다.
- **풀이**: `scipy.stats.f_oneway`를 사용합니다.

    ```python
    from scipy.stats import f_oneway
    # 데이터 로드 및 그룹별 데이터 분리 가정
    # group_A = df[df['name']=='A']['ratio']
    # group_B = ...
    # stat, pval = f_oneway(group_A, group_B, group_C, group_D)
    ```

### 7-2. 사후분석

- **의미 해석**: 만약 ANOVA 결과 p-value가 유의수준보다 커서 귀무가설을 채택했다면, "네 차종 간 범퍼 파손 정도의 평균에는 통계적으로 유의미한 차이가 없다"고 해석합니다.
- **사후분석**: 만약 귀무가설을 기각했다면(p-value < 0.05), 이는 "적어도 한 차종은 다른 차종과 평균 파손 정도가 다르다"는 의미입니다. 구체적으로 어떤 차종들끼리 차이가 나는지 알아보기 위해 **사후분석(Post-hoc analysis)**을 시행해야 합니다. **Tukey의 HSD(Honestly Significant Difference)** 검정이 일반적으로 사용됩니다.

    ```python
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    # ANOVA 결과 귀무가설 기각 시
    # tukey_result = pairwise_tukeyhsd(df['ratio'], df['name'], alpha=0.05)
    # print(tukey_result)
    ```
    - `tukey_result`의 결과 테이블에서 `reject` 컬럼이 `True`인 쌍들이 통계적으로 유의미한 차이를 보이는 차종 조합입니다.

## 8번 문제: 베이즈 정리

- **분석 방법**: 조건부 확률 문제로, **베이즈 정리**를 사용합니다. $P(L1 | \text{불량}) = \frac{P(\text{불량}|L1)P(L1)}{P(\text{불량})}$
- **풀이**:
    1.  **전체 불량률 $P(\text{불량})$ 계산 (전체 확률의 법칙)**:
        $P(\text{불량}) = P(\text{불량}|L1)P(L1) + P(\text{불량}|L2)P(L2) + P(\text{불량}|L3)P(L3)$
        $P(\text{불량}) = (0.011 \times 0.13) + (0.021 \times 0.37) + (0.033 \times 0.50) = 0.00143 + 0.00777 + 0.0165 = 0.0257$
    2.  **베이즈 정리 적용**:
        $P(L1 | \text{불량}) = \frac{0.011 \times 0.13}{0.0257} = \frac{0.00143}{0.0257} \approx 0.05564$
    3.  **결과**: 소수점 둘째자리에서 반올림하면 **0.06** 입니다.
