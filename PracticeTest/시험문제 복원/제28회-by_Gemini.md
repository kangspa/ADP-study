# ADP 28회 실기 문제 풀이 by Gemini

본 문서는 "제28회.md" 파일에 제시된 문제들에 대한 분석 방법론과 풀이 과정을 상세히 설명합니다. 각 문제에 대해 가능한 여러 분석 방법을 소개하고, 실제 문제에 적용하는 과정을 코드 예제와 함께 제시합니다.

---

## 1번 문제: 데이터 탐색 및 품질 개선

### 1-1. EDA 및 차원축소 필요성 확인

- **EDA (탐색적 데이터 분석)**
    1.  **데이터 로드 및 기본 정보 확인**: `df.info()`, `df.describe()`로 데이터 구조, 타입, 결측치, 통계 요약 확인.
    2.  **타겟 변수 분포 확인**: `absences`는 결석 횟수 등급(0~5)으로, 다중 클래스 분류 문제의 타겟 변수입니다. `sns.countplot()`으로 분포를 시각화하여 클래스 불균형 여부를 확인합니다.
    3.  **피처 간 관계 분석**: 숫자형 변수 간의 상관관계를 `sns.heatmap()`으로 시각화하고, 범주형/숫자형 변수와 `absences` 간의 관계를 `sns.boxplot()`이나 `sns.countplot()` 등으로 탐색합니다.

    ```python
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # 데이터 로드 가정
    # df = pd.read_csv('student_data.csv')

    # # 타겟 변수 분포 확인
    # sns.countplot(x='absences', data=df)
    # plt.title('Distribution of Absences Grade')
    # plt.show()

    # # 상관관계 히트맵
    # plt.figure(figsize=(10, 8))
    # # 숫자형 데이터만 선택하여 상관관계 계산
    # numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    # sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
    # plt.title('Correlation Matrix')
    # plt.show()
    ```

- **차원축소 필요성**
    - 이 데이터셋은 약 10~15개의 예측 변수를 가집니다. 이는 '고차원' 데이터로 보기 어렵습니다. 따라서, PCA와 같은 차원축소 기법을 적용했을 때 얻는 계산상의 이점(속도 향상)이 크지 않은 반면, 각 변수(e.g., 'studytime', 'age')가 가지는 고유한 의미를 잃어버려 모델 해석이 어려워지는 단점이 더 큽니다.
    - 상관관계 분석 결과, `Medu`와 `Fedu`처럼 강한 상관관계를 보이는 변수들이 존재할 수 있으나, 이는 모델링 시 다중공선성 문제를 유발할 수 있다는 신호일 뿐, 반드시 차원축소가 필요한 결정적인 이유는 아닙니다. 트리 기반 모델(Random Forest, LightGBM)은 다중공선성에 비교적 강건합니다.
    - **결론**: 현재 데이터셋에서는 차원축소의 **필요성이 낮습니다.** 모델 해석력을 유지하고 각 변수의 영향력을 직접 파악하는 것이 더 중요합니다.

### 1-2. 데이터 품질 개선 및 데이터셋 재생성

- **데이터 품질 개선 방법**
    1.  **범주형 변수 인코딩**: `sex`, `pstatus`, `guardian` 등 문자열로 된 범주형 변수는 모델이 학습할 수 있도록 숫자형으로 변환해야 합니다. 변수 내 순서가 중요하지 않으므로 **원-핫 인코딩(One-Hot Encoding)**을 적용하는 것이 적합합니다.
    2.  **클래스 불균형 해소**: 1-1의 EDA 결과, `absences` 등급의 분포가 불균일할 가능성이 높습니다. 소수 클래스의 데이터가 너무 적으면 모델이 해당 클래스를 잘 학습하지 못합니다. **SMOTE(Synthetic Minority Over-sampling Technique)** 기법을 사용하여 훈련 데이터에서 소수 클래스의 데이터를 인위적으로 생성함으로써 클래스 불균형을 완화할 수 있습니다.

- **데이터셋 재생성**
    ```python
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import SMOTE

    # # 원-핫 인코딩
    # df_processed = pd.get_dummies(df, drop_first=True)

    # # 데이터 분리
    # X = df_processed.drop('absences', axis=1)
    # y = df_processed['absences']
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # # SMOTE 적용 (훈련 데이터에만)
    # smote = SMOTE(random_state=42)
    # X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # print("Original training shape:", y_train.value_counts())
    # print("Resampled training shape:", y_train_resampled.value_counts())
    ```
    - 위 코드를 통해 범주형 변수가 처리되고, 훈련 데이터(`X_train_resampled`, `y_train_resampled`)의 클래스 분포가 균일하게 맞춰진 새로운 데이터셋이 생성됩니다.

### 1-3. 과적합 해결을 위한 개선안

- **가정**: SMOTE와 같은 오버샘플링 기법이 인위적인 샘플 생성으로 인해 모델의 과적합을 유발할 수 있음.

- **개선안 1: 규제 (Regularization)**
    - **설명**: 모델의 복잡도에 페널티를 부과하여 과적합을 방지하는 기법입니다. 신경망에서는 가중치의 크기를 제한하는 L1/L2 규제를 사용하고, 트리 기반 모델에서는 `max_depth`(트리 깊이 제한), `min_samples_leaf`(리프 노드의 최소 샘플 수), `n_estimators`(트리 개수) 등의 하이퍼파라미터를 조정하는 것이 규제와 유사한 역할을 합니다.
    - **장점**: 모델의 일반화 성능을 높여 새로운 데이터에 대한 예측력을 향상시킵니다.
    - **단점**: 최적의 규제 강도(하이퍼파라미터)를 찾기 위한 추가적인 튜닝 과정이 필요합니다.

- **개선안 2: 교차 검증 (Cross-Validation)**
    - **설명**: 데이터를 하나의 훈련/검증 세트로 나누지 않고, 여러 개(K개)의 폴드(fold)로 나누어 K번의 훈련과 검증을 수행하는 방법입니다. 모델 성능을 평가할 때 단일 데이터 분할에 따른 우연성을 줄여주어, 모델의 일반화 성능을 보다 안정적이고 신뢰성 있게 측정할 수 있습니다.
    - **장점**: 데이터의 모든 부분을 훈련과 검증에 사용하므로, 데이터가 적을 때 특히 유용하며, 모델 성능에 대한 신뢰도가 높습니다.
    - **단점**: 모델을 K번 훈련시켜야 하므로 계산 비용이 K배로 증가합니다.

---

## 2번 문제: 결석 예측 모델링 및 평가

### 2-1. 3가지 모델 생성 및 평가

- **분석 방법**: 1-2에서 생성한 데이터셋(`X_train_resampled`, `y_train_resampled`)을 사용하여 Random Forest, Neural Network(MLP), LightGBM 모델을 학습시키고, 원본 테스트 데이터(`X_test`, `y_test`)로 성능을 평가합니다. 다중 클래스 분류이므로, f1-score는 `average='weighted'` 옵션을 사용하여 클래스별 샘플 수를 고려한 가중 평균을 계산합니다.

    ```python
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from lightgbm import LGBMClassifier
    from sklearn.metrics import f1_score

    # # 모델 초기화
    # rf = RandomForestClassifier(random_state=42)
    # nn = MLPClassifier(random_state=42, max_iter=1000)
    # lgbm = LGBMClassifier(random_state=42)

    # # 모델 학습
    # rf.fit(X_train_resampled, y_train_resampled)
    # nn.fit(X_train_resampled, y_train_resampled)
    # lgbm.fit(X_train_resampled, y_train_resampled)

    # # 예측 및 평가
    # rf_preds = rf.predict(X_test)
    # nn_preds = nn.predict(X_test)
    # lgbm_preds = lgbm.predict(X_test)

    # print(f"Random Forest F1-Score: {f1_score(y_test, rf_preds, average='weighted'):.4f}")
    # print(f"Neural Network F1-Score: {f1_score(y_test, nn_preds, average='weighted'):.4f}")
    # print(f"LightGBM F1-Score: {f1_score(y_test, lgbm_preds, average='weighted'):.4f}")
    ```

### 2-2. Hard Voting, Soft Voting 구현 및 비교

- **장단점 설명**
    - **Hard Voting**: 다수결 원칙으로, 여러 모델 중 가장 많이 예측된 클래스를 최종 결과로 선택합니다.
        - **장점**: 구현이 간단하고 직관적입니다.
        - **단점**: 각 모델의 예측 확률(신뢰도) 정보를 활용하지 못합니다. 성능이 낮은 모델도 동등한 한 표를 행사하는 문제가 있습니다.
    - **Soft Voting**: 각 모델이 예측한 클래스별 확률의 평균을 내어, 가장 높은 평균 확률을 가진 클래스를 최종 결과로 선택합니다.
        - **장점**: 각 모델의 신뢰도를 종합적으로 고려하므로, 일반적으로 Hard Voting보다 성능이 우수합니다.
        - **단점**: 모든 기반 모델이 클래스 확률을 예측하는 기능을 제공해야 합니다.

- **구현 및 비교**
    ```python
    from sklearn.ensemble import VotingClassifier

    # # VotingClassifier 초기화
    # # NN은 스케일링에 민감하므로, Voting 전 스케일링된 데이터로 재학습 필요
    # # 여기서는 간결성을 위해 그대로 사용
    # hard_voting = VotingClassifier(estimators=[('rf', rf), ('nn', nn), ('lgbm', lgbm)], voting='hard')
    # soft_voting = VotingClassifier(estimators=[('rf', rf), ('nn', nn), ('lgbm', lgbm)], voting='soft')

    # # 학습 및 평가
    # hard_voting.fit(X_train_resampled, y_train_resampled)
    # soft_voting.fit(X_train_resampled, y_train_resampled)

    # hard_preds = hard_voting.predict(X_test)
    # soft_preds = soft_voting.predict(X_test)

    # print(f"Hard Voting F1-Score: {f1_score(y_test, hard_preds, average='weighted'):.4f}")
    # print(f"Soft Voting F1-Score: {f1_score(y_test, soft_preds, average='weighted'):.4f}")
    ```
    - **비교**: Soft Voting이 Hard Voting보다 F1-score가 더 높게 나올 가능성이 큽니다.

### 2-3. 실시간 온라인 시스템에 가장 적합한 모델 선정

- **선정 모델**: **LightGBM**
- **선정 이유**:
    1.  **추론 속도(Inference Speed)**: 실시간 시스템에서는 예측 응답 시간이 매우 중요합니다. LightGBM은 리프 중심 트리 성장(leaf-wise) 방식을 사용하여 Random Forest보다 학습 및 예측 속도가 매우 빠릅니다. 앙상블 모델인 VotingClassifier는 여러 모델을 실행해야 하므로 가장 느립니다.
    2.  **성능**: LightGBM은 일반적으로 Random Forest와 비슷하거나 더 높은 예측 성능(F1-score)을 보입니다. 즉, 속도 저하 없이 높은 정확도를 유지할 수 있습니다.
    3.  **자원 효율성**: 다른 모델에 비해 메모리 사용량이 적어, 자원이 제한적인 온라인 서버 환경에 더 적합합니다.
    - **결론**: LightGBM은 5개 모델 중 **빠른 속도**와 **높은 성능**을 가장 잘 만족시키므로, 실시간 온라인 시스템에 가장 적합한 모델입니다.

---

## 3번 문제: 모델 적용 및 운영 고려사항

### 3-1. 적정 모델 선정 및 추가 고려사항

- **적정 모델**: **소프트 보팅 (Soft Voting) 앙상블**
- **선정 이유**: 2-3과 달리, 실시간성이 아닌 '정확도'가 최우선이라면, 일반적으로 단일 모델보다 여러 모델의 예측을 종합하는 앙상블 모델이 더 안정적이고 높은 성능을 보입니다. 특히 Soft Voting은 각 모델의 신뢰도까지 고려하므로 가장 정확한 예측을 기대할 수 있습니다.
- **추가 고려사항**:
    1.  **하이퍼파라미터 튜닝**: `GridSearchCV`나 `RandomizedSearchCV`를 사용하여 각 개별 모델(RF, NN, LGBM)과 앙상블 모델의 최적 하이퍼파라미터를 찾아야 합니다. 이를 통해 모델 성능을极限까지 끌어올릴 수 있습니다.
    2.  **모델 해석력 확보**: 최종 모델이 왜 특정 학생을 '결석 위험군'으로 분류했는지 설명할 수 있어야 합니다. `SHAP`이나 `LIME` 같은 XAI(설명가능 AI) 기법을 적용하여, 예측 결과에 대한 근거를 제시하고 모델의 신뢰성을 높여야 합니다.

### 3-2. 모델 적용 및 운영 과정 고려사항

1.  **데이터 드리프트 모니터링**: 학생들의 행동 패턴이나 학교 정책은 시간이 지남에 따라 변할 수 있습니다(Data/Concept Drift). 모델의 예측 성능을 지속적으로 모니터링하고, 성능이 일정 수준 이하로 떨어지면 새로운 데이터로 모델을 주기적으로 재학습시키는 파이프라인을 구축해야 합니다.
2.  **공정성 및 편향성 검토**: 모델이 특정 인구통계학적 그룹(성별, 부모 동거 여부 등)에 대해 불리한 예측을 하지는 않는지, 모델의 편향성을 정기적으로 검토하고 완화하는 노력이 필요합니다.
3.  **실질적인 활용 방안(Actionability)**: 모델이 '결석 위험'을 예측했을 때, 학교 시스템이 자동으로 담당 교사나 상담사에게 알림을 보내고, 해당 학생에게 관심을 기울이거나 상담을 진행하는 등의 실질적인 후속 조치와 연계되어야 모델의 가치가 발휘됩니다.
4.  **피드백 루프 구축**: 모델의 예측에 기반한 개입(상담 등)이 실제로 학생의 결석률에 긍정적인 영향을 미쳤는지 등을 다시 데이터화하여, 모델 개선에 활용하는 피드백 루프를 설계해야 합니다.

---

## 4번 문제: 생존 분석

**참고**: 데이터가 없어 가상 데이터를 생성하여 풀이합니다.

### 4-1. Kaplan-Meier 생존 확률

- **분석 방법**: `lifelines` 라이브러리의 `KaplanMeierFitter`를 사용하여 회사별 생존 곡선을 추정하고, 특정 시점에서의 생존 확률을 계산합니다.

    ```python
    # !pip install lifelines
    from lifelines import KaplanMeierFitter

    # # 가상 데이터 생성
    # data = pd.DataFrame({
    #     'duration': [10, 25, 38, 45, 50, 22, 35, 48, 60, 42],
    #     'status': [1, 1, 0, 1, 0, 1, 1, 0, 0, 1], # 1:사망/고장, 0:생존/정상
    #     'company': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B']
    # })

    # kmf_A = KaplanMeierFitter()
    # kmf_B = KaplanMeierFitter()

    # # 회사별 생존 분석 수행
    # ax = plt.subplot(111)
    # kmf_A.fit(data[data['company']=='A']['duration'], data[data['company']=='A']['status'], label='Company A')
    # kmf_A.plot_survival_function(ax=ax)
    # kmf_B.fit(data[data['company']=='B']['duration'], data[data['company']=='B']['status'], label='Company B')
    # kmf_B.plot_survival_function(ax=ax)
    # plt.title('Survival Function by Company')
    # plt.show()

    # # 특정 시점 생존 확률
    # times = [25, 35, 45]
    # prob_A = kmf_A.predict(times)
    # prob_B = kmf_B.predict(times)

    # print("--- 생존 확률 ---")
    # print("Company A:", np.round(prob_A.values, 3))
    # print("Company B:", np.round(prob_B.values, 3))
    ```

### 4-2. Log-Rank 검정

- **가설 설정**
    - **귀무가설(H0)**: 두 회사의 부품 생존 곡선(생존 시간에 대한 분포)은 동일하다.
    - **대립가설(H1)**: 두 회사의 부품 생존 곡선은 동일하지 않다.
- **검정 수행**
    - **분석 방법**: `lifelines.statistics.logrank_test` 함수를 사용하여 두 그룹 간의 생존 곡선에 통계적으로 유의미한 차이가 있는지 검정합니다.

    ```python
    from lifelines.statistics import logrank_test

    # # Log-Rank 검정
    # results = logrank_test(
    #     durations_A=data[data['company']=='A']['duration'],
    #     durations_B=data[data['company']=='B']['duration'],
    #     event_observed_A=data[data['company']=='A']['status'],
    #     event_observed_B=data[data['company']=='B']['status']
    # )

    # print("\n--- Log-Rank Test ---")
    # print(f"검정 통계량: {results.test_statistic:.3f}")
    # print(f"p-value: {results.p_value:.4f}")

    # if results.p_value < 0.05:
    #     print("귀무가설 기각: 두 회사 간 생존 시간에 유의미한 차이가 있습니다.")
    # else:
    #     print("귀무가설 채택: 두 회사 간 생존 시간에 유의미한 차이가 없습니다.")
    ```

---

## 5번 문제: 맥니마 검정

### 5-1. 가설 설정

- **귀무가설(H0)**: 시식 행위는 구매 의사 변화에 영향을 주지 않는다. (즉, 구매 의사가 '유'에서 '무'로 바뀐 비율과 '무'에서 '유'로 바뀐 비율이 같다.)
- **대립가설(H1)**: 시식 행위는 구매 의사 변화에 영향을 준다.

### 5-2. 검정 및 결과 분석

- **분석 방법**: 시식 전/후 동일한 사람의 구매 의사 변화를 분석하는 것이므로, 대응되는 두 범주형 변수의 비율 차이를 검정하는 **맥니마 검정(McNemar's Test)**을 사용합니다.

    ```python
    from statsmodels.stats.contingency_tables import mcnemar

    # 데이터: 시식전_유 -> 시식후_무 (7명), 시식전_무 -> 시식후_유 (18명)
    # 표 형식: [[시식전유->시식후유, 시식전유->시식후무], [시식전무->시식후유, 시식전무->시식후무]]
    table = [[23, 7],
             [18, 12]]

    # 맥니마 검정 수행
    result = mcnemar(table, exact=True) # 샘플 수가 적을 경우 exact=True 사용

    print(f"검정 통계량: {result.statistic:.3f}")
    print(f"p-value: {result.pvalue:.4f}")

    if result.pvalue < 0.05:
        print("귀무가설 기각: 시식 행위는 구매 의사에 유의미한 영향을 줍니다.")
    else:
        print("귀무가설 채택: 시식 행위가 구매 의사에 영향을 준다고 보기 어렵습니다.")
    ```

---

## 6번 문제: 등분산 검정

- **가설 설정**
    - **귀무가설(H0)**: A, B 지역 학생들 점수의 분산은 동일하다. ($\sigma_A^2 = \sigma_B^2$)
    - **대립가설(H1)**: A, B 지역 학생들 점수의 분산은 동일하지 않다. ($\sigma_A^2 \neq \sigma_B^2$)
- **검정 수행**
    - **분석 방법**: 두 집단의 등분산성을 검정할 때는 **Levene 검정** 또는 **Bartlett 검정**을 사용합니다. Levene 검정이 데이터의 정규성에 덜 민감하여 더 일반적으로 사용됩니다.

    ```python
    from scipy.stats import levene
    import numpy as np

    # 가상 데이터 생성
    np.random.seed(42)
    score_A = np.random.normal(75, 10, 50)
    score_B = np.random.normal(75, 15, 50)

    # Levene 검정
    stat, p_val = levene(score_A, score_B)

    print(f"검정 통계량: {stat:.3f}")
    print(f"p-value: {p_val:.4f}")

    if p_val < 0.05:
        print("귀무가설 기각: 두 지역 학생들 점수의 분산은 다릅니다.")
    else:
        print("귀무가설 채택: 두 지역 학생들 점수의 분산은 같다고 할 수 있습니다.")
    ```

---

## 7번 문제: 편상관 분석

### 7-1. 편상관계수 및 유의확률

- **분석 방법**: 다른 변수('몸무게')의 효과를 통제한 상태에서 두 변수('나이', '콜레스테롤') 간의 순수한 상관관계를 분석하기 위해 **편상관 분석(Partial Correlation)**을 수행합니다. `pingouin` 라이브러리를 사용하면 편리합니다.

    ```python
    # !pip install pingouin
    import pingouin as pg

    # 데이터 로드 가정
    # df_heart = pd.read_csv('erbil_heart_disease.csv')

    # 편상관 분석 수행
    partial_corr = pg.partial_corr(data=df_heart, x='age', y='Cholesterol', covar='weight')

    print(partial_corr)
    r = partial_corr['r'].values[0]
    p_value = partial_corr['p-val'].values[0]

    print(f"\n편상관계수: {r:.4f}")
    print(f"유의확률: {p_value:.4f}")
    ```

### 7-2. 상관계수 검정

- **가설 설정**
    - **귀무가설(H0)**: 몸무게를 통제했을 때, 나이와 콜레스테롤의 편상관계수는 0이다.
    - **대립가설(H1)**: 편상관계수는 0이 아니다.
- **검정 결과**
    - 7-1에서 계산된 **유의확률(p-value)**을 유의수준 0.05와 비교합니다.
    - 만약 p-value가 0.05보다 작으면 귀무가설을 기각하고, "몸무게의 영향을 제외하더라도 나이와 콜레스테롤 사이에는 유의미한 상관관계가 있다"고 결론 내립니다.
    - 만약 p-value가 0.05보다 크면 귀무가설을 기각하지 못하며, "두 변수 간 순수한 상관관계가 있다고 보기 어렵다"고 결론 내립니다.
