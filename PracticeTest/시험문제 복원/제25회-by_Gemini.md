# ADP 25회 실기 문제 풀이 by Gemini

본 문서는 "제25회.md" 파일에 제시된 문제들에 대한 분석 방법론과 풀이 과정을 상세히 설명합니다. 각 문제에 대해 가능한 여러 분석 방법을 소개하고, 실제 문제에 적용하는 과정을 코드 예제와 함께 제시합니다.

---

## 1번 문제: RFM 기반 고객 군집분석

### 1-1. EDA, 이상치 제거, F/M Feature 생성 및 탐색적 분석

고객 데이터를 정제하고, 고객의 구매 빈도(Frequency)와 총 구매액(Monetary)을 나타내는 새로운 특성을 만들어 고객 행동을 분석합니다.

- **사용 가능한 분석 방법**
    - **데이터 정제 (Data Cleaning)**
        - **설명**: 모델링에 방해가 되거나 분석을 왜곡할 수 있는 데이터를 식별하고 처리합니다. 결측치, 비정상적인 값(e.g., 수량/가격 <= 0), 중복 데이터 등을 확인하고 제거합니다.
    - **특성 공학 (Feature Engineering)**
        - **설명**: 기존 데이터를 바탕으로 분석 목적에 맞는 새로운 변수(특성)를 생성합니다. 여기서는 고객별 구매 빈도(F)와 총 구매액(M)을 계산합니다.
    - **탐색적 데이터 분석 (EDA)**
        - **설명**: 생성된 특성의 분포, 통계치, 변수 간 관계 등을 시각화하여 데이터에 대한 인사이트를 얻습니다. 주로 히스토그램, 박스플롯, 산점도 등을 사용합니다.

- **현재 문제에 관한 풀이 방법**
    1.  **이상치 제거**:
        - `Quantity`가 0 이하인 경우는 반품(cancellation)을 의미하므로 분석에서 제외합니다.
        - `UnitPrice`가 0인 경우는 유효한 거래로 보기 어려우므로 제외합니다.
        - `CustomerID`가 없는 데이터는 고객을 특정할 수 없으므로 제외합니다.
    2.  **F, M 특성 생성**:
        - `TotalPrice` 컬럼을 `Quantity * UnitPrice`로 계산합니다.
        - **F (Frequency)**: `CustomerID`로 그룹화하여 고유한 `InvoiceNo`의 개수를 셉니다.
        - **M (Monetary)**: `CustomerID`로 그룹화하여 `TotalPrice`의 합계를 구합니다.
    3.  **탐색적 분석**:
        - 생성된 F와 M 데이터는 분포가 오른쪽으로 매우 치우쳐 있을 가능성이 높습니다. `describe()`로 통계치를 확인하고, `histplot`으로 분포를 시각화하여 이를 확인합니다.
        - 분석의 용이성과 모델 성능 향상을 위해 로그 변환(`np.log1p`)을 적용하여 데이터 분포를 정규분포에 가깝게 만들어 줍니다.

    ```python
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    # 데이터 로드 가정
    # df = pd.read_csv('online_retail.csv', encoding='ISO-8859-1')

    # # 1. 이상치 제거
    # df.dropna(subset=['CustomerID'], inplace=True)
    # df = df[df['Quantity'] > 0]
    # df = df[df['UnitPrice'] > 0]

    # # 2. F, M 특성 생성
    # df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    # rfm_df = df.groupby('CustomerID').agg({
    #     'InvoiceNo': 'nunique', # Frequency
    #     'TotalPrice': 'sum'      # Monetary
    # }).rename(columns={'InvoiceNo': 'F', 'TotalPrice': 'M'})

    # # 3. 탐색적 분석 (로그 변환 전/후)
    # fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    # sns.histplot(rfm_df['F'], ax=axes[0, 0], kde=True).set_title('Frequency Distribution')
    # sns.histplot(rfm_df['M'], ax=axes[0, 1], kde=True).set_title('Monetary Distribution')

    # # 로그 변환
    # rfm_df['F_log'] = np.log1p(rfm_df['F'])
    # rfm_df['M_log'] = np.log1p(rfm_df['M'])

    # sns.histplot(rfm_df['F_log'], ax=axes[1, 0], kde=True).set_title('Log-Transformed Frequency')
    # sns.histplot(rfm_df['M_log'], ax=axes[1, 1], kde=True).set_title('Log-Transformed Monetary')
    # plt.tight_layout()
    # plt.show()
    ```

### 1-2. F, M feature 기반으로 군집분석 실시

- **사용 가능한 분석 방법**
    - **K-평균 군집분석 (K-Means Clustering)**
        - **설명**: 데이터를 K개의 군집으로 나누는 대표적인 비지도 학습 알고리즘입니다. 각 데이터 포인트를 가장 가까운 군집의 중심에 할당하는 과정을 반복하여 군집을 형성합니다.
        - **사전 작업**:
            1.  **특성 스케일링**: K-Means는 거리를 기반으로 하므로, 변수들의 스케일을 맞춰주는 것이 중요합니다. `StandardScaler`를 사용하여 데이터를 표준화합니다.
            2.  **최적의 K 찾기 (Elbow Method)**: K값을 변화시키면서 군집 내 오차 제곱합(WCSS)을 계산하고, 그래프가 팔꿈치처럼 급격히 꺾이는 지점을 최적의 K로 선택합니다.

- **현재 문제에 관한 풀이 방법**
    1.  로그 변환된 `F_log`와 `M_log` 특성을 `StandardScaler`로 표준화합니다.
    2.  Elbow Method를 사용하여 적절한 군집의 수(K)를 결정합니다.
    3.  결정된 K를 사용하여 K-Means 모델을 학습시키고, 각 고객이 어떤 군집에 속하는지 라벨을 부여합니다.
    4.  결과를 산점도로 시각화하여 군집이 어떻게 형성되었는지 확인합니다.

    ```python
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    # # 데이터 준비 (1-1에서 이어짐)
    # rfm_scaled = StandardScaler().fit_transform(rfm_df[['F_log', 'M_log']])

    # # Elbow Method로 최적의 K 찾기
    # wcss = []
    # for i in range(1, 11):
    #     kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    #     kmeans.fit(rfm_scaled)
    #     wcss.append(kmeans.inertia_)

    # plt.figure(figsize=(10, 5))
    # plt.plot(range(1, 11), wcss, marker='o')
    # plt.title('The Elbow Method')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('WCSS')
    # plt.show()

    # # K-Means 군집분석 (Elbow Method 결과 K=3 또는 4로 가정)
    # kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    # rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)

    # # 군집 결과 시각화
    # sns.scatterplot(x='F_log', y='M_log', hue='Cluster', data=rfm_df, palette='viridis')
    # plt.title('Customer Segments')
    # plt.show()
    ```

### 1-3. 군집 결과의 적합성 서술

- **분석 방법**
    - **군집 내 응집도 (Cohesion)**: 하나의 군집 내 데이터들이 얼마나 서로 가깝게 모여 있는지를 나타냅니다. K-Means의 `inertia_` 속성(군집 내 오차 제곱합, WCSS)이 응집도를 나타내는 대표적인 지표이며, 낮을수록 응집도가 높습니다.
    - **군집 간 분리도 (Separation)**: 서로 다른 군집들이 얼마나 멀리 떨어져 있는지를 나타냅니다. 높을수록 군집이 잘 분리된 것입니다.
    - **실루엣 계수 (Silhouette Score)**
        - **설명**: 응집도와 분리도를 종합적으로 고려하는 지표입니다. -1부터 1 사이의 값을 가지며, 1에 가까울수록 군집화가 잘 되었다고 평가합니다.
        - **해석**:
            - 1에 가까움: 현재 군집에 잘 속해 있고, 다른 군집과는 멀리 떨어져 있음.
            - 0에 가까움: 군집의 경계에 위치함.
            - 음수 값: 잘못된 군집에 할당되었을 가능성이 높음.

- **현재 문제에 관한 풀이 방법**
    - K-Means 알고리즘은 본질적으로 **군집 내 응집도를 최대화**(inertia를 최소화)하는 방향으로 동작합니다. Elbow Method는 바로 이 응집도 지표의 변화를 보고 최적의 K를 찾는 방법입니다.
    - 군집화의 적합성을 종합적으로 평가하기 위해 **실루엣 계수**를 계산합니다. `sklearn.metrics.silhouette_score`를 사용하며, 이 점수가 0.5 이상으로 높게 나온다면, 생성된 군집들이 내부적으로는 잘 뭉쳐있고(높은 응집도) 외부적으로는 잘 구분되어(높은 분리도) 적합하게 군집화되었다고 해석할 수 있습니다.

    ```python
    from sklearn.metrics import silhouette_score

    # # 실루엣 계수 계산 (1-2에서 이어짐)
    # score = silhouette_score(rfm_scaled, rfm_df['Cluster'])
    # print(f'Silhouette Score: {score:.3f}')
    ```

### 1-4. 군집 별 특성 및 비즈니스적 판단 제시

- **분석 방법**: 각 군집의 F, M 평균값을 계산하여 군집별 특성을 정의합니다.
- **현재 문제에 관한 풀이 방법**
    1.  `rfm_df.groupby('Cluster')[['F', 'M']].mean()` 코드를 통해 각 군집의 F, M 평균을 계산합니다.
    2.  평균값을 바탕으로 각 군집의 이름을 부여하고 비즈니스 전략을 제안합니다. (K=4 가정)
        - **군집 0 (F 높음, M 높음)**: **VIP 고객**. 구매 빈도와 구매액이 모두 높아 충성도가 매우 높은 핵심 고객층입니다.
            - **비즈니스 판단**: VIP 전용 혜택, 신제품 우선 체험 기회, 개인화된 감사 메시지 등을 통해 관계를 유지하고 이탈을 방지해야 합니다.
        - **군집 1 (F 낮음, M 낮음)**: **휴면/비활성 고객**. 구매 빈도와 금액이 모두 낮아 브랜드에 대한 관여도가 낮은 고객입니다.
            - **비즈니스 판단**: 재방문을 유도하기 위한 할인 쿠폰, 특별 프로모션 등 강력한 인센티브를 제공하는 리마케팅 캠페인이 필요합니다.
        - **군집 2 (F 높음, M 낮음)**: **충성도 높은 일반 고객**. 자주 방문하지만, 한 번에 구매하는 금액은 적습니다.
            - **비즈니스 판단**: 객단가를 높이기 위한 전략이 필요합니다. 관련 상품 추천, '함께 구매하면 좋은 상품' 제안, 일정 금액 이상 구매 시 무료 배송 등의 전략이 유효합니다.
        - **군집 3 (F 낮음, M 높음)**: **큰 손 고객**. 구매 빈도는 낮지만, 한 번 구매할 때 많은 금액을 지출합니다.
            - **비즈니스 판단**: 재방문 주기를 단축시키는 것이 중요합니다. 신제품 출시 알림, 구매 주기 예측을 통한 맞춤형 프로모션 제안 등으로 재방문을 유도할 수 있습니다.

---

## 2번 문제: 시계열 분석

**참고**: 이 문제는 데이터가 제공되지 않았습니다. 따라서 일반적인 월별 판매량 데이터를 가상으로 생성하여 풀이를 진행합니다.

### 2-1. EDA와 시각화

- **사용 가능한 분석 방법**
    - **시계열 플롯 (Time Series Plot)**: 시간에 따른 데이터의 변화를 선 그래프로 그려 추세(Trend), 계절성(Seasonality), 주기성(Cycle) 등을 시각적으로 파악합니다.
    - **시계열 분해 (Time Series Decomposition)**: `statsmodels.tsa.seasonal.seasonal_decompose`를 사용하여 시계열을 추세, 계절, 잔차(Residual) 성분으로 분해하여 각 구성 요소를 명확히 확인합니다.

- **현재 문제에 관한 풀이 방법**
    ```python
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from statsmodels.tsa.seasonal import seasonal_decompose

    # 가상 데이터 생성 (3년간의 월별 판매량)
    date_rng = pd.date_range(start='2020-01-01', end='2022-12-31', freq='M')
    sales = 100 + np.arange(len(date_rng)) * 2 + np.sin(np.arange(len(date_rng)) * np.pi / 6) * 20 + np.random.normal(0, 5, len(date_rng))
    ts_df = pd.DataFrame(date_rng, columns=['date'])
    ts_df['sales'] = sales
    ts_df.set_index('date', inplace=True)

    # 시계열 플롯
    plt.figure(figsize=(12, 6))
    plt.plot(ts_df.index, ts_df['sales'])
    plt.title('Monthly Sales Time Series')
    plt.show()

    # 시계열 분해
    decomposition = seasonal_decompose(ts_df['sales'], model='additive')
    fig = decomposition.plot()
    fig.set_size_inches(10, 8)
    plt.show()
    ```
    시각화 결과, 데이터가 우상향하는 **추세**와 12개월 주기의 **계절성**을 보임을 확인할 수 있습니다.

### 2-2. 결측치 처리 및 논리적 근거 제시

- **사용 가능한 분석 방법 (시계열)**
    - **보간법 (Interpolation)**: 결측치 양쪽의 값을 이용하여 선형적으로 채우는 방법(`df.interpolate()`). 추세가 있는 데이터에 적합합니다.
    - **앞/뒤 값으로 채우기 (Forward/Backward Fill)**: `df.fillna(method='ffill' or 'bfill')`. 데이터의 변화가 크지 않을 때 유용합니다.
    - **계절성 고려 평균 대치**: 특정 월의 결측치는 다른 연도의 동일한 월의 평균값으로 대치할 수 있습니다.

- **현재 문제에 관한 풀이 방법**
    - **선택 방식**: **선형 보간법 (Linear Interpolation)**
    - **논리적 근거**: 시계열 데이터는 시간의 흐름에 따른 연속성이 중요합니다. 특히 판매량 데이터처럼 뚜렷한 추세를 보이는 경우, 결측치 이전과 이후의 값을 직선으로 연결하여 채우는 선형 보간법이 데이터의 추세를 왜곡하지 않고 자연스럽게 값을 채워줄 수 있는 가장 합리적인 방법입니다. 단순 평균이나 ffill/bfill은 추세를 반영하지 못해 데이터의 흐름을 끊을 수 있습니다.

### 2-3. 계절성을 반영한 시계열 모델 및 성능 평가

- **사용 가능한 분석 방법**
    - **SARIMA (Seasonal Auto-Regressive Integrated Moving Average)**
        - **설명**: 계절성을 갖는 시계열 데이터에 특화된 모델입니다. 비계절성 부분(p,d,q)과 계절성 부분(P,D,Q,m)을 함께 고려하여 복잡한 시계열 패턴을 모델링할 수 있습니다.
        - **자동 파라미터 탐색**: `pmdarima.auto_arima` 라이브러리를 사용하면 최적의 (p,d,q)(P,D,Q,m) 파라미터를 자동으로 찾아주어 편리합니다.

- **현재 문제에 관한 풀이 방법**
    1.  `pmdarima.auto_arima`를 사용하여 최적의 SARIMA 모델을 찾습니다. `seasonal=True`, `m=12`(월별 데이터이므로)로 설정합니다.
    2.  데이터를 훈련/테스트 세트로 분리하고, 훈련 데이터로 모델을 학습시킵니다.
    3.  테스트 세트 기간에 대한 예측을 수행하고, 실제값과 비교하여 모델의 정확도를 평가합니다.
    4.  **평가 지표**: **RMSE (Root Mean Squared Error)**. 예측 오차를 실제값과 같은 단위로 보여주어 직관적인 해석이 가능합니다.

    ```python
    # !pip install pmdarima
    from pmdarima import auto_arima
    from sklearn.metrics import mean_squared_error
    import numpy as np

    # # 훈련/테스트 데이터 분리
    # train_data = ts_df['sales'][:-6]
    # test_data = ts_df['sales'][-6:]

    # # auto_arima로 최적 모델 탐색
    # sarima_model = auto_arima(train_data, seasonal=True, m=12,
    #                           stepwise=True, suppress_warnings=True,
    #                           error_action='ignore', max_p=2, max_q=2,
    #                           max_P=2, max_Q=2)

    # print(sarima_model.summary())

    # # 예측
    # predictions = sarima_model.predict(n_periods=len(test_data))

    # # 성능 평가
    # rmse = np.sqrt(mean_squared_error(test_data, predictions))
    # print(f"RMSE: {rmse:.3f}")

    # # 예측 결과 시각화
    # plt.plot(train_data.index, train_data, label='Train')
    # plt.plot(test_data.index, test_data, label='Test')
    # plt.plot(test_data.index, predictions, label='Prediction')
    # plt.legend()
    # plt.show()
    ```

### 2-4. 분석 결과 활용 방안 제안

- **분석 전문가로서의 제안**
    - **재고 관리 최적화**: 분석된 계절성 패턴과 미래 판매량 예측을 통해 성수기에는 충분한 재고를 확보하여 판매 기회 손실을 막고, 비수기에는 재고를 줄여 관리 비용을 절감할 수 있습니다.
    - **마케팅 전략 수립**: 판매량이 급증하는 시기(예: 연말)를 예측하여, 해당 기간 1~2개월 전에 마케팅 캠페인을 집중적으로 실행하여 효과를 극대화할 수 있습니다. 반대로 비수기에는 매출 증대를 위한 특별 프로모션을 기획할 수 있습니다.
    - **인력 및 예산 계획**: 미래 판매량 예측을 기반으로 필요한 인력 규모와 예산을 효율적으로 계획하고 배분할 수 있습니다.
    - **주의사항**: 이 모델은 과거 데이터의 패턴이 미래에도 반복될 것이라는 가정하에 만들어졌습니다. 따라서, 시장의 급격한 변화(예: 경쟁사 출현, 경제 위기)가 발생할 경우 예측 정확도가 떨어질 수 있으므로, **주기적인 모델 성능 모니터링과 새로운 데이터를 활용한 모델 재학습이 필수적**임을 제안합니다.

---

## 3번 문제: 통계 기초

### 3-1. 평균 속도 계산
- **분석 방법**: 이동 거리가 같을 때의 평균 속도는 **조화평균**을 사용합니다.
- **풀이**: `2 / (1/4 + 1/5) = 2 / (9/20) = 40 / 9`
    ```python
    speed = 2 / (1/4 + 1/5)
    print(f"평균 속도: {speed:.4f} km/h")
    ```

### 3-2. 연평균 증가 배수
- **분석 방법**: 평균 변화율, 증가율을 계산할 때는 **기하평균**을 사용합니다.
- **풀이**: 첫 해 대비 증가율은 `4000/3000`, 둘째 해 대비 증가율은 `5000/4000` 입니다. 이 두 증가율의 기하평균을 구합니다.
    ```python
    from scipy.stats.mstats import gmean
    growth_factors = [4000/3000, 5000/4000]
    avg_growth_factor = gmean(growth_factors)
    print(f"연평균 {avg_growth_factor:.4f}배 증가")
    ```

### 3-3. 조건부 확률
- **분석 방법**: $P(\text{등산}|\text{남자}) = \frac{P(\text{등산} \cap \text{남자})}{P(\text{남자})}$
- **풀이**: (등산을 좋아하는 남자 수) / (전체 남자 수) = 20 / (20 + 10) = 20 / 30 = 2/3
    ```python
    prob = 20 / (20 + 10)
    print(f"남성 중에서 등산을 좋아할 확률: {prob:.4f}")
    ```

### 3-4. 모분산의 95% 신뢰구간
- **분석 방법**: 카이제곱($\chi^2$) 분포를 사용합니다. 신뢰구간: $(\frac{(n-1)s^2}{\chi^2_{\alpha/2, n-1}}, \frac{(n-1)s^2}{\chi^2_{1-\alpha/2, n-1}})$
- **풀이**: n=12, s=9.74, df=11. $\chi^2_{0.025, 11}$과 $\chi^2_{0.975, 11}$ 값을 찾아 계산합니다.
    ```python
    from scipy.stats import chi2
    n = 12
    s_sq = 9.74**2
    df = n - 1
    alpha = 0.05
    
    chi2_lower = chi2.ppf(alpha / 2, df)
    chi2_upper = chi2.ppf(1 - alpha / 2, df)
    
    ci_lower = (df * s_sq) / chi2_upper
    ci_upper = (df * s_sq) / chi2_lower
    print(f"모분산의 95% 신뢰구간: [{ci_lower:.4f}, {ci_upper:.4f}]")
    ```

### 3-5. 모분산의 95% 신뢰구간
- **분석 방법**: 3-4와 동일.
- **풀이**: n=10, $s^2=90$, df=9.
    ```python
    n = 10
    s_sq = 90
    df = n - 1
    
    chi2_lower = chi2.ppf(alpha / 2, df)
    chi2_upper = chi2.ppf(1 - alpha / 2, df)
    
    ci_lower = (df * s_sq) / chi2_upper
    ci_upper = (df * s_sq) / chi2_lower
    print(f"모분산의 95% 신뢰구간: [{ci_lower:.4f}, {ci_upper:.4f}]")
    ```

---

## 4번 문제: 가설 검정 (단일표본 t-검정)

### 4-1. 연구가설과 귀무가설
- **귀무가설 (H0)**: 혈압약은 혈압을 낮추는 효과가 없다. (평균 혈압 감소량 $\mu \le 0$)
- **연구가설 (H1)**: 혈압약은 혈압을 낮추는 효과가 있다. (평균 혈압 감소량 $\mu > 0$)

### 4-2. 가설 검정
- **분석 방법**: 모표준편차를 모르고, 표본 크기가 20(<30)이므로 **단일표본 t-검정(One-sample t-test)**을 사용합니다.
- **풀이**:
    1.  검정 통계량 $t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}$를 계산합니다.
    2.  계산된 t-값에 해당하는 p-value를 구합니다. (단측 검정)
    3.  p-value가 유의수준 5%(0.05)보다 작으면 귀무가설을 기각하고, 크면 기각하지 못합니다.

    ```python
    import numpy as np
    from scipy import stats

    n = 20
    x_bar = 25
    s = 9.1
    mu_0 = 0
    alpha = 0.05

    # t-statistic 계산
    t_stat = (x_bar - mu_0) / (s / np.sqrt(n))

    # p-value 계산 (단측 검정)
    df = n - 1
    p_value = 1 - stats.t.cdf(t_stat, df)

    print(f"검정 통계량(t-statistic): {t_stat:.4f}")
    print(f"p-value: {p_value:.4f}")

    if p_value < alpha:
        print("가설 채택 여부: 귀무가설을 기각하고 연구가설을 채택합니다. 즉, 약이 혈압을 실제로 낮춘다고 할 수 있습니다.")
    else:
        print("가설 채택 여부: 귀무가설을 기각하지 못합니다. 즉, 약이 혈압을 낮춘다고 할 통계적 근거가 부족합니다.")
    ```

---

## 5번 문제: 조건부 확률

- **분석 방법**: $P(\text{배구}|\text{여학생}) = \frac{P(\text{배구} \cap \text{여학생})}{P(\text{여학생})}$
- **풀이**: (배구를 선호하는 여학생 수) / (전체 여학생 수) = 65 / (35 + 65) = 65 / 100 = 0.65
    ```python
    prob = 65 / (35 + 65)
    print(f"여학생 중에서 배구를 선호할 확률: {prob:.4f}")
    ```

---

## 6번 문제: 비모수 검정 (크러스칼-월리스 검정)

### 1. 혼합표본 순위 계산
- **풀이**: A, B, C 공장의 모든 데이터를 합쳐서 크기 순으로 정렬한 후 순위를 매깁니다. 동점인 경우 평균 순위를 부여합니다.
    ```python
    A = [52.48, 49.31, 53.24, 57.62, 48.83, 48.83, 57.90, 53.84, 57.62, 57.90]
    B = [47.68, 47.67, 51.21, 40.43, 41.38, 47.19, 44.94, 51.57, 57.62, 57.90]
    C = [72.33, 63.87, 65.34, 57.88, 62.28, 65.55, 59.25, 66.88, 62.00]
    
    all_data = pd.DataFrame({
        'value': A + B + C,
        'group': ['A']*len(A) + ['B']*len(B) + ['C']*len(C)
    })
    all_data['rank'] = all_data['value'].rank()
    print(all_data.sort_values('value'))
    ```

### 2. 연구가설과 귀무가설
- **귀무가설 (H0)**: 세 공장(A, B, C)에서 생산되는 제품 무게의 중앙값은 모두 동일하다.
- **연구가설 (H1)**: 적어도 한 공장의 제품 무게 중앙값은 다른 공장과 다르다.

### 3. 크러스칼-월리스 검정
- **분석 방법**: 세 개 이상의 독립적인 그룹 간의 중앙값 차이를 비교하는 비모수적 방법입니다. 정규성 가정이 만족되지 않을 때 ANOVA 대안으로 사용됩니다. `scipy.stats.kruskal` 함수를 사용합니다.
- **풀이**:
    1.  `scipy.stats.kruskal(A, B, C)`를 사용하여 검정 통계량(H)과 p-value를 계산합니다.
    2.  p-value를 유의수준 95% (alpha=0.05)와 비교합니다.
    3.  p-value < 0.05 이면 귀무가설을 기각하고, 아니면 기각하지 못합니다.

    ```python
    from scipy.stats import kruskal

    # 크러스칼-월리스 검정 수행
    h_stat, p_value = kruskal(A, B, C)

    print(f"검정 통계량(H-statistic): {h_stat:.4f}")
    print(f"p-value: {p_value:.4f}")

    if p_value < 0.05:
        print("가설 채택 여부: p-value가 유의수준 0.05보다 작으므로 귀무가설을 기각합니다. 즉, 적어도 한 공장의 제품 무게 중앙값은 다르다고 할 수 있습니다.")
    else:
        print("가설 채택 여부: p-value가 유의수준 0.05보다 크므로 귀무가설을 기각할 수 없습니다. 즉, 공장별 중앙값에 차이가 있다고 말할 수 없습니다.")
    ```

---

## 7번 문제: 최적화 (정수 계획법)

- **분석 방법**: 이 문제는 주어진 예산 제약 하에 NPV를 최대화하는 자산 조합을 찾는 **0/1 배낭 문제(Knapsack Problem)**의 변형입니다. **정수 선형 계획법(Integer Linear Programming)**으로 해결할 수 있습니다. Python의 `PuLP` 또는 `SciPy` 라이브러리를 사용합니다.

- **문제 정식화**
    - **결정 변수**: $x_i = 1$ (자산 $i$에 투자) 또는 $0$ (투자 안함), for $i=1, ..., 5$
    - **목적 함수**: $Maximize \ NPV = 30x_1 + 20x_2 + 31x_3 + 42x_4 + 44x_5$
    - **제약 조건**:
        1.  1년차 예산: $23x_1 + 15x_2 + 17x_3 + 16x_4 + 24x_5 \le 50$
        2.  2년차 예산: $23x_1 + 15x_2 + 25x_3 + 12x_4 + 23x_5 \le 60$
        3.  3년차 예산: $15x_1 + 12x_2 + 12x_3 + 13x_4 + 17x_5 \le 80$

- **풀이 (PuLP 라이브러리 사용)**
    ```python
    # !pip install pulp
    from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus

    # 데이터 정의
    assets = ['자산1', '자산2', '자산3', '자산4', '자산5']
    costs = {
        '1년차': [23, 15, 17, 16, 24],
        '2년차': [23, 15, 25, 12, 23],
        '3년차': [15, 12, 12, 13, 17]
    }
    npv = [30, 20, 31, 42, 44]
    budgets = {'1년차': 50, '2년차': 60, '3년차': 80}

    # 문제 정의
    prob = LpProblem("NPV_Maximization", LpMaximize)

    # 변수 정의
    x = LpVariable.dicts("x", assets, cat='Binary')

    # 목적 함수 추가
    prob += lpSum([npv[i] * x[assets[i]] for i in range(len(assets))])

    # 제약 조건 추가
    for year in ['1년차', '2년차', '3년차']:
        prob += lpSum([costs[year][i] * x[assets[i]] for i in range(len(assets))]) <= budgets[year]

    # 문제 풀이
    prob.solve()

    # 결과 출력
    print(f"Status: {LpStatus[prob.status]}")
    print(f"Maximum NPV: {prob.objective.value()}")
    print("최적의 투자안:")
    for v in prob.variables():
        if v.varValue > 0:
            print(f"{v.name} = {v.varValue}")
    ```
