# ADP 26회 실기 문제 풀이 by Gemini

본 문서는 "제26회.md" 파일에 제시된 문제들에 대한 분석 방법론과 풀이 과정을 상세히 설명합니다. 각 문제에 대해 가능한 여러 분석 방법을 소개하고, 실제 문제에 적용하는 과정을 코드 예제와 함께 제시합니다.

---

## 머신러닝 (50점)

### 1-1. 결측치 확인 및 제거

- **분석 방법**
    - **결측치 확인**: `df.isnull().sum()` 또는 `df.info()`를 사용하여 어떤 컬럼에 얼마나 많은 결측치가 있는지 확인합니다.
    - **결측치 제거**: 분석 목적에 따라 결측치가 있는 행 또는 열을 제거합니다. 고객 기반 분석에서는 고객을 식별할 수 없는 데이터(`CustomerID`가 null)는 무의미하므로 반드시 제거해야 합니다.

- **현재 문제에 관한 풀이 방법**
    - 온라인 소매 데이터에서는 `Description`과 `CustomerID` 컬럼에 결측치가 주로 발견됩니다.
    - `CustomerID`가 없는 데이터는 어떤 고객의 행동인지 알 수 없으므로, 고객 군집분석 및 추천에 사용할 수 없습니다. 따라서 해당 행들은 제거합니다.

    ```python
    import pandas as pd

    # 데이터 로드 가정
    # df = pd.read_csv('online_retail.csv', encoding='ISO-8859-1')

    # # 결측치 확인
    # print("결측치 확인:")
    # print(df.isnull().sum())

    # # CustomerID 결측치 제거
    # df.dropna(subset=['CustomerID'], inplace=True)

    # # CustomerID가 정수형이 되도록 변환
    # df['CustomerID'] = df['CustomerID'].astype(int)

    # print("\n결측치 제거 후 정보:")
    # print(df.info())
    ```

### 1-2. 이상치 제거 방법 설명 및 결과 제시

- **이상치 제거 방법 설명**
    1.  **업무 지식 기반 제거 (Domain Knowledge-based)**: 데이터의 비즈니스 의미를 파악하여 비정상적인 데이터를 제거합니다. 예를 들어, 온라인 쇼핑 데이터에서 구매 수량(`Quantity`)이 0 이하인 경우는 '반품'을 의미하며, 단가(`UnitPrice`)가 0인 경우는 정상적인 거래가 아니므로 이상치로 간주할 수 있습니다.
    2.  **통계적 기법 (IQR Rule)**: 사분위수 범위를 이용하여 정상 범위를 벗어나는 데이터를 이상치로 판단합니다. 데이터가 정규분포를 따르지 않을 때 효과적인 방법입니다. 구체적으로 `Q1 - 1.5 * IQR` 보다 작거나 `Q3 + 1.5 * IQR` 보다 큰 값을 이상치로 정의합니다. (IQR = Q3 - Q1)

- **현재 문제에 관한 풀이 방법**
    1.  업무 지식을 활용하여 `Quantity > 0` 이고 `UnitPrice > 0` 인 데이터만 필터링합니다.
    2.  그 후, 통계적 기법인 IQR Rule을 `Quantity`와 `UnitPrice`에 적용하여 극단적인 값들을 추가로 제거합니다.
    3.  제거 전후의 데이터 기초 통계량(`describe()`)을 비교하여 이상치 제거 효과를 보입니다.

    ```python
    # # 1. 업무 지식 기반 이상치 제거
    # df_clean = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

    # print("--- 제거 전 통계량 ---")
    # print(df_clean[['Quantity', 'UnitPrice']].describe())

    # # 2. IQR Rule 기반 이상치 제거
    # def remove_outliers(df, column):
    #     Q1 = df[column].quantile(0.25)
    #     Q3 = df[column].quantile(0.75)
    #     IQR = Q3 - Q1
    #     lower_bound = Q1 - 1.5 * IQR
    #     upper_bound = Q3 + 1.5 * IQR
    #     return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    # df_final = remove_outliers(df_clean, 'Quantity')
    # df_final = remove_outliers(df_final, 'UnitPrice')

    # print("\n--- 제거 후 통계량 ---")
    # print(df_final[['Quantity', 'UnitPrice']].describe())
    ```
    - **통계적 결과**: 이상치 제거 후 `describe()` 결과를 보면, `Quantity`와 `UnitPrice`의 `max` 값이 크게 줄어들고, `mean`과 `std`가 변하여 데이터가 더 중앙에 밀집된 형태로 변했음을 통계적으로 확인할 수 있습니다.

### 1-3. K-means, DBSCAN을 이용한 군집 생성

- **데이터 준비 (RFM 특성 생성)**
    - 군집분석을 위해 고객별 행동 특성을 요약하는 RFM(Recency, Frequency, Monetary) 지표를 생성합니다.
    - **Recency**: 최근성. (분석 시점 - 마지막 구매일)
    - **Frequency**: 구매 빈도. (고유한 거래 횟수)
    - **Monetary**: 총 구매액. (`Quantity * UnitPrice`의 합)
    - 생성된 RFM 특성은 분포가 매우 치우쳐 있으므로 로그 변환(`np.log1p`) 후 `StandardScaler`로 표준화하여 사용합니다.

- **사용 가능한 분석 방법**
    - **K-Means**: 거리 기반의 군집분석. 사전에 군집 수(K)를 지정해야 합니다.
    - **DBSCAN**: 밀도 기반의 군집분석. 군집 수를 지정할 필요 없지만, 밀도 파라미터(`eps`, `min_samples`)를 설정해야 합니다. 노이즈(어떤 군집에도 속하지 않는 점)를 구분해내는 장점이 있습니다.

- **현재 문제에 관한 풀이 방법**
    ```python
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans, DBSCAN
    import numpy as np

    # # RFM 특성 생성 (1-2의 df_final 데이터 사용 가정)
    # df_final['InvoiceDate'] = pd.to_datetime(df_final['InvoiceDate'])
    # snapshot_date = df_final['InvoiceDate'].max() + pd.Timedelta(days=1)
    # df_final['TotalPrice'] = df_final['Quantity'] * df_final['UnitPrice']

    # rfm = df_final.groupby('CustomerID').agg({
    #     'InvoiceDate': lambda date: (snapshot_date - date.max()).days, # Recency
    #     'InvoiceNo': 'nunique', # Frequency
    #     'TotalPrice': 'sum' # Monetary
    # }).rename(columns={'InvoiceDate': 'R', 'InvoiceNo': 'F', 'TotalPrice': 'M'})

    # # 로그 변환 및 스케일링
    # rfm_log = np.log1p(rfm)
    # rfm_scaled = StandardScaler().fit_transform(rfm_log)

    # # 1. K-Means 군집 생성 (K=4 가정)
    # kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    # rfm['K_Cluster'] = kmeans.fit_predict(rfm_scaled)

    # # 2. DBSCAN 군집 생성
    # # eps는 NearestNeighbors를 이용해 k-distance plot을 그려 찾는 것이 일반적
    # dbscan = DBSCAN(eps=0.5, min_samples=5)
    # rfm['D_Cluster'] = dbscan.fit_predict(rfm_scaled)
    # # DBSCAN 결과에서 -1은 노이즈(이상치)를 의미
    ```

---

### 2-1. 군집 특성 분석 (성능 지표, 군집 간 차이)

- **군집 성능 지표**
    - **실루엣 계수 (Silhouette Score)**: 군집의 품질을 정량적으로 평가하는 지표. 군집 내 데이터의 응집도와 군집 간 분리도를 함께 고려하며, 1에 가까울수록 군집화가 잘 되었음을 의미합니다.

- **군집 간 차이와 특성**
    - `groupby()`를 사용하여 각 군집별 RFM 값의 평균을 계산하고 비교합니다. 이를 통해 각 군집이 어떤 고객 특성을 나타내는지 해석합니다.

- **현재 문제에 관한 풀이 방법**
    ```python
    from sklearn.metrics import silhouette_score

    # # K-Means 군집 성능 평가
    # kmeans_score = silhouette_score(rfm_scaled, rfm['K_Cluster'])
    # print(f"K-Means Silhouette Score: {kmeans_score:.3f}")

    # # DBSCAN 군집 성능 평가 (노이즈(-1) 제외)
    # dbscan_filtered_rfm = rfm[rfm['D_Cluster'] != -1]
    # dbscan_score = silhouette_score(rfm_scaled[rfm['D_Cluster'] != -1], dbscan_filtered_rfm['D_Cluster'])
    # print(f"DBSCAN Silhouette Score: {dbscan_score:.3f}")

    # # K-Means 군집별 특성 분석
    # print("\nK-Means 군집별 특성 (RFM 평균):")
    # print(rfm.groupby('K_Cluster')[['R', 'F', 'M']].mean())
    ```
    - **결과 해석**: 실루엣 점수를 통해 어떤 군집 모델이 더 적합했는지 평가합니다. `groupby` 결과를 보고 각 군집의 이름을 부여합니다. 예를 들어, R(최근성)이 낮고 F(빈도), M(금액)이 높은 군집은 'VIP 고객', R이 높고 F, M이 낮은 군집은 '이탈 고객' 등으로 명명하고 그 특성을 서술합니다.

### 2-2. 각 군집 별 대표 추천 상품 도출

- **분석 방법**: 각 군집에 속한 고객들이 가장 많이 구매한 상품을 찾습니다. 이는 군집의 소비 성향을 대표하는 상품으로 볼 수 있습니다.

- **현재 문제에 관한 풀이 방법**
    1.  원본 데이터(`df_final`)에 고객별 군집 라벨(`K_Cluster`)을 병합합니다.
    2.  `groupby()`를 사용하여 군집별로 어떤 `Description`(상품명)이 가장 많이 등장했는지(`value_counts()`) 확인합니다.
    3.  군집별 상위 N개 상품을 '대표 추천 상품'으로 제시합니다.
    4.  **타 군집과의 차이**: VIP 고객 군집은 고가의 장식품이나 특정 카테고리 상품을 주로 구매하는 반면, 신규 고객 군집은 저렴하고 대중적인 상품을 구매하는 경향을 보일 수 있습니다. 이처럼 군집별 상위 상품 리스트의 차이점을 비교하여 설명합니다.

    ```python
    # # 군집 라벨을 원본 데이터에 병합
    # df_merged = df_final.merge(rfm[['K_Cluster']], on='CustomerID')

    # # 군집별 대표 상품 Top 5 추출
    # for i in sorted(df_merged['K_Cluster'].unique()):
    #     print(f"--- Cluster {i} Top 5 Products ---")
    #     top_products = df_merged[df_merged['K_Cluster'] == i]['Description'].value_counts().head(5)
    #     print(top_products)
    #     print("\n")
    ```

### 2-3. 특정 고객(12413) 대상 상품 추천

- **분석 방법**: **사용자 기반 협업 필터링 (User-Based Collaborative Filtering)**의 원리를 KNN(K-Nearest Neighbors)을 이용하여 구현합니다.
    1.  **타겟 고객과 유사한 고객 찾기**: RFM 특성 공간에서 타겟 고객(12413)과 가장 가까운 K명의 이웃 고객을 찾습니다.
    2.  **추천 후보 생성**: 이웃 고객들이 구매했지만, 타겟 고객은 아직 구매하지 않은 상품 목록을 만듭니다.
    3.  **추천**: 후보 상품들 중에서 이웃들이 가장 많이 구매한 순서대로 상품을 추천합니다.

- **현재 문제에 관한 풀이 방법**
    ```python
    from sklearn.neighbors import NearestNeighbors

    # # KNN 모델 학습
    # knn = NearestNeighbors(n_neighbors=11, metric='euclidean') # 10명의 이웃 + 자기 자신
    # knn.fit(rfm_scaled)

    # # 타겟 고객(12413)의 데이터 추출
    # target_customer_id = 12413
    # target_customer_data = rfm_log.loc[target_customer_id].values.reshape(1, -1)
    # target_customer_scaled = StandardScaler().fit(rfm_log).transform(target_customer_data)

    # # 가장 가까운 이웃 찾기
    # distances, indices = knn.kneighbors(target_customer_scaled)
    # neighbor_indices = indices.flatten()[1:] # 자기 자신 제외
    # neighbor_customer_ids = rfm.iloc[neighbor_indices].index

    # # 이웃들이 구매한 상품 목록
    # neighbor_purchases = df_final[df_final['CustomerID'].isin(neighbor_customer_ids)]['Description'].unique()

    # # 타겟 고객이 구매한 상품 목록
    # target_purchases = df_final[df_final['CustomerID'] == target_customer_id]['Description'].unique()

    # # 추천 상품 도출 (이웃은 샀지만 나는 안 산 상품)
    # recommendations = [item for item in neighbor_purchases if item not in target_purchases]

    # print(f"CustomerID {target_customer_id}에게 추천하는 상품 (상위 10개):")
    # print(recommendations[:10])
    ```

---

## 통계 분석 (50점)

### 3번 문제: 표본 크기 계산

- **분석 방법**: 비율 추정을 위한 표본 크기 산출 공식을 사용합니다.
  - $n = \frac{Z^2 \cdot p(1-p)}{E^2}$
- **풀이**:
    - **신뢰수준 90%**: $Z_{\alpha/2} = Z_{0.05} = 1.645$
    - **추정오차한계(E)**: 0.05 (5%)
    - **불량률(p)**: 모비율을 모를 때, 표본 크기를 최대로 하는 $p=0.5$를 사용합니다.
    - $n = \frac{1.645^2 \cdot 0.5(1-0.5)}{0.05^2} = \frac{2.706 \cdot 0.25}{0.0025} = 270.6$
    - 표본의 크기는 정수여야 하므로, 올림하여 **271**이 최소값입니다.

### 4번 문제: 시계열 그래프 및 변화율 계산

**참고**: 데이터가 없어 가상 데이터를 생성하여 풀이합니다.

- **4-1. 시계열 그래프**
    ```python
    # # 가상 데이터 생성
    # months = pd.date_range(start='2023-01-01', periods=9, freq='M')
    # prices = [23.1, 22.5, 24.0, 25.1, 24.8, 23.9, 24.5, 24.9, 25.5]
    # silver = pd.Series(prices, index=months)

    # # 이동평균 계산
    # moving_avg = silver.rolling(window=3).mean()

    # # 시각화
    # plt.figure(figsize=(10, 6))
    # plt.plot(silver.index, silver, marker='o', label='Silver Price')
    # plt.plot(moving_avg.index, moving_avg, marker='x', linestyle='--', label='3-Month MA')
    # plt.title('Silver Price Trend')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    ```
- **4-2. 변화율 계산**
    - **풀이**: $(\frac{9월 가격 - 1월 가격}{1월 가격}) \times 100$
    ```python
    # # 변화율 계산
    # percentage_change = ((prices[-1] - prices[0]) / prices[0]) * 100
    # print(f"1월 대비 9월 가격 변화율: {percentage_change:.1f}%")
    ```

### 5번 문제: 동질성 검정 (카이제곱 검정)

- **5-1. 가설 설정**
    - **귀무가설(H0)**: 세 선거구의 지지율은 모두 동일하다. ($p_1 = p_2 = p_3$)
    - **연구가설(H1)**: 적어도 한 선거구의 지지율은 다르다.

- **5-2. 가설 검증**
    - **분석 방법**: 세 집단 이상의 비율 차이를 검정하므로 **카이제곱 동질성 검정**을 사용합니다.
    - **풀이**: `scipy.stats.chi2_contingency` 함수를 사용하면 검정통계량과 p-value를 쉽게 계산할 수 있습니다.

    ```python
    from scipy.stats import chi2_contingency

    # 데이터 (관측도수)
    observed = pd.DataFrame([[176, 193, 159], [124, 107, 141]], 
                              index=['지지함', '지지하지 않음'], 
                              columns=['선거구1', '선거구2', '선거구3'])

    chi2_stat, p_value, df, expected = chi2_contingency(observed)

    print(f"검정통계량: {chi2_stat:.3f}")
    print(f"p-value: {p_value:.4f}")

    if p_value < 0.05:
        print("연구가설 채택: p-value가 0.05보다 작으므로, 선거구별 지지율에 유의미한 차이가 있다고 할 수 있습니다.")
    else:
        print("귀무가설 채택: p-value가 0.05보다 크므로, 선거구별 지지율이 동일하다고 할 수 있습니다.")
    ```

### 6번 문제: 독립표본 t-검정

- **6-1. 가설 설정**
    - **귀무가설(H0)**: 남학생과 여학생의 평균 혈압에 차이가 없다. ($\mu_{남} = \mu_{여}$)
    - **연구가설(H1)**: 남학생과 여학생의 평균 혈압에 차이가 있다. ($\mu_{남} \neq \mu_{여}$)

- **6-2. 가설 검증**
    - **분석 방법**: 두 독립적인 집단의 평균을 비교하고, 등분산이 가정되었으므로 **독립표본 t-검정(Independent Two-sample t-test)**을 사용합니다.
    - **풀이**: `scipy.stats.ttest_ind` 함수에 `equal_var=True` 옵션을 주어 계산합니다.

    ```python
    from scipy.stats import ttest_ind

    male = [124.97, 118.62, 126.48, 135.23, 117.66, 117.66, 135.79, 127.67, 115.31, 125.43, 115.37, 115.34, 122.42, 100.87, 102.75, 114.38]
    female = [114.87, 128.14, 115.92, 110.88, 139.66, 122.74, 125.68, 110.75, 119.56]

    t_stat, p_value = ttest_ind(male, female, equal_var=True)

    print(f"검정통계량: {t_stat:.3f}")
    print(f"p-value: {p_value:.4f}")

    if p_value < 0.05:
        print("연구가설 채택: p-value가 0.05보다 작으므로, 남녀 학생 간 평균 혈압에 유의미한 차이가 있습니다.")
    else:
        print("귀무가설 채택: p-value가 0.05보다 크므로, 평균 혈압에 차이가 있다고 할 수 없습니다.")
    ```

- **6-3. 신뢰구간 계산 및 해석**
    - **분석 방법**: 두 평균의 차이에 대한 95% 신뢰구간을 계산합니다. 공식: $(\bar{x}_1 - \bar{x}_2) \pm t_{\alpha/2, df} \cdot SE_{diff}$
    - **풀이**:
        1.  합동분산($s_p^2$)을 구합니다.
        2.  표준오차($SE_{diff}$)를 구합니다.
        3.  신뢰구간을 계산합니다. (자유도 $df = n_1 + n_2 - 2 = 16+9-2=23$)

    - **결과 지지 설명**: 만약 6-2에서 귀무가설을 기각했다면(차이가 있다면), 계산된 95% 신뢰구간은 0을 포함하지 않을 것입니다. 신뢰구간이 0을 포함하지 않는다는 것은 "두 평균의 차이가 0일 가능성이 95% 신뢰수준에서 매우 낮다"는 의미이므로, 두 평균이 다르다는 연구가설을 지지하게 됩니다. 반대로, 신뢰구간이 0을 포함한다면 귀무가설을 기각할 수 없다는 결론을 지지합니다.

### 7번 문제: 베이지안 회귀분석

**참고**: `pymc` 라이브러리가 필요하며, MCMC 시뮬레이션은 시간이 다소 소요될 수 있습니다. 데이터가 없어 가상 데이터를 생성하여 풀이합니다.

- **7-1. 회귀계수 추정**
    ```python
    # !pip install pymc
    import pymc as pm
    import numpy as np
    import pandas as pd

    # # 가상 데이터 생성
    # np.random.seed(2023)
    # n = 411
    # height = np.random.normal(175, 5, n)
    # waist = np.random.normal(85, 7, n)
    # weight = 0.8 * height + 0.4 * waist - 100 + np.random.normal(0, 5, n)
    # data = pd.DataFrame({'height': height, 'waist': waist, 'weight': weight})

    # with pm.Model() as bayesian_model:
    #     # 사전 분포 정의
    #     beta0 = pm.Flat('beta0') # 부적절한 균일분포
    #     beta1 = pm.Flat('beta1')
    #     beta2 = pm.Flat('beta2')
    #     sigma = pm.InverseGamma('sigma', alpha=0.0005, beta=0.0005)

    #     # 선형 모델 정의
    #     mu = beta0 + beta1 * data['height'] + beta2 * data['waist']

    #     # 가능도 함수 정의
    #     y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=data['weight'])

    #     # MCMC 시뮬레이션
    #     np.random.seed(2023)
    #     trace = pm.sample(draws=10000, tune=1000, cores=1)

    # # 결과 요약 (회귀계수 평균)
    # summary = pm.summary(trace)
    # print(summary)
    # beta0_mean = summary.loc['beta0', 'mean']
    # beta1_mean = summary.loc['beta1', 'mean']
    # beta2_mean = summary.loc['beta2', 'mean']
    ```

- **7-2. 체중 추정**
    - **풀이**: 7-1에서 구한 각 회귀계수의 사후분포 평균값을 사용하여 식에 대입합니다.
    - `추정 체중 = beta0_mean + beta1_mean * 180 + beta2_mean * 80`

    ```python
    # # 추정값 계산
    # estimated_weight = beta0_mean + beta1_mean * 180 + beta2_mean * 80
    # print(f"\n키 180cm, 허리둘레 80cm인 남성의 추정 체중: {estimated_weight:.2f} kg")
    ```
