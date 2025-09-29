### 일반화 선형 모델 (Generalized Linear Model, GLM)

#### 개념 요약
일반화 선형 모델(GLM)은 일반적인 선형 회귀(OLS)의 가정을 완화하여, 정규분포가 아닌 다양한 확률분포를 따르는 종속변수를 모델링할 수 있도록 확장한 프레임워크입니다. OLS는 종속변수가 정규분포를 따르고, 독립변수와 선형 관계를 가진다고 가정하지만, 현실의 데이터는 이러한 가정을 만족하지 않는 경우가 많습니다. (e.g., 개수 데이터, 비율 데이터, 생존 시간 데이터 등)

GLM은 다음 세 가지 요소로 구성됩니다:
1.  **확률 성분 (Random Component)**: 종속변수 $y$가 따르는 확률분포를 지정합니다. (e.g., 정규분포, 푸아송분포, 이항분포, 감마분포 등)
2.  **체계적 성분 (Systematic Component)**: 독립변수들의 선형 결합($\eta = \beta_0 + \beta_1x_1 + ...$)을 정의합니다. 이는 선형 예측자(Linear Predictor)라고 불립니다.
3.  **연결 함수 (Link Function)**: 확률 성분(종속변수의 기댓값 $E(y)=\mu$)과 체계적 성분(선형 예측자 $\eta$)을 연결하는 함수 $g(\cdot)$입니다. 즉, $g(\mu) = \eta$ 입니다. 연결 함수를 통해 종속변수의 제약조건(e.g., 개수는 항상 0 이상)을 만족시키면서 선형 모델을 적용할 수 있습니다.

#### 적용 가능한 상황
- **푸아송 회귀 (Poisson Regression)**: 종속변수가 특정 시간이나 공간 내에서 발생하는 사건의 횟수(count)일 때 사용합니다. (e.g., 하루 동안 웹사이트 방문자 수, 한 달간 특정 지역의 교통사고 건수)
- **감마 회귀 (Gamma Regression)**: 종속변수가 양의 값을 가지며 오른쪽으로 꼬리가 긴 분포(right-skewed)를 따를 때 사용합니다. (e.g., 보험 청구액, 병원 입원 기간, 강우량)
- **음이항 회귀 (Negative Binomial Regression)**: 푸아송 회귀와 같이 카운트 데이터를 다루지만, 데이터의 분산이 평균보다 큰 '과산포(Overdispersion)' 현상이 나타날 때 사용합니다. 푸아송 회귀는 평균과 분산이 같다고 가정하지만, 실제 데이터는 이 가정을 위배하는 경우가 많아 음이항 회귀가 더 적합할 수 있습니다.
- 이 외에도 **로지스틱 회귀**는 종속변수가 이항분포를 따르는 GLM의 한 종류입니다.

#### 구현 방법
GLM은 `statsmodels` 라이브러리를 통해 매우 효과적으로 구현할 수 있습니다. `sm.GLM` 클래스를 사용하며, `family` 인자를 통해 확률분포와 연결 함수를 지정합니다.

##### 용도
- 정규분포 가정을 만족하지 않는 다양한 유형의 종속변수(카운트, 비율, 양의 연속형 등)를 모델링합니다.

##### 주의사항
- **적절한 `family` 선택**: 데이터의 특성에 맞는 확률분포(family)를 선택하는 것이 매우 중요합니다. 잘못된 분포를 선택하면 모델의 해석과 예측 성능이 저하됩니다.
- **과산포 (Overdispersion)**: 푸아송 회귀의 경우, 분산이 평균보다 유의미하게 큰 과산포가 있는지 확인해야 합니다. 과산포가 존재하면 표준오차가 과소추정되어 변수의 유의성이 부풀려질 수 있으므로, 음이항 회귀를 대안으로 고려해야 합니다.
- **노출 (Exposure)**: 카운트 데이터 분석 시, 관찰 기간이나 공간의 크기가 다르다면 이를 '노출' 변수로 처리하여 모델에 반영해야 합니다. (e.g., `exposure` 인자 사용)

##### 코드 예시 (`statsmodels`)

**1. 푸아송 회귀 (Poisson Regression)**
- **상황**: 어떤 상점의 시간대별 고객 방문 횟수를 예측.
```python
import statsmodels.api as sm
import pandas as pd
import numpy as np

# 1. 데이터 생성
np.random.seed(42)
X = pd.DataFrame({
    'hour': np.arange(8, 22).repeat(10), # 오전 8시 ~ 오후 9시
    'is_weekend': np.random.randint(0, 2, 140)
})
# 시간에 따라 방문자 수가 변하고, 주말에 더 많다고 가정
lambda_val = np.exp(0.5 + 0.1 * (X['hour'] - 8) + 0.5 * X['is_weekend'])
y = np.random.poisson(lambda_val)

X_const = sm.add_constant(X)

# 2. 모델 학습
# family=sm.families.Poisson(): 푸아송 분포와 로그 연결 함수(기본값)를 사용
poisson_model = sm.GLM(y, X_const, family=sm.families.Poisson())
poisson_results = poisson_model.fit()

print("--- 푸아송 회귀 결과 ---")
print(poisson_results.summary())

# 결과 해석: coef는 로그 스케일에서의 변화량. np.exp(coef)를 통해 해석.
# e.g., is_weekend의 계수가 0.48이면, 주말일 때 평일보다 고객 수가 exp(0.48) \u2248 1.62배 많아짐을 의미.
```

**2. 음이항 회귀 (Negative Binomial Regression)**
- **상황**: 푸아송 회귀 모델의 잔차 분석 결과 과산포가 의심될 때.
```python
# 3. 음이항 회귀 모델 학습
# family=sm.families.NegativeBinomial(): 음이항 분포와 로그 연결 함수 사용
neg_binom_model = sm.GLM(y, X_const, family=sm.families.NegativeBinomial())
neg_binom_results = neg_binom_model.fit()

print("\n--- 음이항 회귀 결과 ---")
print(neg_binom_results.summary())

# 결과 해석: summary의 [alpha] 값이 음이항 분포의 과산포 파라미터.
# 이 값이 0에 가깝지 않고 통계적으로 유의하면 과산포가 존재하며, 음이항 모델이 더 적합함을 시사.
```

**3. 감마 회귀 (Gamma Regression)**
- **상황**: 자동차 사고 당 보험 청구액(양수, skewed)을 예측.
```python
# 1. 데이터 생성
np.random.seed(123)
X_gamma = pd.DataFrame({
    'driver_age': np.random.randint(18, 70, 100),
    'car_value': np.random.uniform(500, 50000, 100)
})
# 나이가 적고 차 가격이 비쌀수록 청구액이 높다고 가정
mu = np.exp(10 - 0.02 * X_gamma['driver_age'] + 0.00001 * X_gamma['car_value'])
# 감마 분포는 shape(alpha)와 scale(beta) 파라미터를 가짐. 여기서는 shape=2로 고정.
y_gamma = np.random.gamma(shape=2., scale=mu/2.)

X_gamma_const = sm.add_constant(X_gamma)

# 2. 모델 학습
# family=sm.families.Gamma(link=sm.families.links.log()): 감마 분포와 로그 연결 함수 사용
# 감마 회귀는 역수(inverse) 연결 함수가 기본값이지만, 로그 연결이 해석에 용이할 때가 많음.
gamma_model = sm.GLM(y_gamma, X_gamma_const, family=sm.families.Gamma(link=sm.families.links.log()))
gamma_results = gamma_model.fit()

print("\n--- 감마 회귀 결과 ---")
print(gamma_results.summary())
```

##### 결과 해석 방법
- **`summary()` 결과**: 각 모델의 `summary()`는 OLS와 유사한 형태의 결과를 제공합니다.
    - **`coef`**: 각 독립변수의 회귀 계수입니다. 연결 함수를 거친 스케일에서의 값이므로 해석에 주의해야 합니다. (e.g., 로그 연결 함수면 $e^{coef}$를 취해 배수 효과로 해석)
    - **`P>|z|`**: 계수의 유의성을 판단하는 p-value입니다.
    - **`Deviance`, `Log-Likelihood`**: 모델의 적합도를 나타내는 지표로, 모델 간 비교에 사용됩니다. (e.g., AIC, BIC)
- **과산포 확인**: 푸아송 회귀 후, `poisson_results.pearson_chi2 / poisson_results.df_resid` 값을 계산해볼 수 있습니다. 이 값이 1보다 현저히 크면 과산포를 의심하고 음이항 회귀를 고려합니다.

#### 장단점 및 대안
- **장점**: 
    - 다양한 분포의 종속변수를 하나의 통일된 프레임워크로 모델링할 수 있어 유연성이 매우 높습니다.
    - `statsmodels`를 통해 풍부한 통계적 추론 결과(계수 유의성, 신뢰구간 등)를 얻을 수 있습니다.
- **단점**: 
    - 데이터에 적합한 분포와 연결 함수를 선택해야 하는 어려움이 있습니다.
    - 모델의 가정이 복잡하여 해석이 OLS보다 어려울 수 있습니다.
- **대안**: 
    - **변수 변환 후 OLS**: 종속변수에 로그, 제곱근 등 변환을 적용하여 정규성을 만족시킨 후 OLS를 적용하는 간단한 방법을 시도할 수 있습니다.
    - **준최대가능도(Quasi-Likelihood) 모델**: 종속변수의 정확한 분포를 모르더라도, 평균과 분산의 관계만 가정하여 모델을 추정할 수 있습니다. (e.g., `sm.families.QuasiPoisson`)
    - **머신러닝 모델**: 트리 기반 모델(Random Forest, Gradient Boosting) 등은 분포 가정이 필요 없고 비선형 관계도 잘 다루므로 대안이 될 수 있지만, 통계적 추론보다는 예측에 더 초점을 둡니다.

```