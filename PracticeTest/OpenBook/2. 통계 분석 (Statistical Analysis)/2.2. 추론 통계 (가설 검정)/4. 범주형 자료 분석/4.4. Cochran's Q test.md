# 코크란의 Q 검정 (Cochran's Q Test)

- **3개 이상의 대응되는(paired) 표본**에서 **이분형(dichotomous) 결과 변수**의 비율(또는 빈도)이 동일한지를 검정하는 비모수적 통계 방법
- 맥니마 검정(McNemar's Test)을 3개 이상의 집단으로 확장한 것으로, 반복 측정된 분산분석(Repeated Measures ANOVA)의 비모수 버전이라고 볼 수 있음
- 동일한 참가자들을 대상으로 **세 가지 다른 조건(또는 시점)**에서 '성공/실패', '예/아니오'와 같은 **이분형 응답을 수집**했을 때, **각 조건에 따른 성공률에 차이**가 있는지를 분석하는 데 사용
- 검정 통계량 Q는 카이제곱 분포를 따르며, 이를 이용해 가설을 검정

- **귀무가설 (H0)**: 모든 조건(또는 시점)에서의 성공 확률(비율)은 동일하다.
- **대립가설 (H1)**: 적어도 하나의 조건(또는 시점)에서의 성공 확률(비율)은 다른 조건과 다르다.

### 적용 가능한 상황

- **3개 이상의 대응 표본**: 동일한 대상에 대해 3회 이상 반복 측정된 데이터.
- **이분형 변수**: 결과가 '성공/실패', '1/0', '합격/불합격' 등 두 가지 범주로만 구성된 경우.
- 예시:
    - 한 그룹의 학생들이 세 번의 다른 시험(예: 중간, 기말, 재시험)에서 합격/불합격한 비율에 차이가 있는지 비교.
    - 동일한 환자들이 세 가지 다른 치료법(A, B, C)을 받았을 때, 각 치료법에 대한 반응(효과 있음/없음) 비율이 다른지 분석.
    - 동일한 참가자들이 세 가지 다른 광고를 보고 제품을 구매/비구매한 비율이 동일한지 검정.

### 구현 방법

`statsmodels.stats.contingency_tables.cochrans_q` 함수를 사용하여 코크란의 Q 검정을 수행할 수 있습니다.

### 주의사항 (가정)

- **대응 표본**: 데이터는 반드시 동일한 개체로부터 반복 측정된 값이어야 합니다.
- **이분형 데이터**: 데이터는 0과 1 (또는 두 개의 다른 값)으로 코딩된 이분형 변수여야 합니다.
- **표본 크기**: 표본 크기(참가자 수)가 너무 작으면 검정 결과의 신뢰도가 떨어질 수 있습니다. 명확한 기준은 없지만, 일반적으로 참가자 수가 충분히 확보되어야 합니다.

## 코드 예시

`statsmodels.stats.contingency_tables.cochrans_q(x)`

**하이퍼파라미터 (인자) 설명**

- `x`: array_like (2-D). 데이터를 `(참가자 수, 반복 측정 수)` 형태의 2차원 배열로 전달합니다. 각 셀의 값은 0 또는 1과 같은 이분형 데이터여야 합니다.

```python
import numpy as np
from statsmodels.stats.contingency_tables import cochrans_q

# 예시: 10명의 학생이 세 번의 시험(Test1, Test2, Test3)에서 합격(1) 또는 불합격(0)한 결과
# 귀무가설: 세 시험의 합격률은 모두 동일하다.
# 대립가설: 적어도 한 시험의 합격률은 다른 시험과 다르다.

data = np.array([
#   T1, T2, T3
    [1, 1, 0],  # 학생 1
    [1, 1, 1],  # 학생 2
    [0, 1, 1],  # 학생 3
    [0, 0, 0],  # 학생 4
    [1, 1, 0],  # 학생 5
    [0, 1, 0],  # 학생 6
    [1, 0, 0],  # 학생 7
    [1, 1, 1],  # 학생 8
    [0, 1, 0],  # 학생 9
    [0, 1, 1]   # 학생 10
])

# 코크란의 Q 검정 수행
result = cochrans_q(data)

print(f"Cochran's Q statistic: {result.statistic:.4f}") # 3.7143
print(f"P-value: {result.pvalue:.4f}")                  # 0.1561

# 결과 해석: "귀무가설 채택: 세 시험의 합격률은 통계적으로 차이가 없다고 볼 수 있습니다."
alpha = 0.05
if result.pvalue < alpha:
    print("귀무가설 기각: 세 시험의 합격률에는 통계적으로 유의미한 차이가 있습니다.")
else:
    print("귀무가설 채택: 세 시험의 합격률은 통계적으로 차이가 없다고 볼 수 있습니다.")

# 사후 분석 (Post-hoc Test)
# 만약 귀무가설이 기각되었다면, 어느 쌍(예: Test1-Test2, Test1-Test3, Test2-Test3)에서
# 차이가 나는지 확인하기 위해 사후 분석을 수행해야 합니다.
# 일반적으로 각 쌍에 대해 맥니마 검정(McNemar's Test)을 수행하고,
# 다중 비교에 따른 1종 오류 증가를 보정하기 위해 본페로니 교정(Bonferroni correction) 등을 적용합니다.

from statsmodels.stats.contingency_tables import mcnemar
import pandas as pd
import itertools

test_names = ['Test1', 'Test2', 'Test3']

# 사후 분석: 모든 쌍에 대해 McNemar 수행
results = []
for i, j in itertools.combinations(range(data.shape[1]), 2):
    # 2x2 교차표 생성
    table = pd.crosstab(data[:, i], data[:, j])
    # McNemar 검정 수행 (exact=False는 chi-square 근사)
    result = mcnemar(table, exact=False, correction=True)
    results.append({
        'comparison': f'{test_names[i]} vs {test_names[j]}',
        'statistic': result.statistic,
        'p_uncorrected': result.pvalue
    })

# Bonferroni 보정
p_values = [r['p_uncorrected'] for r in results]
p_adjusted = np.minimum(np.array(p_values) * len(p_values), 1.0)
for idx, r in enumerate(results):
    r['p_adjusted'] = p_adjusted[idx]
    r['reject'] = p_adjusted[idx] < alpha

# 결과 출력
posthoc_df = pd.DataFrame(results)
print("\nPost-hoc test (McNemar with Bonferroni correction):")
print(posthoc_df)
'''
Post-hoc test (McNemar with Bonferroni correction):
       comparison  statistic  p_uncorrected  p_adjusted  reject
0  Test1 vs Test2       0.80       0.371093    1.000000   False
1  Test1 vs Test3       0.00       1.000000    1.000000   False
2  Test2 vs Test3       2.25       0.133614    0.400843   False
'''
```

### 결과 해석 방법

- **Q statistic**: 코크란의 Q 검정 통계량. 이 값이 클수록 조건 간의 비율 차이가 크다는 것을 의미합니다.
- **P-value**: 귀무가설(모든 조건의 비율이 동일하다)이 참일 때, 현재와 같은 검정 통계량 또는 더 극단적인 값이 나올 확률입니다.
    - `p-value < 유의수준`: 귀무가설을 기각합니다. 즉, 적어도 하나의 조건(또는 시점)에서의 비율이 다른 조건들과 다르다고 결론 내릴 수 있습니다.

**사후 분석 결과 해석**

사후 분석 결과로 나온 행렬은 각 조건 쌍 간의 p-value를 보여줍니다.<br>
이 p-value를 유의수준과 비교하여 특정 두 조건 간에 유의미한 비율 차이가 있는지를 판단할 수 있습니다.<br>
예를 들어, `Test1`과 `Test3` 사이의 p-value가 0.05보다 작다면, 두 시험의 합격률은 통계적으로 유의미하게 다르다고 해석할 수 있습니다.

## 장단점 및 대안

### 장점

- **맥니마 검정의 확장**: 3개 이상의 대응 표본에 대한 이분형 데이터 분석이 가능합니다.
- **비모수적 방법**: 데이터의 정규성 등 분포에 대한 가정이 필요 없습니다.

### 단점

- **구체성 부족**: 검정 결과가 유의미하더라도, 어느 조건 쌍에서 차이가 발생하는지는 알려주지 않습니다. 이를 위해 별도의 사후 분석이 필수적입니다.
- **이분형 데이터에만 적용**: 결과 변수가 반드시 두 개의 범주로 구성되어야 합니다. 3개 이상의 범주를 가진 명목형 또는 순서형 데이터에는 적용할 수 없습니다.

### 대안

- **반복 측정 분산분석 (Repeated Measures ANOVA)**: 결과 변수가 **연속형**이고 정규성 등 ANOVA의 가정을 만족할 때 사용합니다.
- **프리드만 검정 (Friedman Test)**: 결과 변수가 **순위(ordinal)** 또는 정규성을 만족하지 않는 연속형 변수일 때 사용하는 비모수적 방법입니다. 코크란 Q 검정은 이분형 데이터에 대한 프리드만 검정의 특별한 경우로 볼 수 있습니다.
- **로지스틱 회귀 (Logistic Regression) / 일반화 추정 방정식 (GEE)**: 더 복잡한 모델링이 필요할 때 (예: 공변량 통제) 사용할 수 있는 고급 대안입니다. 특히 GEE는 반복 측정된 범주형 데이터를 모델링하는 데 매우 유용합니다.
