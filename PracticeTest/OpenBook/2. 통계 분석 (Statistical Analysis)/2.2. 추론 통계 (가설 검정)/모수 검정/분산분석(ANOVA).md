# 분산분석 (ANOVA)

## 개념 요약

분산분석(Analysis of Variance, ANOVA)은 **세 개 이상**의 집단 간의 **평균**을 비교하는 데 사용되는 강력한 통계적 방법입니다. 이름은 '분산'분석이지만, 실제 목적은 집단 간 '평균'의 차이를 검정하는 것입니다. T-검정을 여러 번 반복해서 사용하면 1종 오류가 증가하는 문제(다중 비교 문제)가 발생하는데, ANOVA는 이를 해결하기 위해 모든 집단의 평균이 같은지를 한 번에 검정합니다.

ANOVA의 핵심 원리는 데이터의 **총 변동(분산)**을 두 가지 요소로 분해하는 것입니다:
1.  **집단 간 변동 (Between-group variation)**: 각 집단의 평균이 전체 평균으로부터 얼마나 떨어져 있는지를 나타냅니다. 이는 독립변수(요인, factor)의 효과에 의해 발생하는 변동입니다.
2.  **집단 내 변동 (Within-group variation)**: 각 집단 내의 데이터들이 해당 집단의 평균으로부터 얼마나 퍼져 있는지를 나타냅니다. 이는 무작위 오차(random error)에 의해 발생하는 변동입니다.

ANOVA는 이 두 변동의 비율, 즉 **F-통계량**을 계산하여 가설을 검정합니다.
`F = (집단 간 분산) / (집단 내 분산)`

만약 집단 간의 차이(집단 간 분산)가 우연에 의한 변동(집단 내 분산)보다 충분히 크다면, F-값이 커지고 이는 집단 간 평균에 유의미한 차이가 있음을 시사합니다.

**ANOVA의 종류**:
- **일원배치 분산분석 (One-way ANOVA)**: 하나의 독립변수(요인)에 따라 종속변수의 평균이 다른지를 검정합니다. (e.g., 비료 종류(A, B, C)에 따른 식물 키의 차이)
- **이원배치 분산분석 (Two-way ANOVA)**: 두 개의 독립변수와 이들 간의 상호작용(interaction effect)이 종속변수의 평균에 미치는 영향을 검정합니다. (e.g., 비료 종류와 토양 종류에 따른 식물 키의 차이 및 두 요인의 상호작용 효과)

**ANOVA의 기본 가정**:
1.  **독립성**: 각 집단의 표본은 서로 독립적으로 추출되어야 합니다.
2.  **정규성**: 각 집단의 데이터는 정규분포를 따라야 합니다.
3.  **등분산성**: 모든 집단의 분산은 동일해야 합니다.

## 사후 분석 (Post-hoc Analysis)

ANOVA 검정 결과 p-value가 유의수준보다 작아서 귀무가설이 기각되면, 우리는 "적어도 한 집단의 평균은 다르다"는 사실만 알 수 있습니다. 하지만 **어떤 집단들 간에** 차이가 있는지는 알 수 없습니다. 이를 확인하기 위해 수행하는 추가적인 분석을 사후 분석이라고 합니다.

- **Tukey's HSD (Honestly Significant Difference) Test**: 가장 널리 사용되는 사후 분석 방법 중 하나입니다. 모든 가능한 집단 쌍(pair)에 대해 평균 차이를 동시에 검정하면서도, 다중 비교로 인한 1종 오류의 증가를 제어합니다.
- **Bonferroni Correction**: 가장 간단하고 보수적인 방법입니다. 유의수준(α)을 비교하는 쌍의 개수(k)로 나누어(`α/k`), 더 엄격한 기준으로 각 쌍을 t-검정합니다. 1종 오류를 확실히 막지만, 검정력이 낮아져 실제 차이를 놓칠 수 있습니다(2종 오류 증가).

## 구현 방법

### 1. 일원배치 분산분석 (One-way ANOVA)

- **문제**: 세 가지 다른 교육 방법(A, B, C)으로 학생들을 가르친 후, 시험 점수를 비교했다. 교육 방법에 따라 학생들의 평균 점수에 유의미한 차이가 있는가?
- **함수**: `scipy.stats.f_oneway`, `statsmodels.formula.api.ols`, `pingouin.anova`
- **코드 예시**:
  ```python
  import pandas as pd
  from scipy.stats import f_oneway
  from statsmodels.stats.multicomp import pairwise_tukeyhsd

  # 데이터 생성
  method_a = [85, 88, 79, 92, 84]
  method_b = [75, 78, 81, 72, 79]
  method_c = [90, 94, 88, 91, 95]

  # 1. ANOVA 검정 (scipy)
  f_statistic, p_value = f_oneway(method_a, method_b, method_c)
  print("--- One-way ANOVA (scipy) ---")
  print(f"F-statistic: {f_statistic:.4f}")
  print(f"P-value: {p_value:.4f}")

  if p_value < 0.05:
      print("귀무가설 기각: 교육 방법에 따른 평균 점수 차이가 유의미합니다.")
  else:
      print("귀무가설 기각 실패: 평균 점수 차이가 유의미하지 않습니다.")

  # 2. 사후 분석 (Tukey's HSD)
  # 데이터를 long format으로 변환
  df = pd.DataFrame({'score': method_a + method_b + method_c,
                     'group': ['A'] * 5 + ['B'] * 5 + ['C'] * 5})

  tukey_result = pairwise_tukeyhsd(endog=df['score'], groups=df['group'], alpha=0.05)
  print("\n--- Tukey's HSD Post-hoc Test ---")
  print(tukey_result)
  ```
- **결과 해석**:
  - **ANOVA**: p-value(0.0004)가 0.05보다 작으므로, 세 교육 방법 간에 유의미한 평균 점수 차이가 있다고 결론 내립니다.
  - **Tukey's HSD**: 결과 테이블의 `reject` 열을 보면, A-B, A-C, B-C 모든 쌍에서 `True`로 나타납니다. 이는 세 그룹 모두 서로 간에 통계적으로 유의미한 평균 차이가 있음을 의미합니다. `p-adj` 열은 다중 비교를 조정한 p-value를 나타냅니다.

### 2. 이원배치 분산분석 (Two-way ANOVA)

- **문제**: 비료 종류(fertilizer)와 토양 종류(soil)가 식물의 성장(growth)에 미치는 영향을 알아보고자 한다. 각 요인의 주 효과(main effect)와 두 요인 간의 상호작용 효과(interaction effect)를 검정하고 싶다.
- **함수**: `statsmodels.formula.api.ols`, `pingouin.anova`
- **코드 예시**:
  ```python
  import statsmodels.api as sm
  from statsmodels.formula.api import ols
  import pingouin as pg

  # 데이터 생성 (pingouin 내장 데이터 사용)
  df_two_way = pg.read_dataset('anova2')

  # 1. 이원배치 분산분석 (statsmodels)
  # C(Fertilizer): Fertilizer를 범주형 변수로 처리
  # Fertilizer:Soil : 상호작용 항
  model = ols('Yield ~ C(Fertilizer) + C(Soil) + C(Fertilizer):C(Soil)', data=df_two_way).fit()
  anova_table_sm = sm.stats.anova_lm(model, typ=2)
  print("--- Two-way ANOVA (statsmodels) ---")
  print(anova_table_sm)

  # 2. 이원배치 분산분석 (pingouin)
  print("\n--- Two-way ANOVA (pingouin) ---")
  anova_table_pg = pg.anova(data=df_two_way, dv='Yield', between=['Fertilizer', 'Soil'], detailed=True)
  print(anova_table_pg)
  ```
- **결과 해석**: (`statsmodels` 기준)
  - `C(Fertilizer)`: p-value(`PR(>F)`)가 0.05보다 작으므로, 비료 종류에 따른 수확량(Yield)의 차이(주 효과)는 유의미합니다.
  - `C(Soil)`: p-value가 0.05보다 작으므로, 토양 종류에 따른 수확량의 차이(주 효과)도 유의미합니다.
  - `C(Fertilizer):C(Soil)`: p-value가 0.05보다 작으므로, 비료와 토양 종류 간의 **상호작용 효과**도 유의미합니다. 이는 특정 비료의 효과가 토양의 종류에 따라 달라짐을 의미합니다. (e.g., A 비료는 1번 토양에서 효과가 좋지만, 2번 토양에서는 효과가 미미할 수 있음)
  - `pingouin` 라이브러리는 더 간결한 코드와 깔끔한 출력 테이블을 제공하여 편리합니다.
