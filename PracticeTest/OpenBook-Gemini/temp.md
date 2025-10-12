# Scipy (Scientific Python)
- Numpy를 기반으로 과학 및 기술 컴퓨팅을 위한 다양한 고급 함수를 제공
- 통계, 최적화, 신호 처리, 선형대수, 이미지 처리 등 폭넓은 분야의 알고리즘을 포함
### 적용 가능한 상황
- 가설 검정, 확률 분포 등 복잡한 통계 분석이 필요할 때 (`scipy.stats`).
- 함수의 최적해(최소/최대값)를 찾아야 할 때 (`scipy.optimize`).
- 미분 방정식 풀이, 수치 적분, 보간 등 공학적 계산이 필요할 때.
### 주의사항
- Scipy는 방대한 하위 모듈로 구성
- 필요한 기능이 어떤 모듈에 있는지(e.g., `scipy.stats`, `scipy.optimize`) 공식 문서를 통해 확인하고 사용하는 것이 좋음
### 코드 예시
  ```python
  from scipy import stats
  from scipy import optimize

  # 1. 통계 (scipy.stats)
  # T-검정 예시: 두 집단의 평균이 통계적으로 유의미하게 다른지 검정
  group1 = [20, 22, 19, 20, 21, 20, 18, 25]
  group2 = [28, 26, 27, 29, 25, 28, 26, 30]

  # 등분산성 검정 (Levene's test)
  levene_stat, levene_p = stats.levene(group1, group2)
  print(f"Levene test p-value: {levene_p:.4f}")
  # p-value가 0.05보다 크면 등분산성 가정 만족

  # 독립표본 T-검정 (Independent Two-sample t-test)
  # equal_var=True (등분산 가정) 또는 False (이분산 가정, Welch's t-test)
  t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=True)
  print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")
  # p-value가 유의수준(e.g., 0.05)보다 작으면, 두 집단의 평균은 유의미하게 다르다고 결론

  # 2. 최적화 (scipy.optimize)
  # 간단한 1차원 함수의 최소값 찾기
  def f(x):
      return x**2 + 10*np.sin(x)

  # 함수 f(x)의 최소값을 x=0 근처에서 찾기 시작
  result = optimize.minimize(f, x0=0)
  print(f"Minimum value found at x = {result.x[0]:.4f}")
  print(f"Function minimum value = {result.fun:.4f}")
  ```
- **T-검정**: P-value가 매우 작게(e.g., 0.0001) 나왔으므로, 두 그룹의 평균 사이에는 통계적으로 유의미한 차이가 있다고 해석할 수 있습니다.
- **최적화**: `minimize` 함수는 주어진 함수 `f(x)`의 값을 최소로 만드는 `x`의 값과 그때의 함수 값을 찾아줍니다. 이 예시에서는 약 -1.3064에서 최소값 -7.9458을 가짐을 보여줍니다.

# 장단점 및 대안

| 라이브러리 | 장점 | 단점 | 대안 |
|---|---|---|---|
| **Scipy** | 통계, 최적화, 신호 처리 등 광범위한 과학 계산 알고리즘 제공, Numpy와 완벽하게 호환 | 기능이 매우 방대하여 학습 곡선이 존재함 | Statsmodels (통계 분석 및 모델링에 더 특화), scikit-learn (머신러닝 알고리즘에 집중) |
