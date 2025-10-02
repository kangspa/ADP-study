# 시계열 데이터 처리

- 시계열 데이터 : 행과 행에 시간의 순서(흐름)이 있고, 시간간격이 동일한 데이터
                          → Sequential Data 에 포함됨
- pd.to_datetime() : object 형태의 날짜 데이터를 datetime 데이터로 바꿔준다.
    - format = ‘ ‘ : 입력하는 날짜의 형태가 어떤 형식인지 알려주는 옵션
                         ‘%Y-%m-%d’ : ‘몇년-몇월-몇일’ 로 작성된지 알려줌
                         → 왠만해서는 혼자서 format 형식을 잡을 수 있
    - 자세한 내용은 아래 링크, 공식 문서에서 알 수 있음
    
    https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
    
- Series.dt.날짜요소 : 날짜 타입의 변수로부터 날짜 요소를 뽑아낼 수 있다.
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/336edc9f-be08-4746-ae72-23d96c4f092a/26cc622c-1a9c-4006-b631-d23394411840/Untitled.png)
    
- .shift() : 시계열 데이터에서 시간의 흐름 전후로 정보를 이동시킬 때 사용
    - 양수 값은 해당 숫자만큼 더해진 날짜에 넣어줌 (default = 1)
    - 음수 값도 가능하며, 그만큼 당겨서 데이터를 저장
    - 이동시키고 생기는 공백 구간은 NaN 값으로 처리 (결측값)
    
    ```python
    # 전날 매출액 열을 추가
    temp['Amt_lag'] = temp['Amt'].shift() #default = 1
    ```
    
- .rolling(n) + 집계함수 : 시간의 흐름에 따라 일정 기간 동안 값을 이동하면서 구하기
    - n = 며칠을 기준으로 삼을지 (default = 1)
    - min_periods : 계산에 사용할 최소 데이터 수
    
    https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html
    
    ```python
    # 7일 이동평균 매출액
    temp['Amt_MA7_1'] = temp['Amt'].rolling(7).mean() # 데이터 상 6일까지는 NaN
    # 6일까지 있는 데이터만 가지고 평균을 구하려면 min_periods 값을 설정
    temp['Amt_MA7_2'] = temp['Amt'].rolling(7, min_periods = 1).mean() # 1일 데이터도 활용
    ```
    
- .diff(n) : 특정 시점 데이터와 이전 시점 데이터의 차이 구하기 (차분 값)
    - n = 며칠 전 데이터와의 차이를 구할 것인지 (default = 1)
    
    https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.diff.html
    
    ```python
    temp['Amt_D1'] = temp['Amt'].diff() # 하루 간 데이터 차이
    temp['Amt_D2'] = temp['Amt'].diff(2) # 이틀 간 데이터 차
    ```
    
- .resample().집계함수() : 시계열 데이터를 ()안에 넣는 시간 단위로 리샘플링한다.
                                      이 때, 기준은 집계함수를 통해서 결정된다.
    - 시간 단위는 아래 공식 링크 참조
    
    https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
    
    ```python
    # 임시로 시간 데이터 생성
    np.random.seed(1)
    
    arr = np.random.randn(365)
    time_idx = pd.date_range('2021-1-1', periods=365, freq='D')
    
    ts = pd.Series(arr, index=time_idx)
    # 일별 데이터를 월별로 변환 (평균값 기준)
    ts.resample('M').mean()
    ```
    
    - resample 뒤에 .ffill 이나 .bfill 메소드를 통해 결측값을 채울 수도 있다.
        
        ```python
        # forward filling 방식
        ts.resample('D').ffill() # ffill: 각 기간의 첫일을 참고하여 결측값 보간
        # backward filling 방식
        ts.resample('D').bfill() # bfill: 각 기간의 마지막일을 참고하여 결측값 보간
        ```

--------------------------------------------------------------

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
