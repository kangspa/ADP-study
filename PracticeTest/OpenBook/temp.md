- np.zeros( ) : 0으로 채워진 배열
- np.ones( ) : 1로 채워진 배열
- np.full( ) : 특정값으로 채워진 배열
- np.eye( ) : 정방향행렬
- np.random.random( ) : 랜덤값으로채운배열
- np.arange(start, stop, step) : 시작값, 끝값과 몇 씩 숫자가 넘어갈지 정해준다(range와 비슷)
```python
# 0으로 채워진 배열
a = np.zeros((2, 2))
'''
[[ 0. 0.]
 [ 0. 0.]]
'''
# 1로 채워진 배열
b = np.ones((1, 2))
# [[ 1. 1.]]

# 특정 값으로 채워진 배열
c = np.full((2, 2), 7.)
'''
[[ 7. 7.]
 [ 7. 7.]]
'''
# 2x2 단위 행렬(identity matrix)
d = np.eye(2)
'''
[[ 1. 0.]
 [ 0. 1.]]
'''
# 랜덤 값으로 채운 배열
e = np.random.random((2, 2))
'''
[[0.17458815, 0.65392064],
 [0.62288583, 0.25213172]]
'''
# 연속된 값으로 배열을 만들 때
f = np.arange(1, 10)
'''
[ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
'''
g = np.arange(1, 10, 2)
'''
[1, 3, 5, 7, 9]
'''
```

### Numpy Array 정보 확인

- .ndim : rank가 몇인지 보여줌(몇 차원인지)
- .shape : shape이 어떻게 되는지
- .dtype : 배열 내의 데이터 형식이 어떻게 되는지 (요소들의 데이터 형식)

- .reshape() : 배열을 다른 shape으로 바꿔줌, 요소만 안 사라진다면 얼마든지 변형 가능

## 인덱싱

- 1차원 배열은 리스트와 방법이 같으므로 설명을 생략합니다.
- **배열[행, 열]** 형태로 특정 위치의 요소를 조회합니다.
- **배열[[행1,행2,..], :]** 또는 **배열[[행1,행2,..]]** 형태로 특정 행을 조회합니다.
- **배열[:, [열1,열2,...]]** 형태로 특정 열을 조회합니다.
- **배열[[행1,행2,...], [열1,열2,...]]** 형태로 특정 행의 특정 열을 조회합니다.

```python
# (3, 3) 형태의 2차원 배열 만들기
a = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
# 요소 조회
print(a[0, 1]) # 2
print(a[0][1]) # 2
# 행 조회
print(a[[0, 1]])
print(a[[0, 1], :])
'''
[[1 2 3]
 [4 5 6]]
'''
# 열 조회
print(a[:, [0, 1]])
'''
[[1 2]
 [4 5]
 [7 8]]
'''
# 행, 열 조회

# 두 번째 행 두 번째 열의 요소 조회
print(a[[1], [1]]) # [5]
# 세 번째 행 두 번째 열의 요소 조회
print(a[[2], [1]]) # [8]

# [0, 0], [1, 1], [2, 2] 조회 -> [행][열]로 작성
print(a[[0, 1, 2], [0, 1, 2]])
'''
[1 5 9]
'''
```

## 슬라이싱

- **배열[행1:행N,열1:열N]** 형태로 지정해 그 위치의 요소를 조회합니다.
- 조회 결과는 **2차원 배열**이 됩니다.
- 마지막 **범위 값은 대상에 포함되지 않습니다.**
- 즉, **배열[1:M, 2:N]**이라면 1 ~ M-1행, 2 ~ N-1열이 조회 대상이 됩니다.

```python
# (3, 3) 형태의 2차원 배열 만들기
a = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
# 첫 번째 ~ 두 번째 행 조회
print(a[0:2])
print(a[0:2, :])
'''
[[1 2 3]
 [4 5 6]]
'''
# 첫 번째 ~ 두 번째 열을 조회
print(a[:, 0:2])
'''
[[1 2]
 [4 5]
 [7 8]]
'''
# 첫 번째 행, 첫 번째 ~ 두 번째 열 조회
print(a[0, 0:2])
'''
[1 2]
'''
# 결과를 2차원으로 받고 싶을 때
print(a[[0], 0:2])
print(a[0:1, 0:2])
'''
[[1 2]]
'''
# 첫 번째 ~ 세 번째 행, 두 번째 ~ 세 번째 열 조회
print(a[0:3, 1:3])
'''
[[2 3]
 [5 6]
 [8 9]]
'''
# 두 번째 ~ 끝 행, 두 번째 ~ 끝 열 조회
print(a[1:, 1:])
'''
[[5 6]
 [8 9]]
'''
# [0, 1] 위치부터 [1,2] 위치까지 요소 조희
print(a[0:2, 1:3])
print(a[0:2, 1:])
'''
[[2 3]
 [5 6]]
'''
```

## 조건 조회

- **조건에 맞는 요소를 선택**하는 방식이며, **불리안 방식(boolean)**이라고 부릅니다.
- 조회 결과는 **1차원 배열**이 됩니다.
```python
# 2차원 배열 만들기
score= np.array([[78, 91, 84, 89, 93, 65],
                 [82, 87, 96, 79, 91, 73]])
# np.array에 조건문을 작성하면 결과를 boolean 값으로 반환
print(score >= 90)
'''
[[False,  True, False, False,  True, False],
 [False, False,  True, False,  True, False]])
'''
# 요소 중에서 90 이상인 것만 조회
print(score[score >= 90])
'''
[91 93 96 91]
'''
# 검색조건을 변수로 선언해서 사용 가
condition = score >= 90
print(score[condition])
'''
[91 93 96 91]
'''
# 모든 요소 중에서 90 이상 95 미만인 것만 조회
print(score[(score >= 90) & (score <= 95)])
'''
[91 93 91]
'''
```

```python
# 두 개의 (2, 2) 형태의 2차원 배열 만들기
x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6], [7, 8]])

# 배열 더하기
print(x + y)
print(np.add(x, y))
'''
[[ 6  8]
 [10 12]]
'''
# 배열 빼기
print(x - y)
print(np.subtract(x, y))
'''
[[-4 -4]
 [-4 -4]]
'''
# 배열 곱하기
print(x * y)
print(np.multiply(x, y))
'''
[[ 5 12]
 [21 32]]
'''
# 배열 나누기
print(x / y)
print(np.divide(x, y))
'''
[[0.2        0.33333333]
 [0.42857143 0.5       ]]
'''
# 배열 y 제곱
print(x ** y)
print(np.power(x, y))
'''
[[    1    64]
 [ 2187 65536]]
'''
```

## Numpy Array에서 자주 사용되는 함수들

- np.sum(), 혹은 array.sum()
    - axis = 0 : 열 기준 집계
    - axis = 1 : 행 기준 집계
    - 생략하면 : 전체 집계
- 동일한 형태로 사용 가능한 함수 : np.max(), np.min(), np.mean(), np.std()
```python
# array를 생성합니다.
a = np.array([[1,5,7],[2,3,8]])

# 전체 집계
print(np.sum(a)) # 26

# 열기준 집계
print(np.sum(a, axis = 0)) # [ 3  8 15]

# 행기준 집계
print(np.sum(a, axis = 1)) # [13 13]
```

- np.argmax(), np.argmin() : 인덱스 값으로 반환해줌
```python
# 전체 중에서 가장 큰 값의 인덱스
print(np.argmax(a)) # 5

# 행 방향 최대값의 인덱스
print(np.argmax(a, axis = 0)) # [1 0 1]

# 열 방향 최대값의 인덱스
print(np.argmax(a, axis = 1)) # [2 2]
```

- np.where(조건문, True 값, False 값) : 조건문에 True면 2번째 인수로, False면 3번째 인수로 변환
```python
# 선언
a = np.array([1,3,2,7])

# 조건 1
print(np.where(a > 2, 1, 0))   # [0 1 0 1]

# 조건 2
print(np.where(a > 3, -1, 5))  # [ 5  5  5 -1]

# 원래 값 유지하고 싶다면 배열 변수를 사용
print(np.where(a > 2, a, 0))   # [0 3 0 7]

# 마찬가지로 변수를 이용한 연산 값도 반환 가능
print(np.where(a > 2, a, a-1)) # [0 3 1 7]
```

★ np.array 가 지원하는 메소드는 np.array 데이터에 직접적으로 메소드 호출이 가능하다.

```python
# 선언
a = [1,2,3]            # 리스트
b = (1,2,3)            # 튜플
c = np.array([1,2,3])  # 배열

# 평균 구하기 : 함수
print(np.mean(a))      # 2.0
print(np.mean(b))      # 2.0
print(np.mean(c))      # 2.0

# 평균 구하기 : 메서드 방식
print(a.mean()) # AttributeError: 'list' object has no attribute 'mean'
print(b.mean()) # AttributeError: 'tuple' object has no attribute 'mean'

print(c.mean()) # 2.0
```

### DataFrame에서 자주 사용되는 메소드들

- head(): 상위 데이터 확인
- tail(): 하위 데이터 확인
- shape: 데이터프레임 크기
- values: 값 정보 확인(저장하면 2차원 numpy 배열이 됨)
- columns: 열 정보 확인

```python
# 상위 10개 행 데이터
data.head(10)

# 하위 3개 행 데이터
data.tail(3)

# 행 수와 열 수 확인
data.shape

# 데이터프레임 값 확인 (2차원 numpy 배열)
data.values

# 열 확인
print(data.columns)

print(data.columns.values) # np array 형태

list(data) # 데이터프레임을 리스트 함수에 넣으면 열 이름이 리스트로 반환됨.
```

- dtypes: 열 자료형 확인
    - int64: 정수형 데이터(int)
    - float64: 실수형 데이터(float)
    - object: 문자열 데이터(string)

```python
# 열 자료형 확인
data.dtypes
```

- info(): 열에 대한 상세한 정보 확인

```python
# 열 자료형, 값 개수 확인
data.info()
```

- describe(): 기초통계정보 확인
    - 개수(count), 평균(mean), 표준편차(std), 최솟값(min),
    사분위값(25%, 50%, 75%), 최댓값(max)을 표시

```python
# 기초통계정보
data.describe()
```

- sort_index() : 인덱스를 기준으로 정렬하는 메소드
    - **ascending** 옵션을 설정해 오름차순, 내림차순을 설정할 수 있습니다.
        - ascending=True: 오름차순 정렬(기본값)
        - ascending=False: 내림차순 정렬
- sort_values() : 특정 열 기준으로 정렬하는 메소드
    - 2가지 방법 → 인덱스를 기준으로 정렬하는 방법과 특정 열을 기준으로 정렬하는 방법
    - **sort_values()** 메소드로 **특정 열**을 기준으로 정렬합니다.
    - **ascending** 옵션을 설정해 오름차순, 내림차순을 설정할 수 있습니다.
        - ascending=True: 오름차순 정렬(기본값)
        - ascending=False: 내림차순 정렬

```python
# 단일 열 정렬
data.sort_values(by='MonthlyIncome', ascending=False)
# 복합 열 정렬
data.sort_values(by=['JobSatisfaction', 'MonthlyIncome'], ascending=[True, False])
# 복합 열 정렬 : 별도로 저장하고, 인덱스 reset
temp = data.sort_values(by=['JobSatisfaction', 'MonthlyIncome'], ascending=[True, False])
temp.reset_index(drop = True) # 기존 index는 삭제하고, 현재 데이터 기준으로 인덱스를 재작성
```

- unique() : 고유 값 확인

```python
# MaritalStatus 열 고유값 확인
print(data['MaritalStatus'].unique())
```

- value_counts() : 고유 값과 그 개수들을 확인, dtype도 확인 가능

```python
# MaritalStatus 열 고유값 개수 확인
print(data['MaritalStatus'].value_counts())
```

- sum(), max(), mean(), median(), count() : 기본 집계 메소드, Groupby 기능과 함께 많이 사용

```python
# MonthlyIncome 열 합계 조회
print(data['MonthlyIncome'].sum())

# MonthlyIncome 열 최댓값 조회
print(data['MonthlyIncome'].max())

# 'Age', 'MonthlyIncome' 열 평균값 확인
print(data[['Age', 'MonthlyIncome']].mean())

# 'Age', 'MonthlyIncome' 열 중앙값 확인
print(data[['Age', 'MonthlyIncome']].median())

# MonthlyIncome 열 데이터 개수 확인
print(data['MonthlyIncome'].count())
```

## DataFrame 데이터 조회

- 1차원(시리즈)로 조회하기

```python
# total_bill 열 조회(시리즈)
tip['total_bill']
tip.total_bill   ← 메소드인지 column 값인지 구분이 안가서 선호하지 않는 방식
```

- 2차원(데이터프레임)으로 조회

```python
# total_bill 열 조회(데이터프레임)
tip[['total_bill']] # column 이름을 '리스트'로 입력 한 것이다!

# 여러 열 한번에 조회 : Age, DistanceFromHome, Gender열만 조회
data[['Age', 'DistanceFromHome', 'Gender']]
```

- 조건으로 조회 : .loc[행 조건, 열 이름]
    - **df.loc[조건]** 형태로 조건을 지정해 조건에 만족하는 데이터만 조회할 수 있습니다.
    - 우선 조건이 제대로 판단이 되는지 확인한 후 그 **조건을 대 괄호 안에** 넣으면 됩니다.
    
    1) 단일 조건 조회
    
    ```python
    # DistanceFromHome 열 값이 10 보다 큰 행 조회
    data.loc[data['DistanceFromHome'] > 10]
    ```
    
    2) 여러 조건 조회
    
    - [ ]안에 조건을 여러개 연결할 때 ‘and와 or’ 대신에 ‘&와 |’를 사용해야 합니다.
    - 그리고 각 조건들은 **(조건1) & (조건2)** 형태로 **괄호**로 묶어야 합니다.
    
    ```python
    # and로 여러 조건 연결
    data.loc[(data['DistanceFromHome'] > 10) & (data['JobSatisfaction'] == 4)]
    # or 조건 |
    data.loc[(data['DistanceFromHome'] > 10) | (data['JobSatisfaction'] == 4)]
    ```
    
- isin([값1, 값2,…, 값n]) : 값1 또는 값2 또는...값n인 데이터만 조회, 매개변수는 무조건 리스트 형태

```python
# 1이나 4인 값 나열
data.loc[data['JobSatisfaction'].isin([1,4])]
# 위 구문은 아래 or 조건으로 작성한 것과 동일하다
data.loc[(data['JobSatisfaction'] == 1) | (data['JobSatisfaction'] == 4)]
```

- between(값1, 값2) : 값1 ~ 값2까지 범위안의 데이터만 조회합니다.
    - inclusive = 'both' (기본값), 'left', 'right', 'neither'
    
    ```python
    # 범위 지정
    data.loc[data['Age'].between(25, 30)]
    # 위 구문은 아래 and 조건으로 작성한 것과 동일하다
    data.loc[(data['Age'] >= 25) & (data['Age'] <= 30)]
    ```
    
- 조건을 만족하는 행의 일부 열만 조회 : **df.loc[조건, ['열 이름1', '열 이름2',...]]**

```python
# 조건에 맞는 하나의 열 조회
data.loc[data['MonthlyIncome'] >= 10000, ['Age']]
# 조건에 맞는 여러 열 조회
data.loc[data['MonthlyIncome'] >= 10000, ['Age', 'MaritalStatus', 'TotalWorkingYears']]

# 열 이름만 지정하고 모든 행을 가져오려면, 아래와 같이 작성
data.loc[ : , ['Age', 'MaritalStatus', 'TotalWorkingYears'] ]
```

## DataFrame 데이터 집계

- dataframe.groupby( ‘집계기준변수’, as_index = )[‘집계대상변수’].집계함수
    - **집계기준변수:** ~~별에 해당되는 변수 혹은 리스트. 범주형변수(예: 월 별, 지역 별 등)
    - **집계대상변수:** 집계함수로 집계할 변수 혹은 리스트. (예: 매출액 합계)
    - **as_index=True**를 설정(기본값)하면 집계 기준이 되는 열이 인덱스 열이 됩니다.
- 집계 결과가 data 열만 가지니 **’시리즈’**가 됩니다

```python
# MaritalStatus 별 Age 평균 --> 시리즈
data.groupby('MaritalStatus', as_index=True)['Age'].mean()
```

- [['data']].sum()과 같이 하면 열이 여럿이라는 의미여서 결과가 **데이터프레임**이 됩니다.

```python
# MaritalStatus 별 Age 평균 --> 데이터프레임
data.groupby('MaritalStatus', as_index=True)[['Age']].mean()
```

- **as_index=False**를 설정하면 행 번호를 기반으로 한 정수 값이 인덱스로 설정됩니다.

```python
# MaritalStatus 별 Age 평균 --> 데이터프레임
data.groupby('MaritalStatus', as_index=False)[['Age']].mean()
```

★ 집계 결과는 새로운 데이터프레임으로 선언하여 사용하는 경우가 많다

- 여러 열 집계 : 집계 대상 열을 리스트로 지정하면 됨
    - sum() 앞에 아무 열도 지정하지 않으면 **기준열 이외의 모든 열에 대한 집계** 수행
    → 숫자형 변수만 집계되도록 명시적으로 지정할 필요가 있음
    
    ```python
    # 여러 열 집계 방법
    data.groupby('MaritalStatus', as_index=False)[['Age','MonthlyIncome']].mean()
    # 전부 sum하는 함수 형태
    data.groupby('MaritalStatus', as_index=False).sum()
    ```
    
- **by=['feature1', 'feature2']** 과 같이 집계 기준 열을 여럿 설정 가능

```python
# 'MaritalStatus', 'Gender'별 나머지 열들 평균 조회
data_sum = data.groupby(['MaritalStatus', 'Gender'], as_index=False)[['Age','MonthlyIncome']].mean()
```

- df.groupby( )**.agg(['함수1','함수2', ...])** : 여러 함수 한번에 사용 (as_index=False 작동 안함)

```python
# min, max, mean 함수 3가지 한번에 적용
data_agg = data.groupby('MaritalStatus')[['MonthlyIncome']].agg(['min','max','mean'])

# 각 열마다 다른 집계를 한 번에 수행 가능
data.groupby('MaritalStatus', as_index=False).agg({'MonthlyIncome':'mean', 'Age':'min'})
```

# DataFrame 변경

- .rename() : 일부 열 이름 변경 하기

```python
# columns 에 dictionary 데이터로 입력, inplace=True로 작성 시 데이터에 변경사항 바로 적용
data.rename(columns={'DistanceFromHome' : 'Distance', 
                    'EmployeeNumber' : 'EmpNo',
                    'JobSatisfaction' : 'JobSat',
                    'MonthlyIncome' : 'M_Income',
                    'PercentSalaryHike' : 'PctSalHike',
                    'TotalWorkingYears' : 'TotWY'}, inplace=True)
```

- .columns : 모든 열 이름을 변경할 때 사용할 수 있음

```python
# 변경을 원치 않다면 기존 열 이름 그대로 작성
data.columns = ['Attr','Age','Dist','EmpNo','Gen','JobSat','Marital','M_Income', 'OT', 'PctSalHike', 'TotWY']
```

- 열 추가 방법 : 없는 column 명으로 넣고자 하는 데이터를 작성하면 알아서 추가됨

```python
data['Income_LY'] = round(data['M_Income'] / (1+data['PctSalHike']/100 ))
```

 - insert() 메소드 사용 시 원하는 위치에 column 추가 가능 (but, 위치를 크게 신경 안 쓰는게 좋다)

```python
# 첫번째 인수는 index 값(위치), 넣고자하는 column명, 넣고자하는 데이터 순으로 인수 작성
data.insert(1, 'Income_LY', round(data['M_Income'] / (1+data['PctSalHike']/100 ))
```

- .drop() : 데이터 삭제를 위한 메소드
    - axis = 0 : 행 삭제 (default)
    - axis = 1 : 열 삭제
    - inplace=False : 삭제한 데이터 반환만, 기존 데이터에 반영 안함 (default)
    - inplace=True : 기존 데이터에도 삭제한 내용을 반영함
    
     ★ 삭제는 잘못하면 되돌릴 수가 없는만큼, 데이터를 미리 copy 해두고 하는 것도 좋다!
    

```python
# data를 복사합니다.
data2 = data.copy()

# 열 하나 삭제
data2.drop('Income_LY', axis=1, inplace=True)

# 열 두 개 삭제
data2.drop(['JobSat2','Diff_Income'], axis=1, inplace=True)
```

- 값 변경 : 기존 방법 + map, cut 메소드 활용 방법 들이 있다.
    
    ★ 값 변경 또한 데이터를 미리 copy 해두고 하는 것도 좋다!
    

```python
# data를 복사
data2 = data.copy()

# Income_LY의 값을 모두 0로 변경
data2['Income_LY'] = 0

# 조건부 변경 : Diff_Income 의 값이 1000보다 작은 경우, 0로 변경
data2.loc[data2['Diff_Income'] < 1000, 'Diff_Income' ] = 0

# 조건부 변경 : Age가 40보다 많으면 1, 아니면 0으로 변경
data2['Age'] = np.where(data2['Age'] > 40, 1, 0)
```

- .map() : 인자로 dictionary { } 값을 넣어서 key에 해당하는 데이터를 value 값으로 변경함
    - 주로 범주형 데이터를 다룰 때 사용

```python
# Male -> 1, Female -> 0
data['Gen'] = data['Gen'].map({'Male': 1, 'Female': 0})
```

- pd.cut() : 숫자형 변수를 범주형 변수로 변환하는 함수

```python
# Age 열을 3 등분으로 분할
age_group = pd.cut(data2['Age'], 3)
'''
★ 값의 범위를 균등하게 나누는 것이고, 값의 개수를 균등하게 맞추는게 아니다! ★
Age
(32.0, 46.0]      590
(17.958, 32.0]    413
(46.0, 60.0]      193
'''

# Age 열을 3 등분으로 분할후 a,b,c로 이름 붙이기
age_group = pd.cut(data2['Age'], 3, labels = ['a','b','c'])

# 원하는 구간대로 나누고, 이름 붙이기
# 'young'  : =< 40 
# 'junior' : 40 <   =< 50
# 'senior' : 50 < 
age_group = pd.cut(data2['Age'], bins =[0, 40, 50, 100] , labels = ['young','junior','senior'])
```

# DataFrame 결합

- pd.concat() : 합치고자 하는 Dataframe 2개를 리스트로 넣어준다.
    - axis = 0 : 세로(행)로 합침, column 명이 기준
    - axis = 1 : 가로(열)로 합침, 행 인덱스 기준
    - join = ‘inner’ : 같은 행과 열 합치기 (default)
    - join = ‘outer’ : 모든 행과 열 합치기
    
    ```python
    df1 = pd.DataFrame({'A':[10,25], 'B':[15,30]})
    df2 = pd.DataFrame({'A':[20,30, 50], 'C':[35,30, 40]})
    df2.drop([1], inplace = True)
    '''
    df1
    		A	  B
    0|	10	15
    1|	25	30
    
    df2
    		A	  C
    0|	20	35
    2|	50	40
    '''
    # axis = 0, join = 'inner'
    pd.concat([df1, df2], axis = 0, join = 'inner')
    '''
    		A
    0|	10
    1|	25
    0|	20
    2|	50
    '''
    # axis = 0, join = 'outer'
    pd.concat([df1, df2], axis = 0, join = 'outer')
    '''
    		A	  B   	C
    0|	10	15.0	NaN
    1|	25	30.0	NaN
    0|	20	NaN	  35.0
    2|	50	NaN	  40.0
    '''
    # axis = 1, join = 'inner'
    pd.concat([df1, df2], axis = 1, join = 'inner')
    '''
    		A 	B 	A 	C
    0|	10	15	20	35
    '''
    # axis = 1, join = 'outer'
    pd.concat([df1, df2], axis = 1, join = 'outer')
    '''
    		A   	B   	A   	C
    0|	10.0	15.0	20.0	35.0
    1|	25.0	30.0	NaN 	NaN
    2|	NaN 	NaN 	50.0	40.0
    '''
    ```
    
- merge() : 데이터 2가지를 옆으로만 붙이는데, 자동으로 key를 잡아준다.(특정열의 값)
    - on = ‘A’ : ‘A’ 값을 기준으로 합친다.
    - how = ‘inner’ : 중복되는 값이 있는 데이터만 합친다.
    - how = ‘outer’ : 모든 데이터를 전부 합친다.
    - how = ‘left’ : 입력한 데이터 중 좌측 데이터 기준으로 합친다.
    - how = ‘right’ : 입력한 데이터 중 우측 데이터 기준으로 합친다.
    
    ```python
    df1 = pd.DataFrame({'A':[1,2], 'B':[15,30], 'C':[20, 25]})
    df2 = pd.DataFrame({'A':[2,3], 'D':[20, 35]})
    '''
    df1
    		A	B 	C
    0|	1	15	20
    1|	2	30	25
    df2
    		A	D
    0|	2	20
    1|	3	35
    '''
    # A 값 기준으로 inner merge 하기
    pd.merge(df1, df2, how = 'inner', on = 'A')
    pd.merge(df1, df2, how = 'inner') # 위 데이터 기준 동일한 코드
    '''
    		A	B 	C 	D
    0|	2	30	25	20
    '''
    # how = 'outer'
    pd.merge(df1, df2, how = 'outer')
    '''
    		A	B   	C   	D
    0|	1	15.0	20.0	NaN
    1|	2	30.0	25.0	20.0
    2|	3	NaN 	NaN 	35.0
    '''
    # how = 'left'
    pd.merge(df1, df2, how = 'left')
    '''
    		A	B 	C 	D
    0|	1	15	20	NaN
    1|	2	30	25	20.0
    '''
    # how = 'right'
    pd.merge(df1, df2, how = 'right')
    '''
    		A	B   	C   	D
    0|	2	30.0	25.0	20
    1|	3	NaN 	NaN 	35
    '''
    ```
    
- pivot() : 집계 후 데이터프레임 구조 변형해서 조회하는데 사용 (결합은 아님)
    - 순서 : 1→ groupby / 2→ pivot

```python
# 1) 매장1의 일별 카테고리별 판매량을 집계
temp = pd.merge(sales1, products)
temp2 = temp.groupby(['Date', 'Category'], as_index = False)['Qty'].sum()
temp2

# 2) pivot (index = 행으로 삼을 값 / columns = 열로 삼을 값 / values = 나타낼 값)
temp3 = temp2.pivot(index = 'Category', columns = 'Date', values = 'Qty')
temp3
```

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