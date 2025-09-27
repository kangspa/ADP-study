# 파이썬 문법

- Jupyter 노트북 단축키 : 노션 (python 기초) 
- 기초 문법 (기본 문법) : 노션 (python 기초)
- Numpy 라이브러리 메소드 : 노션 (python 데이터 구조)
- Pandas 라이브러리 메소드 : 노션 (python 데이터 구조)

# 데이터 전처리

- 데이터 프레임 변경 메소드 : 노션 (데이터 처리)
- 데이터 프레임 결합 메소드 : 노션 (데이터 처리)
- 시계열 데이터 처리 : 노션 (데이터 처리)
- 결측치 처리 : 노션 (머신러닝 - 전처리 및 이해)
- 가변수화 : 노션 (머신러닝 - 전처리 및 이해)
- MinMaxScaler
- StandardScaler
- 클래스 불균형 해결
    - Under Sampling
    - Over Sampling
    - Class Weight으로 해결
- 이상치 처리
- 로그 변환
- 차원 축소
    - pca
    - t-sne

# 데이터 시각화

- matplot 라이브러리 메소드 : 노션 (데이터 시각화 및 분석)
    - 시각화 차트 꾸미는 메소드들 정리
    - 여러 그래프 겹치기 정리
- seaborn 라이브러리 메소드
- 히트맵

# 데이터 분석

- 단변량분석 : 노션 (데이터 시각화 및 분석)
    - 수치형
    - 범주형
- 이변량 분석 : 노션 (데이터 분석)
    - 수치형 → 수치형
    - 수치형 → 범주형
    - 범주형 → 수치형
    - 범주형 → 범주형

# 모델링

## 데이터셋 구축

- 데이터 분리 : sklearn
- 클래스 불균형 해결
    - Under Sampling
    - Over Sampling
    - Class Weight으로 해결

## 머신러닝

- LinearRegression
- KNeighborsClassifier
- DecisionTreeClassifier
    - export_graphviz
    - from IPython.display import Image
- DecisionTreeRegressor
    - export_graphviz
    - from IPython.display import Image
- LogisticRegression
- 앙상블
- 보팅
- 배깅
- 부스팅
- 스태킹
- 군집화 방법
- 클러스터링
- 이상탐지 모델
    - 고립 포레스트
    - LOF

### 모델 성능 최적화

- cross_val_score
- RandomizedSearchCV
- GridSearchCV

## 딥러닝 모델링

- 데이터셋 구축 : pytorch, keras
- 데이터로더 구축 : pytorch, keras
- 클래스 불균형 해결
    - Under Sampling
    - Over Sampling
    - Class Weight으로 해결
- 모델 구축
    - 단일 구축 및 Sequential, Functional 활용 등
- 레이어 추가
- 활성화 함수 추가
- 손실 함수
- 다중 분류 시 마지막 레이어
- 규제 및 dropout 등
- 학습 코드
    - 가중치 업데이트
- 모델 저장
- 단순 모델부터 시계열 모델링까지 작성

### 비정형 데이터 전처리 및 모델 구축

- 이미지 데이터 전처리 및 분석
    - 이미지 관련 모델 활용 방법
- 텍스트 데이터 전처리 및 분석
    - 텍스트 관련 모델 활용 방법

## 성능 평가

- mean_absolute_error
- accuracy_score
- MSE
- RMSE
- MAE
- MAPE
- R^2
- F1-Score
- precision_score
- recall_score
- confusion_matrix
- classification_report
- 군집 성능 지표

# 통계

- 가설 및 가설 검정 : 노션 (데이터 분석)
- 평균 추정 및 신뢰 구간 : 노션 (데이터 분석)
- 중심 극한 정리 : 노션 (데이터 분석)
- 표준 오차 : 노션 (데이터 분석)
- 카이제곱검정 : 노션 (데이터 분석)
- 기술통계
- 상관관계 분석
- 분포 시각화
- 변화 시각화
- 잔차 플롯 (시각화)
- F-test
- Z-검정
- 베이즈 정리
- 데이터 정재
- EDA
- 이상치 확인 및 처리 방법
- 탐색적 분석
- SARIMA
- 확률 계산 관련 방법들
    - 조건부 확률
- 비모수 검정
    - 크러스칼-월리스 검정
- 최적화 (정수 선형 계획법)
- 표본 크기 산출 공식
    - 신뢰수준
    - 추정오차한계
- 시계열 및 변화율
- 동질성 검정
- 독립표본 t-검정
- 베이지안 회귀분석
- 평균 변화율 계산 (기하평균)
- 신뢰 구간 추정
- 대응표본 t-검정
- 분위수 회귀
- 최소제곱법
- 이원배치 분산분석
- 규제
- 교차 검증
- KaplanMeierFitter
- Log-Rank
- 맥니마 검정
- 등분산 검정
- 편상관 분석
- 상관계수 검정
- 파생변수 생성
- 유의 변수
- SMOTE
- 이항 확률
- 비율 검정
- 사후분석
- 베이즈 정리
- 다중공선성 문제
- 회귀분석
    - 선형성
    - 잔차 독립성
    - 잔차 정규성
    - 등분산성
- 카이제곱 독립성 검정
- 이항분포 확률 및 기댓값
- 변수중요도
- 독립성 통계 검정
- 베이즈 추론
- VIF (분산 팽창 요인)
- 코크란 Q 검정
- 포아송 분포
- 주성분 분석
- 대응표본 t-검정
- 시계열 분석
    - AR, MA, ARMA
    - ACF, PACF

---

## KOCW 비모수통계학

- <http://www.kocw.net/home/cview.do?mty=p&kemId=1004752>
    - 12강 분산분석 파트, 13강 비모수적 방법 파트
- <http://www.kocw.net/home/cview.do?mty=p&kemId=865635&ar=link_gil>
    - 7강 모수검정과 비모수검정 파트
- <http://www.kocw.net/home/cview.do?cid=7cc3a7f9daa84276>
    - 2강 일표본 위치문제 파트 (부호검정 등)

## 전처리

- 사전작업(공통)
- 연속형 변수변환, Scaling
- 범주형 인코딩
- 이상치 탐지+처리
- 결측치 처리
- EDA 시각화
- Sampling
- 시계열 데이터 전처리

## 파이썬 문법

- 핸들링(기초)
- 핸들링(심화)

## 모델링

- 선형회귀
- 정규화 선형모델
- 비선형 모델(앙상블×)
- 앙상블 모델
- Simple DL
- 베이지안 회귀
- 차원축소, 변수선택법
- 군집화
- 연관규칙분석
- 모델 평가 (지표, CV, Voting)
- 모델링 결과 시각화

## 통계

- 단순 추정, 통계 계산
- 선형모델(OLS, 정규화, Poly)
- 로지스틱회귀
- sm 기반 고급 모델
- 단일표본 검정(+정규성)
- 2개 집단 비교(독립표본)
- 2개 집단 비교(대응표본)
- 분산분석(다집단, ANOVA)
- 상관관계 검정
- 범주형 검정(독립,대응)
- 비율 검정
- 표본크기, 검정력
- 신뢰구간
- 다중공선성(Cor,VIF,PCA)
- 베이지안 분석
- 선형계획법
- 이산확률분포
- 연속확률분포
- 시계열(sm, tsa)
- 생존분석
- 샘플 데이터 생성
- 베이지안 모델링