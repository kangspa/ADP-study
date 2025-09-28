# 다양한 후기글 참고해서 작성하는 tip

- 전처리, 모델링 정도는 바로 타이핑할 수 있을 정도
- 자신만의 코드북을 만들면서 익숙해지기
- 시각화가 어렵다면, 관련 자료만 별도 페이지로 오픈북 구축하기
- 통계 부분을 우선 풀기 추천 : 아는 내용은 바로 타이핑, 모르는 내용은 빠르게 스킵하는게 중요
    - 보통 통계 파트에서 어려운 문제(통수 문제)가 한번씩 나오므로, 이외의 ML/통계 문제를 모두 맞추는 것을 목표로 해야함
    - 위와 같은 문제 대비 `help()` 함수 확인하는 습관도 필요
- 통계 1시간, 머신러닝 1번 문제 1시간 반, 2번 문제 1시간 추천
    - 1번 문제 : 난이도 높지만 과정이 복잡하진 않음
    - 2번 문제 : 난이도는 낮지만 전처리 및 요구하는 모델링이 복잡한 경우가 많음
    - 매 시험마다 문제가 다르므로, 시간 배분은 많이해서 익숙해지는 수 밖에 없음
- 좋은 성능의 모델보다, 설득력 있는 분석 내용을 제시하는게 중요
- 주피터 노트북에서 실습 진행하는 것을 습관들여야 함
    - 특히 pdf로 제출해야하는데, 주피터 노트북을 pdf로 출력할 때 안 잘리게 하는 방법 등에 익숙해져야 함

- 곽기영 교수님 통계 강의 확인 : <https://youtube.com/playlist?list=PLY0OaF78qqGAxKX91WuRigHpwBU0C2SB_&si=Kag_O-KOsjV1BJUg>

# 시험문제 복원 출처 관련

- 마크다운으로 작성해둔 모든 복원 기출 문제는 웹서핑 통해서 볼 수 있는, 오픈되어 있는 블로그 글들에서 가져왔습니다.
- 각 문제지 최상단에 출처를 작성해뒀는데, 원본 글로 이동할 경우 블로거가 제작해둔 데이터를 볼 수도 있습니다.

- 가장 많이 참고한 출처
    - 출처 : <https://www.datamanim.com/dataset/ADPpb/index.html>
    - 출처 : <https://statisticsplaybook.com/adp-past-exam-questions/>

# 프롬프트 : 복원 문제별 도출해야하는 분석 방법

1. 사전에 "시험문제 복원" 폴더까지 이동 후 `gemini-cli`를 실행하여 진행했습니다.

2. 아래 프롬프트를 "시험문제 복원" 폴더 하위의 "prompt.txt"에 붙여넣기 해주세요.
```markdown
현재 폴더는 ADP(데이터 분석 전문가) 실기 시험 문제 복원 파일을 모아둔 프로젝트입니다.
해당 시험 진행 시에는 아래와 같은 조건들을 따라야 합니다.

- 사용 언어 : python 3.7
- 사용 패키지별 버전
    - numpy==1.21.6
    - pandas==1.1.2
    - scipy==1.7.3
    - statsmodels==0.13.2
    - scikit-learn==0.23.2
    - matplotlib==3.0.3
    - torch==1.7.1
    - torchvision==0.2.2.post3

{filename}을 읽고, 각 문제별 어떤 분석 방법으로 접근해야하는지 가능한 모든 방법을 전부 알려주세요.
각 문제별로 아래와 같은 내용을 서술해주면 됩니다.

- 사용 가능한 분석 방법
    - 해당 분석 방법에 대한 설명
    - 해당 분석 방법에 대한 코드 예제
- 현재 문제에 관한 풀이 방법

위와 같은 내용을 지켜서, "{filename}-by_Gemini.md" 파일명으로 생성해주면 됩니다.
```

3. 하단의 내용을 `gemini-cli`를 통해 실행합니다.
```bash
`prompt.txt` 파일을 읽고, 해당 내용을 이행해줘. `filename`은 다음과 같은 목록을 순차적으로 순회하면서 prompt를 진행하면 돼. ["제24회.md", "제25회.md", "제26회.md", "제27회.md", "제28회.md", "제29회.md", "제30회.md", "제31회.md", "제33회.md"]. 파일 생성하는 모든 경우에 대해 allow을 할게.
```

# 프롬프트 : 오픈북 생성

- 참고 : <https://well-being-stat.tistory.com/4>

## 사용 방법

- `gemini-cli` 를 활용하여 진행 예정

1. 프로젝트 프롬프트는 `prompt.txt`에 작성
2. 요청하고자 하는 topic에 맞춰서 `프롬프트 : ML, Modeling, Stat` 중 하나를 `prompt.txt` 하단에 추가
3. `gemini-cli`에 아래와 같은 내용으로 요청 (~~~ 부분을 요청 내용에 맞춰서 수정)

```bash
`prompt.txt` 파일을 읽고, 해당 내용을 이행해줘. `topic`은 ~~~야.
```

## 프로젝트 프롬프트

다음과 같은 조건들을 토대로 ADP(데이터 분석 전문가) 실기 시험을 준비를 위한 마크다운 파일을 만들어주세요.

- 사용 언어 : python 3.7
- 사용 패키지별 버전
    - numpy==1.21.6
    - pandas==1.1.2
    - scipy==1.7.3
    - statsmodels==0.13.2
    - scikit-learn==0.23.2
    - matplotlib==3.0.3
    - torch==1.7.1
    - torchvision==0.2.2.post3
- 머신러닝 파트를 다룰 경우, sklearn 기반 코드를 작성해주세요.
- 통계 파트를 다룰 경우, stats, sm, pingouin 중 효율적인 패키지 기반으로 작성해주세요.
- 이모티콘은 사용하지 말고 경어체를 사용해주세요.

### 프롬프트 : ML

{topic}에 관한 내용을 파이썬, 가능하면 sklearn 기반으로 작성해주세요.
해당 방법에 대해, `개념 요약 - 적용 가능한 상황 - 구현 방법(함수 인자의 종류 설명 포함) - 장단점 및 대안` 순서로 설명해주세요.
구현 방법들에 대해, 각각 `용도-주의사항-코드 예시` 순서로 정리해주세요.

### 프롬프트: Modeling
{topic}에 관한 내용을 파이썬, 가능하면 sklearn 기반으로 작성해주세요. 필요하다면 pytorch 기반 모델링을 작성해도 좋습니다.
모델에 대해, `카테고리-용도-주의사항(가정, 필요하면 하이퍼파라미터 옵션 선택 가이드 포함)-사용방범(예시 및 코드)-해석 방법` 순서로 설명해주세요.
Regressor와 Classifier의 공통된 부분을 우선 설명해주고, 명시해주세요.

### 프롬프트: Stat
{topic}에 관한 내용을 파이썬, 가능하면 stats나 sm, pingouin 중 가장 효율적인 코드 기반으로 작성해주세요.
{topic}의 개념에 대해서, `용도-주의사항(가정 등)-사용방법(예시 및 코드)-해석 방법` 순서로 설명해주세요.

# 타인의 오픈북 자료

- [GIL'sLAB](https://gils-lab.tistory.com/8)에서 다운받은 오픈북 자료
    1. [지도학습](<GIL'sLAB/1. 지도학습.html>)
    2. [통계분석](<GIL'sLAB/2. 통계분석.html>)
    3. [연관분석](<GIL'sLAB/3. 연관분석.html>)
    4. [군집화](<GIL'sLAB/4. 군집화.html>)
    5. [데이터시각화](<GIL'sLAB/5. 데이터시각화.html>)
    6. [데이터전처리](<GIL'sLAB/6. 데이터전처리 및 Pandas.html>)
    7. [시계열분석](<GIL'sLAB/7. 시계열분석.html>)
    8. [텍스트마이닝](<GIL'sLAB/8. 텍스트마이닝.html>)

<details>
  <summary><strong>시험 준비를 위한 Topic-List</strong></summary>
  <ul>
    <li>참고
      <ul>
        <li><a href="https://well-being-stat.tistory.com/4">https://well-being-stat.tistory.com/4</a></li>
        <li><a href="https://coding-law.tistory.com/entry/번외4-KT-AIVLE-3기-ai트랙-28회-ADP-실기-합격">https://coding-law.tistory.com/entry/번외4-KT-AIVLE-3기-ai트랙-28회-ADP-실기-합격</a></li>
      </ul>
    </li>
    <li>개인적으로 공부한 내용들(주제)은 별도의 마크다운 파일로 만들었습니다.</li>
  </ul>

  <h2>KOCW 비모수통계학</h2>
  <ul>
    <li><a href="http://www.kocw.net/home/cview.do?mty=p&kemId=1004752">http://www.kocw.net/home/cview.do?mty=p&kemId=1004752</a>
      <ul>
        <li>12강 분산분석 파트, 13강 비모수적 방법 파트</li>
      </ul>
    </li>
    <li><a href="http://www.kocw.net/home/cview.do?mty=p&kemId=865635&ar=link_gil">http://www.kocw.net/home/cview.do?mty=p&kemId=865635&ar=link_gil</a>
      <ul>
        <li>7강 모수검정과 비모수검정 파트</li>
      </ul>
    </li>
    <li><a href="http://www.kocw.net/home/cview.do?cid=7cc3a7f9daa84276">http://www.kocw.net/home/cview.do?cid=7cc3a7f9daa84276</a>
      <ul>
        <li>2강 일표본 위치문제 파트 (부호검정 등)</li>
      </ul>
    </li>
  </ul>

  <h2>전처리</h2>
  <ul>
    <li>사전작업(공통)</li>
    <li>연속형 변수변환, Scaling</li>
    <li>범주형 인코딩</li>
    <li>이상치 탐지+처리</li>
    <li>결측치 처리</li>
    <li>EDA 시각화</li>
    <li>Sampling</li>
    <li>시계열 데이터 전처리</li>
  </ul>

  <h2>파이썬 문법</h2>
  <ul>
    <li>핸들링(기초)</li>
    <li>핸들링(심화)</li>
  </ul>

  <h2>모델링</h2>
  <ul>
    <li>선형회귀</li>
    <li>정규화 선형모델</li>
    <li>비선형 모델(앙상블×)</li>
    <li>앙상블 모델</li>
    <li>Simple DL</li>
    <li>베이지안 회귀</li>
    <li>차원축소, 변수선택법</li>
    <li>군집화</li>
    <li>연관규칙분석</li>
    <li>모델 평가 (지표, CV, Voting)</li>
    <li>모델링 결과 시각화</li>
  </ul>

  <h2>통계</h2>
  <ul>
    <li>단순 추정, 통계 계산</li>
    <li>선형모델(OLS, 정규화, Poly)</li>
    <li>로지스틱회귀</li>
    <li>sm 기반 고급 모델</li>
    <li>단일표본 검정(+정규성)</li>
    <li>2개 집단 비교(독립표본)</li>
    <li>2개 집단 비교(대응표본)</li>
    <li>분산분석(다집단, ANOVA)</li>
    <li>상관관계 검정</li>
    <li>범주형 검정(독립,대응)</li>
    <li>비율 검정</li>
    <li>표본크기, 검정력</li>
    <li>신뢰구간</li>
    <li>다중공선성(Cor,VIF,PCA)</li>
    <li>베이지안 분석</li>
    <li>선형계획법</li>
    <li>이산확률분포</li>
    <li>연속확률분포</li>
    <li>시계열(sm, tsa)</li>
    <li>생존분석</li>
    <li>샘플 데이터 생성</li>
    <li>베이지안 모델링</li>
  </ul>
</details>