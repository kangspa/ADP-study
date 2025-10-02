# 기본 내장 데이터 로드
data(mtcars)
data(iris)
data(airquality)
data(ToothGrowth)
data(PlantGrowth)
data(ChickWeight)
data(USArrests)
data(pressure)
data(Titanic)

# ggplot2 패키지의 diamonds, economics 데이터
library(ggplot2)
data(diamonds)
data(economics)

# EuStockMarkets, nottem (시계열 데이터)
data(EuStockMarkets)
data(nottem)

# Titanic은 table 형식이므로 데이터프레임으로 변환
Titanic_df <- as.data.frame(Titanic)

# 시계열 객체는 data.frame으로 변환
EuStockMarkets_df <- as.data.frame(EuStockMarkets)
nottem_df <- data.frame(date = time(nottem), temp = as.numeric(nottem))

# 내보낼 데이터 리스트
datasets <- list(
  mtcars = mtcars,
  iris = iris,
  airquality = airquality,
  ToothGrowth = ToothGrowth,
  diamonds = diamonds,
  PlantGrowth = PlantGrowth,
  ChickWeight = ChickWeight,
  USArrests = USArrests,
  pressure = pressure,
  Titanic = Titanic_df,
  economics = economics,
  EuStockMarkets = EuStockMarkets_df,
  nottem = nottem_df
)

# CSV로 내보내기 (현재 작업 디렉토리에 저장됨)
for (name in names(datasets)) {
  write.csv(datasets[[name]], paste0(name, ".csv"), row.names = FALSE)
}

# 현재 작업 디렉토리 확인
getwd()
