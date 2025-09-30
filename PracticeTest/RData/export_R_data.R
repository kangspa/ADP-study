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

# ggplot2 패키지의 diamonds 데이터
library(ggplot2)
data(diamonds)

# Titanic은 table 형식이므로 데이터프레임으로 변환
Titanic_df <- as.data.frame(Titanic)

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
  Titanic = Titanic_df
)

# CSV로 내보내기 (현재 작업 디렉토리에 저장됨)
for (name in names(datasets)) {
  write.csv(datasets[[name]], paste0(name, ".csv"), row.names = FALSE)
}

# 현재 작업 디렉토리 확인
getwd()
