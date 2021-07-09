install.packages("moments")
install.packages("ggplot2")
install.packages("corrplot")
install.packages("fBasics")
install.packages("leaps")

library(moments)
library(ggplot2)
library(corrplot)
library(fBasics)
library(leaps)

# Performance evaluation function for regression
perf_eval_reg <- function(tgt_y, pre_y){
  
  # RMSE
  rmse <- sqrt(mean((tgt_y - pre_y)^2))
  # MAE
  mae <- mean(abs(tgt_y - pre_y))
  # MAPE
  mape <- 100*mean(abs((tgt_y - pre_y)/tgt_y))
  
  return(c(rmse, mae, mape))
}

# 이상치 기준 분류 만들기 함수
  
boxplot_outliers <-function(target){
  valuelist <-boxplot(target)$stats
  return(c(valuelist[1,],valuelist[5,]))
}

  
perf_mat <- matrix(0, nrow = 1, ncol = 3)
rownames(perf_mat) <- c("Seoulbike")
colnames(perf_mat) <- c("RMSE", "MAE", "MAPE")


#데이터 불러오기 
seoulbike=read.table("SeoulBikeData.csv",sep=",",header=TRUE)
#운행하는 날만 데이터 분석
seoulbike<-seoulbike[!(seoulbike$Functioning.Day=="No"),]
seoulbike<-seoulbike[-14]
#날짜 ID 제거
seoulbike<-seoulbike[-1]

nbike <- nrow(seoulbike)
nVar <- ncol(seoulbike)


# 카테고리변수 Binary변수로 변환해주기
Spring <- rep(0,nbike)
Summer <- rep(0,nbike)
Autumn <- rep(0,nbike)
Winter <- rep(0,nbike)

Spring_idx <- which(seoulbike$Seasons == "Spring")
Summer_idx <- which(seoulbike$Seasons == "Summer")
Autmumn_idx <- which(seoulbike$Seasons == "Autumn")
Winter_idx <- which(seoulbike$Seasons == "Winter")

Spring[Spring_idx] <-1
Summer[Summer_idx] <-1
Autumn[Autmumn_idx] <-1
Winter[Winter_idx] <-1

Seasons <- data.frame(Winter, Spring, Summer, Autumn)

#Holiday 변수 Binary로 변환해주기
seoulbike$Holiday<-ifelse(seoulbike$Holiday=="No Holiday",0,1)

#분석데이터 최종 정리
seoulbike_mlr_data<-cbind(seoulbike[,-c(11)], Seasons)



#Q3 &Q4 박스플랏
mtable<-numeric(15)
stdtable<-numeric(15)
skwtable<-numeric(15)
kurtable<-numeric(15)

outliers <- matrix(0, nrow = 15, ncol = 2)
rownames(outliers) <- names(seoulbike_mlr_data)
colnames(outliers) <- c("LCL", "UCL")

for(i in 1:15){
  mtable[i]<-mean(seoulbike_mlr_data[,i])
  stdtable[i]<-sd(seoulbike_mlr_data[,i])
  skwtable[i]<-skewness(seoulbike_mlr_data[,i])
  kurtable[i]<-kurtosis(seoulbike_mlr_data[,i])
  boxplot(main=names(seoulbike_mlr_data[i]), seoulbike_mlr_data[,i])
  outliers[i,]<-boxplot_outliers(seoulbike_mlr_data[,i])
}


q3table<-data.frame(mtable,stdtable,skwtable,kurtable)
dimnames(q3table)=list(row=colnames(seoulbike_mlr_data),col=c("mean","std","skw","kurt"))

#이상치 제거

for (i in 1:15){
  seoulbike_mlr_data[,i]<-ifelse(seoulbike_mlr_data[,i] < outliers[i,1] | seoulbike_mlr_data[,i] > outliers[i,2], NA, seoulbike_mlr_data[,i])
}
sum(is.na(seoulbike_mlr_data))
seoulbike_mlr_data_cleared<-na.omit(seoulbike_mlr_data)
seoulbike_mlr_data_cleared<-seoulbike_mlr_data_cleared[,-c(9,10,11,15)]


#Q5 ScatterPlot 및 Corrplot
par(mar=c(1,1,1,1))
pairs(seoulbike_mlr_data_cleared,main="Scatter Plot Matrix")
corrplot(cor(seoulbike_mlr_data_cleared), method = 'number', type="upper")


#Q6 Training Set ,Validation Set 분리
set.seed(123456)
seoulbike_trn_idx <- sample(1:nbike, round(0.7*nbike))
seoulbike_trn_data<- seoulbike_mlr_data_cleared[seoulbike_trn_idx,]
seoulbike_val_data<- seoulbike_mlr_data_cleared[-seoulbike_trn_idx,]

#Q6 MLR 모델 학습
mlr_seoulbike <- lm(Rented.Bike.Count ~ ., data = seoulbike_trn_data)
mlr_seoulbike
summary(mlr_seoulbike)
plot(mlr_seoulbike)

#Q8
mlr_seoulbike_p <- predict(mlr_seoulbike, newdata = seoulbike_val_data)
perf_mat[1,] <- perf_eval_reg(seoulbike_val_data$Rented.Bike.Count, mlr_seoulbike_p)
perf_mat

#Q10

q10seoulbike <- seoulbike_mlr_data_cleared[,-c(5,6,7,8,9,10,11)]
set.seed(123456)
q10seoulbike_trn_idx <- sample(1:nbike, round(0.7*nbike))
q10seoulbike_trn_data<- q10seoulbike[q10seoulbike_trn_idx,]
q10seoulbike_val_data<- q10seoulbike[-q10seoulbike_trn_idx,]

q10mlr_seoulbike <- lm(Rented.Bike.Count ~ ., data = q10seoulbike_trn_data)
q10mlr_seoulbike
summary(q10mlr_seoulbike)
plot(q10mlr_seoulbike)

q10perf_mat <- matrix(0, nrow = 1, ncol = 3)
rownames(q10perf_mat) <- c("Seoulbike")
colnames(q10perf_mat) <- c("RMSE", "MAE", "MAPE")


q10mlr_seoulbike_p <- predict(q10mlr_seoulbike, newdata = q10seoulbike_val_data)
q10perf_mat[1,] <- perf_eval_reg(q10seoulbike_val_data$Rented.Bike.Count, q10mlr_seoulbike_p)


#Extra Question
extramlr_seoulbike <- regsubsets(Rented.Bike.Count ~ ., data = seoulbike_trn_data,nvmax=9, method="exhaustive")
extramlr_seoulbike
summary(extramlr_seoulbike)
bestbic<-summary(extramlr_seoulbike)$bic
bestbic
plot(mlr_seoulbike)
