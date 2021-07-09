install.packages("fBasics")
install.packages("corrplot")
install.packages("ggplot2")

library(fBasics)
library(corrplot)
library(ggplot2)
# Performance Evaluation Function -----------------------------------------
perf_eval2 <- function(cm){
  
  # True positive rate: TPR (Recall)
  TPR <- cm[2,2]/sum(cm[2,])
  # Precision
  PRE <- cm[2,2]/sum(cm[,2])
  # True negative rate: TNR
  TNR <- cm[1,1]/sum(cm[1,])
  # Simple Accuracy
  ACC <- (cm[1,1]+cm[2,2])/sum(cm)
  # Balanced Correction Rate
  BCR <- sqrt(TPR*TNR)
  # F1-Measure
  F1 <- 2*TPR*PRE/(TPR+PRE)
  
  return(c(TPR, PRE, TNR, ACC, BCR, F1))
}

# Initialize the performance matrix
perf_mat <- matrix(0, 1, 6)
colnames(perf_mat) <- c("TPR (Recall)", "Precision", "TNR", "ACC", "BCR", "F1")
rownames(perf_mat) <- "Logstic Regression"

# 이상치 기준 분류 만들기 함수

boxplot_outliers <-function(target){
  valuelist <-boxplot(target)$stats
  return(c(valuelist[1,],valuelist[5,]))
}



# Load dataset
bc <- read.csv("data.csv")

target_idx <- 2

# Conduct the normalization
bc_target <- bc[,target_idx]
bc_target <- ifelse(bc_target=="B",0,1)
bc_input <- bc[,-target_idx]
id_nondata<-c(1,32)
bc_input <-bc_input[,-id_nondata]
bc_input <-scale(bc_input,center=TRUE, scale=TRUE)
bc_data <-data.frame(bc_input,bc_target)

#Q3&4
mtable<-numeric(30)
stdtable<-numeric(30)
skwtable<-numeric(30)
kurtable<-numeric(30)


outliers <- matrix(0, nrow = 30, ncol = 2)
rownames(outliers) <- names(bc_input)
colnames(outliers) <- c("LCL", "UCL")

for(i in 1:30){
  mtable[i]<-mean(bc_input[,i])
  stdtable[i]<-sd(bc_input[,i])
  skwtable[i]<-skewness(bc_input[,i])
  kurtable[i]<-kurtosis(bc_input[,i])
  boxplot(main=names(bc_input[i]), bc_input[,i])
  outliers[i,]<-boxplot_outliers(bc_input[,i])
}


q3table<-data.frame(mtable,stdtable,skwtable,kurtable)
dimnames(q3table)=list(row=colnames(bc_input),col=c("mean","std","skw","kurt"))

#이상치 제거

for (i in 1:30){
  bc_data[,i]<-ifelse(bc_input[,i] < outliers[i,1] | bc_input[,i] > outliers[i,2], NA, bc_input[,i])
}
sum(is.na(bc_data))
bc_data_cleared<-na.omit(bc_data)
bc_target_cleared<-bc_data_cleared[,31]
bc_input_cleared<-bc_data_cleared[,-31]

#Q5 ScatterPlot 및 Corrplot
par(mar=c(1,1,1,1))
pairs(bc_input_cleared,main="Scatter Plot Matrix")
corrplot(cor(bc_input_cleared), method = "circle", type="upper", number.cex=0.65)
q7bc_data_cleared<-bc_data_cleared[,c(1,2,5,6,7,9,10,31)]

#Q6
set.seed(12345)
trn_idx <- sample(1:nrow(bc_data_cleared), round(0.7*nrow(bc_data_cleared)))
bc_trn <- bc_data_cleared[trn_idx,]
bc_tst <- bc_data_cleared[-trn_idx,]

# Train the Logistic Regression Model with all variables
full_lr <- glm(bc_target ~ . , family=binomial, bc_trn)
summary(full_lr)

lr_response <- predict(full_lr, type = "response", newdata = bc_tst)
lr_target <- bc_tst$bc_target
lr_predicted <- rep(0, length(lr_target))
lr_predicted[which(lr_response >= 0.5)] <- 1
cm_full <- table(lr_target, lr_predicted)
cm_full

perf_mat[1,] <- perf_eval2(cm_full)
perf_mat


#reQ6
req6bc_data_cleared<-bc_data_cleared[,c(1,2,3,4,5,6,7,8,9,10,31)]
set.seed(12345)
trn_idx <- sample(1:nrow(req6bc_data_cleared), round(0.7*nrow(req6bc_data_cleared)))
bc_trn <- req6bc_data_cleared[trn_idx,]
bc_tst <- req6bc_data_cleared[-trn_idx,]

# Train the Logistic Regression Model with all variables
full_lr <- glm(bc_target ~ . , family=binomial, bc_trn)
summary(full_lr)

#트레이닝셋 검증
lr_response_train <- predict(full_lr, type = "response", newdata = bc_trn)
lr_target_train  <- bc_trn$bc_target
lr_predicted_train  <- rep(0, length(lr_target_train ))
lr_predicted_train [which(lr_response_train  >= 0.5)] <- 1
cm_full_train  <- table(lr_target_train , lr_predicted_train )
cm_full_train

perf_mat_train <- matrix(0, 1, 6)
colnames(perf_mat_train) <- c("TPR (Recall)", "Precision", "TNR", "ACC", "BCR", "F1")
rownames(perf_mat_train) <- "Logstic Regression"


perf_mat_train[1,] <- perf_eval2(cm_full_train)
perf_mat_train

#테스트셋 검증
lr_response_test <- predict(full_lr, type = "response", newdata = bc_tst)
lr_target_test <- bc_tst$bc_target
lr_predicted_test <- rep(0, length(lr_target_test))
lr_predicted_test[which(lr_response_test >= 0.5)] <- 1
cm_full_test <- table(lr_target_test, lr_predicted_test)
cm_full_test

perf_mat_test <- matrix(0, 1, 6)
colnames(perf_mat_test) <- c("TPR (Recall)", "Precision", "TNR", "ACC", "BCR", "F1")
rownames(perf_mat_test) <- "Logstic Regression"

perf_mat_test[1,] <- perf_eval2(cm_full_test)
perf_mat_test

auroc <- function(lr_r,lr_t){
  TPR<-numeric(15000)
  FPR<-numeric(15000)
  aucval=0
  for (i in 1:14999){
    
    lr_pr<- rep(0, length(lr_t))
    lr_pr[which(lr_r >= i/15000)] <- 1
    cm <- table(lr_t, lr_pr)
    # True positive rate: TPR (Recall)
    TPR[i] <- cm[2,2]/sum(cm[2,])
    
    # False postitive rate : FPR
    FPR[i] <- cm[1,2]/sum(cm[1,])
    
    if (i>1) {
      aucval<-aucval+(FPR[i-1]-FPR[i])*(mean(TPR[i],TPR[i-1]))
    }
    
    
  }
  TP_FP_D<-data.frame(FPR,TPR)
  ggplot(TP_FP_D,aes(x=FPR,y=TPR))+
    geom_ribbon(aes(ymax=TPR, ymin=0), fill = "slategray4")+
    geom_path(color="orangered")
  print(aucval)  
}

auroc(lr_response_train,lr_target_train)
auroc(lr_response_test,lr_target_test)


#Q7
set.seed(12345)
trn_idx <- sample(1:nrow(q7bc_data_cleared), round(0.7*nrow(q7bc_data_cleared)))
q7bc_trn <- q7bc_data_cleared[trn_idx,]
q7bc_tst <- q7bc_data_cleared[-trn_idx,]

# Train the Logistic Regression Model with all variables
q7full_lr <- glm(bc_target ~ . , family=binomial, q7bc_trn)
summary(q7full_lr)

#학습데이터
q7lr_response_trn <- predict(q7full_lr, type = "response", newdata = q7bc_trn)
q7lr_target <- q7bc_trn$bc_target
q7lr_predicted <- rep(0, length(q7lr_target))
q7lr_predicted[which(q7lr_response_trn >= 0.5)] <- 1
q7cm_full <- table(q7lr_target, q7lr_predicted)
q7cm_full

q7perf_mat_test <- matrix(0, 1, 6)
colnames(q7perf_mat_test) <- c("TPR (Recall)", "Precision", "TNR", "ACC", "BCR", "F1")
rownames(q7perf_mat_test) <- "Logstic Regression"

q7perf_mat_test[1,] <- perf_eval2(q7cm_full)
q7perf_mat_test


#테스트 데이터

q7lr_response_tst <- predict(q7full_lr, type = "response", newdata = q7bc_tst)
q7lr_target <- q7bc_tst$bc_target
q7lr_predicted <- rep(0, length(q7lr_target))
q7lr_predicted[which(q7lr_response_tst >= 0.5)] <- 1
q7cm_full <- table(q7lr_target, q7lr_predicted)
q7cm_full

q7perf_mat_test <- matrix(0, 1, 6)
colnames(q7perf_mat_test) <- c("TPR (Recall)", "Precision", "TNR", "ACC", "BCR", "F1")
rownames(q7perf_mat_test) <- "Logstic Regression"

q7perf_mat_test[1,] <- perf_eval2(q7cm_full)
q7perf_mat_test

#AUROC
auroc(q7lr_response_trn,lr_target_train)
auroc(q7lr_response_tst,lr_target_test)


#Q8

full_lr <- glm(bc_target ~ . , family=binomial, bc_trn)
summary(full_lr)

step(full_lr, direction = "both")

q8bc_data_cleared<-bc_data_cleared[,c(1,2,3,4,7,8,10,31)]

lr_response <- predict(full_lr, type = "response", newdata = bc_tst)
lr_target <- bc_tst$bc_target
lr_predicted <- rep(0, length(lr_target))
lr_predicted[which(lr_response >= 0.5)] <- 1
cm_full <- table(lr_target, lr_predicted)
cm_full

perf_mat[1,] <- perf_eval2(cm_full)
perf_mat
