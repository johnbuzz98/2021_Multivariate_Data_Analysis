library(nnet) 
library(tree)
library(rpart)
library(rpart.plot)
library(caret)
library(doParallel)
library(adabag)
library(mlbench)
library(randomForest)
library(ipred)
library(gbm)
library(ROSE)

perf_eval_multi <- function(cm){ 
  ACC = sum(diag(cm))/sum(cm)
  BCR = 1
  for (i in 1:dim(cm)[1]){
    BCR = BCR*(cm[i,i]/sum(cm[i,])) 
  }
  BCR = BCR^(1/dim(cm)[1])
  return(c(ACC, BCR))
}

perf_eval_multi2 <- function(act, pred){
  sum <- cbind(act, pred)
  colnames(sum) <- c("actual","predict")
  sum2 <- as.data.frame(sum)
  ACC = length(which(sum2$actual==sum2$predict))/nrow(sum2)
  BCR = 1
  for (i in unique(sum2$actual)){
    BCR = BCR*(length(which(sum2$actual==i & sum2$predict==i))/length(which(sum2$actual==i)))
  }
  
  BCR = BCR^(1/length(unique(sum2$actual)))
  return(c(ACC, BCR))
}

cl <- makeCluster(4) 
registerDoParallel(cl)


l<-as.factor(c(1:3)) #(정사각 cfm 을 만들어주기 위해서)

# Load data & Preprocessing
data <- read.csv("Earthquake_Damage.csv") 
dim(data)
Nrow <- dim(data)[1]
Ncol <- dim(data)[2]


#non_numeric index
factor_col = c()           
for (i in c(1:Ncol)){
  if (is.numeric(data[,i])==FALSE){
    factor_col = c(factor_col, i)
  }
}
factor_col

input_data <- data[,-40]
scal <- input_data[,-factor_col]

data_scaling <- scale(scal, center = TRUE, scale = TRUE)
data_factor <- data[,factor_col]
data_target <- data[,40]

#one_hot_encoding
for (i in c(1:ncol(data_factor))){
  for(x in unique(data_factor[,i])){
    data_factor[paste(colnames(data_factor[i]),x, sep = "_")] <- ifelse(data_factor[,i] == x, 1, 0)
  }
}

#final_data
data_normal <- data.frame(data_factor,data_scaling, Class = data_target)
data_normal <- data_normal[,-c(1:8)]

#train, validation, test set split
set.seed(123456)                                        
tav_idx <- sample(1:Nrow, size=200000)
data_tav <- data_normal[tav_idx,]
data_tst <- data_normal[-tav_idx,]
val_idx <- sample(tav_idx,size=50000)
data_val <- data_normal[val_idx,]
data_trn <- data_normal[tav_idx[!(tav_idx%in%val_idx)],]

trn_target <- data_trn[,69]
trn_input <- data_trn[,-69]
val_target <- data_val[,69]
val_input <- data_val[,-69]
tav_target <- data_tav[,69]
tav_input <- data_tav[,-69]
tst_target <- data_tst[,69]
tst_input <- data_tst[,-69]

#target one hot enconding ver
class_target <- class.ind(data_normal[,69])
data_class <- data.frame(data_normal[,-69], Class = class_target)
data_class_tav <- data_class[tav_idx,]
data_class_tst <- data_class[-tav_idx,]
data_class_val <- data_class[val_idx,]
data_class_trn <- data_class[tav_idx[!(tav_idx%in%val_idx)],]


# target/input 분할
trn_target_class <- data_class_trn[,c(69,70,71)]
trn_input_class <- data_class_trn[,-c(69,70,71)]
val_target_class <- data_class_val[,c(69,70,71)]
val_input_class <- data_class_val[,-c(69,70,71)]
tav_target_class <- data_class_tav[,c(69,70,71)]
tav_input_class <- data_class_tav[,-c(69,70,71)]
tst_target_class <- data_class_tst[,c(69,70,71)]
tst_input_class <- data_class_tst[,-c(69,70,71)]






#________________________________Q1________________________________
#performance evalutaion matrix
perf_summary<-matrix(0,nrow=3,ncol=2)
colnames(perf_summary)<-c("ACC","BCR")
rownames(perf_summary)<-c("Multinomial LR", "ANN","CART")
#Multinomial logistic regression 
start <- proc.time()
mlr <- multinom(Class ~ ., data = data_tav)
proc.time() - start

mlr_prey <- predict(mlr, newdata = data_tst)
mlr_cfmatrix <- table(data_tst$Class, mlr_prey)
mlr_cfmatrix

perf_summary[1,] <- perf_eval_multi(mlr_cfmatrix)
perf_summary


#ANN
#hyperparameter 후보군 중 최적 조합 찾기
nH <- seq(7,11,2)
max_iter <- seq(100,200,50)
rang <- seq(0.1, 0.7, 0.3)
mat_ANN <- matrix(0,length(nH)*length(max_iter)*length(rang),5)
colnames(mat_ANN) <- c("Hidden Nodes", "Max Iteration","rang", "ACC", "BCR")
mat_ANN

start <- proc.time()
n<-1
for (i in 1:length(nH)) {
  cat("Training ANN: the number of hidden nodes:", nH[i], "\n")
  for (j in 1:length(max_iter)) {
    cat("Training ANN: the number of max iteration: ", max_iter[j], "\n")
    for (k in 1:length(rang)){
      tmp_nnet <- nnet(trn_input_class,trn_target_class, size = nH[i], decay = 5e-4, maxit = max_iter[j], rang=rang[k])
      prey <- predict(tmp_nnet, val_input_class)
      mat_ANN[n,1:3] <- c(nH[i],max_iter[j],rang[k])
      mat_ANN[n,4:5] <- perf_eval_multi2(max.col(val_target_class), max.col(prey))    
      n <- n+1
    }
  }
}
proc.time() - start


#ACC 기준의 성능 평가
ordered_ACC <- mat_ANN[order(mat_ANN[,4], decreasing = TRUE),]    
colnames(ordered_ACC) <- c("Hidden Nodes", "Max Iteration","rang", "ACC", "BCR")
ordered_ACC[c(1:5),]

#BCR 기준의 성능 평가
ordered_BCR <- mat_ANN[order(mat_ANN[,5], decreasing = TRUE),]    
colnames(ordered_BCR) <- c("Hidden Nodes", "Max Iteration","rang", "ACC", "BCR")
ordered_BCR[c(1:5),]

mat_ANN_final <- matrix(0,2,5)
colnames(mat_ANN_final) <- c("Hidden Nodes", "Max Iteration","rang", "ACC", "BCR")

#ACC 기준 
start <- proc.time()
nnet_ACC <- nnet(tav_input_class, tav_target_class, size = 9 , maxit =150, rang = 0.7, decay = 5e-4 )
prey_ACC <- predict(nnet_ACC, tst_input_class)
proc.time() - start

mat_ANN_final[1,1:3] <- c(9,150,0.7)
mat_ANN_final[1,4:5] <- perf_eval_multi2(max.col(tst_target_class), max.col(prey_ACC))

#BCR 기준 
start <- proc.time()
nnet_BCR <- nnet(tav_input_class, tav_target_class, size = 9 ,maxit =200, rang = 0.1, decay = 5e-4)
prey_BCR <- predict(nnet_BCR, tst_input_class)
proc.time() - start

mat_ANN_final[2,1:3] <- c(9,200,0.1)
mat_ANN_final[2,4:5] <- perf_eval_multi2(max.col(tst_target_class), max.col(prey_BCR))
mat_ANN_final

perf_summary[2,]<-mat_ANN_final[2,4:5]
perf_summary


#CART
CART.trn = data.frame(trn_input, Class = as.factor(trn_target))
CART.val = data.frame(val_input, Class = as.factor(val_target))
CART.tav = data.frame(tav_input, Class = as.factor(tav_target))
CART.tst = data.frame(tst_input, Class = as.factor(tst_target))

minsplit <- seq(5,35,10)
maxdepth <- seq(10,50,20)
cp <- seq(0.01,0.03,0.01)

mat_CART = matrix(0,length(minsplit)*length(maxdepth)*length(cp),5)
colnames(mat_CART) <- c("minsplit", "maxdepth", "cp", "ACC", "BCR")
mat_CART

ctrl = rpart.control
start <- proc.time()
n<-1
for (i in 1:length(minsplit)) {
  cat("Training CART: the number of minsplit:", minsplit[i], "\n")
  for (j in 1:length(maxdepth)) {
    cat("Training CART: the number of maxdepth: ", maxdepth[j], "\n")
    for (k in 1:length(cp)){
      ctrl <- rpart.control(minsplit[i],maxdepth[j],cp[k])
      tmp_CART <- rpart(Class ~ ., CART.trn, control = ctrl)
      prey <- predict(tmp_CART, CART.val)
      mat_CART[n,1:3] <- c(minsplit[i],maxdepth[j],cp[k])
      tmp_cm <- table(CART.val$Class,round(prey))
      mat_CART[n,4:5] <- perf_eval_multi2(CART.val$Class,round(prey))   
      n <- n+1
    }
  }
}

CART_ACC <- mat_CART[order(mat_CART[,4], decreasing = TRUE),]    
CART_ACC[,]

ctrl1 <- rpart.control(minsplit=5, maxdepth=10, cp=0.01)
CART_best <- rpart(Class ~ ., CART.tav, control=ctrl1)
prey_CART <- predict(CART_best, newdata= CART.tst)
cm_CART <- table(CART.tst$Class, round(prey_CART))
perf_summary[3,]<-perf_eval_multi2(CART.tst$Class, round(prey_CART))
perf_summary


#______________________Q2______________________
#CART BAGGING

mat_CART.bagging = matrix(0,4,3)
colnames(mat_CART.bagging) <- c("No. of Bootstrap", "ACC", "BCR")
mat_CART.bagging

mf <- seq(30,300,90)
ctrl = rpart.control

start <- proc.time()
n<-1
for (i in 1:4) {
  cat("CART Bagging Bootstrap : ", mf[i], "\n")
  ctrl <- rpart.control(minsplit = 5, maxdepth = 10, cp = 0.01)
  CART.bagging <- bagging(Class ~ ., CART.trn, control = ctrl, nbagg = mf[i], coob=TRUE)
  prey <- predict(CART.bagging, CART.val)
  mat_CART.bagging[n,1] <- mf[i]
  tmp_cm <- table(CART.val$Class,round(prey))
  print(tmp_cm)
  mat_CART.bagging[n,2:3] <- perf_eval_multi2(CART.val$Class,round(prey))
  print(mat_CART.bagging)
  n <- n+1
}
mat_CART.bagging
Time <- proc.time() - start
Time

#최적의 값 test
ctrl <- rpart.control(minsplit = 5, maxdepth = 10, cp = 0.01)
CART.bagging <- bagging(Class ~ ., CART.tav, control = ctrl, nbagg = 30, coob=TRUE)
prey <- predict(CART.bagging, CART.tst, type = "class")
tmp_cm <- table(CART.tst$Class,round(prey))
print(tmp_cm)
perf_eval_multi2(CART.tst$Class,round(prey))


#____________________________Q3______________________________
#Random Forest
RF.trn <- CART.trn
RF.tst <- CART.tst
RF.val <- CART.val
RF.tav <- CART.tav

ntr <- seq(30,300,30)
mat_RF = matrix(0,10,3)
colnames(mat_RF) <- c("No. of Tree", "ACC", "BCR")
mat_RF

ptm <- proc.time()
n<-1
for (i in 1:length(ntr)){
  RF.model <- randomForest(Class ~ ., data = RF.trn, ntree = ntr[i], importance = TRUE, do.trace = TRUE)
  
  # Check the result
  print(RF.model)
  plot(RF.model)  
  
  # Variable importance
  Var.imp <- importance(RF.model)
  barplot(Var.imp[order(Var.imp[,4], decreasing = TRUE),4])
  
  # Prediction
  RF.prey <- predict(RF.model, newdata = RF.val, type = "class")
  RF.cfm <- table(RF.prey, RF.val$Class)
  
  mat_RF[n,1] <- ntr[i]
  mat_RF[n,2:3] <- perf_eval_multi(RF.cfm)    
  RF.cfm
  mat_RF
  n <- n+1
} 
mat_RF
RF.Time <- proc.time() - ptm
RF.Time

#베스트 모델의 평가
RF.model <- randomForest(Class ~ ., data = RF.tav, ntree = 300 , importance = TRUE, do.trace = TRUE)

print(RF.model)
plot(RF.model)  

Var.imp <- importance(RF.model)
barplot(Var.imp[order(Var.imp[,4], decreasing = TRUE),4])
head(Var.imp[order(Var.imp[,4], decreasing = TRUE),4])

RF.prey <- predict(RF.model, newdata = RF.tst, type = "class")
RF.cfm <- table(RF.prey, RF.tst$Class)
perf_eval_multi2(RF.prey, RF.tst$Class)
perf_eval_multi(RF.cfm)
RF.cfm


## CART Bagging and Random Forest Performance Measure Plot
x <- seq(30,300,30)
bag_acc <- rep(0.64492,10)
bag_BCR <- rep(0.4074415,10)
RF_acc <- c(0.7055824, 0.7052029, 0.7099338, 0.7089982, 0.7107638, 0.7095757, 0.7097648, 0.7094602, 0.7108959, 0.7121170)
RF_BCR <- c(0.7028603, 0.7052618, 0.7099243, 0.7081217, 0.7104599, 0.7089964, 0.7089964, 0.7098866, 0.7118976, 0.7123419)

plot(bag_acc,axes=F, main = "ACC Plot",xlab="number of bootstrap",ylab="ACC",type='o', col='green', ylim = c(0.5,0.8))
axis(1, at = 1:10, lab=x)
axis(2, ylim=c(0.4,0.8))
lines(RF_acc, type='o', col='gray')

plot(bag_BCR,axes=F, main = "BCR Plot", xlab="number of bootstrap",ylab="BCR",type='o', col='green', ylim = c(0.1,0.8))
axis(1, at = 1:10, lab=x)
axis(2, ylim=c(0.3,0.6))
lines(RF_BCR, type='o', col='gray')



#______________________Q4_________________________
#ANN 30회 반복 수행

mat_ANN_final <- matrix(0,30,3)
colnames(mat_ANN_final) <- c("No.", "ACC", "BCR")

start <- proc.time()
n <- 1
for (i in 1:30){
  final_nnet <- nnet(tav_input_class,tav_target_class, size = 9 , rang = 0.1, decay = 5e-4, maxit =200 )
  final_prey <- predict(final_nnet, tst_input_class)
  mat_ANN_final[n,1] <- i
  mat_ANN_final[n,2:3] <- perf_eval_multi2(tst_target_class, max.col(final_prey))
  table(tst_target_class,max.col(final_prey))
  mat_ANN_final
  n <- n+1
}
proc.time() - start
mat_ANN_final

mean(mat_ANN_final[,2])
var(mat_ANN_final[,2])
mean(mat_ANN_final[,3])
var(mat_ANN_final[,3])


#_______________________________Q5____________________________
#ANN Bagging

#데이터셋 축소
set.seed(111)             

trn_idx_red <- sample(1:nrow(data_trn), size=1500)
val_idx_red <- sample(1:nrow(data_val), size=500)
tst_idx_red <- sample(1:nrow(data_tst), size=606)

data_trn_red <- data_trn[trn_idx_red,]
data_val_red <- data_val[val_idx_red,]
data_tst_red <- data_tst[tst_idx_red,]
data_tav_red <- rbind(data_trn_red,data_val_red)

trn_target_red <- data_trn_red[,69]
trn_input_red <- data_trn_red[,-69]
val_target_red <- data_val_red[,69]
val_input_red <- data_val_red[,-69]
tav_target_red <- data_tav_red[,69]
tav_input_red <- data_tav_red[,-69]
tst_target_red <- data_tst_red[,69]
tst_input_red <- data_tst_red[,-69]

#축소된 데이터셋을 이용하기 위한 CART data set 
CART.trn.red = data.frame(trn_input_red, Class = as.factor(trn_target_red))
CART.val.red = data.frame(val_input_red, Class = as.factor(val_target_red))
CART.tav.red = data.frame(tav_input_red, Class = as.factor(tav_target_red))
CART.tst.red = data.frame(tst_input_red, Class = as.factor(tst_target_red))


#bootstrap 30 반복
#Training

mat_ANN.Bagging= matrix(0,30,3)
colnames(mat_ANN.Bagging) <- c("Iter. No.", "ACC", "BCR")
mat_ANN.Bagging

boots <- seq(30,300,30)
ptm <- proc.time()
n<-1
for (i in (1:30)){
  cat("ANN Bagging iteration : ",i, "\n")
  Bagging.ANN.model <- avNNet(trn_input_red, as.factor(trn_target_red), size = 9, decay = 5e-4, maxit = 200,
                              repeats = boots[i], bag = TRUE, allowParallel = TRUE, trace = TRUE)
  Bagging.Time <- proc.time() - ptm
  Bagging.Time
  
  Bagging.ANN.prey <- predict(Bagging.ANN.model, newdata = val_input_red)
  Bagging.ANN.cfm <- table(val_target_red, max.col(Bagging.ANN.prey))
  Bagging.ANN.cfm
  
  mat_ANN.Bagging[n,1] <- i
  mat_ANN.Bagging[n,2:3] <- perf_eval_multi2(val_target_red, max.col(Bagging.ANN.prey))    
  n <- n+1
  
} 
mat_ANN.Bagging
mean(mat_ANN.Bagging[,2])
var(mat_ANN.Bagging[,2])
mean(mat_ANN.Bagging[,3])
var(mat_ANN.Bagging[,3])

#Testing
best_Bagging.ANN.model <- avNNet(tav_input_red, as.factor(tav_target_red), size = 9, decay = 5e-4, maxit = 200,
                                 repeats = 240, bag = TRUE, allowParallel = TRUE, trace = TRUE)


best_Bagging.ANN.prey <- predict(best_Bagging.ANN.model, newdata = tst_input_red)
Bagging.ANN.cfm <- table(max.col(tst_target_red), max.col(best_Bagging.ANN.prey))
Bagging.ANN.cfm
perf_eval_multi2(tst_target_red, max.col(best_Bagging.ANN.prey))
table(tst_target_red, max.col(best_Bagging.ANN.prey))



#_____________________________Q6____________________________________
#Adaptive Boosting

bag_ctrl <- rpart.control(minsplit=10, maxdepth=10, cp=0.01)

ada_10 <- boosting(Class~., CART.trn, boos=TRUE, mfinal=10, control=bag_ctrl)
ada_20 <- boosting(Class~., CART.trn, boos=TRUE, mfinal=20, control=bag_ctrl)
ada_30 <- boosting(Class~., CART.trn, boos=TRUE, mfinal=30, control=bag_ctrl)
ada_40 <- boosting(Class~., CART.trn, boos=TRUE, mfinal=40, control=bag_ctrl)
ada_50 <- boosting(Class~., CART.trn, boos=TRUE, mfinal=50, control=bag_ctrl)


pred_10 <- predict.boosting(ada_10, CART.val)
pred_20 <- predict.boosting(ada_20, CART.val)
pred_30 <- predict.boosting(ada_30, CART.val)
pred_40 <- predict.boosting(ada_40, CART.val)
pred_50 <- predict.boosting(ada_50, CART.val)

ada_mat <- matrix(0,5,3)
colnames(ada_mat) <- c("mfinal", "ACC", "BCR")
ada_mat[1,2:3] <- perf_eval_multi(t(pred_10$confusion))
ada_mat[2,2:3] <- perf_eval_multi(t(pred_20$confusion))
ada_mat[3,2:3] <- perf_eval_multi(t(pred_30$confusion))
ada_mat[4,2:3] <- perf_eval_multi(t(pred_40$confusion))
ada_mat[5,2:3] <- perf_eval_multi(t(pred_50$confusion))

ada_mat[1,1] <- 10
ada_mat[2,1] <- 20
ada_mat[3,1] <- 30
ada_mat[4,1] <- 40
ada_mat[5,1] <- 50

t(pred_10$confusion)
t(pred_20$confusion)
t(pred_30$confusion)
t(pred_40$confusion)
t(pred_50$confusion)

#최적의 조합으로 test set으로 성능 평가
ada_best <- boosting(Class~., CART.tav, boos=TRUE, mfinal=50, control=bag_ctrl)
pred_best <- predict.boosting(ada_best, CART.tst)
perf_eval_multi(t(pred_best$confusion))



#_____________________Q7_________________________
#GBM

GBM.trn <- data.frame(trn_input, Class = trn_target)
GBM.val <- data.frame(val_input, Class = val_target)
GBM.tst <- data.frame(tst_input, Class = tst_target)
GBM.tav <- data.frame(tav_input, Class = tav_target)

# Training the GBM
nt <- seq(500,1000,250)
sr <- seq(0.03,0.09, 0.03)
bf <- seq(0.6, 0.8, 0.1)

gbm_mat <- matrix(0,length(nt)*length(sr)*length(bf),5)
colnames(gbm_mat) <- c("n.tress","shrinkage","big.fraction", "ACC", "BCR")

ptm <- proc.time()

n<-1
for (i in 1:length(nt)) {
  cat("Training gbm: the number of trees:", nt[i], "\n")
  for (j in 1:length(sr)) {
    cat("Training gbm: shrinkage is: ", sr[j], "\n")
    for (k in 1:length(bf)){
      GBM.model <- gbm.fit(GBM.trn[,1:68], GBM.trn[,69], distribution = "gaussian", 
                           n.trees = nt[i], shrinkage = sr[j], bag.fraction = bf[k], nTrain = 500)
      GBM.prey <- predict(GBM.model, GBM.val[,1:68], type = "response")
      GBM.prey <- round(GBM.prey)
      GBM.cfm <- table(GBM.prey, GBM.val$Class)
      GBM.cfm
      perf_eval_multi(GBM.cfm)
      gbm_mat[n,1:3] <- c(nt[i],sr[j],bf[k])
      gbm_mat[n,4:5] <- perf_eval_multi(GBM.cfm)    
      n <- n+1
    }
  }
}
gbm_mat
GBM.Time <- proc.time() - ptm
GBM.Time

gbm_ACC <- gbm_mat[order(gbm_mat[,4], decreasing = TRUE),]    
colnames(gbm_ACC) <- c("n.tress","shrinkage","big.fraction", "ACC", "BCR")
gbm_ACC[1,]


#최적의 parameter로 testing 
ptm <- proc.time()

GBM.best <- gbm.fit(GBM.tav[,1:68], GBM.tav[,69], distribution = "gaussian", 
                    n.trees = 750, shrinkage = 0.09, bag.fraction = 0.6, nTrain = 500)
summary(GBM.best)

GBM.prey <- predict(GBM.best, GBM.tst[,1:68], type = "response")
GBM.prey <- round(GBM.prey)
GBM.cfm <- table(GBM.prey, GBM.tst$Class)
perf_eval_multi(GBM.cfm)
GBM.cfm
GBM.Time <- proc.time() - ptm
GBM.Time


#_____________________EXTRA Q___________________________
#Down Sampling
table(CART.tav$Class)

tree_1 <- CART.tav[which(CART.tav$Class==1),]
tree_2 <- CART.tav[which(CART.tav$Class==2),]
tree_3 <- CART.tav[which(CART.tav$Class==3),]
tree2_red <- tree_2[sample(1:nrow(tree_2), size = 19300),]
tree3_red <- tree_3[sample(1:nrow(tree_3), size = 19300),]
tree_red <- rbind(tree_1,tree2_red,tree3_red)
table(tree_red$Class)
str(tree_red$Class)

down <- randomForest(Class ~ ., data = tree_red, ntree = 300, importance = TRUE, do.trace = TRUE)
final_rf_pred <- predict(down, newdata = CART.tst, type = "class")
perf_eval_multi2(CART.tst$Class, final_rf_pred)
table(CART.tst$Class, final_rf_pred)

#Up Sampling

# 종속 변수 비율 조정 
tree_12 <- rbind(tree_1,tree_2)
tree_23 <- rbind(tree_3,tree_2)
table(tree_23$Class)
up_data_12 <- ovun.sample(Class~., data = tree_12, method = "both", N=160000)$data
up_data_23 <- ovun.sample(Class~., data = tree_23, method = "both", N=160000)$data
table(up_data_23$Class)
up_data <- rbind(up_data_12,up_data_23[which(up_data_23$Class==3),])
table(up_data$Class)
str(up_data$Class)

up <- randomForest(Class ~ ., data = up, ntree = 300,  importance = TRUE, do.trace = TRUE)
final_up <- predict(up, newdata = CART.tst, type = "class")
perf_eval_multi2(CART.tst$Class, final_up)
cm<- table(CART.tst$Class, final_up)
cm1 <- cbind(cm[,2],cm[,1],cm[,3])
perf_eval_multi(cm1)
