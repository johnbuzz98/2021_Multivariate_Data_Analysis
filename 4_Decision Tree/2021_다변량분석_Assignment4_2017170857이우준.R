#install.packages("tree")
#install.packages("party")
#install.packages("ROCR")
#install.packages("sandwich")
library(tree)
library(party)
library(ROCR)


# Performance Evaluation Function
perf_eval <- function(cm){
  
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
  #F1-Measure
  F1 <- 2*TPR*PRE/(TPR+PRE)
  
  return(c(TPR, PRE, TNR, ACC, BCR, F1))
}

# 이상치 기준 분류 만들기 함수

boxplot_outliers <-function(target){
  valuelist <-boxplot(target)$stats
  return(c(valuelist[1,],valuelist[5,]))
}



# Performance table
perf_table_d1 <- matrix(0, nrow = 3, ncol = 7)
rownames(perf_table_d1) <- c("Non-Pruning","Post-Pruning","Best_Tree_Tr&Val")
colnames(perf_table_d1) <- c("TPR", "Precision", "TNR", "Accuracy", "BCR", "F1-Measure","AUROC")

perf_table_d2 <- matrix(0, nrow = 3, ncol = 7)
rownames(perf_table_d2) <- c("Non-Pruning","Post-Pruning","Best_Tree_Tr&Val")
colnames(perf_table_d2) <- c("TPR", "Precision", "TNR", "Accuracy", "BCR", "F1-Measure","AUROC")


# Load data1 & Preprocessing
  
  data1 <- read.csv("data.csv")
  data1_input_idx <- c(3:32)
  data1_target_idx <- 2
  data1_input <- data1[,data1_input_idx]
  data1_target <- as.factor(data1[,data1_target_idx])
  data1_origin <- data.frame(data1_input, diagnosis = as.factor(data1_target))
  
  #이상치 제거
  data1_outliers <- matrix(0, nrow = 30, ncol = 2)
  rownames(data1_outliers) <- names(data1_input)
  colnames(data1_outliers) <- c("LCL", "UCL")
  
  for(i in 1:30){
    data1_outliers[i,]<-boxplot_outliers(data1_origin[,i])
  }
  
  for (i in 1:30){
    data1_origin[,i]<-ifelse(data1_origin[,i] < data1_outliers[i,1] | data1_origin[,i] > data1_outliers[i,2], NA,data1_origin[,i])
  }
  
  data1_origin<-na.omit(data1_origin) 
  
  
  ndata1 <- nrow(data1_origin)
  
  #Training, Validating, Testing Set 분리
  set.seed(123456)
  data1_idx <- sample(1:ndata1, round(0.6*ndata1))
  data1_trn <- data1_origin[data1_idx,]
  data1_temp <-data1_origin[-data1_idx,]
  set.seed(123456)
  data1_idx <- sample(1:nrow(data1_temp), round(0.375*nrow(data1_temp)))
  data1_val<- data1_temp[data1_idx,]
  data1_tst<- data1_temp[-data1_idx,]
  
  
  # Load data2 & Preprocessing
  data2 <- read.csv("Challenger_Ranked_Games.csv")
  data2_input_idx <- c(2,4:26,33:38,41:50)
  data2_target_idx <- c(3)
  data2_input <- data2[,data2_input_idx]
  data2_input[,c(2:6)]<-data.frame(apply(data2_input[,c(2:6)], 2, as.factor))
  data2_target <- as.factor(data2[,data2_target_idx])
  data2_origin <- data.frame(data2_input, blueWins = data2_target)
  
  #이상치 제거
  data2_outliers <- matrix(0, nrow = 40, ncol = 2)
  rownames(data2_outliers) <- names(data2_input)
  colnames(data2_outliers) <- c("LCL", "UCL")
  
  for(i in c(1,7:40)){
    data2_outliers[i,]<-boxplot_outliers(data2_origin[,i])
  }
  
  for (i in c(1,7:40)){
    data2_origin[,i]<-ifelse(data2_origin[,i] < data2_outliers[i,1] | data2_origin[,i] > data2_outliers[i,2], NA,data2_origin[,i])
  }
  data2_origin<-na.omit(data2_origin)
  ndata2 <- nrow(data2_origin)

#Training, Validating, Testing Set 분리
set.seed(123456)
data2_idx <- sample(1:ndata2, round(0.6*ndata2))
data2_trn <- data2_origin[data2_idx,]
data2_temp <-data2_origin[-data2_idx,]
set.seed(123456)
data2_idx <- sample(1:nrow(data2_temp), round(0.375*nrow(data2_temp)))
data2_val<- data2_temp[data2_idx,]
data2_tst<- data2_temp[-data2_idx,]

#_____________________________Q2_____________________________________
#dataset1
# Training the tree
data1_tree_q2 <- tree(diagnosis ~ ., data1_trn)
summary(data1_tree_q2)

# Plot the tree
plot(data1_tree_q2)
text(data1_tree_q2, pretty = 1)

# Performance of the tree
data1_tree_q2_prediction<-predict(data1_tree_q2, newdata = data1_tst,type="class")
data1_tree_q2_cm <- table(data1_tst$diagnosis, data1_tree_q2_prediction)
data1_tree_q2_cm


perf_table_d1[1,c(1:6)]<-perf_eval(data1_tree_q2_cm)
data1_q2_pred <- prediction(as.numeric(data1_tree_q2_prediction),as.numeric(data1_tst$diagnosis))
perf_table_d1[1,7]<-as.numeric(performance(data1_q2_pred, measure = "auc")@y.values)
perf_table_d1

#dataset2
# Training the tree
data2_tree_q2 <- tree(blueWins ~ ., data2_trn)
summary(data2_tree_q2)

# Plot the tree
plot(data2_tree_q2)
text(data2_tree_q2, pretty = 1)

# Performance of the tree
data2_tree_q2_prediction<-predict(data2_tree_q2, newdata = data2_tst,type="class")
data2_tree_q2_cm <- table(data2_tst$blueWins, data2_tree_q2_prediction)
data2_tree_q2_cm


perf_table_d2[1,c(1:6)]<-perf_eval(data2_tree_q2_cm)
data2_q2_pred <- prediction(as.numeric(data2_tree_q2_prediction),as.numeric(data2_tst$blueWins))
perf_table_d2[1,7]<-as.numeric(performance(data2_q2_pred, measure = "auc")@y.values)
perf_table_d2

#____________________________Q3-1_______________________________

#data1
set.seed(123456)
data1_tree_q3<-cv.tree(data1_tree_q2, FUN=prune.misclass)
plot(data1_tree_q3)
#size 4 is optimal
data1_tree_q3_pp<-prune.misclass(data1_tree_q2, best=4)
plot(data1_tree_q3_pp)
text(data1_tree_q3_pp, pretty = 1)

# Performance of the tree
data1_tree_q3_prediction<-predict(data1_tree_q3_pp, newdata = data1_tst,type="class")
data1_tree_q3_cm <- table(data1_tst$diagnosis, data1_tree_q3_prediction)
data1_tree_q3_cm

perf_table_d1[2,c(1:6)] <-perf_eval(data1_tree_q3_cm)
data1_q3_pred <-prediction(as.numeric(data1_tree_q3_prediction),as.numeric(data1_tst$diagnosis))
perf_table_d1[2,7]<-as.numeric(performance(data1_q3_pred,measure = "auc")@y.values)
perf_table_d1

#data2
set.seed(123456)
data2_tree_q3<-cv.tree(data2_tree_q2, FUN=prune.misclass)
plot(data2_tree_q3)
#size 5 is optimal
data2_tree_q3_pp<-prune.misclass(data2_tree_q2, best=5)
plot(data2_tree_q3_pp)
text(data2_tree_q3_pp, pretty = 1)

# Performance of the tree
data2_tree_q3_prediction<-predict(data2_tree_q3_pp, newdata = data2_tst,type="class")
data2_tree_q3_cm <- table(data2_tst$blueWins, data2_tree_q3_prediction)
data2_tree_q3_cm

perf_table_d2[2,c(1:6)] <-perf_eval(data2_tree_q3_cm)
data2_q3_pred <-prediction(as.numeric(data2_tree_q3_prediction),as.numeric(data2_tst$blueWins))
perf_table_d2[2,7]<-as.numeric(performance(data2_q3_pred,measure = "auc")@y.values)
perf_table_d2

#____________________________Q3-2_______________________________

# tree parameter settings
d1_min_criterion = c(0.875,0.9, 0.95, 0.975, 0.99)
d1_min_split = c(10,30,50,100,150)
d1_max_depth = c(0,1,3,5,7)
data1_q3_2 = matrix(0,length(d1_min_criterion)*length(d1_min_split)*length(d1_max_depth),11)
colnames(data1_q3_2) <- c("min_criterion", "min_split", "max_depth", 
                                      "TPR", "Precision", "TNR", "ACC", "BCR", "F1", "AUROC", "N_leaves")

iter_cnt = 1

for (i in 1:length(d1_min_criterion)){
  for ( j in 1:length(d1_min_split)){
    for ( k in 1:length(d1_max_depth)){
      
      cat("Min criterion:", d1_min_criterion[i], ", Min split:", d1_min_split[j], ", Max depth:", d1_max_depth[k], "\n")
      tmp_control = ctree_control(mincriterion = d1_min_criterion[i], minsplit = d1_min_split[j], maxdepth = d1_max_depth[k])
      tmp_tree <- ctree(diagnosis ~ ., data = data1_trn, controls = tmp_control)
      tmp_tree_val_prediction <- predict(tmp_tree, newdata = data1_val)
      tmp_tree_val_response <- treeresponse(tmp_tree, newdata = data1_val)
      tmp_tree_val_prob <- 1-unlist(tmp_tree_val_response, use.names=F)[seq(1,nrow(data1_val)*2,2)]
      tmp_tree_val_rocr <- prediction(tmp_tree_val_prob, data1_val$diagnosis)
      # Confusion matrix for the validation dataset
      tmp_tree_val_cm <- table(data1_val$diagnosis, tmp_tree_val_prediction)
      
      # parameters
      data1_q3_2[iter_cnt,1] = d1_min_criterion[i]
      data1_q3_2[iter_cnt,2] = d1_min_split[j]
      data1_q3_2[iter_cnt,3] = d1_max_depth[k]
      # Performances from the confusion matrix
      data1_q3_2[iter_cnt,4:9] = perf_eval(tmp_tree_val_cm)
      # AUROC
      data1_q3_2[iter_cnt,10] = unlist(performance(tmp_tree_val_rocr, "auc")@y.values)
      # Number of leaf nodes
      data1_q3_2[iter_cnt,11] = length(nodes(tmp_tree, unique(where(tmp_tree))))
      iter_cnt = iter_cnt + 1
    }
  }
}

# Find the best set of parameters
data1_q3_2 <- data1_q3_2[order(data1_q3_2[,10], decreasing = T),]
data1_q3_2
d1_best_criterion <- data1_q3_2[1,1]
d1_best_split <- data1_q3_2[1,2]
d1_best_depth <- data1_q3_2[1,3]



#data2

# tree parameter settings
d2_min_criterion = c(0.875,0.9, 0.95, 0.975, 0.99)
d2_min_split = c(10,30,50,70,100)
d2_max_depth = c(0,1,3,5,7)
data2_q3_2 = matrix(0,length(d2_min_criterion)*length(d2_min_split)*length(d2_max_depth),11)
colnames(data2_q3_2) <- c("min_criterion", "min_split", "max_depth", 
                          "TPR", "Precision", "TNR", "ACC", "BCR", "F1", "AUROC", "N_leaves")

iter_cnt = 1

for (i in 1:length(d2_min_criterion)){
  for ( j in 1:length(d2_min_split)){
    for ( k in 1:length(d2_max_depth)){
      
      cat("Min criterion:", d2_min_criterion[i], ", Min split:", d2_min_split[j], ", Max depth:", d2_max_depth[k], "\n")
      tmp_control = ctree_control(mincriterion = d2_min_criterion[i], minsplit = d2_min_split[j], maxdepth = d2_max_depth[k])
      tmp_tree <- ctree(blueWins ~ ., data = data2_trn, controls = tmp_control)
      tmp_tree_val_prediction <- predict(tmp_tree, newdata = data2_val)
      tmp_tree_val_response <- treeresponse(tmp_tree, newdata = data2_val)
      tmp_tree_val_prob <- 1-unlist(tmp_tree_val_response, use.names=F)[seq(1,nrow(data2_val)*2,2)]
      tmp_tree_val_rocr <- prediction(tmp_tree_val_prob, data2_val$blueWins)
      # Confusion matrix for the validation dataset
      tmp_tree_val_cm <- table(data2_val$blueWins, tmp_tree_val_prediction)
      
      # parameters
      data2_q3_2[iter_cnt,1] = d2_min_criterion[i]
      data2_q3_2[iter_cnt,2] = d2_min_split[j]
      data2_q3_2[iter_cnt,3] = d2_max_depth[k]
      # Performances from the confusion matrix
      data2_q3_2[iter_cnt,4:9] = perf_eval(tmp_tree_val_cm)
      # AUROC
      data2_q3_2[iter_cnt,10] = unlist(performance(tmp_tree_val_rocr, "auc")@y.values)
      # Number of leaf nodes
      data2_q3_2[iter_cnt,11] = length(nodes(tmp_tree, unique(where(tmp_tree))))
      iter_cnt = iter_cnt + 1
    }
  }
}

# Find the best set of parameters
data2_q3_2 <- data2_q3_2[order(data2_q3_2[,10], decreasing = T),]
data2_q3_2
d2_best_criterion <- data2_q3_2[1,1]
d2_best_split <- data2_q3_2[1,2]
d2_best_depth <- data2_q3_2[1,3]


#____________________________________Q4_____________________________________
#data1
#Find the best model
d1_tree_control = ctree_control(mincriterion = d1_best_criterion, minsplit = d1_best_split, maxdepth = d1_best_depth)
data1_tree_q4 <- ctree(diagnosis ~ ., data = data1_trn, controls = d1_tree_control)

# Plot the best tree
plot(data1_tree_q4, type="extended")

#data2
#Find the best model
d2_tree_control = ctree_control(mincriterion = d2_best_criterion, minsplit = d2_best_split, maxdepth = d2_best_depth)
data2_tree_q4 <- ctree(blueWins ~ ., data = data2_trn, controls = d2_tree_control)

# Plot the best tree
plot(data2_tree_q4, type="simple")

#__________________________________Q5________________________________________
#data1
# Use the training and validation dataset to train the best tree
data1_trn_q5 <- rbind(data1_trn, data1_val)

data1_tree_q5 <- ctree(diagnosis ~ ., data = data1_trn_q5, controls = d1_tree_control)
data1_q5_pre_prediction <- predict(data1_tree_q5, newdata = data1_tst)
data1_q5_pre_response <- treeresponse(data1_tree_q5, newdata = data1_tst)

# Performance of the best tree
data1_q5_cm <- table(data1_tst$diagnosis, data1_q5_pre_prediction)
data1_q5_cm

perf_table_d1[3,c(1:6)] <- perf_eval(data1_q5_cm)
data1_q5_pred <-prediction(as.numeric(data1_q5_pre_prediction),as.numeric(data1_tst$diagnosis))
perf_table_d1[3,7]<-as.numeric(performance(data1_q5_pred,measure = "auc")@y.values)
perf_table_d1


#data2
# Use the training and validation dataset to train the best tree
data2_trn_q5 <- rbind(data2_trn, data2_val)

data2_tree_q5 <- ctree(blueWins ~ ., data = data2_trn_q5, controls = d2_tree_control)
data2_q5_pre_prediction <- predict(data2_tree_q5, newdata = data2_tst)
data2_q5_pre_response <- treeresponse(data2_tree_q5, newdata = data2_tst)

# Performance of the best tree
data2_q5_cm <- table(data2_tst$blueWins, data2_q5_pre_prediction)
data2_q5_cm

perf_table_d2[3,c(1:6)] <- perf_eval(data2_q5_cm)
data2_q5_pred <-prediction(as.numeric(data2_q5_pre_prediction),as.numeric(data2_tst$blueWins))
perf_table_d2[3,7]<-as.numeric(performance(data2_q5_pred,measure = "auc")@y.values)
perf_table_d2


#___________________________________Q6_____________________________________
#로지스틱 회귀분석을 진행하기 전 주어진 데이터 셋을 표준화 시켜준다
#data1
data1_trn_q6 <- data.frame(scale(data1_trn_q5[,c(1:30)], center = TRUE, scale = TRUE),diagnosis = as.numeric(data1_trn_q5$diagnosis))
data1_tst_q6 <- data.frame(scale(data1_tst[,c(1:30)], center = TRUE, scale = TRUE),diagnosis = as.numeric(data1_tst$diagnosis))
data1_trn_q6$diagnosis<-ifelse(data1_trn_q6$diagnosis==2,1,0)
data1_tst_q6$diagnosis<-ifelse(data1_tst_q6$diagnosis==2,1,0)

perf_table_d1_q6 <- matrix(0, nrow = 2, ncol = 5)
rownames(perf_table_d1_q6) <- c("Logistic Regression", "Decison Tree")
colnames(perf_table_d1_q6) <- c("TPR", "TNR", "Accuracy", "BCR", "F1-Measure")
perf_table_d1_q6[2,]<-perf_table_d1[3,c(1,3,4,5,6)]
perf_table_d1_q6

#data1 로지스틱 회귀
#상한 & 하한선 설정 for ForwardSelection
data1_tmp_x <- paste(colnames(data1_trn_q6)[-31], collapse=" + ")
data1_tmp_x
data1_tmp_xy <- paste("diagnosis ~ ", data1_tmp_x, collapse = "")
as.formula(data1_tmp_xy)


#Forward Selection
data1_logi_forward <- step(glm(diagnosis~ 1, data = data1_trn_q6), 
                      scope = list(upper = as.formula(data1_tmp_xy), lower = diagnosis~ 1), 
                      direction="forward")
summary(data1_logi_forward)

# Make prediction
data1_forward_prob <- predict(data1_logi_forward, type = "response", newdata = data1_tst_q6)
data1_forward_prey <- rep(0, nrow(data1_tst_q6))
data1_forward_prey[which(data1_forward_prob >= 0.5)] <- 1
data1_forward_cm <- table(data1_tst_q6$diagnosis, data1_forward_prey)
data1_forward_cm

# Peformance evaluation
perf_table_d1_q6[1,]<-perf_eval(data1_forward_cm)[c(1,3,4,5,6)]
perf_table_d1_q6 


#data2
data2_trn_q5[,c(2:6)]<-data.frame(apply(data2_trn_q5[,c(2:6)], 2, as.numeric))
data2_tst[,c(2:6)]<-data.frame(apply(data2_tst[,c(2:6)], 2, as.numeric))
data2_trn_q6 <- data.frame(scale(data2_trn_q5[,c(1:39)], center = TRUE, scale = TRUE),blueWins = as.numeric(data2_trn_q5$blueWins))
data2_tst_q6 <- data.frame(scale(data2_tst[,c(1:39)], center = TRUE, scale = TRUE),blueWins = as.numeric(data2_tst$blueWins))
data2_trn_q6$blueWins<-ifelse(data2_trn_q6$blueWins==2,1,0)
data2_tst_q6$blueWins<-ifelse(data2_tst_q6$blueWins==2,1,0)

perf_table_d2_q6 <- matrix(0, nrow = 2, ncol = 5)
rownames(perf_table_d2_q6) <- c("Logistic Regression", "Decison Tree")
colnames(perf_table_d2_q6) <- c("TPR", "TNR", "Accuracy", "BCR", "F1-Measure")
perf_table_d2_q6[2,]<-perf_table_d2[3,c(1,3,4,5,6)]
perf_table_d2_q6

#data2 로지스틱 회귀
#상한 & 하한선 설정 for ForwardSelection
data2_tmp_x <- paste(colnames(data2_trn_q6)[-40], collapse=" + ")
data2_tmp_x
data2_tmp_xy <- paste("blueWins ~ ", data2_tmp_x, collapse = "")
as.formula(data2_tmp_xy)

#Forward Selection_part2
data2_logi_forward <- step(glm(blueWins~ 1, data = data2_trn_q6), 
                           scope = list(upper = as.formula(data2_tmp_xy), lower = blueWins~ 1), 
                           direction="forward")
summary(data2_logi_forward)

# Make prediction
data2_forward_prob <- predict(data2_logi_forward, type = "response", newdata = data2_tst_q6)
data2_forward_prey <- rep(0, nrow(data2_tst_q6))
data2_forward_prey[which(data2_forward_prob >= 0.5)] <- 1
data2_forward_cm <- table(data2_tst_q6$blueWins, data2_forward_prey)
data2_forward_cm

# Peformance evaluation
perf_table_d2_q6[1,]<-perf_eval(data2_forward_cm)[c(1,3,4,5,6)]
perf_table_d2_q6 

