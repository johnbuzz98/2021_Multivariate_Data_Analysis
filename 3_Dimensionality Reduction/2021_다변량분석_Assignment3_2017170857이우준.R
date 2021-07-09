#install.packages("glmnet")
#install.packages("GA")
#install.packages("ROCR")

library(glmnet)
library(GA)
library(ROCR)

# Performance evaluation function for multiple linear regression
perf_eval_mlr <- function(tgt_y, pre_y){
  
  # RMSE
  rmse <- sqrt(mean((tgt_y - pre_y)^2))
  # MAE
  mae <- mean(abs(tgt_y - pre_y))
  # MAPE
  mape <- 100*mean(abs((tgt_y - pre_y)/tgt_y))
  
  return(c(rmse, mae, mape))
}

# Performance evaluation function for logistic regression
perf_eval_logi <- function(cm){
  
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
  
  return(c(ACC, BCR, F1))
}

# 이상치 기준 분류 만들기 함수

boxplot_outliers <-function(target){
  valuelist <-boxplot(target)$stats
  return(c(valuelist[1,],valuelist[5,]))
}





#_______________________________________________Part1__________________________________________________

perf_mat_part1 <- matrix(0, nrow = 4, ncol = 5)
rownames(perf_mat_part1) <- c("Forward Selection", " Backward Elimination", "Stepwise Selection", "GA")
colnames(perf_mat_part1) <- c("Adjusted Rsquare", "Time", "RMSE", "MAE", "MAPE")

# Load the data & Preprocessing
part1_data <- read.csv("seoulBikeData.csv")
part1_data<-part1_data[!(part1_data$Functioning.Day=="No"),] #자전거가 운행하는 날만 측정
part1_data_target<-part1_data$Rented.Bike.Count
part1_data_input <- part1_data[,-c(1,2,14)] #날짜, 종속변수 제거, Fucntioning.Day 항목 제거

#명목형 변수 변환 
nbike<-nrow(part1_data)
Spring <- rep(0,nbike)
Summer <- rep(0,nbike)
Autumn <- rep(0,nbike)
Winter <- rep(0,nbike)
Spring_idx <- which(part1_data$Seasons == "Spring")
Summer_idx <- which(part1_data$Seasons == "Summer")
Autmumn_idx <- which(part1_data$Seasons == "Autumn")
Winter_idx <- which(part1_data$Seasons == "Winter")
Spring[Spring_idx] <-1
Summer[Summer_idx] <-1
Autumn[Autmumn_idx] <-1
Winter[Winter_idx] <-1
seasons <- data.frame(Winter, Spring, Summer, Autumn)

part1_data_input$Holiday<-ifelse(part1_data_input$Holiday=="No Holiday",0,1)

part1_data_input<-cbind(part1_data_input[,-c(10)], seasons)

part1_data_final<-data.frame(part1_data_input,part1_data_target)


#Training Set ,Validation Set 분리
set.seed(123456)
part1_trn_idx <- sample(1:nbike, round(0.7*nbike))
part1_trn_data<- part1_data_final[part1_trn_idx,]
part1_val_data<- part1_data_final[-part1_trn_idx,]


#______________________________________________Q1________________________________________________

#상한 & 하한선 설정 for Forward, Backward, Stepwise Selection
part1_tmp_x <- paste(colnames(part1_trn_data)[-15], collapse=" + ")
part1_tmp_x
part1_tmp_xy <- paste("part1_data_target ~ ", part1_tmp_x, collapse = "")
as.formula(part1_tmp_xy)



#Forward Selection_Part1
start_time <- proc.time()
mlr_part1_forward <- step(lm(part1_data_target~ 1, data = part1_trn_data),
                    scope = list(upper = as.formula(part1_tmp_xy), lower = part1_data_target ~ 1),
                    direction="forward")
end_time<-proc.time()
perf_mat_part1[1,2] <- (end_time-start_time)[3]
perf_mat_part1[1,1]<-summary(mlr_part1_forward)$adj.r.squared
summary(mlr_part1_forward)

#Predict by Forward Selection
mlr_part1_forward_p <- predict(mlr_part1_forward, newdata = part1_val_data)
perf_mat_part1[1,c(3,4,5)]<-perf_eval_mlr(part1_val_data$part1_data_target, mlr_part1_forward_p)
perf_mat_part1



#Backward Selection_Part1
start_time <- proc.time()
mlr_part1_backward <- step(lm(part1_data_target~ ., data = part1_trn_data),
                  scope = list(upper = as.formula(part1_tmp_xy), lower = part1_data_target ~ 1),
                  direction="backward")
end_time<-proc.time()
perf_mat_part1[2,2] <- (end_time-start_time)[3]
perf_mat_part1[2,1]<-summary(mlr_part1_backward)$adj.r.squared
summary(mlr_part1_backward)

#Predict by Backward Selection
mlr_part1_backward_p <- predict(mlr_part1_backward, newdata = part1_val_data)
perf_mat_part1[2,c(3,4,5)]<-perf_eval_mlr(part1_val_data$part1_data_target, mlr_part1_backward_p)
perf_mat_part1

#Stepwise Selection_Part1
start_time <- proc.time()
mlr_part1_stepwise <- step(lm(part1_data_target~ 1, data = part1_trn_data),
                          scope = list(upper = as.formula(part1_tmp_xy), lower = part1_data_target ~ 1),
                          direction="both")
end_time<-proc.time()
perf_mat_part1[3,2] <- (end_time-start_time)[3]
perf_mat_part1[3,1]<-summary(mlr_part1_stepwise)$adj.r.squared
summary(mlr_part1_stepwise)

#Predict by Stepwise Selection
mlr_part1_stepwise_p <- predict(mlr_part1_stepwise, newdata = part1_val_data)
perf_mat_part1[3,c(3,4,5)]<-perf_eval_mlr(part1_val_data$part1_data_target, mlr_part1_stepwise_p)
perf_mat_part1


#__________________________________Q2_____________________________________

# Fitness function: Adjusted R square
fit_ars <- function(string){
  sel_var_idx <- which(string == 1)
  # Use variables whose gene value is 1
  sel_x <- x[, sel_var_idx]
  xy <- data.frame(sel_x, y)
  # Training the model
  GA_lr <- lm(y ~ ., data = xy)
  return(summary(GA_lr)$adj.r.squared)
}

x <- as.matrix(part1_trn_data[,-15])
y <- part1_trn_data[,15]

# Variable selection by Genetic Algorithm
start_time <- proc.time()
GA_ars <- ga(type = "binary", fitness = fit_ars, nBits = ncol(x), 
            names = colnames(x), popSize = 50, pcrossover = 0.5, 
            pmutation = 0.01, maxiter = 100, elitism = 2, seed = 123)
end_time <- proc.time()
end_time - start_time
perf_mat_part1[4,2] <- (end_time-start_time)[3]

best_var_idx <- which(GA_ars@solution[which.min(rowSums(GA_ars@solution)),] == 1) 

# Model training based on the best variable subset
GA_trn_data_p1 <- part1_trn_data[,c(best_var_idx, 15)]
GA_tst_data_p1 <- part1_val_data[,c(best_var_idx, 15)]

GA_model_p1 <- lm(part1_data_target ~ ., GA_trn_data_p1)
summary(GA_model_p1)
perf_mat_part1[4,1]<-summary(GA_model_p1)$adj.r.squared


# Make prediction
GA_model_prob_p1 <- predict(GA_model_p1, newdata = GA_tst_data_p1)
perf_mat_part1[4,c(3,4,5)]<-perf_eval_mlr(GA_tst_data_p1$part1_data_target, GA_model_prob_p1)
perf_mat_part1




#__________________________________Q3_____________________________________

popsizelist<-list(25,50,75)
crossoverlist<-list(0.3, 0.5, 0.7)
mutationlist<-list(0.001,0.01,0.1)
q3perf_mat<- matrix(0, nrow = 27, ncol = 5)
colnames(q3perf_mat) <- c("Adjusted Rsquare", "Time", "RMSE", "MAE", "MAPE")
rowname_q3perf<-list()
repeat_time_q3<-1
repeat_time_q3
for (i in popsizelist){
  for (j in crossoverlist){
    for (k in mutationlist){
      rowname_q3perf<-append(rowname_q3perf,(paste(as.character(i),as.character(j),as.character(k),sep=", ")))
      
      start_time <- proc.time()
      GA_ars_q3 <- ga(type = "binary", fitness = fit_ars, nBits = ncol(x), 
                      names = colnames(x), popSize = i, pcrossover = j, 
                      pmutation = k, maxiter = 100, elitism = 2, seed = 123)
      end_time <- proc.time()
      q3perf_mat[repeat_time_q3,2]<-(end_time-start_time)[3]
      
      best_var_idx <- which(GA_ars@solution[which.min(rowSums(GA_ars@solution)),] == 1) 
      
      # Model training based on the best variable subset
      GA_trn_data_q3 <- part1_trn_data[,c(best_var_idx, 15)]
      GA_tst_data_q3 <- part1_val_data[,c(best_var_idx, 15)]
      
      GA_model_q3 <- lm(part1_data_target ~ ., GA_trn_data_q3)
      summary(GA_model_q3)
      q3perf_mat[repeat_time_q3,1]<-summary(GA_model_q3)$adj.r.squared
      
      
      # Make prediction
      GA_model_prob_q3 <- predict(GA_model_q3, newdata = GA_tst_data_q3)
      q3perf_mat[repeat_time_q3,c(3,4,5)]<-perf_eval_mlr(GA_tst_data_q3$part1_data_target, GA_model_prob_q3)
      
      
      repeat_time_q3<-repeat_time_q3+1
      
      
      
    }
  }
}
rownames(q3perf_mat)<-rowname_q3perf
q3perf_mat



#_______________________________________PART 2 _________________________________

part2_data  <- read.csv("data.csv")
part2_data_input <- part2_data[,-c(1,2,33)] #1은 ID, 2는 목적변수, 33은 잉여 변수 (all NA값)
part2_data_input_scaled <- scale(part2_data_input, center = TRUE, scale = TRUE)
part2_data_target <- part2_data$diagnosis
part2_data_target <- ifelse(part2_data_target=="B",0,1)
part2_data_scaled <- data.frame(part2_data_input_scaled, part2_data_target)

#이상치 제거

outliers <- matrix(0, nrow = 30, ncol = 2)
rownames(outliers) <- names(part2_data_input)
colnames(outliers) <- c("LCL", "UCL")

for(i in 1:30){
  outliers[i,]<-boxplot_outliers(part2_data_input_scaled[,i])
}

for (i in 1:30){
  part2_data_scaled[,i]<-ifelse(part2_data_scaled[,i] < outliers[i,1] | part2_data_scaled[,i] > outliers[i,2], NA, part2_data_scaled[,i])
}

sum(is.na(part2_data_scaled))
part2_data_scaled<-na.omit(part2_data_scaled)

ndata2<-nrow(part2_data_scaled)

#Training Set ,Validation Set 분리
set.seed(123456)
part2_trn_idx <- sample(1:ndata2, round(0.7*ndata2))
part2_trn_data<- part2_data_scaled[part2_trn_idx,]
part2_val_data<- part2_data_scaled[-part2_trn_idx,]

#______________________________Q4_______________________________

perf_mat_part2 <- matrix(0, nrow = 4, ncol = 6)
rownames(perf_mat_part2) <- c("Forward", "Backward","Stepwise","GA")
colnames(perf_mat_part2) <- c("T_AUROC", "Time", "V_AUROC", "Accuracy", "BCR", "F1-Measure")


#상한 & 하한선 설정 for Forward, Backward, Stepwise Selection
part2_tmp_x <- paste(colnames(part2_trn_data)[-31], collapse=" + ")
part2_tmp_x
part2_tmp_xy <- paste("part2_data_target ~ ", part2_tmp_x, collapse = "")
as.formula(part2_tmp_xy)


#Forward Selection_part2
start_time <- proc.time()
part2_forward <- step(glm(part2_data_target~ 1, data = part2_trn_data), 
                      scope = list(upper = as.formula(part2_tmp_xy), lower = part2_data_target ~ 1), 
                      direction="forward")
end_time <- proc.time()
perf_mat_part2[1,2]<-(end_time-start_time)[3]
summary(part2_forward)

#Training AUROC
prob <- predict(part2_forward, type = "response")
pred <- prediction(prob,part2_trn_data$part2_data_target)
perf_mat_part2[1,1]<-as.numeric(performance(pred, measure = "auc")@y.values)


# Make prediction
part2_forward_prob <- predict(part2_forward, type = "response", newdata = part2_val_data)
part2_forward_prey <- rep(0, nrow(part2_val_data))
part2_forward_prey[which(part2_forward_prob >= 0.5)] <- 1
part2_forward_cm <- table(part2_val_data$part2_data_target, part2_forward_prey)
part2_forward_cm

# Peformance evaluation
perf_mat_part2[1,c(4,5,6)]<- perf_eval_logi(part2_forward_cm)
perf_mat_part2


#Validation Auroc
pred <- prediction(part2_forward_prob,part2_val_data$part2_data_target)
perf_mat_part2[1,3]<-as.numeric(performance(pred, measure = "auc")@y.values)
perf_mat_part2



#Backward Selection_part2
start_time <- proc.time()
part2_backward <- step(glm(part2_data_target~ .,family=binomial, data = part2_trn_data),
                      scope = list(upper = as.formula(part2_tmp_xy), lower = part2_data_target ~ 1), 
                      direction="backward",trace=1)
summary(part2_backward)
end_time <- proc.time()
perf_mat_part2[2,2]<-(end_time-start_time)[3]
summary(part2_backward)


#Training AUROC
prob <- predict(part2_backward, type = "response")
pred <- prediction(prob,part2_trn_data$part2_data_target)
perf_mat_part2[2,1]<-as.numeric(performance(pred, measure = "auc")@y.values)


# Make prediction
part2_backward_prob <- predict(part2_backward, type = "response", newdata = part2_val_data)
part2_backward_prey <- rep(0, nrow(part2_val_data))
part2_backward_prey[which(part2_backward_prob >= 0.5)] <- 1
part2_backward_cm <- table(part2_val_data$part2_data_target, part2_backward_prey)


# Peformance evaluation
perf_mat_part2[2,c(4,5,6)]<- perf_eval_logi(part2_backward_cm)


#Validation Auroc
pred <- prediction(part2_backward_prob,part2_val_data$part2_data_target)
perf_mat_part2[2,3]<-as.numeric(performance(pred, measure = "auc")@y.values)


#Stepwise Selection_part2
start_time <- proc.time()
part2_stepwise <- step(glm(part2_data_target~ 1, data = part2_trn_data), 
                      scope = list(upper = as.formula(part2_tmp_xy), lower = part2_data_target ~ 1), 
                      direction="both")
end_time <- proc.time()
perf_mat_part2[3,2]<-(end_time-start_time)[3]
summary(part2_stepwise)

#Training AUROC
prob <- predict(part2_stepwise, type = "response")
pred <- prediction(prob,part2_trn_data$part2_data_target)
perf_mat_part2[3,1]<-as.numeric(performance(pred, measure = "auc")@y.values)


# Make prediction
part2_stepwise_prob <- predict(part2_stepwise, type = "response", newdata = part2_val_data)
part2_stepwise_prey <- rep(0, nrow(part2_val_data))
part2_stepwise_prey[which(part2_stepwise_prob >= 0.5)] <- 1
part2_stepwise_cm <- table(part2_val_data$part2_data_target, part2_forward_prey)
part2_stepwise_cm

# Peformance evaluation
perf_mat_part2[3,c(4,5,6)]<- perf_eval_logi(part2_stepwise_cm)


#Validation Auroc
pred <- prediction(part2_stepwise_prob,part2_val_data$part2_data_target)
perf_mat_part2[3,3]<-as.numeric(performance(pred, measure = "auc")@y.values)


#_________________________Q5________________________________
fit_auroc <- function(string){
  sel_var_idx <- which(string == 1)
  # Use variables whose gene value is 1
  sel_x <- x[, sel_var_idx]
  xy <- data.frame(sel_x, y)
  # Training the model
  GA_lr <- glm(y ~ ., family = binomial, data = xy)
  GA_lr_prob <- predict(GA_lr, type = "response", newdata = xy)
  pred <- prediction(GA_lr_prob,y)
  return(as.numeric(performance(pred, measure = "auc")@y.values))
}

x <- as.matrix(part2_trn_data[,-31])
y <- part2_trn_data[,31]

# Variable selection by Genetic Algorithm
start_time <- proc.time()
GA_auroc <- ga(type = "binary", fitness = fit_auroc, nBits = ncol(x), 
            names = colnames(x), popSize = 50, pcrossover = 0.5, 
            pmutation = 0.01, maxiter = 100, elitism = 2, seed = 123)
end_time <- proc.time()
perf_mat_part2[4,2] <- (end_time-start_time)[3]
GA_auroc@solution


best_var_idx <- which(GA_auroc@solution[which.min(rowSums(GA_auroc@solution)),] == 1)

# Model training based on the best variable subset
GA_trn_data_q5 <- part2_trn_data[,c(best_var_idx, 31)]
GA_tst_data_q5 <- part2_val_data[,c(best_var_idx, 31)]

GA_model_q5 <- glm(part2_data_target ~ ., family=binomial, GA_trn_data_q5)
summary(GA_model_q5)

#Training AUROC
prob_q5 <- predict(GA_model_q5, type = "response")
pred_q5 <- prediction(prob_q5,GA_trn_data_q5$part2_data_target)
perf_mat_part2[4,1]<-as.numeric(performance(pred_q5, measure = "auc")@y.values)




# Make prediction
GA_model_prob_q5 <- predict(GA_model_q5, type = "response", newdata = GA_tst_data_q5)
GA_model_prey_q5 <- rep(0, nrow(GA_tst_data_q5))
GA_model_prey_q5[which(GA_model_prob_q5 >= 0.5)] <- 1
GA_model_cm_q5 <- table(GA_tst_data_q5$part2_data_target, GA_model_prey_q5)
GA_model_cm_q5

# Peformance evaluation
perf_mat_part2[4,c(4,5,6)]<- perf_eval_logi(GA_model_cm_q5)

#Validation Auroc
pred <- prediction(GA_model_prob_q5,part2_val_data$part2_data_target)
perf_mat_part2[4,3]<-as.numeric(performance(pred, measure = "auc")@y.values)

#___________________________Q6_________________________________
popsizelist<-list(25,50,75)
crossoverlist<-list(0.3, 0.5, 0.7)
mutationlist<-list(0.001,0.01,0.1)
q6perf_mat<- matrix(0, nrow = 27, ncol = 6)
colnames(q6perf_mat) <- c("AUROC", "Time", "Length", "Accuracy", "BCR", "F1-Measure")
rowname_q6perf<-list()
repeat_time_q6<-1
for (i in popsizelist){
  for (j in crossoverlist){
    for (k in mutationlist){
      
      rowname_q6perf<-append(rowname_q6perf,(paste(as.character(i),as.character(j),as.character(k),sep=", ")))
      
      start_time <- proc.time()
      GA_auroc_q6 <- ga(type = "binary", fitness = fit_auroc, nBits = ncol(x), 
                      names = colnames(x), popSize = i, pcrossover = j, 
                      pmutation = k, maxiter = 100, elitism = 2, seed = 123)
      end_time <- proc.time()
      q6perf_mat[repeat_time_q6,2]<-(end_time-start_time)[3]
      
      best_var_idx_q6 <- which(GA_auroc_q6@solution[which.min(rowSums(GA_auroc_q6@solution)),] == 1)
      q6perf_mat[repeat_time_q6,3]<-min(rowSums(GA_auroc_q6@solution))

      
      
      # Model training based on the best variable subset
      GA_trn_data_q6 <- part2_trn_data[,c(best_var_idx_q6, 31)]
      GA_tst_data_q6 <- part2_val_data[,c(best_var_idx_q6, 31)]
      
      GA_model_q6 <- glm(part2_data_target ~ ., family=binomial, GA_trn_data_q6)
      
      #Training AUROC
      prob_q6 <- predict(GA_model_q6, type = "response")
      pred_q6 <- prediction(prob_q6,GA_trn_data_q6$part2_data_target)
      q6perf_mat[repeat_time_q6,1]<-as.numeric(performance(pred_q6, measure = "auc")@y.values)
      

      # Make prediction
      GA_model_prob_q6 <- predict(GA_model_q6, type = "response", newdata = GA_tst_data_q6)
      GA_model_prey_q6 <- rep(0, nrow(GA_tst_data_q6))
      GA_model_prey_q6[which(GA_model_prob_q6 >= 0.5)] <- 1
      GA_model_cm_q6 <- table(GA_tst_data_q6$part2_data_target, GA_model_prey_q6)

      # Peformance evaluation
      q6perf_mat[repeat_time_q6,c(4,5,6)]<-perf_eval_logi(GA_model_cm_q6)
      
      repeat_time_q6<-repeat_time_q6+1
      
      
    }
  }
}
rownames(q6perf_mat)<-rowname_q6perf
q6perf_mat






#___________________________________________________________________________________
