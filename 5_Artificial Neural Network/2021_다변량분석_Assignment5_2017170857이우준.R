#install.packages("nnet")
#install.packages("GA")
library(nnet)
library(GA)
library(tree)
library(party)
# Performance evaluation function for multi-class classification 
perf_eval_multi <- function(cm){ 
  ACC = sum(diag(cm))/sum(cm)
  BCR = 1
  for (i in 1:dim(cm)[1]){
    BCR = BCR*(cm[i,i]/sum(cm[i,])) 
  }
  BCR = BCR^(1/dim(cm)[1])
  return(c(ACC, BCR))
}

# Performance Evaluation Function2 
# Confusion Matrix가 3 by 3 형태가 아닐 때 bounds error가 발생하여 이를 방지하기 위한 함수 형성
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


#_________________________Q1_______________________________
# Load data & Preprocessing
origindata_input <- read.csv("train_values.csv")
origindata_target <- read.csv("train_labels.csv")
#결측치 확인 (없음)
table(is.na (origindata_input))
table(is.na(origindata_target))
origindata_instance <- dim(origindata_input)[1]
origindata_var <- dim(origindata_input)[2]
#has_가 붙은 것들은 binary 데이터
#요인형 변수는 9~15번, 27번 데이터
numeric_idx<-c(2:8)
categorical_idx<-c(9:15,27)
binary_idx<-c(16:26,28:39)
origindata_input_num<-origindata_input[,numeric_idx]
origindata_input_cat<-origindata_input[,categorical_idx]
origindata_input_bin<-origindata_input[,binary_idx]

for (i in 1:8){
  barplot(main=colnames(origindata_input_cat[i]),table(origindata_input_cat[,i]),ylab="Count")
}

#_________________________Q2_______________________________
origindata_input_cat_1hot<-matrix(0,nrow=origindata_instance)
for (i in 1:8){
  t<-class.ind(origindata_input_cat[,i])
  colnames(t) <- paste(colnames(origindata_input_cat[i]), colnames(t), sep = "_")
  origindata_input_cat_1hot<-cbind(origindata_input_cat_1hot,t)
}
origindata_target_1hot<-class.ind(origindata_target[,2])
colnames(origindata_target_1hot) <- paste("damagae_grade", colnames(origindata_target_1hot), sep = "_")
#Scale the numeric data and make Final data
origindata_input_num<-scale(origindata_input_num,center=TRUE, scale=TRUE)
finaldata<-data.frame(origindata_input_num, origindata_input_bin, origindata_input_cat_1hot[,-1], origindata_target_1hot)

# Initialize performance matrix
perf_summary_q3 <- matrix(0, nrow = 1, ncol = 2)
colnames(perf_summary_q3) <- c("ACC", "BCR")
rownames(perf_summary_q3) <- c("ANN")

# Split the data into the training/validation sets
set.seed(123456)
trn_idx <- sample(1:origindata_instance, 150000)
earth_trn<-finaldata[trn_idx,]
earth_temp<-finaldata[-trn_idx,]
set.seed(123456)
val_idx <-sample(1:nrow(earth_temp),50000)
earth_val<-earth_temp[val_idx,]
earth_tst<-earth_temp[-val_idx,]

trn_target <- earth_trn[,c(69,70,71)]
trn_input <- earth_trn[,-c(69,70,71)]
val_target <- earth_val[,c(69,70,71)]
val_input <- earth_val[,-c(69,70,71)]
tst_target <- earth_tst[,c(69,70,71)]
tst_input <- earth_tst[,-c(69,70,71)]
#______________________________Q3________________________________
#hidden nodes와 max iteration의 범위 설정
nH <- seq(from=1,to=13,by=2)
max_iter <- seq(from=40, to=100, by=30)
q3_mat <- matrix(0,length(nH)*length(max_iter),4)
colnames(q3_mat) <- c("Hidden Nodes", "Max Iteration", "ACC", "BCR")

start <- proc.time()
n<-1
for (i in 1:length(nH)) {
  cat("Training ANN: the number of hidden nodes:", nH[i], "\n")
  for (j in 1:length(max_iter)) {
    tmp_nnet <- nnet(trn_input,trn_target, size = nH[i], decay = 5e-4, maxit = max_iter[j])
    prey <- predict(tmp_nnet, val_input)
    q3_mat[n,1:2] <- c(nH[i],max_iter[j])
    q3_mat[n,3:4] <- perf_eval_multi2(max.col(val_target), max.col(prey))    
    n <- n+1
  }
}
proc.time() - start


#ACC 기준의 성능 평가
q3_ordered_ACC <- q3_mat[order(q3_mat[,3], decreasing = TRUE),]    
colnames(q3_ordered_ACC) <- c("Hidden Nodes", "maxit", "ACC", "BCR")
q3_ordered_ACC

#BCR 기준의 성능 평가
q3_ordered_BCR <- q3_mat[order(q3_mat[,4], decreasing = TRUE),]    
colnames(q3_ordered_BCR) <- c("Hidden Nodes", "maxit", "ACC", "BCR")
q3_ordered_BCR

#_______________________Q4__________________________________
rang <- seq(0.1, 0.85, 0.15)
q4_mat <- matrix(0,length(rang),3)
colnames(q4_mat) <- c("rang", "ACC", "BCR")

start <- proc.time()
n <- 1
for (i in 1:length(rang)){
  rang_nnet <- nnet(trn_input,trn_target, size =q3_ordered_BCR[1,1] , rang = rang[i], decay = 5e-4, maxit =q3_ordered_BCR[1,2] )
  rang_prey <- predict(rang_nnet, val_input)
  q4_mat[n,1] <- rang[i]
  q4_mat[n,2:3] <- perf_eval_multi2(max.col(val_target), max.col(rang_prey))
  n <- n+1
  
}
proc.time() - start

q4_mat
q4_mat_ACC<-q4_mat[order(q4_mat[,2], decreasing = TRUE),] 
q4_mat_BCR<-q4_mat[order(q4_mat[,3], decreasing = TRUE),] 
q4_mat_ACC[1,]
q4_mat_BCR[1,]

#____________________Q5_________________________
#training set과 validation set 결합
intg_target <- rbind(trn_target,val_target)
intg_input <- rbind(trn_input,val_input)


q5_mat <- matrix(0,10,3)
colnames(q5_mat) <- c("No.", "ACC", "BCR")

start <- proc.time()
n <- 1
for (i in 1:10){
  final_nnet <- nnet(intg_input,intg_target, size = q3_ordered_BCR[1,1] , rang = q4_mat_ACC[1,1], decay = 5e-4, maxit =q3_ordered_BCR[1,2] )
  final_prey <- predict(final_nnet, tst_input)
  q5_mat[n,1] <- i
  q5_mat[n,2:3] <- perf_eval_multi2(max.col(tst_target), max.col(final_prey))
  n <- n+1
}
proc.time() - start
q5_mat
q5_total_mat<-matrix(0,2,3)
colnames(q5_total_mat)<-c("Mean","Varaiance","Standard Deviation")
rownames(q5_total_mat)<-c("ACC","BCR")
q5_total_mat[1,1]<-mean(q5_mat[,2])
q5_total_mat[1,2]<-var(q5_mat[,2])
q5_total_mat[1,3]<-sd(q5_mat[,2])
q5_total_mat[2,1]<-mean(q5_mat[,3])
q5_total_mat[2,2]<-var(q5_mat[,3])
q5_total_mat[2,3]<-sd(q5_mat[,3])
q5_total_mat
#____________________Q6_________________________

fit_func <- function(string){
  sel_var_idx <- which(string == 1)
  # Use variables whose gene value is 1
  sel_x <- x[, sel_var_idx]
  # Training the model
  GA_NN <- nnet(sel_x,y, size = q3_ordered_BCR[1,1] , rang = q4_mat_ACC[1,1], decay = 5e-4, maxit =10 ) #컴퓨팅 능력 제한
  GA_prey <- predict(GA_NN, tst_input[, sel_var_idx])
  return(perf_eval_multi2(max.col(tst_target), max.col(GA_prey))[1])
}

x <- intg_input
y <- intg_target

q6_mat <- matrix(0,3,2)
colnames(q6_mat) <- c("No.", "variable selected num")

for (i in 1:3){
  GA_q3 <- ga(type = "binary", fitness = fit_func, nBits = ncol(x), 
              names = colnames(x), popSize = 25, pcrossover = 0.7, 
              pmutation = 0.01, maxiter = 10, elitism = 2, seed = 123*i)
  
  q6_mat[i,2] <- paste(which(GA_q3@solution[which.min(rowSums(GA_q3@solution)),] == 1), collapse=",")
  q6_mat[i,1]<-i
  q6_mat
}

q6_mat
temp_q6<-matrix(0,1,68)
temp_q6[,c(as.numeric(unlist(strsplit(q6_mat[1,2],split=","))))]<-temp_q6[,c(as.numeric(unlist(strsplit(q6_mat[1,2],split=","))))]+1
temp_q6[,c(as.numeric(unlist(strsplit(q6_mat[2,2],split=","))))]<-temp_q6[,c(as.numeric(unlist(strsplit(q6_mat[2,2],split=","))))]+1
temp_q6[,c(as.numeric(unlist(strsplit(q6_mat[3,2],split=","))))]<-temp_q6[,c(as.numeric(unlist(strsplit(q6_mat[3,2],split=","))))]+1
ga_chosen_idx<-which(temp_q6>=2)
ga_chosen_idx
#______________________Q7___________________________________
intg_input_q7 <- intg_input[,ga_chosen_idx]


q7_mat <- matrix(0,10,3)
colnames(q7_mat) <- c("No.", "ACC", "BCR")

start <- proc.time()
n <- 1
for (i in 1:10){
  final_nnet <- nnet(intg_input_q7,intg_target, size = q3_ordered_BCR[1,1] , rang = q4_mat_ACC[1,1], decay = 5e-4, maxit =q3_ordered_BCR[1,2] )
  final_prey <- predict(final_nnet, tst_input[,ga_chosen_idx])
  q7_mat[n,1] <- i
  q7_mat[n,2:3] <- perf_eval_multi2(max.col(tst_target), max.col(final_prey))
  n <- n+1
}
proc.time() - start
q7_mat

q7_total_mat<-matrix(0,2,3)
colnames(q7_total_mat)<-c("Mean","Varaiance","Standard Deviation")
rownames(q7_total_mat)<-c("ACC","BCR")
q7_total_mat[1,1]<-mean(q7_mat[,2])
q7_total_mat[1,2]<-var(q7_mat[,2])
q7_total_mat[1,3]<-sd(q7_mat[,2])
q7_total_mat[2,1]<-mean(q7_mat[,3])
q7_total_mat[2,2]<-var(q7_mat[,3])
q7_total_mat[2,3]<-sd(q7_mat[,3])
q7_total_mat

#__________________Q8________________________________________
min_criterion = c(0.875,0.9, 0.95, 0.975, 0.99)
min_split = c(3000,5000,10000)
max_depth = c(0,1,3,5)
intg_target_q8<-as.numeric(toupper(substr(names(intg_target), nchar(names(intg_target))-0, nchar(names(intg_target)))[max.col(intg_target)]))
tst_target_q8<-as.numeric(toupper(substr(names(tst_target), nchar(names(tst_target))-0, nchar(names(tst_target)))[max.col(tst_target)]))

q8_data<-cbind(intg_input,damagae_grade=intg_target_q8)
q8_tst_data<-cbind(tst_input,damagae_grade=tst_target_q8)
q8_mat <- matrix(0,length(min_criterion)*length(min_split)*length(max_depth),5)
colnames(q8_mat) <- c("min_criterion","min_split","max_depth","ACC", "BCR")

iter_cnt = 1
l<-(1:3)
for (i in 1:length(min_criterion)){
  for ( j in 1:length(min_split)){
    for ( k in 1:length(max_depth)){
      
      tmp_control = ctree_control(mincriterion = min_criterion[i], minsplit = min_split[j], maxdepth = max_depth[k])
      tmp_tree <- ctree(damagae_grade ~ ., data = q8_data, controls = tmp_control)
      tmp_tree_val_prediction <- predict(tmp_tree, newdata = q8_tst_data)
      tmp_tree_val_response <- treeresponse(tmp_tree, newdata = q8_tst_data)
      round(tmp_tree_val_prediction)
      # Confusion matrix
      cfm <-  table(factor(round(tmp_tree_val_prediction),l), factor(q8_tst_data$damagae_grade,l))
      # parameters
      q8_mat[iter_cnt,1] = min_criterion[i]
      q8_mat[iter_cnt,2] = min_split[j]
      q8_mat[iter_cnt,3] = max_depth[k]
      # Performances from the confusion matrix
      q8_mat[iter_cnt,4:5] = perf_eval_multi(cfm)
      iter_cnt <-iter_cnt+1
    }
  }
}

# Find the best set of parameters
q8_mat <- q8_mat[order(q8_mat[,5], decreasing = T),]
q8_mat
#_______________________Q9____________________________
# Multinominal logistic  regression
mlr_model <- lm(damagae_grade ~ ., data = q8_data)
mlr_prey <- predict(mlr_model, newdata = q8_tst_data)

q9_cfm <-  table(factor(round(mlr_prey),l), factor(q8_tst_data$damagae_grade,l))
perf_eval_multi(q9_cfm)

final_table<-matrix(0,2,4)
colnames(final_table) <- c("ANN","ANN_GA","Decision Tree","Multinomial Logistic Regression")
rownames(final_table) <-c("Accruacy", "BCR")
q7_mat <- q7_mat[order(q7_mat[,3], decreasing = TRUE),]
q5_mat <- q5_mat[order(q5_mat[,3], decreasing = TRUE),]   
final_table[,4]<-perf_eval_multi(q9_cfm)
final_table[,3]<-q8_mat[1,c(4,5)]
final_table[,2]<-q7_mat[1,c(2,3)]
final_table[,1]<-q5_mat[1,c(2,3)]
final_table
#_____________________여기까지 함_____________________________