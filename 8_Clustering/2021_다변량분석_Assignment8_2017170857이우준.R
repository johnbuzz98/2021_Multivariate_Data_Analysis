library(clValid)
library(plotrix)
library(factoextra)
library(dbscan)
library(ISLR)
library(proxy)
library(gtools)

# Load dataset
seed <- read.csv("Seed_Data.csv")

#class값 제거
seed_class<-seed[,8]
seed_x<-seed[,-8]

#scaling
seed_x_scaled <- scale(seed_x, center = TRUE, scale = TRUE)
head(seed_x_scaled)

#____________________Q2________________________________
system.time({seed_clValid_k = clValid(seed_x_scaled, 2:5, clMethods = "kmeans", validation = c("internal", "stability"))})
summary(seed_clValid_k)

#____________________Q3________________________________
#K=3

centroid_3 = list()
size_3 = list()
for(i in 1:10){
  seed_kmc = kmeans(seed_x_scaled,3)
  tmp_centroid = as.data.frame(seed_kmc$centers)
  centroid_3[[i]] = tmp_centroid
  
  tmp_size = as.data.frame(seed_kmc$size)
  size_3[[i]] = tmp_size
}

centroid_3
size_3

write.table(as.data.frame(centroid_3),file="centroid_3.csv", quote=F,sep=",",row.names=F)
write.table(as.data.frame(size_3),file="size_3.csv", quote=F,sep=",",row.names=F)


#____________________Q4________________________________
data_k3 = kmeans(seed_x_scaled ,3)
data_centers = data_k3$centers
rownames(data_centers) = paste("cluster", c(1:3))
data_centers
kmc_centroid=as.data.frame(data_k3$centers)
par(mfrow = c(1,3))
for (i in 1:3){
  plot_title = paste("Radar Chart for Cluster", i, sep=" ")
  radial.plot(data_centers[i,], labels = colnames(data_centers), radial.lim=c(-2,2), rp.type = "p", main = plot_title, line.col = "#8EBB33", lwd = 3, show.grid.labels = 1)
}
dev.off()

#____________________Q5________________________________
#Cluster 1 vs. Cluster 3
kmc_cluster2 = seed_x[data_k3$cluster == 1,]
kmc_cluster3 = seed_x[data_k3$cluster == 3,]

result1 = data.frame()

for (i in 1:ncol(seed_x)){
  result1[i,1] = t.test(kmc_cluster2[, i], kmc_cluster3[, i], alternative = "two.sided")$p.value
  result1[i,2] = t.test(kmc_cluster2[, i], kmc_cluster3[, i], alternative = "greater")$p.value
  result1[i,3] = t.test(kmc_cluster2[, i], kmc_cluster3[, i], alternative = "less")$p.value
}

colnames(result1) = c("two.sided", "greater", "less")
rownames(result1) = c(colnames(seed_x))
result1

#Cluster 2 vs. Cluster 3
kmc_cluster2 = seed_x[data_k3$cluster == 2,]
kmc_cluster3 = seed_x[data_k3$cluster == 3,]

result2 = data.frame()

for (i in 1:ncol(seed_x)){
  result2[i,1] = t.test(kmc_cluster2[, i], kmc_cluster3[, i], alternative = "two.sided")$p.value
  result2[i,2] = t.test(kmc_cluster2[, i], kmc_cluster3[, i], alternative = "greater")$p.value
  result2[i,3] = t.test(kmc_cluster2[, i], kmc_cluster3[, i], alternative = "less")$p.value
}

colnames(result2) = c("two.sided", "greater", "less")
rownames(result2) = c(colnames(data))
result2

#____________________Q6________________________________
#distance matrix
cor_Mat = cor(t(seed_x_scaled), method = "spearman")
dist_seed = as.dist(1-cor_Mat)

#method = "single"
hr_single = hclust(dist_seed, method = "single", members=NULL)
plot(hr_single, hang = -1)
dev.off()

#method = "complete"
hr_complete = hclust(dist_seed, method = "complete", members=NULL)
plot(hr_complete, hang = -1)
dev.off()

#____________________Q7________________________________

#method = "single"
hr_single = hclust(dist_seed, method = "single", members=NULL)
plot(hr_single, hang = -1)
rect.hclust(hr_single, k=3, border="red")
dev.off()

#method = "complete"
hr_complete = hclust(dist_seed, method = "complete", members=NULL)
plot(hr_complete, hang = -1)
rect.hclust(hr_complete, k=3, border="red")
dev.off()


#method = "single"
mycl_single = cutree(hr_single, k = 3)
seed_hc_single = data.frame(seed_x_scaled, clusterID = as.factor(mycl_single))
head(seed_hc_single)
hc_summary_single = data.frame()

for (i in 1:ncol(seed_x_scaled)){
  hc_summary_single = rbind(hc_summary_single, tapply(seed_hc_single[,i], seed_hc_single$clusterID, mean))
}

colnames(hc_summary_single) = paste("cluster", c(1:3))
rownames(hc_summary_single) = c(colnames(seed_x))
hc_summary_single

par(mfrow = c(1,3))
#Rador Chart
for (i in 1:ncol(hc_summary_single)){
  plot_title = paste("Radar Chart for Cluster", i, sep=" ")
  radial.plot(hc_summary_single[,i], labels = rownames(hc_summary_single), 
              radial.lim=c(-2,2), rp.type = "p", main = plot_title, 
              line.col = "#8EBB33", lwd = 3, show.grid.labels=1)
}
dev.off()

#method = "complete"
mycl_complete = cutree(hr_complete, k = 3)
seed_hc_complete = data.frame(seed_x_scaled, clusterID = as.factor(mycl_complete))
head(seed_hc_complete)
hc_summary_complete = data.frame()

for (i in 1:ncol(seed_x_scaled)){
  hc_summary_complete = rbind(hc_summary_complete, tapply(seed_hc_complete[,i], seed_hc_complete$clusterID, mean))
}

colnames(hc_summary_complete) = paste("cluster", c(1:3))
rownames(hc_summary_complete) = c(colnames(seed_x))
hc_summary_complete

#Rador Chart
par(mfrow = c(1,3))
for (i in 1:ncol(hc_summary_complete)){
  plot_title = paste("Radar Chart for Cluster", i, sep=" ")
  radial.plot(hc_summary_complete[,i], labels = rownames(hc_summary_complete), 
              radial.lim=c(-2,2), rp.type = "p", main = plot_title, 
              line.col = "#8EBB33", lwd = 3, show.grid.labels=1)
}
dev.off()
#____________________Q8________________________________
hc_summary_complete=t(hc_summary_complete)
hc_summary_single=t(hc_summary_single)
rownames(kmc_centroid)<-c("cluster 1", "cluster 2", "cluster 3")

kmc_centroid
hc_summary_complete
hc_summary_single


#K-Mean Clustering와 Hierarchical Clustering의 cluster 별 코사인 유사도 matrix 구하는 함수
custom_simliarity<-function(a,b){
  resultmat<-matrix(NA,nrow(a),nrow(b))
  for (i in 1:nrow(a)){
    for (j in 1:nrow(b)){
      tmpmat<-rbind(a[i,],b[j,])
      resultmat[i,j]<-as.matrix(dist(tmpmat, method = "cosine"))[1,2]
    }
  }
  return(resultmat)
}

#위에서 구한 3X3 matrix에서 각 Clustering 방법론 별 cluster들끼리 최적의 조합 찾아주기 (유사도의 합이 최대가 되는 방향)
# ex) K-means의 cluster (1,2,3)은 hierachical의 cluster(2,3,1)과 매치되는구나~
permute<-permutations(3,3,1:3)
bestclustermatch<-function(c){
  maxtmp=0
  pertmp=NA
  for (i in 1:6){
    sumtmp<-c[permute[i,1],1]+c[permute[i,2],2]+c[permute[i,3],3]
    if (sumtmp>=maxtmp){
      maxtmp<-sumtmp
      pertmp<-permute[i,]
    }
  }
  pertmp[4]=maxtmp
  return(pertmp)
}


kmc_X_complete<-custom_simliarity(kmc_centroid,hc_summary_complete)
kmc_X_single<-custom_simliarity(kmc_centroid,hc_summary_single)
kmc_X_complete
kmc_X_single

bestclustermatch(kmc_X_complete)
bestclustermatch(kmc_X_single)

#____________________Q9&10________________________________
ep <- seq(0,3,0.1)
mP <- seq(10,30,1)
DBSCAN_result_matrix<-data.frame(NA,(length(ep)*length(mP)),4)
DBSCAN_result_matrix
n<-1
for (i in 1:length(ep)) {
  for (j in 1:length(mP)){
    DBSCAN_seed = dbscan(seed_x_scaled, eps = ep[i], minPts = mP[j])
    num_cluster<-length(unique(DBSCAN_seed$cluster))
    num_noise<-length(which(DBSCAN_seed$cluster==0))
    if (num_noise!=0){
      num_cluster<-num_cluster-1
    }
    DBSCAN_result_matrix[n,c(1:4)]<-c(DBSCAN_seed$eps, DBSCAN_seed$minPts, num_cluster, num_noise)
    DBSCAN_result_matrix[c(1:5),]  
    n<-n+1
  }
}
colnames(DBSCAN_result_matrix)<-c("eps","minPoints","Number_of_Cluster", "Number_of_Noise")
DBSCAN_result_matrix
DBSCAN_result_matrix_only3<-DBSCAN_result_matrix[DBSCAN_result_matrix[,3]==3,]
DBSCAN_result_matrix_only3<-DBSCAN_result_matrix_only3[c(order(DBSCAN_result_matrix_only3[,4])),]
DBSCAN_result_matrix_only3

#____________________Q11________________________________
#K-MEANS
data_k3$cluster
#Hierarchical Clustering 
mycl_complete
#DBSCAN
DBSCAN_seed = dbscan(seed_x_scaled, eps = 1.1, minPts = 16)
DBSCAN_seed$cluster
#Original Class
seed_class
seed_class[1:70]
seed_class[71:140]
seed_class[141:210]

#entropy 함수 정의
entropy <- function(target) {
  freq <- table(target)/length(target)
  # vectorize
  vec <- as.data.frame(freq)[,2]
  #drop 0 to avoid NaN resulting from log2
  vec<-vec[vec>0]
  #compute entropy
  -sum(vec * log2(vec))
}

#Entropy
KMC_entropy<-entropy(data_k3$cluster[1:70])+entropy(data_k3$cluster[71:140])+entropy(data_k3$cluster[141:210])
HC_entropy<-entropy(mycl_complete[1:70])+entropy(mycl_complete[71:140])+entropy(mycl_complete[141:210])
DB_entropy<-entropy(DBSCAN_seed$cluster[1:70])+entropy(DBSCAN_seed$cluster[71:140])+entropy(DBSCAN_seed$cluster[141:210])
KMC_entropy
HC_entropy
DB_entropy

