library(arules)
library(arulesViz)
library(wordcloud)
library(dplyr)
library(Rgraphviz)

#______________________________Step 1__________________________________

#Q1
#Step 1
mooc <- read.csv("big_student_clear_third_version.csv") 
head(mooc)
str(mooc)

#ItemName var save
Institute = mooc["institute"]
Course = mooc["course_id"]
Region = mooc["final_cc_cname_DI"]
Degree = mooc["LoE_DI"]

#Step2
table(Region)
Region = gsub('\\s', '', Region$final_cc_cname_DI)
table(Region)

#Step3
RawTransactions = paste(Institute$institute, Course$course_id, Region, Degree$LoE_DI, sep = "_")
head(RawTransactions)

#Step4
MOOC_transactions = paste(mooc$userid_DI, RawTransactions, sep = " ")
head(MOOC_transactions)

#step5
write.table(MOOC_transactions, "MOOC_User_Course.csv", row.names = FALSE, col.names = FALSE, sep = ',', quote = FALSE)


#________________STEP 2_________________________

#Q2-1
single = read.transactions("MOOC_User_Course.csv", format = "single", cols = c(1, 2), rm.duplicates = TRUE)
inspect(head(single))
summary(single)

#Q2-2
itemName <- itemLabels(single)
itemCount <- itemFrequency(single)*nrow(single)
col = brewer.pal(8, "Set3")
wordcloud(words = itemName, freq = itemCount, min.freq = 150, scale = c(1.8, 0.1), col = col , random.order = FALSE, random.color = TRUE)

#Q2-3
itemFrequencyPlot(single, support = 0.01, cex.names = 1, , main = "Above 1% ")
itemFrequencyPlot(single, support = 0.01, cex.names = 1, , topN = 5, main = "Top 5")



#________________________________Step3_______________________________________
#Q3-1
rule1 = apriori(single, parameter = list(support = 0.001, confidence = 0.001))
rule2 = apriori(single, parameter = list(support = 0.001, confidence = 0.005))
rule3 = apriori(single, parameter = list(support = 0.001, confidence = 0.01))
rule4 = apriori(single, parameter = list(support = 0.005, confidence = 0.001))
rule5 = apriori(single, parameter = list(support = 0.005, confidence = 0.005))
rule6 = apriori(single, parameter = list(support = 0.005, confidence = 0.01))
rule7 = apriori(single, parameter = list(support = 0.01, confidence = 0.001))
rule8 = apriori(single, parameter = list(support = 0.01, confidence = 0.005))
rule9 = apriori(single, parameter = list(support = 0.01, confidence = 0.01))
rule10 = apriori(single, parameter = list(support = 0.02, confidence = 0.001))
rule11 = apriori(single, parameter = list(support = 0.02, confidence = 0.005))
rule12 = apriori(single, parameter = list(support = 0.02, confidence = 0.01))


#Q3-2
rule = apriori(single, parameter=list(support=0.001, confidence=0.05))
summary(rule)

inspect(rule)
inspect(sort(rule, by = "support"))
inspect(sort(rule, by = "confidence"))
inspect(sort(rule, by = "lift"))

mat = as.data.frame(inspect(rule))
measure = mat$support * mat$confidence
measure = measure * mat$lift
mat_measure = data.frame(mat, measure)
head(mat_measure[rev(order(mat_measure$measure)),])


#조건절과 결과절 위치 변경
mat_measure[(mat_measure$lhs == "{HarvardX_CB22x_UnitedStates_Master's}")&(mat_measure$rhs == "{HarvardX_ER22x_UnitedStates_Master's}"),]
mat_measure[(mat_measure$lhs == "{HarvardX_ER22x_UnitedStates_Master's}")&(mat_measure$rhs == "{HarvardX_CB22x_UnitedStates_Master's}"),]

mat_measure[(mat_measure$lhs == "{MITx_8.02x_India_Secondary}")&(mat_measure$rhs == "{MITx_6.00x_India_Secondary}"),]
mat_measure[(mat_measure$lhs == "{MITx_6.00x_India_Secondary}")&(mat_measure$rhs == "{MITx_8.02x_India_Secondary}"),]

mat_measure[(mat_measure$lhs == "{HarvardX_CS50x_India_Secondary}")&(mat_measure$rhs == "{MITx_6.00x_India_Secondary}"),]
mat_measure[(mat_measure$lhs == "{MITx_6.00x_India_Secondary}")&(mat_measure$rhs == "{HarvardX_CS50x_India_Secondary}"),]



#________________________________________________EXTRA QUESTION_____________________________________________
#matrix
plot(rule, method = "matrix", measure = "lift")

#paracoord
plot(rule, method="paracoord")

#grouped
subrule = head(rule, n = 7, by = "lift");
plot(subrule,method="grouped")  
