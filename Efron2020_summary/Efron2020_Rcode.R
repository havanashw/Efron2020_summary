# Author: Shi Hongwei
# Date: Jan 18th, 2021
# Introduction: 
# Recode all the examples in the Efron article Prediction, Estimation, and Attribution
rm(list = ls())
setwd("/Users/shihongwei/Desktop/文献阅读汇总/") # set your work dictionary


###########################################################
############Section 2 Surface Plus Noise Models############
###########################################################
#----------------------------------------------------------------------------------------#
# Example 1: cubic regression model
require(bootstrap)
require(ggplot2)
require(effects)

# Dataset: cholostyramine data
data(cholost) # require(bootstrap)
cholost$z <- as.numeric(scale(cholost$z))

# Cubic regression model
# fit <- lm(y ~ z + I(z^2) + I(z^3), data = cholost) # method1
fit <- lm(y~poly(z, 3), data=cholost) # method2
sigmahat <- summary(fit)$sigma
df <- summary(fit)$df[2]

# Fit regression function
coeff <- coefficients(fit)
eq <- paste0("y = ", round(coeff[4],5), "*x^3 + ", round(coeff[3],5), "*x^2 + ",
             round(coeff[2],5), "*x + ", round(coeff[1],5))
# The prediction results
pre_cholost <- data.frame(new_z=cholost$z, new_y=fit$fitted.values)

# Construct confidence intervals
# Method 1
new_z <- seq(min(cholost$z), max(cholost$z), length.out=100)
z_sample <- new_z[seq(0, length(new_z), 8)]
se_yhat <- sigmahat * sqrt(1/nrow(cholost) + (z_sample-mean(cholost$z))^2/sum((cholost$z-mean(cholost$z))^2))
y_low <- predict(fit, data.frame(z=z_sample)) - pt(0.05/2, df)^(-1)*se_yhat
y_up <- predict(fit, data.frame(z=z_sample)) + pt(0.05/2, df)^(-1)*se_yhat

#----------------------------------------------------#
# Method 2: require(effects)
# eff<- allEffects(fit, xlevels=list(z=z_sample))
# ci_df <- as.data.frame(eff[[1]])
# y_low <- ci_df$lower
# y_up <- ci_df$upper
#----------------------------------------------------#

# Figure 1
fig1 <- ggplot() +
  geom_point(data=cholost, aes(x=z, y=y), shape=18, size=0.6, color="green") +
  geom_line(data=pre_cholost, aes(x=new_z, y=new_y),  color='black', size=1) +
  geom_segment(aes(x=z_sample, y=y_low, 
                   xend=z_sample, yend=y_up), color='red', size=0.8) +
  labs(x="normalized compliance", y="cholesterol decrease", 
       title=paste0("OLS cubic regression ", "(", eq, ")")) +
  theme_bw()
# Save the image
image <- paste0("fig1", ".pdf")
pdf(image, height=5)
fig1
dev.off()

fig1_ <- ggplot(data=cholost, aes(z, y)) +
  geom_point(size=0.1, color="green") +
  stat_smooth(method=lm, formula=y~poly(x, 3)) +
  geom_line(data=pre_cholost, aes(x=new_z, y=new_y),  color='black', size=1) +
  labs(x="normalized compliance", y="cholesterol decrease", 
       title=paste0("OLS cubic regression ", "(", eq, ")")) +
  theme_bw()
# Save the image
image <- paste0("fig1_", ".pdf")
pdf(image, height=5)
fig1_
dev.off()


#----------------------------------------------------------------------------------------#
# Example 2: plot Newton's second law
require(barsurf)
require(plotrix)
require(scatterplot3d)
require(plot3D)
require(MBA)

# Generate data
m <- seq(1, 5, length = 30)
f <- seq(0, 10, length = 30)
func <- function(m, f) {
  a <- f/m
  return(a)
}
a <- outer(m, f, func)

# Figure 2
persp(f, m, a,
      theta = 30, phi = 15,
      expand = 1, col = "green",
      r = 180,
      ltheta = 120,
      shade = 0.5,
      xlab = "force", ylab = "mass", zlab = "acceleration",
      main = "Newton's 2nd law: acceleration=force/mass")
# Save the image
image <- paste0("fig21", ".pdf")
pdf(image)
persp(f, m, a,
      theta = 30, phi = 15,
      expand = 1, col = "green",
      r = 180,
      ltheta = 120,
      shade = 0.5,
      xlab = "force", ylab = "mass", zlab = "acceleration",
      main = "Newton's 2nd law: acceleration=force/mass")
dev.off()

a_noise <- a + matrix(rnorm(30*30, 0, 0.5), 30, 30) # add noise
persp(f, m, a_noise,
      theta = 30, phi = 15,
      expand = 1, col = "green",
      r = 180,
      ltheta = 120,
      shade = 0.5,
      xlab = "force", ylab = "mass", zlab = "acceleration",
      main = "If Newton had done the experiment")
# Save the image
image <- paste0("fig22", ".pdf")
pdf(image)
persp(f, m, a_noise,
      theta = 30, phi = 15,
      expand = 1, col = "green",
      r = 180,
      ltheta = 120,
      shade = 0.5,
      xlab = "force", ylab = "mass", zlab = "acceleration",
      main = "If Newton had done the experiment")
dev.off()


################################################################
############Section 3 The Pure Prediction Algorithms############
################################################################
#----------------------------------------------------------------------------------------#
# Example 3: Classification tree
require(tidyverse)
require(caret)
require(rpart)
require(rpart.plot)

# Dataset: PimaIndiansDiabetes2 (substitute for neonate data which can't be found)
data("PimaIndiansDiabetes2", package = "mlbench")
df_diabetes <- na.omit(PimaIndiansDiabetes2)

# Split the data into train and test set
set.seed(123)
train_samples <- df_diabetes$diabetes %>% 
  createDataPartition(p = 0.8, list = FALSE)
train.data  <- df_diabetes[train_samples, ]
test.data <- df_diabetes[-train_samples, ]

# Build the model
model <- rpart(diabetes ~., data=train.data, method="class")

# Figure 3
rpart.plot(model, extra = 104) # require(rpart.plot)
# Save the image
image <- paste0("fig3", ".pdf")
pdf(image)
rpart.plot(model, extra = 104) # require(rpart.plot)
dev.off()

# Pridiction error rate
df_predict <- predict(model, test.data[,-ncol(test.data)], type = 'class')
table_mat <- table(test.data$diabetes, df_predict)
accuracy <- sum(diag(table_mat)) / sum(table_mat)
error <- 1 - accuracy
print(paste0("Pridiction error rate is ", round(error*100, 2), "%"))

# Cross-validation method
# Randomly select a Z fold subscript set, n: sample size, seed: random seed
CV <- function(n, split=10, seed=111){
  K <- rep(1:split, ceiling(n/split))[1:n]
  set.seed(seed)
  K <- sample(K, n)
  mm <- list() # mm[[i]] is ith subscript set
  for (i in 1:split) mm[[i]] <- (1:n)[K == i]
  return(mm)
}
mm <- CV(nrow(df_diabetes), split=10)
Er <- c()
for(i in 1:10){
  m <- mm[[i]]
  model_temp <- rpart(diabetes ~., data=df_diabetes[-m,], method="class")
  df_predict <- predict(model_temp, df_diabetes[m,-ncol(df_diabetes)], type='class')
  table_mat <- table(df_diabetes[m,]$diabetes, df_predict)
  accuracy <- sum(diag(table_mat)) / sum(table_mat)
  Er[i] <- 1 - accuracy
}
print(paste0("The 10-fold cross-validation prediction error is ", round(mean(Er)*100, 2), "%"))


#----------------------------------------------------------------------------------------#
# Example 4: randomForest
require(randomForest)
require(e1071)
require(ggplot2)
require(ggrepel)

# Construct randomForest model
rf_model <- randomForest(diabetes ~., data=train.data, ntree=500)
rf_Er <- rf_model$err.rate[,1]
train_model <- train(
  diabetes ~., data=train.data, method="rf",
  trControl=trainControl("cv", number=10),
  ntree=500,
  importance=TRUE
)
rf_final_model <- train_model$finalModel
CV_error <- mean(rf_final_model$err.rate[,1])
plot_error <- data.frame(ntree=1:length(rf_Er), rf_Er)

# Figure4
fig4 <- ggplot() +
  geom_line(data=plot_error, aes(x=ntree, y=rf_Er)) +
  geom_hline(yintercept=CV_error, linetype="dashed", color="red") +
  geom_text_repel(aes(ntree, CV_error, label=paste0(round(CV_error*100, 2), "%")), 
                  data=plot_error[plot_error$ntree==500,], color="red") +
  labs(x="# trees", y="prediction error") +
  theme_bw()
# Save the image
image <- paste0("fig4", ".pdf")
pdf(image, height=4.5)
fig4
dev.off()


#################################################################
############Section 4 A Microarray Prediction Probelm############
#################################################################
#----------------------------------------------------------------------------------------#
# Example 5: randomForest and gbm prostate cancer microarray study
require(tidyverse)
require(caret)
require(rpart)
require(randomForest)
require(e1071)
require(ggplot2)
require(ggrepel)
require(doParallel)
require(foreach)
require(sda)
require(gbm)

# Dataset: prostate cancer microarray
data(singh2002) # p = 6033
# require("SIS")
# data(prostate.train) # p = 12600, raw data
prostate <- data.frame(Y=singh2002$y, singh2002$x)

# RandomForest
# Split the data into train and test set
set.seed(111)
train_index <- prostate$Y %>% 
  createDataPartition(p = 0.5, list = FALSE)
train_pro  <- prostate[train_index, ]
test_pro <- prostate[-train_index, ]

# Prediction rule is obtained by randomForeat applied to the train set 
# and the prediction result is obtained by randomForeat applied to the test set 

#---------------------------------------------------------------#
# Method 1
# rf_Er <- c()
# for (ntree in 1:100) {
#   rf_model <- randomForest(Y~.,data=train_pro, ntree=ntree)
#   rf_predict <- rf_model %>% predict(test_pro[,-1])
#   rf_Er[ntree] <- 1 - mean(rf_predict == test_pro$Y)
#   print(ntree)
# }
#---------------------------------------------------------------#

#---------------------------------------------------------------#
# Method 2
# rf_Er <- sapply(1:100, function(ntree) {
#   rf_model <- randomForest(Y~.,data=train_pro, ntree=ntree)
#   rf_predict <- rf_model %>% predict(test_pro[,-1])
#   rf_Er <- 1 - mean(rf_predict == test_pro$Y)
#   return(rf_Er)
#   })
#---------------------------------------------------------------#

# Method 3
cl <- makeCluster((detectCores()-1), type="FORK")
registerDoParallel(cl)
rf_Er <- foreach(ntree=1:300, .combine="c", .packages=c("randomForest", "dbplyr")) %dopar% {
  rf_model <- randomForest(Y~., data=train_pro, ntree=ntree)
  rf_predict <- rf_model %>% predict(test_pro[,-1])
  rf_Er <- 1 - mean(rf_predict == test_pro$Y)
  return(rf_Er)
}
stopCluster(cl)
plot_error <- data.frame(ntree=1:length(rf_Er), rf_Er)

# Read the result
plot_error <- read.csv("plot_error.csv")

# Figure 5
fig5 <- ggplot() +
  geom_point(data=plot_error, aes(x=ntree, y=rf_Er), shape=0, size=0.8, color="red") +
  geom_line(data=plot_error, aes(x=ntree, y=rf_Er), color="red") +
  geom_text_repel(aes(305, rf_Er, label=paste0(round(rf_Er*100, 2), "%")),
                  data=plot_error[plot_error$ntree==300,], color="red") +
  labs(x="number of trees", y="error rate") +
  geom_hline(yintercept=0, linetype="dashed", color = "black") +
  theme_bw()
# Save the image
image <- paste0("fig5", ".pdf")
pdf(image, height=4.5)
fig5
dev.off()

# 100 random train/test splits of prostate data
data_split <- function(p=0.5) {
  set.seed(proc.time()[1]*1000) # make each split different
  train_index <- prostate$Y %>% 
    createDataPartition(p=p, list=FALSE)
  train_pro  <- prostate[train_index, ]
  test_pro <- prostate[-train_index, ]
  return(list(train_pro=train_pro, test_pro=test_pro))
}

Er_freq <- function() {
  output_split <- data_split()
  train_pro <- output_split$train_pro
  test_pro <- output_split$test_pro
  rf_model <- randomForest(Y~., data=train_pro)
  rf_predict <- rf_model %>% predict(test_pro[,-1])
  Er_frequency <- nrow(test_pro) - sum(rf_predict == test_pro$Y)
  return(Er_frequency)
}

cl <- makeCluster((detectCores()-1), type="FORK")
registerDoParallel(cl)
Er_set <- foreach(i=1:100, .combine="c", .packages=c("randomForest", "caret", "dplyr")) %dopar% {
  Er_set <- Er_freq()
  return(Er_set)
}
stopCluster(cl)

# Read the result
Er_set <- read.csv("Er_set.csv")

# Table 2
# write.csv(table(Er_set$Er_set), "table2.csv", row.names=F) 

# Gbm
# Split the data into train and test set
set.seed(111)
train_index <- prostate$Y %>% 
  createDataPartition(p = 0.5, list = FALSE)
train_pro  <- prostate[train_index, ]
test_pro <- prostate[-train_index, ]
train_pro$Y <- as.numeric(train_pro$Y) - 1
test_pro$Y <- as.numeric(test_pro$Y) - 1

# The pridiction error in test set and train set
cl <- makeCluster((detectCores()-1), type="FORK")
registerDoParallel(cl)
Er <- foreach(ntree=1:400, .combine="rbind", .packages=c("gbm", "dplyr")) %dopar% {
  gbm_model <- gbm(Y~., data=train_pro, n.trees=ntree)
  gbm_prediction <- gbm_model %>% predict(test_pro[,-1])
  gbm_pred_test <- ifelse(gbm_prediction > 0.5, 1, 0)
  gbm_pred_train <- ifelse(gbm_model$fit > 0.5, 1, 0)
  test_Er <- 1 - mean(test_pro$Y == gbm_pred_test)
  train_Er <- 1 - mean(train_pro$Y == gbm_pred_train)
  return(c(test_Er, train_Er))
}
stopCluster(cl)
test_Er <- Er[,1]
train_Er <- Er[,2]
gbm_Er <- data.frame(ntree=1:400, test_Er=test_Er, train_Er=train_Er)

# Read the result
gbm_Er <- read.csv("gbm_Er.csv")

# Figure 6
fig6 <- ggplot() +
  geom_line(data=gbm_Er, aes(x=ntree, y=test_Er, color="red")) +
  geom_line(data=gbm_Er, aes(x=ntree, y=train_Er, color="green")) +
  geom_text_repel(aes(410, test_Er, label=paste0(round(test_Er*100, 2), "%")),
                  data=gbm_Er[gbm_Er$ntree==400,], color="red") +
  labs(x="# trees", y="error rate") +
  geom_hline(yintercept=0, linetype="dashed", color = "black") +
  scale_color_manual(name = "",
                     values = c('red' = 'red','green' = 'green'), 
                     breaks = c('red', 'green'),
                     labels = c("Test Set", "Train Set")) +
  theme_bw()
# Save the image
image <- paste0("fig6", ".pdf")
pdf(image, height=6, width=9)
fig6
dev.off()


############################################################################
############Section 5 Advantages and Disadvantages of Prediction############
############################################################################
#----------------------------------------------------------------------------------------#
# Example 6: randomForest importance measures for prostate cancer

# Split the data into train and test set
set.seed(111)
train_index <- prostate$Y %>% 
  createDataPartition(p = 0.5, list = FALSE)
train_pro  <- prostate[train_index, ]
test_pro <- prostate[-train_index, ]
rf_model <- randomForest(Y~., data=train_pro, ntree=500)
imp <- importance(rf_model)
imp_df <- data.frame(gene=1:length(imp), importance=imp)
imp_order <- imp_df[order(imp_df$MeanDecreaseGini, decreasing=T),]
imp_order$order <- 1:nrow(imp)

# Figure 7
fig7 <- ggplot() +
  geom_point(data=imp_order[imp_order$MeanDecreaseGini>0,], aes(x=order, y=MeanDecreaseGini), shape=7, size=0.1) +
  labs(x="order number", y="importance") +
  geom_hline(yintercept=0, linetype="dashed", color = "black") +
  geom_vline(xintercept=1, linetype="dashed", color = "black") +
  geom_text_repel(aes(order, MeanDecreaseGini, label=paste("gene", gene, sep=" ")), 
                  data=imp_order[imp_order$order==1,], hjust=1, color="red") +
  theme_bw()
# Save the image
image <- paste0("fig7", ".pdf")
pdf(image, height=4.5)
fig7
dev.off()

# Can we use the inmportance scores for attribution
remove <- function(top_set) {
  if (top_set == 0) {
    gene_set <- imp_order[,1] + 1
  } else {
    gene_set <- imp_order[-top_set,1] + 1
  }
  return(gene_set)
}

cl <- makeCluster((detectCores()-1), type="FORK")
registerDoParallel(cl)
top_Er <- foreach(top=c(0, 1, 5, 10, 20, 40, 80, 160, 300, 500, 1000, 3000, 5000), .combine="c") %dopar% {
  temp_train_pro <- train_pro[,c(1, remove(1:top))]
  temp_test_pro <- test_pro[,c(1, remove(1:top))]
  rf_model <- randomForest(Y~., data=temp_train_pro)
  rf_predict <- rf_model %>% predict(temp_test_pro[,-1])
  top_Er <- length(rf_predict) - sum(rf_predict==temp_test_pro$Y)
  return(top_Er)
}
stopCluster(cl)
top_df <- data.frame(droptop=c(0, 1, 5, 10, 20, 40, 80, 160, 300, 500, 1000, 3000, 5000),
                     errors=top_Er)
# Table 3
# write.csv(top_df, "top_df.csv", row.names=F)

# Read the result
top_df <- read.csv("top_df.csv")


################################################################
############Section 6 The Training/Test Set Paradigm############
################################################################
#----------------------------------------------------------------------------------------#
# Example 7: splitting randomization is violated

# Split the data into train and test set determined by early/late
train_set <- prostate[c(1:25, 51:76),]
test_set <- prostate[-c(1:25, 51:76),]

# RandomForest test set error
cl <- makeCluster((detectCores()-1), type="FORK")
registerDoParallel(cl)
rf_Er <- foreach(ntree=1:300, .combine="c", .packages=c("randomForest")) %dopar% {
  rf_model <- randomForest(Y~., data=train_set, ntree=ntree)
  rf_predict <- rf_model %>% predict(test_set[,-1])
  rf_Er <- 1 - mean(rf_predict == test_set$Y)
  return(rf_Er)
}
stopCluster(cl)
plot_error1 <- data.frame(ntree=1:length(rf_Er), rf_Er)

# Read the result
plot_error1 <- read.csv("plot_error1.csv")

# Figure 8
fig8 <- ggplot() +
  geom_point(data=plot_error1, aes(x=ntree, y=rf_Er), shape=0, size=0.8, color="red") +
  geom_line(data=plot_error1, aes(x=ntree, y=rf_Er), color="red") +
  geom_text_repel(aes(320, rf_Er, label=paste0(round(rf_Er*100, 2), "%")),
                  data=plot_error1[plot_error1$ntree==300,], color="black") +
  geom_text_repel(aes(300, rf_Er, label=paste0("previous ", round(rf_Er*100, 2), "%")), 
                  data=plot_error[plot_error$ntree==300,], hjust=1.5, color="black") +
  labs(x="number of trees", y="error rate") +
  geom_hline(yintercept=0, linetype="dashed", color = "black") +
  theme_bw()
# Save the image
image <- paste0("fig8", ".pdf")
pdf(image, height=6.5, width=9)
fig8
dev.off()

# Gbm test set error
train_set$Y <- as.numeric(train_set$Y) - 1
test_set$Y <- as.numeric(test_set$Y) - 1
cl <- makeCluster((detectCores()-1), type="FORK")
registerDoParallel(cl)
Er <- foreach(ntree=1:400, .combine="rbind", .packages=c("gbm", "dplyr")) %dopar% {
  gbm_model <- gbm(Y~., data=train_set, n.trees=ntree)
  gbm_prediction <- gbm_model %>% predict(test_set[,-1])
  gbm_pred_test <- ifelse(gbm_prediction > 0.5, 1, 0)
  gbm_pred_train <- ifelse(gbm_model$fit > 0.5, 1, 0)
  test_Er <- 1 - mean(test_set$Y == gbm_pred_test)
  train_Er <- 1 - mean(train_set$Y == gbm_pred_train)
  return(c(test_Er, train_Er))
}
stopCluster(cl)
test_Er <- Er[,1]
train_Er <- Er[,2]
gbm_Er1 <- data.frame(ntree=1:400, test_Er=test_Er, train_Er=train_Er)

# Read the result
gbm_Er1 <- read.csv("gbm_Er1.csv")

# Figure 9
fig9 <- ggplot() +
  geom_line(data=gbm_Er1, aes(x=ntree, y=test_Er, color="red")) +
  geom_line(data=gbm_Er1, aes(x=ntree, y=train_Er, color="green")) +
  geom_text_repel(aes(420, test_Er, label=paste0(round(test_Er*100, 2), "%")),
                  data=gbm_Er1[gbm_Er1$ntree==400,], color="black") +
  geom_text_repel(aes(400, test_Er, label=paste0("previous ", round(test_Er*100, 2), "%")), 
                  data=gbm_Er[gbm_Er$ntree==400,], hjust=1.5, color="black") +
  labs(x="# trees", y="error rate") +
  geom_hline(yintercept=0, linetype="dashed", color = "black") +
  scale_color_manual(name = "",
                     values = c('red' = 'red','green' = 'green'), 
                     breaks = c('red', 'green'),
                     labels = c("Test Set", "Train Set")) +
  theme_bw()
# Save the image
image <- paste0("fig9", ".pdf")
pdf(image, height=6.5, width=9)
fig9
dev.off()


#----------------------------------------------------------------------------------------#
# Example 8: hypothetical micoarray study to prove the importance of randomization

# Generate hypothetical micoarray data
ndays <- 400
pgenes <- 200
Y <- rep(0, ndays)
Y <- sapply(1:ndays, function(i) {ifelse(i%%2==0, "Control", "Treatment")})

# Each genes has expected number of episodes equal 1
each_gene <- c()
for (i in 1:pgenes) {
  if (sum(each_gene) < 200) {
    each_gene[i] <- sample(c(0:3), 1)
  } else {
    each_gene[i] <- 0
  }
}
each_gene <- sample(each_gene, length(each_gene))

# The starting date for each episode is random
start_day <- lapply(each_gene, function(i) {
  if(i == 0) {
    start_day <- NULL
  } else {
    start_day <- sample(1:ndays, i)
  }
})

# A gene has an active episode of 30 days
X_active <- matrix(0, ndays, pgenes)
for (j in 1:pgenes) {
  if (is.null(start_day[[j]]) == F) {
    sindex <- start_day[[j]]
    eindex <- start_day[[j]] + 30
    adj_eindex <- sapply(eindex, function(x) {
      ifelse(x>ndays, ndays, x)
    })
    for (i in each_gene[j]) {
      X_active[sindex[i]:adj_eindex[i],j] <- j # 把开始到结束的活跃日期定义为基因编号
    }
  }
}
X_vec <- X_active
dim(X_vec) <- c(ndays*pgenes,1)
X_df <- data.frame(X_vec)
X_df$days <- 1:ndays

# Figure 10
fig10 <- ggplot(data=X_df[X_df$X_vec!=0,], aes(x=days, y=X_vec)) + 
  geom_point(size=0.2, shape=20) +
  coord_cartesian(xlim=c(0, ndays), ylim=c(0, pgenes)) +
  labs(x="days (subjects)", y="genes") +
  geom_vline(xintercept=320, linetype="dashed", color = "red") +
  geom_text_repel(aes(330, 10, label=320), data=data.frame(x=320, y=10), color="red", size=3) +
  theme_bw()
# Save the image
image <- paste0("fig10", ".pdf")
pdf(image, height=5)
fig10
dev.off()

# Construct data matrix X
X <- matrix(0, ndays, pgenes)
for (i in 1:ndays) {
  for (j in 1:pgenes) {
    if(X_active[i,j] == 0) {
      X[i,j] <- rnorm(1, 0, 1)
    } else {
      X[i,j] <- rnorm(1, sample(c(1,-1), 1)*2, 1)
    }
  }
}
contr_data <- data.frame(Y=Y, X)
contr_data$Y <- as.factor(contr_data$Y)
contr_data <- contr_data[order(contr_data$Y),]

# RandomForest pridiction on random splitting
set.seed(111)
train_index <- contr_data$Y %>% 
  createDataPartition(p = 0.8, list = FALSE)
train_pro  <- contr_data[train_index, ]
test_pro <- contr_data[-train_index, ]

cl <- makeCluster((detectCores()-1), type="FORK")
registerDoParallel(cl)
Er <- foreach(ntree=1:200, .combine="rbind", .packages=c("randomForest", "dplyr")) %dopar% {
  rf_model <- randomForest(Y~., data=train_pro, ntree=ntree)
  rf_pred_test <- rf_model %>% predict(test_pro[,-1])
  rf_pred_train <- rf_model %>% predict(train_pro[,-1])
  test_Er <- 1 - mean(test_pro$Y == rf_pred_test)
  train_Er <- 1 - mean(train_pro$Y == rf_pred_train) # 这里的错误率到底应该用哪个?
  return(c(test_Er, train_Er))
}
stopCluster(cl)
test_Er <- Er[,1]
train_Er <- Er[,2]
rf_Er <- data.frame(ntree=1:200, test_Er=test_Er, train_Er=train_Er)

# Read the result
rf_Er <- read.csv("rf_Er.csv")

# Figure 11
fig111 <- ggplot() +
  geom_line(data=rf_Er, aes(x=ntree, y=test_Er, color="black"), size=1) +
  geom_line(data=rf_Er, aes(x=ntree, y=train_Er, color="green"), linetype="dashed") +
  labs(x="# trees", y="error rate", title="Random Test") +
  geom_hline(yintercept=0, linetype="dashed", color = "black") +
  scale_color_manual(name = "",
                     values = c('black' = 'black','green' = 'green'),
                     breaks = c('black', 'green'),
                     labels = c("Test Set", "Train Set")) +
  theme_bw()
# Save the image
image <- paste0("fig111", ".pdf")
pdf(image, height=6)
fig111
dev.off()

# RandomForest pridiction on early/late splitting
train_pro  <- contr_data[1:320, ]
test_pro <- contr_data[-(1:320), ]
cl <- makeCluster((detectCores()-1), type="FORK")
registerDoParallel(cl)
Er <- foreach(ntree=1:200, .combine="rbind", .packages=c("randomForest", "dplyr")) %dopar% {
  rf_model <- randomForest(Y~., data=train_pro, ntree=ntree)
  rf_pred_test <- rf_model %>% predict(test_pro[,-1])
  rf_pred_train <- rf_model %>% predict(train_pro[,-1])
  test_Er <- 1 - mean(test_pro$Y == rf_pred_test)
  train_Er <- 1 - mean(train_pro$Y == rf_pred_train)
  return(c(test_Er, train_Er))
}
stopCluster(cl)
test_Er <- Er[,1]
train_Er <- Er[,2]
rf_Er1 <- data.frame(ntree=1:200, test_Er=test_Er, train_Er=train_Er)

# Read the result
rf_Er1 <- read.csv("rf_Er1.csv")

# Figure 11
fig112 <- ggplot() +
  geom_line(data=rf_Er1, aes(x=ntree, y=test_Er, color="black"), size=1) +
  geom_line(data=rf_Er1, aes(x=ntree, y=train_Er, color="green"), linetype="dashed") +
  labs(x="# trees", y="error rate", title="Late Test") +
  geom_hline(yintercept=0, linetype="dashed", color = "black") +
  scale_color_manual(name = "",
                     values = c('black' = 'black','green' = 'green'), 
                     breaks = c('black', 'green'),
                     labels = c("Test Set", "Train Set")) +
  theme_bw()
# Save the image
image <- paste0("fig112", ".pdf")
pdf(image, height=6)
fig112
dev.off()


####################################################################
########################Section 7 Smoothness########################
####################################################################
#----------------------------------------------------------------------------------------#
# Example 9: randomForest and gbm fits to the cholostyramine data
require(bootstrap)

# Dataset: cholostyramine data
data(cholost)
cholost$z <- as.numeric(scale(cholost$z)) 
fit <- lm(y ~ z + I(z^2) + I(z^3), data = cholost)
fit_sum <- summary(fit)
sigmahat <- fit_sum$sigma
df <- fit_sum$df[2]

# Cubic OLS fit
coeff <- coefficients(fit)
eq <- paste0("y = ", round(coeff[4],5), "*x^3 + ", round(coeff[3],5), "*x^2 + ",
             round(coeff[2],5), "*x + ", round(coeff[1],5))
pre_cholost <- data.frame(new_z=cholost$z, new_y=fit$fitted.values)
poly_cholost <- data.frame(y=cholost$y, poly(cholost$z, 8))

# RandomForest prediction
rf_model <- randomForest(y~., data=poly_cholost)
rf_predict <- rf_model$predicted

# Figure 12
fig121 <- ggplot() +
  geom_point(data=data.frame(x=cholost$z, y=rf_predict), aes(x, y), shape=7, size=0.5) +
  geom_line(data=data.frame(x=cholost$z, y=rf_predict), aes(x, y), linetype="dashed", size=0.3) +
  geom_line(data=pre_cholost, aes(new_z, new_y), color="red", size=1) +
  labs(x="normalized compliance", y="cholesterol decrease", title="randomForest") +
  theme_bw()
# Save the image
image <- paste0("fig121", ".pdf")
pdf(image, height=7)
fig121
dev.off()

# Gbm prediction
gbm_model <- gbm(y~., data=poly_cholost)
gbm_predict <- gbm_model$fit

# 8th degree OLS fit
p8_model <- lm(y~poly(z, 8), data=cholost) 
p8_predict <- p8_model$fitted.values

# Figure 12
fig122 <- ggplot() +
  geom_point(data=data.frame(x=cholost$z, y=gbm_predict), aes(x, y), shape=7, size=0.5) +
  geom_line(data=data.frame(x=cholost$z, y=gbm_predict), aes(x, y), linetype="dashed", size=0.3) +
  geom_line(data=pre_cholost, aes(new_z, new_y), color="red", size=1) +
  geom_line(data=data.frame(x=cholost$z, y=p8_predict), aes(x, y), linetype="dashed",color="green", size=1) +
  coord_cartesian(ylim=c(0, 80)) +
  labs(x="normalized compliance", y="cholesterol decrease", title="Boosting algorithm gbm") +
  theme_bw()
# Save the image
image <- paste0("fig122", ".pdf")
pdf(image, height=7)
fig122
dev.off()

fig122_ <- ggplot(data=cholost, aes(z, y)) +
  geom_point(size=0.1, color="green") +
  stat_smooth(method=lm, formula=y~poly(x, 8), color="green", linetype="dashed") +
  geom_point(data=data.frame(x=cholost$z, y=gbm_predict), aes(x, y), shape=7, size=0.5) +
  geom_line(data=data.frame(x=cholost$z, y=gbm_predict), aes(x, y), linetype="dashed", size=0.3) +
  geom_line(data=pre_cholost, aes(new_z, new_y), color="red", size=1) +
  labs(x="normalized compliance", y="cholesterol decrease", title="Boosting algorithm gbm") +
  theme_bw()
# Save the image
image <- paste0("fig122_", ".pdf")
pdf(image, height=7)
fig122_
dev.off()


#----------------------------------------------------------------------------------------#
# Example 10: randomForest and gbm fits to the supernova data

# supernova data (no public data set found, self-made data)
set.seed(357)
X_sup <- matrix(rnorm(75*25, -1, 2), 75, 25)
mag <- rnorm(75, -1, 2)
mag <- sample(mag, length(mag))
indexi <- sample(75, 30)
indexj <- sample(25, 10)
for (i in indexi) {
  for (j in indexj) {
    X_sup[i, j] <- runif(1, 0, 1)*sample(c(1,-1), 1)
  }
}
supernova <- data.frame(mag=mag, X_sup)
rf_model <- randomForest(mag~., data=supernova)
gbm_model <- gbm(mag~., data=supernova)
alpha <- seq(0, 1, length.out=300)

# 1-3
testdata1_3 <- matrix(0, length(alpha), 25)
for (i in 1:length(alpha)) {
  testdata1_3[i,] <- as.numeric(alpha[i]*supernova[1,-1] + (1-alpha[i])*supernova[3,-1])
}
testdata1_3 <- data.frame(testdata1_3)
rf_pre1_3 <- rf_model %>% predict(testdata1_3)
gbm_pre1_3 <- gbm_model %>% predict(testdata1_3)

# Figure 13
fig131 <- ggplot() +
  geom_line(data=data.frame(x=alpha, y=rf_pre1_3), aes(x, y)) +
  labs(x="from point1 to point3", y="prediction", title="randomForest") +
  theme_bw()
# Save the image
image <- paste0("fig131", ".pdf")
pdf(image, height=3, width=4)
fig131
dev.off()

fig132 <- ggplot() +
  geom_line(data=data.frame(x=alpha, y=gbm_pre1_3), aes(x, y)) +
  labs(x="from point1 to point3", y="prediction", title="gbm") +
  theme_bw()
# Save the image
image <- paste0("fig132", ".pdf")
pdf(image, height=3, width=4)
fig132
dev.off()

# 1-39
testdata1_39 <- matrix(0, length(alpha), 25)
for (i in 1:length(alpha)) {
  testdata1_39[i,] <- as.numeric(alpha[i]*supernova[1,-1] + (1-alpha[i])*supernova[39,-1])
}
testdata1_39 <- data.frame(testdata1_39)
rf_pre1_39 <- rf_model %>% predict(testdata1_39)
gbm_pre1_39 <- gbm_model %>% predict(testdata1_39)

# Figure 13
fig133 <- ggplot() +
  geom_line(data=data.frame(x=alpha, y=rf_pre1_39), aes(x, y)) +
  labs(x="from point1 to point39", y="prediction", title="randomForest") +
  theme_bw()
# Save the image
image <- paste0("fig133", ".pdf")
pdf(image, height=3, width=4)
fig133
dev.off()

fig134 <- ggplot() +
  geom_line(data=data.frame(x=alpha, y=gbm_pre1_39), aes(x, y)) +
  labs(x="from point1 to point39", y="prediction", title="gbm") +
  theme_bw()
# Save the image
image <- paste0("fig134", ".pdf")
pdf(image, height=3, width=4)
fig134
dev.off()

# 39-65
testdata39_65 <- matrix(0, length(alpha), 25)
for (i in 1:length(alpha)) {
  testdata39_65[i,] <- as.numeric(alpha[i]*supernova[39,-1] + (1-alpha[i])*supernova[65,-1])
}
testdata39_65 <- data.frame(testdata39_65)
rf_pre39_65 <- rf_model %>% predict(testdata39_65)
gbm_pre39_65 <- gbm_model %>% predict(testdata39_65)

# Figure 13
fig135 <- ggplot() +
  geom_line(data=data.frame(x=alpha, y=rf_pre39_65), aes(x, y)) +
  labs(x="from point39 to point65", y="prediction", title="randomForest") +
  theme_bw()
# Save the image
image <- paste0("fig135", ".pdf")
pdf(image, height=3, width=4)
fig135
dev.off()

fig136 <- ggplot() +
  geom_line(data=data.frame(x=alpha, y=gbm_pre39_65), aes(x, y)) +
  labs(x="from point39 to point65", y="prediction", title="gbm") +
  theme_bw()
# Save the image
image <- paste0("fig136", ".pdf")
pdf(image, height=3, width=4)
fig136
dev.off()

#######################################################################
###########Section 9 Traditonal Methods in the Wide Data Era###########
#######################################################################

###################################################################################################
# Example 11: GWAS, Manhattan plot
#-------------------------------------------------------------------------------------------------#
# Manhattan plots are widely used in genome-wide association studies (GWAS). 
# The idea is to represent many non-significant data points with variable low values 
# and a few clusters of significant data points that will appear as towers in the plot. 
# In its most frequent use, they are used plot p-values, but they are transformed 
# using the -log10(pval) so smaller pvalues have a higher transformed value.

# To show how to use the function, we’ll need some data. We’ll simulate it using regioneR’s 
# randomization functions. As input, kpPlotManhattan needs a GRanges with the SNP positions 
# and the p-values of each SNP (either as a column of the GRanges or as an independent numeric vector). 
# We’ll create a small function to create these random datasets that will return the SNPs 
# and their p-values and the regions of the significant peaks.

# ref: https://bernatgel.github.io/karyoploter_tutorial//Tutorial/PlotManhattan/PlotManhattan.html
#-------------------------------------------------------------------------------------------------#
require(BiocManager)
# BiocManager::install("regioneR")
# BiocManager::install("BSgenome.Hsapiens.UCSC.hg19")
require(BSgenome.Hsapiens.UCSC.hg19)
require(regioneR)
set.seed(123)

createDataset <- function(num.snps=50000, max.peaks=5) {
  hg19.genome <- filterChromosomes(getGenome("hg19"))
  snps <- sort(createRandomRegions(nregions=num.snps, length.mean=1, 
                                   length.sd=0, genome=filterChromosomes(getGenome("hg19"))))
  names(snps) <- paste0("rs", seq_len(num.snps))
  snps$pval <- rnorm(n=num.snps, mean=0.5, sd=1)
  snps$pval[snps$pval<0] <- -1*snps$pval[snps$pval<0]
  # define the "significant peaks"
  peaks <- createRandomRegions(runif(1, 1, max.peaks), 8e6, 4e6)
  peaks
  for(npeak in seq_along(peaks)) {
    snps.in.peak <- which(overlapsAny(snps, peaks[npeak]))
    snps$pval[snps.in.peak] <- runif(n = length(snps.in.peak), 
                                     min=0.1, max=runif(1, 6, 8))
  }
  snps$pval <- 10^(-1*snps$pval)
  return(list(peaks=peaks, snps=snps))
}
ds <- createDataset()

# BiocManager::install("karyoploteR")
require(karyoploteR)
kp <- plotKaryotype(plot.type=4)
kp <- kpPlotManhattan(kp, data=ds$snps, ymax=10)
kpAxis(kp, ymin=0, ymax=10, numticks = 11) # add axis

# Add label
snps <- kp$latest.plot$computed.values$data
suggestive.thr <- kp$latest.plot$computed.values$suggestiveline
# Get the names of the top SNP per chr
top.snps <- tapply(seq_along(snps), seqnames(snps), function(x) {
  in.chr <- snps[x]
  top.snp <- in.chr[which.max(in.chr$y)]
  return(names(top.snp))
})
# Filter by suggestive line
top.snps <- top.snps[snps[top.snps]$y>suggestive.thr]
# And select all snp information based on the names
top.snps <- snps[top.snps]
kpText(kp, data=top.snps, labels=names(top.snps), ymax=10, pos=4, cex=1., col="red")
kpPoints(kp, data=top.snps, pch=1, cex=1., col="red", lwd=2, ymax=10)

# Save the image
image <- paste0("Manhattan_plot", ".pdf")
pdf(image, width=25)
kp <- plotKaryotype(plot.type=4)
kp <- kpPlotManhattan(kp, data=ds$snps, ymax=10)
kpAxis(kp, ymin=0, ymax=10, numticks = 11) # add axis
kpText(kp, data=top.snps, labels=names(top.snps), ymax=10, pos=4, cex=1., col="red")
kpPoints(kp, data=top.snps, pch=1, cex=1., col="red", lwd=2, ymax=10)
dev.off()
###################################################################################################

#----------------------------------------------------------------------------------------#
# Example 12: estimate lacal fdr and posterior effect size 
# from empirical Bayes analysis of prostate data
require(sda)
require(ggplot2)
require(locfdr)
require(REBayes)
# ref: https://stackoverflow.com/questions/52390616/install-rmosek-under-3-4-4-using-macos
# mosek_attachbuilder("/Users/shihongwei/Documents/mosek/9.2/tools/platform/osx64x86/bin")
# install.rmosek()
require(Rmosek)
require(glmnet)

# Dataset: prostate data
data(singh2002)
Y <- as.numeric(singh2002$y)-1
X <- singh2002$x
prostate <- data.frame(Y, X)
z_vec <- c()
for (i in 1:ncol(X)) {
  fit <- lm(Y~X[,i])
  z_vec[i] <- summary(fit)$coefficients[2,3]
}

# Estimated local false discovery fdr(z)
w <- locfdr(z_vec)
result <- w$mat
z <- result[,1]
fdr <- result[,2]
position <- (1:length(z_vec))[which(z_vec<=-3.868710 | z_vec>=3.846295)] # z[which(fdr<=0.2)]
z_0.2 <- z_vec[position]

# Estimate the expected effect size E(delta|z)
Emp_bayes <- GLmix(z_vec)
E_delta <- predict(Emp_bayes,z)/4

# Select significant genes from lasso
fit_lasso <- cv.glmnet(X, Y, family="gaussian")
coef_lasso <- as.matrix(coef(fit_lasso))
coef_sort <- data.frame(index=1:length(z_vec), coef=coef_lasso[-1,1])
nonzero <- coef_sort[coef_sort$coef!= 0,]
nonzero$coef <- abs(nonzero$coef)
nonzero <- nonzero[order(nonzero$coef, decreasing=T),]
z_lasso <- z_vec[nonzero$index[1:length(z_0.2)]]

# Figure 14
fig14 <- ggplot() +
  geom_line(data=data.frame(z=z, fdr=fdr), aes(z, fdr, color="red"), 
            linetype="dashed", size=1) +
  geom_line(data=data.frame(z=z, E_delta=E_delta), aes(z, E_delta, color="black"), size=1) +
  geom_point(data=data.frame(x=z_0.2, y=-1.2), aes(x, y), shape=17, color="red") +
  geom_point(data=data.frame(x=z_lasso, y=-1.3), aes(x, y), shape=15, color="green") +
  geom_vline(xintercept=-3.868710, size=0.3, linetype="dotted") +
  geom_vline(xintercept=3.846295, size=0.3, linetype="dotted") +
  geom_hline(yintercept=0.2, size=0.3, linetype="dotted") +
  geom_hline(yintercept=0, size=0.3, linetype="dotted") +
  coord_cartesian(xlim=c(-6, 6), ylim=c(-1.3, 1.3)) +
  scale_color_manual(name=" ",
                     values=c("red"="red", "black"="black"),
                     breaks=c("red", "black"), 
                     labels=c("fdr(z)", "E{delta|z}/4")) +
  labs(x="z value", y="Empirical Bayes Estimates") +
  theme_bw()
# Save the image
image <- paste0("fig14", ".pdf")
pdf(image, height=6, width=8)
fig14
dev.off()


#----------------------------------------------------------------------------------------#
# Example 13: lasso algorithm applied to the supernova data
require(lars)

# Coin supernova data
set.seed(357)
X_sup <- matrix(rnorm(75*25, -1, 2), 75, 25)
mag <- rnorm(75, -1, 2)
mag <- sample(mag, length(mag))
indexi <- sample(75, 30)
indexj <- sample(25, 10)
for (i in indexi) {
  for (j in indexj) {
    X_sup[i, j] <- runif(1, 0, 1)*sample(c(1,-1), 1)
  }
}
supernova <- data.frame(mag=mag, X_sup)

# Lasso on Rpackage lars
fit_las <- lars(X_sup, mag)
best_step <- which.min(fit_las$Cp) # step on the lowest Cp
coef_best <- coef.lars(fit_las, mode="step", s=best_step)
coef_las <- matrix(0, best_step, ncol(X_sup))
for (i in 1:best_step) {
  coef_las[i,] <- coef.lars(fit_las, mode="step", s=i)
}
coef_las <- data.frame(coef_las)
names(coef_las) <- paste0("variable", 1:ncol(X_sup))
coef_las$step <- 1:best_step
nonzero <- sapply(1:ncol(X_sup), function(j) {
  length(which(coef_las[,j] == 0)) != best_step
})
nonzero_col <- (1:ncol(X_sup))[nonzero]
len <-  sapply(nonzero_col, function(i) {
  length((1:best_step)[coef_las[,i]!=0])
})  
var_sort <- nonzero_col[order(len, decreasing=T)]
var_name <- paste0(var_sort)

# Figure 15
fig15 <- ggplot() +
  geom_point(data=data.frame(x=coef_las$step, y=0), aes(x, y), shape=7) +
  geom_line(data=data.frame(x=coef_las$step, y=coef_las[,var_sort[1]]), aes(x, y, color="1")) +
  geom_line(data=data.frame(x=coef_las$step, y=coef_las[,var_sort[2]]), aes(x, y, color="2")) +
  geom_line(data=data.frame(x=coef_las$step, y=coef_las[,var_sort[3]]), aes(x, y, color="3")) +
  geom_line(data=data.frame(x=coef_las$step, y=coef_las[,var_sort[4]]), aes(x, y, color="4")) +
  geom_line(data=data.frame(x=coef_las$step, y=coef_las[,var_sort[5]]), aes(x, y, color="5")) +
  scale_x_continuous(breaks=seq(1, 6, 1)) +
  scale_color_manual(name="variable",
                     values=c("1"="green", "2"="blue", "3"="purple", "4"="red", "5"="black"),
                     breaks=c("1", "2", "3", "4", "5"),
                     labels=var_name) +
  labs(x="step", y="coefficient") +
  theme_bw() 
# Save the image
image <- paste0("fig15", ".pdf")
pdf(image, height=6, width=8)
fig15
dev.off()

