#I already introduced the business problem for classification in the first file.
#I will state the steps for my classification analysis which is similar to the 
#regression, in some way.
#My approach:
#1. Since EDA is already done, I did not do it here
#2.Start by finding best model with metric as accuracy
#3. Special twist, by changing the metric from accuracy to higher recall. This
#might sound confusing now, but will all be clear later on.

#Most importantly, ENJOY!







library(caret)
library(dplyr)
library(GGally)
library(ggplot2)
library(tidyverse)
library(corrplot)
library(gridExtra)
library(ggpubr)
library(moments)
library(rpart)
library(rpart.plot)
library(pROC)
library(AUC)
library(glmnet)
library(C50)
library(randomForest)
library(doParallel)
library(fastAdaboost)

df <- read.csv(file = "WineQuality.csv", 
               header=T,
               sep=",",
               dec=".",
               stringsAsFactors = T)


#Get rid of outliers
outliers = c()
for ( i in 1:11 ) {
  stats = boxplot.stats(df[[i]])$stats
  bottom_outlier_rows = which(df[[i]] < stats[1])
  top_outlier_rows = which(df[[i]] > stats[5])
  outliers = c(outliers , top_outlier_rows[ !top_outlier_rows %in% outliers ] )
  outliers = c(outliers , bottom_outlier_rows[ !bottom_outlier_rows %in% outliers ] )
}

# Detect/Remove outliers
mod = lm(quality ~ ., data = df)
cooksd = cooks.distance(mod)
head(df[cooksd > 20 * mean(cooksd, na.rm=T), ])
coutliers = as.numeric(rownames(df[cooksd > 20 * mean(cooksd, na.rm=T), ]))
outliers = c(outliers , coutliers[ !coutliers %in% outliers ] )
df = df[-outliers, ]

mod = lm(quality ~ ., data = df)
par(mfrow=c(2,3))
lapply(1:6, function(x) plot(mod, which=x, labels.id= 1:nrow(df))) %>% invisible()



#This is where the magic starts: special categorization (>=6 is good)
new_df = df %>% 
  mutate(good = ifelse(quality >= 6, "1", "0") %>% as.factor()) %>%
  select(-quality); dim(df)
table(new_df$good)

#split into test/train
p=ncol(new_df)
idx0 <- createDataPartition(new_df$good, p = .9, 
                                   list = FALSE, 
                                   times = 1)


off_train.x = new_df[idx0, -p] %>% as.matrix()
dim(off_train.x)
off_train.x = off_train.x %>% scale(center=TRUE)
off_train.y = new_df[idx0, p]
official_train = data.frame(off_train.x, good=off_train.y)
dim(official_train)

off_test.x = new_df[-idx0, -p] %>% as.matrix()
off_test.x = off_test.x %>% scale(., center=attr(off_train.x, "scaled:center"),
                                  scale=attr(off_train.x, "scaled:scale"))
off_test.y = new_df[-idx0, p]
official_test = data.frame(off_test.x, good=off_test.y)


#split train into train/validation
idx = createDataPartition(official_train$good, p = 0.9, list=F)

train.x = official_train[idx, -p] %>% as.matrix(); dim(train.x)
train.y = official_train[idx, p]; table(train.y)
train = data.frame(train.x, good=train.y)

test.x = official_train[-idx, -p] %>% as.matrix(); dim(test.x)
test.y = official_train[-idx, p]; table(test.y)
test = data.frame(test.x, good=test.y)





#logistic regression only on few vars for interpretation purposes
fitprova = glm(good~alcohol*density,data = train, family = binomial)
summary(fitprova)
#Model is the following: 
#log-odds(good) = beta0 + beta1*alcohol + beta2*density + beta3*alcohol*density
#e^(b0) = 2.34 ==> odds of wine being good when alcohol and density are set to their means
#e^(b0+b1) = 8.12 ==> odds of wine being good when alcohol increases by 1 sd and 
#density is equal to its mean
#e^(b2) = 1.14 ==> odds ratio of wine being good when density increases by 1 sd and 
#alcohol is equal to its mean
#e^(b2+b3) = 0.85 ==> odds ratio of wine being good when both alcohol and density
#increase by 1sd.
#We conclude that when alcohol or density increase individually, it leads to more chance
#of wine being good, but when they increase together we observe the contrary
#effect (wine become poorer). We also see that alcohol has a bigger effect on 
#whether the wine is good or not, compared to its counterpart density.
#Worthy to note that density is not quite significant so take this analysis with
#a pinch of salt

#let's fit logistic reg on this classification pb. For now let's use a treshold
#of 0.5, but I later came up with an ingenious way to find the optimal treshold
#for our business problem.
threshold = 0.5
glm_mod = glm(good ~. , data = train, family = binomial)
glm_prob = predict(glm_mod, data.frame(test.x), type="response")
glm_class = as.factor(ifelse(glm_prob >= threshold, 1, 0))
cm = confusionMatrix(glm_class, test.y, positive = "1")
glm_eval = cm; glm_eval



#Now Lasso with 10-fold CV
cvlasso = cv.glmnet(train.x, train.y, 
                       family = "binomial",
                       type.measure = "auc")
coef(cvlasso, s=cvlasso$lambda.1se) 

cvlasso_prob = predict(cvlasso, 
                          test.x, 
                          type="response", 
                          s=cvlasso$lambda.1se)

lasso_class = as.factor(ifelse(cvlasso_prob >= threshold, 1, 0))
cm_lasso = confusionMatrix(lasso_class, test.y, positive = "1")
cvlasso_eval = cm; cvlasso_eval


#Decision tree but this time classification problem with boosting
dt = C5.0(train.x, train.y, trials = 10)
dt_pred = predict(dt, test.x)
confMat = confusionMatrix(dt_pred, test.y, positive="1")
dtboost_eval = list(confusionMatrix = confMat); dtboost_eval




#Now, let's fit a RF in 2 ways: a VANILLA one and one with CV
rfcat = randomForest(good~., data=train, ntree=1000, mtry=sqrt(p))
rfcat_prob = predict(rfcat, test.x)
confMat = confusionMatrix(rfcat_prob, test.y, positive="1")
rfcat_eval = list(confusionMatrix = confMat)
rfcat_eval


#Now, for the RF with CV
ct = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid_rf = expand.grid(.mtry = c(p/3, 3, 3.5, sqrt(p)))

cvrfcat = train(good~., data = train,
                     method = 'rf',
                     metric = "Accuracy",
                     trControl = ct,
                     tuneGrid = grid_rf)


cvrfcat_pred = predict(cvrfcat, test.x)
confMat = confusionMatrix(cvrfcat_pred, test.y, positive = "1")
cvrfcat_eval = list(confusionMatrix = confMat); cvrfcat_eval

importance(rfcat)

#Tried to do reduced model, but as expected got a better result with full model
#(even though the most important variables were selected)
cvrfcat.reduced = caret::train(good~alcohol+density+residual.sugar+chlorides, data = train,
                     method = 'rf',
                     metric = "Accuracy",
                     trControl = ct,
                     tuneGrid = grid_rf)


cvrfcat_pred.reduced = predict(cvrfcat.reduced, test.x)
confMat = confusionMatrix(cvrfcat_pred.reduced, test.y, positive = "1")
cvrfcat_eval.reduced = list(confusionMatrix = confMat); cvrfcat_eval.reduced



#xgboost
control <- trainControl(method="repeatedcv", number=10, repeats=3, search="random")
bestmtry <- tuneRF(train.x, train.y, stepFactor=1.5, improve=1e-5, ntree=500)

fitControl = trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 2)

#Following the trouble I got through in the regression problem to only observe
#an ever-so slight improvement in model performance, I will input the most 
#common hyperparameters that I have observed without doing a 150-line
#script to find the best possible hyperparameters.
tune_grid = 
  expand.grid(
    nrounds = c(10, 20, 30, 40, 50, 60, 70, 80, 90, 100),
    eta = 0.3,
    max_depth=5,
    subsample = 1,
    colsample_bytree = 1,
    min_child_weight = 1,
    gamma = c(0.1, 0.2, 0.5, 0.75, 1)
  )

fit_xg_cv = train(good ~ ., data = train, 
                  method = "xgbTree", 
                  trControl = fitControl,
                  verbose = FALSE, 
                  tuneGrid = tune_grid,
                  objective = "binary:logistic" 
                 )
fit_xg_cv
plot(fit_xg_cv)  

pred_xg_cv = predict(fit_xg_cv, test.x, type="raw")
confMat = confusionMatrix(pred_xg_cv, test.y, positive = "1")
xg_eval = list(confusionMatrix = confMat); xg_eval



glm_ = c(glm_eval$confusionMatrix$overall, 
        glm_eval$confusionMatrix$byClass)
lasso_cv = c(cvlasso_eval$confusionMatrix$overall, 
             cvlasso_eval$confusionMatrix$byClass)
classif_tree_boost = c(dtboost_eval$confusionMatrix$overall, 
                        dtboost_eval$confusionMatrix$byClass)
rf = c(rfcat_eval$confusionMatrix$overall, 
       rfcat_eval$confusionMatrix$byClass)
cv_rf = c(cvrfcat_eval$confusionMatrix$overall, 
          cvrfcat_eval$confusionMatrix$byClass)
rf_cv_reduced = c(cvrfcat_eval.reduced$confusionMatrix$overall,
          cvrfcat_eval.reduced$confusionMatrix$byClass)
xgb_cv = c(xg_eval$confusionMatrix$overall, 
                          xg_eval$confusionMatrix$byClass)

all = cbind(glm_, lasso_cv, 
             classif_tree_boost,
            rf, cv_rf, rf_cv_reduced, xgb_cv) %>% data.frame()

knitr::kable(all %>% round(3),
             caption = "comparing all models")

#Result of best model on test set (the official one, not the validation that
#we were using before)
cvrfcat_off = train(good~., data = official_train,
                method = 'rf',
                metric = "Accuracy",
                trControl = ct,
                tuneGrid = grid_rf)


cvrfcat_pred_off = predict(cvrfcat_off, off_test.x)
confMat = confusionMatrix(cvrfcat_pred_off, off_test.y, positive = "1")
cvrfcat_eval_off = confMat; cvrfcat_eval_off


#Maximizing accuracy is a very good metric, but let's think a bit outside the box 
#How to pick best treshold? Above, I picked threshold based on getting better accuracy.
#But let us think about it. Is accuracy what really matters?
#Let me ask you this: how many times did you really think that a wine tasted 
#horrible? Very rarely right? Let's be real, most of us are not connoisseurs
#and will not notice a huge difference between a wine that is "not bad", a wine
#that is "good" and a wine that is "very good". This is why I think that at the
#beginning when marketing the product in front of the jury, we will use accuracy
#as a metric to show them that we are right more times than not; and when our 
#product goes into the market, we will use our secret tactic, allowing for more 
#FP compared to FN. So our threshold will be less than 0.5 and we will use a cost
#matrix in a fictional scenario that I will create myself.

#1. Predict wine good - actually is good: -40
#2. Predict wine bad - actually is bad: -40
#3. Predict wine good- actually is bad: 10
#4. Predict wine bad - actually is good: 25

#In other words,
#TP = -40
#TN = -40
#FN = 25
#FP = 10

#So we get: Cost = 25FN - 40TP + 10FP - 40TN
#Let me further explain: when we predict that wine is bad but it is actually 
#good, this doesn't mean that it is good according to the user, but just that
#it is good following the parameters we inputted into our model. Indeed, a 
#user won't make a difference between a wine that is rated 5.8 or 5.9 compared
#to 6.1 or 6.2. Thus, we can profit from this margin by penalizing more FN. 
#The concept that I have in mind is hard to explain in words, but I hope to 
#convince you about it in my presentation!

thresh <- seq(0.1,1.0, length = 10)
#cost vector
cost_tr = rep(0,length(thresh))
#for training set, let's see what threshold is the best to use to minimize costs
#for training set
for (i in 1:length(thresh)){
  
  glm = rep("0", length(glm_mod$fitted.values))
  glm[glm_mod$fitted.values > thresh[i]] = "1"
  glm <- as.factor(glm)
  x <- confusionMatrix(glm, train.y, positive = "1")
  TN <- x$table[1]
  FP <- x$table[2]
  FN <- x$table[3]
  TP <- x$table[4]
  cost_tr[i] = 25*FN + TP * (-40) + 10 * FP + TN *(-40)
}

x <- confusionMatrix(glm, train.y, positive = "1")
TN <- x$table[1]
FP <- x$table[2]
FN <- x$table[3]
TP <- x$table[4]
cost_simple_tr = 25*FN + TP * (-40) + 10 * FP + TN *(-40)



# putting results in a dataframe for plotting
dat <- data.frame(
  model = c(rep("optimized",10),"simple"),
  cost_per_customer = c(cost_tr,cost_simple_tr),
  threshold = c(thresh,0.5)
)

# plotting
plot <- ggplot(dat, aes(x = threshold, y = cost_per_customer, group = model, colour = model)) +
  geom_line() +
  geom_point()

plot

# cost as a function of threshold
wine.probs <- predict(glm_mod, data.frame(test.x), type = "response")

cost = rep(0,length(thresh))

for (i in 1:length(thresh)){
  
  glm.pred = rep("0", length(wine.probs))
  glm.pred[wine.probs > thresh[i]] = "1"
  glm.pred <- as.factor(glm.pred)
  x <- confusionMatrix(glm.pred, test.y, positive = "1")
  TN <- x$table[1]
  FP <- x$table[2]
  FN <- x$table[3]
  TP <- x$table[4]
  cost[i] = 25*FN + TP * (-40) + 10 * FP + TN *(-40)
}


#for the simple model, take treshold as 0.5
glm.pred = rep("0", length(glm.pred))
glm.pred[wine.probs > 0.5] = "1"
glm.pred <- as.factor(glm.pred)

x <- confusionMatrix(glm.pred, test.y, positive = "1")
TN <- x$table[1]
FP <- x$table[2]
FN <- x$table[3]
TP <- x$table[4]
cost_simple = 25*FN + TP * (-40) + 10 * FP + TN *(-40)



# putting results in a dataframe for plotting
dat <- data.frame(
  model = c(rep("optimized",10),"simple"),
  cost_per_customer = c(cost,cost_simple),
  threshold = c(thresh,0.5)
)



# plotting
plot2 <- ggplot(dat, aes(x = threshold, y = cost_per_customer, group = model, colour = model)) +
  geom_line() +
  geom_point()

dev.off()
par(mfrow=c(1,2))
grid.arrange(plot,plot2, ncol=2)
