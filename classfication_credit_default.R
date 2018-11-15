pacman::p_load("tidyverse", "readxl", "caret", "MASS", "ROCR", "glmnet", "doParallel", "gridExtra")

setwd("/Users/yangyoyo/Desktop/MMA/M2 - Predictive Modelling/Data")

credit_raw <- read_excel("credit data.xlsx")
new_app <- read.csv("new_applications.csv")

str(credit_raw)
summary(credit_raw)

str(new_app)
summary(new_app)

### Data Cleaning ----------------------------------------------------------------------------------
# Convert credit limit to numeric
credit_raw$LIMIT_BAL <- as.numeric(credit_raw$LIMIT_BAL)

# Convert payment amount (Apr - Sep) to numeric
amt_index <- grepl("PAY_AMT[0-9]", names(credit_raw))
credit_raw[, amt_index] <- lapply(credit_raw[, amt_index], as.numeric)

# Combine credit_raw and new_app
credit <- union_all(credit_raw[, -c(1:2)], new_app[, -1])
str(credit)
rm(credit_raw)

# Recode gender
credit$SEX <- credit$SEX %>% 
    recode("1" = "Male", "2" = "Female") %>% 
    as.factor()

# Recode education
credit$EDUCATION_surrogate <- as.factor(ifelse(credit$EDUCATION == 0, "1", "0"))
credit$EDUCATION <- credit$EDUCATION %>% 
    recode("1" = "graduate school", "2" = "university", "3" = "high school", "4" = "others",
           "5" = "unknown", "6" = "unknown", .default = "unknown") %>% 
    as.factor()

# Recode marrital status
credit$MARRIAGE <- credit$MARRIAGE %>% 
    recode("1" = "married", "2" = "single", "3" = "others", .default = "unknown") %>% 
    as.factor()

# Recode pay status (Apr - Sep)
pay_index <- grepl("PAY_[0-9]", names(credit))
recode.pay <- function(x) {
    x <- recode(x, "-2" = "no consumption", "-1" = "pay duly", "0" = "use of revolving credit",
           "1" = "1mo", "2" = "2mo", "3" = "3mo", "4" = "4mo", "5" = "5mo", .default = "6mo+")
    x <- as.factor(x)
}
credit[, pay_index] <- lapply(credit[, pay_index], recode.pay)
names(credit)[pay_index] <- c("SEP_PAY_STATUS", "AUG_PAY_STATUS", "JUL_PAY_STATUS",
                              "JUN_PAY_STATUS", "MAY_PAY_STATUS", "APR_PAY_STATUS")
rm(pay_index)
rm(recode.pay)

# Recode bill amount names
bill_index <- grepl("BILL_AMT[0-9]", names(credit))
names(credit)[bill_index] <- c("SEP_BILL", "AUG_BILL", "JUL_BILL", "JUN_BILL", "MAY_BILL",
                               "APR_BILL")
rm(bill_index)

# Recode payment amount column names
amt_index <- grepl("PAY_AMT[0-9]", names(credit))
names(credit)[amt_index] <- c("PAYMENT_FOR_AUG", "PAYMENT_FOR_JUL", "PAYMENT_FOR_JUN",
                              "PAYMENT_FOR_MAY", "PAYMENT_FOR_APR", "PAYMENT_FOR_MAR")
rm(amt_index)

# Convert default to factor
credit$default.payment.next.month <- as.factor(credit$default.payment.next.month)
levels(credit$default.payment.next.month) <- c("N", "Y")

# Examine cleaned dataset structure
str(credit)
summary(credit)

### Feature Engineering ----------------------------------------------------------------------------
# Create monthly, 3-month & 6-month average utilization rates
credit <- credit %>%
    mutate(SEP_UTIL = ifelse(SEP_BILL < 0, 0, SEP_BILL/LIMIT_BAL),
           AUG_UTIL = ifelse(AUG_BILL < 0, 0, AUG_BILL/LIMIT_BAL),
           JUL_UTIL = ifelse(JUL_BILL < 0, 0, JUL_BILL/LIMIT_BAL),
           JUN_UTIL = ifelse(JUN_BILL < 0, 0, JUN_BILL/LIMIT_BAL),
           MAY_UTIL = ifelse(MAY_BILL < 0, 0, MAY_BILL/LIMIT_BAL),
           APR_UTIL = ifelse(APR_BILL < 0, 0, APR_BILL/LIMIT_BAL))

credit$"3MO_UTIL" <- rowMeans(credit[, 26:28])
credit$"6MO_UTIL" <- rowMeans(credit[, 26:31])

# Create # of defaults in the past 6 months
credit$SEP_DEFAULT <- grepl("[1-9]", credit$SEP_PAY_STATUS)
credit$AUG_DEFAULT <- grepl("[1-9]", credit$AUG_PAY_STATUS)
credit$JUL_DEFAULT <- grepl("[1-9]", credit$JUL_PAY_STATUS)
credit$JUN_DEFAULT <- grepl("[1-9]", credit$JUN_PAY_STATUS)
credit$MAY_DEFAULT <- grepl("[1-9]", credit$MAY_PAY_STATUS)
credit$APR_DEFAULT <- grepl("[1-9]", credit$APR_PAY_STATUS)

credit$"3MO_DEFAULT" <- rowSums(credit[, 34:36])
credit$"6MO_DEFAULT" <- rowSums(credit[, 34:39])

# Separate new application from dataset
new_app <- filter(credit, is.na(default.payment.next.month))
credit <- filter(credit, !is.na(default.payment.next.month))

### Model Construction with Feature Engineering ----------------------------------------------------
set.seed(543145)
inTrain <- createDataPartition(credit$default.payment.next.month, p = 0.8, list = FALSE)
training <- credit[inTrain, ]
testing <- credit[-inTrain, ]

table(training$default.payment.next.month)

## Gradient Boosting ----
# Search for the best tuning parameters with a grid search
ctrl <- trainControl(method = "cv", number = 10,
                     classProbs = TRUE,
                     summaryFunction = twoClassSummary,
                     sampling = "down",
                     search = "grid",
                     verboseIter = TRUE)

max(0.01, 0.1*min(1, nrow(training)/10000)) #max shrinkage
floor(sqrt(ncol(training))) #max interation depth
grid <- expand.grid(interaction.depth = c(1, 3, 6),
                    n.trees = (0:30)*50,
                    shrinkage = seq(.005, .1, .005),
                    n.minobsinnode = 20)

cl <- makePSOCKcluster(4)
registerDoParallel(cl)

set.seed(42)
mod_gbm <- train(default.payment.next.month ~ ., data = training,
                 method = "gbm",
                 metric = "ROC",
                 trControl = ctrl,
                 tuneGrid = grid,
                 verbose = TRUE)

mod_gbm
plot(mod_gbm)

# Reproduce the best model
grid <- expand.grid(interaction.depth = 6,
                    n.trees = 100,
                    shrinkage = .065,
                    n.minobsinnode = 20)

set.seed(42)
mod_gbm_under <- train(default.payment.next.month ~ ., data = training,
                      method = "gbm",
                      metric = "ROC",
                      trControl = ctrl,
                      tuneGrid = grid,
                      verbose = TRUE)

mod_gbm_under

# Find predicitons
gbm_prob_under <- predict(mod_gbm_under, newdata = testing, type = "prob")[, "Y"]
gbm_class_under <- predict(mod_gbm_under, newdata = testing)

# Confusion matrix  
confusionMatrix(gbm_class_under, testing$default.payment.next.month)

# ROC Curve
gbm_ROC_pred_under <- prediction(gbm_prob_under, testing$default.payment.next.month)
gbm_ROC_under <- performance(gbm_ROC_pred_under, "tpr", "fpr")
plot(gbm_ROC_under) 

# AUC
auc.tmp <- performance(gbm_ROC_pred_under, "auc")
gbm_auc <- as.numeric(auc.tmp@y.values)
gbm_auc

## Random Forest ----
# Search for the best tuning parameter with a random search
ctrl$search <- "random"

set.seed(1087)
mod_rf <- train(default.payment.next.month ~ ., data = training,
                method = "rf",
                metric = "ROC",
                trControl = ctrl)

mod_rf
plot(mod_rf)

# Reproduce the best model
ctrl$search <- "grid"
grid <- expand.grid(.mtry = 36)

set.seed(1087)
mod_rf_best <- train(default.payment.next.month ~ ., data = training,
                     method = "rf",
                     metric = "ROC",
                     trControl = ctrl,
                     tuneGrid = grid)

mod_rf_best
varImp(mod_rf_best)

# Find predicitons
rf_prob <- predict(mod_rf_best, newdata = testing, type = "prob")[, "Y"]
rf_class <- predict(mod_rf_best, newdata = testing)

# Confusion matrix  
confusionMatrix(rf_class, testing$default.payment.next.month)

# ROC Curve
rf_ROC_pred <- prediction(rf_prob, testing$default.payment.next.month)
rf_ROC <- performance(rf_ROC_pred, "tpr", "fpr")
plot(rf_ROC) 

# AUC
auc.tmp <- performance(rf_ROC_pred, "auc")
rf_auc <- as.numeric(auc.tmp@y.values)
rf_auc

# Undersamping
ctrl$sampling <- "down"

set.seed(1087)
mod_rf_under <- train(default.payment.next.month ~ ., data = training,
                     method = "rf",
                     metric = "ROC",
                     trControl = ctrl,
                     tuneGrid = grid)

mod_rf_under
varImp(mod_rf_under)

# Find predicitons
rf_prob_under <- predict(mod_rf_under, newdata = testing, type = "prob")[, "Y"]
rf_class_under <- predict(mod_rf_under, newdata = testing)

# Confusion matrix  
confusionMatrix(rf_class_under, testing$default.payment.next.month)

# ROC Curve
rf_ROC_pred_under <- prediction(rf_prob_under, testing$default.payment.next.month)
rf_ROC_under <- performance(rf_ROC_pred_under, "tpr", "fpr")
plot(rf_ROC_under)

# AUC
auc.tmp <- performance(rf_ROC_pred_under, "auc")
rf_auc_under <- as.numeric(auc.tmp@y.values)
rf_auc_under

### Re-run Best Model without Sex ------------------------------------------------------------------
ctrl$sampling
grid <- expand.grid(interaction.depth = 6,
                    n.trees = 100,
                    shrinkage = .065,
                    n.minobsinnode = 20)

set.seed(42)
mod_gbm_under_wo_sex <- train(default.payment.next.month ~ ., data = training[, -2],
                      method = "gbm",
                      metric = "ROC",
                      trControl = ctrl,
                      tuneGrid = grid,
                      verbose = TRUE)

mod_gbm_under_wo_sex

# Find predicitons
gbm_prob_under_wo_sex <- predict(mod_gbm_under_wo_sex, newdata = testing[, -2], type = "prob")[, "Y"]
gbm_class_under_wo_sex <- predict(mod_gbm_under_wo_sex, newdata = testing[, -2])

# Confusion matrix  
confusionMatrix(gbm_class_under_wo_sex, testing$default.payment.next.month)

# ROC Curve
gbm_ROC_pred_under_wo_sex <- prediction(gbm_prob_under_wo_sex, testing$default.payment.next.month)
gbm_ROC_under_wo_sex <- performance(gbm_ROC_pred_under_wo_sex, "tpr", "fpr")
plot(gbm_ROC_under_wo_sex) 

# AUC
auc.tmp <- performance(gbm_ROC_pred_under_wo_sex, "auc")
gbm_auc <- as.numeric(auc.tmp@y.values)
gbm_auc

# Predict new applications
new_app_prob_wo_sex <- round(predict(mod_gbm_under_wo_sex, newdata = new_app[, -2], type = "prob"), digits = 2)
new_app_prob <- round(predict(mod_gbm_under, newdata = new_app, type = "prob"), digits = 2)
list(with_sex = new_app_prob, without_sex = new_app_prob_wo_sex)

new_app_class_wo_sex <- predict(mod_gbm_under_wo_sex, newdata = new_app[, -2])
new_app_class <- predict(mod_gbm_under, newdata = new_app)
list(with_sex = new_app_class, without_sex = new_app_class_wo_sex)


### Explore the Impact of Gender -------------------------------------------------------------------
male <- filter(credit, SEX == "Male")
female <- filter(credit, SEX == "Female")

dim(male); dim(female)

# Create training & testing sets
male_test <- filter(testing, SEX == "Male")
female_test <- filter(testing, SEX == "Female")

# Find predicitons (with SEX)
male_prob <- predict(mod_gbm_under, newdata = male_test, type = "prob")
female_prob <- predict(mod_gbm_under, newdata = female_test, type = "prob")[, "Y"]

cutoff <- seq(0, 1, 0.01)

pred <- data.frame(threshold = numeric(),
                   male_perc = numeric(), 
                   female_perc = numeric())

for(i in cutoff) {
    
    male_class <- ifelse(male_prob <= i, "Give Credit", "Don't Give Credit")
    female_class <- ifelse(female_prob <= i, "Give Credit", "Don't Give Credit")
    male <- round(mean(male_class == "Give Credit"), digits = 2)
    female <- round(mean(female_class == "Give Credit"), digits = 2)
    
    pred[nrow(pred) + 1, ] <- c(i, male, female)
}

pred

# Find predicitons
male_prob_wo_sex <- predict(mod_gbm_under_wo_sex, newdata = male_test[, -2], type = "prob")[, "Y"]
female_prob_wo_sex <- predict(mod_gbm_under_wo_sex, newdata = female_test[, -2], type = "prob")[, "Y"]

pred_wo_sex <- data.frame(threshold = numeric(),
                   male_perc = numeric(), 
                   female_perc = numeric())

for(i in cutoff) {
    
    male_class_wo_sex <- ifelse(male_prob_wo_sex <= i, "Give Credit", "Don't Give Credit")
    female_class_wo_sex <- ifelse(female_prob_wo_sex <= i, "Give Credit", "Don't Give Credit")
    male <- round(mean(male_class_wo_sex == "Give Credit"), digits = 2)
    female <- round(mean(female_class_wo_sex == "Give Credit"), digits = 2)
    
    pred_wo_sex[nrow(pred_wo_sex) + 1, ] <- c(i, male, female)
}

pred_wo_sex

# Plot results
without <- pred_wo_sex %>% 
    gather(sex, perc, -1) %>% 
    ggplot(aes(x = threshold, y = perc, col = sex)) +
    geom_line() +
    labs(title = "Model without Sex",
         x = "Threshold",
         y = "% of Credit Given")

with <- pred %>%
    gather(sex, perc, -1) %>% 
    ggplot(aes(x = threshold, y = perc, col = sex)) +
    geom_line() +
    labs(title = "Model with Sex",
         x = "Threshold",
         y = "% of Credit Given")


grid.arrange(with, without)

female <- ggplot() +
    geom_line(data = pred, aes(x = threshold, y = female_perc), col = "salmon") +
    geom_line(data = pred_wo_sex, aes(x = threshold, y = female_perc), col = "black") +
    labs(title = "% of Credit Given to Female in Models with & without Sex",
         x = "Threshold",
         y = "% of Credit Given") +
    scale_color_manual(name = "Model", values = c("w Sex", "wo Sex"))

male <- ggplot() +
    geom_line(data = pred, aes(x = threshold, y = male_perc), col = "skyblue3") +
    geom_line(data = pred_wo_sex, aes(x = threshold, y = male_perc), col = "black") +
    labs(title = "% of Credit Given to Male in Models with & without Sex",
         x = "Threshold",
         y = "% of Credit Given") +
    scale_color_discrete(name = "Model", labels = c("w Sex", "wo Sex"))
    

grid.arrange(female, male)
