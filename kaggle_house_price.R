library(tidyverse)
library(caret)
library(glmnet)
library(Metrics)

### Import Dataset ---------------------------------------------------------------------------------
setwd("/Users/yangyoyo/Desktop/MMA/M2 - Predictive Modelling/Individual Ass")
train_raw <- read.csv("train.csv")
test_raw <- read.csv("test.csv")

dim(train_raw)
dim(test_raw)

### Explore Train Dataset --------------------------------------------------------------------------
# Examine train dataset structure
glimpse(train_raw)
summary(train_raw)

# visualisation is done in Tableau

### Data Cleaning ----------------------------------------------------------------------------------
## Combine train & test
tnt <- union_all(train_raw, test_raw)
dim(tnt)

## Convert MSSubClass into factor - it should be a categorical variable, not numeric
summary(tnt)
fac_var <- c("MSSubClass", names(tnt[, lapply(tnt, class)=="character"]))
tnt[, fac_var] <- data.frame(lapply(tnt[, fac_var], factor))

## Relevel ordinal variables
# Check current levels
ord_var <- c("Condition1", "Condition2", "ExterQual", "ExterCond", "BsmtQual", "BsmtCond",
             "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "HeatingQC", "Electrical", "KitchenQual",
             "Functional", "FireplaceQu", "GarageFinish", "GarageQual", "GarageCond", "PavedDrive",
             "PoolQC", "Fence")
lapply(tnt[, ord_var], levels)

# Group variables per their level scale and order them with correct level rank
ord_var_g1 <- c("Condition1", "Condition2")
tnt[, ord_var_g1] <- data.frame(lapply(tnt[, ord_var_g1], ordered,
                                         levels = c("RRAe", "RRNe", "PosA", "PosN", "RRAn", "RRNn",
                                                    "Norm", "Feedr", "Artery")))

ord_var_g2 <- c("ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC", "KitchenQual",
                "FireplaceQu", "GarageQual", "GarageCond", "PoolQC")
tnt[, ord_var_g2] <- data.frame(lapply(tnt[, ord_var_g2], ordered, 
                                         levels = c("NA", "Po", "Fa", "TA", "Gd", "Ex")))

ord_var_g3 <- c("BsmtFinType1", "BsmtFinType2")
tnt[, ord_var_g3] <- data.frame(lapply(tnt[, ord_var_g3], ordered,
                                         levels = c("NA", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ")))

tnt$BsmtExposure <- ordered(tnt$BsmtExposure, levels = c("NA", "No", "Mn", "Av", "Gd"))
tnt$Electrical <- ordered(tnt$Electrical, levels = c("Mix", "FuseP", "FuseF", "FuseA", "SBrkr"))
tnt$Functional <- ordered(tnt$Functional, levels = c("Sal", "Sev", "Maj2", "Maj1", "Mod",
                                                         "Min2", "Min1", "Typ"))
tnt$GarageFinish <- ordered(tnt$GarageFinish, levels = c("NA", "Unf", "RFn", "Fin"))
tnt$PavedDrive <- ordered(tnt$PavedDrive, levels = c("N", "P", "Y"))
tnt$Fence <- ordered(tnt$Fence, levels = c("NA", "MnWw", "GdWo", "MnPrv", "GdPrv"))

## Handle missing values
# Examine columns with missing values
missing <- colSums(is.na(tnt))
sort(missing[missing > 0], decreasing = TRUE)

# Recode or impute columns with missing values one by one based on data description --
# PoolQC - NA means no pool, recode NA as its own factor level
tnt$PoolQC[is.na(tnt$PoolQC)] <- "NA"
sum(is.na(tnt$PoolQC))
summary(tnt$PoolQC)

# MiscFeature - NA means no misc feature, recode NA as a level
tnt$MiscFeature <- addNA(tnt$MiscFeature)
sum(is.na(tnt$MiscFeature))
summary(tnt$MiscFeature)

# Alley - NA means no Alley access, recode NA as a level
tnt$Alley <- addNA(tnt$Alley)
sum(is.na(tnt$Alley))
summary(tnt$Alley)

# Fence - NA means no fence, recode NA as a level
tnt$Fence[is.na(tnt$Fence)] <- "NA"
sum(is.na(tnt$Fence))
summary(tnt$Fence)

# FireplaceQu - NA means no fireplace, recode NA as a level
tnt$FireplaceQu[is.na(tnt$FireplaceQu)] <- "NA"
sum(is.na(tnt$FireplaceQu))
summary(tnt$FireplaceQu)

# LotFrontage - NA means values are missing. Check the percentage of missing values < 0.40, so we
# can impute it. Assuming similar houses will mostly likely have similar sized front lot, we choose
# to impute it with the median impute method
mean(is.na(tnt$LotFrontage))

preproc_imp <- preProcess(tnt[, -c(1, 81)], method = "medianImpute")
tnt$LotFrontage <- predict(preproc_imp, tnt[, -c(1, 81)])$LotFrontage
summary(tnt$LotFrontage)

# Impute GarageYrBlt
mean(is.na(tnt$GarageYrBlt))
tnt$GarageYrBlt <- predict(preproc_imp, tnt[, -c(1, 81)])$GarageYrBlt
summary(tnt$GarageYrBlt)

# Discovered incorrect input (year 2207), correct it
tnt$GarageYrBlt[tnt$GarageYrBlt > 2010] <- 2007

# Garage - NA means no garage, recode NA for factors as a level, numeric variables as 0
garage <- c("GarageFinish", "GarageQual", "GarageCond")
tnt[, garage][is.na(tnt[, garage])] <- "NA"
sum(is.na(tnt[, garage]))
summary(tnt[, garage])

tnt$GarageType <- addNA(tnt$GarageType)

garage_1 <- c("GarageCars", "GarageArea")
tnt[, garage_1][is.na(tnt$GarageCars), ] <- rep(0, 2)

# Basement - NA means no basement, recode NA for factors as a level, numeric variables as 0
bsmt <- c("BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2")
tnt[, bsmt][is.na(tnt[, bsmt])] <- "NA"
sum(is.na(tnt[, bsmt]))
summary(tnt[, bsmt])

bsmt_1 <- c("BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF")
tnt[, bsmt_1][is.na(tnt$TotalBsmtSF), ] <- rep(0, 6)

bsmt_2 <- c("BsmtFullBath", "BsmtHalfBath")
tnt[, bsmt_2][is.na(tnt$BsmtFullBath), ] <- rep(0, 2)

# MasVnrType, MasVnrArea - NA most likely means there is masonry veneer, so recode MasVnrType NA as
# "None", MasVnrArea NA as 0
tnt$MasVnrType <- replace(tnt$MasVnrType, is.na(tnt$MasVnrType), "None")
tnt$MasVnrArea <- replace(tnt$MasVnrArea, is.na(tnt$MasVnrArea), 0)

# impute remaining missing values with the most frequent values
tnt$MSZoning[is.na(tnt$MSZoning)] <- names(sort(table(tnt[, "MSZoning"]), decreasing = TRUE)[1])
tnt$Functional[is.na(tnt$Functional)] <- names(sort(table(tnt[, "Functional"]), decreasing = TRUE)[1])
tnt$Exterior1st[is.na(tnt$Exterior1st)] <- names(sort(table(tnt[, "Exterior1st"]), decreasing = TRUE)[1])
tnt$Exterior2nd[is.na(tnt$Exterior2nd)] <- names(sort(table(tnt[, "Exterior2nd"]), decreasing = TRUE)[1])
tnt$KitchenQual[is.na(tnt$KitchenQual)] <- names(sort(table(tnt[, "KitchenQual"]), decreasing = TRUE)[1])
tnt$Utilities[is.na(tnt$Utilities)] <- names(sort(table(tnt[, "Utilities"]), decreasing = TRUE)[1])
tnt$SaleType[is.na(tnt$SaleType)] <- names(sort(table(tnt[, "SaleType"]), decreasing = TRUE)[1])

# Electrical - only has one NA and it indicates missing value. After examining the missing value, we
# find that the house was built in 2006 and belongs to neighborhood Timber. The electricity type of
# similar houses at Timber is all SBrkr, so recode it as SBrkr
tnt[is.na(tnt$Electrical), c("Neighborhood", "YearBuilt")]
table(tnt[tnt$Neighborhood == "Timber", "Electrical"])
tnt$Electrical[is.na(tnt$Electrical)] <- "SBrkr"
summary(tnt$Electrical)

# Check if there are any remaining NA values other than SalePrice - there is none
sort(missing[missing > 0], decreasing = TRUE)
rm(missing)

summary(tnt)

### Data Preprocessing -----------------------------------------------------------------------------
# Re-split train & test dataset
train <- tnt[!is.na(tnt$SalePrice), ]
test <- tnt[is.na(tnt$SalePrice), ]

dim(train); dim(test)

# Remove outliers from train dataset identified via previous exploration
train <- filter(train, GrLivArea < 4000 & LotArea < 70000 & LotFrontage < 300)

# Remove variables that have near zero variance
nzv0 <- nearZeroVar(train, saveMetrics = TRUE)
summary(nzv0) # get the cutoff value for frequency (> 3rd Qu.) and unique percentage (< 1st Qu.)
nzv <- nearZeroVar(train, freqCut = 21.459, uniqueCut = 0.3443, saveMetrics = TRUE)
nzv[nzv$nzv, ] # check which variables will be removed
nzv <- nearZeroVar(train, freqCut = 21.459, uniqueCut = 0.3443)

train <- train[, -nzv]
test <- test[, -nzv]

# Log transform SalePrice
train$SalePrice <- log(train$SalePrice)

# Create training set (75%) & validation set (25%)
set.seed(1)
in_train <- createDataPartition(y = train$SalePrice, p = 0.75, list = FALSE)
training <- train[in_train, ]
validation <- train[-in_train, ]

### Model Construction -----------------------------------------------------------------------------
## Linear Regression - validation RMSE 0.1230584 ----
# Select numeric independant variables for BoxCox transformation
names(train[, sapply(train, class) == "numeric" | sapply(train, class) == "integer"])
trans_var <- c("LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF",
               "TotalBsmtSF", "X1stFlrSF", "X2ndFlrSF", "LowQualFinSF", "GrLivArea", "GarageArea",
               "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "X3SsnPorch" ,"ScreenPorch",
               "PoolArea", "MiscVal")

# Transform independant variables
tnt_bc <- union_all(train, test)
preproc_bc <- preProcess(tnt_bc[, -c(1, 75)], method = "BoxCox")
tnt_bc[, trans_var] <- predict(preproc_bc, tnt_bc[, -c(1, 75)])[, trans_var]

train_bc <- tnt_bc[tnt_bc$Id <= 1460, ]
test_bc <- tnt_bc[tnt_bc$Id > 1460, ]

training_bc <- train_bc[in_train, ]
validation_bc <- train_bc[-in_train, ]

# Construct linear model
mod_lm <- train(SalePrice ~ ., data = training_bc[, -1], method = "lm", metric = "RMSE")
plot(mod_lm$finalModel$residuals)
plot(mod_lm$finalModel$fitted.values)

val_lm <- predict(mod_lm, validation_bc[, -75])
rmse_lm <- rmse(validation_bc$SalePrice, val_lm)
rmse_lm

## LASSO Regression - validation RMSE 0.1079844 ----
tnt <- union_all(train, test)

# Create interaction terms and x matrix
type_var <- model.matrix(Id ~ Neighborhood*BldgType*HouseStyle*SaleCondition, tnt)[, -1]
qual_var <- model.matrix(Id ~ ExterQual*BsmtQual*KitchenQual*CentralAir, tnt)[, -1]

X <- model.matrix(Id ~ (GrLivArea + log(GrLivArea) + sqrt(GrLivArea)) * (type_var + qual_var) +
                      LotArea + log(LotArea) + sqrt(LotArea) + LotFrontage + log(LotFrontage) +
                      sqrt(LotFrontage) + MSSubClass + MSZoning + Alley + LotShape + LotConfig +
                      Condition1 + Condition2 + OverallQual + OverallCond + YearBuilt +
                      YearRemodAdd + RoofStyle + RoofMatl + Exterior1st + Exterior2nd + MasVnrType +
                      MasVnrArea + ExterCond + Foundation + BsmtCond + BsmtExposure + BsmtFinType1 +
                      BsmtFinType2 + BsmtFinSF1 + BsmtFinSF2 + BsmtUnfSF + TotalBsmtSF + Heating +
                      HeatingQC + Electrical + X1stFlrSF + X2ndFlrSF + LowQualFinSF + BsmtFullBath +
                      BsmtHalfBath + FullBath + HalfBath + BedroomAbvGr + KitchenAbvGr +
                      TotRmsAbvGrd + Functional + Fireplaces + FireplaceQu + GarageType +
                      GarageYrBlt + GarageFinish + GarageCars + GarageArea + GarageQual +
                      GarageCond + PavedDrive + WoodDeckSF + OpenPorchSF + EnclosedPorch +
                      X3SsnPorch + ScreenPorch + PoolArea + Fence + MiscVal + MoSold + YrSold +
                      SaleType, tnt)[, -1]

X <- cbind(Id = tnt$Id, X)
y <- training$SalePrice

# Re-split train & test set, then re-split train set to training & validation
X_train <- X[tnt$Id <= 1460, ]
X_test <- X[tnt$Id > 1460, ]

X_training <- X_train[in_train[, 1], ]
X_validation <- X_train[-in_train[, 1], ]

# Construct LASSO regression
mod_lasso0 <- glmnet(x = X_training, y = y, alpha = 1)
plot(mod_lasso0, xvar = "lambda")

# Select the best lambda
crossval <-  cv.glmnet(x = X_training, y = y, alpha = 1)
plot(crossval)
pen_lasso <- crossval$lambda.min
mod_lasso <- glmnet(x = X_training, y = y, alpha = 1, lambda = pen_lasso)

# Predict performance on validation
val_lasso <- predict(mod_lasso, s = pen_lasso, newx = X_validation)
rmse_lasso <- rmse(validation$SalePrice, val_lasso)
rmse_lasso

## Ridge Regression - validation RMSE 0.1521671 ----
crossval <-  cv.glmnet(x = X_training, y = y, alpha = 0)
plot(crossval)
pen_ridge <- crossval$lambda.min
mod_ridge <- glmnet(x = X_training, y = y, alpha = 0, lambda = pen_ridge)

val_ridge <- predict(mod_ridge, s = pen_ridge, newx = X_validation)
rmse_ridge <- rmse(validation$SalePrice, val_ridge)
rmse_ridge


## Elastic Net - validation RMSE 0.108558 ----
alpha <- 0.5

crossval <-  cv.glmnet(x = X_training, y = y, alpha = alpha)
pen_eln <- crossval$lambda.min
mod_eln <- glmnet(x = X_training, y = y, alpha = alpha, lambda = pen_eln)

val_eln <- predict(mod_eln, s = pen_eln, newx = X_validation)
rmse_eln <- rmse(validation$SalePrice, val_eln)
rmse_eln


## Gradient Boosting - validation RMSE 0.1082946 ----
# Construct gbm model
mod_gbm <- train(SalePrice ~ ., data = training_bc[, -1], method = "gbm", metric = "RMSE",
                 trControl = trainControl(method = "repeatedcv", number = 5))

val_gbm <- predict(mod_gbm, validation_bc[, -75])
rmse_gbm <- rmse(validation_bc$SalePrice, val_gbm)
rmse_gbm


## Random Forest - validation RMSE 0.1257456 ----
mod_rf <- train(SalePrice ~ ., data = training_bc[, -1], method = "rf", metric = "RMSE",
                trControl = trainControl(method = "repeatedcv", number = 5), allowParellel = TRUE)

val_rf <- predict(mod_rf, validation_bc[, -75])
rmse_rf <- rmse(validation_bc$SalePrice, val_rf)
rmse_rf


## Stacking selected models ----
# Based on validation performance, select LASSO, elastic net, gbm as base models for final stacking
val_df <- data.frame(price = validation$SalePrice,
                     pred1 = val_lasso, pred2 = val_eln, pred3 = val_gbm, pred4 = val_rf)
mod_stack <- train(price ~ ., data = val_df, method = "glm", metric = "RMSE")
val_stack <- predict(mod_stack, val_df)
rmse_stack <- rmse(validation$SalePrice, val_stack)
rmse_stack


### Predicting Test Data ---------------------------------------------------------------------------
# Predict using indivial models
pred_lm <- predict(mod_lm, test)
pred_rf <- predict(mod_rf, test_bc)
pred_qrf <- predict(mod_qrf, test_bc)
pred_gbm <- predict(mod_gbm, test_bc)

pred_lasso <- predict(mod_lasso, s = pen_lasso, newx = X_test)
pred_eln <- predict(mod_eln, s = pen_eln, newx = X_test)

# Predict using final stacked model
pred_df <- data.frame(pred1 = pred_lasso, pred2 = pred_eln, pred3 = pred_gbm, pred4 = pred_qrf)
pred <- exp(predict(mod_stack, pred_df))

submission <- data.frame(Id = test$Id, SalePrice = pred)
write.csv(submission, file = "stacked_lasso+eln+gbm+qrf2.csv", row.names = FALSE)
