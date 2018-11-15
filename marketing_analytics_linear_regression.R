# This is assignment 1 of the Marketing Analytics class. The goal is to practice linear regression.
# The assignment itself is really easy, so I took the opportunity to practice writing tidy code,
# plotting with ggplot2 and some dyplyr.


pacman::p_load("tidyverse", "GGally", "gridExtra", "caret", "MASS", "coefplot")

setwd("desktop/MMA/M3 - Marketing Analytics/Data")
adv <- read.csv("adv_sales.csv")

### Preliminary Data Inspection --------------------------------------------------------------------
# Examine the structure of the data
glimpse(adv)

# Check descriptive statistics
summary(adv)

# Check for variable distribution & bivariate correlations
ggpairs(adv)

### Explore Bi-variate Relationships ---------------------------------------------------------------
# sales vs in-store spend
store <- adv %>% 
    ggplot(aes(y = sales, x = store)) +
    geom_point(col = "deeppink3", alpha = 0.6) +
    geom_smooth(method = "lm", col = "grey15", alpha = 0.4, size = 1) +
    labs(x = "Advertisement spend in store",
         y = "Sales",
         title = "Sales vs In-Store Spend")
store

# sales vs billboard spend
billboard <- adv %>% 
    ggplot(aes(y = sales, x = billboard)) +
    geom_point(col = "dodgerblue4", alpha = 0.6) +
    geom_smooth(method = "lm", col = "grey15", alpha = 0.4, size = 1) +
    labs(x = "Advertisement spend on billboards",
         y = "Sales",
         title = "Sales vs Billboard Spend")
billboard

# sales vs print spend
print <- adv %>% 
    ggplot(aes(y = sales, x = printout)) +
    geom_point(col = "brown3", alpha = 0.6) +
    geom_smooth(method = "lm", col = "grey15", alpha = 0.4, size = 1) +
    labs(x = "Advertisement spend on prints",
         y = "Sales",
         title = "Sales vs Print Spend")
print

# sales vs satisfaction level
sat <- adv %>% 
    ggplot(aes(y = sales, x = sat)) +
    geom_point(col = "steelblue4", alpha = 0.6) +
    geom_smooth(method = "lm", col = "grey15", alpha = 0.4, size = 1) +
    labs(x = "Satisfaction",
         y = "Sales",
         title = "Sales vs Satisfaction Level")
sat

# sales vs competitor spend
comp <- adv %>% 
    ggplot(aes(y = sales, x = comp)) +
    geom_point(col = "darkorange3", alpha = 0.6) +
    geom_smooth(method = "lm", col = "grey15", alpha = 0.4, size = 1) +
    labs(x = "Comeptitor Spend",
         y = "Sales",
         title = "Sales vs Comeptitor Spend")
comp

# sales vs price
price <- adv %>% 
    ggplot(aes(y = sales, x = price)) +
    geom_point(col = "turquoise4", alpha = 0.6) +
    geom_smooth(method = "lm", col = "grey15", alpha = 0.4, size = 1) +
    labs(x = "Price",
         y = "Sales",
         title = "Sales vs Price")
price

grid.arrange(store, billboard, print, sat, comp, price)
rm(store, billboard, print, sat, comp, price)

### Model Construction -----------------------------------------------------------------------------
# Create train & test set
set.seed(234)
inTrain <- createDataPartition(adv$sales, p = 0.75, list = FALSE)
train <- adv[inTrain, ]
test <- adv[-inTrain, ]
rm(inTrain)

## Model 1: sales vs price ----
par(mfrow = c(2, 2))

mod1 <- lm(sales ~ price, data = train)
summary(mod1)
plot(mod1)

actual <- test$sales
predict <- predict(mod1, test)
train_rsq1 <- summary(mod1)$r.squared
test_rsq1 <- 1 - (sum((actual - predict)^2) / sum((actual - mean(actual))^2))

# Create a data frame to store results for comparison
results <- data.frame(model = 1,
                      variables = c("price"),
                      train_rsq = train_rsq1,
                      test_rsq = test_rsq1)

## Model 2: sales vs price & store ----
mod2 <- lm(sales ~ price + store, data = train)
summary(mod2)
plot(mod2)

predict <- predict(mod2, test)
train_rsq2 <- summary(mod2)$r.squared
test_rsq2 <- 1 - (sum((actual - predict)^2) / sum((actual - mean(actual))^2))

# Append model 2 result to dataframe
results <- rbind(results,
                 data.frame(
                     model = 2,
                     variables = c("price, store"),
                     train_rsq = train_rsq2,
                     test_rsq = test_rsq2
                 ))

## Model 3: sales vs price, store & billboard ----
mod3 <- lm(sales ~ price + store + billboard, data = train)
summary(mod3)
plot(mod3)

predict <- predict(mod3, test)
train_rsq3 <- summary(mod3)$r.squared
test_rsq3 <- 1 - (sum((actual - predict)^2) / sum((actual - mean(actual))^2))

results <- rbind(results,
                 data.frame(
                     model = 3,
                     variables = c("price, store, billboard"),
                     train_rsq = train_rsq3,
                     test_rsq = test_rsq3
                 ))

## Model 4: sales vs price, store, billboard & printout ----
mod4 <- lm(sales ~ price + store + billboard + printout, data = train)
summary(mod4)
plot(mod4)

predict <- predict(mod4, test)
train_rsq4 <- summary(mod4)$r.squared
test_rsq4 <- 1 - (sum((actual - predict)^2) / sum((actual - mean(actual))^2))

results <- rbind(results,
                 data.frame(
                     model = 4,
                     variables = c("price, store, billboard, print"),
                     train_rsq = train_rsq4,
                     test_rsq = test_rsq4
                 ))

## Model 5: sales vs price, store, billboard, printout & satisfaction ----
mod5 <- lm(sales ~ price + store + billboard + printout + sat, data = train)
summary(mod5)
plot(mod5)

predict <- predict(mod5, test)
train_rsq5 <- summary(mod5)$r.squared
test_rsq5 <- 1 - (sum((actual - predict)^2) / sum((actual - mean(actual))^2))

results <- rbind(results,
                 data.frame(
                     model = 5,
                     variables = c("price, store, billboard, print, satisfaction"),
                     train_rsq = train_rsq5,
                     test_rsq = test_rsq5
                 ))

## Model 6: sales vs price, store, billboard, printout, satisfaction & competitor spend ----
mod6 <- lm(sales ~ price + store + billboard + printout + sat + comp, data = train)
summary(mod6)
plot(mod6)

predict <- predict(mod6, test)
train_rsq6 <- summary(mod6)$r.squared
test_rsq6 <- 1 - (sum((actual - predict)^2) / sum((actual - mean(actual))^2))

results <- rbind(results,
                 data.frame(
                     model = 6,
                     variables = c("price, store, billboard, print, satisfaction, competitor"),
                     train_rsq = train_rsq6,
                     test_rsq = test_rsq6
                 ))

## Model 7: sales vs all variables & store, billboard, printout interactions ----
mod7 <- lm(sales ~ . - X + store*billboard*printout, data = train)
summary(mod7)
plot(mod7)

predict <- predict(mod7, test)
train_rsq7 <- summary(mod7)$r.squared
test_rsq7 <- 1 - (sum((actual - predict)^2) / sum((actual - mean(actual))^2))

results <- rbind(results,
                 data.frame(
                     model = 7,
                     variables = c("all variables + store, billboard, print interactions"),
                     train_rsq = train_rsq7,
                     test_rsq = test_rsq7
                 ))

## Model 8: sales vs all variables except for printout, & store, billboard interaction terms ----
mod8 <- lm(sales ~ . - X - printout + store*billboard, data = train)
summary(mod8)
plot(mod8)

predict <- predict(mod8, test)
train_rsq8 <- summary(mod8)$r.squared
test_rsq8 <- 1 - (sum((actual - predict)^2) / sum((actual - mean(actual))^2))

results <- rbind(results,
                 data.frame(
                     model = 8,
                     variables = c("all variables - printout + store, billboard interactions"),
                     train_rsq = train_rsq8,
                     test_rsq = test_rsq8
                 ))

## Model 9: Stepwise with all variables and advertising interaction terms ----
mod9_lm <- lm(sales ~ . - X + store*billboard*printout, data = train)
mod9 <- stepAIC(mod9_lm, direction = "both")
summary(mod9)
plot(mod9)

predict <- predict(mod9, test)
train_rsq9 <- summary(mod9)$r.squared
test_rsq9 <- 1 - (sum((actual - predict)^2) / sum((actual - mean(actual))^2))

results <- rbind(results,
                 data.frame(
                     model = 9,
                     variables = c("stepwise w all variables + store, billboard, print interactions"),
                     train_rsq = train_rsq9,
                     test_rsq = test_rsq9
                 ))

### Model Comparison -------------------------------------------------------------------------------
# Compare R-squared
results

ggplot(results, aes(x = model)) +
    geom_point(aes(y = train_rsq), col = "blue", alpha = 0.9) +
    geom_line(aes(y = train_rsq), col = "blue", alpha = 0.7) +
    geom_point(aes(y = test_rsq), col = "brown3", alpha = 0.9) +
    geom_line(aes(y = test_rsq), col = "brown3", alpha = 0.7)

# Select the best model based on test rsqure
results %>% 
    arrange(desc(test_rsq)) %>% 
    top_n(1, test_rsq)

# Evaluate coefficients importance
summary(mod6)

train_std <- data.frame(scale(train[, -1]))
mod6_std <- lm(sales ~ ., data = train_std)
summary(mod6_std)

coefplot(mod6_std, intercept=FALSE, outerCI=1.96, lwdOuter=1.5,
         ylab="Rating of Feature",
         xlab="Association with Overall Satisfaction")

