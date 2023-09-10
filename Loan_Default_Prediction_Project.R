################################################################################
### Title: "Prediction Project"
### Course: STA 235H
### Semester: Fall 2021
### Names: Nidhish Nerur, Tanay Sethia, Pavan Agrawal, Sanath Govindarajan
################################################################################
#source(file = "include.R")
# Clears memory
rm(list = ls())
# Clears console
cat("\014")

library(tidyverse)
library(modelsummary)
library(ggplot2)
library(estimatr)
library(caret)
library(rpart)
library(ranger)
library(rattle)
library(rsample)
library(parallel)
library(doParallel)
library(ipred)
library(modelr)
# Now clean your data and conduct your analysis with the d dataset.

#action_taken, outcome variable for classification task

library(rsample)
library(tidyverse)


# Let's use pure functions (i.e. no global variables,
# no shared state (besides function parameters), and no
# modifying function parameters) to keep the different parts modular
# and to simplify debugging (we don't have to re-run the
# entire script from the beginning each time).
# Whenever you need to use a random seed, it should be a parameter
# to your function, so that we can easily change the seed later, if necessary.

# select_data: select the given 1000 rows from the data set
select_data <- function() {
  print('selecting data ...')
  # Before doing anything, load your data and select the sample we will use
  d_total <- read.csv("/Users/nidhish/Documents/UTFall2021/STA235H/PredictionProject/county_48453.csv")
  
  # These are the row numbers you will need (everyone will use the same observations)
  rows <- read.csv("https://raw.githubusercontent.com/maibennett/sta235/main/exampleSite/content/Assignments/Project/data/row_sample.csv") %>%
    pull() # Load it as a vector and not a dataframe.
  
  d <- d_total %>% slice(rows)
  
  #Drop all ID variables (not predictors)
  d[, c('hoepa_status', 'lei', 'derived_msa.md', 'activity_year', 'state_code', 'county_code', 'census_tract', 'denial_reason.1', 'denial_reason.2', 'denial_reason.3', 'denial_reason.4')] <- list(NULL)
  
  #Converts numeric variables that should be factors
  d <- d %>% mutate(across(c(action_taken, purchaser_type, preapproval, loan_type, loan_purpose, lien_status, 
                        reverse_mortgage, open.end_line_of_credit, business_or_commercial_purpose,
                        negative_amortization, interest_only_payment, balloon_payment,
                        other_nonamortizing_features, construction_method, occupancy_type,
                        manufactured_home_secured_property_type, manufactured_home_land_property_interest,
                        applicant_credit_score_type, co.applicant_credit_score_type,
                        applicant_ethnicity.1, applicant_ethnicity.2, applicant_ethnicity.3,
                        applicant_ethnicity.4, applicant_ethnicity.5, co.applicant_ethnicity.1,
                        co.applicant_ethnicity.2, co.applicant_ethnicity.3, co.applicant_ethnicity.4,
                        co.applicant_ethnicity.5, applicant_ethnicity_observed, co.applicant_ethnicity_observed,
                        applicant_race.1, applicant_race.2, applicant_race.3, applicant_race.4,
                        applicant_race.5, co.applicant_race.1, co.applicant_race.2, co.applicant_race.3,
                        co.applicant_race.4, co.applicant_race.5, applicant_race_observed, 
                        co.applicant_race_observed, applicant_sex, co.applicant_sex,
                        applicant_sex_observed, co.applicant_sex_observed, submission_of_application,
                        initially_payable_to_institution, aus.1, aus.2, aus.3, aus.4, aus.5),
                      ~as.factor(.x)))
  
  return(d)
}

# filter_mostly_na_cols: remove columns with too many N/A values, as these are not predictive
# and unnecessarily restrict the number of rows (we want our final data set to
# have no NA values)
filter_mostly_na_cols <- function(df, max_na_vals) {
  print('preprocessing ...')
  # exclude columns where > threshold values are NA
  res <- data.frame(df)
  for (col in colnames(res)) {
    na_vals <- sum(is.na(res[col]))
    if (na_vals > max_na_vals) {
      cat('excluding column "', col, '" (', na_vals, ' NA values)\n', sep='')
      res <- res %>% select(-col)
    }
  }
  return(res)
}

# filter_exempt: get rid of "exempt" rows and change
# numeric variables with "exempt" from character type
# to numeric type
filter_exempt <- function(df) {
  res <- data.frame(df)
  for (col in colnames(res)) {
    if (col != "debt_to_income_ratio" && class(res[[col]]) == 'character' && any(res[col] == "Exempt")) {
      res <- res %>% mutate((!!as.name(col)) := ifelse(!!as.name(col) == "Exempt", NA, !!as.name(col)))
      res[col] <- as.numeric(res[[col]])
    }
    else if (col == "debt_to_income_ratio") {
      #res <- res %>% mutate(debt_to_income_ratio = ifelse(debt_to_income_ratio == "Exempt", NA, stri_extract_first_regex(debt_to_income_ratio, "[0-9]+")))
      res[col] <- as.factor(res[[col]])
    }
    
  }
  return(res)
}


# transform categorical variables from character to factor
chr_to_factor <- function(df) {
  res <- data.frame(df)
  for (col in colnames(res)) {
    if (class(res[[col]]) == 'character')
      res[col] <- factor(res[[col]])
  }
  return(res)
}

# split preprocessed data frame into training and testing data
train_test_split <- function(df, output_col, train_percentage, random_seed) {
  set.seed(100)
  return(df %>% initial_split(prop = train_percentage / 100, strata = output_col))
}

# get data
travis_county_hmda <- select_data()

# preprocess
set.seed(100)
no_mostly_na_cols <- travis_county_hmda %>% filter_mostly_na_cols(80)
clean_data <- no_mostly_na_cols %>% drop_na %>% filter_exempt() %>% drop_na()
preprocessed <- clean_data %>% chr_to_factor()
library(DMwR2)
#library(smotefamily)
#preprocessed1 <- preprocessed %>% mutate(loan_amount = (loan_amount - mean(loan_amount))/sd(loan_amount),
                     #   tract_population = (tract_population - mean(tract_population))/sd(tract_population),
                     #   ffiec_msa_md_median_family_income = (ffiec_msa_md_median_family_income - mean(ffiec_msa_md_median_family_income))/sd(ffiec_msa_md_median_family_income),
                     #   tract_to_msa_income_percentage = (tract_to_msa_income_percentage - mean(tract_to_msa_income_percentage))/sd(tract_to_msa_income_percentage),
                     #   tract_owner_occupied_units = (tract_owner_occupied_units - mean(tract_owner_occupied_units))/sd(tract_owner_occupied_units),
                     #   tract_one_to_four_family_homes = (tract_one_to_four_family_homes - mean(tract_one_to_four_family_homes))/sd(tract_one_to_four_family_homes),
                     #   tract_median_age_of_housing_units = (tract_median_age_of_housing_units - mean(tract_median_age_of_housing_units))/sd(tract_median_age_of_housing_units))

# train/test split
cls_train_test <- preprocessed %>% train_test_split('action_taken', 80, 100)
cls_train <- training(cls_train_test)
cls_test <- testing(cls_train_test)

reg_train_test <- preprocessed %>% train_test_split('loan_amount', 80, 100)
reg_train <- training(reg_train_test)
reg_test <- testing(reg_train_test)
xtabs(~ action_taken, data = clean_data)

#library(performanceEstimation)
#balanced_data <- smote(factor(action_taken) ~., cls_train, perc.over = 2, k = 5, perc.under = 2)
#xtabs(~action_taken, data = balanced_data)
# train models

set.seed(100)
matrixP <- model.matrix(action_taken ~ ., data = cls_train)
head(matrixP)
num_predP <- ncol(matrixP) - 1

xtabs(~action_taken, data = cls_train)
#185 predictors
num_predP

tuneGridP <- expand.grid(mtry = seq(1, num_predP, 1),
                        splitrule = "gini",
                        min.node.size = 5)

rfcvP <- train(factor(action_taken) ~ ., data = cls_train,
               method = "ranger",
               trControl = trainControl("cv", 10),
               importance = "permutation",
               tuneGrid = tuneGridP)

plot(rfcvP)
(varImp(rfcvP))

rfcvP$bestTune

predP2 <- rfcvP %>% predict(cls_test)
cls_test <- cls_test %>% mutate(predictionNew = predP2)
#94.32% accuracy - Preferred Model
cls_train$prop
mean(factor(cls_test$action_taken) == cls_test$predictionNew)
confusionMatrix(cls_test$predictionNew, cls_test$action_taken)

#XG Boosting
set.seed(100)
xgbmP <- train(factor(action_taken) ~ ., data = cls_train,
               method = 'xgbTree',
               trControl = trainControl("cv", number = 10, allowParallel = TRUE))

xgbmP$bestTune
xgbmP$finalModel

#94.32% accuracy - same Random Forest
predPxg <- xgbmP %>% predict(cls_test)
cls_test <- cls_test %>% mutate(prediction2 = predPxg)
mean(factor(cls_test$action_taken) == cls_test$prediction2)

#Adaptive Boosting
set.seed(100)
adaP <- train(factor(action_taken) ~ ., data = cls_train,
              method = "ada",
              trControl = trainControl("cv", number = 10),
              tuneLength = 10)
adaP$bestTune
adaP$finalModel
#Error
predadaP <- adaP %>% predict(cls_test)

#Gradient Boosting
set.seed(100)
gbmP <- train(factor(action_taken) ~ ., data = cls_train,
             method = "gbm",                                 
             trControl = trainControl("cv", number = 10),
             tuneLength = 10)
gbmP$bestTune
gbmP$finalModel
predGbm <- gbmP %>% predict(cls_test)
cls_test <- cls_test %>% mutate(prediction3 = predGbm)
#93.75% accuracy 
mean(factor(cls_test$action_taken) == cls_test$prediction3)

#SVM Model
set.seed(1000)
model_svm <- train(factor(action_taken) ~ ., data = cls_train,
                          method = "svmRadial",
                          tuneLength = 15, 
                          trControl = trainControl("cv", number = 10))
model_svm$bestTune
model_svm$finalModel
predSVM <- model_svm %>% predict(cls_test)
cls_test <- cls_test %>% mutate(prediction4 = predSVM)
#84.08%
mean(factor(cls_test$action_taken) == cls_test$prediction4)

#KNN Model 

set.seed(100)
knncP <- train(
  factor(action_taken) ~., data = cls_train, 
  method = "knn", 
  trControl = trainControl("cv", number = 10), 
  preProcess = c("center","scale"), 
  #tuneLength = 30
  tuneGrid = expand.grid(k = seq(1, 100, 2))
)
plot(knncP)
#7
knncP$bestTune

pred.P <- knncP %>% predict(cls_test)
cls_test <- cls_test %>% mutate(predictionKNN = pred.P)
#88.64% accuracy
mean(cls_test$action_taken == cls_test$predictionKNN)
confusionMatrix(cls_test$predictionKNN, cls_test$action_taken)


#Ridge Regression Model
ridge_action <- train(factor(action_taken) ~ ., data = cls_train,
                      method = "glmnet",
                      preProcess = "scale",
                      trControl = trainControl("cv", number = 10),
                      tuneGrid = expand.grid(alpha = 0, 
                                             lambda = seq(0, 5000, by = 0.1)))
plot(ridge_action)
ridge_action$bestTune$lambda
coef(ridge_action$finalModel, ridge_action$bestTune$lambda)

predRidge <- ridge_action %>% predict(cls_test)
mean(cls_test$action_taken == pred.P)

#Stacking Ensemble
install.library(libh2o)
?h2o.gbm
my_gbm <- h2o.gbm(x = x,
                  y = y,
                  training_frame = train_df_h2o,
                  nfolds = nfolds,
                  keep_cross_validation_predictions = TRUE,
                  seed = 5)

# test models

# compare models

#Distributions of important variables
clean_data_graph = clean_data %>% filter(action_taken == 1)
ggplot(data = clean_data_graph, aes(x = debt_to_income_ratio)) +
  geom_bar()
xtabs(~ debt_to_income_ratio, data = clean_data_graph)
summary(clean_data_graph)

clean_data_graph2 = clean_data %>% filter(action_taken == 3)
ggplot(data = clean_data_graph2, aes(x = debt_to_income_ratio)) +
  geom_bar()
summary(clean_data_graph2)

ggplot(data = clean_data_graph, aes(x = initially_payable_to_institution)) +
  geom_bar()
ggplot(data = clean_data_graph2, aes(x = initially_payable_to_institution)) +
  geom_bar()

ggplot(data = clean_data, aes(x = income, y = loan_amount)) +
  geom_point()

ggplot(data = clean_data, aes(x = property_value, y = loan_amount)) +
  geom_point()

clean_data$de



## Preferred model continuous outcome:
model1 <- train(loan_purpose ~ interest_only_payment, data = d,
                method = "lm")


## Preferred model binary outcome:
model2 <- train(business_or_commercial_purpose ~ interest_only_payment, data = d,
                method = "lm")

# Save your preferred model for binary outcome and continuous outcome
save(lm1, lm2, file = "STA235H_SectionX_GroupY.Rdata") 
# You will need to submit this .Rdata file (if you used setwd(), it will be saved
# in your working directory)

