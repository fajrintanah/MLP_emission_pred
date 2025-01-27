library(tidyverse) # plotting and manipulation
library(grid) # combining plots
library(gridExtra) # combining plots
library(ggpubr) # combining plots
library(patchwork) # combining plots
library(ggfortify) # nice extension for ggplot
library(mgcv) #fitting gam models
library(GGally) # displaying pairs panel
library(caret) # an aggregator package for performing many machine learning models
library(caTools) # split dataset
library(readxl) # read excel
library(randomForest) # RF
library(e1071) # SVM
library(gbm)     # basic implementation
library(xgboost) # a faster implementation of gbm
library(pdp)  # model visualization
library(lime)  # model visualization
library(neuralnet) # neural network
library(rpart)     #rpart for computing decision tree models
library(rsample)     # data splitting
library(dplyr)       # data wrangling
library(rpart.plot)  # plotting regression trees
library(ipred)       # bagging
library(broom) # tidy model
library(ranger) 	#efficient RF
library(NeuralNetTools) # neural network
library(tidymodels) # tidy model
library(earth) 		#MARS model
library(iml)	#most robust and efficient relative importance
library(xgboost)	#extreeme gradient boosting
library(ModelMetrics) #get model metrics
library(Metrics) 	#get ML model metrics
library(Cubist) #Cubist modell
library(iBreakDown)
library(DALEX)
library(viridis)
library(ICEbox)
library(hrbrthemes)
library(tidyverse)
library(hrbrthemes)
library(viridis)
library(vip)
library(fastDummies)
library(brulee)  # for neural network
library(httpgd) # for plotting
library(languageserver) # for code linting
library(renv) # for package management
library(lintr) # for code linting

# Perform an initial split (70% training, 30% testing)
set.seed(123) # For reproducibility
data_split_N <- initial_split(EL_Data_N2, prop = 0.7) 

# Extract training and testing datasets
train_data_N <- training(data_split_N)
test_data_N <- testing(data_split_N)

#set the recipes
  N_rec <-
  recipe(N ~ OP_Age + Thick + Season + D_Canal + D_OPT + Depth,
         data = train_data_N) %>% 
   step_dummy(all_nominal_predictors(), one_hot = TRUE) %>% # dummy coding
   step_zv(all_predictors()) %>% # remove zero variance predictors
   step_normalize(all_numeric_predictors()) %>%  # data normalization
   step_interact(~ all_predictors()^2) # add interaction

mlp_spec_N1 <- 
  mlp(epochs = 1000, hidden_units = 5, penalty = 0.01, learn_rate = 0.1) %>%
  set_engine("brulee", validation = 0) %>%
  set_mode("regression")

mlp_wflow1 <- N_rec %>%
  workflow(mlp_spec_N1)

set.seed(987)
mlp_model1 <- fit(mlp_wflow1, train_data_N)
mlp_model1 %>% extract_fit_engine()

mlp_model1 %>% 
  extract_fit_parsnip() %>% 
  predict(test_data_N) %>% 
  bind_cols(test_data_N) %>% 
  metrics(truth = N, estimate = .pred)

# Perform a 10-fold cross-validation
set.seed(123) 
mlp_vfold <- vfold_cv(train_data_N, v = 10)
