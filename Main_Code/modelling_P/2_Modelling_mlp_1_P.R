# Load required packages
library(tidymodels)
library(brulee)
library(yardstick)
library(doParallel)
library(rsample)

library(tidyverse) # plotting and manipulation
library(grid) # combining plots
library(gridExtra) # combining plots
library(ggpubr) # combining plots
library(patchwork) # combining plots
library(ggfortify) # nice extension for ggplot
library(mgcv) #fitting gam models
library(GGally) # displaying pairs panel
library(caret)
library(caTools) # split dataset
library(readxl)
library(randomForest)
library(e1071)
library(gbm)          # basic implementation
library(xgboost)      # a faster implementation of gbm
library(caret)        # an aggregator package for performing many machine learning models
library(pdp)          # model visualization
library(lime)         # model visualization
library(neuralnet)
library(rpart)     #rpart for computing decision tree models
library(rsample)     # data splitting 
library(dplyr)       # data wrangling
library(rpart.plot)  # plotting regression trees
library(ipred)       # bagging
library(broom)
library(ranger) 	#efficient RF
library(NeuralNetTools)
library(tidymodels)
library(earth) 		#MARS model
library(iml)		#most robust and efficient relative importance 
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
library(brulee )
library(dials)

# 1. Data Splitting -----------------------------------------------------------

load(file='E://Fajrin/Publikasi/Pak Heru B Pulunggono/0 Road to Prof/18 Predicting Macronutrient in peat using ML/Data_Private/modelling_mlp2_19022025_P.RData')

set.seed(123)
data_split_P <- initial_split(EL_Data_P2, prop = 0.7)
train_data_P <- training(data_split_P)
test_data_P <- testing(data_split_P)

# 2. Recipe Setup (Corrected) -------------------------------------------------
# First define the recipe without immediate prep()
P_rec1 <- recipe(P ~ OP_Age + Thick + Season + D_Canal + D_OPT + Depth,
                      data = train_data_P) %>%
  # Convert character variables to factors first
  step_string2factor(all_nominal_predictors()) %>%  # Critical fix
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

# 3. Prepare recipe using training data ---------------------------------------
P_rec_prepped1 <- prep(P_rec1, training = train_data_P)

# 4. Process datasets ---------------------------------------------------------
train_data_processed_P1 <- bake(P_rec_prepped1, new_data = train_data_P)
test_data_processed_P1 <- bake(P_rec_prepped1, new_data = test_data_P)

# 5. Verify processed data structure ------------------------------------------
glimpse(train_data_processed_P1)

# 6. Lightweight Model Spec ---------------------------------------------------
mlp_spec_tune_P1 <- mlp(
  epochs = tune(),
  hidden_units = tune(),
  penalty = tune(),
  learn_rate = tune()
) %>% 
  set_engine("brulee", validation = 0) %>%
  set_mode("regression")

# 7. Minimal Workflow --------------------------------------------------------
mlp_wflow_tune_P1 <- workflow() %>%
  add_recipe(P_rec1) %>%
  add_model(mlp_spec_tune_P1)

# 8. Efficient Parallel Setup -------------------------------------------------
cl <- makePSOCKcluster(max(1, parallel::detectCores() - 2))  # Safer core allocation
registerDoParallel(cl)

# 9. Randomized Grid Search ---------------------------------------------------
set.seed(123)
folds_P1 <- vfold_cv(train_data_P, v = 5)

set.seed(123)
param_grid_P1 <- grid_latin_hypercube(
  epochs(range = c(500, 1500)),
  hidden_units(range = c(5, 500)),
  penalty(range = c(-7, -0.1)),
  learn_rate(range = c(-7, -0.1)),
  size = 15  # 15 random combinations 
)

# 10. Memory-Optimized Tuning --------------------------------------------------
grid_results_P1 <- tune_grid(
  mlp_wflow_tune_P1,
  resamples = folds_P1,
  grid = param_grid_P1,
  metrics = metric_set(yardstick::rmse),
  control = control_grid(
    verbose = TRUE,
    parallel_over = "everything",
    allow_par = TRUE,
    extract = NULL,        # No model extracts
    save_pred = FALSE,     # No predictions storage
    save_workflow = FALSE, # No workflow copies
    pkgs = c("brulee")     # Minimal worker packages
  )
)

# 11. Cleanup & Results --------------------------------------------------------
stopCluster(cl)
registerDoSEQ()

# Show best combinations
show_best(grid_results_P1, n = 10, metric = "rmse")

save.image(file='E://Fajrin/Publikasi/Pak Heru B Pulunggono/0 Road to Prof/18 Predicting Macronutrient in peat using ML/Data_Private/modelling_mlp2_19022025_P1.RData')

autoplot(grid_results_P1)


#Best epoch = 900 - 1100
#Best hidden units = 400 - 450
#Best penalty = -3 - -1
#Best learn rate = -2 - -0.1

# Define a new parameter grid based on the best ranges for P1
param_grid_P1_1 <- grid_latin_hypercube(
  epochs(range = c(900, 1100)),
  hidden_units(range = c(400, 450)),
  penalty(range = c(-3, -1)),
  learn_rate(range = c(-2, -0.1)),
  size = 15  # 15 random combinations 
)

# Create a new workflow for the refined grid search for P1
mlp_wflow_tune_P1_1 <- workflow() %>%
  add_recipe(P_rec1) %>%
  add_model(mlp_spec_tune_P1)

# Efficient Parallel Setup for P1
cl <- makePSOCKcluster(max(1, parallel::detectCores() - 2))  # Safer core allocation
registerDoParallel(cl)

# Perform the grid search with the refined parameter grid for P1
set.seed(123)
grid_results_P1_1 <- tune_grid(
  mlp_wflow_tune_P1_1,
  resamples = folds_P1,
  grid = param_grid_P1_1,
  metrics = metric_set(yardstick::rmse),
  control = control_grid(
    verbose = TRUE,
    parallel_over = "everything",
    allow_par = TRUE,
    extract = NULL,        # No model extracts
    save_pred = FALSE,     # No predictions storage
    save_workflow = FALSE, # No workflow copies
    pkgs = c("brulee")     # Minimal worker packages
  )
)

# Cleanup & Results for P1
stopCluster(cl)
registerDoSEQ()

# Show best combinations for the refined grid search for P1
show_best(grid_results_P1_1, n = 10, metric = "rmse")

autoplot(grid_results_P1_1)

#Best epoch = 1000 - 1050
#Best hidden units = 420 - 435
#Best penalty = -2 - -1.75
#Best learn rate = -0.5 - -0.1

# Define a new parameter grid based on the refined best ranges for P1
param_grid_P1_2 <- grid_latin_hypercube(
  epochs(range = c(1000, 1050)),
  hidden_units(range = c(420, 435)),
  penalty(range = c(-2, -1.75)),
  learn_rate(range = c(-0.5, -0.1)),
  size = 15  # 15 random combinations 
)

# Create a new workflow for the refined grid search for P1
mlp_wflow_tune_P1_2 <- workflow() %>%
  add_recipe(P_rec1) %>%
  add_model(mlp_spec_tune_P1)

# Efficient Parallel Setup for P1
cl <- makePSOCKcluster(max(1, parallel::detectCores() - 2))  # Safer core allocation
registerDoParallel(cl)

# Perform the grid search with the refined parameter grid for P1
set.seed(123)
grid_results_P1_2 <- tune_grid(
  mlp_wflow_tune_P1_2,
  resamples = folds_P1,
  grid = param_grid_P1_2,
  metrics = metric_set(yardstick::rmse),
  control = control_grid(
    verbose = TRUE,
    parallel_over = "everything",
    allow_par = TRUE,
    extract = NULL,        # No model extracts
    save_pred = FALSE,     # No predictions storage
    save_workflow = FALSE, # No workflow copies
    pkgs = c("brulee")     # Minimal worker packages
  )
)

# Cleanup & Results for P1
stopCluster(cl)
registerDoSEQ()

# Show best combinations for the refined grid search for P1
show_best(grid_results_P1_2, n = 10, metric = "rmse")

autoplot(grid_results_P1_2)

#Best epoch = 1020 - 1030
#Best hidden units = 425 - 430
#Best penalty = -1.95 - -1.90
#Best learn rate = -0.4 - -0.3


#------- third try ----------------------------

# Define a new parameter grid based on the refined best ranges for P1_3
param_grid_P1_3 <- grid_latin_hypercube(
  epochs(range = c(1020, 1030)),
  hidden_units(range = c(425, 430)),
  penalty(range = c(-1.95, -1.90)),
  learn_rate(range = c(-0.4, -0.3)),
  size = 15  # 15 random combinations 
)

# Create a new workflow for the refined grid search for P1_3
mlp_wflow_tune_P1_3 <- workflow() %>%
  add_recipe(P_rec1) %>%
  add_model(mlp_spec_tune_P1)

# Efficient Parallel Setup for P1_3
cl <- makePSOCKcluster(max(1, parallel::detectCores() - 2))  # Safer core allocation
registerDoParallel(cl)

# Perform the grid search with the refined parameter grid for P1_3
set.seed(123)
grid_results_P1_3 <- tune_grid(
  mlp_wflow_tune_P1_3,
  resamples = folds_P1,
  grid = param_grid_P1_3,
  metrics = metric_set(yardstick::rmse),
  control = control_grid(
    verbose = TRUE,
    parallel_over = "everything",
    allow_par = TRUE,
    extract = NULL,        # No model extracts
    save_pred = FALSE,     # No predictions storage
    save_workflow = FALSE, # No workflow copies
    pkgs = c("brulee")     # Minimal worker packages
  )
)

# Cleanup & Results for P1_3
stopCluster(cl)
registerDoSEQ()

# Show best combinations for the refined grid search for P1_3
show_best(grid_results_P1_3, n = 10, metric = "rmse")

autoplot(grid_results_P1_3)

#Best epoch = 1028
#Best hidden units = 427
#Best penalty = -1.93
#Best learn rate = -0.35
#Mean RMSE = 604 std 59.6
