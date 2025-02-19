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

load(file='E://Fajrin/Publikasi/Pak Heru B Pulunggono/0 Road to Prof/18 Predicting Macronutrient in peat using ML/Data_Private/modelling_mlp2_19022025_K.RData')

set.seed(123)
data_split_K <- initial_split(EL_Data_K2, prop = 0.7)
train_data_K <- training(data_split_K)
test_data_K <- testing(data_split_K)

# 2. Recipe Setup (Corrected) -------------------------------------------------
# First define the recipe without immediate prep()
K_rec1 <- recipe(K ~ OP_Age + Thick + Season + D_Canal + D_OPT + Depth,
                      data = train_data_K) %>%
  # Convert character variables to factors first
  step_string2factor(all_nominal_predictors()) %>%  # Critical fix
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

# 3. Prepare recipe using training data ---------------------------------------
K_rec_prepped1 <- prep(K_rec1, training = train_data_K)

# 4. Process datasets ---------------------------------------------------------
train_data_processed_K1 <- bake(K_rec_prepped1, new_data = train_data_K)
test_data_processed_K1 <- bake(K_rec_prepped1, new_data = test_data_K)

# 5. Verify processed data structure ------------------------------------------
glimpse(train_data_processed_K1)

# 6. Lightweight Model Spec ---------------------------------------------------
mlp_spec_tune_K1 <- mlp(
  epochs = tune(),
  hidden_units = tune(),
  penalty = tune(),
  learn_rate = tune()
) %>% 
  set_engine("brulee", validation = 0) %>%
  set_mode("regression")

# 7. Minimal Workflow --------------------------------------------------------
mlp_wflow_tune_K1 <- workflow() %>%
  add_recipe(K_rec1) %>%
  add_model(mlp_spec_tune_K1)

# 8. Efficient Parallel Setup -------------------------------------------------
cl <- makePSOCKcluster(max(1, parallel::detectCores() - 2))  # Safer core allocation
registerDoParallel(cl)

# 9. Randomized Grid Search ---------------------------------------------------
set.seed(123)
folds_K1 <- vfold_cv(train_data_K, v = 5)

set.seed(123)
param_grid_K1 <- grid_latin_hypercube(
  epochs(range = c(500, 1500)),
  hidden_units(range = c(5, 500)),
  penalty(range = c(-7, -0.1)),
  learn_rate(range = c(-7, -0.1)),
  size = 15  # 15 random combinations 
)

# 10. Memory-Optimized Tuning --------------------------------------------------
grid_results_K1 <- tune_grid(
  mlp_wflow_tune_K1,
  resamples = folds_K1,
  grid = param_grid_K1,
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
show_best(grid_results_K1, n = 10, metric = "rmse")