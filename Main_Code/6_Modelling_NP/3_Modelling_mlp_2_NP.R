
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


# load file fajrin
load(file='E://Fajrin/Publikasi/Pak Heru B Pulunggono/0 Road to Prof/18 Predicting Macronutrient in peat using ML/Data_Private/modelling_mlp2_19022025_K2.RData')


library(readxl)
R_macro <- read_excel("E:/Fajrin/Publikasi/Pak Heru B Pulunggono/0 Road to Prof/18 Predicting Macronutrient in peat using ML/Data/R_macro.xlsx", 
    col_types = c("text", "text", "skip", 
        "text", "numeric", "numeric", "numeric", 
        "numeric", "numeric", "numeric", 
        "numeric", "numeric", "numeric", 
        "numeric", "numeric", "numeric"))
str(R_macro)

# EDA/exploratory Data Analysis

# EDA/exploratory Data Analysis
remove_outliers_iqr <- function(data, col_range) {
  data %>%
    mutate(across(all_of(names(data)[col_range]), ~ {
      x <- .[. != 0]  # Exclude 0s before computing quantiles
      q1 <- quantile(x, 0.25, na.rm = TRUE)
      q3 <- quantile(x, 0.75, na.rm = TRUE)
      iqr <- q3 - q1

      ifelse(. == 0 | . > q3 + 1.5 * iqr | . < q1 - 1.5 * iqr, NA_real_, .)
    }))
}

# Apply the function to columns 7 to 15
R_macro_rev <- remove_outliers_iqr(R_macro, 7:15)

str(R_macro_rev)


# Select NP dataset -----------------------------------------------------------
str(R_macro_rev)

Data_NP <- R_macro_rev %>% 
                select(-c(11:15)) %>% 
                select(-c(7:9))

str(Data_NP)

EL_Data_NP2 <- Data_NP

# 1. Splitting data--------------------------------------------------------------
set.seed(123)
data_split_NP <- initial_split(EL_Data_NP2, prop = 0.7)
train_data_NP <- training(data_split_NP)
test_data_NP <- testing(data_split_NP)

# 2. Recipe Setup (Corrected) -------------------------------------------------
# First define the recipe without immediate prep()
NP_rec2 <- recipe(NP ~ OP_Age + Thick + Season + D_Canal + D_OPT + Depth,
                      data = train_data_NP) %>%
  # Convert character variables to factors first
  step_string2factor(all_nominal_predictors()) %>%  # Critical fix
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())%>%
  step_interact(terms = ~ D_Canal:D_OPT:Depth)  # Add interaction terms

# 3. Prepare recipe using training data ---------------------------------------
NP_rec_prepped2 <- prep(NP_rec2, training = train_data_NP)

# 4. Process datasets ---------------------------------------------------------
train_data_processed_NP2 <- bake(NP_rec_prepped2, new_data = train_data_NP)
test_data_processed_NP2 <- bake(NP_rec_prepped2, new_data = test_data_NP)

# 5. Verify processed data structure ------------------------------------------
glimpse(train_data_processed_NP2)

# 6. Lightweight Model Spec ---------------------------------------------------
mlp_spec_tune_NP2 <- mlp(
  epochs = tune(),
  hidden_units = tune(),
  penalty = tune(),
  learn_rate = tune()
) %>% 
  set_engine("brulee", validation = 0) %>%
  set_mode("regression")

# 7. Minimal Workflow --------------------------------------------------------
mlp_wflow_tune_NP2 <- workflow() %>%
  add_recipe(NP_rec2) %>%
  add_model(mlp_spec_tune_NP2)

# 8. Efficient Parallel Setup -------------------------------------------------
cl <- makePSOCKcluster(max(1, parallel::detectCores() - 2))  # Safer core allocation
registerDoParallel(cl)

# 9. Randomized Grid Search ---------------------------------------------------
set.seed(123)
folds_NP2 <- vfold_cv(train_data_NP, v = 5)

set.seed(123)
param_grid_NP2 <- grid_latin_hypercube(
  epochs(range = c(500, 1500)),
  hidden_units(range = c(5, 500)),
  penalty(range = c(-7, -0.1)),
  learn_rate(range = c(-7, -0.1)),
  size = 15  # 15 random combinations 
)

# 10. Memory-Optimized Tuning --------------------------------------------------
grid_results_NP2 <- tune_grid(
  mlp_wflow_tune_NP2,
  resamples = folds_NP2,
  grid = param_grid_NP2,
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
show_best(grid_results_NP2, n = 10, metric = "rmse")

save.image(file='E://Fajrin/Publikasi/Pak Heru B Pulunggono/0 Road to Prof/18 Predicting Macronutrient in peat using ML/Data_Private/modelling_mlp2_19022025_NP2.RData')

autoplot(grid_results_NP2)