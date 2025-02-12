library(tidymodels)
library(finetune)   # For tune_lhs()
library(doParallel) # Parallel processing
library(ggplot2)

load(file='E://Fajrin/Publikasi/Pak Heru B Pulunggono/0 Road to Prof/18 Predicting Macronutrient in peat using ML/Data_Private/modelling_mlp2_06022025.RData')


# 1. Data Splitting -----------------------------------------------------------
set.seed(123)
data_split_N <- initial_split(EL_Data_N2, prop = 0.7)
train_data_N <- training(data_split_N)
test_data_N <- testing(data_split_N)


# 2. Recipe Setup -------------------------------------------------------------
N_rec5 <- recipe(N ~ OP_Age + Thick + Season + D_Canal + D_OPT + Depth,
                 data = train_data_N) %>%
  step_string2factor(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

# 3. Lightweight Model Spec ---------------------------------------------------
mlp_spec_tune_N5 <- mlp(
  epochs = tune(),
  hidden_units = tune(),
  penalty = tune(),
  learn_rate = tune(),
  activation = tune()
) %>%
  set_engine("brulee", validation = 0) %>%
  set_mode("regression")

# 4. Minimal Workflow ---------------------------------------------------------
mlp_wflow_tune_N5 <- workflow() %>%
  add_recipe(N_rec5) %>%
  add_model(mlp_spec_tune_N5)

# 5. Efficient Parallel Setup -------------------------------------------------
cl <- makePSOCKcluster(max(1, parallel::detectCores() - 2))
registerDoParallel(cl)

# 6. Latin Hypercube Sampling -------------------------------------------------
set.seed(123)
folds_N5 <- vfold_cv(train_data_N5, v = 5, repeats = 3)

# Define hyperparameter ranges for Latin Hypercube Sampling
param_ranges_N5 <- list(
  epochs = c(500, 1000),  
  activation = c("relu", "elu"),
  hidden_units = c(5, 15),  
  penalty = c(0.01, 0.1),  
  learn_rate = c(1e-4, 1e-2)   
)

# 7. Latin Hypercube Sampling with tune_lhs() -------------------------------
lhs_results_N5 <- tune_lhs(
  mlp_wflow_tune_N5,
  resamples = folds_N5,
  param_info = param_ranges_N5,
  metrics = metric_set(yardstick::rmse, yardstick::mae),
  control = control_lhs(
    verbose = TRUE,
    save_pred = TRUE,  # Required for autoplot()
    save_workflow = TRUE  # Optional, allows model extraction
  )
)

# 8. Cleanup & Results --------------------------------------------------------
stopCluster(cl)
registerDoSEQ()

# 9. Autoplot for Tuning Results ----------------------------------------------
autoplot(lhs_results_N5) +
  theme_minimal()

