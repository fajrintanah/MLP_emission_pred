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
folds_N5 <- vfold_cv(train_data_N, v = 5, repeats = 3)
library(finetune)

# Specify the parameter ranges for tuning
param_ranges_N5 <- parameters(
  epochs(range = c(500, 2000)),  # Expanded range
  hidden_units(range = c(5, 50)),  # More flexibility
  activation(values = c("relu", "elu")),  # Specify categorical values explicitly
  penalty(range = c(0.001, 1)),  # Wider range
  learn_rate(range = c(1e-4, 1e-1))  # Expanded range
)


# Use tune_bayes for hyperparameter tuning
set.seed(123)
grid_results_N5 <- tune_grid(
  mlp_wflow_tune_N5,
  resamples = folds_N5,
  grid = 20,  # Number of grid points (increase for better coverage)
  metrics = metric_set(yardstick::rmse),
  control = control_grid(verbose = TRUE)
  )


# Visualize results
autoplot(grid_results_N5) + theme_minimal()


# 8. Cleanup & Results --------------------------------------------------------
stopCluster(cl)
registerDoSEQ()

# 9. Autoplot for Tuning Results ----------------------------------------------
autoplot(lhs_results_N5) +
  theme_minimal()

