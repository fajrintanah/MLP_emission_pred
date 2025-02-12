library(tidymodels)
library(finetune)   # For tune_race_anova()
library(doParallel) # Parallel processing
library(ggplot2)

# 1. Data Splitting -----------------------------------------------------------
set.seed(123)
data_split_N <- initial_split(EL_Data_N2, prop = 0.7)
train_data_N <- training(data_split_N)
test_data_N <- testing(data_split_N)

# 2. Recipe Setup -------------------------------------------------------------
N_rec1 <- recipe(N ~ OP_Age + Thick + Season + D_Canal + D_OPT + Depth,
                 data = train_data_N) %>%
  step_string2factor(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

# 3. Lightweight Model Spec ---------------------------------------------------
mlp_spec_tune_N1 <- mlp(
  epochs = tune(),
  hidden_units = tune(),
  penalty = tune(),
  learn_rate = tune()
) %>%
  set_engine("brulee", validation = 0) %>%
  set_mode("regression")

# 4. Minimal Workflow ---------------------------------------------------------
mlp_wflow_tune_N1 <- workflow() %>%
  add_recipe(N_rec1) %>%
  add_model(mlp_spec_tune_N1)

# 5. Efficient Parallel Setup -------------------------------------------------
cl <- makePSOCKcluster(max(1, parallel::detectCores() - 2))
registerDoParallel(cl)

# 6. Randomized Grid Search ---------------------------------------------------
set.seed(123)
folds_N <- vfold_cv(train_data_N, v = 5, repeats = 10)

set.seed(123)
param_grid_N1 <- grid_random(
  epochs(range = c(500, 1500)),
  hidden_units(range = c(5, 20)),
  penalty(range = c(-4, -1)),
  learn_rate(range = c(-3, -1)),
  size = 50
)

# 7. Fast Memory-Optimized Tuning with race_anova -----------------------------
grid_results_N1 <- tune_race_anova(
  mlp_wflow_tune_N1,
  resamples = folds_N,
  grid = param_grid_N1,
  metrics = metric_set(yardstick::rmse, yardstick::mae),
  control = control_race(
    verbose_elim = TRUE,
    save_pred = TRUE,       # Required for autoplot()
    save_workflow = TRUE    # Enables extraction
  )
)

# 8. Cleanup & Results --------------------------------------------------------
stopCluster(cl)
registerDoSEQ()

# 9. Autoplot for Tuning Results ----------------------------------------------
autoplot(grid_results_N1) +
  theme_minimal()
