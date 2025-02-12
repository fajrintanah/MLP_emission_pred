library(tidymodels)
library(finetune)   # For tune_race_anova()
library(doParallel) # Parallel processing
library(ggplot2)



load(file='E://Fajrin/Publikasi/Pak Heru B Pulunggono/0 Road to Prof/18 Predicting Macronutrient in peat using ML/Data_Private/modelling_mlp2_06022025.RData')


# 1. Data Splitting -----------------------------------------------------------
set.seed(123)
data_split_N <- initial_split(EL_Data_N2, prop = 0.7)
train_data_N <- training(data_split_N)
test_data_N <- testing(data_split_N)

# 2. Recipe Setup -------------------------------------------------------------
N_rec4 <- recipe(N ~ OP_Age + Thick + Season + D_Canal + D_OPT + Depth,
                 data = train_data_N) %>%
  step_string2factor(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

# 3. Lightweight Model Spec ---------------------------------------------------
mlp_spec_tune_N4 <- mlp(
  epochs = tune(),
  hidden_units = tune(),
  penalty = tune(),
  learn_rate = tune()
) %>%
  set_engine("brulee", validation = 0) %>%
  set_mode("regression")

# 4. Minimal Workflow ---------------------------------------------------------
mlp_wflow_tune_N4 <- workflow() %>%
  add_recipe(N_rec4) %>%
  add_model(mlp_spec_tune_N4)

# 5. Efficient Parallel Setup -------------------------------------------------
cl <- makePSOCKcluster(max(1, parallel::detectCores() - 2))
registerDoParallel(cl)

# 6. Randomized Grid Search ---------------------------------------------------
set.seed(123)
folds_N4 <- vfold_cv(train_data_N, v = 5, repeats = 3)

set.seed(123)
param_grid_N4 <- expand.grid(
  epochs = seq(500, 1500, length.out = 5),  
  hidden_units = seq(5, 20, length.out = 6),  
  penalty = seq(-4, -1, length.out = 4),  
  learn_rate = seq(-3, -1, length.out = 4)  
)


# 7. Fast Memory-Optimized Tuning with race_anova -----------------------------
grid_results_N4 <- tune_race_anova(
  mlp_wflow_tune_N4,
  resamples = folds_N4,
  grid = param_grid_N4,
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
autoplot(grid_results_N4) +
  theme_minimal()
