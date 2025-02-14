# Load required packages
library(tidymodels)
library(brulee)
library(yardstick)
library(doParallel)

# 1. Data Splitting -----------------------------------------------------------
set.seed(123)
data_split_N <- initial_split(EL_Data_N2, prop = 0.7)
train_data_N <- training(data_split_N)
test_data_N <- testing(data_split_N)

# 2. Recipe Setup (Corrected) -------------------------------------------------
N_rec2 <- recipe(N ~ OP_Age + Thick + Season + D_Canal + D_OPT + Depth,
                data = train_data_N) %>%
  # Convert character variables to factors first
  step_string2factor(all_nominal_predictors()) %>%  # Critical fix
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_interact(terms = ~ D_Canal:D_OPT:Depth)  # Add interaction terms

# 3. Prepare recipe using training data ---------------------------------------
N_rec_prepped2 <- prep(N_rec2, training = train_data_N)

# 4. Process datasets ---------------------------------------------------------
train_data_processed_N2 <- bake(N_rec_prepped2, new_data = train_data_N)
test_data_processed2_N2 <- bake(N_rec_prepped2, new_data = test_data_N)

# 5. Verify processed data structure ------------------------------------------
glimpse(train_data_processed_N2)

# 6. Lightweight Model Spec ---------------------------------------------------
mlp_spec_tune_N2 <- mlp(
  epochs = tune(),
  hidden_units = tune(),
  penalty = tune(),
  learn_rate = tune()
) %>% 
  set_engine("brulee", validation = 0) %>%
  set_mode("regression")

# 7. Minimal Workflow --------------------------------------------------------
mlp_wflow_tune_N2 <- workflow() %>%
  add_recipe(N_rec2) %>%
  add_model(mlp_spec_tune_N2)

# 8. Efficient Parallel Setup -------------------------------------------------
cl <- makePSOCKcluster(max(1, parallel::detectCores() - 2))  # Safer core allocation
registerDoParallel(cl)

# 9. Randomized Grid Search ---------------------------------------------------
set.seed(123)
folds_N <- vfold_cv(train_data_N, v = 5)

set.seed(456)
param_grid_N2 <- grid_random(
  epochs(range = c(500, 1500)),
  hidden_units(range = c(5, 20)),
  penalty(range = c(-4, -1)),
  learn_rate(range = c(-3, -1)),
  size = 50  # 50 random combinations
)

# 10. Memory-Optimized Tuning --------------------------------------------------
grid_results_N2 <- tune_grid(
  mlp_wflow_tune_N2,
  resamples = folds_N,
  grid = param_grid_N2,
  metrics = metric_set(yardstick::rmse, yardstick::mae),
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
show_best(grid_results_N2, n = 10, metric = "rmse")

# 12. Visualize Results -------------------------------------------------------
autoplot(grid_results_N2) & coord_cartesian(ylim = c(3000, 4000))
# I found scattered results, with the rmse is likely to decrease as the numbers increasing
# so maybe it needs more tuning by increasing the value of each hyperparameter
# maybe the range must be more stretched


######### ------- Second try ------- #########

# 8. Efficient Parallel Setup -------------------------------------------------
cl <- makePSOCKcluster(max(1, parallel::detectCores() - 2))  # Safer core allocation
registerDoParallel(cl)

# 9. Randomized Grid Search ---------------------------------------------------
set.seed(123)
folds_N2_1 <- vfold_cv(train_data_N, v = 5)

set.seed(456)
param_grid_N2_1 <- grid_random(
  epochs(range = c(1500, 10000)),
  hidden_units(range = c(5, 50)),
  penalty(range = c(-3, -1)),
  learn_rate(range = c(-3,-0.3)),
  size = 25  # maybe lower the random combinations
)

# 10. Memory-Optimized Tuning --------------------------------------------------
grid_results_N2_1 <- tune_grid(
  mlp_wflow_tune_N2,
  resamples = folds_N2_1,
  grid = param_grid_N2_1,
  metrics = metric_set(yardstick::rmse, yardstick::mae),
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
show_best(grid_results_N2_1, n = 10, metric = "rmse")

# maybe we should try lower epoch
# increasing hidden units are likely to produce lower rmse
# penalty at -2.5 to -1.5, likely to produce lower rmse
# learning rate close to 0 is likely to produce lower rmse

######### ------- third try ------- #########

# 8. Efficient Parallel Setup -------------------------------------------------
cl <- makePSOCKcluster(max(1, parallel::detectCores() - 2))  # Safer core allocation
registerDoParallel(cl)

# 9. Randomized Grid Search ---------------------------------------------------
set.seed(123)
folds_N2_2 <- vfold_cv(train_data_N, v = 5)

set.seed(456)
param_grid_N2_2 <- grid_random(
  epochs(range = c(500, 2000)),
  hidden_units(range = c(25, 250)),
  penalty(range = c(-2.5, -1.5)),
  learn_rate(range = c(-1,-0.1)),
  size = 25  # maybe lower the random combinations
)

# 10. Memory-Optimized Tuning --------------------------------------------------
grid_results_N2_2 <- tune_grid(
  mlp_wflow_tune_N2,
  resamples = folds_N2_2,
  grid = param_grid_N2_2,
  metrics = metric_set(yardstick::rmse, yardstick::mae),
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
show_best(grid_results_N2_2, n = 10, metric = "rmse")

autoplot(grid_results_N2_2)+coord_cartesian(ylim = c(3000, 3500))

# higher epoch (1500-2000) seems to yield lower rmse
# higher hidden units seems to yield lower rmse
# penalty at -1.75 to -1.5 seems to yield lower rmse
# learning rate at -0.5 to -0.1 or higher seems to yield lower rmse

######### ------- fourth try ------- #########

# 8. Efficient Parallel Setup -------------------------------------------------
cl <- makePSOCKcluster(max(1, parallel::detectCores() - 2))  # Safer core allocation
registerDoParallel(cl)

# 9. Randomized Grid Search ---------------------------------------------------
set.seed(123)
folds_N2_3 <- vfold_cv(train_data_N, v = 5)

set.seed(456)
param_grid_N2_3 <- grid_random(
  epochs(range = c(1500, 2000)),
  hidden_units(range = c(100, 500)),
  penalty(range = c(-1.75, -1.5)),
  learn_rate(range = c(-0.5,-0.05)),
  size = 25  # maybe lower the random combinations
)

# 10. Memory-Optimized Tuning --------------------------------------------------
grid_results_N2_3 <- tune_grid(
  mlp_wflow_tune_N2,
  resamples = folds_N2_3,
  grid = param_grid_N2_3,
  metrics = metric_set(yardstick::rmse, yardstick::mae),
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
show_best(grid_results_N2_3, n = 10, metric = "rmse")
