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
# First define the recipe without immediate prep()
N_rec <- recipe(N ~ OP_Age + Thick + Season + D_Canal + D_OPT + Depth,
                data = train_data_N) %>%
  # Convert character variables to factors first
  step_string2factor(all_nominal_predictors()) %>%  # Critical fix
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

# 3. Prepare recipe using training data ---------------------------------------
N_rec_prepped <- prep(N_rec, training = train_data_N)

# 4. Process datasets ---------------------------------------------------------
train_data_processed <- bake(N_rec_prepped, new_data = train_data_N)
test_data_processed <- bake(N_rec_prepped, new_data = test_data_N)

# 5. Verify processed data structure ------------------------------------------
glimpse(train_data_processed)

# 6. Lightweight Model Spec ---------------------------------------------------
mlp_spec_tune <- mlp(
  epochs = tune(),
  hidden_units = tune(),
  penalty = tune(),
  learn_rate = tune()
) %>% 
  set_engine("brulee", validation = 0) %>%
  set_mode("regression")

# 7. Minimal Workflow --------------------------------------------------------
mlp_wflow_tune <- workflow() %>%
  add_recipe(N_rec) %>%
  add_model(mlp_spec_tune)

# 8. Efficient Parallel Setup -------------------------------------------------
cl <- makePSOCKcluster(max(1, parallel::detectCores() - 2))  # Safer core allocation
registerDoParallel(cl)

# 9. Randomized Grid Search ---------------------------------------------------
set.seed(123)
folds <- vfold_cv(train_data_N, v = 5)

set.seed(456)
param_grid <- grid_random(
  epochs(range = c(500, 1500)),
  hidden_units(range = c(5, 20)),
  penalty(range = c(-4, -1)),
  learn_rate(range = c(-3, -1)),
  size = 50  # 50 random combinations
)

# 10. Memory-Optimized Tuning --------------------------------------------------
grid_results <- tune_grid(
  mlp_wflow_tune,
  resamples = folds,
  grid = param_grid,
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
show_best(grid_results, n = 10, metric = "rmse")

# Final model training with best params
final_model <- mlp_wflow_tune %>%
  finalize_workflow(select_best(grid_results, "rmse")) %>%
  fit(train_data_N)

# Lean test evaluation
test_preds <- predict(final_model, test_data_N) %>% 
  bind_cols(test_data_N %>% select(N))

test_preds %>% metrics(N, .pred)