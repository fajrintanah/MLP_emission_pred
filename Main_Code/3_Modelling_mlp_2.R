# Load required packages
library(tidymodels)
library(brulee)
library(yardstick)
library(doParallel)
library(rsample)

# 1. Data Splitting -----------------------------------------------------------
set.seed(123)
data_split_N <- initial_split(EL_Data_N2, prop = 0.7)
train_data_N <- training(data_split_N)
test_data_N <- testing(data_split_N)

# 2. Recipe Setup (Corrected) -------------------------------------------------
# First define the recipe without immediate prep()
N_rec1 <- recipe(N ~ OP_Age + Thick + Season + D_Canal + D_OPT + Depth,
                data = train_data_N) %>%
  # Convert character variables to factors first
  step_string2factor(all_nominal_predictors()) %>%  # Critical fix
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

# 3. Prepare recipe using training data ---------------------------------------
N_rec_prepped1 <- prep(N_rec1, training = train_data_N)

# 4. Process datasets ---------------------------------------------------------
train_data_processed_N1 <- bake(N_rec_prepped1, new_data = train_data_N)
test_data_processed_N1 <- bake(N_rec_prepped1, new_data = test_data_N)

# 5. Verify processed data structure ------------------------------------------
glimpse(train_data_processed_N1)

# 6. Lightweight Model Spec ---------------------------------------------------
mlp_spec_tune_N1 <- mlp(
  epochs = tune(),
  hidden_units = tune(),
  penalty = tune(),
  learn_rate = tune()
) %>% 
  set_engine("brulee", validation = 0) %>%
  set_mode("regression")

# 7. Minimal Workflow --------------------------------------------------------
mlp_wflow_tune_N1 <- workflow() %>%
  add_recipe(N_rec1) %>%
  add_model(mlp_spec_tune_N1)

# 8. Efficient Parallel Setup -------------------------------------------------
cl <- makePSOCKcluster(max(1, parallel::detectCores() - 2))  # Safer core allocation
registerDoParallel(cl)

# 9. Randomized Grid Search ---------------------------------------------------
set.seed(123)
folds_N <- vfold_cv(train_data_N, v = 5, repeats = 10)

set.seed(123)
param_grid_N1 <- grid_random(
  epochs(range = c(500, 1500)),
  hidden_units(range = c(5, 20)),
  penalty(range = c(-4, -1)),
  learn_rate(range = c(-3, -1)),
  size = 50  # 50 random combinations
)

# 10. Memory-Optimized Tuning --------------------------------------------------
grid_results_N1 <- tune_grid(
  mlp_wflow_tune_N1,
  resamples = folds_N,
  grid = param_grid_N1,
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
show_best(grid_results_N1, n = 10, metric = "rmse")


save.image(file='E://Fajrin/Publikasi/Pak Heru B Pulunggono/0 Road to Prof/18 Predicting Macronutrient in peat using ML/Data_Private/modelling_mlp2_06022025.RData')


load(file='E://Fajrin/Publikasi/Pak Heru B Pulunggono/0 Road to Prof/18 Predicting Macronutrient in peat using ML/Data_Private/modelling_mlp2_06022025.RData')

# 12. Plotting the Results -----------------------------------------------------
autoplot(grid_results_N1)& coord_cartesian(ylim = c(3000, 4000))

# check the rmse.
# 1. higher hidden layers, lower rmse
# 2. epochs at 1000 to 1500, likely to produce lower rmse
# 3. penalty at -1 to -2.5, likely to produce lower rmse
# 4. learn_rate at -1.5 to -1, likely to produce lower rmse

# second try. change to latin hypercube sampling

# ✅ Improved Grid Search with Latin Hypercube Sampling
library(dials)

cl <- makePSOCKcluster(max(1, parallel::detectCores() - 2))
registerDoParallel(cl)

set.seed(123)
folds_N1_1 <- vfold_cv(train_data_N, v = 5, repeats = 5)

set.seed(123)
param_grid_N1_1 <- grid_latin_hypercube(
  epochs(range = c(1000, 1500)),
  hidden_units(range = c(20, 40)),
  penalty(range = c(-2.5, -1)),
  learn_rate(range = c(-1.5, -1)),
  size = 50  # Reduce total combinations
)

# ✅ Memory-Optimized Tuning
grid_results_N1_1 <- tune_grid(
  mlp_wflow_tune_N1,
  resamples = folds_N1_1,
  grid = param_grid_N1_1,
  metrics = metric_set(yardstick::rmse, yardstick::mae),
  control = control_grid(
    verbose = TRUE,
    parallel_over = "resamples",  # More memory efficient
    extract = NULL,
    save_pred = FALSE,
    save_workflow = FALSE,
    pkgs = c("brulee")
  )
)

# ✅ Cleanup & Show Best
stopCluster(cl)
registerDoSEQ()

show_best(grid_results_N1_1, n = 10, metric = "rmse")
