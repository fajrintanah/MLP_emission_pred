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
# Convert scientific notation to numeric for better plotting
grid_results_plot <- show_best(grid_results_N1, n = 50, metric = "rmse") %>%
  mutate(penalty = log10(penalty),  # Convert to log scale
         learn_rate = log10(learn_rate))  # Convert to log scale

# RMSE vs Hidden Units
p1 <- ggplot(grid_results_plot, aes(x = hidden_units, y = mean)) +
  geom_point() +
  geom_smooth(method = "loess", se = FALSE, color = "blue") +
  labs(title = "RMSE vs Hidden Units", x = "Hidden Units", y = "RMSE") +
  ylim(0, 25000) +
  theme_minimal()

# RMSE vs Epochs
p2 <- ggplot(grid_results_plot, aes(x = epochs, y = mean)) +
  geom_point() +
  geom_smooth(method = "loess", se = FALSE, color = "red") +
  labs(title = "RMSE vs Epochs", x = "Epochs", y = "RMSE") +
  ylim(0, 25000) +
  theme_minimal()

# RMSE vs Penalty (Log Scale)
p3 <- ggplot(grid_results_plot, aes(x = penalty, y = mean)) +
  geom_point() +
  geom_smooth(method = "loess", se = FALSE, color = "green") +
  labs(title = "RMSE vs Penalty (Log Scale)", x = "log10(Penalty)", y = "RMSE") +
  ylim(0, 25000) +
  theme_minimal()

# RMSE vs Learn Rate (Log Scale)
p4 <- ggplot(grid_results_plot, aes(x = learn_rate, y = mean)) +
  geom_point() +
  geom_smooth(method = "loess", se = FALSE, color = "purple") +
  labs(title = "RMSE vs Learning Rate (Log Scale)", x = "log10(Learning Rate)", y = "RMSE") +
  ylim(0, 25000) +
  theme_minimal()

# Arrange in a grid
p1 + p2 + p3 + p4 + plot_layout(ncol = 2)

