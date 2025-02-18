### ---------------- SKRIP MAKRO BARU FIX ------------------------


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

# -------------------------- Modelling for N -------------------------------------------------


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

# look after the histogram
# firstly, cast the wrapper function to automatize the process

library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)

windowsFonts(Palatino = windowsFont("Palatino Linotype"))

plot_histograms_patchwork <- function(data, columns = where(is.numeric), rename_xlab = NULL) {  
  # Select columns properly
  selected_cols <- if (is.numeric(columns)) {
    names(data)[columns]  # Select by index
  } else {
    names(select(data, {{ columns }}))  # Select by condition (e.g., numeric)
  }  
  plots <- list()  # Store individual plots  
  for (col in selected_cols) {
    x_label <- if (!is.null(rename_xlab) && col %in% names(rename_xlab)) rename_xlab[[col]] else "Value"    
    # Conditional y-axis title only for "Ca"
    y_label <- if (col == "Ca") "Frequency" else NULL    
    p <- ggplot(data, aes(x = .data[[col]])) +
      geom_histogram(aes(y = after_stat(count)), bins = 30, fill = "#69b3a2", color = "#e9ecef", alpha = 0.9, na.rm = TRUE) +
      theme_bw() +
      theme(
        text = element_text(family = "Palatino"),
        axis.title.x = element_text(size = 12),
        axis.title.y = if (col == "Ca") element_text(size = 12) else element_blank(),
        plot.title = element_blank()  # Remove titles
      ) +
      labs(x = x_label, y = y_label)    
    plots <- append(plots, list(p))
  }  
  # Arrange all plots using patchwork with 3 columns
  final_plot <- wrap_plots(plots) + plot_layout(ncol = 3)  
  return(final_plot)
}

# Example usage
plot_hist_all <- plot_histograms_patchwork(
  data = R_macro_rev, 
  columns = 7:15, 
  rename_xlab = c("N" = "Total N (mg/kg)", "P" = "Total P (mg/kg)", "K" = "Total K (mg/kg)", 
                  "Ca" = "Total Ca (cmol(+)/kg)", "Mg" = "Total Mg (cmol(+)/kg)", 
                  "NP" = "N:P ", "NK" = "N:K ", "PK" = "P:K ", "CaMg" = "Ca:Mg ")
)

# Print the arranged plot
plot_hist_all


# plot boxplots over all variables

library(ggplot2)
library(patchwork)
library(dplyr)
library(paletteer)  # Using paletteer for color scales

windowsFonts(Palatino = windowsFont("Palatino Linotype"))

plot_boxplots_manual <- function(data, factor_columns = 1:6, value_columns = 7:15, rename_vars = NULL) {
  # Convert factor columns to categorical
  data <- data %>% mutate(across(all_of(names(data)[factor_columns]), as.factor))
  # Get column names for axes
  factor_names <- names(data)[factor_columns]
  value_names <- names(data)[value_columns]
  # Rename columns if rename_vars is provided
  if (!is.null(rename_vars)) {
    factor_names <- ifelse(factor_names %in% names(rename_vars), rename_vars[factor_names], factor_names)
    value_names <- ifelse(value_names %in% names(rename_vars), rename_vars[value_names], value_names)
  }
  # Generate a list of boxplots
  plots <- list()
  num_factors <- length(factor_columns)
  num_values <- length(value_columns)
  for (value_idx in seq_along(value_columns)) {
    for (factor_idx in seq_along(factor_columns)) {
      factor_col <- names(data)[factor_columns[factor_idx]]
      value_col <- names(data)[value_columns[value_idx]]
      factor_label <- factor_names[factor_idx]  # Renamed if applicable
      value_label <- value_names[value_idx]  # Renamed if applicable
      # Determine whether to show axis labels and titles
      show_x_labels <- (value_idx == num_values)  # Only for the last row
      show_y_labels <- (factor_idx == 1)  # Only for the first column
      p <- ggplot(data, aes(x = .data[[factor_col]], y = .data[[value_col]], fill = .data[[factor_col]], color = .data[[factor_col]])) +
        geom_boxplot(outlier.shape = NA, alpha = 0.9) +
        geom_jitter(shape = 21, alpha = 0.3, size = 1.7, stroke = 0.5) + # Shape 21 for circles
        scale_fill_paletteer_d("nationalparkcolors::Acadia") +  # Apply Acadia color palette
        scale_color_paletteer_d("nationalparkcolors::Acadia") + # Ensure jitter points match the box colors
        theme_bw() +
        theme(
          text = element_text(family = "Palatino"),
          legend.position = "none",
          axis.text.x = if (show_x_labels) element_text(angle = 45, hjust = 1, vjust = 1) else element_blank(),
          axis.text.y = if (show_y_labels) element_text() else element_blank(),
          axis.title.x = if (show_x_labels) element_text(size = 12) else element_blank(),
          axis.title.y = if (show_y_labels) element_text(size = 12) else element_blank()
        ) +
        labs(x = if (show_x_labels) factor_label else NULL, 
             y = if (show_y_labels) value_label else NULL)

      plots <- append(plots, list(p))
    }
  }
  # Arrange plots in a grid layout (6 columns)
  final_plot <- wrap_plots(plots) + plot_layout(ncol = num_factors)  
  return(final_plot)
}

# Example usage
plot_boxplots_manual_all <- plot_boxplots_manual(
  data = R_macro_rev, 
  factor_columns 	= 1:6, 
  value_columns 	= 7:15, 
  rename_vars 	= c("N" = "Total N (mg/kg)", "P" = "Total P (mg/kg)", "K" = "Total K (mg/kg)", "Ca" = "Total Ca (cmol(+)/kg)", 
					  "Mg" = "Total Mg (cmol(+)/kg)", "NP" = "N:P", "NK" = "N:K", "PK" = "P:K", "CaMg" = "Ca:Mg",
					  "OP_Age" = "OP Age (year)", "Thick" = "Peat Thickness (m)",
					  "D_Canal" = "Distance from Canal (m)", "D_OPT" = "Distance from OP (m)", 
					  "Depth" = "Sampling Depth (cm)")
)

# Print the final arranged plot
plot_boxplots_manual_all


# Define a function to calculate the desired summary statistics
summarize_stats_df <- function(data, columns = where(is.numeric)) {
  # Convert numeric indices to column names
  if (is.numeric(columns)) {
    columns <- names(data)[columns]
  }  
  # Compute summary statistics
  data %>%
    select(all_of(columns)) %>%
    summarise(across(everything(), ~ list(tibble(
      Metric = c("Min", "1st Qu.", "Median", "Mean", "3rd Qu.", "Max", "StDev", "Skw", "Krt"),
      Value = c(
        min(.x, na.rm = TRUE),
        quantile(.x, 0.25, na.rm = TRUE),
        median(.x, na.rm = TRUE),
        mean(.x, na.rm = TRUE),
        quantile(.x, 0.75, na.rm = TRUE),
        max(.x, na.rm = TRUE),
        sd(.x, na.rm = TRUE),
        (mean(.x, na.rm = TRUE) - median(.x, na.rm = TRUE)) / sd(.x, na.rm = TRUE),
        sum((.x - mean(.x, na.rm = TRUE))^4, na.rm = TRUE) / (length(.x) * (sd(.x, na.rm = TRUE)^4)) - 3
      )
    )))) %>%
    pivot_longer(cols = everything(), names_to = "Variable", values_to = "Statistics") %>%
    unnest(Statistics) %>% # nolint
    pivot_wider(names_from = Metric, values_from = Value) # nolint # nolint
}


summary_R_macro_selected <- summarize_stats_df(data = R_macro_rev, 
												columns = 7:15)  # Using column indices. Also accepts c("column names") or all numeric columns by default
summary_R_macro_selected 

write_xlsx(summary_R_macro_selected, "E:/Fajrin/Publikasi/Pak Heru B Pulunggono/0 Road to Prof/18 Predicting Macronutrient in peat using ML/Data/summary_R_macro_selected.xlsx") # test if the process run well. Result = OK !! 


# ------------- PREPROCESSING ------------------------

Data_N <- R_macro_rev %>% 
				select(-c(8:15))

str(Data_N)

EL_Data_N2 <- Data_N