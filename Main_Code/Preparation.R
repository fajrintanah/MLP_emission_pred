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

# Define a function to calculate the desired summary statistics
summarize_stats <- function(x) {
  tibble(
    Min = min(x, na.rm = TRUE),
    `1st Qu.` = quantile(x, 0.25, na.rm = TRUE),
    Median = median(x, na.rm = TRUE),
    Mean = mean(x, na.rm = TRUE),
    `3rd Qu.` = quantile(x, 0.75, na.rm = TRUE),
    Max = max(x, na.rm = TRUE),
    StDev = sd(x, na.rm = TRUE),
    Skw = (mean(x, na.rm = TRUE) - median(x, na.rm = TRUE)) / sd(x, na.rm = TRUE),
    Krt = sum((x - mean(x, na.rm = TRUE))^4, na.rm = TRUE) /
      (length(x) * (sd(x, na.rm = TRUE)^4)) - 3
  )
}

# Apply the function to each column and reshape the data
summary_R_macro <- R_macro %>%
  select_if(is.numeric) %>% # Select only numeric columns
  summarise(across(everything(), summarize_stats, .names = "{.col}")) %>%
  pivot_longer(cols = everything(), names_to = "Variable", values_to = "Statistics") %>%
  unnest_wider(Statistics)


print(summary_R_macro, n=15)


# ------------- PREPROCESSING ------------------------


Data_N <- R_macro %>% 
				select(-c(8:15))

str(Data_N)



## 1. Outlier Removal using:
### iqr method = boxplot = 1.5*iqr = Tukey method
## 2. eliminate NA 

EL_Data_N <- Data_N %>% 
  mutate(
    # Create the quality control column `qc_N`, flagging outliers based on the IQR method
    qc_N = ifelse(
      N > quantile(N, 0.75, na.rm = TRUE) + 1.5 * IQR(N, na.rm = TRUE) | 
      N < quantile(N, 0.25, na.rm = TRUE) - 1.5 * IQR(N, na.rm = TRUE), 
      0, 
      1
    )
  ) %>% 
  mutate(
    # Replace outlier values with NA
    N = ifelse(qc_N == 0, NA_real_, N)
  ) %>% 
  # Remove rows with NA values
  drop_na()

# Check the structure of the resulting dataframe

str(EL_Data_N )



# eliminate qc_N columns

EL_Data_N2 <- EL_Data_N %>% 
				select(-c(qc_N)) # eliminate N and qc_N columns

str(EL_Data_N2 )
