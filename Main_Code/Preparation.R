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
		tibble [857 × 15] (S3: tbl_df/tbl/data.frame)
		 $ OP_Age : chr [1:857] "<6" "<6" "<6" "<6" ...
		 $ Thick  : chr [1:857] "<3" "<3" "<3" "<3" ...
		 $ Season : chr [1:857] "Rainy" "Rainy" "Rainy" "Rainy" ...
		 $ D_Canal: num [1:857] 10 10 10 10 10 10 10 10 10 10 ...
		 $ D_OPT  : num [1:857] 1 1 1 2 2 2 3 3 3 4 ...
		 $ Depth  : num [1:857] 20 40 70 20 40 70 20 40 70 20 ...
		 $ N      : num [1:857] 21552 15946 11816 13862 14955 ...
		 $ P      : num [1:857] 2082 1361 1931 1303 1641 ...
		 $ K      : num [1:857] 787 237 624 252 259 ...
		 $ Ca     : num [1:857] 1506 654 567 649 499 ...
		 $ Mg     : num [1:857] 282.3 32.7 566.7 89.7 99.8 ...
		 $ NP     : num [1:857] 10.35 11.71 6.12 10.64 9.11 ...
		 $ NK     : num [1:857] 27.4 67.3 18.9 54.9 57.8 ...
		 $ PK     : num [1:857] 2.65 5.75 3.09 5.17 6.35 ...
		 $ CaMg   : num [1:857] 5.33 20.02 1 7.24 5 ...


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
		# A tibble: 12 × 10
		   Variable   Min `1st Qu.`  Median    Mean `3rd Qu.`    Max   StDev    Skw     Krt
		   <chr>    <dbl>     <dbl>   <dbl>   <dbl>     <dbl>  <dbl>   <dbl>  <dbl>   <dbl>
		 1 D_Canal    10     25       75      68.5     100      150    47.1  -0.138  -0.954
		 2 D_OPT       1      1        2       2.50      3        4     1.12  0.444  -1.36 
		 3 Depth      20     20       40      43.4      70       70    20.6   0.165  -1.51 
		 4 N         957.  5457.    8438.   9086.    11724.   34178. 4814.    0.135   2.34 
		 5 P           0    252.     699.   1007.     1592.    7353.  936.    0.329   6.43 
		 6 K           0    154.     276.    419.      446.    8659.  600.    0.238  56.5  
		 7 Ca          0    230.     506.    823.      936.   27186. 1570.    0.202 132.   
		 8 Mg          0     45.9    148.    426.      343.   20577. 1333.    0.209 102.   
		 9 NP          0      6.23    10.1    24.9      24.7    602.   43.0   0.344  45.6  
		10 NK          0     16.8     30.6    49.6      55.0   1012.   76.2   0.250  49.5  
		11 PK          0      0.719    2.67    5.10      5.66   146.    9.88  0.246  71.9  
		12 CaMg        0      1.67     4.24   11.8       8.56   672.   42.8   0.177 142. 



# ------------- PREPROCESSING ------------------------


Data_N <- R_macro %>% 
				select(-c(8:15))

str(Data_N)
		tibble [857 × 7] (S3: tbl_df/tbl/data.frame)
		 $ OP_Age : chr [1:857] "<6" "<6" "<6" "<6" ...
		 $ Thick  : chr [1:857] "<3" "<3" "<3" "<3" ...
		 $ Season : chr [1:857] "Rainy" "Rainy" "Rainy" "Rainy" ...
		 $ D_Canal: num [1:857] 10 10 10 10 10 10 10 10 10 10 ...
		 $ D_OPT  : num [1:857] 1 1 1 2 2 2 3 3 3 4 ...
		 $ Depth  : num [1:857] 20 40 70 20 40 70 20 40 70 20 ...
		 $ N      : num [1:857] 21552 15946 11816 13862 14955 ...


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
		tibble [829 × 8] (S3: tbl_df/tbl/data.frame)
		 $ OP_Age : chr [1:829] "<6" "<6" "<6" "<6" ...
		 $ Thick  : chr [1:829] "<3" "<3" "<3" "<3" ...
		 $ Season : chr [1:829] "Rainy" "Rainy" "Rainy" "Rainy" ...
		 $ D_Canal: num [1:829] 10 10 10 10 10 10 10 10 10 10 ...
		 $ D_OPT  : num [1:829] 1 1 2 2 2 3 3 4 4 4 ...
		 $ Depth  : num [1:829] 40 70 20 40 70 20 70 20 40 70 ...
		 $ N      : num [1:829] 15946 11816 13862 14955 19421 ...
		 $ qc_N   : num [1:829] 1 1 1 1 1 1 1 1 1 1 ...


# eliminate qc_N columns

EL_Data_N2 <- EL_Data_N %>% 
				select(-c(qc_N)) # eliminate N and qc_N columns

str(EL_Data_N2 )
