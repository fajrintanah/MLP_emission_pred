
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

# load file fajrin
load(file='E://Fajrin/Publikasi/Pak Heru B Pulunggono/0 Road to Prof/18 Predicting Macronutrient in peat using ML/Data_Private/modelling_mlp2_19022025_K2.RData')


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


str(R_macro_rev)

Data_Ca <- R_macro_rev %>% 
				select(-c(11:15)) %>% 
				select(-c(7:9))

str(Data_Ca)

EL_Data_Ca2 <- Data_Ca