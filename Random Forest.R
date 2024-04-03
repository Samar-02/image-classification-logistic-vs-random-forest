# Load required libraries
library(randomForest)
library(jpeg)
library(pROC)

# Function to convert image to grayscale
rgb2gray <- function(img) {
  return(0.2989 * img[,,1] + 0.587 * img[,,2] + 0.114 * img[,,3])
}

# Function to calculate ROC curve
roc <- function(y, pred) {
  alpha <- quantile(pred, seq(0, 1, by = 0.01))
  N <- length(alpha)
  
  sens <- rep(NA, N)
  spec <- rep(NA, N)
  for (i in 1:N) {
    predClass <- as.numeric(pred >= alpha[i])
    sens[i] <- sum(predClass == 1 & y == 1) / sum(y == 1)
    spec[i] <- sum(predClass == 0 & y == 0) / sum(y == 0)
  }
  return(list(fpr = 1 - spec, tpr = sens))
}

# Function to calculate AUC
auc <- function(r) {
  sum((r$fpr) * diff(c(0, r$tpr)))
}

# Step 1: Feature Extraction
extract_features <- function(image_path, grid_size = c(10, 10), summary_statistic = "mean") {
  img <- readJPEG(image_path)
  img_gray <- rgb2gray(img)
  height <- dim(img_gray)[1]
  width <- dim(img_gray)[2]
  features <- numeric()
  
  for (i in 1:grid_size[1]) {
    for (j in 1:grid_size[2]) {
      row_start <- floor((i - 1) * (height / grid_size[1])) + 1
      row_end <- floor(i * (height / grid_size[1]))
      col_start <- floor((j - 1) * (width / grid_size[2])) + 1
      col_end <- floor(j * (width / grid_size[2]))
      
      grid_area <- img_gray[row_start:row_end, col_start:col_end]
      if (summary_statistic == "mean") {
        features <- c(features, mean(grid_area))
      } else if (summary_statistic == "median") {
        features <- c(features, median(grid_area))
      }
      # Add more summary statistics if needed
    }
  }
  return(features)
}

# Step 2: Prepare Data
prepare_data <- function(image_folder, metadata_file, grid_size = c(10, 10), summary_statistic = "mean") {
  metadata <- read.csv(metadata_file)
  X <- matrix(nrow = nrow(metadata), ncol = grid_size[1] * grid_size[2])
  y <- as.numeric(metadata$category == "outdoor-day")
  
  for (i in 1:nrow(metadata)) {
    image_path <- file.path(image_folder, metadata$name[i])
    features <- extract_features(image_path, grid_size, summary_statistic)
    X[i,] <- features
  }
  return(list(X = X, y = y))
}

# Step 3: Random Forest Model
train_random_forest <- function(X_train, y_train, n_estimators = 100) {
  rf_model <- randomForest(x = X_train, y = y_train, ntree = n_estimators)
  return(rf_model)
}

# Function to evaluate model
evaluate_model <- function(model, X_test, y_test) {
  y_pred <- predict(model, X_test)
  
  # Calculate AUC
  roc_obj <- roc(y_test, y_pred)
  auc_value <- auc(roc_obj)
  cat("AUC:", auc_value, "\n")
  
  # Calculate True Positives, True Negatives, False Positives, False Negatives
  TP <- sum((as.numeric(y_pred > 0.5) == 1) & (y_test == 1))
  TN <- sum((as.numeric(y_pred > 0.5) == 0) & (y_test == 0))
  FP <- sum((as.numeric(y_pred > 0.5) == 1) & (y_test == 0))
  FN <- sum((as.numeric(y_pred > 0.5) == 0) & (y_test == 1))
  
  # Calculate Sensitivity and Specificity
  Sensitivity <- TP / (TP + FN)
  Specificity <- TN / (TN + FP)
  
  # Calculate Misclassification Error
  Misclassification_Error <- (FP + FN) / length(y_test)
  
  # Print the results
  cat("Sensitivity:", Sensitivity, "\n")
  cat("Specificity:", Specificity, "\n")
  cat("Misclassification Error:", Misclassification_Error, "\n")
}

# Example usage
image_folder <- "C:/Users/samar/OneDrive/Documents/Universities/1. York/2023 - 2024/2. Winter/MATH 3333/Final Project/Final Project-20240325/columbiaImages/columbiaImages/"
metadata_file <- "C:/Users/samar/OneDrive/Documents/Universities/1. York/2023 - 2024/2. Winter/MATH 3333/Final Project/Final Project-20240325/photoMetaData.csv"

data <- prepare_data(image_folder, metadata_file, grid_size = c(10, 10), summary_statistic = "mean")
X <- data$X
y <- data$y

# Split data into training and testing sets (80% training, 20% testing)
set.seed(123) # for reproducibility
train_index <- sample(1:nrow(X), 0.8 * nrow(X))
X_train <- X[train_index, ]
y_train <- y[train_index]
X_test <- X[-train_index, ]
y_test <- y[-train_index]

# Train random forest model
rf_model <- train_random_forest(X_train, y_train)

# Evaluate model
evaluate_model(rf_model, X_test, y_test)
