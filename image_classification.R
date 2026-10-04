# Image Classification: Logistic Regression vs. Random Forest
# Portfolio refactor of an individual MATH 3333 course project.
#
# Expected local data layout:
# data/
#   photoMetaData.csv
#   columbiaImages/
#     <image files referenced by the metadata>
#
# The dataset is not included in this repository.

required_packages <- c("jpeg", "randomForest", "pROC")

missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0) {
  stop(
    "Install the required packages first: ",
    paste(missing_packages, collapse = ", ")
  )
}

library(jpeg)
library(randomForest)
library(pROC)

set.seed(123)

metadata_path <- file.path("data", "photoMetaData.csv")
image_dir <- file.path("data", "columbiaImages")

if (!file.exists(metadata_path)) {
  stop(
    "Metadata file not found at ", metadata_path,
    ". See README.md for the expected data layout."
  )
}

metadata <- read.csv(metadata_path, stringsAsFactors = FALSE)

required_columns <- c("name", "category")
missing_columns <- setdiff(required_columns, names(metadata))

if (length(missing_columns) > 0) {
  stop(
    "Metadata is missing required columns: ",
    paste(missing_columns, collapse = ", ")
  )
}

y <- factor(
  ifelse(metadata$category == "outdoor-day", "outdoor", "indoor"),
  levels = c("indoor", "outdoor")
)

read_image <- function(filename) {
  path <- file.path(image_dir, filename)

  if (!file.exists(path)) {
    stop("Image file not found: ", path)
  }

  readJPEG(path)
}

extract_rgb_medians <- function(img) {
  apply(img, 3, median)
}

rgb_to_gray <- function(img) {
  0.2989 * img[, , 1] +
    0.5870 * img[, , 2] +
    0.1140 * img[, , 3]
}

extract_grid_features <- function(img, grid_rows = 10, grid_cols = 10) {
  gray <- rgb_to_gray(img)
  height <- nrow(gray)
  width <- ncol(gray)

  features <- numeric(grid_rows * grid_cols)
  k <- 1

  for (i in seq_len(grid_rows)) {
    row_start <- floor((i - 1) * height / grid_rows) + 1
    row_end <- floor(i * height / grid_rows)

    for (j in seq_len(grid_cols)) {
      col_start <- floor((j - 1) * width / grid_cols) + 1
      col_end <- floor(j * width / grid_cols)

      features[k] <- mean(
        gray[row_start:row_end, col_start:col_end],
        na.rm = TRUE
      )
      k <- k + 1
    }
  }

  features
}

message("Extracting image features...")

images <- lapply(metadata$name, read_image)

logistic_x <- t(vapply(images, extract_rgb_medians, numeric(3)))
colnames(logistic_x) <- c("red_median", "green_median", "blue_median")

rf_x <- t(vapply(images, extract_grid_features, numeric(100)))
colnames(rf_x) <- paste0("grid_", seq_len(ncol(rf_x)))

stratified_split <- function(labels, train_fraction = 0.80) {
  train_indices <- unlist(
    lapply(levels(labels), function(level) {
      idx <- which(labels == level)
      sample(idx, size = floor(length(idx) * train_fraction))
    })
  )

  sort(train_indices)
}

train_idx <- stratified_split(y)
test_idx <- setdiff(seq_along(y), train_idx)

y_train <- y[train_idx]
y_test <- y[test_idx]

evaluate_binary_classifier <- function(actual, probability, threshold = 0.5) {
  predicted <- factor(
    ifelse(probability >= threshold, "outdoor", "indoor"),
    levels = levels(actual)
  )

  tp <- sum(predicted == "outdoor" & actual == "outdoor")
  tn <- sum(predicted == "indoor" & actual == "indoor")
  fp <- sum(predicted == "outdoor" & actual == "indoor")
  fn <- sum(predicted == "indoor" & actual == "outdoor")

  sensitivity <- tp / (tp + fn)
  specificity <- tn / (tn + fp)
  misclassification_error <- mean(predicted != actual)

  roc_obj <- pROC::roc(
    response = actual,
    predictor = probability,
    levels = c("indoor", "outdoor"),
    direction = "<",
    quiet = TRUE
  )

  c(
    sensitivity = sensitivity,
    specificity = specificity,
    misclassification_error = misclassification_error,
    auc = as.numeric(pROC::auc(roc_obj))
  )
}

logistic_train <- data.frame(
  class = y_train,
  logistic_x[train_idx, , drop = FALSE]
)

logistic_model <- glm(
  class ~ .,
  data = logistic_train,
  family = binomial
)

logistic_probability <- predict(
  logistic_model,
  newdata = data.frame(logistic_x[test_idx, , drop = FALSE]),
  type = "response"
)

logistic_metrics <- evaluate_binary_classifier(
  y_test,
  logistic_probability
)

rf_model <- randomForest(
  x = rf_x[train_idx, , drop = FALSE],
  y = y_train,
  ntree = 500,
  importance = TRUE
)

rf_probability <- predict(
  rf_model,
  rf_x[test_idx, , drop = FALSE],
  type = "prob"
)[, "outdoor"]

rf_metrics <- evaluate_binary_classifier(
  y_test,
  rf_probability
)

results <- rbind(
  logistic_regression = logistic_metrics,
  random_forest = rf_metrics
)

print(round(results, 3))
