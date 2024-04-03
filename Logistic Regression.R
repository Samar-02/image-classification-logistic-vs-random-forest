### Photograph examples

## Read in the library and metadata
library(jpeg)
pm <- read.csv("C:/Users/samar/OneDrive/Documents/Universities/1. York/2023 - 2024/2. Winter/MATH 3333/Final Project/Final Project-20240325/photoMetaData.csv")
n <- nrow(pm)

trainFlag <- (runif(n) > 0.5)
y <- as.numeric(pm$category == "outdoor-day")

X <- matrix(NA, ncol=3, nrow=n)
for (j in 1:n) {
  img <- readJPEG(paste0("C:/Users/samar/OneDrive/Documents/Universities/1. York/2023 - 2024/2. Winter/MATH 3333/Final Project/Final Project-20240325/columbiaImages/columbiaImages/",pm$name[j]))
  X[j,] <- apply(img,3,median)
  print(sprintf("%03d / %03d", j, n))
}



# build a glm model on these median values
out <- glm(y ~ X, family=binomial, subset=trainFlag)
out$iter
summary(out)

# How well did we do?
pred <- 1 / (1 + exp(-1 * cbind(1,X) %*% coef(out)))
y[order(pred)]
y[!trainFlag][order(pred[!trainFlag])]

mean((as.numeric(pred > 0.5) == y)[trainFlag])
mean((as.numeric(pred > 0.5) == y)[!trainFlag])

## ROC curve (see lecture 12)
roc <- function(y, pred) {
  alpha <- quantile(pred, seq(0,1,by=0.01))
  N <- length(alpha)
  
  sens <- rep(NA,N)
  spec <- rep(NA,N)
  for (i in 1:N) {
    predClass <- as.numeric(pred >= alpha[i])
    sens[i] <- sum(predClass == 1 & y == 1) / sum(y == 1)
    spec[i] <- sum(predClass == 0 & y == 0) / sum(y == 0)
  }
  return(list(fpr=1- spec, tpr=sens))
}

r <- roc(y[!trainFlag], pred[!trainFlag])
plot(r$fpr, r$tpr, xlab="false positive rate", ylab="true positive rate", type="l")
abline(0,1,lty="dashed")

# auc
auc <- function(r) {
  sum((r$fpr) * diff(c(0,r$tpr)))
}
glmAuc <- auc(r)
glmAuc




#### Do this!!!!!!!!!
# Calculate True Positives, True Negatives, False Positives, False Negatives
TP <- sum((as.numeric(pred > 0.5) == 1) & (y == 1))
TN <- sum((as.numeric(pred > 0.5) == 0) & (y == 0))
FP <- sum((as.numeric(pred > 0.5) == 1) & (y == 0))
FN <- sum((as.numeric(pred > 0.5) == 0) & (y == 1))

# Calculate Sensitivity and Specificity
Sensitivity <- TP / (TP + FN)
Specificity <- TN / (TN + FP)

# Calculate Misclassification Error
Misclassification_Error <- (FP + FN) / length(y)

# Print the results
cat("Sensitivity:", Sensitivity, "\n")
cat("Specificity:", Specificity, "\n")
cat("Misclassification Error:", Misclassification_Error, "\n")

