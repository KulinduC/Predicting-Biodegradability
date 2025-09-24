library(dplyr)
library(caret)
library(e1071)
library(rpart)
library(Boruta)
library(pROC)
library(knitr)
library(randomForest)
library(gbm)
library(glmnet)
library(xgboost)

# New model libs

set.seed(300)

# read data
featurenames <- read.csv("data/chems_feat.name.csv", header = FALSE, colClasses = "character")

# Training and external test data
cdata.df <- read.csv("data/chems_train.data.csv", header = FALSE)
colnames(cdata.df) <- featurenames$V1

# keep your original test path style
tdata.df <- read.csv("~/MATP-4400/data/chems_test.data.csv", header = FALSE)
colnames(tdata.df) <- featurenames$V1

# Class labels
class <- read.csv("data/chems_train.solution.csv", header = FALSE, colClasses = "factor")$V1


# training data 90-10 split
n <- nrow(cdata.df)
ss <- ceiling(n * 0.90)
train.perm <- sample(1:n, ss)

train <- dplyr::slice(cdata.df, train.perm)
validation <- dplyr::slice(cdata.df, -train.perm)
classtrain <- class[train.perm]
classval <- class[-train.perm]

# Factor levels in your style
levels(classtrain) <- make.names(levels(classtrain), unique = TRUE)
levels(classval)   <- make.names(levels(classval),   unique = TRUE)

# Standardization
scaler <- preProcess(train, method = c("center", "scale"))
train_scaled      <- predict(scaler, train)
validation_scaled <- predict(scaler, validation)
test_scaled       <- predict(scaler, tdata.df)


# Boruta Feature Selection
start_time <- Sys.time()
boruta_output <- Boruta(train_scaled, classtrain, doTrace = 0)
end_time <- Sys.time()

cat("Boruta Feature Selection Time:", round(difftime(end_time, start_time, units = "secs"), 2), "seconds\n")
final_features <- getSelectedAttributes(boruta_output, withTentative = TRUE)
cat("Number of features selected by Boruta:", length(final_features), "\n")

train_selected      <- train_scaled[, final_features, drop = FALSE]
validation_selected <- validation_scaled[, final_features, drop = FALSE]
test_selected       <- test_scaled[, final_features, drop = FALSE]

# 5) Helper functions
sensitivity_from_confmat <- function(cm) cm[2,2] / sum(cm[,2])   # assumes table(Predicted, Actual)
specificity_from_confmat <- function(cm) cm[1,1] / sum(cm[,1])

fitControl <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)
d

# RANDOM FOREST (caret::rf) + Boruta
set.seed(401)
rf_model <- train(
  x = train_selected,
  y = classtrain,
  method = "rf",                 # randomForest pkg backend
  trControl = fitControl,
  metric = "ROC",
  tuneLength = 10
)
pred_rf_cls  <- predict(rf_model, newdata = validation_selected)
pred_rf_prob <- predict(rf_model, newdata = validation_selected, type = "prob")

cm_rf <- table(Predicted = pred_rf_cls, Actual = classval)
kable(cm_rf, type = "html", digits = 2, caption = "Actual vs Predicted (Random Forest + Boruta)")

sens_rf <- sensitivity_from_confmat(cm_rf)
spec_rf <- specificity_from_confmat(cm_rf)
balacc_rf <- (sens_rf + spec_rf) / 2
cat("Balanced Accuracy (RF + Boruta):", round(balacc_rf, 3), "\n")

roc_rf <- roc(response = classval, predictor = as.numeric(pred_rf_prob[, 2]))
auc_rf <- auc(roc_rf)
cat("AUC (RF + Boruta):", round(auc_rf, 3), "\n")


# SVM (RBF) + Boruta
set.seed(402)
svm_model <- train(
  x = train_selected,
  y = classtrain,
  method = "svmRadial",
  trControl = fitControl,
  metric = "ROC",
  tuneLength = 10
)
pred_svm_cls  <- predict(svm_model, newdata = validation_selected)
pred_svm_prob <- predict(svm_model, newdata = validation_selected, type = "prob")

cm_svm <- table(Predicted = pred_svm_cls, Actual = classval)
kable(cm_svm, type = "html", digits = 2, caption = "Actual vs Predicted (SVM-RBF + Boruta)")

sens_svm <- sensitivity_from_confmat(cm_svm)
spec_svm <- specificity_from_confmat(cm_svm)
balacc_svm <- (sens_svm + spec_svm) / 2
cat("Balanced Accuracy (SVM + Boruta):", round(balacc_svm, 3), "\n")

roc_svm <- roc(response = classval, predictor = as.numeric(pred_svm_prob[, 2]))
auc_svm <- auc(roc_svm)
cat("AUC (SVM + Boruta):", round(auc_svm, 3), "\n")


# LOGISTIC REGRESSION (GLM binomial) + Boruta
set.seed(403)
glm_model <- train(
  x = train_selected,
  y = classtrain,
  method = "glm",
  family = "binomial",
  trControl = fitControl,
  metric = "ROC"
)
pred_glm_cls  <- predict(glm_model, newdata = validation_selected)
pred_glm_prob <- predict(glm_model, newdata = validation_selected, type = "prob")

cm_glm <- table(Predicted = pred_glm_cls, Actual = classval)
kable(cm_glm, type = "html", digits = 2, caption = "Actual vs Predicted (Logistic Regression + Boruta)")

sens_glm <- sensitivity_from_confmat(cm_glm)
spec_glm <- specificity_from_confmat(cm_glm)
balacc_glm <- (sens_glm + spec_glm) / 2
cat("Balanced Accuracy (Logistic + Boruta):", round(balacc_glm, 3), "\n")

roc_glm <- roc(response = classval, predictor = as.numeric(pred_glm_prob[, 2]))
auc_glm <- auc(roc_glm)
cat("AUC (Logistic + Boruta):", round(auc_glm, 3), "\n")


# GRADIENT BOOSTING (GBM) + Boruta
set.seed(404)
gbm_model <- train(
  x = train_selected,
  y = classtrain,
  method = "gbm",
  trControl = fitControl,
  metric = "ROC",
  verbose = FALSE,
  tuneLength = 10
)
pred_gbm_cls  <- predict(gbm_model, newdata = validation_selected)
pred_gbm_prob <- predict(gbm_model, newdata = validation_selected, type = "prob")

cm_gbm <- table(Predicted = pred_gbm_cls, Actual = classval)
kable(cm_gbm, type = "html", digits = 2, caption = "Actual vs Predicted (GBM + Boruta)")

sens_gbm <- sensitivity_from_confmat(cm_gbm)
spec_gbm <- specificity_from_confmat(cm_gbm)
balacc_gbm <- (sens_gbm + spec_gbm) / 2
cat("Balanced Accuracy (GBM + Boruta):", round(balacc_gbm, 3), "\n")

roc_gbm <- roc(response = classval, predictor = as.numeric(pred_gbm_prob[, 2]))
auc_gbm <- auc(roc_gbm)
cat("AUC (GBM + Boruta):", round(auc_gbm, 3), "\n")


# XGBOOST (xgbTree) + Boruta
set.seed(405)
xgb_model <- train(
  x = as.matrix(train_selected),
  y = classtrain,
  method = "xgbTree",
  trControl = fitControl,
  metric = "ROC",
  tuneLength = 10
)
pred_xgb_cls  <- predict(xgb_model, newdata = as.matrix(validation_selected))
pred_xgb_prob <- predict(xgb_model, newdata = as.matrix(validation_selected), type = "prob")

cm_xgb <- table(Predicted = pred_xgb_cls, Actual = classval)
kable(cm_xgb, type = "html", digits = 2, caption = "Actual vs Predicted (XGBoost + Boruta)")

sens_xgb <- sensitivity_from_confmat(cm_xgb)
spec_xgb <- specificity_from_confmat(cm_xgb)
balacc_xgb <- (sens_xgb + spec_xgb) / 2
cat("Balanced Accuracy (XGBoost + Boruta):", round(balacc_xgb, 3), "\n")

roc_xgb <- roc(response = classval, predictor = as.numeric(pred_xgb_prob[, 2]))
auc_xgb <- auc(roc_xgb)
cat("AUC (XGBoost + Boruta):", round(auc_xgb, 3), "\n")


# Final Summary (Validation)
cat("\n--- FINAL SUMMARY (Validation) ---\n")
summary_tbl <- data.frame(
  Model = c("Random Forest + Boruta",
            "SVM (RBF) + Boruta",
            "Logistic Regression + Boruta",
            "GBM + Boruta",
            "XGBoost + Boruta"),
  Balanced_Accuracy = c(balacc_rf, balacc_svm, balacc_glm, balacc_gbm, balacc_xgb),
  AUC               = c(as.numeric(auc_rf), as.numeric(auc_svm),
                        as.numeric(auc_glm), as.numeric(auc_gbm), as.numeric(auc_xgb))
)

print(knitr::kable(summary_tbl[order(-summary_tbl$AUC, -summary_tbl$Balanced_Accuracy), ],
                   digits = 3, caption = "Model Comparison (Validation)"))
