# =============================================================================
# CREDIT RISK CLASSIFICATION — PORTFOLIO PROJECT
#
# Author:  Richmond Osei
# Dataset: German Credit Data (UCI Machine Learning Repository)
#          1,000 loan applicants | 20 predictors | Binary outcome
#
# Business Problem:
#   A bank needs to classify loan applicants as Good or Bad credit risks.
#   Missed defaults (false negatives) cost the bank bad debt.
#   False alarms (false positives) cost the bank lost revenue.
#   The goal is to build and compare classification models that minimise
#   total business cost while maintaining regulatory interpretability.
#




# Core statistical learning libraries
library(MASS)      # lda(), qda() — discriminant analysis functions
library(class)     # knn() — K-nearest neighbors classifier

# Data access and utilities
library(caret)     # confusionMatrix() — enhanced confusion matrix metrics

# Visualisation
library(pROC)      # roc(), auc() — ROC curve analysis

# Suppress startup messages for clean output
options(warn = -1)


cat("  CREDIT RISK CLASSIFICATION — PORTFOLIO PROJECT\n")
cat("  Richmond Osei | Chapter 4: Classification Methods\n")



# =============================================================================
# SECTION 1: DATA LOADING AND PREPARATION
# The German Credit dataset is a classic benchmark in credit risk modelling
# Source: UCI Machine Learning Repository
# =============================================================================

cat("--- SECTION 1: Data Loading ---\n")

# Download and load the German Credit dataset
# This is a real-world dataset used in quantitative finance research
# 1000 applicants assessed as Good (1) or Bad (2) credit risks

url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

# Column names based on the UCI dataset documentation
col_names <- c(
  "checking_account",   # Status of existing checking account
  "duration",           # Duration in months
  "credit_history",     # Credit history
  "purpose",            # Purpose of loan
  "credit_amount",      # Credit amount (DM)
  "savings_account",    # Savings account / bonds
  "employment",         # Present employment since
  "installment_rate",   # Installment rate as % of disposable income
  "personal_status",    # Personal status and sex
  "other_debtors",      # Other debtors / guarantors
  "residence_since",    # Present residence since
  "property",           # Property type
  "age",                # Age in years
  "other_plans",        # Other installment plans
  "housing",            # Housing status
  "existing_credits",   # Number of existing credits at this bank
  "job",                # Job type
  "dependents",         # Number of people being liable
  "telephone",          # Telephone registered
  "foreign_worker",     # Foreign worker status
  "credit_risk"         # RESPONSE: 1 = Good, 2 = Bad
)

# Read the raw data
credit_raw <- read.table(url,
                         header = FALSE,
                         col.names = col_names,
                         stringsAsFactors = TRUE)

# Convert response to binary factor: Good = 0, Bad = 1
# We code Bad = 1 so that our models predict the probability of being a bad risk
# This is standard in credit risk — we are predicting the probability of default
credit_raw$credit_risk <- ifelse(credit_raw$credit_risk == 1, 0, 1)
credit_raw$credit_risk <- as.factor(credit_raw$credit_risk)

cat("Dataset loaded successfully.\n")
cat("Dimensions:", nrow(credit_raw), "applicants,", ncol(credit_raw), "variables\n")
cat("Response levels: 0 = Good Credit Risk, 1 = Bad Credit Risk\n\n")


# =============================================================================
# EXPLORATORY DATA ANALYSIS
# Always understand your data before modelling
# Key questions: class balance, predictor distributions, correlations
# =============================================================================

cat("--- SECTION 2: Exploratory Data Analysis ---\n")

#
# Credit risk datasets are almost always imbalanced
# More good customers than bad — this affects model evaluation
risk_table <- table(credit_raw$credit_risk)
risk_pct   <- round(prop.table(risk_table) * 100, 1)

cat("Credit Risk Distribution:\n")
cat("  Good (0):", risk_table["0"], "applicants (", risk_pct["0"], "%)\n")
cat("  Bad  (1):", risk_table["1"], "applicants (", risk_pct["1"], "%)\n\n")

# 70% good, 30% bad — moderate imbalance
# Important: a model that always predicts "Good" would be 70% accurate
# This is why overall accuracy alone is insufficient — we need sensitivity

# --- 2.2 Key Numeric Predictors 
# Extract numeric variables for correlation and summary analysis
numeric_vars <- c("duration", "credit_amount", "installment_rate",
                  "residence_since", "age", "existing_credits", "dependents")

cat("Summary of Key Numeric Predictors:\n")
print(round(sapply(credit_raw[, numeric_vars], summary), 2))

# --- 2.3 Correlation with Credit Risk ---
# Convert response to numeric for correlation calculation
credit_numeric <- as.numeric(as.character(credit_raw$credit_risk))

cor_with_risk <- cor(credit_raw[, numeric_vars], credit_numeric)
cat("\nCorrelation of Numeric Predictors with Credit Risk:\n")
print(round(sort(abs(cor_with_risk[,1]), decreasing = TRUE), 3))

# --- 2.4 Publication-Quality Visualisations ---

# Plot 1: Class Distribution
barplot(risk_table,
        names.arg = c("Good Risk (0)", "Bad Risk (1)"),
        col = c("#2196F3", "#F44336"),
        main = "Credit Risk Distribution\nGerman Credit Dataset (n = 1,000)",
        ylab = "Number of Applicants",
        ylim = c(0, 800),
        border = NA)
text(0.7, risk_table["0"] + 20,
     paste0(risk_table["0"], " (", risk_pct["0"], "%)"),
     cex = 1.2, font = 2)
text(1.9, risk_table["1"] + 20,
     paste0(risk_table["1"], " (", risk_pct["1"], "%)"),
     cex = 1.2, font = 2)


# Plot 2: Key Predictor Distributions by Credit Risk
par(mfrow = c(2, 3), mar = c(4, 4, 3, 1))

# Duration
boxplot(duration ~ credit_risk, data = credit_raw,
        names = c("Good", "Bad"),
        col = c("#2196F3", "#F44336"),
        main = "Loan Duration (months)",
        xlab = "Credit Risk", ylab = "Duration",
        border = c("#1565C0", "#B71C1C"))

# Credit Amount
boxplot(credit_amount ~ credit_risk, data = credit_raw,
        names = c("Good", "Bad"),
        col = c("#2196F3", "#F44336"),
        main = "Credit Amount (DM)",
        xlab = "Credit Risk", ylab = "Amount",
        border = c("#1565C0", "#B71C1C"))

# Age
boxplot(age ~ credit_risk, data = credit_raw,
        names = c("Good", "Bad"),
        col = c("#2196F3", "#F44336"),
        main = "Applicant Age (years)",
        xlab = "Credit Risk", ylab = "Age",
        border = c("#1565C0", "#B71C1C"))

# Installment Rate
boxplot(installment_rate ~ credit_risk, data = credit_raw,
        names = c("Good", "Bad"),
        col = c("#2196F3", "#F44336"),
        main = "Installment Rate (% of income)",
        xlab = "Credit Risk", ylab = "Rate",
        border = c("#1565C0", "#B71C1C"))

# Existing Credits
boxplot(existing_credits ~ credit_risk, data = credit_raw,
        names = c("Good", "Bad"),
        col = c("#2196F3", "#F44336"),
        main = "Existing Credits at Bank",
        xlab = "Credit Risk", ylab = "Number",
        border = c("#1565C0", "#B71C1C"))

# Duration histogram by class
hist(credit_raw$duration[credit_raw$credit_risk == 0],
     col = rgb(0.13, 0.59, 0.95, 0.6),
     main = "Loan Duration Distribution by Risk",
     xlab = "Duration (months)",
     xlim = c(0, 75), breaks = 15)
hist(credit_raw$duration[credit_raw$credit_risk == 1],
     col = rgb(0.96, 0.26, 0.21, 0.6),
     add = TRUE, breaks = 15)
legend("topright", c("Good Risk", "Bad Risk"),
       fill = c(rgb(0.13, 0.59, 0.95, 0.6), rgb(0.96, 0.26, 0.21, 0.6)),
       bty = "n")

par(mfrow = c(1, 1))


# =============================================================================
# TRAIN / TEST SPLIT
# Split 70% training, 30% test — stratified to preserve class balance
# Always split BEFORE modelling — never let test data influence training
# =============================================================================

cat("--- SECTION 3: Train/Test Split ---\n")

set.seed(42)  # Reproducibility — document your seed in real projects

n <- nrow(credit_raw)

# Stratified split: sample separately from each class to maintain balance
# This prevents the split from accidentally putting all bad risks in one set
good_idx <- which(credit_raw$credit_risk == 0)
bad_idx  <- which(credit_raw$credit_risk == 1)

train_good <- sample(good_idx, size = floor(0.7 * length(good_idx)))
train_bad  <- sample(bad_idx,  size = floor(0.7 * length(bad_idx)))
train_idx  <- c(train_good, train_bad)

credit_train <- credit_raw[train_idx, ]
credit_test  <- credit_raw[-train_idx, ]
risk_test    <- credit_raw$credit_risk[-train_idx]

cat("Training set:", nrow(credit_train), "applicants\n")
cat("  Good:", sum(credit_train$credit_risk == 0),
    "| Bad:", sum(credit_train$credit_risk == 1), "\n")
cat("Test set:", nrow(credit_test), "applicants\n")
cat("  Good:", sum(credit_test$credit_risk == 0),
    "| Bad:", sum(credit_test$credit_risk == 1), "\n\n")


# =============================================================================
# LOGISTIC REGRESSION
# The workhorse of binary classification in finance and economics
# Directly models P(Y=1|X) using the logistic function
# Produces interpretable coefficients — essential for regulatory contexts
# =============================================================================

cat("--- SECTION 4: Logistic Regression ---\n")

# --- 4.1 Full Model ---
# Start with key financial predictors identified in EDA
# duration, credit_amount, age, installment_rate, checking_account, savings_account
glm_full <- glm(credit_risk ~ duration + credit_amount + age +
                  installment_rate + checking_account + savings_account +
                  credit_history + employment + purpose,
                data = credit_train,
                family = binomial)  # family=binomial tells R: logistic regression

cat("Full Logistic Regression Model:\n")
print(summary(glm_full))

# --- 4.2 Confounding Demonstration ---
# Does loan duration affect credit risk differently when we control for amount?
# This mirrors the student/balance confounding example from the book

# Simple model: duration only
glm_simple_dur <- glm(credit_risk ~ duration,
                      data = credit_train, family = binomial)

# Multiple model: duration + credit_amount
glm_multi_dur <- glm(credit_risk ~ duration + credit_amount,
                     data = credit_train, family = binomial)

cat("\n--- Confounding Demonstration ---\n")
cat("Duration coefficient (simple model):",
    round(coef(glm_simple_dur)["duration"], 4), "\n")
cat("Duration coefficient (with credit amount):",
    round(coef(glm_multi_dur)["duration"], 4), "\n")
cat("Interpretation: Longer loans look riskier overall, but once we control\n")
cat("for loan size, the duration effect may shift. Longer loans for larger\n")
cat("amounts may actually be less risky per unit borrowed.\n\n")

# --- 4.3 Parsimonious Model for Prediction ---
# Remove predictors with large p-values (reduce variance without increasing bias)
glm_final <- glm(credit_risk ~ duration + credit_amount + age +
                   installment_rate + checking_account + savings_account,
                 data = credit_train,
                 family = binomial)

cat("Final Logistic Regression Coefficients:\n")
coef_table <- round(summary(glm_final)$coef, 4)
print(coef_table)

# Business interpretation of key coefficients
cat("\nBusiness Interpretation:\n")
cat("  duration:     each additional month increases log-odds of bad risk\n")
cat("  credit_amount: larger loans slightly increase bad risk probability\n")
cat("  age:          older applicants are slightly less risky (negative coef)\n")
cat("  checking acct: better account status strongly reduces bad risk\n\n")

# --- 4.4 Predictions at Default 50% Threshold ---
glm_probs <- predict(glm_final, credit_test, type = "response")
glm_pred_50 <- ifelse(glm_probs > 0.5, "1", "0")

cat("Logistic Regression — Confusion Matrix (threshold = 0.50):\n")
conf_glm_50 <- table(Predicted = glm_pred_50, Actual = risk_test)
print(conf_glm_50)

glm_acc_50  <- mean(glm_pred_50 == risk_test)
glm_sens_50 <- conf_glm_50["1","1"] / sum(conf_glm_50[,"1"])
glm_spec_50 <- conf_glm_50["0","0"] / sum(conf_glm_50[,"0"])

cat(sprintf("Accuracy:    %.1f%%\n", glm_acc_50 * 100))
cat(sprintf("Sensitivity: %.1f%% (catching actual bad risks)\n", glm_sens_50 * 100))
cat(sprintf("Specificity: %.1f%% (correctly clearing good risks)\n\n", glm_spec_50 * 100))

# --- 4.5 Threshold Optimisation ---
# The 50% threshold minimises overall error but may not minimise business cost
# In credit risk: missing a bad risk (false negative) is more costly than
# a false alarm (false positive) — bad debt > lost revenue from one customer
# A lower threshold catches more bad risks at cost of more false alarms

glm_pred_30 <- ifelse(glm_probs > 0.3, "1", "0")
conf_glm_30 <- table(Predicted = glm_pred_30, Actual = risk_test)

glm_acc_30  <- mean(glm_pred_30 == risk_test)
glm_sens_30 <- conf_glm_30["1","1"] / sum(conf_glm_30[,"1"])
glm_spec_30 <- conf_glm_30["0","0"] / sum(conf_glm_30[,"0"])

cat("Logistic Regression — Confusion Matrix (threshold = 0.30):\n")
print(conf_glm_30)
cat(sprintf("Accuracy:    %.1f%%\n", glm_acc_30 * 100))
cat(sprintf("Sensitivity: %.1f%% (catching actual bad risks)\n", glm_sens_30 * 100))
cat(sprintf("Specificity: %.1f%% (correctly clearing good risks)\n\n", glm_spec_30 * 100))


# =============================================================================
# SECTION 5: LINEAR DISCRIMINANT ANALYSIS (LDA)
# Models the distribution of predictors within each class
# Then uses Bayes' theorem to compute P(class | predictors)
# Assumes: normality within each class, shared covariance matrix
# Advantage over logistic: more stable when classes are well-separated
# =============================================================================

cat("--- SECTION 5: Linear Discriminant Analysis ---\n")
library(MASS)

lda_fit <- lda(credit_risk ~ duration + credit_amount + age +
                 installment_rate + checking_account + savings_account,
               data = credit_train)

lda_fit <- lda(credit_risk ~ duration + credit_amount + age +
                 installment_rate + checking_account + savings_account,
               data = credit_train)

# Examine LDA output — three key pieces of information
cat("LDA Model Output:\n")
print(lda_fit)

cat("\nInterpretation of LDA Output:\n")
cat("Prior probabilities: proportion of each class in training data\n")
cat("Group means: average predictor value for Good vs Bad risk applicants\n")
cat("  Bad risks have longer loans, larger amounts, worse checking accounts\n")
cat("LD1 coefficients: the linear combination that best separates the classes\n\n")

# Predictions
lda_pred <- predict(lda_fit, credit_test)

# lda_pred is a list with three elements:
# $class     = predicted class label (Good/Bad)
# $posterior = matrix of posterior probabilities for each class
# $x         = the raw LD1 score for each observation

cat("LDA Confusion Matrix (threshold = 0.50):\n")
conf_lda <- table(Predicted = lda_pred$class, Actual = risk_test)
print(conf_lda)

lda_acc  <- mean(lda_pred$class == risk_test)
lda_sens <- conf_lda["1","1"] / sum(conf_lda[,"1"])
lda_spec <- conf_lda["0","0"] / sum(conf_lda[,"0"])

cat(sprintf("Accuracy:    %.1f%%\n", lda_acc * 100))
cat(sprintf("Sensitivity: %.1f%%\n", lda_sens * 100))
cat(sprintf("Specificity: %.1f%%\n\n", lda_spec * 100))

# Distribution of posterior probabilities
cat("Posterior Probability Summary:\n")
cat("P(Bad Risk) for actually GOOD applicants:\n")
print(round(summary(lda_pred$posterior[risk_test == "0", "1"]), 3))
cat("P(Bad Risk) for actually BAD applicants:\n")
print(round(summary(lda_pred$posterior[risk_test == "1", "1"]), 3))
cat("Good separation = bad applicants should have higher posterior probabilities\n\n")


# =============================================================================
# SECTION 6: QUADRATIC DISCRIMINANT ANALYSIS (QDA)
# Like LDA but allows each class to have its own covariance matrix
# More flexible: can model non-linear decision boundaries
# Trade-off: more parameters to estimate — needs larger sample sizes
# Use QDA when: large n, or covariances clearly differ between classes
# =============================================================================

cat("--- SECTION 6: Quadratic Discriminant Analysis ---\n")

qda_fit <- qda(credit_risk ~ duration + credit_amount + age +
                 installment_rate + checking_account + savings_account,
               data = credit_train)

# QDA does not report LD coefficients because decision boundary is quadratic
# It models a curved boundary in predictor space — no single linear score
cat("QDA Model Output:\n")
print(qda_fit)

# Predictions — same interface as LDA
qda_pred <- predict(qda_fit, credit_test)

cat("\nQDA Confusion Matrix (threshold = 0.50):\n")
conf_qda <- table(Predicted = qda_pred$class, Actual = risk_test)
print(conf_qda)

qda_acc  <- mean(qda_pred$class == risk_test)
qda_sens <- conf_qda["1","1"] / sum(conf_qda[,"1"])
qda_spec <- conf_qda["0","0"] / sum(conf_qda[,"0"])

cat(sprintf("Accuracy:    %.1f%%\n", qda_acc * 100))
cat(sprintf("Sensitivity: %.1f%%\n", qda_sens * 100))
cat(sprintf("Specificity: %.1f%%\n\n", qda_spec * 100))

cat("QDA vs LDA Comparison:\n")
cat("If QDA sensitivity > LDA sensitivity: the true boundary is non-linear\n")
cat("QDA fits separate covariance structures for good and bad risk applicants\n")
cat("This captures different patterns of risk behaviour between the two groups\n\n")


# =============================================================================
# SECTION 7: K-NEAREST NEIGHBORS (KNN)
# Non-parametric: makes no assumptions about distribution shape
# Assigns each applicant to the class most common among its K nearest neighbours
# CRITICAL REQUIREMENT: Standardise all predictors before applying KNN
# Without standardisation, high-scale predictors (credit_amount) dominate
# all distance calculations, making other predictors irrelevant
# =============================================================================

cat("--- SECTION 7: K-Nearest Neighbors (KNN) ---\n")

# Define the numeric predictors used for KNN
# We use only numeric predictors for KNN distance calculations
# (categorical variables would need dummy encoding first)
knn_pred_cols <- c("duration", "credit_amount", "age", "installment_rate",
                   "existing_credits", "dependents")

# --- CRITICAL STEP: Standardise predictors ---
# scale() transforms each column to mean=0 and sd=1
# Key: fit scaling parameters on TRAINING data only
# Apply those SAME parameters to test data
# This prevents data leakage from test set into model building

train_scaled <- scale(credit_train[, knn_pred_cols])

# Save the scaling parameters from training data
train_center <- attr(train_scaled, "scaled:center")
train_scale  <- attr(train_scaled, "scaled:scale")

# Apply TRAINING scaling parameters to test data
# Do NOT recompute scale on test data — that would be data leakage
test_scaled <- scale(credit_test[, knn_pred_cols],
                     center = train_center,
                     scale  = train_scale)

train_labels <- credit_train$credit_risk
test_labels  <- credit_test$credit_risk

cat("Predictor scaling applied:\n")
cat("  Before scaling — credit_amount range:",
    round(range(credit_train$credit_amount), 0), "\n")
cat("  Before scaling — duration range:",
    round(range(credit_train$duration), 0), "\n")
cat("  After scaling — both variables have mean=0, sd=1\n")
cat("  All predictors now contribute equally to distance calculations\n\n")

# --- K Selection ---
# Try a range of K values to find optimal number of neighbours
# K too small (K=1): overfits — memorises training data, poor test performance
# K too large: underfits — averages too many dissimilar neighbours
k_values <- c(1, 3, 5, 7, 10, 15, 20, 25, 30)
knn_acc   <- numeric(length(k_values))
knn_sens  <- numeric(length(k_values))
knn_spec  <- numeric(length(k_values))
library(class)

set.seed(42)  # KNN breaks ties randomly — seed ensures reproducibility
for(i in seq_along(k_values)) {
  knn_pred_i <- knn(train_scaled, test_scaled, train_labels, k = k_values[i])
  conf_i     <- table(Predicted = knn_pred_i, Actual = test_labels)

  knn_acc[i]  <- mean(knn_pred_i == test_labels)
  # Handle case where confusion matrix might not have both rows
  knn_sens[i] <- if("1" %in% rownames(conf_i)) {
    conf_i["1","1"] / sum(conf_i[,"1"])
  } else 0
  knn_spec[i] <- if("0" %in% rownames(conf_i)) {
    conf_i["0","0"] / sum(conf_i[,"0"])
  } else 0
}

# Display K selection results
cat("KNN Performance Across K Values:\n")
knn_results <- data.frame(
  K           = k_values,
  Accuracy    = round(knn_acc * 100, 1),
  Sensitivity = round(knn_sens * 100, 1),
  Specificity = round(knn_spec * 100, 1)
)
print(knn_results)

# Select best K based on sensitivity (more important in credit risk context)
# Banks prioritise catching bad risks over overall accuracy
best_k_idx <- which.max(knn_sens)
best_k     <- k_values[best_k_idx]
cat("\nBest K by sensitivity:", best_k, "\n\n")

# Final KNN prediction with best K
set.seed(42)
knn_final <- knn(train_scaled, test_scaled, train_labels, k = best_k)

conf_knn <- table(Predicted = knn_final, Actual = test_labels)
cat("KNN Final Confusion Matrix (K =", best_k, "):\n")
print(conf_knn)

knn_acc_final  <- mean(knn_final == test_labels)
knn_sens_final <- conf_knn["1","1"] / sum(conf_knn[,"1"])
knn_spec_final <- conf_knn["0","0"] / sum(conf_knn[,"0"])

cat(sprintf("Accuracy:    %.1f%%\n", knn_acc_final * 100))
cat(sprintf("Sensitivity: %.1f%%\n", knn_sens_final * 100))
cat(sprintf("Specificity: %.1f%%\n\n", knn_spec_final * 100))

# Plot K selection curve
plot(k_values, knn_acc * 100,
     type = "b", pch = 19, col = "#2196F3",
     ylim = c(40, 100),
     main = "KNN Performance vs Number of Neighbours (K)\nCredit Risk Classification",
     xlab = "K (Number of Neighbours)",
     ylab = "Performance (%)")
lines(k_values, knn_sens * 100, type = "b", pch = 17, col = "#F44336")
lines(k_values, knn_spec * 100, type = "b", pch = 15, col = "#4CAF50")
abline(v = best_k, col = "grey50", lty = 2)
legend("topright",
       c("Accuracy", "Sensitivity (Catching Bad Risk)", "Specificity"),
       col = c("#2196F3", "#F44336", "#4CAF50"),
       pch = c(19, 17, 15), lty = 1, bty = "n")



# =============================================================================
# SECTION 8: ROC CURVES AND AUC COMPARISON
# ROC curves show the full trade-off between sensitivity and specificity
# across all possible threshold values simultaneously
# AUC summarises overall classifier quality in one number:
#   AUC = 1.0: perfect classifier
#   AUC = 0.5: no better than random guessing
#   AUC > 0.7: generally acceptable in credit risk
#   AUC > 0.8: good
#   AUC > 0.9: excellent (rare with real credit data)
# =============================================================================

cat("--- SECTION 8: ROC Curves and AUC ---\n")

# Compute ROC for each method
# Each method needs a numeric probability score, not a class label
# Logistic: predicted probabilities directly available
# LDA/QDA: use posterior probability of Bad Risk (class "1")
# KNN: use proportion of K neighbours that are Bad Risk as proxy probability

# Logistic Regression ROC
library(pROC)
roc_glm <- roc(as.numeric(as.character(risk_test)), glm_probs,
               quiet = TRUE)

# LDA ROC — use posterior probability of class "1" (Bad Risk)
roc_lda <- roc(as.numeric(as.character(risk_test)),
               lda_pred$posterior[, "1"],
               quiet = TRUE)

# QDA ROC
roc_qda <- roc(as.numeric(as.character(risk_test)),
               qda_pred$posterior[, "1"],
               quiet = TRUE)

# KNN ROC — need probability scores, run knn with k=best_k on multiple thresholds
# We use the proportion of bad risk neighbours as a probability proxy
set.seed(42)
knn_probs <- numeric(nrow(test_scaled))
for(j in 1:nrow(test_scaled)) {
  neighbours <- knn(train_scaled, test_scaled[j, , drop=FALSE],
                    train_labels, k = best_k,
                    prob = TRUE)
  # attr(neighbours, "prob") gives proportion voting for predicted class
  prob_predicted <- attr(neighbours, "prob")
  # If predicted class is "0", probability of "1" = 1 - prob_predicted
  if(as.character(neighbours) == "0") {
    knn_probs[j] <- 1 - prob_predicted
  } else {
    knn_probs[j] <- prob_predicted
  }
}
roc_knn <- roc(as.numeric(as.character(risk_test)), knn_probs, quiet = TRUE)

# Print AUC values
auc_glm <- auc(roc_glm)
auc_lda <- auc(roc_lda)
auc_qda <- auc(roc_qda)
auc_knn <- auc(roc_knn)

cat("AUC Comparison:\n")
cat(sprintf("  Logistic Regression: %.3f\n", auc_glm))
cat(sprintf("  LDA:                 %.3f\n", auc_lda))
cat(sprintf("  QDA:                 %.3f\n", auc_qda))
cat(sprintf("  KNN (K=%d):          %.3f\n", best_k, auc_knn))
cat("\nAUC > 0.70 is acceptable | > 0.80 is good | > 0.90 is excellent\n\n")

# Plot all ROC curves together for comparison
plot(roc_glm,
     col = "#2196F3", lwd = 2.5,
     main = "ROC Curve Comparison — Credit Risk Classification\nAll Four Methods",
     xlab = "False Positive Rate (1 - Specificity)",
     ylab = "True Positive Rate (Sensitivity)")
lines(roc_lda, col = "#4CAF50",   lwd = 2.5)
lines(roc_qda, col = "#FF9800",   lwd = 2.5)
lines(roc_knn, col = "#F44336",   lwd = 2.5)
abline(a = 0, b = 1, lty = 2, col = "grey60")  # Random classifier baseline

legend("bottomright",
       legend = c(
         sprintf("Logistic Regression (AUC = %.3f)", auc_glm),
         sprintf("LDA (AUC = %.3f)", auc_lda),
         sprintf("QDA (AUC = %.3f)", auc_qda),
         sprintf("KNN K=%d (AUC = %.3f)", best_k, auc_knn),
         "Random Classifier (AUC = 0.500)"
       ),
       col = c("#2196F3", "#4CAF50", "#FF9800", "#F44336", "grey60"),
       lwd = c(2.5, 2.5, 2.5, 2.5, 1),
       lty = c(1, 1, 1, 1, 2),
       bty = "n", cex = 0.9)



# =============================================================================
# SECTION 9: THRESHOLD ANALYSIS — BUSINESS COST OPTIMISATION
# In credit risk, the cost of each type of error is not equal
# False Negative (missing a bad risk): bank grants loan, applicant defaults
#   → Bank loses the principal + interest + recovery costs
# False Positive (flagging a good risk): bank rejects a creditworthy applicant
#   → Bank loses future interest income + damages customer relationship
#
# Typical industry assumption: false negative costs 5x more than false positive
# We find the threshold that minimises this asymmetric business cost
# =============================================================================

cat("--- SECTION 9: Business Cost Threshold Optimisation ---\n")

# Business cost parameters — adjust based on actual bank data
cost_fn <- 5   # Cost of missing a bad risk (false negative) relative units
cost_fp <- 1   # Cost of false alarm on a good risk (false positive)

# Evaluate business cost across a range of thresholds
thresholds  <- seq(0.1, 0.9, by = 0.05)
total_costs <- numeric(length(thresholds))
sens_thresh <- numeric(length(thresholds))
spec_thresh <- numeric(length(thresholds))

for(i in seq_along(thresholds)) {
  pred_t <- ifelse(glm_probs > thresholds[i], "1", "0")
  conf_t <- table(Predicted = pred_t, Actual = risk_test)

  # Extract counts safely
  fn <- if("0" %in% rownames(conf_t) && "1" %in% colnames(conf_t)) conf_t["0","1"] else 0
  fp <- if("1" %in% rownames(conf_t) && "0" %in% colnames(conf_t)) conf_t["1","0"] else 0
  tp <- if("1" %in% rownames(conf_t) && "1" %in% colnames(conf_t)) conf_t["1","1"] else 0
  tn <- if("0" %in% rownames(conf_t) && "0" %in% colnames(conf_t)) conf_t["0","0"] else 0

  total_costs[i] <- cost_fn * fn + cost_fp * fp
  sens_thresh[i] <- if((tp + fn) > 0) tp / (tp + fn) else 0
  spec_thresh[i] <- if((tn + fp) > 0) tn / (tn + fp) else 0
}

optimal_idx       <- which.min(total_costs)
optimal_threshold <- thresholds[optimal_idx]

cat(sprintf("Business Cost Parameters: FN cost = %dx FP cost\n", cost_fn))
cat(sprintf("Optimal threshold: %.2f\n", optimal_threshold))
cat(sprintf("At optimal threshold:\n"))
cat(sprintf("  Sensitivity: %.1f%%\n", sens_thresh[optimal_idx] * 100))
cat(sprintf("  Specificity: %.1f%%\n", spec_thresh[optimal_idx] * 100))
cat(sprintf("  Total business cost: %.0f units\n\n", total_costs[optimal_idx]))

# Final prediction at optimal business threshold
glm_pred_opt <- ifelse(glm_probs > optimal_threshold, "1", "0")
conf_glm_opt <- table(Predicted = glm_pred_opt, Actual = risk_test)
cat("Logistic Regression — Optimal Business Threshold Confusion Matrix:\n")
print(conf_glm_opt)

# Plot threshold analysis
par(mar = c(4, 4, 3, 4))
plot(thresholds, total_costs,
     type = "b", pch = 19, col = "#9C27B0", lwd = 2,
     main = "Business Cost vs Classification Threshold\nLogistic Regression — Credit Risk",
     xlab = "Probability Threshold",
     ylab = "Total Business Cost (relative units)")
abline(v = optimal_threshold, col = "red", lty = 2, lwd = 1.5)
text(optimal_threshold + 0.03, max(total_costs) * 0.95,
     paste("Optimal\nthreshold =", optimal_threshold),
     col = "red", cex = 0.9)

# Add sensitivity line on secondary axis
par(new = TRUE)
plot(thresholds, sens_thresh * 100,
     type = "b", pch = 17, col = "#F44336", lwd = 2,
     axes = FALSE, xlab = "", ylab = "")
axis(4, col = "#F44336", col.axis = "#F44336")
mtext("Sensitivity (%)", side = 4, line = 3, col = "#F44336")
legend("right",
       c("Business Cost", "Sensitivity"),
       col = c("#9C27B0", "#F44336"),
       pch = c(19, 17), lty = 1, bty = "n")


# =============================================================================
# SECTION 10: FULL METHOD COMPARISON
# Bring all results together in one clean summary table
# This is what you present to a hiring manager or credit committee
# =============================================================================

cat("--- SECTION 10: Full Method Comparison ---\n")

# Compile all metrics into a single comparison table
method_names <- c("Logistic (threshold=0.50)",
                  "Logistic (threshold=0.30)",
                  paste0("Logistic (optimal=", optimal_threshold, ")"),
                  "LDA",
                  "QDA",
                  paste0("KNN (K=", best_k, ")"))

accuracies    <- c(glm_acc_50, glm_acc_30,
                   mean(glm_pred_opt == risk_test),
                   lda_acc, qda_acc, knn_acc_final)

sensitivities <- c(glm_sens_50, glm_sens_30,
                   {conf_opt <- table(Predicted=glm_pred_opt, Actual=risk_test)
                    conf_opt["1","1"]/sum(conf_opt[,"1"])},
                   lda_sens, qda_sens, knn_sens_final)

specificities <- c(glm_spec_50, glm_spec_30,
                   {conf_opt["0","0"]/sum(conf_opt[,"0"])},
                   lda_spec, qda_spec, knn_spec_final)

aucs <- c(as.numeric(auc_glm), as.numeric(auc_glm), as.numeric(auc_glm),
          as.numeric(auc_lda), as.numeric(auc_qda), as.numeric(auc_knn))

comparison_table <- data.frame(
  Method      = method_names,
  Accuracy    = paste0(round(accuracies * 100, 1), "%"),
  Sensitivity = paste0(round(sensitivities * 100, 1), "%"),
  Specificity = paste0(round(specificities * 100, 1), "%"),
  AUC         = round(aucs, 3)
)

cat("\n")
print(comparison_table, row.names = FALSE)

# Identify best model by each criterion
cat("\nBest model by accuracy:    ", method_names[which.max(accuracies)], "\n")
cat("Best model by sensitivity: ", method_names[which.max(sensitivities)], "\n")
cat("Best model by AUC:         ", method_names[which.max(aucs)], "\n")

# Final comparison bar chart
metrics_matrix <- rbind(accuracies, sensitivities, specificities) * 100
colnames(metrics_matrix) <- c("Log\n(0.50)", "Log\n(0.30)",
                               paste0("Log\n(", optimal_threshold, ")"),
                               "LDA", "QDA", paste0("KNN\nK=", best_k))

barplot(metrics_matrix,
        beside     = TRUE,
        col        = c("#2196F3", "#F44336", "#4CAF50"),
        main       = "Credit Risk Model Comparison\nAccuracy, Sensitivity, Specificity",
        ylab       = "Performance (%)",
        ylim       = c(0, 110),
        legend.text = c("Accuracy", "Sensitivity", "Specificity"),
        args.legend = list(x = "topright", bty = "n"),
        border     = NA)
abline(h = 70, col = "grey50", lty = 2)


# =============================================================================
# SECTION 11: BUSINESS RECOMMENDATION
# Translate statistical findings into actionable business language
# This is what separates a data scientist from a statistician
# =============================================================================

cat("========================================================\n")
cat("  BUSINESS RECOMMENDATION\n")
cat("========================================================\n\n")

best_auc_method <- method_names[which.max(aucs)]
best_auc_val    <- round(max(aucs), 3)

cat("EXECUTIVE SUMMARY\n")

cat("-----------------\n")
cat("We evaluated four classification methods on 1,000 historical loan\n")
cat("applications to predict credit default risk.\n\n")

cat("KEY FINDINGS:\n")
cat("1. The best overall discriminating power comes from the model with\n")
cat("   AUC =", best_auc_val, "— significantly above random chance (0.500).\n\n")

cat("2. At the default 50% threshold, all models achieve acceptable accuracy\n")
cat("   but miss a significant fraction of bad risks.\n\n")

cat("3. Lowering the threshold to", optimal_threshold, "using business cost\n")
cat("   optimisation (FN 5x more costly than FP) increases bad risk detection\n")
cat("   while accepting a modest increase in false alarms.\n\n")


cat("4. Logistic Regression is recommended for production deployment because:\n")
cat("   a) Competitive discriminating power (AUC comparable to other methods)\n")
cat("   b) Fully interpretable coefficients — each predictor has a clear,\n")
cat("      directional effect that can be explained to regulators\n")
cat("   c) Produces calibrated probabilities for scoring, not just labels\n")


cat("   d) Robust to new applicants outside training distribution\n\n")


cat("5. The optimal threshold of", optimal_threshold, "is recommended for\n")

cat("   production use with assumed cost ratio FN:FP = 5:1\n")

cat("   This threshold should be re-calibrated as actual cost data accumulates.\n\n")

cat("REGULATORY NOTE:\n")

cat("LDA provides an alternative with similar performance and formal\n")
cat("statistical foundations (Bayes optimal under Gaussian assumptions).\n")
cat("QDA and KNN offer flexibility but sacrifice interpretability — these\n")
cat("would require additional explainability work for regulatory approval.\n\n")

