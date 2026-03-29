Credit Risk Classification

Can we predict who will default on a loan before it happens?

Author: Richmond Osei

Dataset: German Credit Data  1,000 real loan applicants (UCI Machine Learning Repository)

https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)

Methods: Logistic Regression · LDA · QDA · KNN

Built with: R 


Live Report
View the full analysis here

https://richmond771.github.io/Credit-Risk-Classification/credit_risk_analysis--1-.html

The report walks through the entire project end to end from raw data exploration to a final business recommendation  with all code, plots, and interpretation visible. Best viewed on 
desktop.


The Problem

Every time a bank approves a loan, it is making a bet. The applicant looks creditworthy on paper  but will they actually repay? Getting this wrong in either direction costs money:

Miss a genuine defaulter  the bank takes the loss when the loan goes bad

Reject a good customer  the bank loses future interest income and the relationship


What makes this interesting statistically is that these two mistakes are not equally costly. A missed default tends to hurt significantly more than a false alarm. That asymmetry  

changes everything about how you build and evaluate the model and it is exactly what this project explores.

What I Built

I compared four classification methods on 1,000 historical loan applications from the German Credit dataset, a standard benchmark in quantitative finance research:

Method Key Idea Decision Boundary Logistic Regression Models probability of default directly LinearLinear Discriminant Analysis Uses Bayes theorem plus normality 

assumption Linear Quadratic Discriminant Analysis Relaxes the shared covariance constraint Quadratic K-Nearest Neighbors No assumptions pure distance based voting Non-parametric

The analysis goes beyond just fitting models. It covers confounding (where a predictor flips sign when you add another variable), threshold optimisation (finding the cutoff that 

minimises business cost rather than overall error), and ROC curves that show performance across every possible threshold simultaneously.

Results

| Method | Accuracy | Sensitivity | Specificity | AUC |
|--------|----------|-------------|-------------|-----|
| Logistic (threshold = 0.50) | ~75% | ~50% | ~88% | ~0.78 |
| Logistic (optimal threshold) | ~70% | ~70% | ~72% | ~0.78 |
| LDA | ~75% | ~50% | ~88% | ~0.78 |
| QDA | ~73% | ~55% | ~83% | ~0.77 |
| KNN (best K) | ~70% | ~60% | ~75% | ~0.72 |


AUC above 0.75 is generally considered good for credit scoring applications.

Key Findings

1. The default threshold is the wrong threshold.
Every model was first evaluated at the standard 50% probability cutoff. At that level, logistic regression caught only about half of the actual bad risks  meaning the bank would still approve loans for roughly 1 in 2 applicants who will later default. Lowering the threshold to the business optimal level pushed that detection rate up to 70%, which is a meaningful improvement for a real lending operation.

2. Checking account status is the single strongest predictor.

Applicants with no checking account or a consistently overdrawn account were significantly more likely to default  more so than loan size, income, or employment history alone. This makes intuitive sense: how someone manages their day to day finances is a stronger signal of repayment behaviour than how much they are asking to borrow.

3. Longer loans are riskier, but not for the reason you might think.

On the surface, loan duration looks like a strong risk factor. But once you control for loan amount, the picture changes. Longer loans tend to be larger loans, and it is the size  not the duration  that drives much of the risk. Treating duration as an independent risk factor without accounting for amount would lead to misleading credit decisions. This is a classic case of confounding.

4. More complexity did not buy better performance.
 
QDA and KNN  the more flexible methods did not outperform logistic regression in this analysis. The relationship between the predictors and default risk appears to be approximately linear, which means the simpler model is the right one. Adding complexity without a corresponding improvement in accuracy only makes the model harder to explain and harder to defend.

5. The model is genuinely useful  but not perfect.

An AUC of ~0.78 means the model correctly ranks a randomly chosen bad risk above a randomly chosen good risk about 78% of the time. That is well above random chance (50%) and sufficient to meaningfully improve lending decisions. However, about 30% of bad risks will still be missed even at the optimal threshold. The model should be used as a decision-support tool alongside human judgment, not as a replacement for it.

The headline finding: logistic regression and LDA perform almost identically  which is exactly what the theory predicts for binary classification with approximately linear decision 

boundaries. Lowering the threshold from 0.50 to the business optimal value meaningfully improves how many bad risks the model catches, at the cost of a modest increase in false alarms
 a trade off most banks would willingly accept.
 
Recommended for production: Logistic regression with an optimised threshold. It matches the discriminating power of more complex methods while producing coefficients that can be explained to regulators, loan officers, and applicants.


A Few Things Worth Noting

On standardisation: 
KNN measures distances between applicants in predictor space. Without standardisation, loan amount (ranging from 250 to 18,000 DM) completely overwhelms age (19 to 75 years) just 

because of its larger numeric scale. Every KNN analysis requires this step  skipping it silently breaks the model.

On data leakage: The scaling parameters (mean and standard deviation) are computed on training data only and then applied to the test set. Recomputing them on the full dataset before 

splitting would let test information leak into the model a subtle but serious mistake that many practitioners make.

On thresholds:

The default 50% threshold is optimal for overall accuracy but not for business cost. When false negatives are more expensive than false positives, you need to lower the threshold and 

this project shows how to find that optimal point systematically.

Repository Structure

Credit-Risk-Classification/

│
├── credit_risk_report.Rmd   
# R Markdown source — renders the live report
├── credit_risk_analysis.R  
# Clean production R script with full comments
├── README.md  
# You are reading this

│
└── plots/            
# All saved visualisations
    ├── 01_class_distribution.png
    
    ├── 02_predictor_distributions.png
    
    ├── 03_knn_k_selection.png
    
    ├── 04_roc_curves.png
    
    ├── 05_threshold_analysis.png
    
    └── 06_method_comparison.png

How to Run It Yourself

r#
Install the required packages (one-time setup)

install.packages(c("MASS", "class", "pROC", "caret"))

# Option 1 — Run the analysis script directly

source("credit_risk_analysis.R")

# Option 2 — Render the full HTML report

rmarkdown::render("credit_risk_report.Rmd")


The data downloads automatically from the UCI Machine Learning Repository no manual setup needed.
different but the workflow is the same: explore, split, fit, evaluate honestly on held-out data, interpret for a real audience.
