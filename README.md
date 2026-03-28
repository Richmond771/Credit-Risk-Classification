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

Method Accuracy Sensitivity Specificity AUC Logistic (threshold = 0.50)~ 75% ~ 50% ~ 88% ~ 0.78 Logistic (optimal threshold)

~70% ~ 70% ~ 72% ~ 0.78 LDA~ 75% ~ 50% ~ 88% ~ 0.78 QDA ~ 73% ~ 55% ~ 83% ~ 0.77 KNN 

(best K) ~ 70% ~ 60% ~ 75% ~ 0.72

AUC above 0.75 is generally considered good for credit scoring applications.

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
