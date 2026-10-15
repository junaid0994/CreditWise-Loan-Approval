# CreditWise – Loan Approval Prediction

A machine learning project that predicts whether a loan application will be **approved or rejected** based on the applicant's financial and personal profile. It covers the full workflow: cleaning, EDA, encoding, feature engineering, and comparing three classifiers.

## Dataset

- 1,000 loan applications, 20 columns (the target is `Loan_Approved`: Yes/No)
- Features include applicant and co-applicant income, credit score, DTI ratio, savings, collateral value, loan amount and term, employment status, education, property area, and more
- About 5% of values are missing in most columns
- **The CSV is not included in this repo.** Download or place `loan_approval_data.csv` inside a `data/` folder before running the notebook:

```
CreditWise-Loan-Approval/
├── data/
│   └── loan_approval_data.csv   <- add this yourself (git-ignored)
├── Credit_Wise.ipynb
├── requirements.txt
└── README.md
```

## Workflow

1. **Data cleaning** – mean imputation for numeric columns, most-frequent imputation for categorical columns; dropped `Applicant_ID`
2. **EDA** – target distribution, gender/education/employer category counts, income histograms, box plots and credit score vs. approval, correlation heatmap
3. **Encoding** – `LabelEncoder` for education and target, `OneHotEncoder` for the other categorical columns
4. **Preprocessing** – 80/20 train-test split, `StandardScaler`
5. **Modeling** – Logistic Regression, Gaussian Naive Bayes, KNN (tuned with `GridSearchCV`)
6. **Feature engineering** – added squared terms `Credit_Score_sq` and `DTI_Ratio_sq`

## Results

Credit score is the strongest predictor of approval (correlation ≈ 0.45).

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.865 | 0.783 | 0.770 | 0.777 |
| Naive Bayes (baseline) | 0.865 | 0.804 | 0.738 | 0.769 |
| KNN (k=9, baseline) | 0.760 | 0.659 | 0.443 | – |
| **Logistic Regression (+ feature engineering)** | **0.880** | 0.785 | **0.836** | **0.810** |
| Naive Bayes (+ feature engineering) | 0.860 | **0.811** | 0.705 | 0.754 |
| KNN (k=9, + feature engineering) | 0.775 | 0.682 | 0.492 | – |

Feature engineering improved Logistic Regression the most, giving the best overall balance of accuracy, recall and F1. Naive Bayes has the highest precision, which matters if the goal is to avoid approving risky loans.

## Tech Stack

Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · Jupyter Notebook

## How to Run

```bash
git clone https://github.com/<your-username>/CreditWise-Loan-Approval.git
cd CreditWise-Loan-Approval
pip install -r requirements.txt
# put loan_approval_data.csv inside the data/ folder
jupyter notebook Credit_Wise.ipynb
```

## Future Improvements

- Add Random Forest / XGBoost and compare
- Use cross-validation for all models, not just KNN
- Handle class imbalance (SMOTE or class weights)
- Wrap preprocessing in a single `Pipeline` and deploy with Streamlit/Flask

## Author

**<Your Name>** – [GitHub](https://github.com/<your-username>) · [LinkedIn](https://linkedin.com/in/<your-handle>)
