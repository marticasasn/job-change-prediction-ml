# Predicting Job Change Among Data Science Trainees

Classical machine learning study on a tabular HR dataset: supervised classifiers with a robustness analysis, K-Means clustering, and ensemble methods, all built on one leakage-safe scikit-learn pipeline.

Individual coursework for ELEC3612 Pattern Recognition and Machine Intelligence, University of Sydney (2026).

## Problem

Given the profile of a person who completed a data science training course (city, education, experience, current company, training hours), predict whether they are looking for a job change (`target = 1`) or not (`target = 0`).

## Data

[HR Analytics: Job Change of Data Scientists](https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists) (Kaggle, `aug_train.csv`), included in `data/`.

- 19,158 rows, 14 columns (12 features after dropping `enrollee_id` and `target`)
- Class imbalance: 75.1% not looking, 24.9% looking
- Missing values in 8 columns, up to 32.0% (`company_type`) and 31.0% (`company_size`)
- Mixed types: 2 numerical, 2 ordinal and 8 nominal features

## Methods

**Preprocessing.** A `ColumnTransformer` inside a `Pipeline`, always fitted on the training split only:

| Column type | Columns | Steps |
|---|---|---|
| Numerical | `city_development_index`, `training_hours` | median imputation, `StandardScaler` |
| Ordinal | `experience`, `last_new_job` | most frequent imputation, `OrdinalEncoder` |
| Nominal | `city`, `gender`, `relevent_experience`, `enrolled_university`, `education_level`, `major_discipline`, `company_size`, `company_type` | most frequent imputation, `OneHotEncoder` |

This yields 159 features on the 80/20 training split (160 when fitted on the full data for clustering).

**Supervised models.** Decision Tree, Logistic Regression, Random Forest (50 to 200 trees) and SVM (RBF kernel). Decision Tree and Logistic Regression get a full robustness analysis:

- train/test split ratios 60/40, 70/30, 80/20
- stratified vs non-stratified split
- stratified k-fold CV with k = 3, 5, 10
- random seeds 42 and 123
- with and without `GridSearchCV` tuning (scored on weighted F1)

**Unsupervised.** K-Means with k = 2 (k-means++), checked with the elbow method and evaluated with Silhouette, ARI and AMI against the true labels.

**Ensembles.** Bagging (Decision Tree base), AdaBoost, Gradient Boosting, hard and soft Voting (DT + LR + RF) and Stacking (DT + LR + RF with a Logistic Regression meta-learner).

**Evaluation.** Accuracy and weighted precision, recall and F1. Unless stated otherwise, models are compared on the same stratified 80/20 split with seed 42.

## Key results

All models on the same 80/20 stratified test set (3,832 rows), ranked by weighted F1:

| Model | Type | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Decision Tree (tuned) | Base | 0.7863 | 0.7721 | 0.7863 | **0.7758** |
| Gradient Boosting | Ensemble | 0.7858 | 0.7719 | 0.7858 | 0.7757 |
| Stacking | Ensemble | 0.7813 | 0.7608 | 0.7813 | 0.7619 |
| Random Forest (200 trees) | Base | 0.7701 | 0.7541 | 0.7701 | 0.7590 |
| Bagging (DT) | Ensemble | 0.7675 | 0.7538 | 0.7675 | 0.7586 |
| SVM (RBF) | Base | 0.7784 | 0.7567 | 0.7784 | 0.7577 |
| Voting (hard) | Ensemble | 0.7704 | 0.7521 | 0.7704 | 0.7569 |
| Logistic Regression | Base | 0.7777 | 0.7548 | 0.7777 | 0.7547 |
| AdaBoost | Ensemble | 0.7769 | 0.7528 | 0.7769 | 0.7503 |
| Voting (soft) | Ensemble | 0.7518 | 0.7386 | 0.7518 | 0.7438 |
| Decision Tree (no tuning) | Base | 0.7035 | 0.7079 | 0.7035 | 0.7056 |

Takeaways:

- **Tuning matters most for the Decision Tree.** Weighted F1 goes from 0.7056 to 0.7758 with `criterion=entropy`, `max_depth=3`, `min_samples_split=2`. On the minority class, false positives drop from 589 to 295 and true positives rise from 408 to 431.
- **Logistic Regression is the most stable model.** Weighted F1 stays between 0.7538 and 0.7583 across all split ratios and seeds, CV F1 between 0.7480 and 0.7486 for k = 3, 5, 10, and tuning leaves the result unchanged.
- **Ensembles do not clearly beat the best single model.** Gradient Boosting matches the tuned tree (0.7757 vs 0.7758); soft voting is the weakest ensemble.
- **Stratification has little effect here.** The class 1 share in the test set is 0.2492 stratified vs 0.2484 non-stratified.
- **K-Means does not recover the labels.** Silhouette 0.5277, but ARI -0.0180 and AMI 0.0034, so the two clusters are unrelated to job-seeking behaviour.

The full analysis, all tables and the discussion of limitations (minority class recall, high-cardinality `city` encoding) are in the [report](report/ELEC3612_Assignment1_Report_MartiCasas.pdf).

## Repository structure

```
.
├── data/
│   └── aug_train.csv                  # Kaggle dataset (CC0)
├── notebooks/
│   ├── 01_preprocessing.ipynb         # EDA and preprocessing pipeline
│   ├── 02_supervised_models.ipynb     # DT, LR, RF, SVM and robustness analysis
│   ├── 03_unsupervised_models.ipynb   # K-Means, elbow method, Silhouette, ARI, AMI
│   └── 04_ensemble_models.ipynb       # Bagging, boosting, voting, stacking
├── report/
│   └── ELEC3612_Assignment1_Report_MartiCasas.pdf
└── requirements.txt
```

Each notebook is self-contained: it loads the CSV and rebuilds the same preprocessing pipeline, so they can be run in any order.

## How to reproduce

The original runs used Python 3.13.

```bash
git clone https://github.com/marticasasn/job-change-prediction-ml.git
cd job-change-prediction-ml
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd notebooks
jupyter lab
```

Run the notebooks from inside `notebooks/`, since they read the data from `../data/aug_train.csv`. All splits and models use fixed seeds (42, plus 123 in the seed analysis). Exact figures can vary slightly across scikit-learn versions.
