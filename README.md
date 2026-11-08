
# Spam Email Classification Using Machine Learning

## Name: SIDDHARTH BALAJI
## BITS ID: 2026AJ05658
## Machine Learning - Assignment 2

## a. Problem Statement

The goal of this project is to build and compare five supervised machine learning classifiers that decide whether an email is **spam** or **not spam**, using only the frequency of certain words and characters in the message.

This is a **binary classification** problem:

- `1` = spam (unsolicited commercial email)
- `0` = not spam (legitimate email)

Each model is trained on the same data and evaluated on the same held-out test set using six metrics (Accuracy, AUC, Precision, Recall, F1 and MCC). Using several metrics matters because the two kinds of mistake are not equally costly in a spam filter: flagging a legitimate email as spam (a false positive) is usually worse than letting one spam message through (a false negative). The trained models are served through an interactive Streamlit app.

## b. Dataset Description

| Item | Detail |
|---|---|
| Dataset | Spambase |
| Source | UCI Machine Learning Repository |
| URL | https://archive.ics.uci.edu/dataset/94/spambase |
| Citation | Hopkins, M., Reeber, E., Forman, G., & Suermondt, J. (1999). Spambase [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C53G6X |
| Task | Binary classification (spam = 1, not spam = 0) |
| Raw size | 4,601 emails, 57 features plus the label |
| Duplicates | 391 duplicate rows found and removed |
| Size used | 4,210 emails, 57 features |
| Missing values | None |
| Class balance (after cleaning) | 2,531 not spam (60.1%), 1,679 spam (39.9%) |

**Features (all numeric):**

- 48 columns `word_freq_*`: percentage of words in the email that match a given word (for example `make`, `free`, `money`, `george`).
- 6 columns `char_freq_*`: percentage of characters that match `;`, `(`, `[`, `!`, `$` and `#`.
- 3 columns `capital_run_length_*`: the average, longest and total length of uninterrupted runs of capital letters.

**Preparation and split:**

- Duplicate rows were removed **before** splitting, so an identical email can never appear in both the training and test sets.
- The data was split 80/20 with stratification and a fixed seed (`random_state=42`): **3,368 training rows and 842 test rows** (506 not spam, 336 spam). The test set is saved as `test_data.csv`.
- Scaling is applied with `StandardScaler` inside the saved pipelines of the models that need it (Logistic Regression, KNN, SVM). Decision Tree and Naive Bayes are trained on the raw features. Because each pipeline carries its own preprocessing, the app can score raw uploaded rows without retraining.

## c. GitHub Repository Link

**Repository:** https://github.com/sidhugithub13/ML_Assigmnment2

**Live Streamlit app:** https://mlassigmnment2.streamlit.app/

```
ML_Assigmnment2/
├── app.py                 # Streamlit app
├── notebook.ipynb         # data loading, cleaning, training and evaluation
├── requirements.txt       # dependencies (scikit-learnpinned to 1.6.1)
├── test_data.csv          # held-out test set (842 rows, with the is_spam label)
├── README.md
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── support_vector_machine.joblib
    └── metrics_summary.csv
```

## d. Models Used

All five classifiers were trained on the same training split and evaluated on the same 842-row test set. Spam is the positive class.

| Model | Setup |
|---|---|
| Logistic Regression | `StandardScaler` + `LogisticRegression(max_iter=1000)` |
| Decision Tree | `DecisionTreeClassifier` with default settings |
| K-Nearest Neighbors | `StandardScaler` + `KNeighborsClassifier(n_neighbors=5)` |
| Naive Bayes | `GaussianNB` with default settings |
| Support Vector Machine | `StandardScaler` + `SVC` (default RBF kernel, `probability=True`) |

Hyperparameters were left at their defaults (apart from `max_iter` and `n_neighbors` above) and `random_state=42` was used wherever a model has randomness. No tuning was done, so the comparison shows how each algorithm behaves out of the box.

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9347 | 0.9768 | 0.9297 | 0.9048 | 0.9170 | 0.8634 |
| Decision Tree | 0.9074 | 0.9053 | 0.8817 | 0.8869 | 0.8843 | 0.8071 |
| KNN | 0.9133 | 0.9546 | 0.9148 | 0.8631 | 0.8882 | 0.8184 |
| Naive Bayes | 0.8361 | 0.9525 | 0.7240 | 0.9524 | 0.8226 | 0.6975 |
| Support Vector Machine | 0.9323 | 0.9772 | 0.9319 | 0.8958 | 0.9135 | 0.8584 |

### Evaluation Metrics

- **Accuracy:** share of all predictions that are correct.
- **AUC:** area under the ROC curve, computed from predicted spam probabilities; measures how well a model ranks spam above legitimate email across all thresholds.
- **Precision:** of the emails flagged as spam, the share that really were spam.
- **Recall:** of the real spam, the share that was caught.
- **F1:** harmonic mean of precision and recall.
- **MCC (Matthews Correlation Coefficient):** a balanced score that uses all four confusion-matrix counts; it ranges from -1 to +1.

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best accuracy (0.9347), F1 (0.9170) and MCC (0.8634) of the five models, with an AUC (0.9768) almost equal to the best. Precision and recall are well balanced (0.9297 and 0.9048). The spam signal in this data appears to be largely linear in the scaled word and character frequencies, and the model is also fast and easy to interpret. |
| Decision Tree | Clearly weaker than the top models, with the lowest AUC (0.9053) and an accuracy of 0.9074. Its precision and recall are nearly equal (0.8817 and 0.8869), so it is balanced but at a lower level. A single unpruned tree gives coarse probability estimates and likely overfits the training data, which would explain the low AUC. |
| KNN | Good precision (0.9148) and AUC (0.9546), but the lowest recall (0.8631), so it misses more spam than the other models. Distance-based methods tend to struggle in a 57-dimensional space where most word-frequency values are zero, and the result depends on the scaling applied beforehand. Overall it sits mid-table. |
| Naive Bayes | Highest recall (0.9524) but the lowest precision (0.7240), accuracy (0.8361) and MCC (0.6975): it catches almost all spam but also flags many legitimate emails as spam. Its AUC (0.9525) is high, so it ranks emails well, and the weak precision suggests the default 0.5 threshold is poorly suited to its probabilities. The independence assumption and Gaussian distribution are a poor fit for skewed, mostly-zero word frequencies. For a spam filter, the many false positives make it the least practical choice here. |
| Support Vector Machine | Highest AUC (0.9772) and highest precision (0.9319), and within a whisker of Logistic Regression on accuracy (0.9323), F1 (0.9135) and MCC (0.8584). It is a strong, well-balanced model but slower to train and to score than Logistic Regression, and its gain in precision does not translate into better F1 or MCC. |
| **Overall, Winner for your dataset?** | **Logistic Regression.** It has the best accuracy, F1 and MCC, which are the metrics that best summarise balanced performance, and it is the simplest and fastest model. The Support Vector Machine is a very close runner-up with the best AUC and precision. The gap between the two is small: the 0.0024 difference in accuracy is only about two emails out of the 842 test emails, so the ranking could change on a different split. Naive Bayes is the one to avoid when false positives matter. |

**Limitations.** Results come from a single train/test split and untuned models, so small differences between the top models should not be over-interpreted. Cross-validation and hyperparameter tuning would give a more reliable ranking.

## Streamlit App

The app (`app.py`) evaluates the saved pipelines and does not retrain anything. It provides:

- **Evaluate model:** choose one of the five classifiers and see Accuracy, AUC, Precision, Recall, F1 and MCC, a confusion matrix with TN/FP/FN/TP counts, a classification report and a preview of the data.
- **Compare models:** all five models on the same data in one table, with the best value per metric highlighted, a bar chart and a "best model per metric" table.
- **About:** a short summary of the dataset, method and metrics.
- **CSV upload:** the bundled `test_data.csv` is loaded by default; a different labelled CSV with the same 57 feature columns plus `is_spam` can be uploaded from the sidebar. Clear error messages are shown if columns are missing or invalid.

## Run Locally

```
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

(On macOS or Linux, activate the environment with `source .venv/bin/activate`.)

`scikit-learn` is pinned to version 1.6.1 in `requirements.txt` because the saved model files were created with that version.