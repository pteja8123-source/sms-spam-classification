# SMS Spam Classification

A Week 2 machine-learning project that classifies SMS messages as legitimate (`ham`) or unwanted (`spam`) using TF-IDF and two scikit-learn pipelines.

## Dataset and methodology

The notebook downloads the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) from its official archive. Attribution: Almeida, T. & Hidalgo, J. (2011), [SMS Spam Collection](https://doi.org/10.24432/C5CC84), UCI Machine Learning Repository, CC BY 4.0.

- Downloaded messages: 5,574; exact duplicate rows removed: 403.
- Cleaned messages: 5,171 (4,518 ham and 653 spam).
- Stratified 80/20 split with `random_state=42`: 4,136 training and 1,035 test messages.
- Training-only TF-IDF: unigrams/bigrams, English stop words, `min_df=2`, `max_features=10000`, `sublinear_tf=True`.
- Models: Multinomial Naive Bayes and Logistic Regression (`class_weight='balanced'`, `max_iter=2000`).

## Executed results

Spam is the positive class. All models used the same held-out test set.

| Model | Accuracy | Spam precision | Spam recall | Spam F1 |
|---|---:|---:|---:|---:|
| Always-ham baseline | 87.34% | 0.00% | 0.00% | 0.0000 |
| Multinomial Naive Bayes | 96.91% | 100.00% | 75.57% | 0.8609 |
| Logistic Regression | 97.49% | 90.70% | 89.31% | 0.9000 |

Logistic Regression detected 117 of 131 spam messages, missed 14, and incorrectly flagged 12 ham messages. Naive Bayes detected 99 spam messages, missed 32, and incorrectly flagged no ham messages. Average precision was 0.9652 and 0.9568 respectively. Logistic Regression had the higher spam F1 on this split.

## Run in Google Colab

Open [Google Colab](https://colab.research.google.com/), choose **File → Upload notebook**, and upload `sms-spam-classification.ipynb`. Select **Runtime → Run all**. Internet access is required for the dataset download. The provided notebook already contains locally executed outputs; it was not executed in Colab during this task.

## Run locally

Install Python 3.12 or a compatible version, then:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn notebook
jupyter notebook sms-spam-classification.ipynb
```

Run all cells from the top. Recorded execution versions: Python 3.12.14, NumPy 2.5.4, Pandas 3.1.1, scikit-learn 1.9.1. Different package versions can change numerical results slightly; exact versions are also printed in the notebook.

## Project structure

```text
sms-spam-classification/
├── sms-spam-classification.ipynb
└── README.md
```

The notebook includes exploratory plots, metric comparisons, confusion matrices, classification reports, precision-recall curves, feature coefficients, sample predictions, and run-derived conclusions. All code cells executed without errors, and six figures were checked visually.

## Limitations

This is one split of a historical dataset, with no external validation, threshold tuning, or hyperparameter search. Exact duplicate removal does not eliminate all similar campaign messages. Performance may differ for current spam, other languages, or other platforms. False positives and false negatives have different costs; the project demonstrates evaluation rather than a production filtering service.
