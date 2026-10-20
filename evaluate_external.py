import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
)
from scipy.sparse import hstack


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

EXTERNAL_FILE = ROOT / "promptshield_external_test_clean.json"


# ============================================================
# LOAD EXTERNAL DATA
# ============================================================

with open(EXTERNAL_FILE, "r", encoding="utf-8") as f:
    external = json.load(f)

X_text = [r["prompt"] for r in external]
y_true = np.array([int(r["label"]) for r in external])


print("=" * 70)
print("PROMPTSHIELD EXTERNAL / OOD EVALUATION")
print("=" * 70)

print(f"External samples : {len(X_text)}")
print(f"Benign           : {int((y_true == 0).sum())}")
print(f"Malicious        : {int((y_true == 1).sum())}")


# ============================================================
# LOAD FROZEN ARTIFACTS
# ============================================================

word_vec = joblib.load(
    RESULTS / "word_tfidf.joblib"
)

char_vec = joblib.load(
    RESULTS / "char_tfidf.joblib"
)

lr = joblib.load(
    RESULTS / "logistic_regression.joblib"
)

svm = joblib.load(
    RESULTS / "linear_svm.joblib"
)

hybrid = joblib.load(
    RESULTS / "promptshield_hybrid.joblib"
)


with open(
    RESULTS / "frozen_thresholds.json",
    "r",
    encoding="utf-8"
) as f:
    thresholds = json.load(f)


print("\nFrozen thresholds:")
for name, value in thresholds.items():
    print(f"  {name:10s}: {value}")


# ============================================================
# EXACT RULE BASELINE FROM ORIGINAL EXPERIMENT
# ============================================================

RULE_PATTERNS = [
    r"\bignore\s+(all\s+)?(previous|prior|above)\s+instructions?\b",
    r"\bdisregard\s+(all\s+)?(previous|prior|above)\s+instructions?\b",
    r"\bforget\s+(all\s+)?(previous|prior|above)\s+instructions?\b",
    r"\b(system|developer)\s+prompt\b.*\b(reveal|show|print|give)\b",
    r"\breveal\s+(the\s+)?(system|hidden)\s+(prompt|instructions?)\b",
    r"\bpretend\s+you\s+are\b.*\b(system|developer)\b",
    r"\bdo\s+not\s+follow\b.*\binstructions?\b",
]

compiled_rules = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in RULE_PATTERNS
]


def rule_score(text):
    return int(any(pattern.search(text) for pattern in compiled_rules))


def rule_predict(texts):
    return np.array(
        [rule_score(text) for text in texts],
        dtype=int
    )


# ============================================================
# TRANSFORM EXTERNAL DATA USING FROZEN VECTORIZERS
# ============================================================

print("\nTransforming external prompts...")

X_word = word_vec.transform(X_text)
X_char = char_vec.transform(X_text)

X_hybrid = hstack(
    [X_word, X_char],
    format="csr"
)


# ============================================================
# GENERATE SCORES
# ============================================================

print("Generating model scores...")

scores = {}

# Word TF-IDF + Logistic Regression
scores["tfidf_lr"] = lr.predict_proba(X_word)[:, 1]

# Character TF-IDF + Linear SVM
scores["char_svm"] = svm.decision_function(X_char)

# Hybrid
scores["hybrid"] = hybrid.predict_proba(X_hybrid)[:, 1]

# Rules
scores["rules"] = rule_predict(X_text).astype(float)


# ============================================================
# METRIC FUNCTION
# ============================================================

def evaluate(name, score, threshold):

    prediction = (score >= threshold).astype(int)

    accuracy = accuracy_score(y_true, prediction)

    precision = precision_score(
        y_true,
        prediction,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        prediction,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        prediction,
        labels=[0, 1]
    ).ravel()

    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    try:
        pr_auc = average_precision_score(
            y_true,
            score
        )
    except Exception:
        pr_auc = np.nan

    return {
        "detector": name,
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "fnr": fnr,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "pr_auc": pr_auc,
    }


# ============================================================
# EVALUATE ALL DETECTORS
# ============================================================

results = []

for name in [
    "rules",
    "tfidf_lr",
    "char_svm",
    "hybrid",
]:

    result = evaluate(
        name,
        scores[name],
        thresholds[name]
    )

    results.append(result)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame(results)

metrics_file = RESULTS / "external_overall_metrics.csv"

metrics_df.to_csv(
    metrics_file,
    index=False
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

predictions = pd.DataFrame({
    "prompt": X_text,
    "true_label": y_true,

    "rules_score": scores["rules"],
    "rules_pred": (
        scores["rules"] >= thresholds["rules"]
    ).astype(int),

    "tfidf_lr_score": scores["tfidf_lr"],
    "tfidf_lr_pred": (
        scores["tfidf_lr"] >= thresholds["tfidf_lr"]
    ).astype(int),

    "char_svm_score": scores["char_svm"],
    "char_svm_pred": (
        scores["char_svm"] >= thresholds["char_svm"]
    ).astype(int),

    "hybrid_score": scores["hybrid"],
    "hybrid_pred": (
        scores["hybrid"] >= thresholds["hybrid"]
    ).astype(int),
})


predictions_file = RESULTS / "external_predictions.csv"

predictions.to_csv(
    predictions_file,
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("EXTERNAL DATASET RESULTS")
print("=" * 70)

display_columns = [
    "detector",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "fpr",
    "fnr",
    "pr_auc",
]

print(
    metrics_df[display_columns].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nSaved:")
print(f"  {metrics_file}")
print(f"  {predictions_file}")

print("\nExternal evaluation complete.")