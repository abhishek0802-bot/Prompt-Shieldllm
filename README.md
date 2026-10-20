# PromptShield Experimental Pipeline

## Dataset
Download the official `prompt_injection_dataset.csv` from:

https://github.com/mirzaakhi/prompt-injection-detection-dataset

Place the CSV in this folder next to `run_experiment.py`.

The dataset documentation says to preserve its official `train`/`test` split and not replace it with a random split.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python run_experiment.py
```

## Outputs

The script creates:

- `results/dataset_integrity.json`
- `results/overall_metrics.csv`
- `results/category_metrics.csv`
- `results/hard_negative_metrics.csv`
- `results/test_predictions.csv`
- `results/frozen_thresholds.json`
- trained `.joblib` model/vectorizer files

## Experimental models

1. Rule/regex baseline
2. Word TF-IDF + Logistic Regression
3. Character TF-IDF + Linear SVM
4. PromptShield Hybrid: word + character TF-IDF + Logistic Regression

## Important research rules

- Official test split is never used for training.
- Thresholds are selected only on validation data and then frozen.
- Do not tune thresholds separately on the test categories.
- Keep the official dataset unchanged.
- Do not report fabricated results; only use generated experiment outputs.
