![PromptShield LLM Banner](promptshield-banner.png)

# 🛡️ PromptShield LLM
### Evaluating the Robustness of Prompt Injection Detectors under Distribution Shift

PromptShield is a machine-learning research project that evaluates how reliably prompt-injection detectors identify malicious inputs when evaluated beyond their original training distribution.

The project compares rule-based and classical machine-learning approaches to investigate detection performance, false positives, attack-category robustness, and generalization under distribution shift.

## 🎯 Project Objectives

- Evaluate prompt-injection detection methods for LLM security.
- Compare in-distribution performance with external-dataset performance.
- Analyze false-positive and false-negative behavior.
- Investigate difficult attack categories, including payload splitting, obfuscation, indirect injection, and multilingual prompts.
- Examine whether a hybrid feature-based detector offers a useful trade-off between detection and false-positive control.

## 🧠 Detection Approaches

| Detector | Description |
|---|---|
| Rule/Regex Baseline | Uses manually defined patterns to identify explicit malicious instructions. |
| TF-IDF + Logistic Regression | Uses word-level TF-IDF features with Logistic Regression. |
| Character TF-IDF + Linear SVM | Uses character-level n-gram features with a Linear Support Vector Machine. |
| PromptShield Hybrid | Combines word-level and character-level TF-IDF features with Logistic Regression. |

Each detector independently classifies prompts as **Benign** or **Malicious**.

## ⚙️ Technology Stack

- **Language:** Python
- **Machine Learning:** scikit-learn
- **Feature Extraction:** TF-IDF
- **Models:** Logistic Regression, Linear SVM
- **Detection Baseline:** Rule-based pattern matching
- **Data Processing:** CSV and JSON

## 📊 Experimental Results

### In-distribution evaluation

Evaluation was conducted on the held-out Mirza dataset test set containing 1,229 samples.

| Detector | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Rule/Regex | 34.01% | 0.00% | 0.00% | 0.00% |
| TF-IDF + LR | 75.26% | 97.03% | 64.49% | 77.48% |
| Character SVM | 46.62% | 100.00% | 19.11% | 32.09% |
| Hybrid | 72.34% | 99.37% | 58.45% | 73.60% |

### External distribution-shift evaluation

The frozen detectors were evaluated on an external dataset after removing five exact-string overlaps.

| Detector | Accuracy | Precision | Recall | F1-score | FPR |
|---|---:|---:|---:|---:|---:|
| Rule/Regex | 75.62% | 39.60% | 11.66% | 20.87% | 0.02% |
| TF-IDF + LR | 63.40% | 38.24% | 53.15% | 44.48% | 32.70% |
| Character SVM | 71.75% | 37.32% | 3.52% | 6.43% | 2.25% |
| Hybrid | 69.06% | 42.30% | 33.40% | 37.32% | 17.36% |

**Key findings**

- TF-IDF + Logistic Regression achieved the highest F1-score on both the in-distribution test set and the external dataset.
- Hybrid F1-score decreased from 73.60% to 37.32% under the evaluated distribution shift.
- Hybrid had a lower external false-positive rate than TF-IDF + Logistic Regression, but also lower recall and F1-score.
- Obfuscation remained challenging: recall was 34.00% for TF-IDF + Logistic Regression and 34.33% for Hybrid.
- The multilingual subset contained only 16 samples, so its results should be interpreted cautiously.

These results show why detector evaluation should consider multiple metrics and external data rather than accuracy alone.

## 🗂️ Dataset

### Primary dataset

The experiments use the Mirza prompt-injection detection dataset.

- Total samples: 5,441
- Training samples: 4,212
- Held-out test samples: 1,229
- Benign samples: 2,628
- Malicious samples: 2,813

The original train/test split is preserved.

Dataset source: [Prompt Injection Detection Dataset](https://github.com/mirzaakhi/prompt-injection-detection-dataset)

### External evaluation dataset

The external evaluation uses the `hendzh/PromptShield` dataset.

- Original samples: 23,516
- Exact-string overlaps removed: 5
- Final evaluation samples: 23,511

Dataset source: [hendzh/PromptShield on Hugging Face](https://huggingface.co/datasets/hendzh/PromptShield)

Only exact-string overlap was checked and removed; semantic or near-duplicate overlap was not comprehensively assessed.

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/abhishek0802-bot/Prompt-Shieldllm.git
cd Prompt-Shieldllm
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the primary dataset

Download the official `prompt_injection_dataset.csv` from the primary dataset repository linked above.

Place the CSV in the project root, next to `run_experiment.py`.

Preserve the dataset's official train/test split.

### 5. Run the experiments

```bash
python run_experiment.py
```

### 6. Run external evaluation

Prepare the external evaluation files expected by the script, then run:

```bash
python evaluate_external.py
```

### 7. Compare results

```bash
python compare_results.py
```

Refer to the script output and the project files for the specific generated result filenames and expected input formats.

## 📁 Project Structure

```text
Prompt-Shieldllm/
├── .gitignore
├── README.md
├── requirements.txt
├── run_experiment.py
├── evaluate_external.py
└── compare_results.py
```

Dataset files and generated results may be stored locally and are not necessarily included in the public repository.

## 🔬 Research Methodology

1. Prepare the primary dataset while preserving its official split.
2. Train the detector models using the training data.
3. Select decision thresholds using a validation subset.
4. Freeze the models and thresholds.
5. Evaluate performance on the held-out test set.
6. Evaluate the same frozen detectors on the external dataset.
7. Compare classification metrics and analyze attack categories.

Metrics include accuracy, precision, recall, F1-score, false-positive rate (FPR), false-negative rate (FNR), and precision-recall area under the curve (PR-AUC).

## ⚠️ Limitations

- Only one external dataset was evaluated.
- The multilingual and benign hard-negative subsets were small.
- Semantic and near-duplicate dataset overlap was not comprehensively assessed.
- The experiments focus on classical machine-learning detectors rather than transformer-based detection models.
- Results are specific to the evaluated datasets, models, and decision thresholds.

## 🔮 Future Work

- Evaluate transformer-based detectors such as BERT and RoBERTa.
- Expand multilingual, code-mixed, and obfuscated attack datasets.
- Develop larger and more diverse benign hard-negative benchmarks.
- Investigate adversarial training and threshold-sensitivity analysis.
- Evaluate detectors across additional external datasets.

## 👨‍💻 Author

**Abhishek Vinayak**  
B.Tech — Computer Science and Engineering (Data Science)  
Manipal University Jaipur, India

## 📚 Project Context

This project was developed as part of academic research on LLM security, prompt-injection detection, and machine-learning robustness.

**Disclaimer:** This is an experimental research project, not a guarantee that detected prompts are malicious or that undetected prompts are safe. It should not be treated as a standalone security control for production LLM systems.
