import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

mirza = pd.read_csv(RESULTS / "overall_metrics.csv")
external = pd.read_csv(RESULTS / "external_overall_metrics.csv")

metrics = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "fpr",
    "fnr",
    "pr_auc"
]

merged = mirza.merge(
    external,
    on="detector",
    suffixes=("_mirza", "_external")
)

for metric in metrics:
    merged[f"{metric}_change_pp"] = (
        merged[f"{metric}_external"]
        - merged[f"{metric}_mirza"]
    ) * 100

columns = ["detector"]

for metric in metrics:
    columns += [
        f"{metric}_mirza",
        f"{metric}_external",
        f"{metric}_change_pp"
    ]

comparison = merged[columns]

comparison.to_csv(
    RESULTS / "generalization_gap.csv",
    index=False
)

print("=" * 90)
print("PROMPTSHIELD GENERALIZATION-GAP ANALYSIS")
print("=" * 90)

for _, row in comparison.iterrows():

    print(f"\n{row['detector'].upper()}")

    for metric in metrics:

        m = row[f"{metric}_mirza"] * 100
        e = row[f"{metric}_external"] * 100
        c = row[f"{metric}_change_pp"]

        print(
            f"{metric:10s}: "
            f"Mirza={m:7.2f}%  "
            f"External={e:7.2f}%  "
            f"Change={c:+7.2f} pp"
        )

print("\nSaved:")
print(RESULTS / "generalization_gap.csv")