import os
import sys

# Ensure Python can import from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.toxilens.data import load_tox21, get_assay_summary


def main():
    print("Loading Tox21 dataset...")
    df = load_tox21()

    print(f"\nDataset Shape: {df.shape[0]} molecules, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns[:5])} ... {list(df.columns[-3:])}\n")

    summary_df = get_assay_summary(df)

    # Format and print the summary table
    print(summary_df.to_string(index=False))

    # Overall observations
    avg_toxic_pct = summary_df["Toxic %"].mean()
    print(f"\nAverage toxic proportion across all assays: {avg_toxic_pct:.2f}%")
    print("Notice the severe class imbalance and missing values across assays.")


if __name__ == "__main__":
    main()
