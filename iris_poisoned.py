import pandas as pd
import numpy as np
import argparse
import os

def poison_data(input_path, output_path, poison_percentage):
    """
    Loads a clean dataset, poisons a specified percentage of rows by injecting 
    random values into feature columns, and saves the resulting dataset.
    """
    print(f"📥 Loading data from: {input_path}")
    df = pd.read_csv(input_path)

    if poison_percentage == 0:
        print("Poisoning percentage is 0. Saving original dataset.")
        df.to_csv(output_path, index=False)
        return

    # Feature columns to inject noise into
    feature_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

    # Validate that required columns exist in the input dataset
    for col in feature_columns:
        if col not in df.columns:
            raise ValueError(f"Missing expected feature column: '{col}'")

    # Calculate how many rows to poison
    n_rows_to_poison = int(len(df) * (poison_percentage / 100.0))

    if n_rows_to_poison == 0:
        print(f"Percentage too small. No rows will be poisoned. Saving original dataset.")
        df.to_csv(output_path, index=False)
        return

    print(f"💉 Poisoning {n_rows_to_poison}/{len(df)} rows ({poison_percentage}%)")

    # Randomly select unique row indices to poison
    rows_to_poison = np.random.choice(df.index, n_rows_to_poison, replace=False)

    # Inject random noise into selected rows
    for col in feature_columns:
        min_val, max_val = df[col].min(), df[col].max()
        df.loc[rows_to_poison, col] = np.round(
            np.random.uniform(low=min_val, high=max_val, size=n_rows_to_poison), 2
        )

    # Save the modified dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"💾 Saving poisoned data to: {output_path}")
    df.to_csv(output_path, index=False)
    print("✅ Poisoning complete.\n")

# 🔽 Add this block to allow CLI usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Poison a dataset by injecting noise into a percentage of rows.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the poisoned CSV file.")
    parser.add_argument("--percentage", type=float, required=True, help="Percentage of rows to poison.")

    args = parser.parse_args()
    poison_data(args.input, args.output, args.percentage)
