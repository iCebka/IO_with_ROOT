import json
import numpy as np
from pathlib import Path

def inspect_npz(file_path, n_rows=2):
    path = Path(file_path)

    print("=" * 80)
    print(f"FILE: {path}")
    print("=" * 80)

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    metadata_path = Path(str(path) + ".metadata.json")
    metadata = {}

    if metadata_path.exists():
        with metadata_path.open("r") as f:
            metadata = json.load(f)

    with np.load(path, allow_pickle=False) as data:

        print("\n[Arrays in file]\n")

        for key in data.files:
            arr = data[key]

            print(f"Array name : {key}")
            print(f"Shape      : {arr.shape}")
            print(f"Dtype      : {arr.dtype}")
            print(f"Dimensions : {arr.ndim}")
            print(f"Size       : {arr.nbytes:,} bytes")
            print()

        if "X" not in data.files or "y" not in data.files:
            print("Expected arrays 'X' and 'y' not found.")
            return

        X = data["X"]
        y = data["y"]

        if X.ndim != 2 or y.ndim != 1:
            raise ValueError(
                f"Unexpected dimensions: X={X.shape}, y={y.shape}"
            )

        if len(X) != len(y):
            raise ValueError(
                f"Row count mismatch: X={len(X)}, y={len(y)}"
            )

        num_samples, num_features = X.shape

        feature_names = metadata.get("feature_columns", [])

        if len(feature_names) != num_features:
            feature_names = [
                f"feature_{i}" for i in range(num_features)
            ]

        target_name = metadata.get("target_column", "Label")

        print("=" * 40)
        print("Dataset summary")
        print("=" * 40)

        print(f"Entries     : {num_samples}")
        print(f"Features    : {num_features}")
        print(f"Target      : {target_name}")
        print(f"Columns     : {num_features + 1}")
        print(f"X dtype     : {X.dtype}")
        print(f"y dtype     : {y.dtype}")

        print("\n[Columns]\n")

        print(f"  - {target_name}: {y.dtype}")

        for i, name in enumerate(feature_names):
            print(f"  - {name}: {X.dtype}")

        print("\n[First rows as NumPy]")

        for row_idx in range(min(n_rows, num_samples)):
            print(f"\nRow {row_idx}:")

            print(
                f"  {target_name} = "
                f"{y[row_idx].item()}"
            )

            for col_idx, name in enumerate(feature_names):
                value = X[row_idx, col_idx].item()
                print(f"  {name} = {value}")
