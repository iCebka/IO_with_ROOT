from pathlib import Path
import json
import h5py

def inspect_hdf5(file_path, n_rows=2):
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

    with h5py.File(path, "r") as f:

        print("\n[Objects in file]\n")

        def print_object(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"Object name : {name}")
                print("Class       : HDF5 Dataset")
                print(f"Shape       : {obj.shape}")
                print(f"Dtype       : {obj.dtype}")
                print(f"Dimensions  : {obj.ndim}")
                print(f"Chunks      : {obj.chunks}")
                print(f"Compression : {obj.compression}")
                print(f"Logical size: {obj.nbytes:,} bytes")
                print(
                    f"Stored size : "
                    f"{obj.id.get_storage_size():,} bytes"
                )
                print()

            elif isinstance(obj, h5py.Group):
                print(f"Group       : {name}")

        f.visititems(print_object)

        if "X" not in f or "y" not in f:
            print("Expected datasets 'X' and 'y' not found.")
            return

        X = f["X"]
        y = f["y"]

        if not isinstance(X, h5py.Dataset) or not isinstance(y, h5py.Dataset):
            raise ValueError("'X' and 'y' must be HDF5 datasets")

        if X.ndim != 2 or y.ndim != 1:
            raise ValueError(
                f"Unexpected dimensions: X={X.shape}, y={y.shape}"
            )

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Row count mismatch: X={X.shape[0]}, y={y.shape[0]}"
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

        for name in feature_names:
            print(f"  - {name}: {X.dtype}")

        print("\n[Storage layout]\n")

        for name, ds in [("X", X), ("y", y)]:
            print(f"{name}:")
            print(f"  Layout      : {'Chunked' if ds.chunks else 'Contiguous'}")
            print(f"  Chunks      : {ds.chunks}")
            print(f"  Compression : {ds.compression}")
            print(f"  Shuffle     : {ds.shuffle}")
            print(f"  Fletcher32  : {ds.fletcher32}")
            print(f"  Logical size: {ds.nbytes:,} bytes")
            print(f"  Stored size : {ds.id.get_storage_size():,} bytes")
            print()

        print("[First rows as NumPy]")

        rows_to_show = min(n_rows, num_samples)

        if rows_to_show > 0:
            # Read only the requested rows rather than
            # materializing the entire HDF5 datasets.
            X_rows = X[:rows_to_show]
            y_rows = y[:rows_to_show]

            for row_idx in range(rows_to_show):
                print(f"\nRow {row_idx}:")
                print(f"  {target_name} = {y_rows[row_idx].item()}")

                for col_idx, name in enumerate(feature_names):
                    value = X_rows[row_idx, col_idx].item()
                    print(f"  {name} = {value}")

