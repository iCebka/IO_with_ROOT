#!/usr/bin/env python3
"""
Dataset format converter.

Converts a tabular dataset between five representations: ROOT (TTree or
RNTuple), CSV, HDF5, NPZ, and a raw binary record format. Every output
format encodes the exact same numeric feature matrix X and integer label
vector y: same rows, same columns, same values up to floating point
representation, so switching formats never changes what the dataset
contains, only how it is stored on disk.

Typical usage: point the script at a ROOT file and a target column name
to get complete CSV, HDF5, NPZ and binary copies of the same dataset in
one call. Alternatively, start from an existing CSV and derive HDF5, NPZ
and binary from it directly.
"""

import os
import json
import argparse

import numpy as np
import pandas as pd


# ROOT column type strings treated as numeric, and therefore eligible to
# become a feature or the target column when reading directly from ROOT.
NUMERIC_TYPES = {
    "float",
    "double",
    "Float_t",
    "Double_t",
    "int",
    "unsigned int",
    "long",
    "unsigned long",
    "long long",
    "unsigned long long",
    "std::int32_t",
    "std::uint32_t",
    "std::int64_t",
    "std::uint64_t",
}

# dtypes used for every converted format. Keeping these as
# named constants instead of repeating the literal strings makes it clear
# that all formats are meant to be equivalent representations of the same
# underlying data.
X_DTYPE = np.float32
Y_DTYPE = np.int64

# Target bytes per HDF5 chunk when chunk_rows is left on "auto". One
# mebibyte sits within the range HDF5 documentation commonly recommends:
# large enough that the number of chunks (and the per chunk bookkeeping
# HDF5 keeps for each one) does not become a bottleneck, small enough
# that a partial read does not pull in an unrelated large block.
AUTO_CHUNK_TARGET_BYTES = 1024 * 1024


def ensure_parent_dir(path):
    """Create the parent directory of path if it does not exist yet."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def write_metadata(metadata_path, **fields):
    """
    Write a small JSON sidecar describing a converted dataset: sample and
    feature counts, dtypes, and any format specific layout details. This
    lets a reader learn a dataset's shape and byte layout from a small
    JSON file instead of opening a potentially very large data file just
    to ask its shape.
    """
    ensure_parent_dir(metadata_path)
    with open(metadata_path, "w") as f:
        json.dump(fields, f, indent=2)
    return metadata_path


def load_xy_from_root(root_path, tree_name, target_col):
    """
    Read a ROOT file directly into (X, y, feature_names) arrays, without
    writing anything to disk first. TTree and RNTuple objects are both
    read through the same RDataFrame constructor in recent ROOT versions.

    A column is kept as a feature if its declared ROOT type is one of the
    entries in NUMERIC_TYPES; anything else is reported and skipped. The
    target column is always kept, even if its type were not numeric.
    """
    import ROOT

    if not os.path.exists(root_path):
        raise FileNotFoundError(f"ROOT file not found: {root_path}")

    rdf = ROOT.RDataFrame(tree_name, root_path)
    columns = [str(c) for c in rdf.GetColumnNames()]

    if target_col not in columns:
        raise ValueError(
            f"Target column '{target_col}' not found. Available columns: {columns}"
        )

    selected_cols = []
    skipped_cols = []

    for c in columns:
        try:
            ctype = str(rdf.GetColumnType(c))
        except Exception:
            # A handful of column types can raise when queried this way,
            # for example some vector branches. Treat as unknown and skip
            # rather than aborting the whole read.
            ctype = "unknown"

        if c == target_col or ctype in NUMERIC_TYPES:
            selected_cols.append(c)
        else:
            skipped_cols.append((c, ctype))

    feature_names = [c for c in selected_cols if c != target_col]

    print("Reading ROOT source")
    print(f"  file          : {root_path}")
    print(f"  tree / object : {tree_name}")
    print(f"  target column : {target_col}")
    print(f"  feature count : {len(feature_names)}")
    print(f"  skipped count : {len(skipped_cols)}")
    for c, t in skipped_cols:
        print(f"    skipped {c}: {t}")

    data = rdf.AsNumpy([target_col] + feature_names)

    y = np.asarray(data[target_col]).astype(Y_DTYPE)
    # np.column_stack normally already returns a C order array, forced
    # explicitly here anyway so this function carries the same guarantee
    # as load_xy_from_csv, rather than relying on that being incidentally
    # true of column_stack's current implementation.
    X = np.ascontiguousarray(
        np.column_stack([np.asarray(data[c]) for c in feature_names]).astype(X_DTYPE)
    )

    print(f"  rows read     : {len(y)}")

    return X, y, feature_names


def load_xy_from_csv(csv_file, target_col):
    """
    Read a CSV file into (X, y, feature_names) arrays. Non numeric
    columns other than the target column are dropped using pandas'
    numeric dtype detection, and reported rather than silently discarded.
    """
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    df = pd.read_csv(csv_file)

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. Available columns: {list(df.columns)}"
        )

    y = df[target_col].to_numpy(dtype=Y_DTYPE)

    candidate_features = df.drop(columns=[target_col])
    feature_df = candidate_features.select_dtypes(include=["number"])

    dropped = [c for c in candidate_features.columns if c not in feature_df.columns]
    if dropped:
        print(f"  dropped non numeric columns: {dropped}")

    # pandas commonly returns a Fortran (column major) ordered array from
    # to_numpy when every column shares the same dtype, since it stores
    # same dtype columns internally as one column major block. Every
    # downstream writer in this script assumes row major (C order) bytes,
    # some rely on it implicitly (a reader computing a row's byte offset
    # as row index times row size), so the array is forced into C order
    # explicitly here rather than left to whatever pandas happened to
    # produce internally.
    X = np.ascontiguousarray(feature_df.to_numpy(dtype=X_DTYPE))
    feature_names = list(feature_df.columns)

    return X, y, feature_names


def write_csv(X, y, feature_names, target_col, output_csv):
    """Write (X, y) to a CSV file, target column first."""
    ensure_parent_dir(output_csv)

    df = pd.DataFrame(X, columns=feature_names)
    df.insert(0, target_col, y)
    df.to_csv(output_csv, index=False)

    print(f"CSV saved to {output_csv}")
    print(f"  rows    : {len(df)}")
    print(f"  columns : {len(df.columns)}")

    return output_csv


def _auto_chunk_rows(num_features, itemsize, target_bytes=AUTO_CHUNK_TARGET_BYTES):
    """Pick a row count for HDF5 chunking that lands close to target_bytes
    per chunk, given how many bytes a single row occupies."""
    bytes_per_row = max(1, num_features * itemsize)
    rows = max(1, target_bytes // bytes_per_row)
    return int(rows)


def write_hdf5(X, y, feature_names, target_col, output_h5, chunk_rows="auto", compression=None):
    """
    Write (X, y) to a single HDF5 file with two datasets, X and y.

    chunk_rows controls the chunk shape along the row axis. The string
    "auto" (the default) computes a row count targeting roughly one
    mebibyte per chunk. An integer uses that many rows per chunk
    directly. None disables chunking entirely, which also disables
    compression, since HDF5 cannot compress a contiguous dataset.

    compression is left off (None) by default; pass a value such as
    "gzip" to trade disk space for CPU time spent decoding on every read.
    """
    import h5py

    ensure_parent_dir(output_h5)

    num_samples, num_features = X.shape
    itemsize = np.dtype(X_DTYPE).itemsize

    resolved_rows = None
    if chunk_rows == "auto":
        resolved_rows = _auto_chunk_rows(num_features, itemsize)
    elif chunk_rows is not None:
        resolved_rows = int(chunk_rows)

    if resolved_rows is not None:
        resolved_rows = min(resolved_rows, max(1, num_samples))

    x_chunks = (resolved_rows, num_features) if resolved_rows else None
    y_chunks = (resolved_rows,) if resolved_rows else None

    with h5py.File(output_h5, "w") as f:
        f.create_dataset("X", data=X, chunks=x_chunks, compression=compression)
        f.create_dataset("y", data=y, chunks=y_chunks, compression=compression)

    print(f"HDF5 saved to {output_h5}")
    print(f"  samples     : {num_samples}")
    print(f"  features    : {num_features}")
    print(f"  chunk rows  : {resolved_rows}")
    print(f"  compression : {compression}")

    write_metadata(
        output_h5 + ".metadata.json",
        dataset=os.path.splitext(os.path.basename(output_h5))[0],
        format="HDF5",
        num_samples=int(num_samples),
        num_features=int(num_features),
        num_classes=int(len(np.unique(y))),
        target_column=target_col,
        feature_columns=feature_names,
        X_dtype=str(np.dtype(X_DTYPE)),
        y_dtype=str(np.dtype(Y_DTYPE)),
        chunk_rows=resolved_rows,
        compression=compression,
    )

    return output_h5


def write_npz(X, y, feature_names, target_col, output_npz):
    """Write (X, y) to a single npz archive, uncompressed."""
    ensure_parent_dir(output_npz)

    np.savez(output_npz, X=X, y=y)

    print(f"NPZ saved to {output_npz}")
    print(f"  samples  : {len(y)}")
    print(f"  features : {X.shape[1]}")

    write_metadata(
        output_npz + ".metadata.json",
        dataset=os.path.splitext(os.path.basename(output_npz))[0],
        format="NPZ",
        num_samples=int(len(y)),
        num_features=int(X.shape[1]),
        num_classes=int(len(np.unique(y))),
        target_column=target_col,
        feature_columns=feature_names,
        X_dtype=str(np.dtype(X_DTYPE)),
        y_dtype=str(np.dtype(Y_DTYPE)),
    )

    return output_npz


def write_bin(X, y, feature_names, target_col, output_dir, num_shards):
    """
    Write (X, y) as one or more raw binary shard files. Each shard is a
    flat array of fixed size records, one per row, with no header: the
    feature values first, num_features consecutive values of X_DTYPE,
    immediately followed by one label value of Y_DTYPE. Reading row i
    only requires seeking to i times the record size and reading that
    many bytes; no other row needs to be touched.

    num_shards controls how many files the dataset is split across. Use
    1 for a single file containing the whole dataset, or more to split it
    into several files of roughly equal size.
    """
    if num_shards <= 0:
        raise ValueError("num_shards must be greater than 0")

    os.makedirs(output_dir, exist_ok=True)

    num_samples, num_features = X.shape

    # A packed structured dtype: no padding is inserted between the
    # features block and the label, so the on disk layout matches the
    # feature_offset_bytes / label_offset_bytes values written to
    # metadata.json exactly.
    record_dtype = np.dtype([
        ("features", X_DTYPE, (num_features,)),
        ("label", Y_DTYPE),
    ])

    records = np.zeros(num_samples, dtype=record_dtype)
    records["features"] = X
    records["label"] = y

    shard_row_indices = np.array_split(np.arange(num_samples), num_shards)

    shard_files = []
    shard_sizes = []
    for i, idx in enumerate(shard_row_indices):
        shard_name = f"shard_{i:03d}.bin"
        shard_path = os.path.join(output_dir, shard_name)
        records[idx].tofile(shard_path)
        shard_files.append(shard_name)
        shard_sizes.append(int(len(idx)))

    feature_bytes = int(np.dtype(X_DTYPE).itemsize * num_features)
    label_bytes = int(np.dtype(Y_DTYPE).itemsize)
    record_bytes = int(record_dtype.itemsize)

    metadata_path = os.path.join(output_dir, "metadata.json")
    write_metadata(
        metadata_path,
        dataset=os.path.basename(os.path.normpath(output_dir)),
        format="BIN",
        num_samples=int(num_samples),
        num_features=int(num_features),
        num_classes=int(len(np.unique(y))),
        target_column=target_col,
        feature_columns=feature_names,
        X_dtype=str(np.dtype(X_DTYPE)),
        y_dtype=str(np.dtype(Y_DTYPE)),
        record_bytes=record_bytes,
        feature_offset_bytes=0,
        feature_bytes=feature_bytes,
        label_offset_bytes=feature_bytes,
        label_bytes=label_bytes,
        num_shards=int(num_shards),
        shard_files=shard_files,
        shard_sizes=shard_sizes,
    )

    print(f"BIN saved to {output_dir}")
    print(f"  samples      : {num_samples}")
    print(f"  features     : {num_features}")
    print(f"  shards       : {num_shards}")
    print(f"  record bytes : {record_bytes}")
    print(f"  metadata     : {metadata_path}")

    return output_dir


def root_to_csv(root_path, tree_name, target_col, output_csv):
    X, y, feature_names = load_xy_from_root(root_path, tree_name, target_col)
    return write_csv(X, y, feature_names, target_col, output_csv)


def root_to_hdf5(root_path, tree_name, target_col, output_h5, chunk_rows="auto", compression=None):
    X, y, feature_names = load_xy_from_root(root_path, tree_name, target_col)
    return write_hdf5(X, y, feature_names, target_col, output_h5, chunk_rows=chunk_rows, compression=compression)


def root_to_npz(root_path, tree_name, target_col, output_npz):
    X, y, feature_names = load_xy_from_root(root_path, tree_name, target_col)
    return write_npz(X, y, feature_names, target_col, output_npz)


def root_to_bin(root_path, tree_name, target_col, output_dir, num_shards):
    X, y, feature_names = load_xy_from_root(root_path, tree_name, target_col)
    return write_bin(X, y, feature_names, target_col, output_dir, num_shards)


def csv_to_hdf5(csv_file, target_col, output_h5, chunk_rows="auto", compression=None):
    X, y, feature_names = load_xy_from_csv(csv_file, target_col)
    return write_hdf5(X, y, feature_names, target_col, output_h5, chunk_rows=chunk_rows, compression=compression)


def csv_to_npz(csv_file, target_col, output_npz):
    X, y, feature_names = load_xy_from_csv(csv_file, target_col)
    return write_npz(X, y, feature_names, target_col, output_npz)


def csv_to_bin(csv_file, target_col, output_dir, num_shards):
    X, y, feature_names = load_xy_from_csv(csv_file, target_col)
    return write_bin(X, y, feature_names, target_col, output_dir, num_shards)

def root_to_npz_native(root_path, tree_name, target_col, output_npz):
    """ROOT -> uncompressed NPZ, preserving original numeric column dtypes."""
    X, y, names, types, skipped = load_xy_from_root_native(
        root_path, tree_name, target_col
    )
    ensure_parent_dir(output_npz)
    np.savez(output_npz, X=X, y=y)
    _write_native_metadata(output_npz, "NPZ", X, y, names, target_col,
                           types, skipped, compression=None)
    print(f"NPZ saved: {output_npz} ({os.path.getsize(output_npz):,} bytes)")
    return output_npz


# Independent, dtype-preserving ROOT conversions. The original commands above
# remain unchanged so existing campaigns are reproducible.
def load_xy_from_root_native(root_path, tree_name, target_col):
    """Read numeric ROOT columns without narrowing or widening their dtypes.

    X remains a homogeneous 2-D array for compatibility with the existing
    training loaders. Mixed feature dtypes are therefore rejected explicitly:
    silently promoting them would no longer preserve each column's dtype.
    Non-numeric columns are excluded and documented in the metadata.
    """
    import ROOT

    if not os.path.isfile(root_path):
        raise FileNotFoundError(root_path)

    rdf = ROOT.RDataFrame(tree_name, root_path)
    columns = [str(col) for col in rdf.GetColumnNames()]
    if target_col not in columns:
        raise ValueError(f"Target column {target_col!r} not in ROOT object: {columns}")

    feature_names, skipped_columns, root_types = [], [], {}
    for name in columns:
        try:
            root_type = str(rdf.GetColumnType(name))
        except Exception:
            root_type = "unknown"
        root_types[name] = root_type
        if root_type in NUMERIC_TYPES:
            if name != target_col:
                feature_names.append(name)
        elif name == target_col:
            raise TypeError(f"Non-numeric target {name!r}: {root_type}")
        else:
            skipped_columns.append({"name": name, "root_type": root_type})

    if not feature_names:
        raise ValueError("No supported numeric feature columns")

    print(f"Reading ROOT: {root_path} / {tree_name}")
    print(f"  features: {len(feature_names)}; skipped: {skipped_columns}")
    data = rdf.AsNumpy([target_col] + feature_names)
    y = np.asarray(data[target_col])
    arrays = [np.asarray(data[name]) for name in feature_names]

    for name, arr in [(target_col, y)] + list(zip(feature_names, arrays)):
        if arr.ndim != 1 or arr.dtype.kind not in "iuf":
            raise TypeError(f"Expected one-dimensional numeric column {name!r}, got {arr.shape}, {arr.dtype}")
        if arr.shape[0] != y.shape[0]:
            raise ValueError(f"Length mismatch in column {name}")

    feature_types = {str(a.dtype) for a in arrays}
    if len(feature_types) != 1:
        raise TypeError(
            "Feature dtypes differ: "
            + str({name: str(a.dtype) for name, a in zip(feature_names, arrays)})
            + ". A single homogeneous X matrix cannot preserve all original dtypes."
        )

    # column_stack copies values but does not cast them when all types match.
    X = np.ascontiguousarray(np.column_stack(arrays))
    if X.dtype != arrays[0].dtype:
        raise AssertionError("Unexpected feature dtype conversion")
    y = np.ascontiguousarray(y)

    print(f"  entries: {len(y)}; X: {X.shape}, {X.dtype}; y: {y.shape}, {y.dtype}")
    return X, y, feature_names, root_types, skipped_columns


def _write_native_metadata(output_path, format_name, X, y, feature_names,
                           target_col, root_types, skipped_columns, **extra):
    write_metadata(
        output_path + ".metadata.json",
        dataset=os.path.splitext(os.path.basename(output_path))[0],
        format=format_name,
        num_samples=int(len(y)),
        num_features=int(X.shape[1]),
        num_classes=int(len(np.unique(y))),
        target_column=target_col,
        feature_columns=feature_names,
        X_dtype=str(X.dtype),
        y_dtype=str(y.dtype),
        source_root_types=root_types,
        skipped_columns=skipped_columns,
        dtype_policy="preserve source NumPy numeric dtypes; no explicit casts",
        **extra,
    )



def root_to_hdf5_native(root_path, tree_name, target_col, output_h5,
                        chunk_rows="auto", compression=None):
    """ROOT -> HDF5 with native numeric dtypes and explicit chunk settings."""
    import h5py

    X, y, names, types, skipped = load_xy_from_root_native(
        root_path, tree_name, target_col
    )
    nrows, nfeatures = X.shape
    if chunk_rows == "auto":
        resolved_rows = _auto_chunk_rows(nfeatures, X.dtype.itemsize)
    elif chunk_rows is None:
        resolved_rows = None
    else:
        resolved_rows = int(chunk_rows)
        if resolved_rows <= 0:
            raise ValueError("chunk_rows must be positive, auto, or none")

    if resolved_rows is not None:
        resolved_rows = min(resolved_rows, max(1, nrows))
    if nrows == 0 and resolved_rows is not None:
        raise ValueError("Cannot create chunked datasets with zero rows")
    if resolved_rows is None and compression is not None:
        raise ValueError("Compression requires HDF5 chunking")

    ensure_parent_dir(output_h5)
    with h5py.File(output_h5, "w") as f:
        f.create_dataset("X", data=X,
                         chunks=(resolved_rows, nfeatures) if resolved_rows else None,
                         compression=compression)
        f.create_dataset("y", data=y,
                         chunks=(resolved_rows,) if resolved_rows else None,
                         compression=compression)

    _write_native_metadata(output_h5, "HDF5", X, y, names, target_col,
                           types, skipped, chunk_rows=resolved_rows,
                           compression=compression)
    print(f"HDF5 saved: {output_h5} ({os.path.getsize(output_h5):,} bytes)")
    return output_h5


def csv_to_all(csv_file, target_col, output_dir, basename, num_shards, chunk_rows="auto", compression=None):
    """
    Derive HDF5, NPZ and binary shards from an existing CSV file, reusing
    the same (X, y) arrays for all three so they describe exactly the
    same data.
    """
    if basename is None:
        basename = os.path.splitext(os.path.basename(csv_file))[0]

    os.makedirs(output_dir, exist_ok=True)

    X, y, feature_names = load_xy_from_csv(csv_file, target_col)

    output_h5 = os.path.join(output_dir, f"{basename}.h5")
    output_npz = os.path.join(output_dir, f"{basename}.npz")
    output_bin_dir = os.path.join(output_dir, f"{basename}_bin")

    write_hdf5(X, y, feature_names, target_col, output_h5, chunk_rows=chunk_rows, compression=compression)
    write_npz(X, y, feature_names, target_col, output_npz)
    write_bin(X, y, feature_names, target_col, output_bin_dir, num_shards)

    print("CSV to ALL completed")
    print(f"  CSV input : {csv_file}")
    print(f"  HDF5      : {output_h5}")
    print(f"  NPZ       : {output_npz}")
    print(f"  BIN dir   : {output_bin_dir}")


def root_to_all(root_file, tree_name, target_col, output_dir, basename, num_shards,
                 chunk_rows="auto", compression=None):
    """
    Convert a ROOT file to CSV, HDF5, NPZ and binary shards in one call.
    The source is read once, and every output format is derived from
    that single in memory copy, so the formats cannot drift apart due to
    re parsing text or reading the source file more than once.
    """
    if basename is None:
        basename = os.path.splitext(os.path.basename(root_file))[0]

    os.makedirs(output_dir, exist_ok=True)

    X, y, feature_names = load_xy_from_root(root_file, tree_name, target_col)

    output_csv = os.path.join(output_dir, f"{basename}.csv")
    output_h5 = os.path.join(output_dir, f"{basename}.h5")
    output_npz = os.path.join(output_dir, f"{basename}.npz")
    output_bin_dir = os.path.join(output_dir, f"{basename}_bin")

    write_csv(X, y, feature_names, target_col, output_csv)
    write_hdf5(X, y, feature_names, target_col, output_h5, chunk_rows=chunk_rows, compression=compression)
    write_npz(X, y, feature_names, target_col, output_npz)
    write_bin(X, y, feature_names, target_col, output_bin_dir, num_shards)

    print("ROOT to ALL completed")
    print(f"  ROOT input : {root_file}")
    print(f"  CSV        : {output_csv}")
    print(f"  HDF5       : {output_h5}")
    print(f"  NPZ        : {output_npz}")
    print(f"  BIN dir    : {output_bin_dir}")



# Additional independent commands retaining the categorical ROOT column.
# Existing conversion commands and training-compatible X/y layouts are unchanged.
def load_xy_category_from_root_native(root_path, tree_name, target_col,
                                      category_col="sample_category"):
    """Read X/y with native numeric dtypes plus an exact UTF-8 category column.

    The category is stored separately because a homogeneous numeric X matrix
    cannot also preserve ROOT's std::string values. Fails if the requested
    category does not exist or any value cannot be represented as UTF-8 text.
    """
    import ROOT

    if not os.path.isfile(root_path):
        raise FileNotFoundError(root_path)
    if category_col == target_col:
        raise ValueError("category_col and target_col must differ")

    rdf = ROOT.RDataFrame(tree_name, root_path)
    columns = [str(c) for c in rdf.GetColumnNames()]
    if target_col not in columns or category_col not in columns:
        raise ValueError(f"Required columns {target_col!r}, {category_col!r} not both found: {columns}")

    root_types = {name: str(rdf.GetColumnType(name)) for name in columns}
    if root_types[target_col] not in NUMERIC_TYPES:
        raise TypeError(f"Target {target_col!r} is not numeric: {root_types[target_col]}")
    if root_types[category_col] not in ("std::string", "string", "std::__cxx11::basic_string<char>"):
        raise TypeError(f"Category {category_col!r} is not a supported ROOT string: {root_types[category_col]}")

    features = [c for c in columns if c not in (target_col, category_col)
                and root_types[c] in NUMERIC_TYPES]
    skipped = [{"name": c, "root_type": root_types[c]}
               for c in columns if c not in features and c not in (target_col, category_col)]
    if not features:
        raise ValueError("No supported numeric feature columns")

    print(f"Reading ROOT: {root_path} / {tree_name}")
    print(f"  numeric features: {len(features)}; category: {category_col}; skipped: {skipped}")
    data = rdf.AsNumpy([target_col] + features + [category_col])
    y = np.asarray(data[target_col])
    feature_arrays = [np.asarray(data[c]) for c in features]
    for name, a in [(target_col, y)] + list(zip(features, feature_arrays)):
        if a.ndim != 1 or a.dtype.kind not in "iuf" or len(a) != len(y):
            raise TypeError(f"Invalid numeric column {name!r}: {a.shape}, {a.dtype}")
    if len({a.dtype.str for a in feature_arrays}) != 1:
        raise TypeError("Mixed feature dtypes cannot be represented faithfully in one X matrix: "
                        + str({name: str(a.dtype) for name, a in zip(features, feature_arrays)}))

    X = np.ascontiguousarray(np.column_stack(feature_arrays))
    if X.dtype != feature_arrays[0].dtype:
        raise AssertionError("Unexpected X dtype change")
    y = np.ascontiguousarray(y)

    raw_category = np.asarray(data[category_col])
    if raw_category.ndim != 1 or len(raw_category) != len(y):
        raise ValueError("Category length or dimensions do not match y")

    decoded = []
    for i, v in enumerate(raw_category):
        if isinstance(v, bytes):
            decoded.append(v.decode("utf-8", errors="strict"))
        elif isinstance(v, str):
            decoded.append(v)
        else:
            raise TypeError(f"Category element {i} is {type(v).__name__}, not str/bytes")
    # Fixed-width Unicode instead of object dtype: no pickle required for np.load.
    category = np.asarray(decoded, dtype=np.str_)
    print(f"  rows: {len(y)}; X: {X.dtype} {X.shape}; y: {y.dtype}; "
          f"{category_col}: Unicode ({len(set(decoded))} distinct)")
    return X, y, category, features, root_types, skipped


def root_to_npz_native_with_category(root_path, tree_name, target_col, output_npz,
                                     category_col="sample_category"):
    """Keep numeric source types and sample_category in a separate NPZ array."""
    X, y, category, names, types, skipped = load_xy_category_from_root_native(
        root_path, tree_name, target_col, category_col
    )
    ensure_parent_dir(output_npz)
    np.savez(output_npz, X=X, y=y, **{category_col: category})
    _write_native_metadata(
        output_npz, "NPZ", X, y, names, target_col, types, skipped,
        compression=None, categorical_columns=[category_col],
        categorical_encoding="NumPy Unicode (UTF-32 in .npy; no pickle)",
        category_dtype=str(category.dtype),
    )
    print(f"NPZ with category saved: {output_npz} ({os.path.getsize(output_npz):,} bytes)")
    return output_npz


def root_to_hdf5_native_with_category(root_path, tree_name, target_col, output_h5,
                                      chunk_rows="auto", compression=None,
                                      category_col="sample_category"):
    """Keep numeric source types and sample_category as a UTF-8 HDF5 dataset."""
    import h5py

    X, y, category, names, types, skipped = load_xy_category_from_root_native(
        root_path, tree_name, target_col, category_col
    )
    nrows, nfeatures = X.shape
    if chunk_rows == "auto":
        resolved_rows = _auto_chunk_rows(nfeatures, X.dtype.itemsize)
    elif chunk_rows is None:
        resolved_rows = None
    else:
        resolved_rows = int(chunk_rows)
        if resolved_rows <= 0:
            raise ValueError("chunk_rows must be positive, auto, or none")
    if resolved_rows is not None:
        resolved_rows = min(resolved_rows, max(1, nrows))
    if nrows == 0 and resolved_rows is not None:
        raise ValueError("Cannot chunk an empty dataset")
    if resolved_rows is None and compression is not None:
        raise ValueError("Compression requires chunking")

    ensure_parent_dir(output_h5)
    with h5py.File(output_h5, "w") as f:
        f.create_dataset("X", data=X,
                         chunks=(resolved_rows, nfeatures) if resolved_rows else None,
                         compression=compression)
        f.create_dataset("y", data=y,
                         chunks=(resolved_rows,) if resolved_rows else None,
                         compression=compression)
        # vlen UTF-8 retains the exact text without arbitrary truncation.
        utf8 = h5py.string_dtype(encoding="utf-8")
        f.create_dataset(category_col, data=category.astype(object), dtype=utf8,
                         chunks=(resolved_rows,) if resolved_rows else None,
                         compression=compression)

    _write_native_metadata(
        output_h5, "HDF5", X, y, names, target_col, types, skipped,
        chunk_rows=resolved_rows, compression=compression,
        categorical_columns=[category_col], categorical_encoding="HDF5 variable-length UTF-8",
    )
    print(f"HDF5 with category saved: {output_h5} ({os.path.getsize(output_h5):,} bytes)")
    return output_h5

def _parse_chunk_rows_arg(value):
    """Translate a chunks command line value into what write_hdf5
    expects: the string auto, None, or an integer row count."""
    if value is None:
        return "auto"
    lowered = value.lower()
    if lowered == "auto":
        return "auto"
    if lowered == "none":
        return None
    return int(value)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Convert a tabular dataset between ROOT, CSV, HDF5, NPZ and raw binary formats."
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    p = subparsers.add_parser("root-to-csv", help="Convert a ROOT file to CSV.")
    p.add_argument("root_file")
    p.add_argument("--tree-name", default="tree")
    p.add_argument("--target-col", default="Label")
    p.add_argument("--output", required=True)

    p = subparsers.add_parser("root-to-hdf5", help="Convert a ROOT file to HDF5.")
    p.add_argument("root_file")
    p.add_argument("--tree-name", default="tree")
    p.add_argument("--target-col", default="Label")
    p.add_argument("--output", required=True)
    p.add_argument("--chunks", default="auto", help="auto, none, or an integer row chunk size.")
    p.add_argument("--compression", default=None, help="for example gzip. Default: none.")

    p = subparsers.add_parser("root-to-npz", help="Convert a ROOT file to NPZ.")
    p.add_argument("root_file")
    p.add_argument("--tree-name", default="tree")
    p.add_argument("--target-col", default="Label")
    p.add_argument("--output", required=True)

    p = subparsers.add_parser("root-to-bin", help="Convert a ROOT file to binary shards.")
    p.add_argument("root_file")
    p.add_argument("--tree-name", default="tree")
    p.add_argument("--target-col", default="Label")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--output-dir", required=True)

    p = subparsers.add_parser("root-to-npz-native", help="Preserve ROOT numeric dtypes in NPZ.")
    p.add_argument("root_file")
    p.add_argument("--tree-name", default="tree")
    p.add_argument("--target-col", default="Label")
    p.add_argument("--output", required=True)

    p = subparsers.add_parser("root-to-hdf5-native", help="Preserve ROOT numeric dtypes in HDF5.")
    p.add_argument("root_file")
    p.add_argument("--tree-name", default="tree")
    p.add_argument("--target-col", default="Label")
    p.add_argument("--output", required=True)
    p.add_argument("--chunks", default="auto")
    p.add_argument("--compression", default=None)

    p = subparsers.add_parser("root-to-all", help="Convert a ROOT file to CSV, HDF5, NPZ and binary shards.")
    p.add_argument("root_file")
    p.add_argument("--tree-name", default="tree")
    p.add_argument("--target-col", default="Label")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--basename", default=None)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--chunks", default="auto")
    p.add_argument("--compression", default=None)

    p = subparsers.add_parser("csv-to-hdf5", help="Convert a CSV file to HDF5.")
    p.add_argument("--csv-file", required=True)
    p.add_argument("--target-col", default="Label")
    p.add_argument("--output", required=True)
    p.add_argument("--chunks", default="auto")
    p.add_argument("--compression", default=None)

    p = subparsers.add_parser("csv-to-npz", help="Convert a CSV file to NPZ.")
    p.add_argument("--csv-file", required=True)
    p.add_argument("--target-col", default="Label")
    p.add_argument("--output", required=True)

    p = subparsers.add_parser("csv-to-bin", help="Convert a CSV file to binary shards.")
    p.add_argument("--csv-file", required=True)
    p.add_argument("--target-col", default="Label")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--output-dir", required=True)

    p = subparsers.add_parser("csv-to-all", help="Convert a CSV file to HDF5, NPZ and binary shards.")
    p.add_argument("--csv-file", required=True)
    p.add_argument("--target-col", default="Label")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--basename", default=None)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--chunks", default="auto")
    p.add_argument("--compression", default=None)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "root-to-csv":
        root_to_csv(args.root_file, args.tree_name, args.target_col, args.output)

    elif args.command == "root-to-hdf5":
        root_to_hdf5(
            args.root_file, args.tree_name, args.target_col, args.output,
            chunk_rows=_parse_chunk_rows_arg(args.chunks), compression=args.compression,
        )

    elif args.command == "root-to-npz":
        root_to_npz(args.root_file, args.tree_name, args.target_col, args.output)

    elif args.command == "root-to-bin":
        root_to_bin(args.root_file, args.tree_name, args.target_col, args.output_dir, args.num_shards)

    elif args.command == "root-to-npz-native":
        root_to_npz_native(args.root_file, args.tree_name, args.target_col, args.output)

    elif args.command == "root-to-hdf5-native":
        root_to_hdf5_native(
            args.root_file, args.tree_name, args.target_col, args.output,
            chunk_rows=_parse_chunk_rows_arg(args.chunks), compression=args.compression,
        )

    elif args.command == "root-to-all":
        root_to_all(
            args.root_file, args.tree_name, args.target_col, args.output_dir, args.basename,
            args.num_shards, chunk_rows=_parse_chunk_rows_arg(args.chunks), compression=args.compression,
        )

    elif args.command == "csv-to-hdf5":
        csv_to_hdf5(
            args.csv_file, args.target_col, args.output,
            chunk_rows=_parse_chunk_rows_arg(args.chunks), compression=args.compression,
        )

    elif args.command == "csv-to-npz":
        csv_to_npz(args.csv_file, args.target_col, args.output)

    elif args.command == "csv-to-bin":
        csv_to_bin(args.csv_file, args.target_col, args.output_dir, args.num_shards)

    elif args.command == "csv-to-all":
        csv_to_all(
            args.csv_file, args.target_col, args.output_dir, args.basename, args.num_shards,
            chunk_rows=_parse_chunk_rows_arg(args.chunks), compression=args.compression,
        )

    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
