# Parsing and Extending a ROOT dataset

The presented notebooks and scripts conduct an exploration on parsing and extending a ROOT dataset to validate a method for both parsing and extending.

## Notebooks

`root_to_npz_hdf5_step_by_step.ipynb`: Step by step transformations for the current parsing method.

`parsing_root.ipynb`: Use of current parsing method on ROOT dataset with ~700.000 events. The obtained sizes per file are:

|Format|Events|Size|
|--|--|--|
|ROOT|757448|91Mb|
|NPZ|757448|58Mb|
|HDF5|757448|59Mb|


`extending_root.ipynb`: extending a ROOT dataset from ~700.000 events to 1 million events and parsing to NPZ and HDF5.

Its sizes are:

|Format|Events|Size|
|--|--|--|
|ROOT|1 million|120Mb|
|NPZ|1 million|77Mb|
|HDF5|1 million|77Mb|

Modifications of the original parsing method were also tested on the sizes of the parsed datasets.

`test_on_native.ipynb`: Parsing the dataset maintaining float64 and int32 types for the values.

|Format|Events|Size|
|--|--|--|
|ROOT|757448|91Mb|
|NPZ|757448|107Mb|
|HDF5|757448|108Mb|

`test_on_non_filter.ipynb`: Parsing the dataset maintaining float64 and int32 types for the values and non skipping `sample_category` feature. This could be the closest to a exact parsing approach.

|Format|Events|Size|
|--|--|--|
|ROOT|757448|91Mb|
|NPZ|757448|122Mb|
|HDF5|757448|137Mb|

## Scripts

`extend.py`: implements the method for extending a ROOT dataset by repiting the number of events at the end of the dataset.

`inspect_root.py`: implements a method for printing a summary of a ROOT dataset.

`in_hdf5.py`: implements a method for printing a summary of an hdf5 dataset.

`in_npz.py`: implements a method for printing a summary of an npz dataset.

`dataset_converter.py`: implements the main parsing methods from ROOT to NPZ, HDF5 and binary shards. Original `root_to_npz` and `root_to_hdf5` implements the float32 original approach and skips the `sample_category` feature. `root_to_*_native` implement the float64 and int32 approach, and `root_to_*_native_with_category` implement the float64, int32 and non-skip of `sample_category` feature.

## Directory
data:
- original: ~700.000 (ROOT and parsed)
- 1M: 1 million events dataset (ROOT and parsed)
- 5M: 5 million events dataset (only ROOT)
- 10M: 10 million events dataset (only ROOT)
- 50M: 50 million events dataset (only ROOT)
- conversion_walkthrough: generated during `root_to_npz_hdf5.ipynb`.