# Pentinsula

[![Build Status](https://travis-ci.com/jl-wynen/pentinsula.svg?branch=master)](https://travis-ci.com/jl-wynen/pentinsula)

This Python package provides utilities for managing time series data and chunked datasets in HDF5 files using the [h5py](https://www.h5py.org/) package.
In this context, a time series is a sequence of arrays ('items') that may be constructed or read one item after the other.

# Installation

```shell script
pip3 install .
```

# Usage

Pentinsula provides low level routines for handling a chunked HDF5 dataset through the class `pentinsula.ChunkBuffer`.
Higher level abstractions are implemented in the class `pentinsula.TimeSeries` and in particular the iterators `TimeSeries.read_iter` and `TimeSeries.write_iter`.

## Creating a time series.

The following code creates a new dataset and fills it with `(2, 3)`-arrays. 
```python
from pentinsula import TimeSeries
import numpy as np

# Construct a new time series with chunk size 5 and item shape (2, 3).
# Does not access any files.
series = TimeSeries("myfile.h5", "series", 5, (2, 3))
# Create a dataset in the file.
# write=False means that no data is written yet (no items have been set yet).
series.create_dataset(filemode="w", write=False)
# Iterate over items from time 0 to time 7 and fill with some data.
for _, (time_index, item) in zip(range(7), series.write_iter()):
    item[...] = np.array([[time_index, time_index, time_index],
                          [time_index * 100, time_index * 100, time_index * 100]])
```
Items are numpy array views and can be set using numpy indexing.
Here an ellipsis (`...`) is used to set the entire item.

Note how the code above uses `zip` and `range` to terminate the loop.
It is possible to use `break` instead but in that case, the last item that was produced by the iterator is counted as being part of the series and gets written to file.
This means that
```python
# !!! BAD probably not doing what you expect!!!
for time_index, item in series.write_iter():
    if time_index >= 7:
        break
    item[...] = time_index
```
writes _8_ items, not the intended 7 and the content of the last one is undefined.

## Reading a time series

The time series constructed in the previous section can be read using
```python
import h5py

# Load an existing time series starting at index 0.
series = TimeSeries.load("myfile.h5", "series", 0)
# Open the file manually to keep it open during reading.
with h5py.File("myfile.h5", "r") as f:
    dataset = f["series"]
    # Loop over all times.
    for time_index, item in series.read_iter(file=f, dataset=dataset):
        print(time_index, ":\n", item)
```
which prints
```text
1 :
 [[  1.   1.   1.]
 [100. 100. 100.]]
2 :
 [[  2.   2.   2.]
 [200. 200. 200.]]
3 :
 [[  3.   3.   3.]
 [300. 300. 300.]]
4 :
 [[  4.   4.   4.]
 [400. 400. 400.]]
5 :
 [[  5.   5.   5.]
 [500. 500. 500.]]
6 :
 [[  6.   6.   6.]
 [600. 600. 600.]]
```
In contrast to `write_iter`, `read_iter` terminates on its own if it reaches the end of the dataset.

The read example shows how you can keep the file open by passing a file and dataset handle (the latter can be omitted) to the iterator.
In the `write_iter` example, the file is opened automatically whenever data needs to be written.
Both iterators support those two modes. 

# License

Distributed under the MIT license. See `LICENSE` for more information.
