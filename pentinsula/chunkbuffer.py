from contextlib import contextmanager, nullcontext
from io import BytesIO
from pathlib import Path

import h5py as h5
import numpy as np


def _open_or_pass_file(file, stored_filename, *args, **kwargs):
    if stored_filename is not None:
        if file is not None and not isinstance(file, BytesIO):
            filename = Path(file.filename) if isinstance(file, h5.File) else Path(file)
            if filename != stored_filename:
                raise ValueError(f"Argument file ({filename}) does not match stored file ({stored_filename}.")
        else:
            file = stored_filename
    else:
        if file is None:
            raise ValueError("Arguments file and stored_filename cannot both be None.")

    return nullcontext(file) if isinstance(file, h5.File) else h5.File(file, *args, **kwargs)


def _normalise_chunk_index(chunk_index, nchunks):
    if len(chunk_index) != len(nchunks):
        raise IndexError(f"Invalid index dimension {len(chunk_index)} for dataset dimension {len(nchunks)}")

    normalised = []
    for index, length in zip(chunk_index, nchunks):
        if length <= index <= -length:
            raise IndexError(f"chunk_index {chunk_index} is out of range with number of chunks {nchunks}")
        normalised.append(index if index >= 0 else length - index)
    return tuple(normalised)


def _tuple_floordiv(numerator, denominator):
    return tuple(num // den for num, den in zip(numerator, denominator))


def _chunk_slices(chunk_index, chunk_shape):
    return tuple(slice(i * n, (i + 1) * n)
                 for i, n in zip(chunk_index, chunk_shape))


def _required_dataset_shape(chunk_slices):
    return tuple(chunk_slice.stop for chunk_slice in chunk_slices)


class ChunkBuffer:
    def __init__(self, file, dataset, shape=None, dtype=None, data=None, maxshape=None):
        # special casing on str instead of converting any file to Path allows for streams
        self._filename = (Path(file.filename) if isinstance(file, h5.File)
                          else (Path(file) if isinstance(file, str) else file))
        self._dataset_name = dataset.name if isinstance(dataset, h5.Dataset) else dataset

        if data is not None:
            self._buffer = np.array(data, dtype=dtype)
            if shape is not None:
                self._buffer = self._buffer.reshape(shape)
        else:
            self._buffer = np.empty(shape, dtype=dtype)

        self._maxshape = tuple(maxshape) if isinstance(maxshape, (tuple, list)) else (None,) * self._buffer.ndim
        if len(self._maxshape) != len(self._buffer.shape):
            raise ValueError(f"Argument maxshape {maxshape} has wrong number of dimensions. "
                             f"Expected {len(self._buffer.shape)} according to buffer shape.")

        self._chunk_index = (0,) * self._buffer.ndim

    @classmethod
    def load(cls, file, dataset, chunk_index):
        with _open_or_pass_file(file, None, "r") as h5f:
            dataset = dataset if isinstance(dataset, h5.Dataset) else h5f[dataset]
            if dataset.chunks is None:
                raise RuntimeError(f"Dataset {dataset.name} is not chunked.")

            chunk_buffer = cls(file, dataset, dataset.chunks, dtype=dataset.dtype, maxshape=dataset.maxshape)
            chunk_buffer.select(_normalise_chunk_index(chunk_index,
                                                       _tuple_floordiv(dataset.shape, chunk_buffer._buffer.shape)))

            chunk_buffer.read(dataset=dataset)

            return chunk_buffer

    @property
    def data(self):
        # return view to prevent metadata changes in _buffer
        return self._buffer.view()

    @property
    def shape(self):
        return self._buffer.shape

    @property
    def ndim(self):
        return self._buffer.ndim

    @property
    def dtype(self):
        return self._buffer.dtype

    @property
    def maxshape(self):
        return self._maxshape

    @property
    def chunk_index(self):
        return self._chunk_index

    @property
    def filename(self):
        return self._filename

    @property
    def dataset_name(self):
        return self._dataset_name

    def select(self, chunk_index):
        """
        Does not read!
        :param chunk_index: must be positive
        :return:
        """

        # validate index
        if len(chunk_index) != self.ndim:
            raise IndexError(f"Invalid index dimension {len(chunk_index)} for dataset dimension {self.ndim}.")
        for dim, (index, length, maxlength) in enumerate(zip(chunk_index, self._buffer.shape, self._maxshape)):
            if index < 0:
                raise IndexError(f"Negative chunk_index in dimension {dim}. Only positive values allowed.")
            if maxlength is not None and index * length >= maxlength:
                raise IndexError(f"chunk_index {chunk_index} out of bounds in dimension {dim} "
                                 f"with maxshape {self._maxshape}")

        self._chunk_index = chunk_index

    @contextmanager
    def _load_or_pass_dataset(self, file, dataset, filemode):
        if dataset is None:
            with _open_or_pass_file(file, self._filename, filemode) as h5f:
                yield h5f[self._dataset_name]
        else:
            # Only check if self._filename is a Path in order to allow for storing streams.
            if isinstance(self._filename, Path) and dataset.file.filename != str(self._filename):
                raise ValueError(f"Dataset is not in the stored file ({self._filename}).")
            yield dataset

    @contextmanager
    def _retrieve_dataset(self, file, dataset, filemode):
        with self._load_or_pass_dataset(file, dataset, filemode) as dataset:
            def raise_error(name, in_file, in_memory):
                raise RuntimeError(f"The {name} of dataset {dataset.name} in file {dataset.file.filename} ({in_file}) "
                                   f"does not match the {name} of ChunkBuffer ({in_memory}).")

            if dataset.chunks != self._buffer.shape:
                raise_error("chunk shape", dataset.chunks, self._buffer.shape)
            if dataset.dtype != self._buffer.dtype:
                raise_error("datatype", dataset.dtype, self._buffer.dtype)
            if dataset.maxshape != self._maxshape:
                raise_error("maximum shape", dataset.maxshape, self._maxshape)

            yield dataset

    def read(self, chunk_index=None, file=None, dataset=None):
        with self._retrieve_dataset(file, dataset, "r") as dataset:
            if chunk_index is not None:
                self.select(chunk_index)
            dataset.read_direct(self._buffer, _chunk_slices(self._chunk_index, self._buffer.shape))

    def write(self, must_exist, file=None, dataset=None):
        with self._retrieve_dataset(file, dataset, "a") as dataset:
            chunk_slices = _chunk_slices(self._chunk_index, self._buffer.shape)
            is_big_enough = all(chunk_slice.stop <= dataset_length
                                for chunk_slice, dataset_length in zip(chunk_slices, dataset.shape))

            if not is_big_enough:
                if must_exist:
                    raise RuntimeError(f"The currently selected chunk ({self._chunk_index}) "
                                       f"does not exist in dataset {dataset.name}. "
                                       "Use must_exist=False to resize.")
                else:
                    dataset.resize(max(required, existing)
                                   for required, existing in zip(_required_dataset_shape(chunk_slices),
                                                                 dataset.shape))
            dataset.write_direct(self._buffer, dest_sel=chunk_slices)

    def create_dataset(self, file=None, filemode="a", write=True):
        with _open_or_pass_file(file, self._filename, filemode) as h5f:
            dataset = h5f.create_dataset(self._dataset_name,
                                         _required_dataset_shape(_chunk_slices(self._chunk_index, self._buffer.shape)),
                                         chunks=self._buffer.shape,
                                         maxshape=self._maxshape,
                                         dtype=self.dtype)
            if write:
                self.write(True, dataset=dataset)
