from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Tuple, Union

import h5py as h5
import numpy as np

from .h5utils import get_dataset_name, open_or_pass_file, open_or_pass_dataset
from .types import File, Dataset, Shape, DType


class ChunkBuffer:
    def __init__(self, file: File,
                 dataset: Dataset,
                 shape: Shape = None,
                 dtype: DType = None,
                 data: Optional[np.ndarray] = None,
                 maxshape: Shape = None):
        # special casing on str instead of converting any file to Path allows for streams
        self._filename = (Path(file.filename) if isinstance(file, h5.File)
                          else (Path(file) if isinstance(file, str) else file))
        self._dataset_name = get_dataset_name(dataset)

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
    def load(cls, file: File,
             dataset: Dataset,
             chunk_index: Shape,
             o_fill_level: Optional[List[int]] = None):
        with open_or_pass_dataset(file, dataset, None, "r") as dataset:
            chunk_buffer = cls(file, dataset, dataset.chunks, dtype=dataset.dtype, maxshape=dataset.maxshape)
            chunk_buffer.select(_normalise_chunk_index(chunk_index,
                                                       _chunk_number(dataset.shape, chunk_buffer._buffer.shape)))
            fill_level = chunk_buffer.read(dataset=dataset)

            if o_fill_level is not None:
                o_fill_level.clear()
                o_fill_level.extend(fill_level)
            return chunk_buffer

    @property
    def data(self) -> np.ndarray:
        # return view to prevent metadata changes in _buffer
        return self._buffer.view()

    @property
    def shape(self) -> Shape:
        return self._buffer.shape

    @property
    def ndim(self) -> int:
        return self._buffer.ndim

    @property
    def dtype(self) -> np.dtype:
        return self._buffer.dtype

    @property
    def maxshape(self) -> Shape:
        return self._maxshape

    @property
    def chunk_index(self) -> Shape:
        return self._chunk_index

    @property
    def filename(self) -> Path:
        return self._filename

    @property
    def dataset_name(self) -> Path:
        return self._dataset_name

    def select(self, chunk_index: Shape):
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
    def _load_or_pass_dataset(self, file: Optional[File], dataset: Optional[Dataset], filemode: str):
        if dataset is None:
            with open_or_pass_file(file, self._filename, filemode) as h5f:
                yield h5f[str(self._dataset_name)]
        else:
            dataset_name = get_dataset_name(dataset)
            if dataset_name != self.dataset_name:
                raise ValueError(f"Wrong dataset. Stored: {self.dataset_name}, you passed in {dataset_name}.")
            # Only check if self._filename is a Path in order to allow for storing streams.
            if isinstance(self._filename, Path) and dataset.file.filename != str(self._filename):
                raise ValueError(f"Dataset is not in the stored file ({self._filename}).")

            if isinstance(dataset, h5.Dataset):
                yield dataset
            else:
                with open_or_pass_file(file, self._filename, filemode) as h5f:
                    yield h5f[str(dataset)]

    @contextmanager
    def _retrieve_dataset(self, file: Optional[File], dataset: Optional[Dataset], filemode: str):
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

    def read(self, chunk_index: Optional[Shape] = None,
             file: Optional[File] = None,
             dataset: Optional[Dataset] = None) -> Union[List[int], Tuple[int, ...]]:
        with self._retrieve_dataset(file, dataset, "r") as dataset:
            if chunk_index is not None:
                self.select(chunk_index)
            nchunks = _chunk_number(dataset.shape, self._buffer.shape)
            for dim, (i, n) in enumerate(zip(self.chunk_index, nchunks)):
                if i >= n:
                    raise IndexError(f"Chunk index {i} out of bounds in dimension {dim} with number of chunks = {n}")

            fill_level = _chunk_fill_level(dataset.shape, self._buffer.shape, self._chunk_index, nchunks)
            dataset.read_direct(self._buffer,
                                source_sel=_chunk_slices(self._chunk_index, self._buffer.shape),
                                dest_sel=tuple(slice(0, n) for n in fill_level))
            return fill_level

    def write(self, must_exist: bool,
              fill_level: Optional[Union[List[int], Tuple[int, ...]]] = None,
              file: Optional[File] = None,
              dataset: Optional[Dataset] = None):
        fill_level = self._buffer.shape if fill_level is None else fill_level
        required_shape = _required_dataset_shape(self._chunk_index,
                                                 self._buffer.shape,
                                                 fill_level)

        with self._retrieve_dataset(file, dataset, "a") as dataset:
            if any(required > current
                   for required, current in zip(required_shape, dataset.shape)):
                if must_exist:
                    raise RuntimeError(f"The currently selected chunk {self._chunk_index} "
                                       f"does not exist in dataset {dataset.name}. "
                                       "Use must_exist=False to resize.")
                else:
                    dataset.resize(max(required, existing)
                                   for required, existing in zip(required_shape,
                                                                 dataset.shape))

            dataset.write_direct(self._buffer,
                                 source_sel=tuple(slice(0, n) for n in fill_level),
                                 dest_sel=_chunk_slices(self._chunk_index, self._buffer.shape))

    def create_dataset(self, file: Optional[File] = None,
                       filemode: str = "a",
                       write: bool = True,
                       fill_level: Optional[Union[List[int], Tuple[int, ...]]] = None):
        fill_level = self._buffer.shape if fill_level is None else fill_level

        with open_or_pass_file(file, self._filename, filemode) as h5f:
            dataset = h5f.create_dataset(str(self._dataset_name),
                                         _required_dataset_shape(self._chunk_index,
                                                                 self._buffer.shape,
                                                                 fill_level),
                                         chunks=self._buffer.shape,
                                         maxshape=self._maxshape,
                                         dtype=self.dtype)
            if write:
                self.write(True, dataset=dataset, fill_level=fill_level)


def _normalise_chunk_index(chunk_index: Shape, nchunks: Shape) -> Shape:
    if len(chunk_index) != len(nchunks):
        raise IndexError(f"Invalid index dimension {len(chunk_index)} for dataset dimension {len(nchunks)}")

    normalised = []
    for index, length in zip(chunk_index, nchunks):
        if not (-length <= index < length):
            raise IndexError(f"chunk_index {chunk_index} is out of range with number of chunks {nchunks}")
        normalised.append(index if index >= 0 else length + index)
    return tuple(normalised)


def _tuple_ceildiv(numerator: Shape, denominator: Shape) -> Shape:
    # -(-n // d) computes ceil(n / d) but to infinite precision.
    return tuple(-(-num // den) for num, den in zip(numerator, denominator))


def _chunk_number(full_shape: Shape, chunk_shape: Shape) -> Shape:
    return _tuple_ceildiv(full_shape, chunk_shape)


def _chunk_fill_level(full_shape: Shape, chunk_shape: Shape, chunk_index: Shape, nchunks: Shape) -> Shape:
    # The Modulo operation evaluates to
    # for i in range(2*n):   n - (-i % n)
    #   -> n, 1, 2, ..., n-2, n-1, n, 1, 2, ..., n-2, n-1
    # This is needed because remainder = 0 means, the chunk is fully filled, i.e. fill_level = n.
    return tuple(chunk - (-full % chunk) if idx == nchunk - 1 else chunk
                 for full, chunk, idx, nchunk in zip(full_shape, chunk_shape, chunk_index, nchunks))


def _chunk_slices(chunk_index: Shape, chunk_shape: Shape) -> Tuple[slice, ...]:
    return tuple(slice(i * n, (i + 1) * n)
                 for i, n in zip(chunk_index, chunk_shape))


def _required_dataset_shape(chunk_index: Shape, chunk_shape: Shape, fill_level: Union[Shape, List[int]]) -> Shape:
    for dim, (length, fl) in enumerate(zip(chunk_shape, fill_level)):
        if fl > length:
            raise ValueError(f"Fill level {fill_level} is greater than chunk shape {chunk_shape} in dimension {dim}.")
    return tuple(idx * length + fl
                 for idx, length, fl in zip(chunk_index, chunk_shape, fill_level))
