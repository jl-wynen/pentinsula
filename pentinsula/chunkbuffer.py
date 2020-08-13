from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Tuple, Union

import h5py as h5
import numpy as np

from .h5utils import get_dataset_name, open_or_pass_file, open_or_pass_dataset
from .types import File, Dataset, Shape, DType


class ChunkBuffer:
    """
    Hold a buffer to a single chunk of an HDF5 dataset.

    This class stores a chunk of an HDF5 dataset in memory as a numpy array and provides methods
    for synchronizing the buffer with the file.
    It is important to note that ChunkBuffer never reads from or writes to the file on its own.
    The user must call the corresponding methods to ensure that the buffer and file are up to date.

    ChunkBuffer maintains an index to a chunk in the dataset.
    A chunk index is a tuple of ndim numbers, where ndim is the number of dimensions (ranks)
    of the dataset.
    This index can be changed with the 'select' method.
    The 'read' and 'write' methods use the currently selected index if not given a different index
    as an argument (only 'read').

    It is possible to handle chunks that are not fully filled (size of dataset is not a multiple
    of the chunk size).
    However, ChunkBuffer always holds a full chunk in memory and does not maintain any information
    on how much the chunk is filled as the user has direct write access to the underlying array.
    The read and write methods can return / take as argument the fill level of the chunk and the
    user has to use those appropriately.
    The 'fill level' is a tuple or list of integers that indicates for each dimension how many
    elements are in use.
    For example, for a fully filled chunk, fill_level = chunk_shape.
    For a half filled chunk, fill_level = [size // 2 for size in chunk_shape].
    """

    def __init__(self, file: File,
                 dataset: Dataset,
                 shape: Shape = None,
                 dtype: DType = None,
                 data: Optional[np.ndarray] = None,
                 maxshape: Shape = None):
        """
        Construct a ChunkBuffer in memory.
        Does not verify if a suitable dataset exists in the file or create one.
        The first chunk of the dataset is selected.

        :param file: The file that the dataset lives in.
        :param dataset: The dataset to buffer.
        :param shape: Shape of the *chunk*, not the whole dataset. Required if data is None.
        :param dtype: Datatype of the dataset.
        :param data: Initial data for the *chunk*, not the whole dataset.
                     The chunk shape is inferred from this if argument shape is None, otherwise,
                     attempts to reshape the array.
        :param maxshape: Maximum shape of the dataset.
        """

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
        """
        Load a chunk of an existing dataset.

        :param file: The file containing the dataset.
        :param dataset: The dataset to load. Must be chunked.
        :param chunk_index: The chunk to load.
        :param o_fill_level: If given a list, it is filled with the fill level of the loaded chunk.
        :return: A newly constructed ChunkBuffer.
        """

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
        """
        A view of the stored buffer.
        You can read and modify the data contained in the buffer through this view.

        Note that this only accesses the buffer in memory, you need to call
        read / write to synchronise with the file.
        """
        return self._buffer.view()

    @property
    def shape(self) -> Shape:
        """
        The shape of the buffer, that is the shape of a single chunk.
        """
        return self._buffer.shape

    @property
    def ndim(self) -> int:
        """
        The number of dimensions (ranks) of the dataset.
        """
        return self._buffer.ndim

    @property
    def dtype(self) -> np.dtype:
        """
        The datatype of the dataset.
        """
        return self._buffer.dtype

    @property
    def maxshape(self) -> Shape:
        """
        The maximum shape of the dataset.
        """
        return self._maxshape

    @property
    def chunk_index(self) -> Shape:
        """
        The current chunk index (immutable).
        """
        return self._chunk_index

    @property
    def filename(self) -> Path:
        """
        The name of the HDF5 file.
        """
        return self._filename

    @property
    def dataset_name(self) -> Path:
        """
        The full path of the dataset inside of the HDF5 file.
        """
        return self._dataset_name

    def select(self, chunk_index: Shape):
        """
        Select a chunk.

        This function verifies that the index is valid based on metadata that is available in memory.
        No synchronisation with the file happens, in particular the chunk is not read from the file
        and the buffer keeps its prior contents.

        :param chunk_index: A tuple of indices of the chunk to select. All indices must be positive.
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
        """
        Contextmanager to load a dataset from file or pass along the argument.
        """

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
        """
        Contextmanager to get a handle to the dataset.
        Checks metadata of self against the file.
        """

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
        """
        Read a chunk from the file.

        The chunk must exist in the dataset in the HDF5 file.
        All stored metadata is checked against the file and an error is raised if there is a mismatch.

        An existing file or dataset handle to a currently open connection can be passed in as arguments
        to avoid opening the file on every call to this function.

        :param chunk_index: Index of the chunk to read.
                            If None, use currently selected chunk, i.e. self.chunk_index.
        :param file: Indicates the file to read from. If given, it must match the filename stored in the buffer.
        :param dataset: Indicates the dataset to read from.
        :return: The fill level of the chunk.
        """

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
        """
        Write the currently selected chunk to the file.

        All stored metadata is checked against the file and an error is raised if there is a mismatch.

        An existing file or dataset handle to a currently open connection can be passed in as arguments
        to avoid opening the file on every call to this function.

        :param must_exist: If True, raise an error if the chunk is not already allocated in the dataset.
                           If False, resize the dataset to include the chunk but only up to the fill level.
        :param fill_level: For each dimension, indicate the fill level of the chunk.
                           Only the parts of the buffer within the fill level are written.
        :param file: Indicates the file to write to. If given, it must match the filename stored in the buffer.
        :param dataset: Indicates the dataset to write to.
        """

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
        """
        Create a new dataset in the file big enough to contain the currently selected chunk.

        :param file: If given, use this file handle to access the HDF5 file, otherwise use the stored filename.
        :param filemode: Open-mode of the file, see documentation of h5py.File.
        :param write: If True, write the buffer to the dataset.
                      Only the selected chunk is written, the content of the other chunks is undefined.
                      If False, no data is written, the contents of the dataset are undefined.
        :param fill_level: For each dimension, indicate the fill level of the chunk.
                           Used for computing the shape of the dataset and which parts of the buffer to write.
        """

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
    """
    Make sure the chunk index is within bounds and return a new tuple where all negative indices are
    replaced by corresponding positive onces.
    """

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
    """
    :param full_shape: Shape of the entire dataset.
    :param chunk_shape: Shape of a single chunk
    :return: Number of chunks in every dimension.
    """
    return _tuple_ceildiv(full_shape, chunk_shape)


def _chunk_fill_level(full_shape: Shape, chunk_shape: Shape, chunk_index: Shape, nchunks: Shape) -> Shape:
    """
    :param full_shape: Shape of the entire dataset.
    :param chunk_shape: Shape of a single chunk.
    :param chunk_index: Index of a chunk.
    :param nchunks: Number of chunks.
    :return: Fill level of the given chunk.
    """

    # The Modulo operation evaluates to
    # for i in range(2*n):   n - (-i % n)
    #   -> n, 1, 2, ..., n-2, n-1, n, 1, 2, ..., n-2, n-1
    # This is needed because remainder = 0 means, the chunk is fully filled, i.e. fill_level = n.
    return tuple(chunk - (-full % chunk) if idx == nchunk - 1 else chunk
                 for full, chunk, idx, nchunk in zip(full_shape, chunk_shape, chunk_index, nchunks))


def _chunk_slices(chunk_index: Shape, chunk_shape: Shape) -> Tuple[slice, ...]:
    """
    :param chunk_index: Index of a chunk.
    :param chunk_shape: Shape of chunks.
    :return: Slices into the dataset to address the given chunk.
    """
    return tuple(slice(i * n, (i + 1) * n)
                 for i, n in zip(chunk_index, chunk_shape))


def _required_dataset_shape(chunk_index: Shape, chunk_shape: Shape, fill_level: Union[Shape, List[int]]) -> Shape:
    """
    Return the minimum dataset shape to include the given chunk with the given fill level.
    """
    for dim, (length, fl) in enumerate(zip(chunk_shape, fill_level)):
        if fl > length:
            raise ValueError(f"Fill level {fill_level} is greater than chunk shape {chunk_shape} in dimension {dim}.")
    return tuple(idx * length + fl
                 for idx, length, fl in zip(chunk_index, chunk_shape, fill_level))
