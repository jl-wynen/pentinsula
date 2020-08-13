from enum import Flag, auto
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .chunkbuffer import ChunkBuffer
from .h5utils import open_or_pass_dataset
from .types import File, Dataset, Shape, DType


class BufferPolicy(Flag):
    """
    Flag indicating what happens when a TimeSeries changes chunks.
    """

    NOTHING = auto()
    READ = auto()
    WRITE = auto()
    READ_WRITE = READ | WRITE


class TimeSeries:
    """
    Build or consume a time series of items in an HDF5 dataset.

    In this context a 'time series' is understood to be a sequence of 'items' that gets
    constructed one item after the other.
    Items can be an array of any shape and are stacked into the time series along a new
    leading dimension.
    This means that the underlying HDF5 dataset has shape (time, *item.shape).

    This class provides methods both for building such a series and for reading it.
    It uses a chunked dataset and buffers a complete chunk in memory to reduce the
    number of file operation.
    The memory required for this is entirely determined by the chunk size of the dataset.

    It is possible to manually process the series by using the methods TimeSeries.select
    and TimeSeries.item to pick a time and access the corresponding item.
    This requires the user to pay attention to the buffering.
    For easier usage, there are also high level iterators that either read items from a
    time series successively (TimeSeries.read_iter) or write them, either overwriting
    or extending a series (TimeSeries.write_iter.
    """

    def __init__(self,
                 file_or_buffer: Union[File, ChunkBuffer],
                 dataset: Optional[Dataset] = None,
                 buffer_length: Optional[int] = None,
                 shape: Shape = (),
                 dtype: Optional[DType] = None,
                 maxshape: Optional[Shape] = None):
        """
        Construct a TimeSeries with underlying buffer in memory.
        Does not verify if a suitable dataset exists in the file or create one.
        Time 0 is selected.

        :param file_or_buffer: Either a file name / file handle or an existing ChunkBuffer.
                               In case of a file, a new buffer is created, and the arguments
                               dataset and buffer_length are required.
                               Otherwise, the existing buffer with all its properties is used.
        :param dataset: Name or dataset object for initialising a new buffer.
        :param buffer_length: Number of times stored in the buffer, i.e. first rank of a chunk.
        :param shape: Shape of *items*, that is the total shape of a chunk is (buffer_length,) + shape.
        :param dtype: Datatype of the buffer.
        :param maxshape: Maximum shape of the dataset. Must satisfy len(maxshape) = 1 + len(shape).
        """

        if isinstance(file_or_buffer, ChunkBuffer):
            self._buffer = file_or_buffer
        else:
            if dataset is None or buffer_length is None:
                raise ValueError("dataset and buffer_length must be provided when "
                                 "file_or_buffer indicates a file.")
            self._buffer = ChunkBuffer(file_or_buffer, dataset, shape=(buffer_length,) + shape,
                                       dtype=dtype, maxshape=maxshape)
        self._buffer_time_index = 0  # into the buffer, not total time

    @classmethod
    def load(cls, file: File, dataset: Dataset, time_index: int):
        """
        Load an existing time series from file.

        :param file: The file containing the dataset.
        :param dataset: The dataset to load. Must be chunked along its first dimension.
        :param time_index: This time is selected and the corresponding chunk is loaded.
                           Must be > 0.
        :return: A newly constructed TimeSeries.
        """

        with open_or_pass_dataset(file, dataset, None, "r") as dataset:
            series = cls(file, dataset, dataset.chunks[0], shape=dataset.shape[1:],
                         dtype=dataset.dtype, maxshape=dataset.maxshape)
            series.read(time_index, file=dataset.file, dataset=dataset)
            return series

    @classmethod
    def pick_up(cls, file: File, dataset: Dataset):
        """
        Extend an existing time series.

        Selects the time *after* the last stored time.
        This means that the content of the current item is undefined.

        :param file: The file containing the dataset.
        :param dataset: The dataset to load. Must be chunked along its first dimension.
        :return: A newly constructed TimeSeries.
        """

        with open_or_pass_dataset(file, dataset, None, "r") as dataset:
            series = cls(file, dataset, dataset.chunks[0], shape=dataset.shape[1:],
                         dtype=dataset.dtype, maxshape=dataset.maxshape)
            if dataset.shape[0] % dataset.chunks[0] == 0:
                # First element of chunk, nothing to read.
                series.select(dataset.shape[0], BufferPolicy.NOTHING)
            else:
                # Item at shape[0] does not exist, read the one before that and advance.
                series.read(dataset.shape[0] - 1, file=dataset.file, dataset=dataset)
                series.advance(BufferPolicy.NOTHING)
            return series

    @property
    def item(self) -> np.ndarray:
        """
        A view of the currently selected item.
        This is always a numpy.ndarray even for scalar items to allow modifications
        through the returned object.

        Note that this only accesses the buffer in memory, you need to call
        read / write to synchronise with the file.
        """
        if len(self.shape) == 0:
            # Return an array for scalar items to allow assignment.
            return self._buffer.data.reshape(-1, 1)[self._buffer_time_index]
        return self._buffer.data[self._buffer_time_index]

    @property
    def time_index(self) -> int:
        """
        The current time index (immutable).
        """
        return self._buffer.chunk_index[0] * self._buffer.shape[0] + self._buffer_time_index

    @property
    def buffer_length(self) -> int:
        """
        Number of times in a buffer.
        """
        return self._buffer.shape[0]

    @property
    def shape(self) -> Shape:
        """
        The shape of items. Does not include the time dimension.
        """
        return self._buffer.shape[1:]

    @property
    def ndim(self) -> int:
        """
        The number of dimensions (ranks) of items.
        Does not include the time dimension.
        """
        return self._buffer.ndim - 1

    @property
    def dtype(self) -> DType:
        """
        The datatype of the dataset.
        """
        return self._buffer.dtype

    @property
    def maxtime(self) -> int:
        """
        The maximum time that can be stored in the dataset.
        May be None.
        """
        return self._buffer.maxshape[0]

    @property
    def filename(self) -> Path:
        """
        The name of the HDF5 file.
        """
        return self._buffer.filename

    @property
    def dataset_name(self) -> Path:
        """
        The full path of the dataset inside of the HDF5 file.
        """
        return self._buffer.dataset_name

    def select(self, time_index: int,
               on_chunk_change: BufferPolicy = BufferPolicy.NOTHING,
               file: Optional[File] = None,
               dataset: Optional[Dataset] = None):
        """
        Change the stored time index.

        This function switches chunks as necessary but only reads from / writes to the file
        if the argument on_buffer_change is set accordingly.

        :param time_index: New time index.
        :param on_chunk_change: Controls what happen if the chunk is changed.
                                Data for the new time index is read from the file only if the READ
                                flag is set, otherwise, the buffer in memory is unchanged.
                                If the WRITE bit is set, the current buffer is written to file
                                before changing the time index, otherwise, the file is not modified.
        :param file: Indicates the file to read from. If given, it must match the filename stored in the buffer.
        :param dataset: Indicates the dataset to read from.
        """

        if time_index < 0:
            raise IndexError("Time index must be positive.")
        if self.maxtime is not None and time_index >= self.maxtime:
            raise IndexError(f"Time index out of bounds, index {time_index}"
                             f"larger than maxtime {self.maxtime}")

        time_chunk = time_index // self._buffer.shape[0]
        if time_chunk != self._buffer.chunk_index[0]:
            # need to change buffered chunk
            if on_chunk_change & BufferPolicy.WRITE:
                # save current
                self._buffer.write(must_exist=False, file=file, dataset=dataset)
            self._buffer.select((time_chunk,) + self._buffer.chunk_index[1:])
            if on_chunk_change & BufferPolicy.READ:
                # read new
                self._buffer.read(file=file, dataset=dataset)

        self._buffer_time_index = time_index % self._buffer.shape[0]

    def advance(self, on_buffer_change: BufferPolicy = BufferPolicy.NOTHING,
                file: Optional[File] = None,
                dataset: Optional[Dataset] = None):
        """
        Move to the next time index.

        See TimeSeries.select for more information.
        """
        self.select(self.time_index + 1, on_buffer_change, file=file, dataset=dataset)

    def read(self, time_index: Optional[int] = None,
             file: Optional[File] = None,
             dataset: Optional[Dataset] = None):
        """
        Read a chunk from file.

        The time must exist in the dataset in the HDF5 file.
        All stored metadata is checked against the file and an error is raised if there is a mismatch.

        An existing file or dataset handle to a currently open connection can be passed in as arguments
        to avoid opening the file on every call to this function.

        A call to this function ensures only that the item for the given time index is read.
        Whether or not other items are read depends on the details of chunking and should not be relied upon.

        :param time_index: Time index to load.
                           If None, the currently selected time index is used.
        :param file: Indicates the file to read from. If given, it must match the filename stored in the buffer.
        :param dataset: Indicates the dataset to read from.
        """

        if time_index is not None:
            self.select(time_index, BufferPolicy.NOTHING)
        fill_level = self._buffer.read(file=file, dataset=dataset)
        if self._buffer_time_index >= fill_level[0]:
            raise RuntimeError(f"Cannot read data for time index {self.time_index}. The dataset only contains items "
                               f"up to time {self._buffer.chunk_index[0] * self._buffer.shape[0] + fill_level[0] - 1}.")

    def write(self, file: Optional[File] = None, dataset: Optional[File] = None):
        """
        Write the buffer up to the *currently selected time* to the file.

        Only the current time index is relevant for determining what is written.
        For example, given a time series with buffer_length = 10, the code
            series.select(3)
            series.item[...] = 3
            series.select(2)
            series.item[...] = 2
            series.write()
        only writes times 0, 1, 2 to the file.
        The data stored in the first assignment to item is *not* written to the file!

        All stored metadata is checked against the file and an error is raised if there is a mismatch.

        An existing file or dataset handle to a currently open connection can be passed in as arguments
        to avoid opening the file on every call to this function.

        Note that in contrast to ChunkBuffer.write, the dataset is always resized to be big enough to
        include the current time and the fill level is determined automatically.

        :param file: Indicates the file to write to. If given, it must match the filename stored in the buffer.
        :param dataset: Indicates the dataset to write to.
        """

        self._buffer.write(must_exist=False,
                           fill_level=(self._buffer_time_index + 1,) + self.shape,
                           file=file,
                           dataset=dataset)

    def create_dataset(self, file: Optional[File] = None, filemode: str = "a", write: bool = True):
        """
        Create a new dataset in the file big enough to contain the currently selected time.

        :param file: If given, use this file handle to access the HDF5 file, otherwise use the stored filename.
        :param filemode: Open-mode of the file, see documentation of h5py.File.
        :param write: If True, write the buffer to the dataset.
                      Only the selected chunk is written, the content of the other chunks is undefined.
                      If False, no data is written, the contents of the dataset are undefined.
        """
        self._buffer.create_dataset(file, filemode, write,
                                    fill_level=(self._buffer_time_index + 1,) + self.shape)

    def read_iter(self, times: slice = slice(None), file: Optional[File] = None, dataset: Optional[Dataset] = None):
        """
        Return an iterator to read items successively from the file.

        This iterator starts at the given time, iterates up to the given maximum time or last time
        in the dataset and read data from file as needed.

        Note that no data is written to the file by this iterator.
        It is not save to modify the yielded items, use TimeSeries.write_iter for writing.

        :param times: Slice to indicate which times to iterator over.
                      Each element can be None, meaning:
                      - times.start is None: Start at the currently selected time.
                      - times.stop is None: Iterate to the end of the dataset.
                      - times.step is None: Equivalent to times.step = 1.
        :param file: Indicates the file to read from. If given, it must match the filename stored in the buffer.
        :param dataset: Indicates the dataset to read from.
        :return: An iterator yielding tuples of time indices and items.
        """

        file = self._buffer.filename if file is None else file
        dataset = self._buffer.dataset_name if dataset is None else dataset
        with open_or_pass_dataset(file, dataset, None, "r") as dataset:
            ntimes = dataset.shape[0]
        if times.stop is not None and times.stop > ntimes:
            raise ValueError(f"Number of times {times.stop} out of bounds, "
                             f"the dataset only contains {ntimes} time points.")

        start, stop, step = times.indices(ntimes)
        if start is None:
            start = self.time_index

        for time_index in range(start, stop, step):
            self.select(time_index, BufferPolicy.READ, file=file, dataset=dataset)
            yield time_index, self.item

    def write_iter(self, flush: bool = True, file: Optional[File] = None, dataset: Optional[Dataset] = None):
        """
        Return an iterator to write items successively to the file.

        This iterator starts at the currently selected time and iterates up to
        the maximum time of the dataset or, if that is None, iterates indefinitely.
        Chunks are written as needed.
        If the last chunk is not filled completely, it is only written if flush = True.

        It is save to break out of a loop over this iterator.
        Note, however, that the last chunk is only written if flush = True.

        Note that no data is read from the file by this iterator.
        The items retain their value unless overwritten by the user.
        Use TimeSeries.read_iter for reading.

        :param flush: If True, the last chunk is written to file when the iterator stops.
                      Otherwise, it is not written.
        :param file: Indicates the file to write to. If given, it must match the filename stored in the buffer.
        :param dataset: Indicates the dataset to read from.
        :return: An iterator yielding tuples of time indices and items.
        """

        # Like builtin range but allows for infinite loops with stop=None.
        def range_(start, stop):
            if stop is None:
                idx = start
                while True:
                    yield idx
                    idx += 1
            else:
                yield from range(start, stop)

        try:
            yield self.time_index, self.item
            for time_index in range_(self.time_index + 1, self._buffer.maxshape[0]):
                self.advance(BufferPolicy.WRITE, file=file, dataset=dataset)
                yield time_index, self.item
        finally:
            if flush:
                # Note on optimisation:
                # In the last advance, the time index was incremented and the current item was not written.
                # This line cannot lead to writing the same dataset twice.
                self.write(file, dataset)
