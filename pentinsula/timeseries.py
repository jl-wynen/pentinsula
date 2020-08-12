from enum import Flag, auto
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from .chunkbuffer import ChunkBuffer
from .h5utils import open_or_pass_dataset
from .types import File, Dataset, Shape, DType


class BufferPolicy(Flag):
    NOTHING = auto()
    READ = auto()
    WRITE = auto()
    READ_WRITE = READ | WRITE


class TimeSeries:
    def __init__(self,
                 file_or_buffer: Union[File, ChunkBuffer],
                 dataset: Optional[Dataset] = None,
                 buffer_length: Optional[int] = None,
                 shape: Shape = (),
                 dtype: Optional[DType] = None,
                 maxshape: Optional[Shape] = None):
        """

        :param file:
        :param dataset:
        :param buffer_length:
        :param shape:
        :param dtype:
        :param maxshape: len = 1 + len(shape)
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

    # load stored stuff
    # index >= 0
    @classmethod
    def load(cls, file: File, dataset: Dataset, time_index: int):
        with open_or_pass_dataset(file, dataset, None, "r") as dataset:
            series = cls(file, dataset, dataset.chunks[0], shape=dataset.shape[1:],
                         dtype=dataset.dtype, maxshape=dataset.maxshape)
            series.read(time_index, file=dataset.file, dataset=dataset)
            return series

    # extend a series
    @classmethod
    def pick_up(cls, file: File, dataset: Dataset):
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
    def filename(self) -> Path:
        return self._buffer.filename

    @property
    def dataset_name(self) -> Path:
        return self._buffer.dataset_name

    @property
    def buffer_length(self) -> int:
        return self._buffer.shape[0]

    @property
    def shape(self) -> Shape:
        return self._buffer.shape[1:]

    @property
    def ndim(self) -> int:
        return self._buffer.ndim - 1

    @property
    def dtype(self) -> DType:
        return self._buffer.dtype

    @property
    def maxtime(self) -> int:
        return self._buffer.maxshape[0]

    @property
    def time_index(self) -> int:
        return self._buffer.chunk_index[0] * self._buffer.shape[0] \
               + self._buffer_time_index

    @property
    def item(self) -> np.ndarray:
        if len(self.shape) == 0:
            # Return an array for scalar items to allow assignment.
            return self._buffer.data.reshape(-1, 1)[self._buffer_time_index]
        return self._buffer.data[self._buffer_time_index]

    def select(self, time_index: int,
               on_buffer_change: BufferPolicy = BufferPolicy.NOTHING,
               file: Optional[File] = None,
               dataset: Optional[Dataset] = None):
        """
        this does not read:
        ts.select(3)
        ts.select(3, BufferPolicy.READ)

        :param time_index:
        :param on_buffer_change:
        :return:
        """

        if time_index < 0:
            raise IndexError("Time index must be positive.")
        if self.maxtime is not None and time_index >= self.maxtime:
            raise IndexError(f"Time index out of bounds, index {time_index}"
                             f"larger than maxtime {self.maxtime}")

        time_chunk = time_index // self._buffer.shape[0]
        if time_chunk != self._buffer.chunk_index[0]:
            # need to change buffered chunk
            if on_buffer_change & BufferPolicy.WRITE:
                # save current
                self._buffer.write(must_exist=False, file=file, dataset=dataset)
            self._buffer.select((time_chunk,) + self._buffer.chunk_index[1:])
            if on_buffer_change & BufferPolicy.READ:
                # read new
                self._buffer.read(file=file, dataset=dataset)

        self._buffer_time_index = time_index % self._buffer.shape[0]

    def advance(self, on_buffer_change: BufferPolicy = BufferPolicy.NOTHING,
                file: Optional[File] = None,
                dataset: Optional[Dataset] = None):
        self.select(self.time_index + 1, on_buffer_change, file=file, dataset=dataset)

    def read(self, time_index: Optional[int] = None, file: Optional[File] = None, dataset: Optional[Dataset] = None):
        if time_index is not None:
            self.select(time_index, BufferPolicy.NOTHING)
        fill_level = self._buffer.read(file=file, dataset=dataset)
        if self._buffer_time_index >= fill_level[0]:
            raise RuntimeError(f"Cannot read data for time index {self.time_index}. The dataset only contains items "
                               f"up to time {self._buffer.chunk_index[0] * self._buffer.shape[0] + fill_level[0] - 1}.")

    def write(self, file: Optional[File] = None, dataset: Optional[File] = None):
        self._buffer.write(must_exist=False,
                           fill_level=(self._buffer_time_index + 1,) + self.shape,
                           file=file,
                           dataset=dataset)

    def create_dataset(self, file: Optional[File] = None, filemode: str = "a", write: bool = True):
        self._buffer.create_dataset(file, filemode, write,
                                    fill_level=(self._buffer_time_index + 1,) + self.shape)

    def read_iter(self, times: slice = slice(None), file: Optional[File] = None, dataset: Optional[Dataset] = None):
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
