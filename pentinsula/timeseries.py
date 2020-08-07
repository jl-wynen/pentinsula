from enum import Flag, auto

import h5py as h5
import numpy as np

from .chunkbuffer import ChunkBuffer
from .h5utils import open_or_pass_file, open_or_pass_dataset


# name for 'entry for given time'? -> slot / element / timepoint

# registry for collective writes? -> benchmark, hack next()

# iterators for reading / writing entries


class BufferPolicy(Flag):
    NOTHING = auto()
    READ = auto()
    WRITE = auto()
    READ_WRITE = READ | WRITE


def _normalise_time_index(index, ntimes):
    if not (-ntimes <= index < ntimes):
        raise IndexError(f"Time index {index} is out of bounds with number of times {ntimes}.")
    return index if index >= 0 else ntimes - index


class TimeSeries:
    def __init__(self, file_or_buffer, dataset=None, buffer_length=None,
                 shape=(), dtype=None, maxshape=None):
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
    @classmethod
    def load(cls, file, dataset, time_index):
        with open_or_pass_dataset(file, dataset, None, "r") as dataset:
            ntimes = dataset.shape[0]
            time_index = _normalise_time_index(time_index, ntimes)
            chunk_length = dataset.chunks[0]
            chunk_index = (time_index // chunk_length,) + (0,) * (len(dataset.shape) - 1)
            buffer = ChunkBuffer.load(dataset.file, dataset, chunk_index)

            series = cls(buffer)
            series._buffer_time_index = time_index % chunk_length
            return series

    # extend a series
    @classmethod
    def pick_up(cls, file, dataset):
        ...

    @property
    def shape(self):
        return self._buffer.shape[1:]

    @property
    def ndim(self):
        return self._buffer.ndim - 1

    @property
    def dtype(self):
        return self._buffer.dtype

    @property
    def maxtime(self):
        return self._buffer.maxshape[0]

    @property
    def time_index(self):
        return self._buffer.chunk_index[0] * self._buffer.shape[0] \
               + self._buffer_time_index

    @property
    def item(self):
        return self._buffer.data[self._buffer_time_index]

    def select(self, time_index, on_buffer_change=BufferPolicy.NOTHING):
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
                self._buffer.write(must_exist=False)
            self._buffer.select((time_chunk,) + self._buffer.chunk_index[1:])
            if on_buffer_change & BufferPolicy.READ:
                # read new
                self._buffer.read()

        self._buffer_time_index = time_index % self._buffer.shape[0]

    def advance(self, on_buffer_change=BufferPolicy.NOTHING):
        self.select(self.time_index + 1, on_buffer_change)

    # def append(self, entry):
    #     self.get_buffer()[...] = entry
    #     self.next()
    #
    # # 'write'?
    # def flush(self):
    #     ...
    #
    # def read(self):
    #     # read current chunk?
    #     ...
