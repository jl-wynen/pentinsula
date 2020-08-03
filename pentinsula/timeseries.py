import h5py as h5
import numpy as np

from .chunkbuffer import ChunkBuffer
from .h5utils import open_or_pass_file


# name for 'entry for given time'? -> slot / element / timepoint

# registry for collective writes? -> benchmark, hack next()


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
        with open_or_pass_file(file, None, "r") as h5f:
            dataset = dataset if isinstance(dataset, h5.Dataset) else h5f[dataset]
            if dataset.chunks is None:
                raise RuntimeError(f"Dataset {dataset.name} is not chunked.")

            ntimes = dataset.shape[0]
            time_index = _normalise_time_index(time_index, ntimes)
            chunk_length = dataset.chunks[0]
            chunk_index = (time_index // chunk_length,) + (0,) * (len(dataset.shape) - 1)
            buffer = ChunkBuffer.load(h5f, dataset, chunk_index)

            series = cls(buffer)
            series._buffer_time_index = time_index % chunk_length
            return series

    # extend a series
    @classmethod
    def pick_up(cls, file, dataset):
        ...

    @property
    def time_index(self):
        return self._buffer.chunk_index[0] * self._buffer.shape[0] \
               + self._buffer_time_index

    # def get_buffer(self):
    #     return self.buffer[self.buffer_index]
    #
    # # advance / store / commit ??
    # def next(self):
    #     self.buffer_index += 1
    #     if self.buffer_index >= self.buffer.shape[0]:
    #         # check maxshape in first dim
    #
    #
    #         self.buffer.write(...)
    #         next_chunk_index = self.buffer.chunk_index
    #         next_chunk_index = (next_chunk_index[0]+1,) + next_chunk_index[1:]
    #         self.buffer.select(next_chunk_index)
    #
    # def append(self, entry):
    #     self.get_buffer()[...] = entry
    #     self.next()
    #
    # def __iter__(self):
    #     # iterate over chunks and entries of chunks
    #     ...
    #
    # # 'write'?
    # def flush(self):
    #     ...
    #
    # def read(self):
    #     # read current chunk?
    #     ...


def main():
    ts = TimeSeries()
    # print(type(ts.get_buffer()))
    x = ts.get_buffer()[...] = [1, 2]
    # x[...] = [1, 2]

    print(ts.buffer)


if __name__ == "__main__":
    main()