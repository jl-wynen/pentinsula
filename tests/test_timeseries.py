from io import BytesIO
from itertools import product
from pathlib import Path
import random
import unittest

import h5py as h5
import numpy as np

from pentinsula import ChunkBuffer, TimeSeries, BufferPolicy

try:
    from .utils import random_string, random_int_tuple, product_range, repeat
except ImportError:
    from utils import random_string, random_int_tuple, product_range, repeat

N_REPEAT_TEST_CASE = 5


class MyTestCase(unittest.TestCase):
    @repeat(N_REPEAT_TEST_CASE)
    def test_construction(self):
        # valid arguments, individual shape, dtype
        for dtype, maxshape in product((int, float, np.float32, np.int32, None), (None, (None,))):
            filename = random_string(random.randint(1, 10))
            dataset_name = random_string(random.randint(1, 10))
            buffer_length = random.randint(1, 10)
            shape = random_int_tuple(1, 10, random.randint(0, 4))
            maxshape = maxshape if maxshape is None else maxshape * (len(shape) + 1)
            series = TimeSeries(filename, dataset_name, buffer_length,
                                shape=shape, dtype=dtype, maxshape=maxshape)
            self.assertEqual(series.filename, Path(filename))
            self.assertEqual(series.dataset_name.relative_to("/"), Path(dataset_name))
            self.assertEqual(series.buffer_length, buffer_length)
            self.assertEqual(series.shape, shape)
            self.assertEqual(series.item.shape, shape if shape else (1,))
            self.assertEqual(series.dtype, dtype if dtype else np.float64)
            self.assertEqual(series.item.dtype, dtype if dtype else np.float64)
            self.assertIsNone(series.maxtime)

        # valid arguments, from array
        for dtype in (int, float, np.float32, np.int32, None):
            shape = random_int_tuple(1, 10, random.randint(1, 5))
            array = np.random.uniform(-10, 10, shape).astype(dtype)
            buffer = ChunkBuffer("", "", data=array)
            series = TimeSeries(buffer)
            self.assertEqual(series.shape, shape[1:])
            self.assertEqual(buffer.dtype, dtype if dtype else np.float64)
            for i in range(shape[0]):
                series.select(i)
                np.testing.assert_allclose(series.item, array[i])

    @repeat(N_REPEAT_TEST_CASE)
    def test_load(self):
        for ndim in range(0, 4):
            shape = random_int_tuple(1, 10, ndim)
            buffer_length = random.randint(1, 10)
            nchunks = random.randint(1, 4)
            array = np.random.uniform(-10, 10, (nchunks * buffer_length,) + shape)

            stream = BytesIO()
            with h5.File(stream, "w") as h5f:
                h5f.create_dataset("data", data=array, chunks=(buffer_length,) + shape)

            # valid
            for time_index in range(buffer_length * nchunks):
                series = TimeSeries.load(stream, "data", time_index)
                self.assertEqual(series.dataset_name.relative_to("/"), Path("data"))
                self.assertEqual(series.buffer_length, buffer_length)
                self.assertEqual(series.shape, shape)
                self.assertEqual(series.item.shape, shape if shape else (1,))
                self.assertEqual(series.ndim, ndim)
                self.assertEqual(series.dtype, array.dtype)
                self.assertEqual(series.maxtime, buffer_length * nchunks)
                np.testing.assert_allclose(series.item, array[time_index])

            # invalid, load non-existent chunk
            # outside of maxshape, discoverable through maxshape
            with self.assertRaises(IndexError):
                TimeSeries.load(stream, "data", buffer_length * nchunks)
            # outside of maxshape, not discoverable through maxshape
            with self.assertRaises(IndexError):
                TimeSeries.load(stream, "data", buffer_length * (nchunks + 1))
            # within maxshape but not stored
            with h5.File(stream, "w") as h5f:
                h5f.create_dataset("partially_filled", shape=array.shape, chunks=(buffer_length,) + shape,
                                   maxshape=(None,) * array.ndim)
            with self.assertRaises(IndexError):
                TimeSeries.load(stream, "partially_filled", buffer_length * (nchunks + 1))

        # invalid, contiguous dataset
        stream = BytesIO()
        with h5.File(stream, "w") as h5f:
            h5f.create_dataset("data", data=np.random.uniform(-10, 10, (5, 3)))
        with self.assertRaises(RuntimeError):
            TimeSeries.load(stream, "data", 0)

    @repeat(N_REPEAT_TEST_CASE)
    def test_pick_up(self):
        for ndim in range(0, 4):
            shape = random_int_tuple(1, 10, ndim)
            buffer_length = random.randint(1, 10)
            nchunks = random.randint(1, 4)
            array = np.random.uniform(-10, 10, (nchunks * buffer_length,) + shape)

            for nstored in range(1, buffer_length * nchunks):
                stream = BytesIO()
                with h5.File(stream, "w") as h5f:
                    h5f.create_dataset("data", data=array[:nstored], chunks=(buffer_length,) + shape,
                                       maxshape=(None,) * array.ndim)

                series = TimeSeries.pick_up(stream, "data")
                self.assertEqual(series.dataset_name.relative_to("/"), Path("data"))
                self.assertEqual(series.buffer_length, buffer_length)
                self.assertEqual(series.shape, shape)
                self.assertEqual(series.item.shape, shape if shape else (1,))
                self.assertEqual(series.ndim, ndim)
                self.assertEqual(series.dtype, array.dtype)
                self.assertEqual(series.maxtime, None)
                self.assertEqual(series.time_index, nstored)

    @repeat(N_REPEAT_TEST_CASE)
    def test_dataset_creation(self):
        for ndim in range(0, 4):
            shape = random_int_tuple(1, 10, ndim)
            buffer_length = random.randint(1, 10)
            nchunks = random.randint(1, 4)
            dtype = random.choice((int, float))

            for time_index in range(0, buffer_length * nchunks):
                stream = BytesIO()
                series = TimeSeries(stream, "data", buffer_length, shape, dtype=dtype)
                series.select(time_index)
                series.create_dataset(write=False)

                loaded = TimeSeries.load(stream, "data", time_index)
                self.assertEqual(loaded.shape, shape)
                self.assertEqual(loaded.dtype, dtype)
                self.assertEqual(loaded.time_index, time_index)

    @repeat(N_REPEAT_TEST_CASE)
    def test_select(self):
        for ndim in range(0, 4):
            shape = random_int_tuple(1, 10, ndim)
            buffer_length = random.randint(1, 10)
            nchunks = random.randint(1, 4)
            base_data = np.random.uniform(-10, 10, (nchunks * buffer_length,) + shape)
            chunk_data = np.random.uniform(-10, 10, (buffer_length,) + shape)
            repeated_chunk_data = np.tile(chunk_data, (nchunks,) + (1,) * len(shape))

            stream = BytesIO()
            with h5.File(stream, "w") as h5f:
                h5f.create_dataset("data", data=base_data, chunks=(buffer_length,) + shape,
                                   maxshape=(None,) * base_data.ndim)

            # no changes to file or buffer with policy = NOTHING
            series = TimeSeries(ChunkBuffer(stream, "data", data=chunk_data))
            for time_index in range(buffer_length * nchunks):
                series.select(time_index, BufferPolicy.NOTHING)
                self.assertEqual(series.time_index, time_index)
                with h5.File(stream, "r") as h5f:
                    read = h5f["data"][()]
                    np.testing.assert_allclose(read, base_data)
                np.testing.assert_allclose(series.item, chunk_data[time_index % buffer_length])

            # reading fills at least the current item when crossing chunk boundary, file is unchanged
            series = TimeSeries(ChunkBuffer(stream, "data", data=chunk_data))
            for time_index in range(buffer_length * nchunks):
                series.select(time_index, BufferPolicy.READ)
                with h5.File(stream, "r") as h5f:
                    read = h5f["data"][()]
                    np.testing.assert_allclose(read, base_data)
                if time_index >= buffer_length:
                    np.testing.assert_allclose(series.item, base_data[time_index])
                else:
                    np.testing.assert_allclose(series.item, chunk_data[time_index % buffer_length])

            # writing overwrites the file content if crossing a chunk boundary but does not modify the buffer
            series = TimeSeries(ChunkBuffer(stream, "data", data=chunk_data))
            for time_index in range(buffer_length * nchunks):
                series.select(time_index, BufferPolicy.WRITE)
                with h5.File(stream, "r") as h5f:
                    read = h5f["data"][()]
                separatrix = time_index // buffer_length * buffer_length
                np.testing.assert_allclose(read[:separatrix], repeated_chunk_data[:separatrix])
                np.testing.assert_allclose(read[separatrix:], base_data[separatrix:])
                np.testing.assert_allclose(series.item, chunk_data[time_index % buffer_length])

            # reading and writing overwrites file and reads current content if crossing a chunk boundary
            with h5.File(stream, "w") as h5f:
                h5f["data"][...] = base_data
            series = TimeSeries(ChunkBuffer(stream, "data", data=chunk_data))
            for time_index in range(buffer_length * nchunks):
                series.select(time_index, BufferPolicy.READ | BufferPolicy.WRITE)
                if time_index < buffer_length:
                    # first chunk -> nothing has been read
                    np.testing.assert_allclose(series.item, chunk_data[time_index])
                else:
                    # later chunks -> chunk was read
                    np.testing.assert_allclose(series.item, base_data[time_index])
                # set item to fill file with repeated chunk data
                series.item[...] = repeated_chunk_data[time_index]

                with h5.File(stream, "r") as h5f:
                    read = h5f["data"][()]
                separatrix = time_index // buffer_length * buffer_length
                np.testing.assert_allclose(read[:separatrix], repeated_chunk_data[:separatrix])
                np.testing.assert_allclose(read[separatrix:], base_data[separatrix:])

    @repeat(N_REPEAT_TEST_CASE)
    def test_read_iter(self):
        for ndim in range(0, 4):
            shape = random_int_tuple(1, 10, ndim)
            buffer_length = random.randint(1, 10)
            nchunks = random.randint(1, 4)
            fill_level = random.randint(1, buffer_length) if nchunks > 1 else buffer_length
            array = np.random.uniform(-10, 10, ((nchunks - 1) * buffer_length + fill_level,) + shape)

            stream = BytesIO()
            with h5.File(stream, "w") as h5f:
                h5f.create_dataset("data", data=array, chunks=(buffer_length,) + shape)

            series = TimeSeries.load(stream, "data", 0)
            # all times
            file_arg = None if random.random() < 0.5 else stream
            dataset_arg = None if random.random() < 0.5 else "data"
            for desired_index, (time_index, item) in enumerate(series.read_iter(file=file_arg,
                                                                                dataset=dataset_arg)):
                self.assertEqual(time_index, desired_index)
                np.testing.assert_allclose(item, array[time_index])
            with h5.File(stream, "r") as h5f:
                np.testing.assert_allclose(h5f["data"][()], array)

            # random slice
            times = slice(random.randint(0, buffer_length * nchunks),
                          random.randint(1, array.shape[0]),
                          random.randint(1, buffer_length * nchunks // 2))
            for desired_index, (time_index, item) in zip(range(times.start, times.stop, times.step),
                                                         series.read_iter(times,
                                                                          file=file_arg,
                                                                          dataset=dataset_arg)):
                self.assertEqual(time_index, desired_index)
                np.testing.assert_allclose(item, array[time_index])

            # stop index out of bounds
            with self.assertRaises(ValueError):
                for _ in series.read_iter(slice(0, buffer_length * nchunks * 2)):
                    break

    @repeat(N_REPEAT_TEST_CASE)
    def test_write_iter(self):
        for ndim in range(0, 4):
            shape = random_int_tuple(1, 10, ndim)
            buffer_length = random.randint(1, 10)
            nchunks = random.randint(1, 4)
            fill_level = random.randint(1, buffer_length)
            array = np.random.uniform(-10, 10, ((nchunks - 1) * buffer_length + fill_level,) + shape)

            stream = BytesIO()
            series = TimeSeries(stream, "data", buffer_length, shape)
            series.create_dataset(write=False)

            file_arg = None if random.random() < 0.5 else stream
            dataset_arg = None if random.random() < 0.5 else "data"
            for desired_index, (time_index, item) in zip(range(array.shape[0]),
                                                         series.write_iter(flush=True,
                                                                           file=file_arg,
                                                                           dataset=dataset_arg)):
                self.assertEqual(time_index, desired_index)
                item[...] = array[time_index]

            with h5.File(stream, "r") as h5f:
                np.testing.assert_allclose(h5f["data"][()], array)


if __name__ == '__main__':
    unittest.main()
