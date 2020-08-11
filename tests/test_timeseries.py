from io import BytesIO
from itertools import product
from pathlib import Path
import random
import unittest

import h5py as h5
import numpy as np

from pentinsula import ChunkBuffer, TimeSeries

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


if __name__ == '__main__':
    unittest.main()
