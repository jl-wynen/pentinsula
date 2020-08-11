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


if __name__ == '__main__':
    unittest.main()
