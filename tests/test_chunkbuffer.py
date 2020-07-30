from itertools import product
from pathlib import Path
import random
import string
import unittest

import numpy as np

from pentinsula import ChunkBuffer


def _random_int_tuple(a, b, n):
    return tuple(random.randint(a, b) for _ in range(n))


def _random_string(n):
    letters = string.ascii_letters
    return "".join(random.choice(letters) for _ in range(n))


class TestChunkBuffer(unittest.TestCase):
    def test_construction(self):
        # valid arguments, individual shape, dtype
        for dtype, maxshape in product((int, float, np.float32, np.int32, None), (None, (None,))):
            filename = _random_string(random.randint(1, 10))
            dataset_name = _random_string(random.randint(1, 10))
            shape = _random_int_tuple(1, 10, 4)
            maxshape = maxshape if maxshape is None else maxshape*len(shape)
            buffer = ChunkBuffer(filename, dataset_name,
                                 shape=shape, dtype=dtype,
                                 maxshape=maxshape)
            self.assertEqual(buffer.filename, Path(filename))
            self.assertEqual(buffer.dataset_name, dataset_name)
            self.assertEqual(buffer.shape, shape)
            self.assertEqual(buffer.data.shape, shape)
            self.assertEqual(buffer.dtype, dtype if dtype else np.float64)
            self.assertEqual(buffer.data.dtype, dtype if dtype else np.float64)
            self.assertEqual(buffer.maxshape, (None,)*len(shape))

        # valid arguments, from array
        for dtype in (int, float, np.float32, np.int32, None):
            shape = _random_int_tuple(1, 10, 4)
            array = np.random.uniform(-10, 10, shape).astype(dtype)
            buffer = ChunkBuffer(_random_string(random.randint(1, 10)), _random_string(random.randint(1, 10)),
                                 data=array)
            self.assertEqual(buffer.shape, shape)
            self.assertEqual(buffer.dtype, dtype if dtype else np.float64)
            np.testing.assert_allclose(array, buffer.data)

        # valid arguments, from array with reshaping
        in_shape = (10, 4)
        for target_shape in ((20, 2), (40, ), (5, 8)):
            array = np.random.uniform(-10, 10, in_shape)
            buffer = ChunkBuffer(_random_string(random.randint(1, 10)), _random_string(random.randint(1, 10)),
                                 data=array, shape=target_shape)
            self.assertEqual(buffer.shape, target_shape)

        # invalid reshaping
        array = np.random.uniform(-10, 10, (4, 10))
        with self.assertRaises(ValueError):
            ChunkBuffer("test.h5", "test", data=array, shape=(3,))

        # invalid maxshape
        with self.assertRaises(ValueError):
            ChunkBuffer("test.h5", "test", shape=(1, 2), maxshape=(1,))
        with self.assertRaises(ValueError):
            ChunkBuffer("test.h5", "test", shape=(1, 2), maxshape=(1, 2, 3))


if __name__ == '__main__':
    unittest.main()
