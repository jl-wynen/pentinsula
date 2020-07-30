from io import BytesIO
from itertools import product
from pathlib import Path
import random
import string
import unittest

import h5py as h5
import numpy as np

from pentinsula import ChunkBuffer
from pentinsula.chunkbuffer import _chunk_slices


def _capture_variables(**kwargs):
    return "  " + "\n  ".join(f"{name} := {value}" for name, value in kwargs.items())


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

    def test_load(self):
        for ndim in range(1, 4):
            chunk_shape = _random_int_tuple(1, 10, ndim)
            nchunks = _random_int_tuple(1, 4, ndim)
            total_shape = tuple(n*c for n, c in zip(chunk_shape, nchunks))
            array = np.random.uniform(-10, 10, total_shape)

            stream = BytesIO()
            with h5.File(stream, "w") as h5f:
                h5f.create_dataset("data", data=array, chunks=chunk_shape)
            # valid, load all chunks
            for chunk_index in product(*tuple(map(range, nchunks))):
                buffer = ChunkBuffer.load(stream, "data", chunk_index)
                np.testing.assert_allclose(buffer.data, array[_chunk_slices(chunk_index, chunk_shape)],
                                           err_msg=_capture_variables(ndim=ndim,
                                                                      chunk_shape=chunk_shape,
                                                                      nchunks=nchunks,
                                                                      chunk_index=chunk_index))

            # invalid, load non-existent chunk
            with self.assertRaises(IndexError):
                ChunkBuffer.load(stream, "data", nchunks)

        # invalid, contiguous dataset
        stream = BytesIO()
        with h5.File(stream, "w") as h5f:
            h5f.create_dataset("data", data=np.random.uniform(-10, 10, (5, 3)))
        with self.assertRaises(RuntimeError):
            ChunkBuffer.load(stream, "data", (0, 0))


if __name__ == '__main__':
    unittest.main()
