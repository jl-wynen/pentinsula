from io import BytesIO
from itertools import chain, product
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest

import h5py as h5
import numpy as np

from pentinsula import ChunkBuffer
from pentinsula.chunkbuffer import _chunk_slices

try:
    from .utils import random_string, capture_variables, random_int_tuple, product_range, repeat
except ImportError:
    from utils import random_string, capture_variables, random_int_tuple, product_range, repeat


N_REPEAT_TEST_CASE = 5


class TestChunkBuffer(unittest.TestCase):
    @repeat(N_REPEAT_TEST_CASE)
    def test_construction(self):
        # valid arguments, individual shape, dtype
        for dtype, maxshape in product((int, float, np.float32, np.int32, None), (None, (None,))):
            filename = random_string(random.randint(1, 10))
            dataset_name = random_string(random.randint(1, 10))
            shape = random_int_tuple(1, 10, 4)
            maxshape = maxshape if maxshape is None else maxshape * len(shape)
            buffer = ChunkBuffer(filename, dataset_name,
                                 shape=shape, dtype=dtype,
                                 maxshape=maxshape)
            self.assertEqual(buffer.filename, Path(filename))
            self.assertEqual(buffer.dataset_name.relative_to("/"), Path(dataset_name))
            self.assertEqual(buffer.shape, shape)
            self.assertEqual(buffer.data.shape, shape)
            self.assertEqual(buffer.dtype, dtype if dtype else np.float64)
            self.assertEqual(buffer.data.dtype, dtype if dtype else np.float64)
            self.assertEqual(buffer.maxshape, (None,) * len(shape))

        # valid arguments, from array
        for dtype in (int, float, np.float32, np.int32, None):
            shape = random_int_tuple(1, 10, 4)
            array = np.random.uniform(-10, 10, shape).astype(dtype)
            buffer = ChunkBuffer(random_string(random.randint(1, 10)), random_string(random.randint(1, 10)),
                                 data=array)
            self.assertEqual(buffer.shape, shape)
            self.assertEqual(buffer.dtype, dtype if dtype else np.float64)
            np.testing.assert_allclose(array, buffer.data)

        # valid arguments, from array with reshaping
        in_shape = (10, 4)
        for target_shape in ((20, 2), (40,), (5, 8)):
            array = np.random.uniform(-10, 10, in_shape)
            buffer = ChunkBuffer(random_string(random.randint(1, 10)), random_string(random.randint(1, 10)),
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

    @repeat(N_REPEAT_TEST_CASE)
    def test_load(self):
        for ndim in range(1, 4):
            chunk_shape = random_int_tuple(1, 10, ndim)
            nchunks = random_int_tuple(1, 4, ndim)
            total_shape = tuple(n * c for n, c in zip(chunk_shape, nchunks))
            array = np.random.uniform(-10, 10, total_shape)

            stream = BytesIO()
            with h5.File(stream, "w") as h5f:
                h5f.create_dataset("data", data=array, chunks=chunk_shape)
            # valid, load all chunks, positive indices
            for chunk_index in product_range(nchunks):
                buffer = ChunkBuffer.load(stream, "data", chunk_index)
                np.testing.assert_allclose(buffer.data, array[_chunk_slices(chunk_index, chunk_shape)],
                                           err_msg=capture_variables(ndim=ndim,
                                                                      chunk_shape=chunk_shape,
                                                                      nchunks=nchunks,
                                                                      chunk_index=chunk_index))
            # negative index
            neg_index = (-1,) * ndim
            pos_index = tuple(n - 1 for n in nchunks)
            buffer = ChunkBuffer.load(stream, "data", neg_index)
            np.testing.assert_allclose(buffer.data, array[_chunk_slices(pos_index, chunk_shape)],
                                       err_msg=capture_variables(ndim=ndim,
                                                                  chunk_shape=chunk_shape,
                                                                  nchunks=nchunks,
                                                                  chunk_index=neg_index))

            # invalid, load non-existent chunk
            # outside of maxshape, discoverable through maxshape
            with self.assertRaises(IndexError):
                ChunkBuffer.load(stream, "data", nchunks)
            # outside of maxshape, not discoverable through maxshape
            with self.assertRaises(IndexError):
                ChunkBuffer.load(stream, "data", (nchunks[0] + 1,) + nchunks[1:])
            # within maxshape but not stored
            with h5.File(stream, "w") as h5f:
                h5f.create_dataset("partially_filled", shape=total_shape, chunks=chunk_shape,
                                   maxshape=tuple(n * 2 for n in total_shape))
            with self.assertRaises(IndexError):
                ChunkBuffer.load(stream, "partially_filled", (nchunks[0] + 1,) + nchunks[1:])

        # invalid, contiguous dataset
        stream = BytesIO()
        with h5.File(stream, "w") as h5f:
            h5f.create_dataset("data", data=np.random.uniform(-10, 10, (5, 3)))
        with self.assertRaises(RuntimeError):
            ChunkBuffer.load(stream, "data", (0, 0))

    @repeat(N_REPEAT_TEST_CASE)
    def test_dataset_creation(self):
        for ndim in range(1, 4):
            max_nchunks = random_int_tuple(1, 4, ndim)
            for chunk_index in product_range(max_nchunks):
                chunk_shape = random_int_tuple(1, 10, ndim)
                for fill_level in chain((None,), product_range((1,) * ndim, chunk_shape)):
                    if fill_level is None:
                        total_shape = tuple(n * (i + 1)
                                            for n, i in zip(chunk_shape, chunk_index))
                    else:
                        total_shape = tuple(n * i + fl
                                            for n, i, fl in zip(chunk_shape, chunk_index, fill_level))
                    chunk_data = np.random.uniform(-10, 10, chunk_shape).astype(random.choice((float, int)))

                    stream = BytesIO()
                    buffer = ChunkBuffer(stream, "data", data=chunk_data, maxshape=(None,) * ndim)
                    buffer.select(chunk_index)
                    buffer.create_dataset(stream if random.random() < 0.5 else None, filemode="w",
                                          write=True, fill_level=fill_level)

                    with h5.File(stream, "r") as h5f:
                        dataset = h5f["data"]
                        self.assertEqual(dataset.shape, total_shape)
                        self.assertEqual(dataset.chunks, chunk_shape)
                        self.assertEqual(dataset.dtype, chunk_data.dtype)
                        self.assertEqual(dataset.maxshape, buffer.maxshape)
                        fill_slices = tuple(map(slice, fill_level)) if fill_level is not None else ...
                        np.testing.assert_allclose(ChunkBuffer.load(h5f, "data", chunk_index).data[fill_slices],
                                                   chunk_data[fill_slices])

    @repeat(N_REPEAT_TEST_CASE)
    def test_select(self):
        for ndim in range(1, 5):
            chunk_shape = random_int_tuple(1, 10, ndim)
            nchunks = random_int_tuple(1, 4, ndim)
            maxshape = tuple(f * n if random.random() < 0.25 else None
                             for f, n in zip(nchunks, chunk_shape))
            buffer = ChunkBuffer("file", "data", shape=chunk_shape, maxshape=maxshape)

            # valid calls
            for chunk_index in product_range(nchunks):
                buffer.select(chunk_index)
                self.assertEqual(buffer.chunk_index, chunk_index)

            def random_chunk_index():
                return tuple(map(lambda n: random.randint(0, n - 1), nchunks))

            # invalid number of dimensions
            too_many_dims = random_chunk_index() + (0,)
            with self.assertRaises(IndexError):
                buffer.select(too_many_dims)
            too_few_dims = random_chunk_index()[:-1]
            with self.assertRaises(IndexError):
                buffer.select(too_few_dims)

            # index out of bounds
            for dim in range(ndim):
                chunk_index = random_chunk_index()
                negative = chunk_index[:dim] + (random.randint(-10, -1),) + chunk_index[dim + 1:]
                with self.assertRaises(IndexError):
                    buffer.select(negative)
                if maxshape[dim] is not None:
                    too_large = chunk_index[:dim] + (nchunks[dim] + random.randint(1, 10),) + chunk_index[dim + 1:]
                    with self.assertRaises(IndexError):
                        buffer.select(too_large)

    @repeat(N_REPEAT_TEST_CASE)
    def test_read(self):
        for ndim in range(1, 4):
            chunk_shape = random_int_tuple(1, 10, ndim)
            nchunks = random_int_tuple(1, 4, ndim)
            for fill_level in chain((None,), product_range((1,) * ndim, chunk_shape)):
                if fill_level is None:
                    total_shape = tuple(n * c for n, c in zip(chunk_shape, nchunks))
                else:
                    total_shape = tuple(n * (c - 1) + fl
                                        for n, c, fl in zip(chunk_shape, nchunks, fill_level))
                array = np.random.uniform(-10, 10, total_shape).astype(random.choice((int, float)))
                stream = BytesIO()
                with h5.File(stream, "w") as h5f:
                    h5f.create_dataset("data", data=array, chunks=chunk_shape, maxshape=(None,) * ndim)

                def validate_fill_level(chunk_index, actual_fill_level):
                    target_fill_level = chunk_shape if fill_level is None else fill_level
                    for idx, n, length, actual, target in zip(chunk_index, nchunks, chunk_shape,
                                                              actual_fill_level, target_fill_level):
                        if idx == n - 1:
                            self.assertEqual(actual, target)
                        else:
                            self.assertEqual(actual, length)

                # valid
                buffer = ChunkBuffer(stream, "data", shape=chunk_shape, dtype=array.dtype)
                for chunk_index in product_range(nchunks):
                    # separate select / read
                    buffer.select(chunk_index)
                    read_fill_level = buffer.read()
                    validate_fill_level(chunk_index, read_fill_level)
                    fill_slices = tuple(map(slice, fill_level)) if fill_level is not None else ...
                    np.testing.assert_allclose(buffer.data[fill_slices],
                                               array[_chunk_slices(chunk_index, chunk_shape)][fill_slices])

                    # read with index arg
                    buffer.data[...] = np.random.uniform(-20, 20, chunk_shape).astype(buffer.dtype)
                    read_fill_level = buffer.read(chunk_index)
                    validate_fill_level(chunk_index, read_fill_level)
                    np.testing.assert_allclose(buffer.data[fill_slices],
                                               array[_chunk_slices(chunk_index, chunk_shape)][fill_slices])

                # index out of bounds
                with self.assertRaises(IndexError):
                    buffer.read(nchunks)

            # dataset does not exist
            buffer = ChunkBuffer(stream, "wrong_name", shape=chunk_shape, dtype=array.dtype)
            with self.assertRaises(KeyError):
                buffer.read()

            # invalid chunk shape
            buffer = ChunkBuffer(stream, "data", shape=tuple(random.randint(1, 10) + n for n in chunk_shape))
            with self.assertRaises(RuntimeError):
                buffer.read()

            # invalid datatype
            buffer = ChunkBuffer(stream, "data", shape=chunk_shape, dtype=np.float32)
            with self.assertRaises(RuntimeError):
                buffer.read()

            # invalid maxshape
            buffer = ChunkBuffer(stream, "data", shape=chunk_shape, dtype=array.dtype, maxshape=chunk_shape)
            with self.assertRaises(RuntimeError):
                buffer.read()

    @repeat(N_REPEAT_TEST_CASE)
    def test_write_overwrite(self):
        for ndim in range(1, 4):
            chunk_shape = random_int_tuple(1, 10, ndim)
            nchunks = random_int_tuple(1, 4, ndim)
            total_shape = tuple(n * c for n, c in zip(chunk_shape, nchunks))

            stream = BytesIO()
            chunk = np.random.uniform(-10, 10, chunk_shape).astype(random.choice((int, float)))
            file_content = np.random.uniform(-10, 10, total_shape).astype(chunk.dtype)
            with h5.File(stream, "w") as h5f:
                h5f.create_dataset("data", data=file_content, chunks=chunk_shape, maxshape=(None,) * ndim)

            buffer = ChunkBuffer(stream, "data", data=chunk)
            # valid indices
            for chunk_index in product_range(nchunks):
                with h5.File(stream, "a") as h5f:
                    h5f["data"][...] = file_content

                buffer.select(chunk_index)
                buffer.write(must_exist=True)
                desired_file_content = file_content.copy()
                desired_file_content[_chunk_slices(chunk_index, chunk_shape)] = chunk
                with h5.File(stream, "r") as h5f:
                    np.testing.assert_allclose(h5f["data"][()], desired_file_content)

            # index out of bounds
            for dim in range(ndim):
                chunk_index = tuple(map(lambda n: random.randint(0, n - 1), nchunks))
                chunk_index = chunk_index[:dim] + (nchunks[dim] + random.randint(1, 10),) + chunk_index[dim + 1:]
                buffer.select(chunk_index)
                with self.assertRaises(RuntimeError):
                    buffer.write(must_exist=True)

    @repeat(N_REPEAT_TEST_CASE)
    def test_write_extend(self):
        for ndim in range(1, 4):
            chunk_shape = random_int_tuple(1, 10, ndim)
            nchunks = random_int_tuple(1, 5, ndim)
            chunks = []

            stream = BytesIO()
            with h5.File(stream, "w") as h5f:
                h5f.create_dataset("data", shape=chunk_shape, dtype=float,
                                   chunks=chunk_shape, maxshape=(None,) * ndim)

            buffer = ChunkBuffer(stream, "data", shape=chunk_shape, dtype=float)
            for chunk_index in product_range(nchunks):
                chunks.append((_chunk_slices(chunk_index, chunk_shape), np.random.uniform(-10, 10, chunk_shape)))
                buffer.select(chunk_index)
                buffer.data[...] = chunks[-1][1]
                buffer.write(must_exist=False)

                with h5.File(stream, "r") as f:
                    dataset = f["data"]
                    for chunk_slice, expected in chunks:
                        np.testing.assert_allclose(dataset[chunk_slice], expected)

    def test_real_files(self):
        with TemporaryDirectory() as tempdir:
            filename = Path(tempdir) / "test_file.h5"
            chunk_shape = (1, 2, 3)
            array = np.random.uniform(-10, 10, chunk_shape)
            buffer = ChunkBuffer(filename, "data", data=array)
            buffer.create_dataset(filemode="w")

            self.assertTrue(filename.exists())
            with h5.File(filename, "r") as h5f:
                np.testing.assert_allclose(h5f["data"][()], array)

            # extend dataset with stored filename
            array = np.random.uniform(-10, 10, chunk_shape)
            buffer.select((1, 0, 0))
            buffer.data[...] = array
            buffer.write(must_exist=False)
            with h5.File(filename, "r") as h5f:
                np.testing.assert_allclose(h5f["data"][1:, :, :], array)

            # extend dataset with passed in filename
            array = np.random.uniform(-10, 10, chunk_shape)
            buffer.select((1, 1, 0))
            buffer.data[...] = array
            buffer.write(must_exist=False, file=filename)
            with h5.File(filename, "r") as h5f:
                np.testing.assert_allclose(h5f["data"][1:, 2:, :], array)

            # extend dataset with passed in dataset
            array = np.random.uniform(-10, 10, chunk_shape)
            buffer.select((1, 0, 1))
            buffer.data[...] = array
            with h5.File(filename, "r+") as h5f:
                dataset = h5f["data"]
                buffer.write(must_exist=False, dataset=dataset)
                np.testing.assert_allclose(dataset[1:, :2, 3:], array)

            # wrong filename
            with self.assertRaises(ValueError):
                buffer.write(must_exist=False, file="wrong_file.h5")

            # wrong dataset
            with h5.File(filename, "a") as h5f:
                wrong_dataset = h5f.create_dataset("wrong_data", (1,))
                with self.assertRaises(ValueError):
                    buffer.write(must_exist=False, dataset=wrong_dataset)


if __name__ == '__main__':
    unittest.main()
