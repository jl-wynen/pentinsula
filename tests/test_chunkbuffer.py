from io import BytesIO
from itertools import product
from pathlib import Path
import random
import string
from tempfile import TemporaryDirectory
import unittest

import h5py as h5
import numpy as np

from pentinsula import ChunkBuffer
from pentinsula.chunkbuffer import _chunk_slices


N_REPEAT_TEST_CASE = 30


def repeat(n):
    def wrapper(func):
        def repeater(*args):
            for i in range(n):
                func(*args)

        return repeater

    return wrapper


def _capture_variables(**kwargs):
    return "  " + "\n  ".join(f"{name} := {value}" for name, value in kwargs.items())


def _random_int_tuple(a, b, n):
    return tuple(random.randint(a, b) for _ in range(n))


def _random_string(n):
    letters = string.ascii_letters
    return "".join(random.choice(letters) for _ in range(n))


def _chunk_indices(nchunks):
    yield from product(*tuple(map(range, nchunks)))


class TestChunkBuffer(unittest.TestCase):
    @repeat(N_REPEAT_TEST_CASE)
    def test_construction(self):
        # valid arguments, individual shape, dtype
        for dtype, maxshape in product((int, float, np.float32, np.int32, None), (None, (None,))):
            filename = _random_string(random.randint(1, 10))
            dataset_name = _random_string(random.randint(1, 10))
            shape = _random_int_tuple(1, 10, 4)
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
            shape = _random_int_tuple(1, 10, 4)
            array = np.random.uniform(-10, 10, shape).astype(dtype)
            buffer = ChunkBuffer(_random_string(random.randint(1, 10)), _random_string(random.randint(1, 10)),
                                 data=array)
            self.assertEqual(buffer.shape, shape)
            self.assertEqual(buffer.dtype, dtype if dtype else np.float64)
            np.testing.assert_allclose(array, buffer.data)

        # valid arguments, from array with reshaping
        in_shape = (10, 4)
        for target_shape in ((20, 2), (40,), (5, 8)):
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

    @repeat(N_REPEAT_TEST_CASE)
    def test_load(self):
        for ndim in range(1, 4):
            chunk_shape = _random_int_tuple(1, 10, ndim)
            nchunks = _random_int_tuple(1, 4, ndim)
            total_shape = tuple(n * c for n, c in zip(chunk_shape, nchunks))
            array = np.random.uniform(-10, 10, total_shape)

            stream = BytesIO()
            with h5.File(stream, "w") as h5f:
                h5f.create_dataset("data", data=array, chunks=chunk_shape)
            # valid, load all chunks
            for chunk_index in _chunk_indices(nchunks):
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

    @repeat(N_REPEAT_TEST_CASE)
    def test_dataset_creation(self):
        for ndim in range(1, 4):
            max_nchunks = _random_int_tuple(1, 4, ndim)
            for chunk_index in _chunk_indices(max_nchunks):
                chunk_shape = _random_int_tuple(1, 10, ndim)
                total_shape = tuple(n * (i + 1) for n, i in zip(chunk_shape, chunk_index))
                chunk_data = np.random.uniform(-10, 10, chunk_shape).astype(random.choice((float, int)))

                stream = BytesIO()
                buffer = ChunkBuffer(stream, "data", data=chunk_data, maxshape=(None,) * ndim)
                buffer.select(chunk_index)
                buffer.create_dataset(stream if random.random() < 0.5 else None, filemode="w", write=True)

                with h5.File(stream, "r") as h5f:
                    dataset = h5f["data"]
                    self.assertEqual(dataset.shape, total_shape)
                    self.assertEqual(dataset.chunks, chunk_shape)
                    self.assertEqual(dataset.dtype, chunk_data.dtype)
                    self.assertEqual(dataset.maxshape, buffer.maxshape)
                    np.testing.assert_allclose(ChunkBuffer.load(h5f, "data", chunk_index).data, chunk_data)

    @repeat(N_REPEAT_TEST_CASE)
    def test_select(self):
        for ndim in range(1, 5):
            chunk_shape = _random_int_tuple(1, 10, ndim)
            nchunks = _random_int_tuple(1, 4, ndim)
            maxshape = tuple(f * n if random.random() < 0.25 else None
                             for f, n in zip(nchunks, chunk_shape))
            buffer = ChunkBuffer("file", "data", shape=chunk_shape, maxshape=maxshape)

            # valid calls
            for chunk_index in _chunk_indices(nchunks):
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
            chunk_shape = _random_int_tuple(1, 10, ndim)
            nchunks = _random_int_tuple(1, 4, ndim)
            total_shape = tuple(n * c for n, c in zip(chunk_shape, nchunks))
            array = np.random.uniform(-10, 10, total_shape).astype(random.choice((int, float)))

            stream = BytesIO()
            with h5.File(stream, "w") as h5f:
                h5f.create_dataset("data", data=array, chunks=chunk_shape, maxshape=(None,) * ndim)

            # valid
            buffer = ChunkBuffer(stream, "data", shape=chunk_shape, dtype=array.dtype)
            for chunk_index in _chunk_indices(nchunks):
                # separate select / read
                buffer.select(chunk_index)
                buffer.read()
                np.testing.assert_allclose(buffer.data, array[_chunk_slices(chunk_index, chunk_shape)])

                # read with index arg
                buffer.data[...] = np.random.uniform(-20, 20, chunk_shape).astype(buffer.dtype)
                buffer.read(chunk_index)
                np.testing.assert_allclose(buffer.data, array[_chunk_slices(chunk_index, chunk_shape)])

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
            chunk_shape = _random_int_tuple(1, 10, ndim)
            nchunks = _random_int_tuple(1, 4, ndim)
            total_shape = tuple(n * c for n, c in zip(chunk_shape, nchunks))

            stream = BytesIO()
            chunk = np.random.uniform(-10, 10, chunk_shape).astype(random.choice((int, float)))
            file_content = np.random.uniform(-10, 10, total_shape).astype(chunk.dtype)
            with h5.File(stream, "w") as h5f:
                h5f.create_dataset("data", data=file_content, chunks=chunk_shape, maxshape=(None,) * ndim)

            buffer = ChunkBuffer(stream, "data", data=chunk)
            # valid indices
            for chunk_index in _chunk_indices(nchunks):
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
            chunk_shape = _random_int_tuple(1, 10, ndim)
            nchunks = _random_int_tuple(1, 5, ndim)
            chunks = []

            stream = BytesIO()
            with h5.File(stream, "w") as h5f:
                h5f.create_dataset("data", shape=chunk_shape, dtype=float,
                                   chunks=chunk_shape, maxshape=(None,) * ndim)

            buffer = ChunkBuffer(stream, "data", shape=chunk_shape, dtype=float)
            for chunk_index in _chunk_indices(nchunks):
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
            filename = Path(tempdir)/"test_file.h5"
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
                wrong_dataset = h5f.create_dataset("wrong_data", (1, ))
                with self.assertRaises(ValueError):
                    buffer.write(must_exist=False, dataset=wrong_dataset)


if __name__ == '__main__':
    unittest.main()
