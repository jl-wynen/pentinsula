"""
General utilities for handling h5py.File and h5py.Dataset.
Mainly for internal usage.
"""

from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import Optional, Union

import h5py as h5

from .types import File, Dataset


@contextmanager
def open_or_pass_file(file: Optional[File],
                      stored_filename: Optional[Union[str, Path, BytesIO]],
                      *args, **kwargs) -> h5.File:
    """
    Open a file or pass an existing file along.

    :param file: Either an h5py.File or an object pointing to a file.
    :param stored_filename: If given, the name of argument 'file' must match this.
    :param args: Positional arguments to pass to h5py.File.
    :param kwargs: Keyword arguments to pass to h5py.File.
    :return: Context manager for the file.
    """

    if stored_filename is not None:
        if file is not None and not isinstance(file, BytesIO):
            filename = Path(file.filename) if isinstance(file, h5.File) else Path(file)
            if filename != stored_filename:
                raise ValueError(f"Argument file ({filename}) does not match stored file ({stored_filename}.")
        else:
            file = stored_filename
    else:
        if file is None:
            raise ValueError("Arguments file and stored_filename cannot both be None.")

    yield file if isinstance(file, h5.File) else h5.File(file, *args, **kwargs)


@contextmanager
def open_or_pass_dataset(file: Optional[File],
                         dataset: Dataset,
                         stored_filename: Optional[Union[str, Path, BytesIO]] = None,
                         *args, **kwargs) -> h5.Dataset:
    """
    Open a dataset or pass an existing dataset along.

    :param file: Either an h5py.File or an object pointing to a file.
    :param dataset: Either an h5py.Dataset or a path in file pointing to a dataset.
    :param stored_filename: If given, the name of argument 'file' must match this.
    :param args: Positional arguments to pass to h5py.File.
    :param kwargs: Keyword arguments to pass to h5py.File.
    :return: Context manager for the dataset.
    """

    with open_or_pass_file(file, stored_filename, *args, **kwargs) as h5f:
        dataset = dataset if isinstance(dataset, h5.Dataset) else h5f[str(dataset)]
        if dataset.chunks is None:
            raise RuntimeError(f"Dataset {dataset.name} is not chunked.")
        yield dataset


def get_dataset_name(dataset: Dataset) -> Path:
    """
    Return the name of a dataset.
    """

    name = Path(dataset.name if isinstance(dataset, h5.Dataset) else dataset)
    if not name.is_absolute():
        name = "/" / name
    return name
