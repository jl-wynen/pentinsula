from contextlib import contextmanager
from io import BytesIO
from pathlib import Path

import h5py as h5


@contextmanager
def open_or_pass_file(file, stored_filename, *args, **kwargs):
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
def open_or_pass_dataset(file, dataset, stored_filename=None, *args, **kwargs):
    with open_or_pass_file(file, stored_filename, *args, **kwargs) as h5f:
        dataset = dataset if isinstance(dataset, h5.Dataset) else h5f[str(dataset)]
        if dataset.chunks is None:
            raise RuntimeError(f"Dataset {dataset.name} is not chunked.")
        yield dataset


def get_dataset_name(dataset):
    name = Path(dataset.name if isinstance(dataset, h5.Dataset) else dataset)
    if not name.is_absolute():
        name = "/"/name
    return name
