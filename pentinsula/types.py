"""
Defines type aliases commonly used in pentinsula.
"""

from io import BytesIO
from pathlib import Path
from typing import Union, Tuple

import h5py as h5
from numpy import dtype

File = Union[str, Path, h5.File, BytesIO]
Dataset = Union[str, Path, h5.Dataset]
Shape = Tuple[int, ...]
DType = Union[int, float, complex, dtype]
