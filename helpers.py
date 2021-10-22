import numpy as np

from io import BytesIO
from typing import cast


def bytes_to_ndarray(tensor: bytes) -> np.ndarray:
    """Deserialize NumPy ndarray from bytes."""
    bytes_io = BytesIO(tensor)
    ndarray_deserialized = np.load(bytes_io, allow_pickle=False)
    return cast(np.ndarray, ndarray_deserialized)


def ndarray_to_bytes(ndarray: np.ndarray) -> bytes:
    """Serialize NumPy ndarray to bytes."""
    bytes_io = BytesIO()
    np.save(bytes_io, ndarray, allow_pickle=False)
    return bytes_io.getvalue()