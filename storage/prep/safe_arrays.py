"""Restricted data-only readers for pinned publisher split archives."""

import io
import pickle
import struct
from collections import OrderedDict
import numpy as np


class SparseCSR:
    """Data-only state holder; never invokes a SciPy constructor or method."""

    def __setstate__(self, state):
        allowed = {
            "_shape",
            "maxprint",
            "data",
            "indices",
            "indptr",
            "_has_sorted_indices",
            "_has_canonical_format",
        }
        if not isinstance(state, dict) or set(state) - allowed:
            raise ValueError("Unapproved sparse split state")
        shape = state["_shape"]
        data, indices, indptr = (state[k] for k in ("data", "indices", "indptr"))
        if len(shape) != 2 or any(not isinstance(n, int) or n < 0 for n in shape):
            raise ValueError("Invalid sparse split shape")
        if any(
            not isinstance(v, np.ndarray) or v.ndim != 1 or v.dtype.hasobject
            for v in (data, indices, indptr)
        ):
            raise ValueError("Invalid sparse split arrays")
        if (
            indices.dtype.kind not in "iu"
            or indptr.dtype.kind not in "iu"
            or len(data) != len(indices)
            or len(indptr) != shape[0] + 1
        ):
            raise ValueError("Invalid sparse split array lengths/types")
        if (
            indptr[0] != 0
            or indptr[-1] != len(data)
            or (indptr[1:] < indptr[:-1]).any()
            or (indices < 0).any()
            or (indices >= shape[1]).any()
        ):
            raise ValueError("Invalid sparse split indices")
        self.shape, self.data, self.indices, self.indptr = shape, data, indices, indptr


def tensor(
    storage, offset, shape, stride, requires_grad=False, hooks=None, metadata=None
):
    if (
        requires_grad
        or hooks
        or metadata
        or len(shape) != len(stride)
        or len(shape) > 4
    ):
        raise ValueError("Unsupported split tensor state")
    if any(not isinstance(n, int) or n < 0 for n in (*shape, *stride, offset)):
        raise ValueError("Invalid tensor dimensions")
    return np.ndarray(
        shape,
        dtype=storage.dtype,
        buffer=storage,
        offset=offset * storage.dtype.itemsize,
        strides=tuple(s * storage.dtype.itemsize for s in stride),
    ).copy()


class StorageUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        types = {
            "LongStorage": "<i8",
            "IntStorage": "<i4",
            "FloatStorage": "<f4",
            "DoubleStorage": "<f8",
            "ByteStorage": "u1",
            "BoolStorage": "?",
        }
        if module != "torch" or name not in types:
            raise ValueError("Unapproved storage constructor")
        return types[name]

    def persistent_load(self, value):
        if (
            len(value) != 6
            or value[0] != "storage"
            or value[3] != "cpu"
            or value[5] is not None
        ):
            raise ValueError("Only standalone CPU split storages are supported")
        return value


def storage_from_bytes(body):
    # torch.storage._load_from_bytes would invoke unrestricted pickle internally.
    # Decode this narrow legacy storage format ourselves, without importing torch.
    file = io.BytesIO(body)
    reader = StorageUnpickler(file)
    if reader.load() != 119547037146038801333356 or reader.load() != 1001:
        raise ValueError("Unsupported tensor storage format")
    system = reader.load()
    if system.get("little_endian") is not True:
        raise ValueError("Unexpected tensor endianness")
    _, dtype, key, _, count, _ = reader.load()
    if reader.load() != [key] or struct.unpack("<Q", file.read(8))[0] != count:
        raise ValueError("Storage length/key disagreement")
    dtype = np.dtype(dtype)
    raw = file.read()
    if len(raw) != count * dtype.itemsize:
        raise ValueError("Storage byte count disagreement")
    return np.frombuffer(raw, dtype=dtype)


class DataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        permitted = {
            ("scipy.sparse.csr", "csr_matrix"): SparseCSR,
            ("scipy.sparse._csr", "csr_matrix"): SparseCSR,
            ("torch._utils", "_rebuild_tensor_v2"): tensor,
            ("torch._utils", "_rebuild_tensor"): tensor,
            ("torch.storage", "_load_from_bytes"): storage_from_bytes,
            ("collections", "OrderedDict"): OrderedDict,
            ("numpy", "ndarray"): np.ndarray,
            ("numpy", "dtype"): np.dtype,
            ("numpy.core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
            (
                "numpy._core.multiarray",
                "_reconstruct",
            ): np._core.multiarray._reconstruct,
            ("numpy.core.multiarray", "scalar"): np._core.multiarray.scalar,
            ("numpy._core.multiarray", "scalar"): np._core.multiarray.scalar,
            ("numpy.core.numeric", "_frombuffer"): np._core.numeric._frombuffer,
            ("numpy._core.numeric", "_frombuffer"): np._core.numeric._frombuffer,
        }
        if (module, name) not in permitted:
            raise ValueError(f"Unapproved pickle constructor: {module}.{name}")
        return permitted[module, name]

    def persistent_load(self, _):
        raise ValueError("Persistent pickle references are prohibited")


def load(data, *, numpy_file=False):
    file = io.BytesIO(data)
    if numpy_file:
        version = np.lib.format.read_magic(file)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(file)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(file)
        else:
            raise ValueError("Unsupported NPY split format")
        if not dtype.hasobject:
            return np.load(io.BytesIO(data), allow_pickle=False)
    return DataUnpickler(file).load()


def describe(value, depth=0):
    if isinstance(value, SparseCSR):
        return {"type": "csr", "shape": list(value.shape), "nnz": len(value.data)}
    if isinstance(value, np.ndarray):
        result = {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        if value.dtype.hasobject and depth < 4:
            result["first"] = [describe(v, depth + 1) for v in value[:2]]
        return result
    if isinstance(value, (list, tuple)):
        return {
            "type": type(value).__name__,
            "length": len(value),
            "first": [describe(v, depth + 1) for v in value[:2]] if depth < 4 else [],
        }
    if isinstance(value, dict):
        return {
            "type": "dict",
            "length": len(value),
            "first": [
                [str(k), describe(v, depth + 1)] for k, v in list(value.items())[:2]
            ]
            if depth < 4
            else [],
        }
    return {"type": type(value).__name__}
