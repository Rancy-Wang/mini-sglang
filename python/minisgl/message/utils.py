from __future__ import annotations

from typing import Any, Dict, Type

import numpy as np
import torch


def _serialize_any(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _serialize_any(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return type(value)(_serialize_any(v) for v in value)
    elif isinstance(value, (int, float, str, type(None), bool, bytes)):
        return value
    else:
        return serialize_type(value)


def serialize_type(self) -> Dict:
    # find all member variables
    serialized = {}

    if isinstance(self, torch.Tensor):
        if not self.is_cpu:
            raise ValueError("Message tensors must reside on CPU before serialization.")
        tensor = self.contiguous()
        serialized["__type__"] = "Tensor"
        serialized["buffer"] = tensor.numpy().tobytes()
        serialized["dtype"] = str(tensor.dtype)
        serialized["shape"] = list(tensor.shape)
        return serialized

    # normal type
    serialized["__type__"] = self.__class__.__name__
    for k, v in self.__dict__.items():
        serialized[k] = _serialize_any(v)
    return serialized


def _deserialize_any(cls_map: Dict[str, Type], data: Any) -> Any:
    if isinstance(data, dict):
        if "__type__" in data:
            return deserialize_type(cls_map, data)
        else:
            return {k: _deserialize_any(cls_map, v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(_deserialize_any(cls_map, d) for d in data)
    elif isinstance(data, (int, float, str, type(None), bool, bytes)):
        return data
    else:
        raise ValueError(f"Cannot deserialize type {type(data)}")


def deserialize_type(cls_map: Dict[str, Type], data: Dict) -> Any:
    type_name = data["__type__"]
    if type_name == "Tensor":
        buffer = data["buffer"]
        dtype_str = data["dtype"].replace("torch.", "")
        # NumPy 1.24 removed the deprecated ``np.bool`` alias while PyTorch
        # continues to spell this wire dtype ``torch.bool``.
        np_dtype = np.bool_ if dtype_str == "bool" else getattr(np, dtype_str)
        assert isinstance(buffer, bytes)
        np_tensor = np.frombuffer(buffer, dtype=np_dtype)
        shape = data.get("shape")
        if shape is not None:
            np_tensor = np_tensor.reshape(shape)
        return torch.from_numpy(np_tensor.copy())

    cls = cls_map[type_name]
    kwargs = {}
    for k, v in data.items():
        if k == "__type__":
            continue
        kwargs[k] = _deserialize_any(cls_map, v)
    return cls(**kwargs)
