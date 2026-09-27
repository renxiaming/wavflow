# Copyright (c) Facebook, Inc. and its affiliates.
"""
List serialization code adopted from
https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/common.py
"""

from typing import Any, List, Optional

import numpy as np
import pickle
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

try:
    from multiprocessing.reduction import ForkingPickler
except ImportError:  # older PyTorch
    ForkingPickler = mp.reductions.ForkingPickler


def get_rank():
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def all_gather(data, group=None):
    world_size = dist.get_world_size()
    if world_size == 1:
        return [data]

    output = [None for _ in range(world_size)]
    dist.all_gather_object(output, data, group=group)
    return output


class NumpySerializedList:
    def __init__(self, lst: list):
        def _serialize(data):
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)

        print(f"Serializing {len(lst)} elements to byte tensors and concatenating them all ...")
        self._lst = [_serialize(x) for x in lst]
        self._addr = np.asarray([len(x) for x in self._lst], dtype=np.int64)
        self._addr = np.cumsum(self._addr)
        self._lst = np.concatenate(self._lst)
        print(f"Serialized dataset takes {len(self._lst) / 1024**2:.2f} MiB")

    def __len__(self):
        return len(self._addr)

    def __getitem__(self, idx):
        start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
        end_addr = self._addr[idx].item()
        data_bytes = memoryview(self._lst[start_addr:end_addr])
        return pickle.loads(data_bytes)


class TorchSerializedList(NumpySerializedList):
    def __init__(self, lst: list):
        super().__init__(lst)
        self._addr = torch.from_numpy(self._addr)
        self._lst = torch.from_numpy(self._lst)

    def __getitem__(self, idx):
        start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
        end_addr = self._addr[idx].item()
        data_bytes = memoryview(self._lst[start_addr:end_addr].numpy())
        return pickle.loads(data_bytes)


def local_scatter(array: Optional[List[Any]]):
    if get_world_size() == 1:
        return array[0]
    if get_rank() == 0:
        assert array is not None and len(array) == get_world_size()
        all_gather(array)
        return array[0]
    all_data = all_gather(None)
    return all_data[0][get_rank()]


class TorchShmSerializedList(TorchSerializedList):
    def __init__(self, lst: list):
        if get_rank() == 0:
            super().__init__(lst)
            handles = [None] + [
                bytes(ForkingPickler.dumps((self._addr, self._lst)))
                for _ in range(get_world_size() - 1)
            ]
        else:
            handles = None
        handle = local_scatter(handles)

        if get_rank() > 0:
            self._addr, self._lst = ForkingPickler.loads(handle)
            print(f"Worker {get_rank()} obtains a dataset of length={len(self)} from its local leader.")
