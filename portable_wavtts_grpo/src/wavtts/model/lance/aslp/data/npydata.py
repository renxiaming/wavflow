import numpy
import pyarrow
from dataclasses import dataclass, field
from typing import List

from .base import Base


@dataclass
class FloatData(Base):
    data: numpy.ndarray = field(default=None, compare=False)
    shape: List[int] = field(default=None, compare=False)

    def __post_init__(self):
        if isinstance(self.shape, numpy.ndarray):
            self.shape = self.shape.tolist()
        if self.data is None:
            return
        assert isinstance(self.data, numpy.ndarray), "data should be numpy array"
        assert self.data.dtype in (numpy.float16, numpy.float32), f"expected float32, got {self.data.dtype}"
        if self.shape is not None:
            assert len(self.data.shape) == 1, (
                f"When a shape is specified, only one-dimensional data can be provided. Got {self.data.shape}"
            )
            self.data = self.data.reshape(*self.shape)
        else:
            self.shape = self.data.shape.tolist() if isinstance(self.data.shape, numpy.ndarray) else self.data.shape

    @staticmethod
    def get_example():
        return FloatData(
            data_id="example_data",
            data=numpy.random.rand(2, 1000).astype(numpy.float32),
        )

    @staticmethod
    def get_schema():
        return pyarrow.schema([
            pyarrow.field("data_id", pyarrow.string(), nullable=False),
            pyarrow.field("data", pyarrow.list_(pyarrow.float32())),
            pyarrow.field("shape", pyarrow.list_(pyarrow.int32())),
        ])

    def to_dict(self, only_not_none: bool = False, str_wrap: bool = False):
        data_id = f"'{self.data_id}'" if str_wrap else self.data_id
        return {
            "data_id": data_id,
            "data": self.data.reshape(-1),
            "shape": self.shape,
        }


@dataclass
class IntData(Base):
    data: numpy.ndarray = field(default=None, compare=False)
    shape: List[int] = field(default=None, compare=False)

    def __post_init__(self):
        if self.data is None and self.shape is None:
            return
        assert isinstance(self.data, numpy.ndarray), "data should be numpy array"
        assert self.data.dtype in (numpy.int16, numpy.int32), f"expected int32, got {self.data.dtype}"
        if self.shape is not None:
            assert len(self.data.shape) == 1, (
                f"When a shape is specified, only one-dimensional data can be provided. Got {self.data.shape}"
            )
            self.data = self.data.reshape(*self.shape)
        else:
            self.shape = self.data.shape

    @staticmethod
    def get_example():
        return IntData(
            data_id="example_data",
            data=numpy.random.randint(0, 200, size=(2, 1000)).astype(numpy.int32),
        )

    @staticmethod
    def get_schema():
        return pyarrow.schema([
            pyarrow.field("data_id", pyarrow.string(), nullable=False),
            pyarrow.field("data", pyarrow.list_(pyarrow.int32())),
            pyarrow.field("shape", pyarrow.list_(pyarrow.int32())),
        ])

    def to_dict(self, only_not_none: bool = False, str_wrap: bool = False):
        data_id = f"'{self.data_id}'" if str_wrap else self.data_id
        return {
            "data_id": data_id,
            "data": self.data.reshape(-1),
            "shape": self.shape,
        }
