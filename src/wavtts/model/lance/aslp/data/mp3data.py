import os

import pyarrow
from dataclasses import dataclass, field

from .base import Base


@dataclass
class Mp3Data(Base):
    data_id: str = field(default=None)
    name: str = field(default=None, compare=False)
    mp3_binary: bytes = field(default=None, compare=False)

    def __post_init__(self):
        if isinstance(self.mp3_binary, (bytes, type(None))):
            return
        if isinstance(self.mp3_binary, str) and os.path.isfile(self.mp3_binary):
            with open(self.mp3_binary, "rb") as f:
                self.mp3_binary = f.read()
            return
        raise NotImplementedError(
            f"Unsupported type {type(self.mp3_binary)} for mp3_binary; expected bytes or file path"
        )

    @staticmethod
    def get_example():
        return Mp3Data(
            data_id="example_mp3_data",
            mp3_binary=b"example_mp3_binary_data",
        )

    @staticmethod
    def get_schema():
        return pyarrow.schema([
            pyarrow.field("data_id", pyarrow.string(), nullable=False),
            pyarrow.field("mp3_binary", pyarrow.binary(), nullable=True),
        ])
