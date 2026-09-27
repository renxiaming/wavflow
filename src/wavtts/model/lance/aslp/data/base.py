import logging
from dataclasses import dataclass, field
from inspect import signature

import pandas

from .limitations import AutoNameValue


@dataclass
class Base:
    data_id: str = field(default=None)
    _rowid: int = field(default=None, compare=False)

    @classmethod
    def from_kwargs(cls, **kwargs):
        cls_fields = {field for field in signature(cls).parameters}
        native_args, new_args = {}, {}
        for name, val in kwargs.items():
            if name in cls_fields:
                native_args[name] = val
            else:
                new_args[name] = val

        ret = cls(**native_args)

        for new_name, new_val in new_args.items():
            logging.warn(f"Ignore {new_name}:{new_val}")
        return ret

    @classmethod
    def from_pandas(cls, items: pandas.DataFrame, progress: bool = False):
        from tqdm import tqdm

        cls_fields = {field for field in signature(cls).parameters}
        keys = items.keys()
        ignore_keys = [i for i in keys if i not in cls_fields]
        exists_keys = [i for i in keys if i in cls_fields]
        if len(ignore_keys) > 1:
            logging.warn(f"Ignore {ignore_keys}")
        data_list = []
        bar = range(len(items))
        if progress:
            bar = tqdm(range(len(items)))
        for idx in bar:
            native_args = {}
            for key in exists_keys:
                native_args[key] = items[key][idx]
            data_list.append(cls(**native_args))
        return data_list

    @staticmethod
    def get_example():
        raise NotImplementedError

    @staticmethod
    def get_schema():
        raise NotImplementedError

    def to_dict(self, only_not_none: bool = False, str_wrap: bool = False):
        source_dict = self.__dict__
        new_dict = {}
        for key, value in source_dict.items():
            if isinstance(value, AutoNameValue):
                value = value.value
            if str_wrap:
                if isinstance(value, str):
                    value = f"'{value}'"
                elif value is not None:
                    value = str(value)
            if key == "_rowid":
                continue
            if value is not None:
                new_dict[key] = value
            elif not only_not_none:
                new_dict[key] = value
        return new_dict
