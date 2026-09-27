from enum import Enum, auto
from math import inf
import logging

MAX_MOS = 5
MIN_MOS = -5
MIN_DURATION = 0
MAX_DURATION = inf


class AutoNameValue(Enum):
    @staticmethod
    def _generate_next_value_(name, start, count, last_values) -> str:
        return name.lower()

    @classmethod
    def check(cls: Enum, obj: object) -> Enum:
        if isinstance(obj, cls):
            return obj
        try:
            item = cls(obj)
        except ValueError:
            if obj is None or obj == "None":
                item = cls(None)
            else:
                logging.warning(
                    f"{cls.__name__} has not attribute with {obj}! Create NotExistEnum({obj})"
                )
                item = NotExistEnum.create_not_exist_enum(obj, obj)
        return item


class NotExistEnum(AutoNameValue):
    @staticmethod
    def create_not_exist_enum(name: str, value: object):
        return AutoNameValue("NotExistEnum", {name: value})

    @staticmethod
    def is_same_type(cls) -> bool:
        return type(cls) == type(NotExistEnum)
