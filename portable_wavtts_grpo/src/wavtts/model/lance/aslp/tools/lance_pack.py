import datetime
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import List

import lance
import pandas
import pyarrow
from tqdm import tqdm

from ..data.base import Base


def to_pyarrow_table_with_schema(data_list: List[Base], schema: pyarrow.Schema):
    dataframe = pandas.DataFrame.from_dict([item.to_dict() for item in data_list])
    return pyarrow.Table.from_pandas(dataframe, schema=schema)


def load(uri: str, auto_create: bool = False, target_cls: Base = Base) -> lance.LanceDataset:
    if auto_create and not os.path.exists(uri):
        ds = create_new_dataset(uri, target_cls.get_example)
    else:
        ds = lance.dataset(uri)
    return ds


def create_new_dataset(uri: str, example_func) -> lance.LanceDataset:
    example_data: Base = example_func()
    table = to_pyarrow_table_with_schema([example_data], example_data.get_schema())
    ds = lance.write_dataset(table, uri)
    ds.create_scalar_index("data_id", "BTREE")
    ds.delete("(`data_id` IS NOT NULL) OR (`data_id` IS NULL)")
    return ds


def ceildiv(a, b):
    return -(a // -b)


class LanceReader:
    def __init__(self, uri: str, filter_str: str = None, target_cls: Base = Base) -> None:
        self.ds = load(uri)
        self.filter = filter_str
        self.target_cls = target_cls

    def get_ids(self, ids_col: str = "data_id", filter_str: str = None, raw: bool = False, progress: bool = False):
        items = self.ds.to_table([ids_col], filter=filter_str, with_row_id=True).to_pandas()
        if raw:
            return items
        return self.target_cls.from_pandas(items, progress)

    def get_datas_by_rows(self, rows: List[int], cols: List[str] = None, raw: bool = False):
        table = self.ds.take(rows, columns=cols, with_row_id=True)
        items = table.to_pandas()
        if raw:
            return items
        return self.target_cls.from_pandas(items)

    def get_datas_by_rowids(self, row_ids: List[int], cols: List[str] = None, raw: bool = False):
        table = self.ds._take_rows(row_ids=row_ids, columns=cols, with_row_id=True)
        items = table.to_pandas()
        if raw:
            return items
        return self.target_cls.from_pandas(items)

    def get_datas_by_filter(self, cols: List[str] = None, filter_str: str = None, raw: bool = False):
        table = self.ds.to_table(cols, filter_str, with_row_id=True)
        items = table.to_pandas()
        if raw:
            return items
        return self.target_cls.from_pandas(items)


class LanceWriter(LanceReader):
    def __init__(self, uri: str, filter_str: str = None, target_cls: Base = Base) -> None:
        self.ds = load(uri, auto_create=True, target_cls=target_cls)
        self.uri = uri
        self.filter = filter_str
        self.target_cls = target_cls

    def insert(self, data_list: List[Base], index: str = "data_id"):
        schema = self.target_cls.get_schema()
        table = to_pyarrow_table_with_schema(data_list, schema)
        return (
            self.ds.merge_insert(index)
            .when_not_matched_insert_all()
            .execute(table, schema=schema)
        )

    def update(self, data_list: List[Base], index: str = "data_id"):
        for item in data_list:
            if isinstance(item, Base):
                item_dict = item.to_dict(only_not_none=True, str_wrap=True)
            elif isinstance(item, dict):
                item_dict = {k: v for k, v in item.items() if v is not None}
            else:
                raise TypeError(f"Expected item of type {Base} or dict, but got {type(item)}")
            index_value = item_dict.pop(index, None)
            if index_value is None:
                continue
            self.ds.update(item_dict, where=f"{index}={index_value}")

    def delete(self, ids: List[str], index: str = "data_id"):
        list_str = ", ".join([f"'{i}'" for i in ids])
        self.ds.delete(f"`{index}` IN ({list_str})")

    def write_parallel(self, data_list: List[Base], n_jobs: int = 5, max_items_per_frag: int = 10_000, progress: bool = False, **kwargs):
        part_size = ceildiv(len(data_list), n_jobs)
        part_size = min(part_size, max_items_per_frag)
        data_parts = []
        for i in range(n_jobs):
            start = i * part_size
            end = min((i + 1) * part_size, len(data_list))
            data_parts.append(data_list[start:end])

        fragments = []

        def task(data, **task_kwargs):
            new_frags = lance.fragment.write_fragments(
                to_pyarrow_table_with_schema(data, self.target_cls.get_schema()),
                self.uri,
                **task_kwargs,
            )
            fragments.extend(new_frags)

        threads = []
        executor = ThreadPoolExecutor(n_jobs)
        for part in data_parts:
            if len(part) >= 1:
                threads.append(executor.submit(partial(task, part, **kwargs)))

        if progress:
            threads = tqdm(threads)

        for thread in threads:
            thread.result()

        operation = lance.LanceOperation.Append(fragments)
        self.ds = lance.LanceDataset.commit(self.uri, operation, read_version=self.ds.latest_version)

    def clean_up(self):
        last_version = self.ds.latest_version
        self.ds.checkout_version(last_version)
        delta = datetime.datetime.now() - self.ds.versions()[-1]["timestamp"]
        return self.ds.cleanup_old_versions(delta)
