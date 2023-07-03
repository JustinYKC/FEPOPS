from . import FepopsDBSqlite, FepopsDBJSON
from typing import Union
from pathlib import Path


def get_persistent_fepops_storage_object(
    database_file: Union[str, Path], parallel: bool = True, n_jobs: int = -1
):
    if str(database_file).endswith(
        (".sqlite", ".sqlite3", ".db", ".db3", ".s3db", ".sl3")
    ):
        return FepopsDBSqlite(database_file, parallel=parallel, n_jobs=n_jobs)
    if str(database_file).endswith((".json")):
        return FepopsDBJSON(database_file, parallel=parallel, n_jobs=n_jobs)
