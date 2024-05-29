from pathlib import Path
from typing import Union

from . import FepopsDBJSON, FepopsDBSqlite


def get_persistent_fepops_storage_object(
    database_file: Union[str, Path],
    kmeans_method: str = "sklearn",
    parallel: bool = True,
    n_jobs: int = -1,
) -> Union[FepopsDBSqlite, FepopsDBJSON]:
    if str(database_file).endswith(
        (".sqlite", ".sqlite3", ".db", ".db3", ".s3db", ".sl3")
    ):
        return FepopsDBSqlite(
            database_file, kmeans_method=kmeans_method, parallel=parallel, n_jobs=n_jobs
        )
    if str(database_file).endswith((".json")):
        return FepopsDBJSON(
            database_file, kmeans_method=kmeans_method, parallel=parallel, n_jobs=n_jobs
        )
