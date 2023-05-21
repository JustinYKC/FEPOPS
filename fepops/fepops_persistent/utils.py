from . import FepopsDBSqlite, FepopsDBJSON

def get_persistent_fepops_storage_object(database_file: str):
        if database_file.endswith(
            (".sqlite", ".sqlite3", ".db", ".db3", ".s3db", ".sl3")
        ):
            return FepopsDBSqlite(database_file)
        if database_file.endswith((".json")):
            return FepopsDBJSON(database_file)
