import bz2
import sqlite3
from pathlib import Path
from typing import Union
from rdkit import Chem
import numpy as np
from fepops.fepops import GetFepopStatusCode
from .fepops_persistent_abc import FepopsPersistentAbstractBaseClass


class FepopsDBSqlite(FepopsPersistentAbstractBaseClass):
    def __init__(
        self,
        database_file: Union[str, Path],
        kmeans_method: str = "pytorch-cpu",
        parallel: bool = True,
        n_jobs: int = -1,
    ):
        super().__init__(
            database_file=database_file,
            kmeans_method=kmeans_method,
            parallel=parallel,
            n_jobs=n_jobs,
        )
        if not self.database_file.exists():
            print(f"Database {self.database_file} not found, a new one will be created")
        self._register_sqlite_adaptors()
        self.con = sqlite3.connect(
            self.database_file, detect_types=sqlite3.PARSE_DECLTYPES
        )
        self.cur = self.con.cursor()
        res = self.cur.execute("SELECT name FROM sqlite_master")
        if res.fetchone() is None:
            print(f"Creating new table in {self.database_file}")
            self.cur.execute(
                "CREATE TABLE fepops_lookup_table(cansmi text primary key, fepops array)"
            )

    def _register_sqlite_adaptors(self) -> None:
        def adapt_array(nparray):
            """
            Adapted from
            http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
            """
            return sqlite3.Binary(bz2.compress(nparray.tobytes()))

        def convert_array(text):
            return np.frombuffer(bz2.decompress(text))

        sqlite3.register_adapter(np.ndarray, adapt_array)
        sqlite3.register_converter("array", convert_array)

    def add_fepop(self, rdkit_canonical_smiles: str, fepops: Union[np.ndarray, None]):
        if fepops is None:
            fepops = np.array([np.NaN])
        super().add_fepop(rdkit_canonical_smiles=rdkit_canonical_smiles, fepops=fepops)
        if not self.fepop_exists(rdkit_canonical_smiles=rdkit_canonical_smiles):
            self.cur.execute(
                "insert into fepops_lookup_table (cansmi, fepops) values (?,?)",
                (rdkit_canonical_smiles, fepops),
            )
            self.con.commit()

    def fepop_exists(self, rdkit_canonical_smiles: str) -> bool:
        """Check if Fepop exists in the database

        If the fepops object was constructed with a database file, then
        query if the supplied canonical SMILES is included.  If no database
        is present, then False is returned, as if it is not included.

        Parameters
        ----------
        rdkit_canonical_smiles : str
                Canonical smiles to check

        Returns
        -------
        bool
                True if supplied canonical smiles exists in the database
        """

        if self.database_file is None:
            return False
        res = self.cur.execute(
            f"""SELECT EXISTS(SELECT 1 FROM fepops_lookup_table WHERE cansmi="{rdkit_canonical_smiles}" LIMIT 1);"""
        )
        found = res.fetchone()
        if found[0] != 1:
            return False
        return True

    def get_fepops(self, smiles, is_canonical=False) -> Union[np.ndarray, None]:
        super().get_fepops(smiles=smiles)

        if isinstance(smiles, str):
            rdkit_canonical_smiles, mol = self._get_can_smi_mol_tuple(
                smiles, is_canonical=is_canonical
            )
        elif isinstance(smiles, Chem.rdchem.Mol):
            mol = smiles
            rdkit_canonical_smiles = Chem.MolToSmiles(mol)
        else:
            # At this point is is guaranteed to be np.ndarray (type checking
            # performed by super), so just return the array (smiles) and
            # success.
            return GetFepopStatusCode.SUCCESS, smiles

        if self.fepop_exists(rdkit_canonical_smiles):
            res = self.cur.execute(
                f"""SELECT fepops FROM fepops_lookup_table where cansmi="{rdkit_canonical_smiles}" """
            )
            fepop = res.fetchone()[0]
            if np.isnan(fepop).any():
                return GetFepopStatusCode.FAILED_RETRIEVED_NONE, None
            else:
                return GetFepopStatusCode.SUCCESS, fepop.reshape(
                    -1,
                    (
                        self.fepops_object.num_centroids_per_fepop
                        * self.fepops_object.num_features_per_fepop
                    )
                    + self.fepops_object.num_distances_per_fepop,
                )
        else:
            status, fepops_descriptors = self.fepops_object.get_fepops(mol)
            if status == GetFepopStatusCode.SUCCESS:
                self.add_fepop(
                    rdkit_canonical_smiles=rdkit_canonical_smiles,
                    fepops=fepops_descriptors,
                )
                return fepops_descriptors
            else:
                return status, None
