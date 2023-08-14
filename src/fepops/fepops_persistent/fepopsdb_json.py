import bz2
import json
from base64 import b64decode, b64encode
from pathlib import Path
from typing import Union

import numpy as np

from .fepops_persistent_abc import FepopsPersistentAbstractBaseClass


class FepopsDBJSON(FepopsPersistentAbstractBaseClass):
    """FepopsDBJSON - allows reading and writing to a simple JSON style cache"""

    def __init__(
        self,
        database_file: Union[str, Path],
        kmeans_method: str = "sklearn",
        parallel: bool = True,
        n_jobs: int = -1,
    ):
        """FepopsDBJSON constructor

        Allows reading and writing to a JSON file in place of a database/cache

        Parameters
        ----------
        database_file : Union[str, Path]
            Filename as a Path or string denoting where the file is, or should
            be created
        kmeans_method : str, optional
            KMeans method which should be used by the OpenFEPOPS object, by
            default "sklearn"
        parallel : bool, optional
            Run in parallel (using joblib), by default True
        n_jobs : int, optional
            Number of jobs to be spawned with joblib. If -1, then use
            all available cores. By default -1
        """
        super().__init__(
            database_file=database_file,
            kmeans_method=kmeans_method,
            parallel=parallel,
            n_jobs=n_jobs,
        )

        if not self.database_file.parent.exists():
            self.database_file.parent.mkdir(parents=True)
        if self.database_file.exists():
            self.db = json.load(open(self.database_file, "r"))
        else:
            self.db = {}
        self._db_changed = False
        self._was_written = False

    def add_fepop(self, rdkit_canonical_smiles: str, fepops: Union[np.ndarray, None]):
        """Add a FEPOP to the database using the supplied SMILES as a key

        Parameters
        ----------
        rdkit_canonical_smiles : str
            Canonical SMILES string generated by RDKit which represents the
            molecule used to generate the FEPOPS
        fepops : Union[np.ndarray, None]
            Array containing calculated FEPOPS descriptors. If None, then None
            is stored in the database, which is useful for indicating that the
            canonical SMILES supplied did not succeed in generating a molecule
            and subsequent FEPOPS. Marking these difficult SMILES in the
            database means they can be checked and ignored without further time
            being spent to regenerate them again.
        """
        if fepops is None:
            fepops = np.array([np.NaN])
        super().add_fepop(rdkit_canonical_smiles=rdkit_canonical_smiles, fepops=fepops)
        if not self.fepop_exists(rdkit_canonical_smiles=rdkit_canonical_smiles):
            self.db[rdkit_canonical_smiles] = b64encode(
                bz2.compress(fepops.tobytes())
            ).decode("ascii")
            self._db_changed = True

    def get_fepops(
        self, smiles: str, is_canonical: bool = False
    ) -> Union[np.ndarray, None]:
        """Get FEPOPS from the database for a given SMILES

        Parameters
        ----------
        smiles : str
            The SMILES string of the molecule
        is_canonical : bool, optional
            If True, then we guarantee that the SMILES string supplied is
            canonical and generated by RDKit and in which case, we may skip a
            cleaning step, by default False

        Returns
        -------
        Union[np.ndarray, None]
            Returns an array representing the retrieved FEPOPS, or None if None
            was stored in the database under the supplied SMILES key
        """
        super().get_fepops(smiles=smiles)
        if not is_canonical:
            smiles, mol = self._get_can_smi_mol_tuple(smiles)
        if self.fepop_exists(rdkit_canonical_smiles=smiles):
            res = np.frombuffer(bz2.decompress(b64decode(self.db[smiles].encode())))
            if np.isnan(res).any():
                return None
            else:
                return res.reshape(
                    -1,
                    self.openfepops_object.num_centroids_per_fepop
                    * self.openfepops_object.num_features_per_fepop,
                )
        else:
            new_fepops = self.openfepops_object.get_fepops(mol)
            self.add_fepop(rdkit_canonical_smiles=smiles, fepops=new_fepops)
            return new_fepops

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
            True if the canonical smiles exists in the database
        """
        return rdkit_canonical_smiles in self.db

    def write(self):
        """Write any changes to the database/cache to the original file"""
        if self._db_changed:
            json.dump(self.db, open(self.database_file, "w"))
            self._was_written = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.write()

    def __del__(self):
        if self._db_changed and not self._was_written:
            print(
                "New fepops were added but changes were not written. Either call .write() or use FepopsDBJSON in a context, like with FepopsDBJSON() as fepops_jsondb..."
            )
