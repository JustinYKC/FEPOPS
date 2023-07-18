from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Union

import numpy as np
from rdkit import Chem
from scipy.spatial.distance import cdist, pdist, squareform
from tqdm import tqdm
from fepops.fepops import GetFepopStatusCode
from ..fepops import Fepops
import multiprocessing as mp


class FepopsPersistentAbstractBaseClass(metaclass=ABCMeta):
    """Abstract base class for persistent fepops storage

    New storage methods may be implemented as demonstrated in fepopsdb_json.py
    by extending this abstract base class which provides some required
    functionality like:

    - save_descriptors(smiles: Union[str, Path, list[str]]), to save a smiles file/list of smiles to the persistent storage
    - get_cansmi_to_mol_dict_not_in_database(smiles: Union[str, Path, list[str]]), to retrieve a unique dictionary with canonical smiles as keys not already stored in the database and rdkit mol objects as values.

    When writing your own persistent storage methods, you must override the following methods:

    add_fepop(rdkit_canonical_smiles: str, fepops: np.ndarray)
    ----------------------------------------------------------
    Add the fepop to persistent storage. super().add_fepop may be called by the overridden function to perform type checks on arguments.

    fepop_exists(rdkit_canonical_smiles: str)
    -----------------------------------------
    Return True if the canonical smiles is already in the database, and False if not. super().fepop_exists may be called by the overridden
    function to perform type checks on arguments.

    get_fepops(rdkit_canonical_smiles: str)
    --------------------------------------
    Return a fepop from persistent storage. If it does not exist, then generate it by calling self.fepops_object.get_fepops which is supplied
    by this base class. super().get_fepops may be called by the overridden function to perform type checks on arguments.
    With this function in place it allows interface compatibility with a standard Fepops object.

    Inheriting functions may also define __enter__ and __exit__ methods for use with context handlers. If none are defined, then empty ones
    are provided. This can be useful in doing things like writing out large files after descriptor generation if incremental writes are not
    possible, like in the case of the FepopsDBJSON child class.


    Parameters
    ----------
    database_file : Union[str, Path]
            File to use for persistent storage.
    kmeans_method : str, optional
            Method which should be used for kmeans calculation by
            fepops objects, can be one of "sklearn", "pytorchgpu",
            or "pytorchcpu".
    parallel : bool, optional
            Run in parallel (using joblib), by default True
    n_jobs : int, optional
            Number of jobs to be spawned with joblib. If -1, then use
            all available cores. By default -1.
    """

    @staticmethod
    def _parallel_init_worker_get_cansmi_mol_tuple(smiles_is_rdkit_canonical):
        global sirdkc
        sirdkc = smiles_is_rdkit_canonical

    @staticmethod
    def _parallel_get_cansmi_tuple(m):
        global sirdkc
        return FepopsPersistentAbstractBaseClass._get_can_smi_mol_tuple(
            m, smiles_guaranteed_rdkit_canonical=sirdkc
        )

    @staticmethod
    def _parallel_init_worker_desc_gen_shared_fepops_ob(fepops_object):
        global shared_fepops_ob
        shared_fepops_ob = fepops_object

    @staticmethod
    def _parallel_get_gen_fepops_descriptors(m):
        global shared_fepops_ob
        return m[0], shared_fepops_ob.get_fepops(m[1])

    @abstractmethod
    def __init__(
        self,
        database_file: Union[str, Path],
        kmeans_method: str = "sklearn",
        parallel: bool = True,
        n_jobs: int = -1,
    ):
        self.database_file = Path(database_file)
        self.fepops_object = Fepops(kmeans_method=kmeans_method)
        self.parallel = parallel
        self.n_jobs = n_jobs

    def save_descriptors(
        self,
        smiles: Union[str, Path, list[str]],
        add_failures_to_database: bool = True,
        smiles_guaranteed_rdkit_canonical: bool = False,
        fepops_object_constructor_kwargs: dict = {},
    ):
        """Pregenerate FEPOPS descriptors for a set of SMILES strings"""

        canonical_smiles_to_mol_dict = self.get_cansmi_to_mol_dict_not_in_database(
            smiles, smiles_guaranteed_rdkit_canonical=smiles_guaranteed_rdkit_canonical
        )
        if len(canonical_smiles_to_mol_dict) == 0:
            print("Nothing to add to database")
            return
        if not self.parallel:
            for rdkit_canonical_smiles, mol in tqdm(
                canonical_smiles_to_mol_dict.items(), desc="Generating fepops"
            ):
                status, fepops_array = self.fepops_object.get_fepops(mol)
                if status == GetFepopStatusCode.SUCCESS or add_failures_to_database:
                    self.add_fepop(rdkit_canonical_smiles, fepops_array)
            print(
                f"Added {len(canonical_smiles_to_mol_dict)} new molecues to the database ({self.database_file})"
            )
        else:  # Do it in parallel
            n_successes = 0
            n_failures = 0
            for rdkit_canonical_smiles, (status, new_fepop) in tqdm(
                mp.Pool(
                    # processes=min(len(canonical_smiles_to_mol_dict), mp.cpu_count()),
                    processes=2,
                    initializer=self._parallel_init_worker_desc_gen_shared_fepops_ob,
                    initargs=(Fepops(**fepops_object_constructor_kwargs),),
                ).imap(
                    self._parallel_get_gen_fepops_descriptors,
                    canonical_smiles_to_mol_dict.items(),
                ),
                desc="Generating descriptors (parallel)",
                total=len(canonical_smiles_to_mol_dict),
            ):
                if status == GetFepopStatusCode.SUCCESS:
                    n_successes += 1
                else:
                    n_failures += 1
                if status == GetFepopStatusCode.SUCCESS or add_failures_to_database:
                    self.add_fepop(rdkit_canonical_smiles, new_fepop)
            print(
                f"Successfully added {n_successes} new molecues to the database ({self.database_file}), {n_failures} failed"
            )

    @abstractmethod
    def add_fepop(self, rdkit_canonical_smiles: str, fepops: np.ndarray):
        """Add canonical smiles and fepop to database.

        Must be overridden
        """
        if not isinstance(rdkit_canonical_smiles, str):
            raise ValueError(
                f"Expected an rdkit canonical smiles string, but a {type(rdkit_canonical_smiles)} was passed"
            )
        if not isinstance(fepops, np.ndarray):
            raise ValueError(f"Expected a fepop, but a {type(fepops)} was passed")

    @abstractmethod
    def get_fepops(
        self, smiles: Union[str, Chem.rdchem.Mol, np.ndarray], is_canonical: bool = True
    ) -> None:
        if not isinstance(smiles, (str, Chem.rdchem.Mol, np.ndarray)):
            raise ValueError(
                f"Expected an rdkit canonical smiles string, rdkit mol, or a numpy array of descriptors but a {type(smiles)} was passed: {smiles}"
            )

    @abstractmethod
    def fepop_exists(self, rdkit_canonical_smiles: str) -> bool:
        """Return True if canonical smiles already exist in the database"""
        if not isinstance(rdkit_canonical_smiles, str):
            raise ValueError(
                f"Expected an rdkit canonical smiles string, but a {type(rdkit_canonical_smiles)} was passed"
            )

    @staticmethod
    def _get_can_smi_mol_tuple(s: str, smiles_guaranteed_rdkit_canonical: bool = False):
        try:
            mol = Chem.MolFromSmiles(s)
        except:
            try:
                mol = Chem.MolFromSmiles(s, sanitize=False)
            finally:
                mol = None
        if mol is None:
            print(f"Could not parse smiles to a valid molecule, smiles was: {s}")
            return (s, mol)
        if smiles_guaranteed_rdkit_canonical:
            return (s, mol)
        else:
            return (Chem.MolToSmiles(mol), mol)

    def get_cansmi_to_mol_dict_not_in_database(
        self,
        smiles: Union[str, Path, list[str]],
        smiles_guaranteed_rdkit_canonical: bool = False,
    ):
        if isinstance(smiles, str):
            smiles = Path(smiles)
        if isinstance(smiles, Path):
            if smiles.exists():
                smiles = [
                    s.strip() for s in open(smiles).readlines() if len(s.strip()) > 0
                ]
            else:
                raise ValueError(
                    f"smiles file ({smiles}) not found. If you are passing smiles, place it into a list first"
                )
        if not isinstance(smiles, list):
            raise ValueError(
                "smiles should be a str or Path denoting the location of a smiles file, or a list of smiles"
            )
        smiles = list(set(smiles))
        print(f"Got {len(smiles)} unique SMILES strings")

        if not self.parallel:
            # Ensure unique (canonical, also storing intermediate mol)
            canonical_smiles_to_mol_dict = dict(
                self._get_can_smi_mol_tuple(
                    s,
                    smiles_guaranteed_rdkit_canonical=smiles_guaranteed_rdkit_canonical,
                )
                for s in tqdm(smiles, desc="Uniquifying input smiles (non-parallel)")
            )
        else:
            tmp_res_list = []
            # Ensure unique (canonical, also storing intermediate mol)
            for res in tqdm(
                mp.Pool(
                    initializer=self._parallel_init_worker_get_cansmi_mol_tuple,
                    initargs=(smiles_guaranteed_rdkit_canonical,),
                    processes=max(1, min(len(smiles) / 50, mp.cpu_count())),
                ).map(self._parallel_get_cansmi_tuple, smiles, chunksize=1),
                desc="Uniquifying input smiles (parallel)",
                total=len(smiles),
            ):
                tmp_res_list.append(res)

            canonical_smiles_to_mol_dict = dict(tmp_res_list)

            del tmp_res_list
        # Make sure none are already in the database
        canonical_smiles_to_mol_dict = {
            cansmi: mol
            for cansmi, mol in tqdm(
                canonical_smiles_to_mol_dict.items(),
                desc="Checking if mols already exist in the database",
            )
            if not self.fepop_exists(cansmi)
        }
        print(
            f"Got {len(canonical_smiles_to_mol_dict)} unique molecules not already in the database"
        )

        return canonical_smiles_to_mol_dict

    def calc_similarity(
        self,
        fepops_features_1: Union[np.ndarray, str, None],
        fepops_features_2: Union[np.ndarray, str, None],
        is_canonical=True,
    ):
        """Calculate FEPOPS similarity

        A static method for calculating molecular similarity based on their FEPOPS descriptors.

        Parameters
        ----------
        fepops_features_1 : Union[np.ndarray, str, None]
            A Numpy array containing the FEPOPS descriptors of the query molecule
            or a smiles string from which to generate FEPOPS descriptors for the
            query molecule.
        fepops_features_2 : Union[np.ndarray, str, None, list[np.ndarray, str, None]]
            A Numpy array containing the FEPOPS descriptors of the candidate
            molecule or a smiles string from which to generate FEPOPS descriptors
            for the candidate molecule.  Can also be None, in which case, np.nan is
            returned as a score, or a list of any of these. If it is a list,
            then a list of scores against the single candidate is returned.

        Returns
        -------
        float
                Fepops similarity between two molecules
        """
        if fepops_features_1 is None:
            return np.nan
        if isinstance(fepops_features_1, (str, Chem.rdchem.Mol)):
            status, fepops_features_1 = self.get_fepops(
                fepops_features_1, is_canonical=is_canonical
            )
            if status != GetFepopStatusCode.SUCCESS:
                return np.nan

        if isinstance(fepops_features_2, list):
            new_fepops_features_2 = []
            for item in fepops_features_2:
                status, fpop = self.get_fepops(item, is_canonical=is_canonical)
                new_fepops_features_2.append(
                    fpop if status == GetFepopStatusCode.SUCCESS else None
                )
            return self.fepops_object.calc_similarity(
                fepops_features_1, new_fepops_features_2
            )

        if isinstance(fepops_features_2, (str, Chem.rdchem.Mol)):
            status, fepops_features_2 = self.get_fepops(
                fepops_features_2, is_canonical=is_canonical
            )
            if status != GetFepopStatusCode.SUCCESS:
                return np.nan
        if any(x is None for x in (fepops_features_1, fepops_features_2)):
            return np.nan
        score = self.fepops_object.calc_similarity(fepops_features_1, fepops_features_2)
        return score if score is not None else np.nan

    def write(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.write()