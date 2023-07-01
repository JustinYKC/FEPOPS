import time
import fire
import os
import numpy as np
from sklearn.datasets import make_classification
from tqdm import tqdm
from fepops import Fepops
from dataclasses import dataclass
from typing import Callable, Union
from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit.Chem import PandasTools
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.metrics import roc_auc_score
from fepops.fepops_persistent import get_persistent_fepops_storage_object
from typing import Union, Optional
import multiprocessing as mp


@dataclass
class SimilarityMethod:
    name: str
    score: Callable


class Filter:
    def __init__(self, min_atoms: int = 4) -> None:
        self.min_atoms = min_atoms
        self.lfc = rdMolStandardize.LargestFragmentChooser()

    def _remove_salt(self, mol: Chem.rdchem.Mol) -> Union[Chem.rdchem.Mol, None]:
        """Romve salts or ions from molecules

        Remove salts or ions from molecules

        Parameters
        ----------
        mol : Chem.rdchem.Mol
        The Rdkit mol object of the input molecule.

        Returns
        -------
        Chem.rdchem.Mol or None
        Return a Rdkit mol object if the input molecule meets the criterion, otherwise return None.
        """
        try:
            cleaned_molecule = rdMolStandardize.Cleanup(mol)
            smiles = Chem.MolToSmiles(cleaned_molecule)
            if "." in smiles:
                cleaned_molecule = rdMolStandardize.Cleanup(
                    self.lfc.choose(cleaned_molecule)
                )
        except:
            return None
        return cleaned_molecule

    def _filter_by_atom_num(self, mol: Chem.rdchem.Mol) -> Union[Chem.rdchem.Mol, None]:
        """Filter molecules by number of atoms

        Filter out molecules by the number of atoms.

        Parameters
        ----------
        mol : Chem.rdchem.Mol
        The Rdkit mol object of the input molecule.
        cutoff : int
        The criterion of atom numbers for filtering. By default 4.

        Returns
        -------
        Chem.rdchem.Mol or None
        Return a Rdkit mol object if the input molecule meets the criterion, otherwise return None.
        """
        atom_num = mol.GetNumAtoms()
        if atom_num > self.min_atoms:
            return mol
        else:
            return None

    def __call__(self, mol: Chem.rdchem.Mol) -> Union[Chem.rdchem.Mol, None]:
        """Apply all filters

        Parameters
        ----------
        mol : Chem.rdchem.Mol
        The Rdkit mol object of the input molecule.

        Returns
        -------
        filter_result
        Return a Rdkit mol object if the input molecule meets all criteria, otherwise return None.
        """
        filter_result = self.filter_mol(mol)
        return filter_result

    def filter_mol(self, mol: Chem.rdchem.Mol) -> Union[Chem.rdchem.Mol, None]:
        """Apply all filters

        Perform two filters (minimum number of atoms, and salt/ion) in sequence
        to obtain the final molecules as required.

        Parameters
        ----------
        mol : Chem.rdchem.Mol
        The Rdkit mol object of the input molecule.

        Returns
        -------
        Chem.rdchem.Mol or None
        Return a Rdkit mol object if the input molecule meets all criteria, otherwise return None.
        """
        mol = self._remove_salt(mol)
        if mol is None:
            return None
        else:
            mol = self._filter_by_atom_num(mol)
            if mol is None:
                return None
            else:
                return mol


class DudePreprocessor:
    def __init__(
        self,
        dude_directory: Union[Path, str] = "data/dude/",
    ) -> None:
        self.dude_path = Path(dude_directory)
        self.dude_unprocessed_path = self.dude_path / Path("unprocessed")
        if not self.dude_path.exists():
            raise FileNotFoundError(f"Dude dataset not found in path: {self.dude_path}")
        self.fepops_ob = Fepops()
        print(self.dude_path)
        print(self.dude_unprocessed_path)

    def __call__(
        self,
    ):
        self.process()

    def process(
        self,
    ):
        dude_targets = [
            t.parent.name for t in self.dude_path.glob("unprocessed/*/actives_final.ism")
        ]
        print(dude_targets)
        for target in tqdm(dude_targets, desc=f"Preparing targets"):
            self.create_dude_target_csv_data(target)

    @staticmethod
    def _parallel_init_worker_desc_gen_shared_fepops_ob():
        global shared_fepops_ob
        shared_fepops_ob = Fepops()

    @staticmethod
    def _parallel_get_rdkit_cansmi(s):
        global shared_fepops_ob
        mol = shared_fepops_ob._mol_from_smiles(s)
        if mol is None:
            return ""
        return Chem.MolToSmiles(mol)

    def create_dude_target_csv_data(
        self,
        dude_target: Path,
        actives_file: Path = Path("actives_final.ism"),
        decoys_file: Path = Path("decoys_final.ism"),
        seperator: str = " ",
    ):
        if not self.dude_unprocessed_path.exists():
            raise FileNotFoundError(
                f"no directory under {self.dude_path} called 'unprocessed'. This unprocessed directory should contain all dude files downloaded and extracted under subdirectories with the names of targets"
            )
        processed_path = self.dude_path / Path("processed")
        processed_path.mkdir(parents=True, exist_ok=True)
        actives = pd.read_csv(
            self.dude_unprocessed_path / Path(dude_target) / actives_file,
            sep=seperator,
            header=None,
            names=["SMILES", "DUDEID", "CHEMBLID"],
        )
        actives["Active"] = 1
        decoys = pd.read_csv(
            self.dude_unprocessed_path / Path(dude_target) / decoys_file,
            sep=seperator,
            header=None,
            names=["SMILES", "DUDEID"],
        )
        decoys["Active"] = 0
        df = pd.concat([actives, decoys]).reset_index().drop(columns="index")
        df["rdkit_canonical_smiles"] = tqdm(
            mp.Pool(
                initializer=self._parallel_init_worker_desc_gen_shared_fepops_ob
            ).imap(self._parallel_get_rdkit_cansmi, df.SMILES, chunksize=100),
            desc=f"Generating {dude_target} benchmark file",
            total=len(df),
        )
        df.to_csv(processed_path / f"dude_target_{dude_target}.csv", index=False)


class ROCScorer:
    def __init__(self, methods: list[SimilarityMethod]) -> None:
        self.similarity_methods = methods

    def get_auroc_scores(
        self,
        df: pd.DataFrame,
        smiles_column_title="SMILES",
        active_flag_column_title="Active",
    ) -> dict[str, float]:
        if active_flag_column_title not in df.columns:
            raise ValueError(
                f"Could not find a '{active_flag_column_title}' column in the DataFrame to indicate if the compound is active, or a decoy. Consider passing a column title to the 'active_flag_column_title' argument to get_auroc_scores"
            )
        if smiles_column_title not in df.columns:
            raise ValueError(
                f"Could not find a '{smiles_column_title}' column in the DataFrame. Consider passing a column title to the 'active_flag_column_title' argument to get_auroc_scores"
            )
        labels = np.array(df[active_flag_column_title].tolist(), dtype=int)
        scores_dict = {sm.name: [] for sm in self.similarity_methods}
        for active in df.query(f"{active_flag_column_title}==1")[
            smiles_column_title
        ].tolist():
            scores = np.array(
                [
                    [
                        sim_method.score(active, s)
                        for sim_method in self.similarity_methods
                    ]
                    for s in df[smiles_column_title].tolist()
                ]
            )
            roc_scores = [roc_auc_score(labels, s) for s in scores.T]
            [
                scores_dict[m.name].append(rs)
                for m, rs in zip(self.similarity_methods, roc_scores)
            ]
        return scores_dict


class FepopsBenchmarker:
    def __init__(self, database_file: str = "benchmark.db"):
        """FEPOPS benchmarker

        Contains test data useful in assessing the peformance of the FEPOPS object
        """
        self.fepops = get_persistent_fepops_storage_object(database_file=database_file)

    def get_1k_x_1024_fepops(self):
        n_useful_features = 22
        return np.array(
            [
                make_classification(
                    n_samples=1024,
                    n_features=n_useful_features,
                    n_informative=n_useful_features,
                    n_redundant=0,
                    n_repeated=0,
                    n_classes=21,
                    n_clusters_per_class=1,
                    weights=None,
                    flip_y=0,
                    class_sep=1.0,
                    hypercube=True,
                    shift=0.0,
                    scale=1.0,
                    shuffle=True,
                )[0]
                for _ in tqdm(range(1000), "Generating test data")
            ]
        )

    def kmedoids(self):
        """Benchmark kmedoid code using the standard classification dataset

        All benchmarks run using github codepace running Python 3.10.9 on a
        2cpu instance (Intel(R) Xeon(R) Platinum 8272CL CPU @ 2.60GHz)

        Github commit 0e9f282, using the first kmeans implementation gives
        a median time of 0.04102 seconds to extract 7 centroid fepops from
        1024 fepops.

        """

        cached_1k_1024_fepops = self.get_1k_x_1024_fepops()
        timings = np.empty(cached_1k_1024_fepops.shape[0])
        for i, f_1024 in enumerate(tqdm(cached_1k_1024_fepops, "Benchmarking")):
            start = time.time()
            self._get_k_medoids(f_1024)
            timings[i] = time.time() - start
        print(
            f"Median time to compute medoids with sklearn from 1024 fepops={np.median(timings)}, mean={np.mean(timings)}"
        )

    def kmeans(self):
        """Benchmark kmedoid code using the standard classification dataset

        All benchmarks run using github codepace running Python 3.10.9 on a
        2cpu instance (Intel(R) Xeon(R) Platinum 8272CL CPU @ 2.60GHz)

        Github commit 0e9f282, using the first kmeans implementation gives
        a median time of 0.04102 seconds to extract 7 centroid fepops from
        1024 fepops.

        """
        fepops_sklearn = Fepops(kmeans_method="sklearn")

        cached_1k_1024_fepops = self.get_1k_x_1024_fepops()
        timings = np.empty(cached_1k_1024_fepops.shape[0])
        for i, f_1024 in enumerate(tqdm(cached_1k_1024_fepops, "Benchmarking")):
            start = time.time()
            fepops_sklearn._perform_kmeans(f_1024, 7, kmeans_method="sklearn")
            timings[i] = time.time() - start
        print(
            f"Median time to compute kmeans with sklearn from 1024 fepops={np.median(timings)}, mean={np.mean(timings)}"
        )

        fepops_ptcpu = Fepops(kmeans_method="pytorch-cpu")

        timings = np.empty(cached_1k_1024_fepops.shape[0])
        for i, f_1024 in enumerate(tqdm(cached_1k_1024_fepops, "Benchmarking")):
            start = time.time()
            fepops_ptcpu._perform_kmeans(f_1024, 7, kmeans_method="pytorch-cpu")
            timings[i] = time.time() - start
        print(
            f"Median time to compute kmeans with sklearn from 1024 fepops={np.median(timings)}, mean={np.mean(timings)}"
        )

    @staticmethod
    def _score_morgan(m1: str, m2: str, smi_to_mol_func) -> float:
        m1 = smi_to_mol_func(m1)
        m2 = smi_to_mol_func(m2)
        fp1 = AllChem.GetMorganFingerprint(m1, 2)
        fp2 = AllChem.GetMorganFingerprint(m2, 2)
        return DataStructs.DiceSimilarity(fp1, fp2)

    def auroc_performance(
        self,
        data_tsv: Union[Path, str],
        smiles_column_title: str = "Std_SMILES",
        active_flag_column_title: str = "Activity",
    ):
        roc_scorer = ROCScorer(
            [
                SimilarityMethod(
                    "Morgan",
                    lambda x, y: self._score_morgan(
                        x, y, self.fepops.fepops_object._mol_from_smiles
                    ),
                ),
                SimilarityMethod("FEPOPS", self.fepops.calc_similarity),
            ]
        )
        scores_dict = roc_scorer.get_auroc_scores(
            pd.read_csv(
                Path(data_tsv),
                sep="\t",
                index_col=[0],
                header=0,
            ).reset_index(),
            smiles_column_title="Std_SMILES",
            active_flag_column_title="Activity",
        )
        print(pd.DataFrame.from_dict(scores_dict).describe())


if __name__ == "__main__":
    # fire.Fire(FepopsBenchmarker)
    fire.Fire(DudePreprocessor)
