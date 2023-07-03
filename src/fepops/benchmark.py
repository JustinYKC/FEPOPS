import time
import numpy as np
from sklearn.datasets import make_classification
from tqdm import tqdm
from fepops import Fepops
from dataclasses import dataclass
from typing import Callable, Union, Optional
from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.metrics import roc_auc_score
from fepops.fepops_persistent import get_persistent_fepops_storage_object
from typing import Union, Optional


@dataclass
class SimilarityMethod:
    name: str
    supports_multiple_candidates: bool
    score: Callable
    descriptor_calc_func: Optional[Callable] = None
    descriptor_score_func: Optional[Callable] = None


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

        smiles_list = df[smiles_column_title].tolist()
        labels_list = df[active_flag_column_title].tolist()
        descriptors = []
        for sm in self.similarity_methods:
            descriptors.append([])
            for smiles in tqdm(smiles_list, desc=f"Caching descriptors for {sm.name}"):
                res = sm.descriptor_calc_func(smiles)
                if isinstance(res, tuple):
                    descriptors[-1].append(res[-1])
                else:
                    descriptors[-1].append(res)

        problematic_molecule_ilocs = []
        assert (
            len(set([len(d) for d in descriptors])) == 1
        ), "Generation of descriptors yielded different lengths, something went very wrong"
        for j in range(len(descriptors[0])):
            vals = [descriptors[i][j] is not None for i in range(len(descriptors))]
            if not all(vals):
                problematic_molecule_ilocs.append(j)
        problematic_molecule_smiles = df.iloc[problematic_molecule_ilocs][
            smiles_column_title
        ].tolist()
        if len(problematic_molecule_smiles) > 0:
            print('─' * 80)
            print("Warning")
            print(
                "Problematic molecules for which no descriptors could be retrieved were found. "
            )
            print("The following molecules will not be included in AUROC calculations:")
            for pm in problematic_molecule_smiles:
                print(pm)
            print('─' * 80)
            problematic_molecule_ilocs.sort(reverse=True)
            for pmi in problematic_molecule_ilocs:
                del labels_list[pmi]
                for d_i in range(len(descriptors)):
                    del descriptors[d_i][pmi]

        labels = np.array(labels_list, dtype=int)
        scores_dict = {sm.name: [] for sm in self.similarity_methods}
        for sm_i, sm in enumerate(self.similarity_methods):
            for active_i in tqdm(
                np.argwhere(np.array(labels_list) == 1).flatten(),
                desc=f"Assessing active recall (AUROC) for {sm.name}",
            ):
                if sm.supports_multiple_candidates:
                    scores = np.array(
                        sm.descriptor_score_func(
                            descriptors[sm_i][active_i], descriptors[sm_i]
                        ),
                        dtype=float,
                    )
                else:
                    scores = np.array(
                        [
                            sm.descriptor_score_func(
                                descriptors[sm_i][active_i], descriptors[sm_i][smiles_i]
                            )
                            for smiles_i in range(len(descriptors[0]))
                        ],
                        dtype=float,
                    )
                scores_dict[sm.name].append(
                    roc_auc_score(
                        labels[np.argwhere(~np.isnan(scores))],
                        scores[np.argwhere(~np.isnan(scores))],
                    )
                )
        return scores_dict

    def get_auroc_scores_fast(
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
        descriptors = {
            sm.name: {m: sm.descriptor_calc_func(m) for m in df[smiles_column_title]}
            for sm in self.similarity_methods
        }
        scores_dict = {sm.name: [] for sm in self.similarity_methods}
        for active in tqdm(
            df.query(f"{active_flag_column_title}==1")[smiles_column_title].tolist(),
            desc="Assessing active recall (AUROC)",
        ):
            scores = np.array(
                [
                    [
                        sim_method.score(
                            descriptors[sim_method.name][active],
                            descriptors[sim_method.name][s],
                        )
                        for sim_method in self.similarity_methods
                    ]
                    for s in df[smiles_column_title].tolist()
                ]
            )
            roc_scores = [
                roc_auc_score(
                    labels[np.argwhere(~np.isnan(s))], s[np.argwhere(~np.isnan(s))]
                )
                for s in scores
            ]
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
    def _score_morgan(m1: str, m2: Union[str, None], smi_to_mol_func) -> float:
        m1 = smi_to_mol_func(m1)
        if m2 is None:
            return AllChem.GetMorganFingerprint(m1, 2)
        m2 = smi_to_mol_func(m2)
        fp1 = AllChem.GetMorganFingerprint(m1, 2)
        fp2 = AllChem.GetMorganFingerprint(m2, 2)
        return DataStructs.DiceSimilarity(fp1, fp2)

    def auroc_performance(
        self,
        data_tsv: Union[Path, str],
        data_tsv_contains_canonical_smiles: bool = True,
        smiles_column_title: str = "SMILES",
        active_flag_column_title: str = "Active",
        results_output_dir: Optional[Union[str, Path]] = None,
    ):
        roc_scorer = ROCScorer(
            [
                SimilarityMethod(
                    "Morgan",
                    False,
                    lambda x, y: self._score_morgan(
                        x, y, self.fepops.fepops_object._mol_from_smiles
                    ),
                    lambda x: self._score_morgan(
                        x, None, self.fepops.fepops_object._mol_from_smiles
                    ),
                    lambda x, y: DataStructs.DiceSimilarity(x, y),
                ),
                SimilarityMethod(
                    "FEPOPS",
                    True,
                    self.fepops.calc_similarity,
                    lambda x: self.fepops.get_fepops(
                        x, is_canonical=data_tsv_contains_canonical_smiles
                    ),
                    self.fepops.calc_similarity,
                ),
            ]
        )
        scores_dict = roc_scorer.get_auroc_scores(
            pd.read_csv(
                Path(data_tsv),
                sep="\t",
                index_col=[0],
                header=0,
            ).reset_index(),
            smiles_column_title=smiles_column_title,
            active_flag_column_title=active_flag_column_title,
        )
        auroc_results_df = pd.DataFrame.from_dict(scores_dict)
        if results_output_dir is None:
            results_output_dir = Path(data_tsv).parent
        results_output_dir.mkdir(parents=True, exist_ok=True)
        print(data_tsv + "\n" + auroc_results_df.describe())
        auroc_results_df.to_csv(
            results_output_dir
            / f"results_benchmarking_{Path(data_tsv).stem.replace('benchmarking_','').replace('_molecules','')}.csv"
        )

    def auroc_performance_on_dir(
        self,
        working_dir: Union[Path, str] = Path("."),
        data_tsv_contains_canonical_smiles: bool = True,
        smiles_column_title: str = "SMILES",
        active_flag_column_title: str = "Active",
        results_output_dir: Optional[Union[Path, str]] = None,
    ):
        working_dir = Path(working_dir)
        csv_files = [Path(f) for f in working_dir.glob("benchmarking_*_molecules.csv")]
        db_and_tsv = []
        for csvf in csv_files:
            target = str(csvf.stem).split("_")[1]
            db_file_path = (
                working_dir / f"benchmarking_dud_e_diverse_mols_by_target_{target}.db"
            )
            if db_file_path.exists():
                db_and_tsv.append((db_file_path, csvf))
        print(db_and_tsv)
        for db_file_path, csv_file_path in tqdm(db_and_tsv):
            self.auroc_performance(
                csv_file_path,
                data_tsv_contains_canonical_smiles=data_tsv_contains_canonical_smiles,
                smiles_column_title=smiles_column_title,
                active_flag_column_title=active_flag_column_title,
                results_output_dir=results_output_dir,
            )
