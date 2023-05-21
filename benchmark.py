import time
import fire
import numpy as np
from sklearn.datasets import make_classification
from tqdm import tqdm
from fepops import Fepops
from dataclasses import dataclass
from typing import Callable, Union
from pathlib import Path
import pandas as pd
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from sklearn.metrics import roc_auc_score
from fepops.fepops_persistent import get_persistent_fepops_storage_object


@dataclass
class SimilarityMethod:
    name: str
    score: Callable


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
    fire.Fire(FepopsBenchmarker)
