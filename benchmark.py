import time
import fire
import numpy as np
from sklearn.datasets import make_classification
from tqdm import tqdm
from fepops import Fepops


class FepopsBenchmarker:
    def __init__(self):
        """FEPOPS benchmarker

        Contains test data useful in assessing the peformance of the FEPOPS object
        """
        self.fepops = Fepops()

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
            self.fepops._get_k_medoids(f_1024)
            timings[i] = time.time() - start
        print(
            f"Median time to compute medoid from 1024 fepops={np.median(timings)}, mean={np.mean(timings)}"
        )


if __name__ == "__main__":
    fire.Fire(FepopsBenchmarker)
