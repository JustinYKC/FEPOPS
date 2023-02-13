import pytest
from sklearn.datasets import make_classification


@pytest.fixture
def fepops_1024_unclustered():
    """Classification dataset, 7 classes, 7 clusters, 100 samples, 10 features"""
    n_useful_features = 22
    data, _ = make_classification(
        n_samples=1024,
        n_features=n_useful_features,
        n_informative=n_useful_features,
        n_redundant=0,
        n_repeated=0,
        n_classes=7,
        n_clusters_per_class=1,
        weights=None,
        flip_y=0,
        class_sep=1.0,
        hypercube=True,
        shift=0.0,
        scale=1.0,
        shuffle=True,
        random_state=42,
    )
    return data


@pytest.fixture
def test_mol_smiles():
    """Dictionary of test molecule smiles"""
    return {
        "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
        "Ibuprofen": "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",
        "Diclofenac": "O=C(O)Cc1ccccc1Nc2c(Cl)cccc2Cl",
        "Nicotine": "CN1CCC[C@H]1c2cccnc2",
        "Vanilin": "O=Cc1ccc(O)c(OC)c1",
    }
