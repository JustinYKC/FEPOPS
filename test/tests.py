"""Initial framework of FEPOPS test suite"""

from pathlib import Path
from fepops import Fepops
from fepops.fepops_persistent import get_persistent_fepops_storage_object
import tempfile
import numpy as np


def test_k_medoids_simple(fepops_1024_unclustered):
    """Test k_medoid clustering.  Needs expansion"""
    f = Fepops()
    assert f._get_k_medoids(fepops_1024_unclustered).shape[0] == 7


def test_diclofenac_vs_all(test_mol_smiles):
    """Test similarity of paracetamol to itself, diclofenac, ibuprofen, nicotine and vanilin"""
    f = Fepops()
    fepops_dict = {
        mol_name: f.get_fepops(mol_smiles)[1]
        for mol_name, mol_smiles in test_mol_smiles.items()
    }
    scores = {
        f"Diclofenac_{mol_name}_similarity": f.calc_similarity(
            fepops_dict["Diclofenac"], fepop
        )
        for mol_name, fepop in fepops_dict.items()
    }
    print(scores)
    sorted_scores = {
        k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)
    }
    it = iter(sorted_scores.keys())
    assert "Diclofenac" in next(it).split("_")[1]


def test_nicotine_vs_all(test_mol_smiles):
    """Test similarity of nicotine to itself, diclofenac, ibuprofen, paracetamol and vanilin"""
    f = Fepops()
    fepops_dict = {
        mol_name: f.get_fepops(mol_smiles)[1]
        for mol_name, mol_smiles in test_mol_smiles.items()
    }
    scores = {
        f"Nicotine_{mol_name}_similarity": f.calc_similarity(
            fepops_dict["Nicotine"], fepop
        )
        for mol_name, fepop in fepops_dict.items()
    }
    sorted_scores = {
        k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)
    }
    it = iter(sorted_scores.keys())
    assert "Nicotine" in next(it).split("_")[1]


def test_fepops_persistent_sqlite(test_mol_smiles):
    """Test creating and saving fepops descriptors to a sqlite3 database"""

    file = Path(tempfile.gettempdir()) / Path(
        next(tempfile._get_candidate_names()) + ".db"
    )
    print(file.name)
    try:
        f = get_persistent_fepops_storage_object(database_file=str(file.name))
        fepops_dict = {
            mol_name: f.get_fepops(mol_smiles)[1]
            for mol_name, mol_smiles in test_mol_smiles.items()
        }
        fepops_dict2 = {
            mol_name: f.get_fepops(mol_smiles)[1]
            for mol_name, mol_smiles in test_mol_smiles.items()
        }
        assert fepops_dict.keys() == fepops_dict2.keys()

    finally:
        file.unlink(missing_ok=True)
