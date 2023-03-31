"""Initial framework of FEPOPS test suite"""

from fepops.fepops import Fepops


def test_k_medoids_simple(fepops_1024_unclustered):
    """Test k_medoid clustering.  Needs expansion"""
    f = Fepops()
    assert f._get_k_medoids(fepops_1024_unclustered).shape[0] == 7


def test_paracetamol_vs_all(test_mol_smiles):
    """Test similarity of paracetamol to itself, diclofenac, ibuprofen, nicotine and vanilin"""
    f = Fepops()
    fepops_dict = {
        mol_name: f.get_fepops(mol_smiles)
        for mol_name, mol_smiles in test_mol_smiles.items()
    }
    scores = {
        f"Paracetamol_{mol_name}_similarity": f.calc_similarity(
            fepops_dict["Paracetamol"], fepop
        )
        for mol_name, fepop in fepops_dict.items()
    }
    sorted_scores = {
        k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)
    }
    it = iter(sorted_scores.keys())
    assert "Paracetamol" in next(it).split("_")[1]
