"""Initial framework of FEPOPS test suite"""

from pathlib import Path
from fepops import Fepops
from fepops.fepops import GetFepopStatusCode
from fepops.fepops_persistent import get_persistent_fepops_storage_object


def test_k_medoids_simple(fepops_1024_unclustered):
    """Test k_medoid clustering.  Needs expansion"""
    f = Fepops()
    assert f._get_k_medoids(fepops_1024_unclustered).shape[0] == 7

def test_all_test_mols_rank_themselves_top(test_mol_smiles):
    """Test similarity of paracetamol to itself, diclofenac, ibuprofen, nicotine and vanilin"""
    f = Fepops()
    fepops_dict = {
        mol_name: f.get_fepops(mol_smiles)
        for mol_name, mol_smiles in test_mol_smiles.items()
    }
    assert all([t[0]==GetFepopStatusCode.SUCCESS for t in fepops_dict.values()]), "Could not generate valid fepops for all molecules within the test set"
    scores={}
    for query in fepops_dict.keys():
        scores[query] = {
            f"{query}_{candidate_name}_similarity": f.calc_similarity(
                fepops_dict[query][1], candidate_fepops
            )
            for candidate_name, (_, candidate_fepops) in fepops_dict.items()
        }
    sorted_results={
        query:{
        k: v for k, v in sorted(scores[query].items(), key=lambda item: item[1], reverse=True)
        } for query in fepops_dict}
    for query, results in sorted_results.items():
        assert query in next(iter(results.keys())).split("_")[1]


def test_fepops_persistent_sqlite(test_mol_smiles, tmp_path):
    """Test creating and saving fepops descriptors to a sqlite3 database"""

    d = tmp_path/"test_dir"
    d.mkdir()
    file = d/"test.db"
    f = get_persistent_fepops_storage_object(database_file=str(file.name))
    fepops_dict = {
        mol_name: f.get_fepops(mol_smiles)[1]
        for mol_name, mol_smiles in test_mol_smiles.items()
    }
    fepops_dict2 = {
        mol_name: f.get_fepops(mol_smiles)[1]
        for mol_name, mol_smiles in test_mol_smiles.items()
    }
    file.unlink(missing_ok=True)
    assert fepops_dict.keys() == fepops_dict2.keys()
   