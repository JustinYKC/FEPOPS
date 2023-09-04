import sys
from rdkit import Chem
from fepops import Fepops
from io import StringIO

smiles_input_file_path = "./COCONUT_after_filter.smi"
sio = sys.stderr = StringIO()
f = Fepops()
with open(smiles_input_file_path) as in_file:
    for line in in_file:
        parts = line.strip().split("\t")
        id = parts[-1].strip()
        smiles = parts[0].strip()
        print (f"{id}, {smiles}:")
        mol = Chem.MolFromSmiles(smiles)
        err = sio.getvalue()
        if err:
            sio = sys.stderr = StringIO()
            if "Explicit valence" in err:
                print (f"# {id} Bad_valence\n")
                continue
            elif "SMILES Parse Error" in err:
                print (f"# {id} SMILES_parse_error\n")
                continue

        fepops_features = f.get_fepops(smiles)
        print (f"{fepops_features}\n")