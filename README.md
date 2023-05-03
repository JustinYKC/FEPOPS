# FEPOPS(feature point pharmacophores)
Python implementation of the FEPOPS molecular similarity method and descriptor generator. The FEPOPS descriptors is a 3D method of molecular representation by four centroids with five pharmacophoric features (i.e. atomic logP, atomic partial charges, hydrogen bond acceptors, hydrogen bond donors, and 6 distances between the four centroids). This implementation was recreated following the original paper: [https://pubs.acs.org/doi/10.1021/jm049654z](https://pubs.acs.org/doi/10.1021/jm049654z)

### Steps for implementation:
1. Preprocessing: Molecules were filtered by the number of atoms (< 4), the number of rings (> 9), and the number of rotatable bonds (< 40). A salt/ions filter was then applied. Filtering carried out using `mol_filter.py`.
2. Tautomer enumeration: Tautomers for each molecule were enumerated and saved as mol objects (codes in `fepops.py`).
3. Sample conformers through rotation of flexible bonds: Each tautomer underwent conformer generation, changing the angles of all rotatable bonds. Up to 1024 conformers are sampled from the pool of conformers if more than 5 rotatable bonds are found in a molecule (codes in `fepops.py`).
4. Calculate the 4 centroids: 4 centroids were calculated using K-means for each sampled conformer. With these 4 centroids, each conformer was represented as a 4-point molecular representation with each atom clustered into one of these 4 centroid groups (codes in `fepops.py`).
5. Calculate and assign pharmacophoric features: Each of the four centroids was assigned five pharmacophoric features from their atom cluster members (codes in `fepops.py`).
6. Select most representitive conformers: The FEPOPS conformers were further clustered by k-medoids to find a small number of representative conformers for each molecule (codes in `fepops.py`).
7. Calculate FEPOPS similarity: The FEPOPS similarity between two molecules is measured by Pearson correlation between two FEPOPS descriptors after transformation of each to sum to zero and have a variance of 1 (codes in `fepops.py`). 

# Requirements
This FEPOPS implementation requires the following packages:
- rdkit (>=2019.09.x.x)
- numpy (>=1.19.x)
- pandas (>=1.5.0)
- scikit-learn (>=0.20.x)
- scipy (>=1.7.x)
- PyTorch (>=1.0.0)
- fast-pytorch-kmeans (>=0.1.9)
- tqdm (>=4.48.0)

# Usage
A quickstart example to generate the FEPOPS descriptors for a molecule directly from its SMILES as follows: In terminal:
```
python fepops.py get_fepops -ismi "O=C1OC2=CC3(C)C(CC4OC(=O)C(OC(=O)C)C5C6(OCC45C3C(O)C6O)C(=O)OC)C(C2=C1)C" 
```

A quickstart example to calculate the FEPOPS similarity between two molecules using their SMILES as follows: In terminal:
```
python fepops.py calc_sim -ismi1 "O=C1OC2=CC3(C)C(CC4OC(=O)C(OC(=O)C)C5C6(OCC45C3C(O)C6O)C(=O)OC)C(C2=C1)C" -ismi2 "OC=1C=C(O)C=C(C1)C=2OC=3C=CC=CC3C2"
```

An example of filtering molecules in the dataset of natural products: `COCONUT.DB.smi` [COCONUT.DB.smi](https://coconut.naturalproducts.net/download), for further use of the FEPOPS generation:
```
python mol_filter.py
```

This implementation is also importable and callable within custom scripts for the FEPOPS generation of a batch of molecules. For example:
```
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
```