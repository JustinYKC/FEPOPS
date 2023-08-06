# OpenFEPOPS(FEature POint PharmacophoreS)

OpenFEPOPS is an open-source Python implementation of the FEPOPS molecular similarity technique enabling descriptor generation, comparison, and ranking of molecules in virtual screening campaigns. The central idea behind FEPOPS is reducing the complexity of molecules by merging of local atomic environments and atom properties into ‘feature points’. This compressed feature point representation has been used to great effect as noted in literature, helping researchers identify active and potentially therapeutically valuable small molecules. This implementation was recreated following the original paper: [https://pubs.acs.org/doi/10.1021/jm049654z](https://pubs.acs.org/doi/10.1021/jm049654z). By default, OpenFEPOPS uses literature reported parameters which show good performance in retrieval of active lead- and drug-like small molecules within virtual screening campaigns, with feature points capturing charge, lipophilicity, and hydrogen bond acceptor and donor status. When run with default parameters, OpenFepops compactly represents molecules using sets of four feature points, with each feature point encoded into 22 numeric values, resulting in a compact representation of 616 bytes per molecule. By extension, this allows the indexing of a compound archive containing 1 million small molecules using only 587.5 MB of data. Whilst more compact representations are readily available, the FEPOPS technique strives to capture tautomer and conformer information, first through enumeration and then through diversity driven selection of representative FEPOPS descriptors to capture the diverse states that a molecule may adopt.

At the time of writing, `OpenFEPOPS` is the only publicly available implementation of the FEPOPS molecular similarity technique. Whilst used within industry and referenced extensively in literature, it has been unavailable to researchers as an open-source tool. This truly open implementation allows researchers to use and contribute to the advancement of FEPOPS within the rapid development and collaborative framework provided by open-source software. It is therefore hoped that this will allow the technique to be used not only for traditional small molecule molecular similarity, but also in new emerging fields such as protein design and featurization of small- and macromolecules for both predictive and generative tasks.


The OpenFEPOPS descriptor generation process as outlined in \autoref{fig:descriptor_generation} follows; for a given small molecule, OpenFEPOPS iterates over tautomers and conformers, picking four (by default) K-mean derived points, into which the atomic information of neighbouring atoms is collapsed. As standard, the atomic properties of charge, logP, hydrogen bond donor, and hydrogen bond acceptor status are collapsed into four feature points per unique tautomer conformation. These feature points are encoded to 22 numeric values (a FEPOP) comprising four points, each with four properties, and six pairwise distances between these points. With four FEPOPS representing every enumerated conformer for every enumerated tautomer of a molecule, this set of representative FEPOPS should capture every possible state of the original molecule. From this list, the K-medoid algorithm is applied to identify seven diverse FEPOPS which are thought to capture a fuzzy representation of the molecule using seven FEPOPS comprising 22 descriptors each, totalling 154 32-bit floating point numbers or 616 bytes.

OpenFEPOPS has been uploaded to the Python Packaging Index under the name 'fepops' and as such is installable using the pip package manager and the command 'pip install fepops'. With the package installed, entrypoints are used to expose commonly used OpenFEPOPS tasks such as descriptor generation and calculation on molecular similarity, enabling simple command line access without the need to explicitly invoke a Python interpreter. Whilst OpenFEPOPS may be used solely via the command line interface, a robust API is available and may be used within other programs or integrated into existing pipelines to enable more complex workflows.  Extensive documentation is available online.


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
fepops get_fepops -ismi "O=C1OC2=CC3(C)C(CC4OC(=O)C(OC(=O)C)C5C6(OCC45C3C(O)C6O)C(=O)OC)C(C2=C1)C" 
```

A quickstart example to calculate the FEPOPS similarity between two molecules using their SMILES as follows: In terminal:
```
python fepops.py calc_sim "O=C1OC2=CC3(C)C(CC4OC(=O)C(OC(=O)C)C5C6(OCC45C3C(O)C6O)C(=O)OC)C(C2=C1)C" "OC=1C=C(O)C=C(C1)C=2OC=3C=CC=CC3C2"
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