# OpenFEPOPS(FEature POint PharmacophoreS)

OpenFEPOPS is an open-source Python implementation of the FEPOPS molecular similarity technique enabling descriptor generation, comparison, and ranking of molecules in virtual screening campaigns. The central idea behind FEPOPS is reducing the complexity of molecules by merging of local atomic environments and atom properties into ‘feature points’. This compressed feature point representation has been used to great effect as noted in literature, helping researchers identify active and potentially therapeutically valuable small molecules. This implementation was recreated following the original paper: [https://pubs.acs.org/doi/10.1021/jm049654z](https://pubs.acs.org/doi/10.1021/jm049654z). By default, OpenFEPOPS uses literature reported parameters which show good performance in retrieval of active lead- and drug-like small molecules within virtual screening campaigns, with feature points capturing charge, lipophilicity, and hydrogen bond acceptor and donor status. When run with default parameters, OpenFepops compactly represents molecules using sets of four feature points, with each feature point encoded into 22 numeric values, resulting in a compact representation of 616 bytes per molecule. By extension, this allows the indexing of a compound archive containing 1 million small molecules using only 587.5 MB of data. Whilst more compact representations are readily available, the FEPOPS technique strives to capture tautomer and conformer information, first through enumeration and then through diversity driven selection of representative FEPOPS descriptors to capture the diverse states that a molecule may adopt.

At the time of writing, `OpenFEPOPS` is the only publicly available implementation of the FEPOPS molecular similarity technique. Whilst used within industry and referenced extensively in literature, it has been unavailable to researchers as an open-source tool. This truly open implementation allows researchers to use and contribute to the advancement of FEPOPS within the rapid development and collaborative framework provided by open-source software. It is therefore hoped that this will allow the technique to be used not only for traditional small molecule molecular similarity, but also in new emerging fields such as protein design and featurization of small- and macro-molecules for both predictive and generative tasks.

## Method description
Whilst OpenFEPOPS has included functionality for descriptor caching and profiling of libraries, the core functionality of the package is descriptor generation and scoring.
### _Descriptor generation:_

1. Tautomer enumeration
    - For a given small molecule, OpenFEPOPS uses RDKit to iterate over molecular tautomers. By default, there is no limit to the number of recoverable tautomers, but a limit may be imposed which may be necessary if adapting the OpenFEPOPS code to large macromolecules and not just small molecules.
2.  Conformer enumeration
    - For each tautomer, up to 1024 conformers are sampled by either complete enumeration of rotatable bond states (at the literature reported optimum increment of 90 degrees) if there are five or less rotatable bonds, or through random sampling of 1024 possible states if there are more than 5 rotatable bonds.
3.  Defining feature points
    - The KMeans algorithm is applied to each conformer of each tautomer to identify four (by default) representative or centrol points, into which the atomic information of neighbouring atoms is collapsed. As standard, the atomic properties of charge, logP, hydrogen bond donor, and hydrogen bond acceptor status are collapsed into four feature points per unique tautomer conformation. These feature points are encoded to 22 numeric values (a FEPOP) comprising four points, each with four properties, and six pairwise distances between these points. With many FEPOPS descriptors collected from a single molecule through tautomer and conformer enumeration, this set of representative FEPOPS should capture every possible state of the original molecule.
4.  Selection of diverse FEPOPS
    - From the collection of FEPOPS derived from every tautomer conformation of a molecule, the K-Medoid algorithm is applied to identify seven (by default) diverse FEPOPS which are thought to best capture a fuzzy representation of the molecule. These seven FEPOPS each comprise 22 descriptors each, totalling 154 32-bit floating point numbers or 616 bytes.

![OpenFEPOPS descriptor generation showing the capture of tautomer and conformer information from a single input molecule.\label{fig:descriptor_generation}](Figure1.png)

Descriptor generation with OpenFEPOPS is a compute intensive task and as noted in literature, designed to be run in situations where large compound archives have had their descriptors pre-generated and are queried against realatively small numbers of new molecules for which descriptors are not known and are generated. To enable use in this manner, OpenFEPOPS provides functionality to cache descriptors through specification of database files, either in the SQLite or JSON formats.


### Scoring and comparison of molecules based on their molecular descriptors

1.  Sorting
    - With seven (by default) diverse FEPOPS representing a small molecule, the FEPOPS are sorted by ascending charge.
2.  Scaling
    - Due to the different scales and distributions of features comprising FEPOPS descriptors, each FEPOP is centered and scaled according to observed mean and standard deviations of the same features within a larger pool of molecules. By default, these means and standard deviations have been derived from the DUDE diversity set which captures known actives and decoys for a diverse set of therapeutic targets.
3.  Scoring
    - The Pearson correlation coefficient is calculated for the scaled descriptors of the first molecule to the scaled descriptors of the second.

Literature highlights that the choice of the Pearson correlation coefficient leads to high background scores as it is highly unlikely to see little correlation between any molecule due to fundamental limitations of chemistry and geometry. Therefore, unrelated molecules are likely to have FEPOPS similarity scores higher than those encountered with more traditional techniques such as bitstring fingeprints and Tanimoto or Dice similarity measures.

## Usage

OpenFEPOPS has been uploaded to the Python Packaging Index under the name 'fepops' and as such is installable using the pip package manager and the command 'pip install fepops'. With the package installed, entrypoints are used to expose commonly used OpenFEPOPS tasks such as descriptor generation and calculation on molecular similarity, enabling simple command line access without the need to explicitly invoke a Python interpreter. Whilst OpenFEPOPS may be used solely via the command line interface, a robust API is available and may be used within other programs or integrated into existing pipelines to enable more complex workflows.  API documentation is available at https://justinykc.github.io/FEPOPS.

### Command line usage:
With OpenFEPOPS installed to a Python environment, entrypoints in the code and a command line interface control object allow omission of the python interpreter program when calling common OpenFEPOPS tasks. This allows command line usage of the form:

```console
fepops <subcommand> <arguments>
```

These sub-commands to carry out common tasks are:
1. calc_sim
    - Calculates the molecular similarity of two supplied molecules as SMILES strings.
2. get_fepops
    - Calculate and print out the molecular descriptors of a molecule supplied as a SMILES string
3. save_descriptors
    - To be used in conjunction with the --database_file switch (see below) which enables writing of generated descriptors to a SQLite or JSON database/file cache. A list of smiles, or the location of a SMILES file may be passed here. If passing a single molecule, then enclose the smiles in square brackets so that it is passed as a list containing one item.

A database or cache file may be supplied for use with all subcommands using the --database_file switch before the subcommand as follows:
```console
fepops --database_file=<DB_file_path> <subcommand> <arguments>
```
Depending on the supplied file extension of <DB_file_path>, either a SQLite file (".sqlite", ".sqlite3", ".db", ".db3", ".s3db", or ".sl3" file extensions) is created or loaded for the appending of data, or a JSON file is used (when the extension is ".json").

Running OpenFEPOPS with the -h switch prints help to the terminal and it may also be used after sub-command switches to inspect required arguments.

#### Example: Calculating molecular similarity between two molecules (ibuprofen and diclofenac)
```console
fepops calc_sim "CC(Cc1ccc(cc1)C(C(=O)O)C)C" "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl"
```
Note the use of quotes around the smiles strings. This is required as BASH and other shells will try to parse the brackets often present in smiles to denote branching.

#### Example: Get the FEPOPS descriptors for diclofenac
```console
fepops get_fepops "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl"
```

#### Example: pregenerating descriptors for an in-house compound archive
With a SMILES file called 'inhouse_compounds.smi', we may pre-generate their FEPOPS descriptors for faster use and comparison against new molecules.
```console
fepops --database_file=inhouse_compounds.db save_descriptors inhouse_compounds.smi
```

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

