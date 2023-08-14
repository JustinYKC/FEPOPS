---
title: 'OpenFEPOPS: A Python implementation of the FEPOPS molecular similarity technique'
tags:
  - Python
  - molecular similarity
  - virtual screening
  - pharmacophores
  - feature points
authors:
  - name: Yan-Kai Chen
    orcid: 0000-0001-7161-9503
    affiliation: 1
  - name: Douglas R. Houston
    orcid: 0000-0002-3469-1546
    affiliation: 1
  - name: Manfred Auer
    orcid: 0000-0001-8920-3522
    affiliation: "1, 2"
  - name: Steven Shave
    orcid: 0000-0001-6996-3663
    corresponding: true
    affiliation: 1
affiliations:
  - name: School of Biological Sciences, University of Edinburgh, The King’s Buildings, Max Born Crescent, CH Waddington Building, Edinburgh, EH9 3BF, United Kingdom.
    index: 1
  - name: Xenobe Research Institute, P. O. Box 3052, San Diego, California, 92163, United States.
    index: 2
date: 6 August 2023
bibliography: paper.bib

---



# Summary

OpenFEPOPS is an open-source Python implementation of the FEPOPS molecular similarity technique [@jenkins20043d; @nettles2007flexible; @jenkins2013feature] enabling descriptor generation, comparison, and ranking of molecules in virtual screening campaigns. Ligand based virtual screening [@ripphausen2011state] is a fundamental approach undertaken to expand hit series or perform scaffold hopping whereby new chemistries and synthetic routes are made available in efforts to remove undesirable molecular properties and discover better starting points in the early stages of drug discovery [@hughes2011principles]. Typically, these techniques query hit molecules against proprietary, in-house, or publicly available repositories of small molecules in the hope of finding close matches which will display similar activities to the query based on the molecular similarity principle (similar molecules should have similar properties and make similar interactions [@cortes2020qsar]). Often batteries of these similarity measures are used in parallel, helping to score molecules from many different subjective viewpoints and measures of similarity [@baber2006use]. The central idea behind FEPOPS is reducing the complexity of molecules by merging of local atomic environments and atom properties into ‘feature points’. This compressed feature point representation has been used to great effect as noted in literature, helping researchers identify active and potentially therapeutically valuable small molecules. By default, OpenFEPOPS uses literature reported parameters which show good performance in retrieval of active lead- and drug-like small molecules within virtual screening campaigns, with feature points capturing charge, lipophilicity, and hydrogen bond acceptor and donor status. When run with default parameters, OpenFepops compactly represents molecules using sets of four feature points, with each feature point encoded into 22 numeric values, resulting in a compact representation of 616 bytes per molecule. By extension, this allows the indexing of a compound archive containing 1 million small molecules using only 587.5 MB of data. Whilst more compact representations are readily available, the FEPOPS technique strives to capture tautomer and conformer information, first through enumeration and then through diversity driven selection of representative FEPOPS descriptors to capture the diverse states that a molecule may adopt.

# Statement of need

At the time of writing, `OpenFEPOPS` is the only publicly available implementation of the FEPOPS molecular similarity technique. Whilst used within industry and referenced extensively in literature, it has been unavailable to researchers as an open-source tool. This truly open implementation allows researchers to use and contribute to the advancement of FEPOPS within the rapid development and collaborative framework provided by open-source software. It is therefore hoped that this will allow the technique to be used not only for traditional small molecule molecular similarity, but also in new emerging fields such as protein design and featurization of small- and macro-molecules for both predictive and generative tasks.

# Brief software description 

Whilst OpenFEPOPS has included functionality for descriptor caching and profiling of libraries, the core functionality of the package is descriptor generation and scoring.

## _Descriptor generation:_

The OpenFEPOPS descriptor generation process as outlined in \autoref{fig:descriptor_generation} follows;

1. Tautomer enumeration
    - For a given small molecule, OpenFEPOPS uses RDKit to iterate over molecular tautomers. By default, there is no limit to the number of recoverable tautomers, but a limit may be imposed which may be necessary if adapting the OpenFEPOPS code to large macromolecules and not just small molecules.
2.  Conformer enumeration
    - For each tautomer, up to 1024 conformers are sampled by either complete enumeration of rotatable bond states (at the literature reported optimum increment of 90 degrees) if there are five or less rotatable bonds, or through random sampling of 1024 possible states if there are more than 5 rotatable bonds.
3.  Defining feature points
    - The KMeans algorithm [@arthur2007k] is applied to each conformer of each tautomer to identify four (by default) representative or centrol points, into which the atomic information of neighbouring atoms is collapsed. As standard, the atomic properties of charge, logP, hydrogen bond donor, and hydrogen bond acceptor status are collapsed into four feature points per unique tautomer conformation. These feature points are encoded to 22 numeric values (a FEPOP) comprising four points, each with four properties, and six pairwise distances between these points. With many FEPOPS descriptors collected from a single molecule through tautomer and conformer enumeration, this set of representative FEPOPS should capture every possible state of the original molecule.
4.  Selection of diverse FEPOPS
    - From the collection of FEPOPS derived from every tautomer conformation of a molecule, the K-Medoid algorithm [@park2009simple] is applied to identify seven (by default) diverse FEPOPS which are thought to best capture a fuzzy representation of the molecule. These seven FEPOPS each comprise 22 descriptors each, totalling 154 32-bit floating point numbers or 616 bytes.

![OpenFEPOPS descriptor generation showing the capture of tautomer and conformer information from a single input molecule.\label{fig:descriptor_generation}](Figure1.png)

Descriptor generation with OpenFEPOPS is a compute intensive task and as noted in literature, designed to be run in situations where large compound archives have had their descriptors pre-generated and are queried against realatively small numbers of new molecules for which descriptors are not known and are generated. To enable use in this manner, OpenFEPOPS provides functionality to cache descriptors through specification of database files, either in the SQLite or JSON formats.

## Scoring and comparison of molecules based on their molecular descriptors

1.  Sorting
    - With seven (by default) diverse FEPOPS representing a small molecule, the FEPOPS are sorted by ascending charge.
2.  Scaling
    - Due to the different scales and distributions of features comprising FEPOPS descriptors, each FEPOP is centered and scaled according to observed mean and standard deviations of the same features within a larger pool of molecules. By default, these means and standard deviations have been derived from the DUDE diversity set which captures known actives and decoys for a diverse set of therapeutic targets.
3.  Scoring
    - The Pearson correlation coefficient is calculated for the scaled descriptors of the first molecule to the scaled descriptors of the second.

Literature highlights that the choice of the Pearson correlation coefficient leads to high background scores as it is highly unlikely to see little correlation between any molecule due to fundamental limitations of chemistry and geometry. Therefore, unrelated molecules are likely to have FEPOPS similarity scores higher than those encountered with more traditional techniques such as bitstring fingeprints and Tanimoto or Dice similarity measures.

The predictive performance of OpenFEPOPS has been evaluated using the DUDE [@mysinger2012directory] diversity set. This dataset comprises eight protein targets accompanied by decoy ligands and known active ligands. Macro-averaged AUROC scores for each target were generated using every known active small molecule to retrieve the entire set of actives for the target. Table 1 shows the average AUROC scores for DUDE diversity set targets along with scores obtained using the popular Morgan 2, MACCS, and RDKit fingerprints as implemented in RDKit and scored using the Tanimoto distance metric. See the Jupyter notebook 'Explore_DUDE_diversity_set.ipynb' in the source repository for further methods and details. 

|  | Morgan 2 | MACCS | RDKit |OpenFEPOPS|
|--------:|----------:|-------:|-------:|------------:|
|akt1  |0.836|0.741 |0.833 |0.829|
|ampc  |0.784|0.673|0.660 |0.639|
|cp3a4 |0.603|0.582|0.613 |0.650|
|cxcr4 |0.697|0.854|0.592 |0.899|
|gcr   |0.670|0.666|0.708  |0.616|
|hivpr |0.780|0.681|0.759 |0.678|
|hivrt |0.651|0.670 |0.660 |0.584|
|kif11 |0.763|0.668 |0.672  |0.713|

**Table 1:** Macro averaged AUROC scores by target and molecular similarity technique for the DUDE diversity set. Across all datasets, 19 small molecules out of 112,796 were excluded from analysis either due to issues in parsing to valid structures using RDKit, or failed in OpenFEPOPS descriptor generation.


## Availability, usage and documentation

OpenFEPOPS has been uploaded to the Python Packaging Index under the name 'fepops' and as such is installable using the pip package manager and the command `pip install fepops`. With the package installed, entrypoints are used to expose commonly used OpenFEPOPS tasks such as descriptor generation and calculation on molecular similarity, enabling simple command line access without the need to explicitly invoke a Python interpreter. Whilst OpenFEPOPS may be used solely via the command line interface, a robust API is available and may be used within other programs or integrated into existing pipelines to enable more complex workflows.  Extensive API documentation is available at https://justinykc.github.io/FEPOPS, along with a concise user-guide at https://justinykc.github.io/FEPOPS/readme.html

# References