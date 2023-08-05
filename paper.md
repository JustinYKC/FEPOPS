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
    affiliation: 1
  - name: Steven Shave
    orcid: 0000-0001-6996-3663
    corresponding: true
    affiliation: 1
affiliations:
  - name: School of Biological Sciences, University of Edinburgh, The King’s Buildings, Max Born Crescent, CH Waddington Building, Edinburgh, EH9 3BF, U.K.
    index: 1
date: 6 August 2023
bibliography: paper.bib

---



# Summary

OpenFEPOPS is an open-source Python implementation of the FEPOPS molecular similarity technique [@jenkins20043d; @nettles2007flexible; jenkins2013feature] enabling descriptor generation, comparison, and ranking of molecules in virtual screening campaigns. Ligand based virtual screening [@ripphausen2011state] is a fundamental approach undertaken to expand hit series or perform scaffold hopping whereby new chemistries and synthetic routes are made available in efforts to remove undesirable molecular properties and discover better starting points in the early stages of drug discovery [@hughes2011principles]. Typically, these techniques query hit molecules against proprietary, in-house, or publicly available repositories of small molecules in the hope of finding close matches which will display similar activities to the query based on the molecular similarity principle (similar molecules should have similar properties and make similar interactions [@cortes2020qsar]). Often batteries of these similarity measures are used in parallel, helping to score molecules from many different subjective viewpoints and measures of similarity [@baber2006use]. The central idea behind FEPOPS is reducing the complexity of molecules by merging of local atomic environments and atom properties into ‘feature points’. This compressed feature point representation has been used to great effect as noted in literature, helping researchers identify active and potentially therapeutically valuable small molecules. By default, OpenFEPOPS uses literature reported parameters which show good performance in retrieval of active lead- and drug-like small molecules within virtual screening campaigns, with feature points capturing charge, lipophilicity, and hydrogen bond acceptor and donor status. When run with default parameters, OpenFepops compactly represents molecules using 4 feature points, each encoded into 22 numeric values, resulting in a compact representation of 352 bytes per molecule. By extension, this allows the indexing of a compound archive containing 1 million small molecules using only 356 MB of data. Whilst more compact representations are readily available, the FEPOPS technique strives to capture tautomer and conformer information, first through enumeration and then through diversity driven selection of representative FEPOPS descriptors to capture the diverse range of descriptor representations that a molecule may adopt.

# Statement of need

At the time of writing, `OpenFEPOPS` is the only publicly available implementation of the FEPOPS molecular similarity technique. Whilst used within industry and referenced extensively in literature, it has been unavailable to researchers as an open-source tool. This truly open implementation allows researchers to use and contribute to the advancement of FEPOPS within the rapid development and collaborative framework provided by open-source software. It is therefore hoped that this will allow the technique to be used not only for traditional small molecule molecular similarity, but also in new emerging fields such as protein design and featurization of small- and macromolecules for both predictive and generative tasks.

# Brief software description 

The FEPOPS descriptor generation process follows; for a given small molecule, OpenFEPOPS iterates over tautomers and conformers, picking four (by default) K-median atoms of each, into which the atomic information of neighbouring atoms is collapsed. As standard, the atomic properties of charge, logP, hydrogen bond donor, and hydrogen bond acceptor status are collapsed into four feature points per unique tautomer conformation. These feature points are encoded to 22 numeric values (a FEPOP) comprising four points, each with four properties, and six pairwise distances between these points. With four FEPOPS representing every enumerated conformer for every enumerated tautomer of a molecule, this set of representative FEPOPS should capture every possible state of the molecule. From this list, the K-means algorithm is applied to identify four diverse FEPOPS which are thought to capture a fuzzy representation of the molecule using four FEPOPS comprising 22 descriptors each, totalling 88 32-bit floating point numbers or 352 bytes.
OpenFEPOPS has been benchmarked against the DUD:E diversity set [@mysinger2012directory] and compared with three other common molecular similarity techniques. OpenFEPOPS achieves an average AUROC score of XXX versus XXX, XXX, and XXX respectively for three other similarity techniques which are; i) Morgan fingerprints with a radius of 2 and the Jaccard metric, ii) RDKit Fingerprints with a Radius of 6 and the Jaccard metric [@landrum2013rdkit], and finally, iii) the USRCAT molecular similarity technique [@schreyer2012usrcat]. Whilst the more traditional 2D graph-based fingeprints score higher than the two 3D descriptor-based techniques (OpenFEPOPS and USRCAT), the importance of applying diverse techniques in the early stages of drug discovery cannot be overstated, with many studies focusing on application of consensus schemes from diverse techniques to build robust predictors.

OpenFEPOPS has been uploaded to the Python Packaging Index under the name ‘fepops’ and as such is installable using the pip package manager and the command ‘pip install fepops’. With the package installed, entrypoints are used to expose commonly used OpenFEPOPS tasks such as descriptor generation and calculation on molecular similarity, enabling simple command line access without the need to explicitly invoke a Python interpreter. Whilst OpenFEPOPS may be used solely via the command line interface, a robust API is available and may be used within other programs or integrated into existing pipelines to enable more complex workflows.  Extensive online documentation is available at XXX. 

# References