.. OpenFEPOPS documentation master file, created by
   sphinx-quickstart on Mon Aug  7 00:33:31 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to OpenFEPOPS's documentation
=====================================

OpenFEPOPS is an open-source Python implementation of the FEPOPS molecular similarity technique enabling
descriptor generation,comparison, and ranking of molecules in virtual screening campaigns. The central
idea behind FEPOPS is reducing the complexity of molecules by merging of local atomic environments and
atom properties into ‘feature points’. This compressed feature point representation has been used to
great effect as noted in literature, helping researchers identify active and potentially therapeutically
valuable small molecules. This implementation was recreated following the original paper:
`<https://pubs.acs.org/doi/10.1021/jm049654z>`_


Source code is available on the GitHub repo here: `OpenFEPOPS GitHub Repository <https://github.com/JustinYKC/FEPOPS>`_

An online version of the GitHub README.md can be found here which documents how to use the package for
calculation of descriptors and subsequent molecular similarity calculations and is available here: `readme`_

.. toctree::
   :maxdepth: 2
   :caption: API documentation:

   modules
   readme

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
