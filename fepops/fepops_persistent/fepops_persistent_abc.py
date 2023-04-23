from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MolStandardize
from rdkit.Chem import Crippen
from rdkit.Chem import rdMolTransforms
from sklearn.cluster import KMeans as _SKLearnKMeans
from fast_pytorch_kmeans import KMeans as _FastPTKMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist, squareform, pdist
import numpy as np
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
import io
from ..fepops import Fepops
import sys, random, itertools, argparse
import torch
from typing import Union, Callable, Optional
from abc import ABCMeta, abstractmethod
import json


class FepopsPersistentAbstractBaseClass(metaclass=ABCMeta):
	"""Abstract base class for persistent fepops storage

	New storage methods may be implemented as demonstrated in fepopsdb_json.py
	by extending this abstract base class which provides some required
	functionality like:
	
	- save_descriptors(smiles: Union[str, Path, list[str]]), to save a smiles file/list of smiles to the persistent storage
	- get_cansmi_to_mol_dict_not_in_database(smiles: Union[str, Path, list[str]]), to retrieve a unique dictionary with canonical smiles as keys not already stored in the database and rdkit mol objects as values.
	
	When writing your own persistent storage methods, you must override the following methods:
	
	add_fepop(rdkit_canonical_smiles: str, fepops: np.ndarray)
	----------------------------------------------------------
	Add the fepop to persistent storage. super().add_fepop may be called by the overridden function to perform type checks on arguments.
	
	fepop_exists(rdkit_canonical_smiles: str)
	-----------------------------------------
	Return True if the canonical smiles is already in the database, and False if not. super().fepop_exists may be called by the overridden
	function to perform type checks on arguments.

	get_fepop(rdkit_canonical_smiles: str)
	--------------------------------------
	Return a fepop from persistent storage. If it does not exist, then generate it by calling self.fepops_object.get_fepop which is supplied
	by this base class. super().get_fepop may be called by the overridden function to perform type checks on arguments.
	With this function in place it allows interface compatibility with a standard Fepops object.
	
	Inheriting functions may also define __enter__ and __exit__ methods for use with context handlers. If none are defined, then empty ones
	are provided. This can be useful in doing things like writing out large files after descriptor generation if incremental writes are not
	possible, like in the case of the FepopsDBJSON child class.


	Parameters
	----------
	database_file : Union[str, Path]
		File to use for persistent storage.
	kmeans_method : str, optional
		Method which should be used for kmeans calculation by
		fepops objects, can be one of "sklearn", "pytorch-gpu",
		or "pytorch-cpu".
	parallel : bool, optional
		Run in parallel (using joblib), by default True
	n_jobs : int, optional
		Number of jobs to be spawned with joblib. If -1, then use
		all available cores. By default -1.
	"""

	@abstractmethod
	def __init__(
		self,
		database_file: Union[str, Path],
		kmeans_method: str = "pytorch-cpu",
		parallel: bool = True,
		n_jobs: int = -1,
	):
		self.database_file = Path(database_file)
		self.fepops_object = Fepops(kmeans_method=kmeans_method)
		self.parallel = parallel
		self.n_jobs = n_jobs

	def save_descriptors(
		self,
		smiles: Union[str, Path, list[str]],
	):
		canonical_smiles_to_mol_dict = self.get_cansmi_to_mol_dict_not_in_database(
			smiles
		)
		if not self.parallel:
			for rdkit_canonical_smiles, mol in canonical_smiles_to_mol_dict.items():
				self.add_fepop(
					rdkit_canonical_smiles, self.fepops_object._get_fepops(mol)
				)
			print(
				f"Added {len(canonical_smiles_to_mol_dict)} new molecues to the database ({self.database_file})"
			)
		else:  # Do it in parallel
			cansmi_fepops_tuples = Parallel(n_jobs=self.n_jobs, prefer="threads")(
				delayed(lambda sm: (sm[0], self.fepops_object.get_fepops(sm[1])))(
					(cs, m)
				)
				for cs, m in canonical_smiles_to_mol_dict.items()
			)
			for rdkit_canonical_smiles, new_fepop in cansmi_fepops_tuples:
				self.add_fepop(
					rdkit_canonical_smiles, new_fepop
				)
			print(
				f"Added {len(canonical_smiles_to_mol_dict)} new molecues to the database ({self.database_file})"
			)

	@abstractmethod
	def add_fepop(self, rdkit_canonical_smiles: str, fepops: np.ndarray):
		"""Add canonical smiles and fepop to database.

		Must be overridden
		"""
		if not isinstance(rdkit_canonical_smiles, str):
			raise ValueError(
				f"Expected an rdkit canonical smiles string, but a {type(rdkit_canonical_smiles)} was passed"
			)
		if not isinstance(fepops, np.ndarray):
			raise ValueError(f"Expected a fepop, but a {type(fepops)} was passed")

	@abstractmethod
	def get_fepop(self, smiles: str) -> Union[np.ndarray, None]:
		if not isinstance(smiles, str):
			raise ValueError(
				f"Expected an rdkit canonical smiles string, but a {type(smiles)} was passed"
			)

	@abstractmethod
	def fepop_exists(self, rdkit_canonical_smiles: str) -> bool:
		"""Return True if canonical smiles already exist in the database"""
		if not isinstance(rdkit_canonical_smiles, str):
			raise ValueError(
				f"Expected an rdkit canonical smiles string, but a {type(rdkit_canonical_smiles)} was passed"
			)

	@staticmethod
	def _mol_from_smiles(smiles_string: str) -> Chem.rdchem.Mol:
		"""Parse smiles to mol, catching errors

		Parameters
		----------
		smiles_string : str
						Smiles string

		Returns
		-------
		Chem.rdchem.Mol
						RDkit molecule

		Raises
		------
		ValueError
						Unable to parse smiles into a molecule
		"""
		if not isinstance(smiles_string, str):
			raise ValueError(
				f"Expected a smiles string but a {type(smiles_string)} was passed"
			)
		try:
			mol = Chem.MolFromSmiles(smiles_string)
		except:
			try:
				mol = Chem.MolFromSmiles(smiles_string, sanitize=False)
			finally:
				mol = None
		if mol is None:
			raise ValueError(
				f"Could not parse smiles to a valid molecule, smiles was:{smiles_string}"
			)
		return mol

	@staticmethod
	def _get_can_smi_mol_tuple(s: str, is_canonical: bool = False):
		try:
			mol = Chem.MolFromSmiles(s)
		except:
			try:
				mol = Chem.MolFromSmiles(s, sanitize=False)
			finally:
				mol = None
		if mol is None:
			raise ValueError(
				f"Could not parse smiles to a valid molecule, smiles was:{s}"
			)
		if is_canonical:
			return (s, mol)
		else:
			return (Chem.CanonSmiles(Chem.MolToSmiles(mol)), mol)

	def get_cansmi_to_mol_dict_not_in_database(
		self, smiles: Union[str, Path, list[str]]
	):
		if isinstance(smiles, str):
			smiles = Path(smiles)
		if isinstance(smiles, Path):
			if smiles.exists():
				smiles = [s.strip() for s in open(smiles).readlines() if len(s.strip())>0]
			else:
				raise ValueError(
					f"smiles file ({smiles}) not found. If you are passing smiles, make it into a list"
				)
		if not isinstance(smiles, list):
			raise (
				"smiles should be a str or Path denoting the location of a smiles file, or a list of smiles"
			)
		smiles = list(set(smiles))
		if not self.parallel:
			# Ensure unique (canonical, also storing intermediate mol)
			canonical_smiles_to_mol_dict = dict(
				self._get_can_smi_mol_tuple(s)
				for s in tqdm(smiles, desc="Ensuring unique")
			)
		else:
			# Ensure unique (canonical, also storing intermediate mol)
			canonical_smiles_to_mol_dict = dict(
				Parallel(n_jobs=self.n_jobs)(
					delayed(FepopsPersistentAbstractBaseClass._get_can_smi_mol_tuple)(s)
					for s in smiles
				)
			)
		# Make sure none are already in the database
		canonical_smiles_to_mol_dict = {
			cansmi: mol
			for cansmi, mol in canonical_smiles_to_mol_dict.items()
			if not self.fepop_exists(cansmi)
		}
		return canonical_smiles_to_mol_dict
	
	def calc_similarity(
		self,
		query: Union[str, Path, list[str]],
		candidate: Union[str, Path, list[str]],
	):
		"""Calculate FEPOPS similarity

		A static method for calculating molecular similarity based on their FEPOPS descriptors.

		Parameters
		----------
		fepops_features_1 : Union[np.ndarray, str]
			A Numpy array containing the FEPOPS descriptors of the query molecule
			or a smiles string from which to generate FEPOPS descriptors for the
			query molecule.
		fepops_features_2 : Union[np.ndarray, str]
			A Numpy array containing the FEPOPS descriptors of the candidate
			molecule or a smiles string from which to generate FEPOPS descriptors
			for the query molecule.
		
		Returns
		-------
		float
			Fepops similarity between two molecules
		"""
		if isinstance(query, str):
			query = self.get_fepops(query)
		if isinstance(candidate, str):
			candidate = self.get_fepops(candidate)
		return np.max(cdist(query, candidate, metric=self.fepops_object._score))
	
	def write(self):
		pass

	def __enter__(self):
		return self
	
	def __exit__(self, exc_type, exc_value, traceback):
		self.write()
