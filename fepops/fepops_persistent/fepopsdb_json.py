from .fepops_persistent_abc import FepopsPersistentAbstractBaseClass
from typing import Union
from pathlib import Path
import bz2
from joblib import Parallel, delayed
import numpy as np
import json
from base64 import b64encode, b64decode

class FepopsDBJSON(FepopsPersistentAbstractBaseClass):
	def __init__(
		self,
		database_file: Union[str, Path],
		kmeans_method: str = "pytorch-cpu",
		parallel: bool = True,
		n_jobs: int = -1,
	):
		super().__init__(
			database_file=database_file,
			kmeans_method=kmeans_method,
			parallel=parallel,
			n_jobs=n_jobs,
		)
		
		if not self.database_file.parent.exists():
			self.database_file.parent.mkdir(parents=True)
		if self.database_file.exists():
			self.db=json.load(open(self.database_file, "r"))
		else:
			self.db={}
		self._db_changed=False
		self._was_written=False
			

	def add_fepop(self, rdkit_canonical_smiles: str, fepops: np.ndarray):
		super().add_fepop(rdkit_canonical_smiles=rdkit_canonical_smiles, fepops=fepops)
		if not self.fepop_exists(rdkit_canonical_smiles=rdkit_canonical_smiles):
			self.db[rdkit_canonical_smiles]=b64encode(bz2.compress(fepops.tobytes())).decode('ascii')
			self._db_changed=True
			

	def get_fepop(self, smiles: str, is_canonical:bool=False) -> Union[np.ndarray, None]:
		super().get_fepop(smiles=smiles)
		if not is_canonical:
			smiles, mol = self._get_can_smi_mol_tuple(smiles)
		if self.fepop_exists(rdkit_canonical_smiles=smiles):
			return np.frombuffer(bz2.decompress(b64decode(self.db[smiles].encode()))).reshape(7,-1)
		else:
			new_fepops=self.fepops_object.get_fepops(mol)
			self.add_fepop(rdkit_canonical_smiles=smiles, fepops=new_fepops)
			return new_fepops
		
	def fepop_exists(self, rdkit_canonical_smiles: str):
		"""Check if Fepop exists in the database

		If the fepops object was constructed with a database file, then
		query if the supplied canonical SMILES is included.  If no database
		is present, then False is returned, as if it is not included.

		Parameters
		----------
		rdkit_canonical_smiles : _type_
						_description_

		Returns
		-------
		_type_
						_description_
		"""
		return rdkit_canonical_smiles in self.db
	
	def write(self):
		if self._db_changed:
			json.dump(self.db, open(self.database_file, "w"))
			self._was_written=True

	def __enter__(self):
		return self
	def __exit__(self, exc_type, exc_value, traceback):
		self.write()

	def __del__(self):
		if self._db_changed and not self._was_written:
			print("New fepops were added but changes were not written. Either call .write() or use FepopsDBJSON in a context, like with FepopsDBJSON() as fepops_jsondb...")
			


	