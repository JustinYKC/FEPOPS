from typing import Optional

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize


class Filter:
    """Filter molecules by property

    Filter object allows filtering by min atoms, max rings, and max rotatable
    bonds present within a molecule.
    """

    def __init__(
        self, min_atoms: int = 4, max_rings: int = 9, max_n_rot: int = 40
    ) -> None:
        self.min_atoms = min_atoms
        self.max_rings = max_rings
        self.max_n_rot = max_n_rot

    def __call__(self, mol: Chem.rdchem.Mol) -> Optional[Chem.rdchem.Mol]:
        """Apply all filters

        Apply four filters, removing molecules which violate the defined minimum
        cutoff for the number of atoms present and maximum cutoffs for rings and
        rotatable bonds. Additionally remove salts/ions.

        Parameters
        ----------
        mol : Chem.rdchem.Mol
          The Rdkit Mol object of the input molecule.

        Returns
        -------
        Optional[Chem.rdchem.Mol]
          Chem.rdchem.Mol if the molecule passes all filters, otherwise None.
        """
        return self.filter_mol(mol)

    def filter_mol(self, mol: Chem.rdchem.Mol) -> Optional[Chem.rdchem.Mol]:
        """Apply all filters

        Apply four filters, removing molecules which violate the defined minimum
        cutoff for the number of atoms present and maximum cutoffs for rings and
        rotatable bonds. Additionally remove salts/ions.

        Parameters
        ----------
        mol : Chem.rdchem.Mol
          The Rdkit Mol object of the input molecule.

        Returns
        -------
        Optional[Chem.rdchem.Mol]
          Chem.rdchem.Mol if the molecule passes all filters, otherwise None.
        """
        mol = self.filter_num_atoms(mol, self.min_atoms)
        if mol is None:
            return None
        mol = self.filter_num_rings(mol, self.max_rings)
        if mol is None:
            return None
        mol = self.rotation_bond_num(mol, self.max_n_rot)
        if mol is None:
            return None
        mol = self.remove_salts(mol)
        return mol

    def smiles_to_mol(self, query_smiles: str) -> Chem.rdchem.Mol:
        """Read SMILES strings from a file

        Read molecular SMILES strings as Rdkit mol objects from a file.

        Parameters
        ----------
        query_smiles : str
          A molecular SMILES string.

        Returns
        -------
        Chem.rdchem.Mol
          Return a Rdkit mol object.
        """
        return Chem.MolFromSmiles(query_smiles)

    def filter_num_atoms(
        self, mol: Chem.rdchem.Mol, cutoff: int = 4
    ) -> Optional[Chem.rdchem.Mol]:
        """Filter out molecules with atom counts less than cutoff

        Parameters
        ----------
        mol : Chem.rdchem.Mol
          Input molecule
        cutoff : int
          The number of atoms which the molecule must have to pass this filter
          step. By default 4.

        Returns
        -------
        Optional[Chem.rdchem.Mol]
          Chem.rdchem.Mol if the input molecule meets the cutoff criteria,
          otherwise None.
        """
        atom_num = mol.GetNumAtoms()
        if atom_num > cutoff:
            return mol
        else:
            return None

    def filter_num_rings(
        self, mol: Chem.rdchem.Mol, cutoff: int = 9
    ) -> Optional[Chem.rdchem.Mol]:
        """Filter out molecules with ring counts greater than cutoff

        Parameters
        ----------
        mol : Chem.rdchem.Mol
          Input molecule
        cutoff : int
          The number of rings which the molecule must have less than to pass
          this filter. By default 9

        Returns
        -------
        Optional[Chem.rdchem.Mol]
          Chem.rdchem.Mol if the input molecule meets the cutoff criteria,
          otherwise None.
        """
        ring_num = mol.GetRingInfo().NumRings()
        if ring_num < cutoff:
            return mol
        else:
            return None

    def rotation_bond_num(
        self, mol: Chem.rdchem.Mol, cutoff: int = 40
    ) -> Optional[Chem.rdchem.Mol]:
        """Filter out molecules with rotatable bond counts greater than cutoff

        Parameters
        ----------
        mol : Chem.rdchem.Mol
          Input molecule
        cutoff : int
          The number of rotatable bonds which the molecule must have less than
          to pass this filter. By default 40

        Returns
        -------
        Optional[Chem.rdchem.Mol]
          Chem.rdchem.Mol if the input molecule meets the cutoff criteria,
          otherwise None.
        """
        rotation_bond_num = rdMolDescriptors.CalcNumRotatableBonds(mol)
        if rotation_bond_num < cutoff:
            return mol
        else:
            return None

    def remove_salts(self, mol: Chem.rdchem.Mol) -> Optional[Chem.rdchem.Mol]:
        """Remove salts and ions from molecules

        Parameters
        ----------
        mol : Chem.rdchem.Mol
          Input molecule/salt

        Returns
        -------
        Optional[Chem.rdchem.Mol]
          Standardised molecule or None if an error in standardisation occurs
        """
        lfc = rdMolStandardize.LargestFragmentChooser()
        try:
            cleaned_molecule = rdMolStandardize.Cleanup(mol)
            smiles = Chem.MolToSmiles(cleaned_molecule)
            if "." in smiles:
                cleaned_molecule = rdMolStandardize.Cleanup(
                    lfc.choose(cleaned_molecule)
                )
        except:
            return None
        return cleaned_molecule
