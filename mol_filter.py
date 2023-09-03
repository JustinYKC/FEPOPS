from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
import pandas as pd
from typing import Optional


class Filter:
    """Filter object allows filtering of molecules by property

    Filter object allowing filtering by min atoms, max rings, and max rotatable bonds in a molecule.
    """

    def __init__(
        self, min_atoms: int = 4, max_rings: int = 9, max_n_rot: int = 40
    ) -> None:
        self.min_atoms = min_atoms
        self.max_rings = max_rings
        self.max_n_rot = max_n_rot

    def __call__(self, mol: Chem.rdchem.Mol) -> Optional[Chem.rdchem.Mol]:
        """Apply all filters

        Parameters
        ----------
        mol : Chem.rdchem.Mol
          The Rdkit mol object of the input molecule.

        Returns
        -------
        filter_result
          Return a Rdkit mol object if the input molecule meets all criteria, otherwise return None.
        """
        filter_result = self.filter_mol(mol)
        return filter_result

    def filter_mol(self, mol: Chem.rdchem.Mol) -> Optional[Chem.rdchem.Mol]:
        """Apply all filters

        Perform four filters (minimum number of atoms, maximum number of rings, maximum number of rotatable bonds,
        and salt/ion) in sequence to obtain the final molecules as required.

        Parameters
        ----------
        mol : Chem.rdchem.Mol
          The Rdkit mol object of the input molecule.

        Returns
        -------
        Optional[Chem.rdchem.Mol]
          Return a Rdkit mol object if the input molecule meets all criteria, otherwise return None.
        """
        mol = self.atom_num(mol, self.min_atoms)
        if mol is None:
            return None
        else:
            mol = self.ring_num(mol, self.max_rings)
            if mol is None:
                return None
            else:
                mol = self.rotation_bond_num(mol, self.max_n_rot)
                if mol is None:
                    return None
                else:
                    mol = self.remove_salt(mol)
                    if mol is None:
                        return None
                    else:
                        return mol

    def read_as_mol(self, query_smiles: str) -> Chem.rdchem.Mol:
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

    def atom_num(
        self, mol: Chem.rdchem.Mol, cutoff: int = 4
    ) -> Optional[Chem.rdchem.Mol]:
        """Filter molecules by number of atoms

        Filter out molecules by the number of atoms.

        Parameters
        ----------
        mol : Chem.rdchem.Mol
          The Rdkit mol object of the input molecule.
        cutoff : int
          The criterion of atom numbers for filtering. By default 4.

        Returns
        -------
        Optional[Chem.rdchem.Mol]
          Return a Rdkit mol object if the input molecule meets the criterion, otherwise return None.
        """
        atom_num = mol.GetNumAtoms()
        if atom_num > cutoff:
            return mol
        else:
            return None

    def ring_num(
        self, mol: Chem.rdchem.Mol, cutoff: int = 9
    ) -> Optional[Chem.rdchem.Mol]:
        """Fliter molecules by number of rings

        Fliter out molecules by the number of rings.

        Parameters
        ----------
        mol : Chem.rdchem.Mol
          The Rdkit mol object of the input molecule.
        cutoff : int
          The criterion of ring numbers for filtering. By default, 9.

        Returns
        -------
        Optional[Chem.rdchem.Mol]
          Return a Rdkit mol object if the input molecule meets the criterion, otherwise return None.
        """
        ring_num = mol.GetRingInfo().NumRings()
        if ring_num < cutoff:
            return mol
        else:
            return None

    def rotation_bond_num(
        self, mol: Chem.rdchem.Mol, cutoff: int = 40
    ) -> Optional[Chem.rdchem.Mol]:
        """Fliter molecules by number of rotatable bonds

        Fliter out molecules by the number of rotatable bonds.

        Parameters
        ----------
        mol : Chem.rdchem.Mol
          The Rdkit mol object of the input molecule.
        cutoff : int
          The criterion of rotatable bond numbers for filtering. By default 40.

        Returns
        -------
        Optional[Chem.rdchem.Mol]
          Return a Rdkit mol object if the input molecule meets the criterion, otherwise return None.
        """
        rotation_bond_num = rdMolDescriptors.CalcNumRotatableBonds(mol)
        if rotation_bond_num < cutoff:
            return mol
        else:
            return None

    def remove_salt(self, mol: Chem.rdchem.Mol) -> Optional[Chem.rdchem.Mol]:
        """Romve salts or ions from molecules

        Remove salts or ions from molecules

        Parameters
        ----------
        mol : Chem.rdchem.Mol
          The Rdkit mol object of the input molecule.

        Returns
        -------
        Optional[Chem.rdchem.Mol]
          Return a Rdkit mol object if the input molecule meets the criterion, otherwise return None.
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

    def print_valid_cpd(self, ori_df: pd.DataFrame) -> None:
        """Print out the result of filtering

        Parameters
        ----------
        ori_df : pd.DataFrame
          The resulting dataframe after applying a filter.
        """
        print(f"Number of the original molecules: {ori_df.shape[0]}")
        # print (f"{ori_df.shape[0] - filtered_df.shape[0]} invalid smiles found, dataset contains {filtered_df.shape[0]} valid molecules")
        print(f"Number of molecules should be removed: {ori_df.Mol.isnull().sum()}")


if __name__ == "__main__":
    dataset_raw = pd.read_csv(
        "./COCONUT_DB.smi", sep=" ", header=0, names=["SMILES", "ID"]
    )
    print(dataset_raw)

    filter = Filter()
    dataset_raw["Mol"] = dataset_raw.SMILES.apply(filter.read_as_mol)
    filter.print_valid_cpd(dataset_raw)
    dataset_vali = dataset_raw.loc[~dataset_raw["Mol"].isna()]

    dataset_vali["Mol"] = dataset_vali.Mol.apply(filter)
    filter.print_valid_cpd(dataset_vali)

    dataset_vali = dataset_vali.loc[~dataset_vali["Mol"].isna()]
    print(dataset_vali)

    dataset_vali["Std_SMILES"] = dataset_vali.Mol.apply(Chem.MolToSmiles)
    dataset_vali.to_csv(
        "./COCONUT_after_filter.smi",
        columns=["Std_SMILES", "ID"],
        sep="\t",
        index=False,
        header=False,
    )
