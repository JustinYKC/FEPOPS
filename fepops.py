from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MolStandardize
from rdkit.Chem import Crippen
from rdkit.Chem import rdMolTransforms
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist, squareform, pdist
import numpy as np
import sys, random, itertools, argparse
import torch
import kmeans_pytorch


class Fepops:
    """Fepops molecular similarity object

    Fepops allows the comparison of molecules using feature points, see
    the original publication for more information. https://pubs.acs.org/doi/10.1021/jm049654z
    """

    def __init__(self):
        self.tautomer_enumerator = MolStandardize.tautomer.TautomerEnumerator()
        self.donor_mol_from_smarts = Chem.MolFromSmarts("[!H0;#7,#8,#9]")
        self.acceptor_mol_from_smarts = Chem.MolFromSmarts(
            "[!$([#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]"
        )
        self.rotatable_bond_from_smarts = Chem.MolFromSmarts(
            "[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]"
        )
        self.scaler = StandardScaler()

    def _get_k_medoids(self, input_x: np.array, k: int = 7) -> np.array:
        """Select k Fopops conformers to generate the final Fepops descriptors

        A private method used to perform k-medoids in order to derive the final Fepops descriptors
        by selecting k representative Fepops conformers.

        Parameters
        ----------
        input_x : np.array
                The pharmacophore features of all conformers.
        k : int
                The number of medoids for clustering. By default 7.

        Returns
        -------
        np.array
                The final Fepops descriptors of the k representative conformers.
        """
        input_x = np.unique(input_x, axis=0)

        if input_x.shape[0] <= k:
            return input_x

        point_to_centroid_map = np.ones(input_x.shape[0])
        point_to_centroid_map_prev = np.zeros_like(point_to_centroid_map)

        medoids = input_x[
            np.random.choice(np.arange(input_x.shape[0]), size=k, replace=False), :
        ]
        # medoids = input_x[np.argsort(np.sum(distances, axis=0))[np.round(np.linspace(0, len(input_x.shape[0]) - 1, k)).astype(int)]]

        while (point_to_centroid_map != point_to_centroid_map_prev).any():
            point_to_centroid_map_prev = point_to_centroid_map
            point_to_centroid_map = np.argmin(
                np.square(cdist(input_x, medoids)), axis=1
            )
            for i in range(k):
                medoid_members = input_x[point_to_centroid_map == i]
                if len(medoid_members) == 0:
                    chosen_x_point = np.random.choice(np.arange(input_x.shape[0]))
                    medoids[i] = input_x[chosen_x_point, :]
                    point_to_centroid_map[chosen_x_point] = i
                medoids[i] = np.median(
                    input_x[point_to_centroid_map == i], axis=0
                )  # Using median to ensure the minimum distance to its medoid_members to be returned
        return medoids

    def annotate_atom_idx(self, mol: Chem.rdchem.Mol):
        for i, atom in enumerate(mol.GetAtoms()):
            # For each atom, set the property "molAtomMapNumber" to a custom number, let's say, the index of the atom in the molecule
            atom.SetProp("molAtomMapNumber", str(atom.GetIdx()))

    def _get_tautomers(self, mol: Chem.rdchem.Mol) -> list:
        """Enumerate all tautomers for an input molecule

        A private function used for generating all tautomers for a molecule.

        Parameters
        ----------
        mol : Chem.rdchem.Mol
                The Rdkit mol object of the input molecule.

        Returns
        -------
        List
                List containing enumerated tautomers.
        """
        return self.tautomer_enumerator.enumerate(mol)

    def _calculate_atomic_logPs(self, mol: Chem.rdchem.Mol) -> dict:
        """Calculate logP contribution for each of atom in a molecule

        Parameters
        ----------
        mol : Chem.rdchem.Mol
                The Rdkit mol object of the input molecule.

        Returns
        -------
        dict
                A dictionary containing all atom symbols with their logP values.
        """
        return {
            atom.GetIdx(): float(contribution[0])
            for atom, contribution in zip(
                mol.GetAtoms(), Crippen.rdMolDescriptors._CalcCrippenContribs(mol)
            )
        }

    def _calculate_partial_charges(self, mol: Chem.rdchem.Mol) -> dict:
        """Calculate the charge of each atom in a molecule

        Parameters
        ----------
        mol : Chem.rdchem.Mol
                The Rdkit mol object of the input molecule.

        Returns
        -------
        dict
                A dictionary containing all atom symobls with their charges.
        """
        Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
        return {
            atom.GetIdx(): float(atom.GetProp("_GasteigerCharge"))
            for atom in mol.GetAtoms()
        }

    def _sum_of_atomic_features_by_centroids(
        self, feature_dict: dict, centroid_atom_idx: np.array
    ) -> int:
        """Sum all atomic features according to their centroid group

        A method that that is used to sum all the atomic features based on their centroid group.

        Parameters
        ----------
        feature_dict : dict
                A dictionary containing the atomic features.
        centroid_atom_idx : np.array
                A Numpy array containing the label of the centroid index for each atom.

        Returns
        -------
        int
                Sum of the features across the centroid group of atoms.
        """
        return sum([v for k, v in feature_dict.items() if k in centroid_atom_idx])

    def _get_dihedrals(self, mol: Chem.rdchem.Mol) -> tuple:
        """Identify dihedrals in order to obtain rotatable bonds in a molecule

        Identify dihedrals using flanking atoms of rotatable bonds

        Parameters
        ----------
        mol : Chem.rdchem.Mol
                The Rdkit mol object of the input molecule.

        Returns
        -------
        tuple
                A tuple containing all identified dihedrals with the index of their four defined atoms.
        """
        dihedrals = []
        for atom_j, atom_k in mol.GetSubstructMatches(self.rotatable_bond_from_smarts):
            atom_i, atom_l = self._get_flanking_atoms(mol.GetBonds(), atom_j, atom_k)
            if atom_i is not None and atom_l is not None:
                dihedrals.append((atom_i, atom_j, atom_k, atom_l))
        return dihedrals
    
    def _perform_kmeans(
        self, mol:Chem.rdchem.Mol, num_centroids:int=4, kmeans_methods:str='torch_cpu'
    ) -> tuple:
      mol_coors = mol.GetConformer(0).GetPositions()
      if kmeans_methods == 'sklearn':
        kmeans = KMeans(n_clusters=num_centroids, random_state=0).fit(mol_coors)
        centroid_coors = kmeans.cluster_centers_
        instance_cluster_labels = kmeans.labels_
      if kmeans_methods.startswith('torch_'):
        mol_coors_torch = torch.from_numpy(mol_coors)
        if kmeans_methods == 'torch_gpu':
          instance_cluster_labels, centroid_coors = kmeans_pytorch.kmeans(
            X=mol_coors_torch, num_clusters=num_centroids, distance='euclidean', device=torch.device('cuda:0')
          )
        if kmeans_methods == 'torch_cpu':
          instance_cluster_labels, centroid_coors = kmeans_pytorch.kmeans(
            X=mol_coors_torch, num_clusters=num_centroids, distance='euclidean', device=torch.device('cpu')
          )
        instance_cluster_labels = instance_cluster_labels.numpy()
        centroid_coors = centroid_coors.numpy()
      else: 
        print ("The method selected for the k-means calculation is invalid, please use the valid format in terms of 'sklearn', 'torch_cpu', or 'torch_gpu'")
        sys.exit(-1)
      return centroid_coors, instance_cluster_labels

    def _get_flanking_atoms(
        self, bonds: Chem.rdchem._ROBondSeq, atom_1_idx: int, atom_2_idx: int
    ) -> tuple:
        """Search for two atoms connecting to either atom in a rotatable bond

        A private method to identify two atoms flanking atoms in a rotatable bond

        Parameters
        ----------
        bonds : Chem.rdchem._ROBondSeq
                The Rdkit molecule bond object that contains the indexes of both begin and end atoms in a bond.
        atom_1_idx : int
                The index of the first atom in a rotatable bond.
        atom_2_inx : int
                The index of the second atom in a rotatable bond.

        Returns
        -------
        tuple
                A tuple containing the indexes of two flanking atoms for the given atoms of a rotatable bond.
        """
        bound_to_atom_1 = None
        bound_to_atom_2 = None

        for bond in bonds:
            bond_indexes = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if atom_1_idx not in bond_indexes and atom_2_idx not in bond_indexes:
                continue

            if atom_1_idx in bond_indexes:  # Atom 1 in bond indexes
                if atom_2_idx in bond_indexes:
                    continue
                if atom_1_idx == bond_indexes[0]:
                    bound_to_atom_1 = bond_indexes[1]
                    continue
                if atom_1_idx == bond_indexes[1]:
                    bound_to_atom_1 = bond_indexes[0]
                    continue
            else:  # Atom 2 in bond indexes
                if atom_2_idx == bond_indexes[0]:
                    bound_to_atom_2 = bond_indexes[1]
                    continue
                if atom_2_idx == bond_indexes[1]:
                    bound_to_atom_2 = bond_indexes[0]
                    continue

                if bound_to_atom_1 is not None and bound_to_atom_2 is not None:
                    return bound_to_atom_1, bound_to_atom_2
        return bound_to_atom_1, bound_to_atom_2

    def _sample_bond_states(self, n_rot: int) -> list:
        """Sample a set of conformers with different rotation angles

        A private method used to generate a set of bond angle multipliers (0 to 3) using
        all rotatable bonds within a molecule. Up to 1024 conformers are sampled for a molecule if
        n_rot is greater than five, otherwise all n_rot^4 are returned.

        Parameters
        ----------
        n_nor : int
                The number of rotatable bonds in a molecule.

        Returns
        -------
        List
                A list containing the sampled bond states (rotation angles) for all of the rotatable bonds of a molecule.
        """
        if n_rot <= 5:
            return list(itertools.product(range(4), repeat=n_rot))

        rotation_list = set(
            tuple(random.choice(range(4)) for _ in range(n_rot))
            for counter in range(1024)
        )
        while len(rotation_list) < 1024:
            rots = [random.choice(range(4)) for _ in range(n_rot)]
            rotation_list.add(tuple(rots))
        return list(rotation_list)

    def generate_conformers(self, mol: Chem.rdchem.Mol) -> list:
        """Generate conformers with rotatable bonds

        Generate conformers for a molecule, enumerating rotatable bonds over 90 degree angles.

        Parameters
        ----------
        mol : Chem.rdchem.Mol
                The Rdkit mol object of the input molecule.

        Returns
        -------
        List
                A list containing mol objects of different conformers with different angles of rotetable bonds.
        """
        AllChem.EmbedMolecule(mol, randomSeed=10)
        original_conformer = mol.GetConformer(0)
        dihedrals = self._get_dihedrals(mol)
        starting_angles = (
            rdMolTransforms.GetDihedralDeg(original_conformer, *dihedral_atoms)
            for dihedral_atoms in dihedrals
        )
        bond_states = self._sample_bond_states(len(dihedrals))

        new_conf_mol_list = []
        for bond_state in bond_states:
            self._generate_conf(
                original_conformer, dihedrals, starting_angles, bond_state
            )
            new_conf_mol_list.append(mol)
        # return [self._generate_conf(original_conformer, dihedrals, starting_angles, bond_state) for bond_state in bond_states]
        return new_conf_mol_list

    def _generate_conf(
        self,
        conformer: Chem.rdchem.Conformer,
        dihedrals: tuple,
        starting_angles: tuple,
        bond_state: tuple,
    ) -> None:
        """Rotate the assigned rotatable bonds

        Change conformers by Rotating the assigned rotatable bond based onset dihedral angles defined
        by four flanking atoms.

        Parameters
        ----------
        conformer : Chem.rdchem.Conformer
                The Rdkit conformer object.
        dihedrals : tuple
                A tuple containing all identified dihedrals with the index of their four defined atoms.
        starting_angles : tuple
                A tuple containing the orignal states (dihedral angles) of all the rotatable bond before rotating.
        bond_state : tuple
                A tuple containing a specific bond state (a combination of various rotation angles) for all
                rotatable bonds of a molecule.
        """
        for dihedral_atoms, torsion_angle_multiplier, orig_torsion_angle in zip(
            dihedrals, bond_state, starting_angles
        ):
            rdMolTransforms.SetDihedralDeg(
                conformer,
                *dihedral_atoms,
                orig_torsion_angle + torsion_angle_multiplier * 90.0
            )

    def get_centroid_pharmacophoric_features(
        self, mol: Chem.rdchem.Mol, num_centroids: int = 4
    ) -> np.array:
        """Obtain the four centroids and their corresponding pharmacophoric features

        Obtain the four centroids and then calucate and assign their corresponding pharmacophoric
        features (logP, charges, HBA, HBD, six distances between four centroids).

        Parameters
        ----------
        mol : Chem.rdchem.Mol
                The Rdkit mol object of the input molecule.
        num_centroids : int
                The number of centoids used for clustering. By default 4.

        Returns
        -------
        np.array
                A Numpy array containing 22 pharmacophoric features for all conformers.
        """
        centroid_coors, instance_cluster_labels = self._perform_kmeans(mol, num_centroids, "torch_cpu")
        centroid_dist_arr = cdist(centroid_coors, centroid_coors)
        centroid_dist = list(
            centroid_dist_arr[np.triu_indices_from(centroid_dist_arr, k=1)]
        )

        atomic_logP_dict = self._calculate_atomic_logPs(mol)
        atomic_charge_dict = self._calculate_partial_charges(mol)
        hb_acceptors = set(
            i[0] for i in mol.GetSubstructMatches(self.acceptor_mol_from_smarts)
        )
        hb_donors = set(
            i[0] for i in mol.GetSubstructMatches(self.donor_mol_from_smarts)
        )

        pharmacophore_features_arr = np.array([], ndmin=1)
        for centroid in range(num_centroids):
            hba, hbd = 0.0, 0.0
            centroid_atomic_id = np.where(instance_cluster_labels == centroid)[0]
            sum_of_logP = self._sum_of_atomic_features_by_centroids(
                atomic_logP_dict, centroid_atomic_id
            )
            sum_of_charge = self._sum_of_atomic_features_by_centroids(
                atomic_charge_dict, centroid_atomic_id
            )
            if len(hb_acceptors.intersection(set(centroid_atomic_id))) > 0:
                hba = 1
            if len(hb_donors.intersection(set(centroid_atomic_id))) > 0:
                hbd = 1
            pharmacophore_features_arr = np.append(
                pharmacophore_features_arr, [sum_of_logP, sum_of_charge, hba, hbd]
            )

        pharmacophore_features_arr = np.append(
            pharmacophore_features_arr, centroid_dist
        )
        # print (pharmacophore_features_arr, pharmacophore_features_arr.shape)
        return pharmacophore_features_arr

    def get_fepops(self, smiles_string: str) -> np.array:
        """Get Fepops descriptors

        This method returns a set of the Fepops descriptors when feeding a molecular SMILES string.

        Parameters
        ----------
        smiles_string : str
                SMILES string of an input molecule.

        Returns
        -------
        np.array
                A Numpy array containing the calculated Fepops descriptors of an input molecule.
        """

        tautomers_list = self._get_tautomers(
            Chem.AddHs(Chem.MolFromSmiles(smiles_string))
        )
        each_mol_with_all_confs_list = []
        for index, t_mol in enumerate(tautomers_list):
            conf_list = self.generate_conformers(t_mol)
            each_mol_with_all_confs_list.extend(conf_list)

        for index, each_mol in enumerate(each_mol_with_all_confs_list):
            pharmacophore_feature = self.get_centroid_pharmacophoric_features(each_mol)
            if index == 0:
                pharmacophore_feature_all_confs = pharmacophore_feature
                continue
            pharmacophore_feature_all_confs = np.vstack(
                (pharmacophore_feature_all_confs, pharmacophore_feature)
            )
        return self._get_k_medoids(pharmacophore_feature_all_confs, 7)

    def _score(self, x1: np.array, x2: np.array) -> float:
        """Score function for the similarity calculation

        The score function for the similarity calculation on the FEPOPS descriptors.

        Parameters
        ----------
        x1 : np.array
                A Numpy array containing the FEPOPS descriptors 1.
        x2 : np.array
                A Numpy array containing the FEPOPS descriptors 2.

        Returns
        -------
        float
                The FEPOPS similarity score (Pearson correlation).
        """
        x1 = self.scaler.fit_transform(x1.reshape(-1, 1))
        x2 = self.scaler.fit_transform(x2.reshape(-1, 1))
        return np.corrcoef(x1.flatten(), x2.flatten())[0, 1]

    def calc_similarity(
        self, fepops_features_1: np.array, fepops_features_2: np.array
    ) -> float:
        """Calculate FEPOPS similarity

        A static method for calculating molecular similarity based on their FEPOPS descriptors.

        Parameters
        ----------
        fepops_features_1 : np.array
                A Numpy array containing the FEPOPS descriptors of the first molecule
        fepops_features_2 : np.array
                A Numpy array containing the FEPOPS descriptors of the second molecule

        Returns
        -------
        float
                Fepops similarity between two molecules
        """
        return np.max(cdist(fepops_features_1, fepops_features_2, metric=self._score))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the FEPOPS descriptors")
    subparsers = parser.add_subparsers(
        dest="subcmd", help="subcommands", metavar="SUBCOMMAND"
    )
    subparsers.required = True

    f1_parser = subparsers.add_parser(
        "get_fepops", help="Generate the FEPOPS descriptors from an input SMILES string"
    )
    f1_parser.add_argument(
        "-ismi",
        "--input_smiles",
        help="An input SMILES string of a molecule",
        dest="insmile",
        required=True,
    )

    f2_parser = subparsers.add_parser(
        "cal_sim", help="Calculate FEPOPS similarity between two molecules"
    )
    f2_parser.add_argument(
        "-ismi1",
        "--input_smiles_1",
        help="An input SMILES string of the first molecule",
        dest="insmiles1",
        required=True,
    )
    f2_parser.add_argument(
        "-ismi2",
        "--input_smiles_2",
        help="An input SMILES string of the second molecule",
        dest="insmiles2",
        required=True,
    )

    args = parser.parse_args()
    f = Fepops()
    if args.subcmd == "get_fepops":
        fepops_features = f.get_fepops(args.insmile)
        print(fepops_features)

    if args.subcmd == "cal_sim":
        fepops_features_1 = f.get_fepops(args.insmiles1)
        fepops_features_2 = f.get_fepops(args.insmiles2)
        print(f.calc_similarity(fepops_features_1, fepops_features_2))
