from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MolStandardize
from rdkit.Chem import Crippen, Lipinski
from rdkit.Chem import rdMolTransforms
from sklearn.cluster import KMeans as _SKLearnKMeans
from fast_pytorch_kmeans import KMeans as _FastPTKMeans
from scipy.spatial.distance import cdist, squareform, pdist
from scipy.special import softmax
import numpy as np
from tqdm import tqdm
import itertools, zlib
import torch
from typing import Union, Optional, Tuple
from enum import Enum
import multiprocessing as mp
from multiprocessing import SimpleQueue

GetFepopStatusCode = Enum(
    "GetFepopStatusCode",
    ["SUCCESS", "FAILED_TO_GENERATE", "FAILED_TO_RETRIEVE", "FAILED_RETRIEVED_NONE"],
)


class Fepops:
    """Fepops molecular similarity object

    Fepops allows the comparison of molecules using feature points, see
    the original publication for more information https://pubs.acs.org/doi/10.1021/jm049654z
    In short, featurepoints reduce the number of points used to represent a molecule and can
    be used to compare molecules in the hope of discovering biosimilars based on queries.

    Parameters
    ----------
    kmeans_method : str, optional
        Method which should be used for kmeans calculation, can be
        one of "sklearn", "pytorch-gpu", or "pytorch-cpu". By
        default "pytorch-cpu".

    Raises
    ------
    ValueError
        Invalid kmeans method
    """

    def __init__(
        self,
        kmeans_method: str = "pytorch-cpu",
        max_tautomers: Optional[int] = None,
        *,
        num_fepops_per_mol: int = 7,
        num_centroids_per_fepop: int = 4,
    ):
        self.implemented_kmeans_methods = ["sklearn", "pytorch-cpu", "pytorch-gpu"]
        self.sort_by_features_col_index_dict = {
            name: sort_order_index
            for sort_order_index, name in enumerate(["charge", "logP", "hba", "hbd"])
        }
        self.num_fepops_per_mol = num_fepops_per_mol
        self.num_centroids_per_fepop = num_centroids_per_fepop
        self.num_features_per_fepop = len(self.sort_by_features_col_index_dict)
        self.num_distances_per_fepop = (
            (self.num_centroids_per_fepop**2) - self.num_centroids_per_fepop
        ) // 2
        self.donor_mol_from_smarts = Chem.MolFromSmarts("[!H0;#7,#8,#9]")
        self.acceptor_mol_from_smarts = Chem.MolFromSmarts(
            "[!$([#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]"
        )
        self.rotatable_bond_from_smarts = Chem.MolFromSmarts(
            "[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]"
        )
        if kmeans_method not in self.implemented_kmeans_methods:
            raise ValueError(
                f"Supplied argument kmeans_method '{kmeans_method}' not found, please supply a string denoting an implemented kmeans method from {self.implemented_kmeans_methods}"
            )
        self.kmeans_method_str = kmeans_method
        self.tautomer_enumerator = MolStandardize.tautomer.TautomerEnumerator(
            **{"max_tautomers": max_tautomers} if max_tautomers is not None else {}
        )

    def _get_k_medoids(
        self, input_x: np.ndarray, k: int = 7, seed: int = 42
    ) -> np.ndarray:
        """Select k Fepops from conformers

        Gets k mediods from conformers (and conformers of tautomers) which
        are representative of the molcule.

        Parameters
        ----------
        input_x : np.ndarray
            The pharmacophore features of all conformers.
        k : int
            The number of medoids for clustering. By default 7.

        Returns
        -------
        np.ndarray
            The final Fepops descriptors of the k representative conformers.
        """
        input_x = np.unique(input_x, axis=0)

        if input_x.shape[0] <= k:
            return input_x

        point_to_centroid_map = np.ones(input_x.shape[0])
        point_to_centroid_map_prev = np.zeros_like(point_to_centroid_map)

        np_rng = np.random.default_rng(seed=seed)
        medoids = input_x[
            np_rng.choice(np.arange(input_x.shape[0]), size=k, replace=False), :
        ]

        while (point_to_centroid_map != point_to_centroid_map_prev).any():
            point_to_centroid_map_prev = point_to_centroid_map
            point_to_centroid_map = np.argmin(
                np.square(cdist(input_x, medoids)), axis=1
            )
            for i in range(k):
                medoid_members = input_x[point_to_centroid_map == i]
                if len(medoid_members) == 0:
                    chosen_x_point = np_rng.choice(np.arange(input_x.shape[0]))
                    medoids[i] = input_x[chosen_x_point, :]
                    point_to_centroid_map[chosen_x_point] = i
                medoids[i] = np.median(input_x[point_to_centroid_map == i], axis=0)
        # Sorting at this stage for reproducibility with existing pregenerated
        # descriptor sets.
        return medoids[np.lexsort(medoids.T[::-1])]

    def annotate_atom_idx(self, mol: Chem.rdchem.Mol):
        for i, atom in enumerate(mol.GetAtoms()):
            # For each atom, set the property "molAtomMapNumber" to a custom number, let's say, the index of the atom in the molecule
            atom.SetProp("molAtomMapNumber", str(atom.GetIdx()))

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
        self, feature_dict: dict, centroid_atom_idx: np.ndarray
    ) -> int:
        """Sum all atomic features according to their centroid group

        A method that that is used to sum all the atomic features based on their centroid group.

        Parameters
        ----------
        feature_dict : dict
            A dictionary containing the atomic features.
        centroid_atom_idx : np.ndarray
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
        self,
        atom_coords: np.ndarray,
        num_centroids: int = 4,
        kmeans_method: str = "pytorch-cpu",
        seed: int = 42,
    ) -> tuple:
        """Perform kmeans calculation

        Carry out kmeans calcaultion based on a given kmeans method and number of centroids.

        Parameters
        ----------
        atom_coords : ndarray
            A Numpy array containing the 3D coordinates of a molecule.
        num_centroids : int
            The number of centoids used for clustering. By default 4.
        kmeans_method : str
            Method used to perform the kmeans calculation. Can be 'sklearn',
            'pytorch-cpu' or 'pytorch-gpu'. By default 'pytorch-cpu'.
        seed : int
            Seed for sklearn kmeans initialisation. By default 42.

        Returns
        -------
        tuple
            A tuple containing the centroid coordinates and the cluster labels of molecular atoms.
        """
        if kmeans_method == "sklearn":
            kmeans = _SKLearnKMeans(
                n_clusters=num_centroids, random_state=seed, n_init="auto"
            ).fit(atom_coords)
            centroid_coors = kmeans.cluster_centers_
            instance_cluster_labels = kmeans.labels_
        elif kmeans_method.startswith("pytorch"):
            torch.manual_seed(seed)
            mol_coors_torch = torch.from_numpy(atom_coords).to(
                "cuda" if kmeans_method.endswith("gpu") else "cpu"
            )
            kmeans = _FastPTKMeans(n_clusters=num_centroids, max_iter=300)
            instance_cluster_labels = kmeans.fit_predict(
                mol_coors_torch,
                centroids=torch.tensor(
                    atom_coords[:num_centroids], device=mol_coors_torch.device
                ),
            ).numpy()
            centroid_coors = kmeans.centroids.numpy()
        else:
            raise ValueError(
                f"The method selected for the k-means calculation is invalid, please use one of {self.implemented_kmeans_methods}"
            )
        return centroid_coors, instance_cluster_labels

    def _get_centroid_distances(
        self, centroid_coords_or_distmat: np.ndarray, is_distance_matrix: bool
    ) -> np.ndarray:
        """Get centroid distances array

        In the fepops paper using 4 centroids, there is a specific order in
        which to return the 4 distances:
        d1-4, d1-2, d2-3, d3-4, d1-3, d2-4.
        This order is the same as the way matrix determinants are calculated,
        and as such this function generalises to other cardinalities of points.


        Parameters
        ----------
        centroid_coords : np.ndarray
            MxN array of centroid coords, where M is the number of centroids,
            and N is the number of coordinates (should be 3).

        Returns
        -------
        np.ndarray
            Ordered centroid distances
        """
        if not is_distance_matrix:
            dmat = squareform(pdist(centroid_coords_or_distmat))
        else:
            dmat = centroid_coords_or_distmat.copy()
        distances = np.hstack([dmat[0,-1]]+[np.diagonal(dmat, offset=k) for k in range(1, dmat.shape[0]-1)])
        return distances

    def _mol_from_smiles(self, smiles_string: str) -> Chem.rdchem.Mol:
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
        try:
            mol = Chem.MolFromSmiles(smiles_string)
        except:
            try:
                mol = Chem.MolFromSmiles(smiles_string, sanitize=False)
            except:
                f"Could not parse smiles to a valid molecule, smiles was: {smiles_string}"
                return None
        return mol

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

    def _sample_bond_states(self, n_rot: int, seed: int) -> list:
        """Sample a set of conformers with different rotation angles

        A private method used to generate a set of bond angle multipliers (0 to 3) using
        all rotatable bonds within a molecule. Up to 1024 conformers are sampled for a molecule if
        n_rot is greater than five, otherwise all n_rot^4 are returned.

        Parameters
        ----------
        n_nor : int
            The number of rotatable bonds in a molecule.
        seed : int
            Seed for random sampling of rotamer space. Typically the hash of molecule coords.
        Returns
        -------
        List
            A list containing the sampled bond states (rotation angles) for all of the rotatable bonds of a molecule.
        """
        if n_rot <= 5:
            return list(itertools.product(range(4), repeat=n_rot))

        np_rng = np.random.default_rng(seed=seed)

        rotation_list = set(
            tuple(np_rng.choice(range(4), size=n_rot)) for counter in range(1024)
        )

        while len(rotation_list) < 1024:
            rotation_list.add(tuple(np_rng.choice(range(4), n_rot)))
        return list(rotation_list)

    def generate_conformers(self, mol: Chem.rdchem.Mol, random_seed: int = 42) -> list:
        """Generate conformers with rotatable bonds

        Generate conformers for a molecule, enumerating rotatable bonds over 90 degree angles.

        Parameters
        ----------
        mol : Chem.rdchem.Mol
            The Rdkit mol object of the input molecule.
        random_seed : reproducibility

        Returns
        -------
        List
            A list containing mol objects of different conformers with different angles of rotetable bonds.
        """
        try:
            mol = Chem.AddHs(mol)

            params = AllChem.ETKDGv3()
            params.useSmallRingTorsions = True
            params.randomSeed = random_seed
            original_conformer = mol.GetConformer(AllChem.EmbedMolecule(mol, params))
        except ValueError:
            params = AllChem.ETKDGv2()
            id = AllChem.EmbedMolecule(mol, params)
            if id == -1:
                print(
                    "Coords could not be generated without using random coords. using random coords now"
                )
                params.useRandomCoords = True
            try:
                original_conformer = mol.GetConformer(
                    AllChem.EmbedMolecule(mol, params)
                )
            except ValueError:
                print("Conformer embedding failed")
                return []
        dihedrals = self._get_dihedrals(mol)
        starting_angles = [
            rdMolTransforms.GetDihedralDeg(original_conformer, *dihedral_atoms)
            for dihedral_atoms in dihedrals
        ]
        bond_states = self._sample_bond_states(
            len(dihedrals), zlib.crc32(mol.GetConformer(0).GetPositions().tobytes())
        )

        new_conf_mol_list = []
        for bond_state in bond_states:
            self._generate_conf(
                original_conformer, dihedrals, starting_angles, bond_state
            )
            new_conf_mol_list.append(Chem.Mol(mol))

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
                orig_torsion_angle + torsion_angle_multiplier * 90.0,
            )

    def get_centroid_pharmacophoric_features(
        self,
        mol: Chem.rdchem.Mol,
        kmeans_method_str: str,
    ) -> np.ndarray:
        """Obtain the four centroids and their corresponding pharmacophoric features

        Obtain the four centroids and then calucate and assign their corresponding pharmacophoric
        features (logP, charges, HBA, HBD, six distances between four centroids).

        Parameters
        ----------
        mol : Chem.rdchem.Mol
            The Rdkit mol object of the input molecule.
        num_centroids : int
            sThe number of centoids used for clustering. By default 4.

        Returns
        -------
        np.ndarray
            A Numpy array containing 22 pharmacophoric features for all conformers.
        """
        centroid_coords, instance_cluster_labels = self._perform_kmeans(
            mol.GetConformer(0).GetPositions(),
            num_centroids=self.num_centroids_per_fepop,
            kmeans_method=kmeans_method_str,
        )

        atomic_logP_dict = self._calculate_atomic_logPs(mol)
        atomic_charge_dict = self._calculate_partial_charges(mol)
        hb_acceptors = set(
            i[0] for i in mol.GetSubstructMatches(self.acceptor_mol_from_smarts)
        )
        hb_donors = set(
            i[0] for i in mol.GetSubstructMatches(self.donor_mol_from_smarts)
        )
        pharmacophore_features_arr = np.empty(shape=[self.num_centroids_per_fepop, 4])
        for centroid in range(self.num_centroids_per_fepop):
            centroid_atomic_id = np.where(instance_cluster_labels == centroid)[0]

            sum_of_logP = self._sum_of_atomic_features_by_centroids(
                atomic_logP_dict, centroid_atomic_id
            )
            sum_of_charge = self._sum_of_atomic_features_by_centroids(
                atomic_charge_dict, centroid_atomic_id
            )

            if any(atom_id in hb_acceptors for atom_id in centroid_atomic_id):
                hba = 1
            else:
                hba = 0
            if any(atom_id in hb_donors for atom_id in centroid_atomic_id):
                hbd = 1
            else:
                hbd = 0

            pharmacophore_features_arr[centroid, :] = (
                sum_of_charge,
                sum_of_logP,
                hbd,
                hba,
            )

        sorted_index_rank_arr = np.lexsort(pharmacophore_features_arr.T[::-1])
        centroid_coords = centroid_coords[sorted_index_rank_arr]
        pharmacophore_features_arr = pharmacophore_features_arr[sorted_index_rank_arr]

        centroid_dist = self._get_centroid_distances(
            centroid_coords, is_distance_matrix=False
        )
        pharmacophore_features_arr = np.append(
            pharmacophore_features_arr, centroid_dist
        )
        return pharmacophore_features_arr

    def get_fepops(
        self, mol: Union[str, None, Chem.rdchem.Mol], is_canonical:bool=False,
    ) -> Tuple[GetFepopStatusCode, Union[np.ndarray, None]]:
        """Get Fepops descriptors

        This method returns Fepops descriptors from a smiles string.


        Parameters
        ----------
        mol : Union[str, None, Chem.rdchem.Mol]
            Molecule as a SMILES string or RDKit molecule. Can also be None,
            in which case a failure error status is returned along with None
            in place of the requested Fepops descriptors.

        Returns
        -------
        Tuple[GetFepopStatusCode, Union[np.ndarray, None]]
            Returns a tuple, with the first value being a GetFepopStatusCode
            (enum) denoting SUCCESS or FAILED_TO_GENERATE. The second tuple
            element is either None (if unsuccessful), or a np.ndarray containing
            the calculated Fepops descriptors of the requested input molecule.
        """
        original_smiles = None
        if isinstance(mol, np.ndarray):
            return GetFepopStatusCode.SUCCESS, mol
        if isinstance(mol, str):
            original_smiles = mol
            mol = self._mol_from_smiles(mol)
        if mol is None:
            print(
                f"Failed to make a molecule{' from '+original_smiles if original_smiles is not None else ''}"
            )
            return GetFepopStatusCode.FAILED_TO_GENERATE, None
        if Lipinski.HeavyAtomCount(mol) < self.num_centroids_per_fepop:
            print(
                f"Number of heavy atoms ({Lipinski.HeavyAtomCount(mol)}) below requested feature points ({self.num_centroids_per_fepop}) for molecule {original_smiles if original_smiles is not None else ''}"
            )
            return GetFepopStatusCode.FAILED_TO_GENERATE, None
        mol = Chem.AddHs(mol)

        tautomers_list = self.tautomer_enumerator.enumerate(mol)
        each_mol_with_all_confs_list = []
        for index, t_mol in enumerate(tautomers_list):
            conf_list = self.generate_conformers(t_mol)
            each_mol_with_all_confs_list.extend(conf_list)

        if each_mol_with_all_confs_list == []:
            print(
                f"Failed to generate conformers/tautomers {' for '+original_smiles if original_smiles is not None else ''}"
            )
            return GetFepopStatusCode.FAILED_TO_GENERATE, None

        pharmacophore_feature_all_confs = np.array(
            [
                self.get_centroid_pharmacophoric_features(
                    each_mol,
                    kmeans_method_str=self.kmeans_method_str,
                )
                for each_mol in each_mol_with_all_confs_list
            ]
        )

        medoids = self._get_k_medoids(
            pharmacophore_feature_all_confs, self.num_fepops_per_mol
        )
        return GetFepopStatusCode.SUCCESS, medoids
    def _score_scaler(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Score function for the similarity calculation

        The score function for the similarity calculation on the FEPOPS descriptors.

        Parameters
        ----------
        x1 : np.ndarray
            A Numpy array containing the FEPOPS descriptors 1.
        x2 : np.ndarray
            A Numpy array containing the FEPOPS descriptors 2.

        Returns
        -------
        float
            The FEPOPS similarity score (Pearson correlation).
        """
        x1 = self.scaler.fit_transform(x1.reshape(-1, 1))
        x2 = self.scaler.fit_transform(x2.reshape(-1, 1))
        return np.corrcoef(x1.flatten(), x2.flatten())[0, 1]
    
    def pairwise_correlation(self, A, B):
        am = A - np.mean(A, axis=0, keepdims=True)
        bm = B - np.mean(B, axis=0, keepdims=True)
        return am.T @ bm /  (np.sqrt(
            np.sum(am**2, axis=0,
                keepdims=True)).T * np.sqrt(
            np.sum(bm**2, axis=0, keepdims=True)))

    def _score_combialign(self, x1: np.ndarray, x2: np.ndarray):
        """Score fepops using CombiAlign

        Instead of sorting feature points by charge, this algorithm matches 2
        sets of medoids by holding one set constant and enumerating all
        permutations of the other, and performing pearson correlation calculations
        in a row-pairwise manner.  The highest summed correlation score permutation
        is then used for the second set, and scoring proceeds using softmax of the
        full fepops descriptors and the pearson correlation coefficient between
        them.

        The CombiAlign algorithm as defined in Nettles, James H., et al. "Flexible
        3D pharmacophores as descriptors of dynamic biological space." Journal of
        Molecular Graphics and Modelling 26.3 (2007): 622-633.

        Parameters
        ----------
        x1 : np.ndarray
            Query fepop
        x2 : np.ndarray
            Candidate fepop

        Returns
        -------
        float
            Fepops score, higher is better. 1 is the maximum.
        """
        x1_desc = x1[: -self.num_distances_per_fepop].reshape(
            self.num_centroids_per_fepop, self.num_features_per_fepop
        )
        x2_desc = x2[: -self.num_distances_per_fepop].reshape(
            self.num_centroids_per_fepop, self.num_features_per_fepop
        )
        x2_dists = x2[-self.num_distances_per_fepop :]

        permutation_tuples = list(
            itertools.permutations(range(self.num_centroids_per_fepop))
        )

        # Vectorised correlation coefficient calculations performed in numpy,
        # calculating all 24 permutations and the best fit (highest sum
        # correlation) used as the best fepop ordering of centroids.
        # This is magnitudes faster than a previous approach taken whereby
        # numpy corrcoef was run many times and resulted in extremely slow
        # scoring code. The folowing blog post helped immesely with this
        # code:
        # https://waterprogramming.wordpress.com/2014/06/13/numpy-vectorized-correlation-coefficient/

        # Make mega block of fepops for x2, containing all perturbations
        x2_mega_desc = x2_desc[permutation_tuples]
        x1_mean = x1_desc.mean(axis=-2)
        # As all permutations give the same mean (of columns/features),
        # just do the first.
        x2_mean = x2_mega_desc[0].mean(axis=-2)
        x1_residual = x1_desc - x1_mean
        x2_residual = x2_mega_desc - x2_mean
        numerator = np.sum(x1_residual * x2_residual, axis=-1)
        denominator = np.sqrt(
            np.sum(x1_residual**2, axis=-1)
            * np.sum((x2_mega_desc - x2_mean) ** 2, axis=-1)
        )
        best_permutaion = permutation_tuples[
            np.argmax(np.sum(numerator / denominator, axis=-1))
        ]
        
        # Rebuild x2 distances to squareform matrix, then reorder as per the best permutation and extract in required FEPOPS order.
        dmat = np.zeros((self.num_centroids_per_fepop, self.num_centroids_per_fepop))

        dmat[0, -1]=x2_dists[0]
        dist_position=1
        for offset in range(1, self.num_centroids_per_fepop-1):
            num_in_diagonal=self.num_centroids_per_fepop-offset
            dmat+=np.diag(x2_dists[dist_position:dist_position+num_in_diagonal], k=offset)
            dist_position+=num_in_diagonal
        # Not required, but for completeness, copy upper triangle to lower
        # triangle of dmat matrix
        dmat = dmat + dmat.T - np.diag(np.diag(dmat))
        
        # Reorder the distance matrix using best permutation
        
        new_dmat = np.zeros_like(dmat)
        for i, p in enumerate(best_permutaion):
            for j in range(dmat.shape[0]):
                if i == j:
                    continue
                new_dmat[i, j] = dmat[p, best_permutaion[j]]
        distances = self._get_centroid_distances(new_dmat, is_distance_matrix=True)

        # Reform x2 with reordered medoids and medoid distances
        x2 = np.hstack([x2_desc[[best_permutaion]].flatten(), distances])
        # Apply softmax and return pearson correlation between the two
        
        
        #return np.corrcoef(softmax(x1), softmax(x2))[0, 1]
        #return np.corrcoef(x1, x2)[0, 1]
        #x1 = self.scaler.fit_transform(x1.reshape(-1, 1))
        #x2 = self.scaler.fit_transform(x2.reshape(-1, 1))
        return self.pairwise_correlation((x1 - x1.mean())/x1.std(),(x2 - x2.mean())/x2.std())
        #return np.corrcoef(x1.flatten(), x2.flatten())[0, 1]
        #A= [0.4, 0.6]
        #SoftA = []

    def _init_worker_calc_similarity(self,query_descriptors_, candidate_descriptors_):
        global shared_query_descriptors, shared_candidate_descriptors
        shared_query_descriptors=query_descriptors_ 
        shared_candidate_descriptors=candidate_descriptors_

    def _work_calc_similarity(self, candidate_descriptor_i):
        global shared_query_descriptors, shared_candidate_descriptors
        return self.calc_similarity(shared_query_descriptors, shared_candidate_descriptors[candidate_descriptor_i])

    def calc_similarity(
        self,
        query: Union[np.ndarray, str, None],
        candidate: Union[np.ndarray, str, None, list[np.ndarray, str, None]],
    ) -> float:
        """Calculate FEPOPS similarity

        Method for calculating molecular similarity based on their FEPOPS descriptors.

        Parameters
        ----------
        query : Union[np.ndarray, str]
            A Numpy array containing the FEPOPS descriptors of the query molecule
            or a smiles string from which to generate FEPOPS descriptors for the
            query molecule. Can also be None, in which case, np.nan is returned
            as a score.
        candidate : Union[np.ndarray, str, None, list[np.ndarray, str, None]],
            A Numpy array containing the FEPOPS descriptors of the candidate
            molecule or a smiles string from which to generate FEPOPS descriptors
            for the candidate molecule.  Can also be None, in which case, np.nan is
            returned as a score, or a list of any of these. If it is a list,
            then a list of scores against the single candidate is returned.

        Returns
        -------
        float
            Fepops similarity between two molecules
        """
        if not isinstance(query, np.ndarray):
            query_status, query = self.get_fepops(query)
            if query_status != GetFepopStatusCode.SUCCESS:
                return np.nan

        if isinstance(candidate, list):
            shared_queue = SimpleQueue()
            scores=[]
            with mp.Pool(initializer=self._init_worker_calc_similarity, initargs=(query, candidate)) as pool:
                return pool.map(self._work_calc_similarity,(range(len(candidate))))
        if not isinstance(candidate, np.ndarray):
            candidate_status, candidate = self.get_fepops(candidate)
            if candidate_status != GetFepopStatusCode.SUCCESS:
                return np.nan
        return np.max(cdist(query, candidate, metric=self._score_combialign))

        



    def __call__(
        self,
        query: Union[np.ndarray, str],
        candidate: Union[np.ndarray, str],
    ) -> float:
        return self.calc_similarity(query, candidate)
