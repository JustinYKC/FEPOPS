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
import itertools, zlib
import torch
from typing import Union, Optional, Tuple, Literal
from enum import Enum
import multiprocessing as mp
from multiprocessing import SimpleQueue
import logging

GetFepopStatusCode = Enum(
    "GetFepopStatusCode",
    ["SUCCESS", "FAILED_TO_GENERATE", "FAILED_TO_RETRIEVE", "FAILED_RETRIEVED_NONE"],
)


class OpenFEPOPS:
    """OpenFEPOPS (Feature Points) molecular similarity object

    Fepops allows the comparison of molecules using feature points, see the
    original publication for more information:
    https://doi.org/10.1021/jm049654z. In short, featurepoints reduce the number
    of points used to represent a molecule by combining atoms and their
    properties. Typically used to compare libraries of small molecules against
    known actives in the hope of discovering biosimilars based on queries.

    Parameters
    ----------
    kmeans_method : str, optional
        String literal denoting the method which should be used for kmeans
        calculations. May be one of "sklearn", "pytorchgpu", or "pytorchcpu". If
        "sklearn" is passed then Scikit-learn's kmeans implementation is used.
        However a faster implementation from the fast_pytorch_kmeans package can
        also be used if Pytorch is available and may be run in cpu-only mode, or
        GPU accelerated mode. Note: GPU accelerated mode should only be used if
        you are stretching the capabilities in terms of feature points for large
        molecules.  Small molecules will not benefit at all from GPU
        acceleration due to overheads.  By default "sklearn"
    max_tautomers : Optional[int], optional
        Maximum number of tautomers which should be generated. Internally, this
        implementation of FEPOPS relies upon RDKit's TautomerEnumerator to
        generate tautomers and may optionally pass in a limit to the number of
        Tautomers to generate. Unless the molecules (or macromolecules) you are
        working with generate massive numbers of tautomers, this should be None
        implying that no limit should be placed on tautomer generation. By
        default None
    num_fepops_per_mol : int, optional
        Number of feature points to use in the representation of a molecule.
        Literature notes that 7 has been empirically found to be a good number
        of feature points for performant representations of small molecules.
        This might be increased if you are dealing with large and very flexible
        molecules, by default 7
    num_centroids_per_fepop : int, optional
        Each fepop is represented by a number of centres, into which atom
        properties are compressed. Literature notes that this has been
        empirically determined to be 4 for a performant representation of small
        molecules. By default 4
    descriptor_means : Tuple[float, ...], optional
        Due to the need to apply scaling to FEPOPS, the DUDE diversity set has
        been profiled and the means collected for all contained FEPOPS. This
        this allows centering and scaling of FEPOPS before scoring. This field
        contains default values for FEPOP means calculated with
        num_fepops_per_mol = 7, num_centroids_per_fepop=4, and kmeans_method =
        'sklearn'. New values should be supplied if the FEPOPS object is using
        different numbers for these values.  By default (-0.28932319,0.5166312,
        0.37458883,0.99913668,-0.04193182,1.03616917,0.27327129,0.99839024,
        0.09701198,1.12969387,0.23718642,0.99865705,0.35968991,0.6649304,
        0.4123743,0.99893657,5.70852885,6.3707943,6.47354071,6.26385429,
        6.19229367,6.22946713)
    descriptor_sds : Tuple[float, ...], optional
        Due to the need to apply scaling to FEPOPS, the DUDE diversity set has
        been profiled and the means collected for all contained FEPOPS. This
        this allows centering and scaling of FEPOPS before scoring. This field
        contains default values for FEPOP standard deviations calculated with
        num_fepops_per_mol = 7, num_centroids_per_fepop=4, and kmeans_method =
        'sklearn'. New values should be supplied if the FEPOPS object is using
        different numbers for these values.  By default (0.35067291,1.00802116,
        0.48380817,0.02926675,0.15400475,0.86220776,0.44542581,0.03999429,
        0.16085455,0.92042695,0.42515847,0.03655217,0.35778578,1.36108994,
        0.49210665,0.03252466,1.96446927,2.30792259,2.5024708,2.4155645,
        2.29434487,2.31437527)

    Raises
    ------
    ValueError
        Invalid kmeans method

    """

    def __init__(
        self,
        *,
        kmeans_method: Literal['sklearn', 'pytorchcpu', 'pytorchgpu'] = 'sklearn',
        max_tautomers: Optional[int] = None,
        num_fepops_per_mol: int = 7,
        num_centroids_per_fepop: int = 4,
        descriptor_means: Tuple[float, ...] = (
            -0.28971602,
            0.5181022,
            0.37487135,
            0.99922747,
            -0.04187301,
            1.03382471,
            0.27407036,
            0.99853436,
            0.09725517,
            1.12824307,
            0.23735556,
            0.99882914,
            0.35977538,
            0.66653514,
            0.41238282,
            0.99902545,
            5.71261449,
            6.37716992,
            6.47293777,
            6.26134733,
            6.20354385,
            6.23201498,
        ),
        descriptor_stds: Tuple[float, ...] = (
            0.35110473,
            1.00839329,
            0.4838859,
            0.02769204,
            0.15418035,
            0.86446056,
            0.44583626,
            0.0381767,
            0.16095862,
            0.92079483,
            0.42526185,
            0.03413741,
            0.35756229,
            1.36093993,
            0.4921059,
            0.0311619,
            1.9668792,
            2.31266486,
            2.50699385,
            2.41269982,
            2.30018205,
            2.31527129,
        ),
    ):
        """OpenFEPOPS (Feature Points) molecular similarity object

        Fepops allows the comparison of molecules using feature points, see the
        original publication for more information:
        https://doi.org/10.1021/jm049654z. In short, featurepoints reduce the number
        of points used to represent a molecule by combining atoms and their
        properties. Typically used to compare libraries of small molecules against
        known actives in the hope of discovering biosimilars based on queries.

        Parameters
        ----------
        kmeans_method : str, optional
            String literal denoting the method which should be used for kmeans
            calculations. May be one of "sklearn", "pytorchgpu", or "pytorchcpu". If
            "sklearn" is passed then Scikit-learn's kmeans implementation is used.
            However a faster implementation from the fast_pytorch_kmeans package can
            also be used if Pytorch is available and may be run in cpu-only mode, or
            GPU accelerated mode. Note: GPU accelerated mode should only be used if
            you are stretching the capabilities in terms of feature points for large
            molecules.  Small molecules will not benefit at all from GPU
            acceleration due to overheads.  By default "sklearn"
        max_tautomers : Optional[int], optional
            Maximum number of tautomers which should be generated. Internally, this
            implementation of FEPOPS relies upon RDKit's TautomerEnumerator to
            generate tautomers and may optionally pass in a limit to the number of
            Tautomers to generate. Unless the molecules (or macromolecules) you are
            working with generate massive numbers of tautomers, this should be None
            implying that no limit should be placed on tautomer generation. By
            default None
        num_fepops_per_mol : int, optional
            Number of feature points to use in the representation of a molecule.
            Literature notes that 7 has been empirically found to be a good number
            of feature points for performant representations of small molecules.
            This might be increased if you are dealing with large and very flexible
            molecules, by default 7
        num_centroids_per_fepop : int, optional
            Each fepop is represented by a number of centres, into which atom
            properties are compressed. Literature notes that this has been
            empirically determined to be 4 for a performant representation of small
            molecules. By default 4
        descriptor_means : Tuple[float, ...], optional
            Due to the need to apply scaling to FEPOPS, the DUDE diversity set has
            been profiled and the means collected for all contained FEPOPS. This
            this allows centering and scaling of FEPOPS before scoring. This field
            contains default values for FEPOP means calculated with
            num_fepops_per_mol = 7, num_centroids_per_fepop=4, and kmeans_method =
            'sklearn'. New values should be supplied if the FEPOPS object is using
            different numbers for these values.  By default (-0.28932319,0.5166312,
            0.37458883,0.99913668,-0.04193182,1.03616917,0.27327129,0.99839024,
            0.09701198,1.12969387,0.23718642,0.99865705,0.35968991,0.6649304,
            0.4123743,0.99893657,5.70852885,6.3707943,6.47354071,6.26385429,
            6.19229367,6.22946713)
        descriptor_stds : Tuple[float, ...], optional
            Due to the need to apply scaling to FEPOPS, the DUDE diversity set has
            been profiled and the means collected for all contained FEPOPS. This
            this allows centering and scaling of FEPOPS before scoring. This field
            contains default values for FEPOP standard deviations calculated with
            num_fepops_per_mol = 7, num_centroids_per_fepop=4, and kmeans_method =
            'sklearn'. New values should be supplied if the FEPOPS object is using
            different numbers for these values.  By default (0.35067291,1.00802116,
            0.48380817,0.02926675,0.15400475,0.86220776,0.44542581,0.03999429,
            0.16085455,0.92042695,0.42515847,0.03655217,0.35778578,1.36108994,
            0.49210665,0.03252466,1.96446927,2.30792259,2.5024708,2.4155645,
            2.29434487,2.31437527)
        Raises
        ------
        ValueError
            Invalid kmeans method
        """
        # Descriptor stds may contain zeros. If they do, then we mimic Scikit-Learn's
        # StandardScaler, whereby if unit variance is not achievable, no scaling is
        # applied (value of 1.0)
        self.descriptor_stds_no_zeros = np.array(descriptor_stds)
        self.descriptor_stds_no_zeros[self.descriptor_stds_no_zeros == 0.0] = 1.0
        self.descriptor_means = np.array(descriptor_means)

        try:
            self.kmeans_func = getattr(self, f"_perform_kmeans_{kmeans_method}")
        except:
            raise ValueError(
                f"Supplied kmeans_method argument ({kmeans_method}) does not match a callable method of the form (_perfom_kmeans_{kmeans_method}). Implemented methods seem to be: {[m.split('_')[3] for m in OpenFEPOPS.__dict__.keys() if m.startswith('_perform_kmeans_')]}"
            )

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

        self.tautomer_enumerator = MolStandardize.tautomer.TautomerEnumerator(
            **{"max_tautomers": max_tautomers} if max_tautomers is not None else {}
        )

    def _get_k_medoids(
        self, input_x: np.ndarray, k: int = 7, random_state: int = 42
    ) -> np.ndarray:
        """Select k FEPOPS from conformers and tautomers

        Gets k mediods from conformers (and tautomers) which are representative
        of the molecule as a function of conformer and tautomer states by virtue
        of chosen FEPOPS being diverse.

        Parameters
        ----------
        input_x : np.ndarray
            The pharmacophore features of all conformers.
        k : int
            The number of medoids for clustering. By default 7.
        random_state : int
            Integer to use as a random state when seeding the random number
            generator.  By default 42.

        Returns
        -------
        np.ndarray
            The final Fepops descriptors comprised of k representative
            conformers/tautomers.
        """
        input_x = np.unique(input_x, axis=0)

        if input_x.shape[0] <= k:
            return input_x

        # Apply standard scaling to FEPOP features. Behaviour when std dev is 0 mimics
        # Scikit-Learn's StandardScaler, whereby if unit variance is not achievable, no
        # scaling is applied (value of 1.0)
        input_x_std = np.std(input_x, axis=0)
        input_x_std[input_x_std == 0.0] = 1.0
        X = (input_x - np.mean(input_x, axis=0)) / input_x_std
        point_to_centroid_map = np.ones(X.shape[0])
        point_to_centroid_map_prev = np.zeros_like(point_to_centroid_map)

        np_rng = np.random.default_rng(seed=random_state)
        medoids = X[np_rng.choice(np.arange(X.shape[0]), size=k, replace=False), :]

        while (point_to_centroid_map != point_to_centroid_map_prev).any():
            point_to_centroid_map_prev = point_to_centroid_map
            point_to_centroid_map = np.argmin(np.square(cdist(X, medoids)), axis=1)
            for i in range(k):
                medoid_members = X[point_to_centroid_map == i]
                if len(medoid_members) == 0:
                    chosen_x_point = np_rng.choice(np.arange(X.shape[0]))
                    medoids[i] = X[chosen_x_point, :]
                    point_to_centroid_map[chosen_x_point] = i
                medoids[i] = np.median(X[point_to_centroid_map == i], axis=0)
        # Sorting at this stage for reproducibility with existing pregenerated
        # descriptor sets and convention with early FEPOPS versions which relied
        # upon FEPOPS within a molecule being sorted by charge (before moving to
        # the newer CombiAlign scoring algorithm)
        return input_x[np.lexsort(medoids.T[::-1])]

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
        Chem.rdPartialCharges.ComputeGasteigerCharges(mol, throwOnParamFailure=True)
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

    def _perform_kmeans_sklearn(
        self,
        atom_coords: np.ndarray,
        num_centroids: int = 4,
        seed: int = 42,
    ) -> tuple:
        """Perform kmeans calculation (sklearn method)

        Parameters
        ----------
        atom_coords : ndarray
            A Numpy array containing the 3D coordinates of a molecule.
        num_centroids : int
            The number of centoids used for clustering. By default 4.
        seed : int
            Seed for sklearn kmeans initialisation. By default 42.

        Returns
        -------
        tuple
            A tuple containing the centroid coordinates and the cluster labels
            of molecular atoms.
        """
        kmeans = _SKLearnKMeans(
            n_clusters=num_centroids,
            random_state=seed,
            n_init="auto",
        ).fit(atom_coords)
        centroid_coords = kmeans.cluster_centers_
        instance_cluster_labels = kmeans.labels_
        return centroid_coords, instance_cluster_labels

    def _perform_kmeans_pytorchcpu(
        self,
        atom_coords: np.ndarray,
        num_centroids: int = 4,
        seed: int = 42,
    ) -> tuple:
        """Perform kmeans calculation using pytorch (CPU only)

        Parameters
        ----------
        atom_coords : ndarray
            A Numpy array containing the 3D coordinates of a molecule.
        num_centroids : int
            The number of centoids used for clustering. By default 4.
        seed : int
            Seed for sklearn kmeans initialisation. By default 42.

        Returns
        -------
        tuple
            A tuple containing the centroid coordinates and the cluster labels
            of molecular atoms.
        """
        torch.manual_seed(seed)
        mol_coors_torch = torch.from_numpy(atom_coords)
        kmeans = _FastPTKMeans(n_clusters=num_centroids, max_iter=300)
        instance_cluster_labels = kmeans.fit_predict(
            mol_coors_torch,
            centroids=torch.tensor(
                atom_coords[:num_centroids], device=mol_coors_torch.device
            ),
        ).numpy()
        centroid_coords = kmeans.centroids.numpy()
        return centroid_coords, instance_cluster_labels

    def _perform_kmeans_pytorchgpu(
        self,
        atom_coords: np.ndarray,
        num_centroids: int = 4,
        seed: int = 42,
    ) -> tuple:
        """Perform kmeans calculation using pytorch (gpu accelerated)

        Parameters
        ----------
        atom_coords : ndarray
            A Numpy array containing the 3D coordinates of a molecule.
        num_centroids : int
            The number of centoids used for clustering. By default 4.
        seed : int
            Seed for sklearn kmeans initialisation. By default 42.

        Returns
        -------
        tuple
            A tuple containing the centroid coordinates and the cluster labels
            of molecular atoms.
        """
        torch.manual_seed(seed)
        mol_coors_torch = torch.from_numpy(atom_coords).to("cuda")
        kmeans = _FastPTKMeans(n_clusters=num_centroids, max_iter=300)
        instance_cluster_labels = kmeans.fit_predict(
            mol_coors_torch,
            centroids=torch.tensor(
                atom_coords[:num_centroids], device=mol_coors_torch.device
            ),
        ).numpy()
        centroid_coords = kmeans.centroids.numpy()
        return centroid_coords, instance_cluster_labels

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
        distances = np.hstack(
            [dmat[0, -1]]
            + [np.diagonal(dmat, offset=k) for k in range(1, dmat.shape[0] - 1)]
        )
        return distances

    def _mol_from_smiles(self, smiles_string: str) -> Chem.rdchem.Mol:
        """Parse smiles to mol, catching errors

        This SMILES->RDKit mol converter is used throughout OpenFEPOPS and as
        such, any read in/parsing of a SMILES stirng should use this method.

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

        A private method used to generate a set of bond angle multipliers (0 to
        3) using all rotatable bonds within a molecule. Up to 1024 conformers
        are sampled for a molecule if n_rot is greater than five, otherwise all
        n_rot^4 are returned.

        Parameters
        ----------
        n_nor : int
            The number of rotatable bonds in a molecule.
        seed : int
            Seed for random sampling of rotamer space. Typically the hash of
            molecule coords.
        Returns
        -------
        List
            A list containing the sampled bond states (rotation angles) for all
            of the rotatable bonds of a molecule.
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

    def generate_conformers(self, mol: Chem.rdchem.Mol, random_state: int = 42) -> list:
        """Generate conformers with rotatable bonds

        Generate conformers for a molecule, enumerating rotatable bonds over 90
        degree angles. This 90 degree increment was deemed opimal in literature.

        Parameters
        ----------
        mol : Chem.rdchem.Mol
            The Rdkit mol object of the input molecule.
        random_state : int
            Integer to use as a random state when seeding the random number
            generator.  By default 42.


        Returns
        -------
        List
            A list containing mol objects of different conformers with different
            angles of rotetable bonds

        """

        try:
            mol = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            params.useSmallRingTorsions = True
            params.randomSeed = random_state
            original_conformer = mol.GetConformer(AllChem.EmbedMolecule(mol, params))
        except ValueError:
            params = AllChem.ETKDGv2()
            id = AllChem.EmbedMolecule(mol, params)
            if id == -1:
                logging.warning(
                    "Coords could not be generated without using random coords. using random coords now"
                )
                params.useRandomCoords = True
            try:
                original_conformer = mol.GetConformer(
                    AllChem.EmbedMolecule(mol, params)
                )
            except ValueError:
                logging.warning("Conformer embedding failed")
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

        Change conformers by rotating the assigned rotatable bond based on a set
        of dihedral angles defined by four flanking atoms.

        Parameters
        ----------
        conformer : Chem.rdchem.Conformer
            The Rdkit conformer object.
        dihedrals : tuple
            A tuple containing all identified dihedrals with the index of their
            four defined atoms.
        starting_angles : tuple
            A tuple containing the orignal states (dihedral angles) of all the
            rotatable bond before rotating.
        bond_state : tuple
            A tuple containing a specific bond state (a combination of various
            rotation angles) for all rotatable bonds of a molecule.
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
    ) -> np.ndarray:
        """Obtain centroids and their corresponding pharmacophoric features

        Obtain centroids and then calucate and assign their corresponding
        pharmacophoric features (logP, charges, HBA, HBD, and distances
        between the centroids, following the pattern used for calculation of
        matrix determinants - in the case of 4 centroids, this is:
        d1-4, d1-2, d2-3, d3-4, d1-3, d2-4)

        Parameters
        ----------
        mol : Chem.rdchem.Mol
            The Rdkit mol object of the input molecule.

        Returns
        -------
        np.ndarray
            A Numpy array containing 22 pharmacophoric features for all conformers.
        """
        centroid_coords, instance_cluster_labels = self.kmeans_func(
            mol.GetConformer(0).GetPositions(),
            num_centroids=self.num_centroids_per_fepop,
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
        self,
        mol: Union[str, None, Chem.rdchem.Mol],
        is_canonical: bool = False,
    ) -> Tuple[GetFepopStatusCode, Union[np.ndarray, None]]:
        """Get Fepops descriptors for a molecule

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
            logging.error(
                f"Failed to make a molecule{' from '+original_smiles if original_smiles is not None else ''}"
            )
            return GetFepopStatusCode.FAILED_TO_GENERATE, None
        if Lipinski.HeavyAtomCount(mol) < self.num_centroids_per_fepop:
            logging.error(
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
            logging.error(
                f"Failed to generate conformers/tautomers {' for '+original_smiles if original_smiles is not None else ''}"
            )
            return GetFepopStatusCode.FAILED_TO_GENERATE, None

        try:
            pharmacophore_feature_all_confs = np.array(
                [
                    self.get_centroid_pharmacophoric_features(each_mol)
                    for each_mol in each_mol_with_all_confs_list
                ]
            )
        except ValueError as e:
            if original_smiles is not None:
                logging.error(f"Failed molecule had SMILES: {original_smiles}")
            logging.error(e)
            return GetFepopStatusCode.FAILED_TO_GENERATE, None

        medoids = self._get_k_medoids(
            pharmacophore_feature_all_confs, self.num_fepops_per_mol
        )
        return GetFepopStatusCode.SUCCESS, medoids

    def pairwise_correlation(self, A: np.ndarray, B: np.ndarray):
        """Fast method to generate pairwise correlation values (Pearson)

        Parameters
        ----------
        A : np.ndarray
            First features array (1D)
        B : np.ndarray
            Second features array (1D)

        Returns
        -------
        np.ndarray
            2D matrix containing A vs B feature correlations
        """
        if len(A) < len(B):
            A = np.pad(A, (0, len(B) - len(A)), mode='constant', constant_values=0)
        if len(B) < len(A):
            B = np.pad(B, (0, len(A) - len(B)), mode='constant', constant_values=0)
        am = A - np.mean(A, axis=0, keepdims=True)
        bm = B - np.mean(B, axis=0, keepdims=True)
        return (
            am.T
            @ bm
            / (
                np.sqrt(np.sum(am**2, axis=0, keepdims=True)).T
                * np.sqrt(np.sum(bm**2, axis=0, keepdims=True))
            )
        )

    def calc_similarity(
        self,
        query: Union[np.ndarray, str, None],
        candidate: Union[np.ndarray, str, None, list[np.ndarray, str, None]],
    ) -> float:
        """Calculate FEPOPS similarity

        Method for calculating molecular similarity based on their OpenFEPOPS
        descriptors. Centres and scales FEPOPS descriptors using parameters
        passed upon object initialisation.

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
            scores = []
            for c in candidate:
                scores.append(self.calc_similarity(query, c))
            return scores
        if not isinstance(candidate, np.ndarray):
            candidate_status, candidate = self.get_fepops(candidate)
            if candidate_status != GetFepopStatusCode.SUCCESS:
                return np.nan

        if not isinstance(query, np.ndarray):
            raise ValueError("query was not, or could not be coerced into a np.ndarray")
        if not isinstance(candidate, np.ndarray):
            raise ValueError(
                "candidate was not, or could not be coerced into a np.ndarray"
            )

        q = (query - self.descriptor_means) / self.descriptor_stds_no_zeros
        c = (candidate - self.descriptor_means) / self.descriptor_stds_no_zeros
        return self.pairwise_correlation(q.flatten(), c.flatten())

    def __call__(
        self,
        query: Union[np.ndarray, str],
        candidate: Union[np.ndarray, str],
    ) -> float:
        """Calling the object has the same effect as calling calc_similarity

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
        return self.calc_similarity(query, candidate)
