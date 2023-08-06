import numpy as np
import fire
from .fepops import OpenFEPOPS, GetFepopStatusCode
from .fepops_persistent import get_persistent_fepops_storage_object


class FepopsCMDLineInterface:
    def __init__(self, database_file: str = None):
        self.database_file = database_file

    def get_fepops(self, smi: str):
        """Get Fepops descriptors from a SMILES string

        Parameters
        ----------
        smi : str
            Molecule as a SMILES string. Can also be None, in which case a failure error
            status is returned along with None in place of the requested Fepops
            descriptors.

        Returns
        -------
        Tuple[GetFepopStatusCode, Union[np.ndarray, None]]
            Returns a tuple, with the first value being a GetFepopStatusCode
            (enum) denoting SUCCESS or FAILED_TO_GENERATE. The second tuple
            element is either None (if unsuccessful), or a np.ndarray containing
            the calculated Fepops descriptors of the requested input molecule.
        """
        if self.database_file is not None:
            with get_persistent_fepops_storage_object(self.database_file) as f:
                return f.get_fepops(smi)
        else:
            f = OpenFEPOPS()
            return f.get_fepops(smi)

    def calc_sim(self, smi1: str, smi2: str):
        """Calculate FEPOPS similarity between two smiles strings

        Parameters
        ----------
        smi1 : str
            Smiles string representing molecule 1
        smi2 : str
            Smiles string representing molecule 2

        Returns
        -------
        float
            Fepops similarity between the two supplied molecules
        """
        if self.database_file is not None:
            with get_persistent_fepops_storage_object(self.database_file) as f:
                fepops_status1, fepops_features1 = f.get_fepops(smi1)
                fepops_status2, fepops_features2 = f.get_fepops(smi2)
                if (
                    fepops_status1 is not GetFepopStatusCode.SUCCESS
                    or fepops_status2 is not GetFepopStatusCode.SUCCESS
                ):
                    return np.nan
                else:
                    return f.calc_similarity(fepops_features1, fepops_features2)
        else:
            f = OpenFEPOPS()
            print(f.calc_similarity(smi1, smi2))

    def save_descriptors(self, smi):
        """Pregenerate FEPOPS descriptors for a set of SMILES strings"""
        f_persistent = get_persistent_fepops_storage_object(self.database)
        f_persistent.save_descriptors(smi)
        if self.database_file.endswith((".json")):
            f_persistent.write()

    def dude_preprocessor(self, dude_directory: str = "data/dude/"):
        """Access functions for preparation and benchmarking of the DUDE dataset

        The DUDE dataset consists of 102 targets, 22,886 active compounds along with
        decoys. More information is available here:
        https://dude.docking.org/

        """
        from .dude_preprocessor import DudePreprocessor

        print("Dude DIR========", dude_directory)
        dude_preprop = DudePreprocessor(dude_directory=dude_directory)
        return dude_preprop


def fepops_entrypoint():
    fire.Fire(FepopsCMDLineInterface)


if __name__ == '__main__':
    fire.Fire(FepopsCMDLineInterface)
