import numpy as np
import fire
import json
from pathlib import Path
from typing import Optional
from .fepops import OpenFEPOPS, GetFepopStatusCode
from .fepops_persistent import get_persistent_fepops_storage_object


class FepopsCMDLineInterface:
    """Object to organise the command line interface to FEPOPS

    Used as an entrypoint to allow use of simply 'fepops' on the command line
    and provides access to the following sub commands:
    - get_fepops
    - calc_sim
    - save_descriptors
    - dude_preprocessor

    """

    def __init__(self, database_file: str = None, json_file: Optional[str] = None):
        """Constructor for the command line inteface object

        Allows singular definition of a database file and subsequent use in
        all subcommands and setting of JSON output file if required.

        Parameters
        ----------
        database_file : str, optional
            If a string, then this is used as the database/cache file used by
            all subprocesses. If None, then no database of cache file is used,
            by default None.
        json_file : str, optional
            If a string, then this is used as the JSON output file for results
            from fepops generation and similarity calculations. If None, then
            results are printed to the terminal, By default None
        """
        self.database_file = database_file
        self.json_file = None if json_file is None else Path(json_file)
        if self.json_file is not None and not self.json_file.parent.resolve().exists():
            self.json_file.parent.resolve().mkdir(parents=True)

    def get_fepops(self, smi: str):
        """Get Fepops descriptors from a SMILES string

        Parameters
        ----------
        smi : str
            Molecule as a SMILES string. Can also be None, in which case a
            failure status code is returned along with None in place of the
            requested Fepops descriptors.

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
                fepops_result = f.get_fepops(smi)
        else:
            f = OpenFEPOPS()
            fepops_result = f.get_fepops(smi)
        if self.json_file is None:
            return fepops_result
        else:
            fepops_status, fepops_features = fepops_result
            with open(Path(self.json_file), 'w') as json_output:
                json.dump(
                    {
                    "SMILES": smi,
                    "FepopStatusCode": str(fepops_status.name),
                    "Fepops": fepops_features.tolist()
                    }, json_output, indent=4
                )

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
                    similarity_score = np.nan
                else:
                    similarity_score = f.calc_similarity(fepops_features1, fepops_features2)
        else:
            f = OpenFEPOPS()
            similarity_score = f.calc_similarity(smi1, smi2)
            
        if self.json_file is not None:
            with open(Path(self.json_file), 'w') as json_output:
                json.dump(
                    {
                    "SMI1": smi1,
                    "SMI2": smi2,
                    "FEPOPS Similarity Score": similarity_score if len(similarity_score)>1 else similarity_score[0]
                    }, json_output, indent=4
                )
        else:
            return similarity_score if len(similarity_score)>1 else similarity_score[0]

    def save_descriptors(self, smi: str):
        """Pregenerate FEPOPS descriptors for a set of SMILES strings

        Parameters
        ----------
        smi :
            String containing the path to a SMILES file which should be read in
            and have each molecule with in added to the database
        """
        f_persistent = get_persistent_fepops_storage_object(self.database_file)
        f_persistent.save_descriptors(smi)
        if self.database_file.endswith((".json")):
            f_persistent.write()

    def dude_preprocessor(self, dude_directory: str = "data/dude/"):
        """Access a dude preprocessor object for preprocessing and preparation
        of the DUDE benchmarking dataset

        The DUDE dataset consists of 102 targets, 22,886 active compounds along with
        decoys, bringing the total number of molecules to 1,434,022.
        More information is available here: https://dude.docking.org/
        """
        from .utils import DudePreprocessor

        dude_preprop = DudePreprocessor(dude_directory=dude_directory)
        return dude_preprop


def fepops_entrypoint():
    """Entrypoint for the fepops module"""
    fire.Fire(FepopsCMDLineInterface)


if __name__ == '__main__':
    """Entrypoint if run directly with the Python interpreter"""
    fire.Fire(FepopsCMDLineInterface)
