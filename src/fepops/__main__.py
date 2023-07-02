import numpy as np
import fire
from .fepops import Fepops, GetFepopStatusCode
from .fepops_persistent import get_persistent_fepops_storage_object

class FepopsCMDLineInterface():
    def __init__(self, database:str=None):
        self.database=database
    
    def get_fepops(self, smi):
        if self.database_file is not None:
            with get_persistent_fepops_storage_object(self.database_file) as f:
                status, fepops_features = f.get_fepops(smi)
        else:
            f = Fepops()
            status, fepops_features = f.get_fepops(smi)
        if status == GetFepopStatusCode.SUCCESS:
            return(fepops_features)
        else:
            return f"Failed to generate FEPOPS descriptors for {smi}"
    def calc_sim(self, smi1, smi2):
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
            f = Fepops()
            print(f.calc_similarity(smi1, smi2))

    def save_descriptors(self, smi):
        """Pregenerate FEPOPS descriptors for a set of SMILES strings"""
        f_persistent = get_persistent_fepops_storage_object(self.database)
        f_persistent.save_descriptors(smi)
        if self.database_file.endswith((".json")):
            f_persistent.write()

def fepops_entrypoint():
    fire.Fire(FepopsCMDLineInterface)

if __name__ == '__main__':
  fire.Fire(FepopsCMDLineInterface)