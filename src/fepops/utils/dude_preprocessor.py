from typing import Union, Optional
from pathlib import Path
from tqdm import tqdm
from fepops import OpenFEPOPS
import pandas as pd
import multiprocessing as mp
from rdkit import Chem
import logging


class DudePreprocessor:
    def __init__(
        self,
        *,
        dude_directory: Union[Path, str] = "data/dude/",
    ):
        """Perform preprocessing on the DUDE database.

        Assumes that the DUDE database has been downloaded and placed in the path
        "data/dude/unprocessed". Here, we expect directories for each target containing
        actives_final.ism and decoys_final.ism. These files are then processed into CSVs
        representing the molecules (actives and decoys) for each DUDE target.

        Parameters
        ----------
        dude_directory : Union[Path, str], optional
            Directory containg , by default "data/dude/"

        """
        print(dude_directory)
        self.dude_path = Path(dude_directory)
        self.dude_unprocessed_path = self.dude_path / Path("unprocessed")
        if not self.dude_path.exists():
            raise FileNotFoundError(f"Dude dataset not found in path: {self.dude_path}")
        self.dude_processed_path = self.dude_path / Path("processed")
        self.dude_processed_path.parent.mkdir(parents=True, exist_ok=True)
        self.fepops_ob = OpenFEPOPS()

    def __call__(
        self,
        targets: Optional[Union[str, list[str]]] = None,
        skip_existing: bool = True,
    ):
        """Perform processing

        Parameters
        ----------
        targets : Optional[Union[str, list[str]]], optional
            Optionally provide a list of target names to process. If None, then
            all targets are processed, by default None
        skip_existing : bool, optional
            If True, then existing processed target files are not regenerated, by
            default True
        """
        self.process(targets=targets, skip_existing=skip_existing)

    def process(
        self,
        targets: Optional[list[str]] = None,
        skip_existing: bool = True,
    ):
        """Perform processing

        Parameters
        ----------
        targets : Optional[Union[str, list[str]]], optional
            Optionally provide a list of target names to process. If None, then
            all targets are processed, by default None
        skip_existing : bool, optional
            If True, then existing processed target files are not regenerated, by
            default True
        """
        if targets is None:
            dude_targets = [
                t.parent.name
                for t in self.dude_path.glob("unprocessed/*/actives_final.ism")
            ]
        else:
            if isinstance(targets, str):
                targets = [targets]
        print(f"Processing the following DUDE targets: {dude_targets}")
        for target in tqdm(dude_targets, desc=f"Preparing targets"):
            self._create_dude_target_csv_data(target, skip_existing=skip_existing)

    @staticmethod
    def _parallel_init_worker_desc_gen_shared_fepops_ob():
        global shared_fepops_ob
        shared_fepops_ob = OpenFEPOPS()

    @staticmethod
    def _parallel_get_rdkit_cansmi(s):
        global shared_fepops_ob
        mol = shared_fepops_ob._mol_from_smiles(s)
        if mol is None:
            return ""
        return Chem.MolToSmiles(mol)

    def _create_dude_target_csv_data(
        self,
        dude_target: Path,
        actives_file: Path = Path("actives_final.ism"),
        decoys_file: Path = Path("decoys_final.ism"),
        seperator: str = " ",
        skip_existing: bool = True,
    ):
        target_output_file = self.dude_processed_path / f"dude_target_{dude_target}.csv"
        if skip_existing and target_output_file.exists():
            logging.warning(
                f"Found existing {target_output_file}, skipping due to skip_existing = True, rerun as False to regenerate"
            )
        actives = pd.read_csv(
            self.dude_unprocessed_path / Path(dude_target) / actives_file,
            sep=seperator,
            header=None,
            names=["SMILES", "DUDEID", "CHEMBLID"],
        )
        actives["Active"] = 1
        decoys = pd.read_csv(
            self.dude_unprocessed_path / Path(dude_target) / decoys_file,
            sep=seperator,
            header=None,
            names=["SMILES", "DUDEID"],
        )
        decoys["Active"] = 0
        df = pd.concat([actives, decoys]).reset_index().drop(columns="index")
        df["rdkit_canonical_smiles"] = tqdm(
            mp.Pool(
                initializer=self._parallel_init_worker_desc_gen_shared_fepops_ob
            ).imap(self._parallel_get_rdkit_cansmi, df.SMILES, chunksize=100),
            desc=f"Generating {dude_target} benchmark file",
            total=len(df),
        )
        df.to_csv(target_output_file, index=False)

    def cache_mols_from_csv(
        self,
        csv_path: Union[Path, str],
        rdkit_canonical_smiles_column_header: str = "rdkit_canonical_smiles",
    ):
        """Cache mols from a CSV into a db for faster recall later

        Parameters
        ----------
        csv_path : Union[Path, str]
            Path of CSV file. If None, then all CSV files in the DUDE datasets
            processed path are used.
        rdkit_canonical_smiles_column_header : str, optional
            Column header containing RDKit canonical SMILES, by default
            "rdkit_canonical_smiles"
        """
        from ...fepops.fepops_persistent import get_persistent_fepops_storage_object

        for csv_path in (
            [Path(csv_path)]
            if csv_path is not None
            else self.dude_processed_path.glob("dude_target_*.csv")
        ):
            df = pd.read_csv(csv_path)
            if df[rdkit_canonical_smiles_column_header].isnull().values.any():
                logging.warning(
                    f"Whilst working on caching {csv_path}, the following mol rows did not contain RDKit canonical SMILES:"
                )
                print(df[df[rdkit_canonical_smiles_column_header].isnull()])
            smiles = [
                s
                for s in df[rdkit_canonical_smiles_column_header].tolist()
                if not pd.isnull(s)
            ]
            with get_persistent_fepops_storage_object(csv_path.with_suffix(".db")) as f:
                f.save_descriptors(smiles)
