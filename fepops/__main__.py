import numpy as np
import argparse
from .fepops import Fepops, GetFepopStatusCode
from .fepops_persistent import get_persistent_fepops_storage_object

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
    f1_parser.add_argument(
        "-db",
        "--database_file",
        help="sqlite3 database file to use as a descriptor cache",
        dest="database_file",
        required=False,
    )

    f2_parser = subparsers.add_parser(
        "calc_sim", help="Calculate FEPOPS similarity between two molecules"
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
    f2_parser.add_argument(
        "-db",
        "--database_file",
        help="sqlite3 database file to use as a descriptor cache",
        dest="database_file",
        required=False,
    )

    f3_parser = subparsers.add_parser(
        "save_descriptors",
        help="Pregenerate FEPOPS descriptors for a set of SMILES strings",
    )
    f3_parser.add_argument(
        "-ismi",
        "--input_smiles",
        help="Input file containing SMILES strings",
        dest="smiles_file",
        required=True,
    )
    f3_parser.add_argument(
        "-db",
        "--database_file",
        help="sqlite3 database file to use as a descriptor cache",
        dest="database_file",
        required=True,
    )

    args = parser.parse_args()
    if args.subcmd == "get_fepops":
        if args.database_file is not None:
            with get_persistent_fepops_storage_object(args.database_file) as f:
                status, fepops_features = f.get_fepops(args.insmile)
        else:
            f = Fepops()
            status, fepops_features = f.get_fepops(args.insmile)
        if status == GetFepopStatusCode.SUCCESS:
            print(fepops_features)
        else:
            print(f"Failed to generate FEPOPS descriptors for {args.insmile}")

    if args.subcmd == "calc_sim":
        if args.database_file is not None:
            with get_persistent_fepops_storage_object(args.database_file) as f:
                fepops_status1, fepops_features1 = f.get_fepops(args.insmiles1)
                fepops_status2, fepops_features2 = f.get_fepops(args.insmiles2)
                if (
                    fepops_status1 is not GetFepopStatusCode.SUCCESS
                    or fepops_status2 is not GetFepopStatusCode.SUCCESS
                ):
                    print(np.nan)
                else:
                    print(f.calc_similarity(fepops_features1, fepops_features2))
        else:
            f = Fepops()
            print(f.calc_similarity(args.insmiles1, args.insmiles2))

    if args.subcmd == "save_descriptors":
        f_persistent = get_persistent_fepops_storage_object(args.database_file)
        f_persistent.save_descriptors(args.smiles_file)
        if args.database_file.endswith((".json")):
            f_persistent.write()
