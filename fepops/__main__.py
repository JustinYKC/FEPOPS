import argparse
from .fepops import Fepops
from .fepops_persistent import FepopsDBSqlite, FepopsDBJSON
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the FEPOPS descriptors")
    subparsers = parser.add_subparsers(
        dest = "subcmd", help="subcommands", metavar="SUBCOMMAND"
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
        "save_descriptors", help="Pregenerate FEPOPS descriptors for a set of SMILES strings"
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

    def get_persistent_fepops_storage_object(database_file:str):
        if args.database_file.endswith((".sqlite", ".sqlite3", ".db", ".db3", ".s3db", ".sl3")):
            return FepopsDBSqlite(args.database_file)
        if args.database_file.endswith((".json")):
            return FepopsDBJSON(args.database_file)

    args = parser.parse_args()
    if args.subcmd == "get_fepops":
        if args.database_file is not None:
            f_persistent = get_persistent_fepops_storage_object(args.database_file)
            fepops_features = f_persistent.get_fepop(args.insmile)
        else:
            f=Fepops()
            fepops_features = f.get_fepops(args.insmile)
        print(fepops_features)

    if args.subcmd == "calc_sim":
        if args.database_file is not None:
            with get_persistent_fepops_storage_object(args.database_file) as f_persistent:
                fepops_features1 = f_persistent.get_fepop(args.insmiles1)
                fepops_features2 = f_persistent.get_fepop(args.insmiles2)
                print(f_persistent.calc_similarity(fepops_features1, fepops_features2))
        else:
            f=Fepops()
            fepops_features_1 = f.get_fepops(args.insmiles1)
            fepops_features_2 = f.get_fepops(args.insmiles2)
            print(f.calc_similarity(fepops_features_1, fepops_features_2))

    if args.subcmd == "save_descriptors":
        f_persistent = get_persistent_fepops_storage_object(args.database_file)
        f_persistent.save_descriptors(args.smiles_file)
      