#!/usr/bin/env python3

"""
ABCC Audit Main Script 
Greg Conan: gregmconan@gmail.com
Created 2022-07-06
Updated 2024-02-26
"""
# Standard imports
import argparse
from datetime import datetime
import os
import pdb
from typing import Any, Dict, Set

# Local imports
from src.BidsDB import AllBidsDBs
from src.utilities import (
    dt_format, get_most_recent_FTQC_fpath, make_logger, PATH_NGDR,
    s3_get_info, UserAWSCreds, valid_output_dir, valid_readable_dir,
    valid_readable_file, WHICH_DBS
)

def main():
    cli_args = _cli()
    cli_args["to_save"] = pick_DBs_to_save(cli_args)

    make_logger_from_cli_args(cli_args)

    # Make all BIDS DBs: tier1 (NGDR), tier 2 (s3), and fast-track QC DB
    client = make_s3_client_from_cli_args(cli_args)
    all_DBs = AllBidsDBs(cli_args, client=client,
                         sub_col="subject", ses_col="session")
    for key in cli_args["audit"]:
        all_DBs.add_DB(key)

    if cli_args["debugging"]:
        pdb.set_trace()

    if len(cli_args["audit"]) == 3:
        all_DBs.save_all_summary_tsv_files()


def pick_DBs_to_save(cli_args: Dict[str, Any]) -> Set[str]:
    to_save = set()
    dont_save = set(cli_args["dont_save"])
    for which_DB in cli_args["audit"]:
        if not (which_DB in dont_save or cli_args.get(f"{which_DB}_DB_file")):
            to_save.add(which_DB)
    return to_save


def make_s3_client_from_cli_args(cli_args):
    # Get AWS credentials to access s3 buckets
    if not cli_args["aws_keys"]:
        my_s3info = s3_get_info()
        if my_s3info:
            cli_args["aws_keys"] = [my_s3info["access key"],
                                    my_s3info["secret key"]]
    return UserAWSCreds(cli_args, argparse.ArgumentParser()).get_s3_client()


def make_logger_from_cli_args(cli_args):
    # Get logger, and prepare it to log to a file if the user said to
    log_to = dict()
    if cli_args.get("log"):
        if len(os.path.split(cli_args["log"])) > 1:
            os.makedirs(os.path.dirname(cli_args["log"]), exist_ok=True)
        else:
            cli_args["log"] = os.path.join(cli_args["output"], cli_args["log"])
        log_to = dict(out=cli_args["log"], err=cli_args["log"])
    return make_logger(cli_args["verbosity"], **log_to)


def _cli() -> Dict[str, Any]: 
    """ 
    :return: Dict with all command-line arguments from user
    """
    # audits = [audit.split("abcd-")[-1] for audit in audit_names]
    parser = argparse.ArgumentParser()
    dt_str = dt_format(datetime.now())
    DEFAULT_BUCKETS = ("ABCC_year1", "ABCC_year2", "ABCC_year4")
    DEFAULT_DTYPES = ("anat", "func", "fmap") #, dwi
    DEFAULT_FILE_EXT = ".json"
    DEFAULT_HOST = "https://s3.msi.umn.edu"
    DEFAULT_FTQC = get_most_recent_FTQC_fpath(
        "/home/rando149/shared/code/internal/utilities/"
        "abcd-dicom2bids/spreadsheets/fastqc{}/abcd_fastqc01.txt"
    )
    DEFAULT_OUT_DIR = os.path.join(os.getcwd(), "audit-outputs", dt_str)
    MSG_DB = ("Valid path to .tsv file saved by this script "
              "containing one row per subject session, and either {}s or "
              "blanks in every column except the first (subject ID) and "
              "second (session ID) columns. If this does not already exist "
              "at the given file path, then the script will save one there.")
    MSG_DEFAULT = " By default, the value for this argument will be `{}`"
    MSG_VALID_PATH = "Valid path to existing local"
    PROG_DESCRIP = ("Check tier 1 and 2 locations to find certain "
                    "files for every subject and session of some ABCD "
                    "BIDS data, then save the files' paths. Plus more")  # TODO
    parser = argparse.ArgumentParser(description=PROG_DESCRIP)
    parser.add_argument(
        "-a", "-audit", "--audit", nargs="+",
        choices=WHICH_DBS,
        default=WHICH_DBS
    )
    parser.add_argument(
        "-bids", "--bids-dir", "--tier-1-dir", "--local-dir", "--dirpath",
        dest="tier1_dir",
        default=PATH_NGDR,
        type=valid_readable_dir,
        help=(f"{MSG_VALID_PATH} directory structured in BIDS input data "
              f"format.{MSG_DEFAULT.format(PATH_NGDR)}")
    )
    parser.add_argument(
        "-b", "--buckets", "--s3-buckets", "--s3-bucket-names", 
        nargs='+',
        dest="buckets",
        default=DEFAULT_BUCKETS,
        help=("List of bucket names containing ABCD BIDS data."
              + MSG_DEFAULT.format(DEFAULT_BUCKETS))
    )
    parser.add_argument(
        "-d", "--debug", "--debugging",
        dest="debugging",
        action="store_true",
        help=("Include this flag to interactively debug errors and exceptions.")
    )
    parser.add_argument(
        "--exclude", "--dont-save", "--no-save", "--dont-keep-DBs",
        # "-save", "--save", "--save-DBs", "-keep", "--keep", "--keep-DBs",
        dest="dont_save",
        default=list(),
        choices=WHICH_DBS,
        nargs="+",
        help=("Which BIDS DBs you do not want to save to files. By default, "
              "all BIDS DBs created will be saved. BIDS DBs created "
              "by previous runs of this script will not be saved.")
    )
    parser.add_argument(
        "-e", "-ext", "--extension", "--file-ext", "--file-extension",
        dest="file_ext",
        default=DEFAULT_FILE_EXT,
        help=("Valid file extension of all of the files you want to check "
              "on tiers 1 and 2." + MSG_DEFAULT.format(DEFAULT_FILE_EXT))
    )
    parser.add_argument(  # Only required if running DICOM DB Flow
        "-ftqc", "--fasttrack-qc",
        dest="fpath_QC_CSV",
        default=DEFAULT_FTQC,
        type=valid_readable_file, 
        help=(f"{MSG_VALID_PATH} abcd_fastqc01.txt file from the NDA."
              f"{MSG_DEFAULT.format(DEFAULT_FTQC)}")
    )
    parser.add_argument(
        "-ftqc-DB", "-ftqc-db", "--fast-track-DB", "--ftqc-DB-file",
        dest="ftqc_DB_file",
        # type=valid_readable_file,
        help=MSG_DB.format("'ftq_usable' score (1.0 or 0.0)")
    )
    parser.add_argument(
        "-host", "--host", "--s3-host", "--host-URL", "--s3-URL",
        dest="host",
        default=DEFAULT_HOST, 
        help=("URL path to AWS s3 host." + MSG_DEFAULT.format(DEFAULT_HOST))
    )
    parser.add_argument(
        "-k", "-keys", "-creds", "--aws-keys", "--aws-creds",
        dest="aws_keys",
        nargs="*",
        default=list(),
        help=("AWS s3 access key and secret key. Optionally, you may choose "
              "to give this flag two arguments, your AWS access key followed "
              "by your AWS secret key. Otherwise, by default this script will "
              "try to infer both keys by running the 's3info' command, and if "
              "that fails then you will be prompted to manually enter both "
              "keys (hidden from the terminal). Enter the word 'manual' to "
              "skip the s3info check and enter keys manually.")
    )
    parser.add_argument(
        "-l", "-log", "-log-to", "--log-file", "--log-file-name",
        dest="log",
        help=("Name of text file to write output logs to. If the file name "
              "is given alone, then the log file will be written in the "
              "--output directory. You can also give a file path to a "
              "nonexistent file.")
    )
    parser.add_argument(
        "-o", "-out", "--output", "--out-dir", "--output-dir",
        dest="output",
        default=DEFAULT_OUT_DIR,
        type=valid_output_dir,
        help=(f"{MSG_VALID_PATH} directory to save output files into. If no "
              "directory exists at this path yet, then one will be created."
              f"{MSG_DEFAULT.format(DEFAULT_OUT_DIR)}")
    )
    parser.add_argument(
        "-s3-db", "-s3-DB", "-s3db", "--s3-DB-file",
        dest="s3_DB_file",
        # type=valid_readable_file,
        help=MSG_DB.format("AWS s3 path within a given bucket")
    )
    parser.add_argument(
        "-tier1-DB", "-tier1-db", "--tier1-DB-file-path",
        dest="tier1_DB_file",
        # type=valid_readable_file,
        help=MSG_DB.format("valid local file path")
    )
    parser.add_argument(
        "-t", "--type", "--dtype", "--types", "--dtypes",
        dest="dtypes",
        choices=DEFAULT_DTYPES,
        default=DEFAULT_DTYPES
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        dest="verbosity",
        help=("Include this flag to print more info to stdout while running. "
              "Including the flag more times will print more information.")
    )
    return vars(parser.parse_args())  


if __name__ == "__main__":
    main()
    