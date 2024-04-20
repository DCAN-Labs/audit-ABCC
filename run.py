#!/usr/bin/env python3

"""
ABCC Audit Main Script 
Greg Conan: gregmconan@gmail.com
Created: 2022-07-06
Updated: 2024-04-18
"""
# Standard imports
import argparse
from datetime import datetime
import os
import pdb
from typing import Any, Dict, Mapping, Set

# Local imports
from src.BidsDB import AllBidsDBs
from src.utilities import (
    dt_format, get_most_recent_FTQC_fpath, IMG_DSC_2_COL_HDR_FNAME,
    PATH_NGDR, SplitLogger, UserAWSCreds, valid_output_dir,
    valid_readable_dir, valid_readable_file, WHICH_DBS
)

def main():
    cli_args = _cli()
    cli_args["to_save"] = pick_DBs_to_save(cli_args)

    make_logger_from_cli_args(cli_args)


    # Make all BIDS DBs: fast-track QC DB, tier1 (NGDR) DB, and finally tier 2
    # (s3), finishing with s3 so that it can use the subject-session pairings 
    # from the other BIDS DBs to know which to pull(?)
    client = UserAWSCreds(cli_args, argparse.ArgumentParser()).get_s3_client()
    all_DBs = AllBidsDBs(cli_args, client=client,
                         sub_col="subject", ses_col="session")
    for key in WHICH_DBS:  # First "ftqc", then "tier1", then "s3"
        if key in cli_args["audit"]:
            all_DBs.add_DB(key)
            

    if len(cli_args["audit"]) == 3:       
        all_DBs.summarize()

        if cli_args["debugging"]:
            pdb.set_trace()

        if "summary" not in cli_args["dont_save"]:
            all_DBs.save_all_summary_tsv_files()


def pick_DBs_to_save(cli_args: Dict[str, Any]) -> Set[str]:
    """
    _summary_ 
    :param cli_args: Mapping[str, Any] of command-line input arguments
    :return: Set[str], _description_
    """
    to_save = set()
    dont_save = set(cli_args["dont_save"])
    for which_DB in cli_args["audit"]:
        if not (which_DB in dont_save or cli_args.get(f"{which_DB}_DB_file")):
            to_save.add(which_DB)
    return to_save


def make_logger_from_cli_args(cli_args: Mapping[str, Any]):
    """
    Get logger, and prepare it to log to a file if the user said to 
    :param cli_args: Mapping[str, Any] of command-line input arguments
    :return: logging.Logger, _description_
    """
    log_to = dict()
    if cli_args.get("log"):
        if len(os.path.split(cli_args["log"])) > 1:
            os.makedirs(os.path.dirname(cli_args["log"]), exist_ok=True)
        else:
            cli_args["log"] = os.path.join(cli_args["output"], cli_args["log"])
        log_to = dict(out=cli_args["log"], err=cli_args["log"])
    return SplitLogger(cli_args["verbosity"], **log_to)


def _cli() -> Dict[str, Any]: 
    """ 
    :return: Dict with all command-line arguments from user
    """
    parser = argparse.ArgumentParser()
    dt_str = dt_format(datetime.now())

    SCRIPT_DIR = os.path.dirname(__file__)

    DEFAULT_BUCKETS = ("ABCC_year1", "ABCC_year2", "ABCC_year4")
    DEFAULT_DTYPES = ("anat", "func", "fmap", "dwi")
    DEFAULT_FILE_EXT = ".json"
    DEFAULT_HOST = "https://s3.msi.umn.edu"
    DEFAULT_IMG2HDR = os.path.join(SCRIPT_DIR, IMG_DSC_2_COL_HDR_FNAME)
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
        # type=valid_readable_dir,
        help=(f"{MSG_VALID_PATH} directory structured in BIDS input data "
              f"format.{MSG_DEFAULT.format(PATH_NGDR)}")
    )
    parser.add_argument(
        "-b", "--buckets", "--s3-buckets", "--s3-bucket-names", 
        nargs='+',
        dest="buckets",
        default=DEFAULT_BUCKETS[:2],  # TODO default=DEFAULT_BUCKETS,  # once I actually can access ABCC_year4 
        help=("List of bucket names containing ABCD BIDS data. Excluding "
              "this argument is the same as including `--buckets "
              f"{' '.join(DEFAULT_BUCKETS)}`")
    )
    parser.add_argument(
        "-d", "--debug", "--debugging",
        dest="debugging",
        action="store_true",
        help=("Include this flag to interactively debug errors and exceptions.")
    )
    parser.add_argument(
        "--exclude", "--dont-save", "--no-save", "--dont-keep-DBs",
        dest="dont_save",
        default=list(),
        choices=[*WHICH_DBS, "summary"],
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
        dest="fpath_FTQC",
        default=DEFAULT_FTQC,
        type=valid_readable_file, 
        help=(f"{MSG_VALID_PATH} abcd_fastqc01.txt file from the NDA."
              f"{MSG_DEFAULT.format(DEFAULT_FTQC)}")
    )
    parser.add_argument(
        "-ftqc-DB", "-ftqc-db", "--fast-track-DB", "--ftqc-DB-file",
        dest="ftqc_DB_file",
        type=valid_readable_file,
        help=MSG_DB.format("'ftq_usable' score (1.0 or 0.0)")
    )
    parser.add_argument(
        "-host", "--host", "--s3-host", "--host-URL", "--s3-URL",
        dest="host",
        default=DEFAULT_HOST, 
        help=("URL path to AWS s3 host." + MSG_DEFAULT.format(DEFAULT_HOST))
    )
    parser.add_argument(
        "-i", "-img2hdr", "-img-dsc-2-hdr", "--image-desc-to-header",
        dest="img2hdr_fpath",
        default=DEFAULT_IMG2HDR,
        type=valid_readable_file,
        help=(f"{MSG_VALID_PATH} .JSON file mapping `image_description` "
              "strings from the Fast Track QC spreadsheet to names of "
              "header columns in the output spreadsheets."
              f"{MSG_DEFAULT.format(DEFAULT_IMG2HDR)}")
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
        type=valid_readable_file,
        help=MSG_DB.format("AWS s3 path within a given bucket")
    )
    parser.add_argument(
        "-tier1-DB", "-tier1-db", "--tier1-DB-file-path",
        dest="tier1_DB_file",
        type=valid_readable_file,
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
    return validate_cli_args(vars(parser.parse_args()), parser)


def validate_cli_args(cli_args: Mapping[str, Any],
                      parser: argparse.ArgumentParser) -> Dict[str, Any]:
    if "tier1" in cli_args["audit"] and not cli_args.get("tier1_DB_file"):
        valid_readable_dir(cli_args["tier1_dir"])
    return cli_args


if __name__ == "__main__":
    main()
    