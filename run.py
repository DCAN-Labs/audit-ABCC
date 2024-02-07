#!/usr/bin/env python3

"""
ABCC Audit Main Script 
Greg Conan: gconan@umn.edu
Created 2022-07-06
Updated 2024-02-07
"""
# Standard imports
import argparse
from datetime import datetime
import itertools
import os
import pandas as pd
import pdb
from typing import Any, Callable, Dict, Mapping, Optional

# Local imports
from src.BidsDB import (FastTrackQCDB, Tier1BidsDB, S3BidsDB, BidsDB)
from src.utilities import (
    debug_or_raise, dt_format, get_and_log_time_since, LazyDict, invert_dict,
    is_nothing, log, make_default_out_file_name, make_logger, mutual_reindex,
    PATH_DICOM_DB, PATH_NGDR, s3_get_info, UserAWSCreds, valid_output_dir,
    valid_readable_dir, valid_readable_file
)

# Constant: Shorthand names for the different kinds of BIDS DBs
WHICH_DBS = ("ftqc", "tier1", "s3")

def main():
    cli_args = _cli()
    make_logger_from_cli_args(cli_args)

    # Make all BIDS DBs: tier1 (NGDR), tier 2 (s3), and fast-track QC DB
    client = make_s3_client_from_cli_args(cli_args)
    all_DBs = AllBidsDBs(cli_args, client=client,
                         sub_col="subject", ses_col="session")
    for key in cli_args["audit"]:
        all_DBs.add_DB(key)

    # TODO Move this block to BidsDB.FastTrackQCDB.make_FTQC_df_from
    # Collapse T?w-NORM into T?w by, per subject session, preferring the
    # former except when there's none. 
    ftqc_df = all_DBs.get_DB("ftqc").df
    for t in (1, 2):
        for col_NORM in ftqc_df.columns:
            if col_NORM.startswith(f"T{t}-NORM"):
                col_Tw = f"T{t}_{col_NORM.split('_')[-1]}"
                if col_Tw in ftqc_df.columns:
                    ftqc_df[col_Tw] = \
                        ftqc_df.pop(col_NORM).combine_first(ftqc_df[col_Tw])
                else:
                    ftqc_df.rename(columns={col_NORM: col_Tw}, inplace=True)
                    # ftqc_df[col_Tw] = ftqc_df.pop(col_NORM)

    # Summarize results; identify subject sessions not matching expectations
    summary_DB_df = all_DBs.summarize_all()
    if cli_args["debug"]:
        pdb.set_trace()

    # summary_DB_df["coverage"]

    # Save summary dataframe to .tsv file  # TODO CLEAN UP & MODULARIZE &c
    summary_DB_df.to_csv(os.path.join(
        cli_args["output"], make_default_out_file_name("audit-summary")
    ), sep="\t", index=True, header=True, columns=["complete", "coverage_s3", "coverage_tier1"])
    summary_DB_df.to_csv(os.path.join(
        cli_args["output"], make_default_out_file_name("audit-full")
    ), sep="\t", index=True, header=True)
    summary_DB_df.reset_index(inplace=True)
    summary_DB_df[(summary_DB_df['coverage_s3']!=1.0) &
                  (summary_DB_df['coverage_tier1']!=1.0) &
                  (summary_DB_df['session'].isin({'ses-baselineYear1Arm1',
                                                  'ses-2YearFollowUpYArm1'}))
                  ].to_csv(os.path.join(
        cli_args["output"], make_default_out_file_name("incomplete-subj-ses")
    ), sep="\t", index=False, header=True, columns=[all_DBs.COLS.sub_col,
                                                    all_DBs.COLS.ses_col])


class AllBidsDBs(LazyDict):

    def __init__(self, cli_args: Mapping[str, Any], 
                 # dtypes: Optional[Iterable[str]] = 
                 sub_col: str, ses_col: str, client=None) -> None:
                 # which: Iterable[str] = ["ftqc", "tier1", "s3"]
        self.update(cli_args) # Save cli_args as attributes for easy access
        self.client = client  # client: boto3.session.Session.client
        self.COLS = LazyDict(sub_col=sub_col, ses_col=ses_col)
        self.DB = list()      # List[BidsDB]
        self.elapsed = list() # List[timedelta]: how long making DBs took
        self.ix_of = dict()   # Dict[str, int]: indices of each DB in self.DB
        self.out_fpaths = {key: self.build_out_fpath(key)
                           for key in self["to_save"]}
        self.timestamps = [datetime.now()]  # datetime when each DB is done
        
        # self.add_DB input parameters specific to each kind of BidsDB
        self.PARAMS = pd.DataFrame([
            [FastTrackQCDB, "ABCD FastTrack01 QC", "the NDA",
             ["fpath_QC_CSV"]],
            [Tier1BidsDB, "Tier 1", "the NGDR space on the MSI",
             ["tier1_dir", "file_ext"]],
            [S3BidsDB, "Tier 2", "these AWS s3 buckets: " +
             ", ".join(self["buckets"]), ["client", "buckets", "file_ext"]]
        ], columns=["DB", "db_name", "src", "kwargs"], index=WHICH_DBS)


    # TODO Add option to load BIDS DB objects from file
    def add_DB(self, key: str):  # make_DB: Callable, *its_args, **its_kwargs):
        """
        Given the shorthand name for a BidsDB, create it and store it 
        """
        which = self.PARAMS.loc[key]
        CURRENT_TASK = f"started making {which.db_name} DB from {which.src}"
        log(f"Just {CURRENT_TASK}")
        self.ix_of[key] = len(self.DB)
        self.DB.append(self.make_DB(key, which.DB, **{
            kwarg: self[kwarg] for kwarg in which.kwargs
        })) # *its_args, **its_kwargs)) # , in_fpath=in_fpath,  dtypes=self["dtypes"]))
        # self.DB[-1].df.name = key  #?
        self.elapsed.append(get_and_log_time_since(CURRENT_TASK,
                                                   self.timestamps[-1]))
        self.timestamps.append(datetime.now())


    def make_DB(self, key: str, initalize_DB: Callable, **kwargs: Any
                ) -> BidsDB:
        """
        :param key: String, shorthand name for which BIDS DB to create:
                    "ftqc"=FastTrackQCDB, "tier1"=Tier1BidsDB, "s3"=S3BidsDB
        :param initalize_DB: Class object to create a BidsDB
        :return: BidsDB
        """
        return initalize_DB(
            in_fpath=self[f"{key}_DB_file"], dtypes=self["dtypes"],
            out_fpath=self.out_fpaths.get(key), **self.COLS, **kwargs
        )
    

    def get_DB(self, key_DB: str) -> BidsDB:
        """
        :param key_DB: String, shorthand name for which BIDS DB to fetch
        :return: BidsDB (already created) whose shorthand name is key_DB
        """
        return self.DB[self.ix_of[key_DB]]
    

    def build_out_fpath(self, key_DB: str):
        return os.path.join(self["output"], self.lazyget(
            f"{key_DB}_DB_file", lambda: make_default_out_file_name(key_DB)
        ))


    def summarize_all(self) -> pd.DataFrame:
        """
        Get boolified versions of each df, then combine them by
        assigning a separate value for any combination of:
        - "bids" if FastTrackQCDB.df[.ftq_usable] else "delete"
        - "(tier1)" if Tier1BidsDB.df[.NGDR_fpath] else ""
        - "(s3)" if S3BidsDB.df[.s3_subpath] else ""
        The result will include "bids (s3) (tier1)", "delete (tier1)", etc
        :return: pd.DataFrame, _description_
        """
        self.LOC_KEYS = ("tier1", "s3")
        booled_list = [eachDB.df.applymap(lambda x: not is_nothing(x)
                                          ) for eachDB in self.DB]

        # Combine all dataframes' indices so each has every subject session
        booled_list = mutual_reindex(*booled_list, fill_value=False)
        self.key_of = invert_dict(self.ix_of)
        self.booled = LazyDict({self.key_of[ix]: booled_list[ix]
                                for ix in range(len(booled_list))})
        # self.COLS.bool = set(booled_list[0].columns).union(set(booled_list[1].columns)).union(set(booled_list[2].columns))

        self.COLS.booled = booled_list[0].columns
        for i in range(1, 3):
            self.COLS.booled = self.COLS.booled.intersection(booled_list[i].columns)
        detail_df = self.booled["ftqc"].apply(self.summarize_col)
        bids_DB_df = detail_df.applymap(
            lambda cell: "" if is_nothing(cell) else cell["summary"]
        )
        for location in self.LOC_KEYS:
            COVG_LOC = f"coverage_{location}"
            bids_DB_df[COVG_LOC] = detail_df.applymap(
                # TODO Should NaN mean good (1.0) or bad (0.0)?
                lambda cell: 1.0 if is_nothing(cell) else cell[COVG_LOC]  
            ).apply(lambda row: row.sum() / row.shape[0], axis=1)
        bids_DB_df["complete"] = bids_DB_df.apply(self.sub_ses_is_complete,
                                                  axis=1)
        return bids_DB_df
    

    def sub_ses_is_complete(self, sub_ses_row):  # , location_keys):
        # is_complete_on = dict()
        is_complete_on = list()
        for loc_key in self.LOC_KEYS:
            if sub_ses_row.get(f"coverage_{loc_key}") == 1.0:
                is_complete_on.append(loc_key)
        completion = {2: "complete (both)",
                      1: "complete ({})",
                      0: "incomplete"}[len(is_complete_on)]
        if len(is_complete_on) == 1:  # TODO clean up
            completion = completion.format(is_complete_on[0])
        return completion


    def summarize_col(self, ftqc_col: pd.Series) -> pd.Series:  #, ixs: pd.Index):
        # tier1_col.apply(lambda cell: self.summarize_1_cell(
        # location_keys = self.booled.keys()
        # if all([ftqc_col.name in self.get_DB(lockey) for lockey in location_keys]):
        if ftqc_col.name in self.COLS.booled:
            summary_col = ftqc_col.to_frame(ftqc_col.name).apply(
                self.summarize_1_cell, axis=1
                # lambda sub_ses_cell: self.summarize_1_cell(sub_ses_cell, location_keys), axis=1
            )  # lambda sub_ses_row: self.summarize_1_cell(sub_ses_row, ftqc_col.name  #, ixs, )
            # pdb.set_trace()
        else:
            summary_col =  pd.Series().reindex_like(ftqc_col)
        return summary_col
    

    def summarize_1_cell(self, sub_ses_cell):  # , loc_keys):
        cells = {"ftqc": sub_ses_cell[0]}
        details = dict()
        for key in self.LOC_KEYS:
            cells[key] = self.booled[key].loc[sub_ses_cell.name,
                                              sub_ses_cell.index[0]]
            details[f"coverage_{key}"] = float(cells["ftqc"] == cells[key])
        details["summary"] = build_summary_str(**cells)
        return details
    

def build_summary_str(ftqc: bool, s3: bool, tier1: bool) -> str:
    return "".join(["goodQC" if ftqc else "badQC",
                    " (s3)" if s3 else "",
                    " (tier1)" if tier1 else ""])


def make_s3_client_from_cli_args(cli_args):
    # Get AWS credentials to access s3 buckets
    if not cli_args["aws_keys"]:
        my_s3info = s3_get_info()
        if my_s3info:
            cli_args["aws_keys"] = [my_s3info["access key"],
                                    my_s3info["secret key"]]
    # cli_args = LazyDict(host="https://s3.msi.umn.edu", buckets=["ABCC_year1", "ABCC_year2"], aws_keys=[my_s3info["access key"], my_s3info["secret key"]])
    return UserAWSCreds(cli_args, argparse.ArgumentParser()).get_s3_client()


def make_logger_from_cli_args(cli_args):
    # Get logger, and prepare it to log to a file if the user said to
    log_to = dict()
    if cli_args.get("log"):
        if len(os.path.split(cli_args["log"])) > 1:
            os.makedirs(os.path.dirname(cli_args["log"]), exist_ok=True)
        else:
            cli_args["log"] = os.path.join(cli_args["output"], cli_args["log"])
        # if os.path.isdir(os.path.dirname(cli_args["log"])):
        log_to = dict(out=cli_args["log"], err=cli_args["log"])
    return make_logger(cli_args["verbosity"], **log_to)


def _cli() -> Dict[str, Any]: 
    """ 
    :return: Dict with all command-line arguments from user
    """
    # audits = [audit.split("abcd-")[-1] for audit in audit_names]
    parser = argparse.ArgumentParser()
    dt_str = dt_format(datetime.now())
    DEFAULT_BUCKETS = ("ABCC_year1", "ABCC_year2")  # , "ABCC_year4")  # TODO I can't access year 4 apparently
    DEFAULT_DTYPES = ("anat", "func", "fmap") #, dwi
    DEFAULT_FILE_EXT = ".json"
    DEFAULT_HOST = "https://s3.msi.umn.edu"
    DEFAULT_FTQC = ("/home/rando149/shared/code/internal/utilities/"
                    "abcd-dicom2bids/spreadsheets/fastqc20240112/"
                    "abcd_fastqc01.txt")
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
        "-d", "--debug",
        action="store_true",
        help=("Include this flag to interactively debug errors and exceptions.")
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
        "-save", "--save", "--save-DBs", "-keep", "--keep", "--keep-DBs",
        dest="to_save",
        default=list(),
        choices=WHICH_DBS,
        nargs="+",
        help=("Which BIDS DBs you want to save to files. By default, none "
              "will be saved.")
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