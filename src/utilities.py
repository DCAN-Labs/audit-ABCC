#!/usr/bin/env python3

"""
Utilities for ABCC auditing workflow
Originally written by Anders Perrone
Updated by Greg Conan on 2024-03-05
"""
# Standard imports
import argparse
from collections.abc import Callable, Hashable
from datetime import datetime
import functools
from getpass import getpass
from glob import glob
import json
import logging
import os
import pdb
import re
import subprocess as sp
import sys
from typing import (Any, Dict, Generator, Iterable, List,  # Literal, 
                    Mapping, Optional, Tuple, Union)

# External (pip/PyPI) imports
import boto3
import nibabel as nib
import numpy as np
import pandas as pd

# Constants

# Pipeline names, col names
BIDS_COLUMNS = ("bids_subject_id", "bids_session_id")
BIDSPIPELINE = "abcd-bids-pipeline"
DICOM2BIDS = "abcd-dicom2bids"

# Column names to split BIDS filename into for each data type (dtype)
DTYPE_2_UNIQ_COLS = {"anat": ["rec-normalized", "run", "Tw"],
                     "dwi":  ["run"],
                     "fmap": ["acq", "dir", "run"],
                     "func": ["task", "run"]}

# Database path, data types definition fpath, temporarily hardcoded dirpath
IMG_DSC_2_COL_HDR_FNAME = "image_description_to_header_col.json"
LOGGER_NAME = "BidsDBLogger"
PATH_ABCD_BIDS_DB = "/home/rando149/shared/projects/ABCC_year2_processing/s3_status_report.csv"
PATH_DICOM_DB = "/home/rando149/shared/code/internal/utilities/abcd-dicom2bids/src/audit/ABCD_BIDS_db.csv"
PATH_NGDR = "/spaces/ngdr/ref-data/abcd/nda-3165-2020-09/"
SESSION_DICT = {'baseline_year_1_arm_1': 'ses-baselineYear1Arm1',
                '2_year_follow_up_y_arm_1': 'ses-2YearFollowUpYArm1',
                '4_year_follow_up_y_arm_1': 'ses-4YearFollowUpYArm1'}

# Shorthand names for the different kinds of BIDS DBs
WHICH_DBS = ("ftqc", "tier1", "s3")


def main():
    pd.set_option('display.max_columns', None)


def attrs_in(an_obj: Any) -> List[str]:
    """
    :param an_obj: Any
    :return: List of strings naming every public attribute in an_obj
    """
    return uniqs_in([attr_name if not attr_name.startswith("_")
                     else None for attr_name in dir(an_obj)])


def build_NGDR_fpath(root_BIDS_dir: str, parent_dirname: str,
                     which_BIDS_file: str) -> str:
    """
    :param root_BIDS_dir: str, valid path to existing top-level BIDS directory
                          with "sub-*/ses-*/" subdirectories
    :param parent_dirname: str naming the subdirectory of "sub-*/ses-*/" to
                           get files from 
    :param which_BIDS_file: str naming the files in parent_dirname to get
    :return: str, (BIDS-)valid path to existing local files
    """
    return os.path.join(root_BIDS_dir, "sub-*", "ses-*", parent_dirname,
                        f"sub-*_ses-*_{which_BIDS_file}")


def boolify_and_clean_col(exploded_col: pd.Series) -> pd.Series:
    """
    :param exploded_col: pd.Series of strings, each element either the empty
                         string or a certain other string
    :return: pd.Series of bools, replacing the empty string with False and the
             other string with True; that other string is now the Series name
    """
    new_col = exploded_col != ""
    new_col.name = exploded_col[new_col].unique()[0]
    return new_col
    

def build_summary_str(ftqc: bool, s3: bool, tier1: bool, **_: Any) -> str:
    """ 
    :param ftqc: True if a given file passed QC (ftq_usable == 1.0) else False
    :param s3: True if a given file is present in an ABCC s3 bucket else False
    :param tier1: True if a given file is present in NGDR on tier 1 else False
    :return: String summarizing the given file's completeness
    """
    return "".join(["goodQC" if ftqc else "badQC",
                    " (s3)" if s3 else "",
                    " (tier1)" if tier1 else ""])


def df_is_nothing(df: Optional[pd.DataFrame] = None) -> bool:
    """
    :param df: Object to check if it's nothing/empty/falsy
    :return: True if df is None or empty; otherwise False
    """
    return (df is None or df.empty)


def debug(an_err: Exception, local_vars: Mapping[str, Any]) -> None:
    """
    :param an_err: Exception (any)
    :param local_vars: Dict[str, Any] mapping variables' names to their
                       values; locals() called from where an_err originated
    """
    locals().update(local_vars)
    if verbosity_is_at_least(2):
        logging.getLogger(LOGGER_NAME).exception(an_err)  # .__traceback__)
    if verbosity_is_at_least(1):
        show_keys_in(locals())  # , logging.getLogger(LOGGER_NAME).info)
    pdb.set_trace()


def default_pop(poppable: Any, key: Optional[Any] = None,
                default: Optional[Any] = None) -> Any:
    """
    :param poppable: Any object which implements the .pop() method
    :param key: Input parameter for .pop(), or None to call with no parameters
    :param default: Object to return if running .pop() raises an error
    :return: Object popped from poppable.pop(key), if any; otherwise default
    """
    try:
        if key is None:
            to_return = poppable.pop()
        else:
            to_return = poppable.pop(key)
    except (AttributeError, IndexError):
        to_return = default
    return to_return


def dt_format(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%d_%H-%M-%S")


def get_and_log_time_since(event_name, event_time):
    """
    Print and return a string showing how much time has passed since the
    current running script reached a certain part of its process
    :param event_name: String to print after 'Time elapsed since '
    :param event_time: datetime object representing a time in the past
    :param logger: logging.Logger object to show messages and raise warnings
    :return: datetime.datetime representing the moment this function is called
    """
    right_now = datetime.now()  # dt.datetime.now()
    log(f"\nTime elapsed since {event_name}: {right_now - event_time}")
    return right_now


def get_and_print_time_if(will_print: bool, event_time: datetime,
                          event_name: str) -> datetime:
    """
    Print and return a string showing how much time has passed since the
    current running script reached a certain part of its process
    :param will_print: True to print an easily human-readable message
                       showing how much time has passed since {event_time}
                       when {event_name} happened, False to skip printing
    :param event_time: datetime object representing a time in the past
    :param event_name: String to print after 'Time elapsed '
    :return: datetime object representing the current moment
    """
    timestamp = datetime.now()
    if will_print:
        print(f"\nTime elapsed {event_name}: {timestamp - event_time}")
    return timestamp 


def get_col_headers_for(dtype: str, df: Optional[pd.DataFrame] = None) -> set:
    # headers = set(headers) if headers else set()
    headers = set()
    prefix, uniq_col = get_header_vars_for(dtype)
    if not (df is None or df.empty): # not is_nothing(df):
        uniqs = df[uniq_col].unique()
        runs = df["run"].unique()
        # pdb.set_trace()
        for run in runs:
            for uniq in uniqs:
                headers.add(make_col_header(f"{prefix}{uniq}", run))
    return headers


def get_most_recent_FTQC_fpath(incomplete_dirpath_FTQC):
    # Get all readable abcd_fastqc01.txt file paths
    ftqc_paths = pd.DataFrame({"fpath":
                               glob(incomplete_dirpath_FTQC.format("*"))})
    ftqc_paths["readable"] = ftqc_paths["fpath"].apply(
        lambda fpath: os.access(fpath, os.R_OK)
    )
    # ftqc_paths.drop(index=~ftqc_paths["readable"], inplace=True)
    ftqc_paths = ftqc_paths[ftqc_paths["readable"]]

    # If there are no readable abcd_fastqc01.txt file paths, return None
    if ftqc_paths.empty:
        most_recent_FTQC = None
    else:

        # Get the datetimestamp (int) from each abcd_fastqc01.txt file name
        prefix, suffix = incomplete_dirpath_FTQC.split("{}")
        ftqc_paths["dtstamp"] = \
            ftqc_paths["fpath"].str.strip(prefix).str.rstrip(suffix)
        ftqc_paths["dtstamp"] = ftqc_paths["dtstamp"].astype(int)

        # Return the path to the most recent abcd_fastqc01.txt file
        most_recent_FTQC = \
            ftqc_paths.loc[ftqc_paths["dtstamp"].idxmax(), "fpath"]
    return most_recent_FTQC


def get_ERI_filepath(bids_dir_path: str) -> str:
    return build_NGDR_fpath(os.path.join(bids_dir_path, "sourcedata"),
                            "func", "task-*_run-*"
                            "_bold_EventRelatedInformation.txt")


def get_header_vars_for(dtype: str) -> Tuple[str]:
    return {"func":      ("task-", "task"),
            "anat":      ("T", "Tw"),
            "fmap":      ("FM-", "dir"),
            "dwi":       ("", "dtype") # "dtype" -> uniq_col with only 1 value
            }.get(dtype, ("", ""))


def get_tier1_or_tier2_ERI_db_fname(parent_dir: str, tier: int) -> str:
    return os.path.join(parent_dir, f"ERI_tier{tier}_paths_bids_db.csv")


def get_variance_of_each_row_of(dtseries_path: str):
    """
    :param dtseries_path: String, valid path to existing dtseries.nii file
    :return: np.ndarray of floats; each is the variance of a row in the dtseries file
    """
    return load_matrix_from(dtseries_path).var(axis=1)


def invert_dict(a_dict: Dict[Hashable, Hashable],
                keep_collision: bool = False) -> Dict[Hashable, Hashable]:
    """
    Switch a dict's keys and values.
    If two keys mapped to the same value, then they will collide when the keys
    and values are swapped: which key should be kept? Calling this function
    with keep_collision will put the colliding old keys into sets.
    Otherwise, the old value will only be mapped as a new key to 
    whichever old key comes up last during iteration.
    :param a_dict: Dictionary (any with hashable values)
    :param keep_collision: True to keep old keys with the same value
    :return: dict mapping old values (new keys) to old keys (new values).
    """
    if keep_collision:
        new_dict = dict()
        for k, v in a_dict.items():
            if v in new_dict:
                if isinstance(new_dict[v], set):
                    new_dict[v].add(k)
                else:
                    new_dict[v] = {k, new_dict[v]}
            else:
                new_dict[v] = k
    else:  # By default, simply set every value to a key and vice versa
        new_dict = {v: k for k, v in a_dict.items()}
    return new_dict


class ImgDscColNameSwapper:  # (LazyDict): 
    def __init__(self, json_fpath: Optional[str] = None) -> None:
        self.fpath = self.find_fpath(json_fpath)
        self.dsc2hdr = extract_from_json(self.fpath)
        self.hdr2dsc = invert_dict(self.dsc2hdr,
                                   keep_collision=True)
        self.hdr2dsc.pop(None, None)

        # Map each header column name and image_description to its dtype
        self.dtype_of = LazyDict()
        HDR_PFX_2_DTYPE = {"T1": "anat", "T2": "anat", "task": "func",
                           "FM": "fmap", "dwi": "dwi"}
        for dsc, hdr in self.dsc2hdr.items():
            self.dtype_of[hdr] = (HDR_PFX_2_DTYPE.get(hdr.split("-", 1)[0], None)
                                  if hdr else None)  # self.hdr_col_to_dtype(hdr)
            self.dtype_of[dsc] = self.dtype_of[hdr]

    def find_fpath(self, json_fpath: Optional[str] = None) -> str:
        to_check = list()
        if json_fpath:
            fname, dir_to_check = os.path.split(json_fpath)
            to_check = [dir_to_check]
        else:
            fname = IMG_DSC_2_COL_HDR_FNAME
            to_check = list()
        to_check += [os.path.dirname(sys.argv[0]),
                     os.path.dirname(__file__), os.getcwd()]
        return search_for_readable_file(fname, *to_check)
        
    def to_header_col(self, image_description: str) -> str:
        return self.dsc2hdr.get(image_description)
    
    def to_image_desc(self, header_col_name: str) -> str:
        return self.hdr2dsc.get(header_col_name)


def iter_attr(an_obj: Any) -> Generator[str, None, None]:
    try:
        iterator = iter(an_obj)
    except TypeError:
        iterator = iter(dir(an_obj))
    for element_or_attribute_name in iterator:
        yield element_or_attribute_name


class LazyDict(dict):
    """
    Dictionary subclass that can get/set items...
    ...as object-attributes: self.item is self['item']. Benefit: You can
       get/set items by using '.' or by using variable names in brackets.
    ...and ignore the 'default=' code until it's needed, ONLY evaluating it
       after failing to get/set an existing key. Benefit: The 'default='
       code does not need to be valid if self already has the key.
    Extended version of LazyButHonestDict from stackoverflow.com/q/17532929
    Does not change core functionality of the Python dict type.
    TODO: Right now, trying to overwrite a LazyDict method or a core dict
          attribute will silently fail: the new value can be accessed through
          dict methods but not as an attribute. Maybe worth fixing eventually?
    """
    def __getattr__(self, __name: str):
        """
        For convenience, access items as object attributes.
        :param __name: String naming this instance's item/attribute to return
        :return: Object (any) mapped to __name in this instance
        """
        return self.__getitem__(__name)
    
    def __setattr__(self, __name: str, __value: Any) -> None:
        """
        For convenience, set items as object attributes.
        :param __name: String, the key to map __value to in this instance
        :param __value: Object (any) to store in this instance
        """
        self.__setitem__(__name, __value)

    def lazyget(self, key: Hashable,
                get_if_absent:Optional[Callable] = lambda: None) -> Any:
        """
        LazyButHonestDict.lazyget from stackoverflow.com/q/17532929
        :param key: Object (hashable) to use as a dict key
        :param get_if_absent: function that returns the default value
        :return: _type_, _description_
        """
        return self[key] if self.get(key) is not None else get_if_absent()
    
    def lazysetdefault(self, key: Hashable,
                       set_if_absent:Optional[Callable] = lambda: None) -> Any:
        """
        LazyButHonestDict.lazysetdefault from stackoverflow.com/q/17532929 
        :param key: Object (hashable) to use as a dict key
        :param set_if_absent: function that returns the default value
        :return: _type_, _description_
        """
        return (self[key] if self.get(key) is not None else
                self.setdefault(key, set_if_absent()))
    
    # def subset(self, *keys: Hashable) -> "LazyDict":
    #     return LazyDict({key: self.get(key) for key in keys})


def float_is_nothing(thing: Optional[float]) -> bool:
    # return True if thing in {None, np.nan} else not thing
    return thing is None or np.isnan(thing) or not thing


def load_matrix_from(matrix_path: str):
    """
    :param matrix_path: String, the absolute path to a .gii or .nii matrix file
    :return: numpy.ndarray matrix of data from the matrix file
    """
    return {".gii": lambda x: x.agg_data(),
            ".nii": lambda x: np.array(x.get_fdata().tolist()),
            }[os.path.splitext(matrix_path)[-1]](nib.load(matrix_path))


def log(msg: str, level: int = logging.INFO):
    logging.getLogger(LOGGER_NAME).log(level, msg)


def make_col_header(prefix: str, run_num: Optional[float] = 1,
                    debugging: bool = False) -> str:
    try:  
        if float_is_nothing(run_num):  # run_num is None or np.isnan(run_num):  # is_nothing(run_num):
            run_num = 1
        return f"{prefix}_{run_num_to_str(run_num)}"
                # if isnt_nothing(run_num) else prefix)  # TODO Ensure run number added
    except ValueError as e:
        if debugging:
            debug(e, locals())
        else:
            raise e


def make_col_header_from(row: pd.Series, dtype: str, uniq=None,
                         debugging: bool = False) -> str:
    if uniq:
        uniq_col = uniq
        prefix, _ = get_header_vars_for(dtype)
    else:
        prefix, uniq_col = get_header_vars_for(dtype)
    try:
        return make_col_header(f"{prefix}{row.get(uniq_col)}", row.get("run"))
    except ValueError as e:
        if debugging:
            debug(e, locals())
        else:
            raise e
    

def make_default_out_file_name(key_DB: str):
    return f"{key_DB}_BIDS_DB_{dt_format(datetime.now())}.tsv"


def make_ERI_filepath(parent: str, subj_ID: str, session: str, task: str,
                      run: Union[int, str]) -> str:  # run: int | str
    """
    :param parent: String, valid path to tier1 directory (e.g.
                   "/home/nda-3165-2020-09/") or s3 bucket (e.g. "s3://bucket")
    :param subj_ID: String naming a subject ID
    :param session: String naming a session
    :param task: String naming a task
    :param run: Int or string, the run number
    :return: String, valid path to an EventRelatedInformation.txt file
    """
    return os.path.join(parent, "sourcedata", subj_ID, session, "func",
                        f"{subj_ID}_{session}_task-{task}_run-{run}_bold_"
                        "EventRelatedInformation.txt")


def make_logger(verbosity: int, out: Optional[str] = None,
                err: Optional[str] = None) -> logging.Logger:
    """
    Make logger to log status updates, warnings, and other important info
    :return: logging.Logger able to print info to stdout and problems to stderr
    """  # TODO stackoverflow.com/a/33163197 ?
    err_log = dict(filename=err) if err else dict(stream=sys.stderr)
    out_log = dict(filename=out) if out else dict(stream=sys.stdout)
    FMT = "\n%(levelname)s %(asctime)s: %(message)s"
    if verbosity > 0:
        logging.basicConfig(**out_log, format=FMT, level=logging.INFO) 
    if verbosity > 2:
        logging.basicConfig(**out_log, format=FMT, level=logging.DEBUG)
    logging.basicConfig(**err_log, format=FMT, level=logging.ERROR)
    logging.basicConfig(**err_log, format=FMT, level=logging.WARNING)
    return logging.getLogger(LOGGER_NAME)  # os.path.basename(sys.argv[0]))


def mutual_reindex(*dfs: pd.DataFrame, fill_value:
                   Optional[Hashable] = np.nan) -> List[pd.DataFrame]:
    # Combine all dataframes' indices so each has every subject session
    return mutual_reindex_dfs(*dfs, fill_value=fill_value
                              ) if len(dfs) > 1 else dfs


def mutual_reindex_dfs(*dfs: pd.DataFrame, fill_value:
                       Optional[Hashable] = np.nan) -> List[pd.DataFrame]:
    combined_ixs = functools.reduce(lambda x, y: x.union(y),
                                    [df.index for df in dfs])
    return [df.reindex(combined_ixs, fill_value=fill_value) for df in dfs]


def reformat_BIDS_df_col(col: pd.Series) -> pd.Series:
    reformatters = {"rec-normalized": bool, "Tw": int,
                    "run": lambda x: float(x) if x != "" else np.nan}
    return col.apply(reformatters.get(col.name, lambda x: x))


def reformat_pGUID(pguid: str) -> str:
    uid_start = "INV"
    try:
        uid = pguid.split(uid_start, 1)[1]
    except ValueError:
        print(f"WARNING: {pguid} is improperly formatted. "
              "Assuming last 8 chars is the uid")
        uid = pguid[-8:]
    return f"sub-NDARINV{''.join(uid)}"


class RegexForBidsDetails(LazyDict):
    # Regex patterns to find each detail identifying what a str is about/for
    FIND = LazyDict({
        "- (": r"(?:\s{1}\()?",  # Exclude space & opening parenthesis if any
        "-)": r"(?:\))?",        # Exclude closing parenthesis if any
        "-_": r"(?:_)",          # Exclude underscore
        "-/": r"\/",             # Exclude slash
        "-ses": r"(?:ses-.*?)",  # Exclude text from ses ID until next group
        "-subj": r"(?:sub-.*?)", # Exclude text from subj ID until next group
        "any": r"(.*?)",         # Get all text before the next capture group
        "acq": r"(?:acq-)?(.*?)",  # Get what's after "acq-" if it's present
        "dir": r"(dir-.*?)",    # Get all text from "dir-" until next group
        "nothing else": r"(?:.*)",  # Exclude everything from here onward
        "rec": r"(rec-.*?)?",   # Get any text from "rec-" until next group
        "run": r"(?:run-)?([0-9]{2})?",  # Get 2-digit run number
        "s3": r"(s3)?",         # Get the string "s3" if it's present
        "ses": r"(ses-.*?)",    # Get all text from "ses-" until next group
        "subj": r"(sub-NDARINV.{8})",  # Get all text from "sub-" until next
        "task": r"(?:task-)(.*?)",  # Get all text after "task-" until next
        "tier1": r"(tier1)?",   # Get the string "tier1" if it's present
        "Tw": r"(?:T)([0-9]?)(?:\w.*)"  # Get the digit between T and w
    })
    FIND["-_?"] = FIND["-_"] + "?"  # Exclude underscore, if any is present

    SPLIT = LazyDict({  # Keys to split BIDS strings into identifying details

        # BIDS-valid file name -> tuple of 5 strings: subject ID, session ID,
        # rec if any, run number if any, and whether the image is T1 or T2
        "anat": ("subj", "-_", "ses", "-_?", "rec", "?", "-_?", "run",
                 "-_", "Tw"),

        # Subj-ses audit string -> tuple of 3 strings: 'bids' or 'delete',
        # 'tier1' if present, and 's3' if present
        "audit": ("^", "any", "- (", "tier1", "-)", "- (", "s3", "-)", "$"),

        # BIDS-valid file path -> string naming data type (parent dir)
        "dtype": ("-subj", "-/", "-ses", "-/", "any", "-/", "-)"),

        # BIDS-valid file name -> tuple of 3 strings: subject ID, session ID,
        # and run number if any
        "dwi": ("subj", "-_", "ses", "-_", "run"),

        # BIDS-valid file name -> tuple of 5 strings: subject ID, session ID,
        # task name, acquisition name/ID if any, and run number if any
        "fmap": ("subj", "-_", "ses", "-_", "acq", "-_?", "dir", "?", "-_",
                 "run", "-_?"),

        # BIDS-valid file name -> tuple of 4 strings: subject ID, session ID,
        # task name, acquisition name/ID if any, and run number if any
        "func": ("subj", "-_", "ses", "-_?", "task", "-_", # "acq", "?", "-_?",
                 "run", "-_?", "nothing else")
    })

    def __init__(self, *pattern_names: str) -> None:
        for pattern in pattern_names:
            self[pattern] = self.create(*self.SPLIT.lazyget(pattern, list))

    def create(self, *args: str) -> re.Pattern:
        """
        :return: re.Pattern to use for splitting a string into desired details
        """
        return re.compile("".join([self.FIND.get(key, key) for key in args]))
    

def explode_col(ser: pd.Series, re_patterns: RegexForBidsDetails, dtype: str,
                debugging: bool = False) -> list:
    try:
        num_new_cols = DTYPE_2_UNIQ_COLS.get(dtype)
        exploded = ser.str.findall(re_patterns[dtype]
                                   ).explode(ignore_index=True)
        # exploded[exploded.isna()] = [''] * 5
        ixs_NaNs = exploded.isna()
        if ixs_NaNs.any():
            # assert exploded[~ixs_NaNs].any()
            # num_new_cols = len(exploded[~ixs_NaNs].iloc[0])
            if not num_new_cols:  # is_nothing(num_new_cols):
                num_new_cols = len(exploded[~ixs_NaNs].iloc[0])
            exploded[ixs_NaNs] = exploded[ixs_NaNs].apply(
                lambda _: [np.nan] * num_new_cols
            )
    except (AttributeError, KeyError, ValueError) as e:  # AssertionError, 
        if debugging:
            debug(e, locals())
        else:
            raise e
    return exploded.to_list()


def extract_from_json(json_path: str) -> Dict:
    """
    :param json_path: String, a valid path to a real readable .json file
    :return: Dictionary, the contents of the file at json_path
    """
    with open(json_path, 'r') as infile:
        return json.load(infile)


def run_num_to_str(run_num:Optional[float] = 1):
    return f"run-{int(run_num):02d}"


def s3_get_info() -> LazyDict:
    """
    :return: Dictionary containing all of the information in the output of the
            "s3info" command in the Unix Bash terminal (via subprocess)
    """
    user_s3info = sp.check_output(("s3info")).decode("utf-8").split("\n")
    aws_s3info = LazyDict()
    for eachline in user_s3info:
        # if eachline != "":
        split = eachline.split(":")
        if len(split) > 1:
            split = [x.strip() for x in split]
            aws_s3info[split[0].lower()] = split[-1]
    return aws_s3info


def save_to_hash_map_table(sub_ses_df: pd.DataFrame, subj_col: str,
                           ses_col: str, in_dirpath: str, out_dirpath: str,
                           outfile_path: str, subdirnames=list()) -> None:
    """
    Save out a text file mapping (for each subject-session) old paths to new 
    paths, with 2 space-separated paths (i.e. 1 old-to-new mapping) per line
    :param sub_ses_df: pandas.DataFrame with columns named subj_col and ses_col
    :param subj_col: String naming a column of sub_ses_df with subject IDs
    :param ses_col: String naming a column of sub_ses_df with session IDs
    :param in_dirpath: String, valid path to existing dir to copy files OUT FROM
    :param out_dirpath: String, valid path to existing dir to copy files INTO
    :param outfile_path: String, valid path to text file to save out
    :param subdirnames: List of strings, each naming a subdir between an
                        in_/out_dirpath and the subject and session subdirs;
                        empty by default
    """
    # Add in_dirpath/subj/ses and out_dirpath/subj/ses, each as a column
    dirpaths = {"in": in_dirpath, "out": out_dirpath}
    for prefix in dirpaths.keys():
        sub_ses_df["{}put_path".format(prefix)] = sub_ses_df.apply(
            lambda row: os.path.join(dirpaths[prefix], *subdirnames,
                                     row.get(subj_col), row.get(ses_col)),
            axis="columns"
        )

    # Save those two new columns, space-separated, into a text file
    sub_ses_df.get(["input_path", "output_path"]
                   ).to_csv(outfile_path, sep=" ", header=False, index=False)
    print(f"Saved hash mapping table of paths to {outfile_path}")


def search_for_readable_file(fname: str, *dir_paths_to_search: str) -> str:
    file_found_at = None
    to_search = [fname] + [os.path.join(dir_path, fname)
                           for dir_path in dir_paths_to_search]
    ix = 0
    while ix < len(to_search) and not file_found_at:
        next_path_to_check = to_search[ix]
        if os.access(next_path_to_check, os.R_OK):
            file_found_at = next_path_to_check
        else:
            ix += 1
    return file_found_at


def show_keys_in(a_dict: Mapping[str, Any],# log:Callable = print,
                 what_keys_are: str = "Local variables",
                 level: int = logging.INFO) -> None:
    """
    :param a_dict: Dictionary mapping strings to anything
    :param log: Function to log/print text, e.g. logger.info or print
    :param what_keys_are: String naming what the keys are
    """
    log(f"{what_keys_are}: {stringify_list(uniqs_in(a_dict))}", level=level)


def stringify_list(a_list: list) -> str:
    """ 
    :param a_list: List (any)
    :return: String of all items in a_list, single-quoted and comma-separated
    """
    result = ""
    if a_list and isinstance(a_list, list):
        list_with_str_els = [str(el) for el in a_list]
        if len(a_list) > 1:
            result = "'{}'".format("', '".join(list_with_str_els))
        else:
            result = list_with_str_els[0]
    return result


def uniqs_in(listlike: Iterable[Hashable]) -> list:
    """
    Get an alphabetized list of unique, non-private local variables' names
    by calling locals() and then passing it into this function
    :param listlike: List-like collection (or dict) of strings
    :return: List (sorted) of all unique strings in listlike that don't start
             with an underscore
    """
    uniqs = set([v if not v.startswith("_") else None
                 for v in listlike]) - {None}
    uniqs = [x for x in uniqs]
    uniqs.sort()
    return uniqs


class UserAWSCreds:
    def __init__(self, cli_args: dict, parser: argparse.ArgumentParser
                 ) -> None:
        """
        _summary_ 
        :param cli_args: dict, _description_
        :param parser: argparse.ArgumentParser to raise an argparse error 
                       immediately if any parameters the user gave are invalid
        """
        self.error = parser.error
        self.host = cli_args["host"]
        self.lens =  {"access": 20, "secret": 40} # AWS s3 key names and their lengths
        self.names = list(self.lens.keys())
        self.keys = self.get_and_validate(cli_args.get("aws_keys"))

    def get_and_validate(self, keys_initial=list()) -> LazyDict:  # , cli_args, parser):
        """
        Get AWS credentials, either from s3info or from the user entering them
        :param keys_initial: List/tuple of user's AWS S3 keys, or list saying
                             "manual" to prompt the user for those keys, or
                             None to try to get keys other ways, or something
                             else to raise an error  
        :return: LazyDict mapping 
        """
        aws_creds = LazyDict()  # Return value
        if keys_initial:
            if keys_initial != ["manual"]:
                if len(keys_initial) != 2:
                    self.error("Please give exactly two keys, your access "
                               "key and your secret key (in that order).") 
                for i in range(len(self.names)):
                    aws_creds[self.names[i]] = keys_initial[i]
        else:
            try:
                aws_creds = s3_get_info()
            except (sp.CalledProcessError, ValueError):
                pass
        for key_name in self.names:
            aws_creds.lazysetdefault(key_name, lambda:
                                     self.get_credential(key_name))
            self.validate_key(key_name, aws_creds[key_name])
        return aws_creds
    
    def validate_key(self, key_name: str, key_to_validate: str) -> None:
        key_len = self.lens[key_name]
        if len(key_to_validate) != key_len:
            self.error(f"Your AWS {key_name} key must be {key_len} "
                       "characters long.")

    def get_s3_client(self): # , host, access_key, secret_key):
        return boto3.session.Session().client(
            "s3", endpoint_url=self.host, aws_access_key_id=self.keys.access,
            aws_secret_access_key=self.keys.secret
        )

    def get_credential(self, cred_name: str, input_fn=getpass) -> str:  # , cli_args):
        """
        If AWS credential was a CLI arg, return it; otherwise prompt user for it 
        :param cred_name: String naming which credential (username or password)
        :param input_fn: Function to get the credential from the user
        :return: String, user's NDA credential
        """
        return input_fn(f"Enter your AWS S3 {cred_name} key: ")  # cli_args.get(cred_name, )


def valid_output_dir(path: Any) -> str:
    """
    Try to make a folder for new files at path; throw exception if that fails
    :param path: String which is a valid (not necessarily real) folder path
    :return: String which is a validated absolute path to real writeable folder
    """
    return validate(path, lambda x: os.access(x, os.W_OK),
                    valid_readable_dir, "Cannot create directory at '{}'", 
                    lambda y: os.makedirs(y, exist_ok=True))


def valid_readable_dir(path: Any) -> str:
    """
    :param path: Parameter to check if it represents a valid directory path
    :return: String representing a valid directory path
    """
    return validate(path, os.path.isdir, valid_readable_file,
                    "Cannot read directory at '{}'")


def valid_readable_file(path: Any) -> str:
    """
    Throw exception unless parameter is a valid readable filepath string. Use
    this, not argparse.FileType('r') which leaves an open file handle.
    :param path: Parameter to check if it represents a valid filepath
    :return: String representing a valid filepath
    """
    return validate(path, lambda x: os.access(x, os.R_OK),
                    os.path.abspath, "Cannot read file at '{}'")


def validate(to_validate: Any, is_real: Callable, make_valid: Callable,
             err_msg: str, prepare:Callable = None):
    """
    Parent/base function used by different type validation functions. Raises an
    argparse.ArgumentTypeError if the input object is somehow invalid.
    :param to_validate: String to check if it represents a valid object 
    :param is_real: Function which returns true iff to_validate is real
    :param make_valid: Function which returns a fully validated object
    :param err_msg: String to show to user to tell them what is invalid
    :param prepare: Function to run before validation
    :return: to_validate, but fully validated
    """
    try:
        if prepare:
            prepare(to_validate)
        assert is_real(to_validate)
        return make_valid(to_validate)
    except (OSError, TypeError, AssertionError, ValueError, 
            argparse.ArgumentTypeError):
        raise argparse.ArgumentTypeError(err_msg.format(to_validate))


def verbosity_is_at_least(verbosity: int) -> bool:
    """
    :param verbosity: Int, the number of times that the user included the
                      --verbose flag when they started running the script.
                      This number corresponds to the logging levels like so:
                      verbosity == 0 corresponds to logging.ERROR(==40)
                      verbosity == 1 corresponds to logging.WARNING(==30)
                      verbosity == 2 corresponds to logging.INFO(==20)
                      verbosity >= 3 corresponds to logging.DEBUG(==10)
    :return: Bool indicating whether the program is being run in verbose mode
    """
    lowest_log_level = max(10, 40 - (10 * verbosity))
    return logging.getLogger().getEffectiveLevel() <= lowest_log_level


if __name__ == "__main__":
    main()
