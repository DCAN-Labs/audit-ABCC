#!/usr/bin/env python3

"""
Utilities for ABCC auditing workflow
Originally written by Anders Perrone
Updated by Greg Conan on 2024-02-07
"""
# Standard imports
import argparse
from collections.abc import Callable, Hashable
from datetime import datetime
import functools
from getpass import getpass
import logging
import os
import pdb
import re
import subprocess as sp
import sys
from typing import (Any, Generator, Iterable, List, Mapping,
                    Optional, Tuple, Union)  # Literal, 
import urllib

# External imports
import boto3
import nibabel as nib
import numpy as np
import pandas as pd

# Constants: Pipeline names, database path, & temporarily hardcoded dirpath
BIDS_COLUMNS = ("bids_subject_id", "bids_session_id")
BIDSPIPELINE = "abcd-bids-pipeline"
DICOM2BIDS = "abcd-dicom2bids"
LOGGER_NAME = "BidsDBLogger"
PATH_ABCD_BIDS_DB = "/home/rando149/shared/projects/ABCC_year2_processing/s3_status_report.csv"
PATH_DICOM_DB = "/home/rando149/shared/code/internal/utilities/abcd-dicom2bids/src/audit/ABCD_BIDS_db.csv"
PATH_NGDR = "/spaces/ngdr/ref-data/abcd/nda-3165-2020-09/"
SESSION_DICT = {'baseline_year_1_arm_1': 'ses-baselineYear1Arm1',
                '2_year_follow_up_y_arm_1': 'ses-2YearFollowUpYArm1',
                '4_year_follow_up_y_arm_1': 'ses-4YearFollowUpYArm1'}

# Column names to split BIDS filename into for each data type (dtype)
DTYPE_2_UNIQ_COLS = {"anat": ["rec-normalized", "run", "Tw"],
                     "dwi":  ["run"],  # TODO is "dwi" needed?
                     "fmap": ["acq", "dir", "run"],
                     "func": ["task", "run"]}


def main():
    pd.set_option('display.max_columns', None)


def attrs_in(an_obj: Any):
    return uniqs_in([attr_name if not attr_name.startswith("_")
                     else None for attr_name in dir(an_obj)])


def build_NGDR_fpath(root_BIDS_dir: str, parent_dirname: str,
                     which_BIDS_file: str) -> str:
    return os.path.join(root_BIDS_dir, "sub-*", "ses-*", parent_dirname,
                        f"sub-*_ses-*_{which_BIDS_file}")


def boolify_and_clean_col(exploded_col: pd.Series) -> pd.Series:
    new_col = exploded_col != ""
    new_col.name = exploded_col[new_col].unique()[0]
    return new_col


def debug_or_raise(an_err: Exception, local_vars: Mapping[str, Any]) -> None:
    """
    :param an_err: Exception
    :param local_vars: Dict<str:obj> mapping variables' names to their values;
                       locals() called from where an_err originated
    :raises an_err: if self.debug is False; otherwise pause to debug
    """
    # vrs = LazyDict(local_vars)
    if in_debug_mode():
        locals().update(local_vars)
        if verbosity_is_at_least(2):
            logging.getLogger(LOGGER_NAME).exception(an_err)  # .__traceback__)
        if verbosity_is_at_least(1):
            show_keys_in(locals())  # , logging.getLogger(LOGGER_NAME).info)
        pdb.set_trace()
    else:
        raise an_err


def default_pop(poppable: Any, key:Optional[Any] = None,
                default:Optional[Any] = None):
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

    
def fix_split_col(qc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Because qc_df's ftq_notes column contains values with commas, it is split
    into multiple columns on import. This function puts them back together.
    :param qc_df: pandas.DataFrame with all QC data
    :return: pandas.DataFrame which is qc_df, but with the last column(s) fixed
    """
    def trim_end_columns(row: pd.Series) -> None:
        """
        Local function to check for extra columns in a row, and fix them
        :param row: pandas.Series which is one row in the QC DataFrame
        :param columns: List of strings where each is the name of a column in
        the QC DataFrame, in order
        :return: N/A
        """
        ix = int(row.name)
        if not pd.isna(qc_df.at[ix, columns[-1]]):
            qc_df.at[ix, columns[-3]] += " " + qc_df.at[ix, columns[-2]]
            qc_df.at[ix, columns[-2]] = qc_df.at[ix, columns[-1]]

    # Keep checking and dropping the last column of qc_df until it's valid
    columns = qc_df.columns.values.tolist()
    last_col = columns[-1]
    while any(qc_df[last_col].isna()):
        qc_df.apply(trim_end_columns, axis="columns")
        print("Dropping '{}' column because it has NaNs".format(last_col))
        qc_df = qc_df.drop(last_col, axis="columns")
        columns = qc_df.columns.values.tolist()
        last_col = columns[-1]
    return qc_df


# TODO Is there a reason that s3_audit.py originally counted runs using range() (see s3_get_bids_anats) instead of just getting them from the file names like the funcs?


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


def get_col_headers_for(dtype: str, df:Optional[pd.DataFrame] = None) -> set:
    # headers = set(headers) if headers else set()
    headers = set()
    prefix, uniq_col = get_header_vars_for(dtype)
    if not is_nothing(df):
        uniqs = df[uniq_col].unique()
        runs = df["run"].unique()
        for run in runs:
            for uniq in uniqs:
                headers.add(make_col_header(f"{prefix}{uniq}", run))
    return headers


def get_ERI_filepath(bids_dir_path: str) -> str:
    return build_NGDR_fpath(os.path.join(bids_dir_path, "sourcedata"),
                            "func", "task-*_run-*"
                            "_bold_EventRelatedInformation.txt")


def get_header_vars_for(dtype: str) -> Tuple[str]:
    return {"func":      ("task-", "task"),
            "anat":      ("T", "Tw"),
            "fmap":      ("", "dir"),  # TODO "FM" for fmap?
            "dwi":       ("", "dtype") # "dtype" -> uniq_col with only 1 value
            }.get(dtype, ("", ""))


def get_subj_ses_dict(subj_ses_QC_DF: pd.DataFrame) -> Mapping[str, Any]:
    """
    :param subj_ses_QC_DF: All df_QC rows for a specific subject and session,
                           but no others. Passed in by DataFrameGroupBy.apply
    :return: list<dict<str:obj>> to add as a row in a new pd.DataFrame
    """
    subject_dict = None  # Return value: None if session df is empty
    pid = subj_ses_QC_DF.pGUID
    ses = subj_ses_QC_DF.EventName
    print(f'Checking {pid} {ses}')
    if not subj_ses_QC_DF.empty:
        ses_df_QC = subj_ses_QC_DF[subj_ses_QC_DF['QC'] == 1.0]
        subject_dict = {'subject': reformat_pGUID(pid),
                        'session': SESSION_DICT[ses]} 
        for t in range(2):
            subject_dict = ImageTypeIn(ses_df_QC, t, is_anat=True
                                       ).add_own_run_counts_to(subject_dict)
        for task in ("rest", "MID", "SST", "nback"):
            subject_dict = ImageTypeIn(ses_df_QC, task, is_anat=False
                                       ).add_own_run_counts_to(subject_dict)
    return subject_dict


def get_tier1_or_tier2_ERI_db_fname(parent_dir: str, tier: int) -> str:
    return os.path.join(parent_dir, f"ERI_tier{tier}_paths_bids_db.csv")


def get_variance_of_each_row_of(dtseries_path: str):
    """
    :param dtseries_path: String, valid path to existing dtseries.nii file
    :return: np.ndarray of floats; each is the variance of a row in the dtseries file
    """
    return load_matrix_from(dtseries_path).var(axis=1)


def invert_dict(a_dict: dict) -> dict:
    return {v: k for k, v in a_dict.items()}


class ImageTypeIn:
    def __init__(self, ses_df: pd.DataFrame, which: str,
                 is_anat: bool) -> None:
        self.is_anat = is_anat
        self.ses_df = ses_df
        self.which = which

    def count_own_rows(self) -> int:
        # Count how many ses_df rows fit this DTypeCol's ses_df.image_description 
        if self.is_anat:
            img_dsc = f"ABCD-{self.which}"
            df = self.get_rows_by_img_desc(f"{img_dsc}-NORM")
            if df.empty:
                df = self.get_rows_by_img_desc(img_dsc)
        else:
            swapper = TaskNames()
            df = self.get_rows_by_img_desc(swapper.swap(self.which))
        return df.drop_duplicates(subset='image_file', keep='first').shape[0]

    def get_rows_by_img_desc(self, img_dsc: str) -> pd.DataFrame:
        # Select own subset of the dataframe
        return self.ses_df[self.ses_df["image_description"] == img_dsc]

    def add_own_run_counts_to(self, subject_dict: dict) -> dict:
        for run_num in range(0, self.count_own_rows()):
            subject_dict[make_col_header(self.which, run_num + 1)] = 'no bids'
        return subject_dict


def is_truthy(a_value: Any) -> bool:
    result = bool(a_value)
    # if not a_value:
    #     result = False
    if result and isinstance(a_value, float):
        result = not np.isnan(a_value)
    return result  # False if not a_value else (not np.isnan(a_value))


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


def in_debug_mode() -> bool:
    """
    :return: Boolean indicating whether or not the program is being run in
             debug mode
    """
    return verbosity_is_at_least(2)  # TODO
    """
    try:
        currently_in_debug_mode = DEBUG
    except NameError as e:
        if globals().get("DEBUG") is None:
            show_keys_in(locals())
            pdb.set_trace()
        else:
            currently_in_debug_mode = globals().get("DEBUG")
    return currently_in_debug_mode
    """
    

def is_nothing(thing: Any) -> bool:
    if thing is None:
        result = True
    elif isinstance(thing, float):
        result = np.isnan(thing)
    elif hasattr(thing, "empty"):  # get_module_name_of(thing) == "pandas":
        result = thing.empty
    else:
        result = not thing
    return result


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


def run_num_to_str(run_num:Optional[float] = 1):
    return f"run-{int(run_num):02d}"


def make_col_header(prefix: str, run_num:Optional[float] = 1) -> str:
    try:  
        if is_nothing(run_num):
            run_num = 1
        return f"{prefix}_{run_num_to_str(run_num)}"
                # if isnt_nothing(run_num) else prefix)  # TODO Ensure run number added
    except ValueError as e:
        debug_or_raise(e, locals())


def make_col_header_from(row: pd.Series, dtype: str, uniq=None) -> str:
    if uniq:
        uniq_col = uniq
        prefix, _ = get_header_vars_for(dtype)
    else:
        prefix, uniq_col = get_header_vars_for(dtype)
    try:
        return make_col_header(f"{prefix}{row.get(uniq_col)}", row.get("run"))
    except ValueError as e:
        debug_or_raise(e, locals())
    

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
        # (sub-NDARINV.{8})(?:_)(ses-.*?)(?:_)(?:acq-)?(.*?)(?:_)?(?:dir-)(.*?)?(?:_)(?:run-)?([0-9]{2})?
        "fmap": ("subj", "-_", "ses", "-_", "acq", "-_?", "dir", "?", "-_",
                 "run", "-_?"),

        # BIDS-valid file name -> tuple of 4 strings: subject ID, session ID,
        # task name, acquisition name/ID if any, and run number if any
        "func": ("subj", "-_", "ses", "-_?", "task", "-_", # "acq", "?", "-_?",
                 "run", "-_?", "nothing else")
    })

    def __init__(self, *pattern_names: str) -> None:
        # if self.SPLIT.keys()
        for pattern in pattern_names:
            self[pattern] = self.create(*self.SPLIT.lazyget(pattern, list))

    def create(self, *args: str) -> re.Pattern:
        """
        :return: re.Pattern to use for splitting a string into desired details
        """
        return re.compile("".join([self.FIND.get(key, key) for key in args]))
    

def explode_col(ser: pd.Series, re_patterns: RegexForBidsDetails, dtype: str) -> list:
    try:
        num_new_cols = DTYPE_2_UNIQ_COLS.get(dtype)
        exploded = ser.str.findall(re_patterns[dtype]
                                   ).explode(ignore_index=True)
        # exploded[exploded.isna()] = [''] * 5
        ixs_NaNs = exploded.isna()
        if ixs_NaNs.any():
            # assert exploded[~ixs_NaNs].any()
            # num_new_cols = len(exploded[~ixs_NaNs].iloc[0])
            if is_nothing(num_new_cols):
                num_new_cols = len(exploded[~ixs_NaNs].iloc[0])
            exploded[ixs_NaNs] = exploded[ixs_NaNs].apply(
                lambda _: [np.nan] * num_new_cols
            )
    except (AssertionError, AttributeError, KeyError, ValueError) as e:
        debug_or_raise(e, locals())
    return exploded.to_list()


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


def show_keys_in(a_dict: Mapping[str, Any],# log:Callable = print,
                 what_keys_are:str = "Local variables",
                 level:int = logging.INFO) -> None:
    """
    :param a_dict: Dictionary mapping strings to anything
    :param log: Function to log/print text, e.g. logger.info or print
    :param what_keys_are: String naming what the keys are
    """
    log(f"{what_keys_are}: {stringify_list(uniqs_in(a_dict))}", level=level)


def split_into_anats_and_funcs(s3_data: list) -> Mapping[str, list]:
    try:
        funcs = set()
        anats_T = {1: set(), 2: set()}
        for obj in s3_data:  # TODO OPTIMIZE by making s3_data a pd.DataFrame?
            key = urllib.parse.unquote(obj["Key"])
            parts = key.split("_")
            
            # Only get .nii.gz file name keys
            last_part = parts[-1].split(".")
            if last_part[-2] == "nii" and last_part[-1] == "gz":
            # if parts[-1].endswith(".nii.gz"):
                # dtype = {"bold": "func", "T1w": "anat", "T2w": "anat"}.get(last_part[0])
                # if dtype == "anat":
                if last_part[0] == "bold":  # TODO Optimize
                    funcs.add(get_func_name(key))
                elif last_part[0] == "T1w":
                    anats_T[1].add(get_anat_name(key, 1))
                elif last_part[0] == "T2w":
                    anats_T[2].add(get_anat_name(key, 2))

        return {"anat": [*anats_T[1], *anats_T[2]], "func": list(funcs)}
    except KeyError:
        return


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


class TaskNames:
    def __init__(self) -> None:
        self.all = LazyDict({"task-MID": "ABCD-MID-fMRI",
                             "task-SST": "ABCD-SST-fMRI",
                             "task-nback": "ABCD-nBack-fMRI",
                             "task-rest": "ABCD-rsfMRI"}) 
        self.inv = invert_dict(self.all)

    def get_all(self, is_img_dsc: bool = False):
        to_abbreviate = self.inv if is_img_dsc else self.all
        return [x for x in to_abbreviate.keys()]

    def get_all_abbreviated(self, is_img_dsc: bool = False):
        return [self.abbreviate(x) for x in self.get_all(is_img_dsc)]
    
    def abbreviate(self, to_abbreviate: str) -> str:
        abbreviated = to_abbreviate.split("-")[1]
        return {"rsfMRI": "rs"}.get(abbreviated, abbreviated)

    def swap(self, to_swap: str) -> str:
        return self.all.get(to_swap, self.inv.get(to_swap, to_swap))


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


# def unquote(a_str: str) -> str: return urllib.parse.unquote(a_str)


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
        :return: Dictionary mapping 
        """
        aws_creds = LazyDict()  # Return value
        if keys_initial:
            if keys_initial != ["manual"]:
                if len(keys_initial) != 2:
                    self.error("Please give exactly two keys, your access "
                               "key and your secret key (in that order).") 
                # self.validate(keys_initial)
                for i in range(len(self.names)):
                    aws_creds[self.names[i]] = keys_initial[i]
        else:
            try:
                aws_creds = s3_get_info()
                # aws_keys = [aws_creds.get(key_name) for key_name in self.names]
            except (sp.CalledProcessError, ValueError):
                pass
        for key_name in self.names:
            aws_creds.lazysetdefault(key_name, lambda:
                                     self.get_credential(key_name))
            self.validate_key(key_name, aws_creds[key_name])
            # self.validate([aws_creds["access"], aws_creds["secret"]])
            # aws_creds[f"{key_name}_key"] = aws_creds.pop(key_name)
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


def validate_dtseries(dtseries_path: str) -> str: 
    variances = get_variance_of_each_row_of(dtseries_path)
    # invalid_rows = variances.
    """
    invalid_rows = 0
    for eachvar in variances:
        if not eachvar:
            invalid_rows += 1
    if invalid_rows > THRESHOLD: pass  
    """ # TODO


def verbosity_is_at_least(verbosity: int) -> bool:
    """
    :param verbosity: Int, the number of times that the user included the
                      --verbose flag when they started running the script.
                      This number corresponds to the logging levels like so:
                      verbosity 0: logging.ERROR == 40
                      verbosity 1: logging.WARNING == 30
                      verbosity 2: logging.INFO == 20
                      verbosity 3+: logging.DEBUG == 10
    :return: Boolean indicating whether or not the program is being run in
             verbose mode
    """
    lowest_log_level = max(10, 40 - (10 * verbosity))
    return logging.getLogger().getEffectiveLevel() <= lowest_log_level


if __name__ == "__main__":
    main()
