#!/usr/bin/env python3

"""
Utilities for ABCC auditing workflow
Originally written by Anders Perrone
Updated by Greg Conan on 2024-04-10
"""
# Standard imports
import argparse
from collections.abc import Callable, Hashable
from datetime import datetime
import dateutil
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
                    Mapping, Optional, Set, Tuple, Union)

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
IMG_DSC_2_COL_HDR_FNAME = "img_desc_to_header_1_fmap_col.json"  # "img_desc_to_header_different_fmap_cols.json"
LOGGER_NAME = __name__  # "BidsDBLogger"
PATH_NGDR = "/spaces/ngdr/ref-data/abcd/nda-3165-2020-09/"

# Shorthand names for the different kinds of BIDS DBs
WHICH_DBS = ("ftqc", "tier1", "s3")


def main():  # Display all columns in a Pandas DataFrame
    pd.set_option("display.max_columns", None)


def attrs_in(an_obj: Any) -> List[str]:
    """
    Get anything's public attributes. Convenience function for debugging.
    :param an_obj: Any
    :return: List of strings naming every public attribute in an_obj
    """
    return uniqs_in([attr_name for attr_name in list(dir(an_obj))
                     if not attr_name.startswith("_")])


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


class Debuggable:  # I put the debugger function in a class so it can use its
                   # implementer classes' self.debugging variable
    def debug_or_raise(self, an_err: Exception, local_vars: Mapping[str, Any]
                       ) -> None:
        """
        :param an_err: Exception (any)
        :param local_vars: Dict[str, Any] mapping variables' names to their 
                           values; locals() called from where an_err originated
        :raises an_err: if self.debugging is False; otherwise pause to debug
        """
        if self.debugging:
            debug(an_err, local_vars)
        else:
            raise an_err


class ColHeaderFactory(Debuggable):
    # 
    DTYPE_2_PREFIX_AND_UNIQ = {"func": ("task-", "task"),
                               "anat": ("T", "Tw"),
                               "fmap": ("FM_", "dir"),
                               "dwi":  ("", "dwi")} # "dtype" -> uniq_col with only 1 value
    ACQ_2_FM = {"func": "fMRI_", "dwi": "Diffusion_"}
    

    def __init__(self, debugging: bool = False, will_add_run: bool = False,
                 dtype: Optional[str] = None, run_col: Optional[str] = "run",
                 uniq_col: Optional[str] = None) -> None:  #, prefix: Optional[str] = ""
        """
        _summary_ 
        :param debugging: True to pause & interact on error or False to crash
        :param will_add_run: True to include run number in column headers; 
                             otherwise (and by default) False
        :param dtype: Optional[str], data type, a key in DTYPE_2_UNIQ_COLS.
                      Exclude it to create dtype-agnostic column headers.
        :param run_col: Optional[str],_description_, defaults to "run"
        :param uniq_col: Optional[str],_description_, defaults to None
        """
        self.will_add_run = will_add_run
        self.debugging = debugging
        self.dtype = dtype
        self.run_col = run_col
        
        if dtype:  # and ((not prefix) or (not uniq_col)):
            self.prefix, self.uniq_col = \
                self.get_prefix_and_uniq_col_name_from(dtype)
        else:
            self.prefix = ""
        if uniq_col:
            self.uniq_col = uniq_col


    def add_run_to(self, hdr_base: str, run_num: float) -> str:
        """
        :param hdr_base: str to append run number onto
        :param run_num: float, run number (preferably a positive integer)
        :return: str, hdr_base + "_run-" + 2-digit (0-padded) run_num, or
                 just hdr_base if run_num is NaN
        """
        try:
            return (hdr_base if float_is_nothing(run_num) else
                    hdr_base + stringify_run_num(run_num))
        except ValueError as e:
            self.debug_or_raise(e, locals())


    def get_all_for(self, dtype: Optional[str] = None,
                    df: Optional[pd.DataFrame] = None) -> Set[str]:
        """
        :param dtype: Optional[str], data type, a key in DTYPE_2_UNIQ_COLS.
                      Exclude it to either use the dtype string given at
                      instantiation or create dtype-agnostic column headers.
        :param df: pd.DataFrame to get data from to create column header
        :return: Set[str] of all BidsDB.df column header strings for dtype
        """
        headers = set()
        if not dtype:
            dtype = self.dtype
        if not df_is_nothing(df):
            prefix, uniq_col = self.get_prefix_and_uniq_col_name_from(dtype)

            runs = df[self.run_col].unique()
            
            uniqs = [dtype] if uniq_col == "dwi" else df[uniq_col].unique()
            acqs = df["acq"].unique() if dtype == "fmap" else [""]

            for uniq in uniqs:
                for acq in acqs:
                    hdr_base = f"{prefix}{self.ACQ_2_FM.get(acq, '')}{uniq}"
                    for run in runs:
                        headers.add(self.add_run_to(hdr_base, run))
                        
        headers.discard(np.nan)
        return headers


    def get_all_simply_for(self, dtype: Optional[str] = None,
                           df: Optional[pd.DataFrame] = None) -> Set[str]:
        """
        Annoyingly, this is ~20x slower than get_all_for despite seeming
        much more elegant than get_all_for's triply nested for loop.
        :param dtype: Optional[str],_description_, defaults to None
        :param df: Optional[pd.DataFrame],_description_, defaults to None
        :return: Set[str] of all BidsDB.df column header strings for dtype
        """
        headers = set()
        if not dtype:
            dtype = self.dtype
        if not df_is_nothing(df):
            uniq_cols = DTYPE_2_UNIQ_COLS.get(dtype)
            if dtype == "anat":
                uniq_cols = uniq_cols[1:]

            df_1_row_per_hdr = df.groupby(uniq_cols).head(1)
            headers = set(df_1_row_per_hdr.apply(self.new_from_row, axis=1))

        headers.discard(np.nan)
        return headers


    @classmethod
    def get_prefix_and_uniq_col_name_from(cls, dtype: str) -> Tuple[str]:
        """
        Given a dtype, get both of the other variables needed to make a
        BidsDB.df column header: the header prefix and the name of the
        BidsDB.df column with values uniquely identifying each row
        :param dtype: str, data type, a key in DTYPE_2_UNIQ_COLS
        :return: Tuple[str] with 2 elements: [0] prefix and [1] uniq_col.
        """
        return cls.DTYPE_2_PREFIX_AND_UNIQ.get(dtype, ("", "")) 


    def new_from_row(self, row: pd.Series, uniq_col: Optional[str] = None,
                     include_non_run: bool = False) -> str:
        """
        _summary_ 
        :param row: pd.Series, _description_
        :param uniq_col: Optional[str],_description_, defaults to None
        :param include_non_run: True to return a column header string even if
                                row has no run number; else (by default) False
        :return: str, _description_
        """
        if not uniq_col:
            uniq_col = self.uniq_col
        # insert Diffusion or fMRI based on acq-[func/dwi] in the filename 
        acq_hdr = (self.ACQ_2_FM.get(row.get("acq", ""), "")
                   if row.get("dtype") == "fmap" else "")
        hdr_base = f"{self.prefix}{acq_hdr}{row.get(uniq_col, '')}"

        # Add run number if the row has one
        run = row.get(self.run_col)
        if not float_is_nothing(run):
            hdr = hdr_base + stringify_run_num(run)

        # If it doesn't, return what the caller specified via include_non_run
        elif include_non_run:
            hdr = hdr_base
        else:
            hdr = None
        return hdr


class DataFrameQuery(Debuggable):
    def __init__(self) -> None:
        pass  # Unused so calling DataFrameQuery(...) returns a pd.DataFrame


    def __new__(self, df: pd.DataFrame, will_clean: bool = False,
                **conditions: Any) -> pd.DataFrame:
        """
        :param df: pd.DataFrame to query/search (BidsDB.df)
        :param will_clean: True to return a transposed dataframe without
                           Nones or NaNs, to be convenient and readable
        :param conditions: Dict[str, Any] mapping df column names to the
                           values to find in those columns
        :return: pd.DataFrame, df subset containing only rows matching 
                 the given conditions
        """
        self.conditions = list()
        self.df = df
        try:
            for col_name, col_value in conditions.items():
                self.add(self, col_name, col_value)
        except (TypeError) as e:
            self.debug_or_raise(self, e, locals())
        self.df.query(" & ".join(self.conditions), inplace=True)
        if will_clean:
            self.clean()
        return self.df
   

    def add(self, col_name: str, col_value: Any) -> None:
        """
        The pd.DataFrame returned from calling this DataFrameQuery will only
        include rows with a {col_name} value of {col_value}.
        :param col_name: String naming a column or index of self.df to filter
        :param col_value: Any value to find in self.df  
        """
        if col_name in self.df.index.names:
            col_value = f"'{col_value}'"
        else:
            col_name = f"`{col_name}`"
        self.conditions.append(f"{col_name} == {col_value}")


    def clean(self) -> None:
        """
        Transpose self.df and remove its Nones/NaNs so it becomes more
        convenient and readable, especially in pdb.set_trace calls
        """
        df = self.df.T
        self.df = df[~df.isna()]


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
        to_return = poppable.pop() if key is None else poppable.pop(key)
    except (AttributeError, IndexError):
        to_return = default
    return to_return


def dt_format(moment: datetime) -> str:
    """
    :param moment: datetime, a specific moment
    :return: String, that moment in "YYYY-mm-dd_HH-MM-SS" format
    """
    return moment.strftime("%Y-%m-%d_%H-%M-%S")


def fill_run(ser):
    """
    _summary_ 
    :param ser: _type_, _description_
    :return: _type_, _description_
    """
    run_nums = ser.dropna()
    if not run_nums.empty:
        if run_nums.shape[0] == 1:
            ser = run_nums.iloc[0]
        else:
            pdb.set_trace()
    return ser


def get_most_recent_FTQC_fpath(incomplete_dirpath_FTQC: str) -> str:
    """
    :param incomplete_dirpath_FTQC: Fast Track QC directory paths with "{}"
                                    in place of any differences between them
    :return: str, valid path to most recent readable Fast Track QC
             spreadsheet text file, or None if no readable FTQC files found
    """
    COLS = LazyDict(fpath="fpath", DT="datetimestamp", can_read="is_readable")

    # Get all readable abcd_fastqc01.txt file paths
    ftqc_paths = pd.DataFrame({COLS.fpath:
                               glob(incomplete_dirpath_FTQC.format("*"))})
    ftqc_paths[COLS.can_read] = ftqc_paths[COLS.fpath].apply(
        lambda fpath: os.access(fpath, os.R_OK)
    )
    # ftqc_paths.drop(index=~ftqc_paths["readable"], inplace=True)
    ftqc_paths = ftqc_paths[ftqc_paths[COLS.can_read]]

    # If there are no readable abcd_fastqc01.txt file paths, return None
    if ftqc_paths.empty:
        most_recent_FTQC = None
    else:

        # Get the datetimestamp (int) from each abcd_fastqc01.txt file name
        prefix, suffix = incomplete_dirpath_FTQC.split("{}")
        ftqc_paths[COLS.DT] = \
            ftqc_paths[COLS.fpath].str.strip(prefix).str.rstrip(suffix)
        ftqc_paths[COLS.DT] = ftqc_paths[COLS.DT].astype(int)

        # Return the path to the most recent abcd_fastqc01.txt file
        most_recent_FTQC = ftqc_paths.loc[ftqc_paths[COLS.DT].idxmax(),
                                          COLS.fpath]
    return most_recent_FTQC


def get_ERI_filepath(bids_dir_path: str) -> str:
    """
    :param bids_dir_path: str, valid path to BIDS root directory
    :return: str, globbable incomplete path to EventRelatedInformation file
    """
    return build_NGDR_fpath(os.path.join(bids_dir_path, "sourcedata"),
                            "func", "task-*_run-*_bold_"
                            "EventRelatedInformation.txt")


def get_tier1_or_tier2_ERI_db_fname(parent_dir: str, tier: int) -> str:
    """
    :param parent_dir: str, path to directory with an ERI database file
    :param tier: int, 1 to mean local storage (particularly MSI NGDR) or 2 to
                 mean cloud storage (particularly AWS s3)
    :return: str, path to database of EventRelatedInformation file paths
    """
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
    

class ImgDscColNameSwapper:  # (LazyDict, Debuggable): 
    def __init__(self, json_fpath: Optional[str] = None) -> None:
        """
        Class to create image_description strings, create BidsDB.df header
        column strings, and convert one into the other. Each image_description
        is a substring of `ftq_series_id`s in the FastTrackQC spreadsheet:
        everything between the subject/session IDs and the date-time-stamp.
        :param json_fpath: Optional[str], valid path to existing .JSON file
        """
        # Read in the .JSON file mapping FastTrackQC image_description strings
        # to names of their respective final BidsDB.df column header strings
        self.fpath = self.find_fpath(json_fpath)
        self.dsc2hdr = extract_from_json(self.fpath)

        # Map column header names to image_description strings
        self.hdr2dsc = invert_dict(self.dsc2hdr,
                                   keep_collision=True)
        self.hdr2dsc.pop(None, None)

        # Map each header column name and image_description to its dtype
        self.dtype_of = LazyDict()
        HDR_PFX_2_DTYPE = {"T1": "anat", "T2": "anat", "task": "func",
                           "FM": "fmap", "dwi": "dwi"}
        SPLIT_BY = re.compile("-|_")  # image_description delimiters
        for dsc, hdr in self.dsc2hdr.items():
            self.dtype_of[hdr] = HDR_PFX_2_DTYPE.get(SPLIT_BY.split(hdr, 1)[0]
                                                     ) if hdr else None
            self.dtype_of[dsc] = self.dtype_of[hdr]


    def find_fpath(self, json_fpath: Optional[str] = None) -> str:
        """
        :param json_fpath: Optional[str], .JSON file path if provided
        :return: str, valid path to existing .JSON file mapping FastTrackQC
                 image_descriptions to their BidsDB.df column headers
        """
        to_check = list()
        if json_fpath:
            dir_to_check, fname = os.path.split(json_fpath)
            to_check = [dir_to_check]
        else:
            fname = IMG_DSC_2_COL_HDR_FNAME
            to_check = list()
        to_check += [os.path.dirname(sys.argv[0]),
                     os.path.dirname(__file__), os.getcwd()]
        return search_for_readable_file(fname, *to_check)
        

    def to_header_col(self, img_desc: str) -> str:
        """
        :param img_desc: str, image_description from FastTrackQC spreadsheet;
        :return: str, header column in final BidsDB.df
        """
        return self.dsc2hdr.get(img_desc)
    

    def to_image_desc(self, header_col_name: str) -> str:
        """
        :param header_col_name: str, header column in final BidsDB.df
        :return: str, image_description from FastTrackQC spreadsheet
        """
        return self.hdr2dsc.get(header_col_name)


def iter_attr(an_obj: Any) -> Generator[str, None, None]:
    """
    Iterate over an object's attributes. Convenience function for debugging.
    :yield: Generator[str, None, None] to iterate over an_obj attributes
    """
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


    def lazyget(self, key: Hashable, get_if_absent:
                Optional[Callable] = lambda: None) -> Any:
        """
        LazyButHonestDict.lazyget from stackoverflow.com/q/17532929
        :param key: Object (hashable) to use as a dict key
        :param get_if_absent: function that returns the default value
        :return: _type_, _description_
        """
        return self[key] if self.get(key) is not None else get_if_absent()
    

    def lazysetdefault(self, key: Hashable, get_if_absent:
                       Optional[Callable] = lambda: None) -> Any:
        """
        LazyButHonestDict.lazysetdefault from stackoverflow.com/q/17532929 
        :param key: Object (hashable) to use as a dict key
        :param get_if_absent: function that returns the default value
        :return: _type_, _description_
        """
        return (self[key] if self.get(key) is not None else
                self.setdefault(key, get_if_absent()))
    
    # def subset(self, *keys: Hashable) -> "LazyDict":
    #     return LazyDict({key: self.get(key) for key in keys})


def is_nan(thing: Any) -> bool:
    """
    :return: True if thing is np.nan (NaN), otherwise False, raising no error
    """
    try:
        thing_is_nan = np.isnan(thing)
    except TypeError:
        thing_is_nan = False
    return thing_is_nan


def float_is_nothing(thing: Optional[float]) -> bool:
    """
    :return: True if thing is falsy or NaN, otherwise False
    """
    return thing is None or is_nan(thing) or not thing


def load_matrix_from(matrix_path: str):
    """
    :param matrix_path: String, the absolute path to a .gii or .nii matrix file
    :return: numpy.ndarray matrix of data from the matrix file
    """
    return {".gii": lambda x: x.agg_data(),
            ".nii": lambda x: np.array(x.get_fdata().tolist()),
            }[os.path.splitext(matrix_path)[-1]](nib.load(matrix_path))


def log(msg: str, level: int = logging.INFO):
    """
    _summary_ 
    :param msg: str, _description_
    :param level: int,_description_, defaults to logging.INFO
    """
    SplitLogger.logAtLevel(level, msg)


def stringify_run_num(run_num: Union[int, float]) -> str:
    """
    :param run_num: Union[int, float], run number (ideally positive integer)
                    to turn into a string
    :return: str, "_run-" + 2-digit (0-padded) run_num
    """
    return f"_run-{int(run_num):02d}"
    

def make_default_out_file_name(key_DB: str):
    """
    _summary_ 
    :param key_DB: str, _description_
    :return: _type_, _description_
    """
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
                        "EventRelatedInformation.txt")  # {task}{stringify_run_num(run)}_


def mutual_reindex(*dfs: pd.DataFrame, fill_value:
                   Optional[Hashable] = np.nan) -> List[pd.DataFrame]:
    """
    This one DOES accept an empty dfs list.
    Combine all dataframes' indices so each has every subject session, and
    fill missing values (in 1 df but not the other) with np.nan (NaN) 
    :param fill_value: Optional[Hashable],_description_, defaults to np.nan
    :return: List[pd.DataFrame], the same dfs but now sharing indices, or
             an empty list if no dfs were given 
    """
    return mutual_reindex_dfs(*dfs, fill_value=fill_value
                              ) if len(dfs) > 1 else dfs


def mutual_reindex_dfs(*dfs: pd.DataFrame, fill_value:
                       Optional[Hashable] = np.nan) -> List[pd.DataFrame]:
    """
    This one DOES NOT accept an empty dfs list.
    Combine all dataframes' indices so each has every subject session, and
    fill missing values (in 1 df but not the other) with np.nan (NaN) 
    :param fill_value: Optional[Hashable],_description_, defaults to np.nan
    :return: List[pd.DataFrame], the same dfs but now sharing indices
    """
    combined_ixs = functools.reduce(lambda x, y: x.union(y),
                                    [df.index for df in dfs])
    return [df.reindex(combined_ixs, fill_value=fill_value) for df in dfs]


def reformat_pGUID(pguid: str) -> str:
    """
    :param pguid: str, _description_
    :return: str, now BIDS-valid subject ID
    """
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
        "dir": r"(dir-.*?)",     # Get all text from "dir-" until next group
        "nothing else": r"(?:.*)",  # Exclude everything from here onward
        "rec": r"(rec-.*?)?",    # Get any text from "rec-" until next group
        "run": r"(?:run-)?([0-9]{2})?",  # Get 2-digit run number
        "s3": r"(s3)?",          # Get the string "s3" if it's present
        "ses": r"(ses-.*?)",     # Get all text from "ses-" until next group
        "subj": r"(sub-NDARINV.{8})",  # Get all text from "sub-" until next
        "task": r"(?:task-)(.*?)",  # Get all text after "task-" until next
        "tier1": r"(tier1)?",    # Get the string "tier1" if it's present
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
        """
        This class is just somewhere to put all of the Regex patterns I use
        without constantly rebuilding them every time I need to use them.
        """
        for pattern in pattern_names:
            self[pattern] = self.create(*self.SPLIT.lazyget(pattern, list))


    def create(self, *args: str) -> re.Pattern:
        """
        :return: re.Pattern to use for splitting a string into desired details
        """
        return re.compile("".join([self.FIND.get(key, key) for key in args]))
    

def explode_col(ser: pd.Series, re_patterns: RegexForBidsDetails, dtype: str,
                debugging: bool = False) -> list:
    """
    _summary_ 
    :param ser: pd.Series, _description_
    :param re_patterns: RegexForBidsDetails, _description_
    :param dtype: str, data type, a key in DTYPE_2_UNIQ_COLS
    :param debugging: True to pause & interact on error or False to crash
    :raises e: _description_
    :return: list, _description_
    """
    try:
        new_cols = DTYPE_2_UNIQ_COLS.get(dtype)
        num_new_cols = len(new_cols) if new_cols else None
        exploded = ser.str.findall(re_patterns[dtype]
                                   ).explode(ignore_index=True)
        ixs_NaNs = exploded.isna()
        if ixs_NaNs.any():
            if not num_new_cols:
                num_new_cols = len(exploded[~ixs_NaNs].iloc[0])
            exploded[ixs_NaNs] = exploded[ixs_NaNs].apply(
                lambda _: [np.nan] * num_new_cols
            )
    except (AttributeError, KeyError, ValueError) as e:
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
    """
    _summary_ 
    :param fname: str, _description_
    :return: str, _description_
    """
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


def show_keys_in(a_dict: Mapping[str, Any],  # log: Callable = print,
                 what_keys_are: str = "Local variables",
                 level: int = logging.INFO) -> None:
    """
    :param a_dict: Dictionary mapping strings to anything
    :param log: Function to log/print text, e.g. logger.info or print
    :param what_keys_are: String naming what the keys are
    """
    log(f"{what_keys_are}: {stringify_list(uniqs_in(a_dict))}", level=level)


class ShowTimeTaken:
    def __init__(self, doing_what: str, show: Callable = log) -> None:
        """
        Context manager to time and log the duration of any block of code 
        :param doing_what: String describing what is being timed
        :param show: Function to print/log/show messages to the user
        """
        self.doing_what = doing_what
        self.show = show


    def __call__(self):
        pass


    def __enter__(self):
        """
        Log the moment that script execution enters the context manager and
        what it is about to do. 
        """
        self.show(f"Just started {self.doing_what}")
        self.start = datetime.now()
        return self
    

    def __exit__(self, exc_type: Optional[type] = None,
                 exc_val: Optional[BaseException] = None, exc_tb=None):
        """
        Log the moment that script execution exits the context manager and
        what it just finished doing. 
        :param exc_type: Exception type
        :param exc_val: Exception value
        :param exc_tb: Exception traceback
        """
        self.elapsed = datetime.now() - self.start
        self.show(f"\nTime elapsed {self.doing_what}: {self.elapsed}")


class SplitLogger(logging.getLoggerClass()): 
    # Container class for message-logger and error-logger ("split" apart)
    FMT = "\n%(levelname)s %(asctime)s: %(message)s"
    LVL = LazyDict(OUT={logging.DEBUG, logging.INFO},
                   ERR={logging.CRITICAL, logging.ERROR, logging.WARNING})
    NAME = LOGGER_NAME  # "BidsDBLogger"


    def __init__(self, verbosity: int, out_fpath: Optional[str] = None,
                 err_fpath: Optional[str] = None) -> None:
        """
        Make logger to log status updates, warnings, and other important info.
        SplitLogger can log errors/warnings/problems to one stream/file and
        log info/outputs/messages to a different stream/file.
        :param verbosity: Int, the number of times that the user included the
                          --verbose flag when they started running the script.
        :param out_fpath: Valid path to text file to write output logs into
        :param err_fpath: Valid path to text file to write error logs into
        """  # TODO stackoverflow.com/a/33163197 ?
        super().__init__(self.NAME, level=verbosity_to_log_level(verbosity))
        self.addSubLogger("out", sys.stdout, out_fpath)
        self.addSubLogger("err", sys.stderr, err_fpath)


    def addSubLogger(self, sub_name: str, log_stream,
                     log_file_path: Optional[str] = None):
        """
        Make a child Logger to handle one kind of message (namely err or out) 
        :param name: String naming the child logger, accessible as
                     self.getLogger(f"{self.NAME}.{sub_name}")
        :param log_stream: io.TextIOWrapper, namely sys.stdout or sys.stderr
        :param log_file_path: Valid path to text file to write logs into
        """
        sublogger = self.getChild(sub_name)
        sublogger.setLevel(self.level)
        handler = (logging.FileHandler(log_file_path, encoding="utf-8")
                   if log_file_path else logging.StreamHandler(log_stream))
        handler.setFormatter(logging.Formatter(fmt=self.FMT))
        sublogger.addHandler(handler)


    @classmethod
    def logAtLevel(cls, level: int, msg: str) -> None:
        """
        Log a message, using the sub-logger specific to that message's level 
        :param level: logging._levelToName key; level to log the message at
        :param msg: String, the message to log
        """
        logger = logging.getLogger(cls.NAME)
        if level in cls.LVL.ERR:
            sub_log_name = "err"
        elif level in cls.LVL.OUT:
            sub_log_name = "out"
        sublogger = logger.getChild(sub_log_name)
        sublogger.log(level, msg)


def stringify_list(a_list: list) -> str:
    """ 
    :param a_list: List (any)
    :return: String containing all items in a_list, single-quoted and
             comma-separated if there are multiple
    """
    result = ""
    if a_list and isinstance(a_list, list):
        list_with_str_els = [str(el) for el in a_list]
        if len(a_list) > 1:
            result = "'{}'".format("', '".join(list_with_str_els))
        else:
            result = list_with_str_els[0]
    return result


def checkout(df):
    """
    Convenience function for debugging 
    :param df: _type_, _description_
    """
    if df.empty:
        log(df.index.values)
    elif df.shape[0] > 1:# and not df[df["image_description"]=="ABCD-rsfMRI"].empty:
        log("Checking out df")
        pdb.set_trace()
        log("Done")


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
    # AWS s3 key names and their lengths
    NAMES = ["access", "secret"]
    LENS = {key: keylen for key, keylen in zip(NAMES, [20, 40])}


    def __init__(self, cli_args: Mapping[str, Any],
                 parser: argparse.ArgumentParser) -> None:
        """
        _summary_ 
        :param cli_args: Mapping[str, Any] of command-line input arguments
        :param parser: argparse.ArgumentParser to raise an argparse error 
                       immediately if any parameters the user gave are invalid
        """
        self.error = parser.error
        self.host = cli_args["host"]
        self.keys = self.get_and_validate(default_pop(cli_args, "aws_keys")) #cli_args.get("aws_keys"))

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
                for i in range(len(self.NAMES)):
                    aws_creds[self.NAMES[i]] = keys_initial[i]
        else:
            try:
                aws_creds = s3_get_info()
            except (sp.CalledProcessError, ValueError):
                pass
        for key_name in self.NAMES:
            aws_creds.lazysetdefault(key_name, lambda:
                                     self.get_credential(key_name))
            self.validate_key(key_name, aws_creds[key_name])
        return aws_creds
    
    def validate_key(self, key_name: str, key_to_validate: str) -> None:
        key_len = self.LENS[key_name]
        if len(key_to_validate) != key_len:
            self.error(f"Your AWS {key_name} key must be {key_len} "
                       "characters long.")

    def get_s3_client(self) -> boto3.session.Session.client: # , host, access_key, secret_key):
        ses = boto3.session.Session()

        # Speed up datetime parsing to speed up listing boto3 bucket contents
        # See https://adobke.com/blog/posts/speeding-up-boto3-list-objects/
        parser_factory = ses._session.get_component("response_parser_factory")
        parser_factory.set_parser_defaults(
            timestamp_parser=dateutil.parser.isoparse
        )
        
        return ses.client(
            "s3", endpoint_url=self.host, aws_access_key_id=self.keys.access,
            aws_secret_access_key=self.keys.secret
        )

    def get_credential(self, cred_name: str, input_fn: Callable = getpass
                       ) -> str:  # , cli_args):
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


def verbosity_to_log_level(verbosity: int) -> int:
    """
    :param verbosity: Int, the number of times that the user included the
                      --verbose flag when they started running the script.
    :return: Level for logging, corresponding to verbosity like so:
             verbosity == 0 corresponds to logging.ERROR(==40)
             verbosity == 1 corresponds to logging.WARNING(==30)
             verbosity == 2 corresponds to logging.INFO(==20)
             verbosity >= 3 corresponds to logging.DEBUG(==10)
    """
    return max(10, 40 - (10 * verbosity))


def verbosity_is_at_least(verbosity: int) -> bool:
    """
    :param verbosity: Int, the number of times that the user included the
                      --verbose flag when they started running the script.
    :return: Bool indicating whether the program is being run in verbose mode
    """
    return logging.getLogger().getEffectiveLevel() \
               <= verbosity_to_log_level(verbosity)


if __name__ == "__main__":
    main()
