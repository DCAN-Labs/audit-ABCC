#!/usr/bin/env python3

"""
ABCC Audit BIDS DB Object Classes
Greg Conan: gregmconan@gmail.com
Created: 2024-01-18
Updated: 2024-04-09
"""
# Standard imports
import functools
from glob import glob
import os
import pdb
import re
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional
import urllib.parse as urlparse

# External (pip/PyPI) imports
import boto3
import numpy as np
import pandas as pd

# Local imports
from src.utilities import (
    build_NGDR_fpath, build_summary_str, ColHeaderFactory, DataFrameQuery,
    Debuggable, df_is_nothing, DTYPE_2_UNIQ_COLS, explode_col,
    ImgDscColNameSwapper, invert_dict, float_is_nothing, LazyDict,
    make_default_out_file_name, mutual_reindex, RegexForBidsDetails,
    ShowTimeTaken, WHICH_DBS
)

# Constants:

# Map s3 bucket name to the name/ID of the session it has data from
BUCKET2SES = {"ABCC_year1": "ses-baselineYear1Arm1",
              "ABCC_year2": "ses-2YearFollowUpYArm1",
              "ABCC_year4": "ses-4YearFollowUpYArm1"}

# Regex compiled strings to quickly find subject ID, session ID, data type,
# or data type detail substrings within any given string
RGX_SPLIT = RegexForBidsDetails("anat", "audit", "dtype",
                                "dwi", "fmap", "func")


class BidsDBColumnNames(LazyDict, Debuggable):
    DEFAULT_ID = ("subject", "session")
    
    
    def __init__(self, prev_COLS: "BidsDBColumnNames" = None,
                 sub: Optional[str] = None, ses: Optional[str] = None,
                 **kwargs: Any) -> None:
        """
        :param df: pd.DataFrame that this object represents the columns of
        :param prev_COLS: BidsDBColumnNames object that already exists
        :param sub: String naming the subject/participant ID column
        :param ses: String naming the session ID column
        """
        if prev_COLS:
            self.update(prev_COLS)
        self.update(kwargs)
        self.SWAPPER = ImgDscColNameSwapper(self.get("img2hdr_fpath"))
        self.lazysetdefault("ID", LazyDict)
        self.add_ID_COLS(sub_col=sub, ses_col=ses)
        if self.get("df") is not None:
            self.create_from_own_df()


    def add_ID_COLS(self, sub_col: Optional[str] = None,
                    ses_col: Optional[str] = None) -> None:
        """
        Add subject ID and session ID column names
        :param sub_col: String naming the subject/participant ID column
        :param ses_col: String naming the session ID column
        """
        sub_ses_passed_in = (sub_col, ses_col)
        for i in range(len(self.DEFAULT_ID)):
            self.ID[self.DEFAULT_ID[i][:3]] = (
                sub_ses_passed_in[i] if sub_ses_passed_in[i] else
                self.lazyget("all", lambda: self.DEFAULT_ID)[i]
            )
        self.ID.sub_ses = [self.ID.sub, self.ID.ses]


    def add_dtype_COLS(self) -> None:
        """
        Add column names for "func", "anat", "dwi", & "fmap" files' details 
        """
        for dtype, extra_cols in DTYPE_2_UNIQ_COLS.items():
            self.ID[dtype] = [*self.ID.sub_ses, *extra_cols]


    def create_from_own_df(self) -> None:
        """
        Get column names from self.df (pd.DataFrame) columns
        """
        self.lazyget_all()
        try:
            self.task = self.get_subset_with("task-")
        except TypeError as e:
            self.debug_or_raise(e, locals())
        self.ID.lazysetdefault("non", self.get_non_ID_cols)


    def lazyget_all(self, df: Optional[pd.DataFrame] = None) -> Iterable[str]:
        if df is None:
            df = self.get("df")
        return self.lazysetdefault("all", lambda: df.columns.values)


    def get_non_ID_cols(self) -> pd.Series:
        """
        :return: pd.Series of all self.df column name strings
                 EXCEPT subject- or session-ID columns. 
        """
        return pd.Series(index=self.df.columns
                         ).drop(self.ID.sub_ses, errors="ignore"
                                ).reset_index().drop(columns=[0])["index"]
    

    def get_subset_with(self, to_find: str) -> List[str]:
        """
        :param to_find: String to look for in every column name
        :return: List[str] of column names containing to_find
        """
        return [col for col in self.lazyget_all() if re.search(to_find, col)]


    def rename_sub_ses_cols(self, new_sub_col: str, new_ses_col: str) -> None:
        """
        :param new_sub_col: String, subject/participant ID column's new name
        :param new_ses_col: String, session ID column's new name
        """
        self.df.rename(columns={self.ID.sub: new_sub_col,
                                self.ID.ses: new_ses_col}, inplace=True)
        self.add_ID_COLS(sub_col=new_sub_col, ses_col=new_ses_col)
        return self.df


class BidsDBDetailsForType(Debuggable):
    REFORMATTERS = {"rec-normalized": bool, "Tw": int,
                    "run": lambda x: float(x) if x != "" else np.nan}

    def __init__(self, dtype: str, df: pd.DataFrame, col_to_keep: str,
                 db_COLS: BidsDBColumnNames, from_FTQC: bool,
                 drop_NaN_from: Optional[Iterable[str]] = list(),
                 debugging: bool = False) -> None:
        """
        :param df: pd.DataFrame with a column named self.COLS.fname of file
                   name/path strings to split into new columns if not from_FTQC
        :param dtype: str, data type, a key in DTYPE_2_UNIQ_COLS
        :param col_to_get: String naming the column of the BidsDB pd.DataFrame
                           to keep values from in the final pd.DataFrame
        :param db_COLS: BidsDBColumnNames, _description_
        """
        self.COLS = db_COLS
        self.col_to_keep = col_to_keep
        self.debugging = debugging
        self.df = df
        self.dtype = dtype

        # Build "factory" object ready to create all column headers
        specifier = (dict(uniq_col=self.COLS.hdr) if from_FTQC else
                     dict(dtype=dtype))
        self.hdr_mkr = ColHeaderFactory(debugging=debugging,
                                        will_add_run=True, **specifier,
                                        run_col=self.COLS.run)
        
        if from_FTQC:
            self.add_header_col()
            self.COLS[dtype] = set(self.df[self.COLS.hdr].dropna().unique())
        else:
            self.make_df_from_fpaths()
            
            self.COLS[dtype] = self.hdr_mkr.get_all_for(df=self.df)

        # Remove any rows containing NaN in the drop_NaN_from column(s)
        if drop_NaN_from:
            self.df.dropna(subset=drop_NaN_from, inplace=True)
        # if dtype == "anat":
        if not from_FTQC:
            self.df.loc[self.df[self.COLS.run] == "", self.COLS.run] = 1
            self.add_header_col()  # TODO ?


    def add_header_col(self) -> Dict[str, pd.DataFrame]:
        """
        :param dfs_BIDS: Dict[str, pd.DataFrame] mapping each data type str
                         ("anat", "func", etc) to pd.DataFrame of data from
                         files with that data type
        :return: dict, dfs_BIDS but with a new column in its pd.DataFrame
                 values: the column header in the final self.df
        """
        self.df[self.COLS.hdr] = \
            self.df.apply(self.hdr_mkr.new_from_row, axis=1)
        

    def make_df_from_fpaths(self):  # , *drop_NaN_from: str):
        """
        Split filenames into new columns with all relevant details for its dtype
        :param drop_NaN_from: Iterable[str], names of columns to drop NaNs from
        """
        try:
            dtype_ID_COLS = self.COLS.ID.get(self.dtype)
            if float_is_nothing(self.df.get(dtype_ID_COLS)):
                self.df[dtype_ID_COLS] = explode_col(self.df[self.COLS.fname],
                                                     RGX_SPLIT, self.dtype)
            assert not df_is_nothing(self.df)
        except (AssertionError, AttributeError, KeyError, ValueError) as e:
            self.debug_or_raise(e, locals())


    def make_final_df(self) -> pd.DataFrame:
        """
        Given a dataframe with one BIDS file per line, where each has its own
        subject/session *and* its own task/run/etc., restructure it to store
        data by subject-session instead of by file. Makes final output df 
        :param col_to_get: str, column of df to store values from in output df
        :param df: pd.DataFrame, DataFrame with one row per file
        :param dtype: str, data type, a key in DTYPE_2_UNIQ_COLS
        :return: pd.DataFrame, df with one subject-session per row and new
                 columns for every different value in the "header" column
        """
        try:
            sub_ses_dict_cols = self.COLS[self.dtype]
            # sub_ses_dict_cols = self.COLS.lazyget(self.dtype, lambda: self.df[self.COLS.hdr].unique())
        except (AttributeError, KeyError) as e:
            self.debug_or_raise(e, locals())
        # sub_ses_dict_cols = self.COLS.dtype  # self.df[self.COLS.hdr].unique() if self.COLS.hdr in self.df else 
        dicts_for_df = self.df.groupby(self.COLS.ID.sub_ses).apply(
            lambda sub_ses_df: self.make_subj_ses_dict(
                sub_ses_df, sub_ses_dict_cols
            )
        )
        final_df = pd.DataFrame(dicts_for_df.values.tolist(),
                                columns=[*self.COLS.ID.sub_ses,
                                         *sub_ses_dict_cols])  # *self.COLS.dtype])
        if self.debugging:
            pdb.set_trace()
        return final_df.dropna(how="all", axis=1)
        


    def make_subj_ses_dict(self, a_df: pd.DataFrame, dict_cols: Iterable[str]
                           ) -> Dict[str, Any]:
        """
        :param a_df: pd.DataFrame with the same ID in each row in its subject
                     column and the same ID in each row in its session column
        :param dict_cols: Iterable[str] of a_df column names to return as keys
        :return: Dict[str, Any] mapping dict_cols to their values in a_df
        """
        sub_ses_dict = {col: None for col in dict_cols}
        if not a_df.empty:
            for col_ID in self.COLS.ID.sub_ses:
                sub_ses_dict[col_ID] = a_df[col_ID].iloc[0]
            def add_to_sub_ses_dict(row):
                sub_ses_dict[row.get(self.COLS.hdr)] = \
                    row.get(self.col_to_keep)
            a_df.apply(add_to_sub_ses_dict, axis=1)
        return sub_ses_dict


    @classmethod
    def reformat_col(cls, col: pd.Series) -> pd.Series:
        return col.apply(cls.REFORMATTERS.get(col.name, lambda x: x))


class BidsDB(LazyDict, Debuggable):
    def __init__(self, in_files: Optional[Iterable[str]] = list(),
                 out_fpath: Optional[str] = None, df:
                 Optional[pd.DataFrame] = None, sub_col: Optional[str] = None,
                 ses_col: Optional[str] = None, debugging: bool = False,
                 img2hdr_fpath: Optional[str] = None) -> None:
        """
        :param in_files: List of valid paths to existing file(s) to read from
        :param out_fpath: Valid output file path to write the final df to
        :param df: pd.DataFrame, this BIDS DB's contents, if already made
        :param sub_col: String naming the subject/participant ID column
        :param ses_col: String naming the session ID column
        :param debugging: True to pause & interact on error or False to crash
        """
        self.debugging = debugging
        self.lazysetdefault("fpath", LazyDict)

        # "if df:" wouldn't work because bool(df) raises an exception
        if df is not None:
            self.df = df
        elif in_files:
            self.df = self.read_df_from(*in_files)
        else:
            self.debug_or_raise(ValueError("No data provided"), locals())

        self.ensure_has_COLS(sub=sub_col, ses=ses_col, fname="image_file",
                             img2hdr_fpath=img2hdr_fpath, df=self.df)
        self.df = self.set_sub_ses_cols_as_index(self.df)

        if out_fpath:
            self.save_to(out_fpath)


    def __str__(self) -> str:
        """ 
        :return: String representing the DB dataframe and naming its columns
        """
        name = self.__class__.__name__
        df = self.get("df", None)
        return  (f"{name}.COLS:\n\n{self.COLS.lazyget_all(df=df)}\n\n"
                 f"{name}.df:\n{df}")
    __repr__ = __str__


    def ensure_has_COLS(self, sub: str, ses: str, **kwargs: Any) -> None:
        """
        Make sure that this BidsDB has a COLS attribute containing all
        necessary column name strings. 
        :param sub: String naming the subject/participant ID column
        :param ses: String naming the session ID column
        """
        self.COLS = BidsDBColumnNames(
            prev_COLS=self.lazysetdefault("COLS", LazyDict),
            sub=sub, ses=ses, hdr="header", run="run", QC="QC",
            img_dsc="image_description", DT="datetimestamp",
            debugging=self.debugging, **kwargs  #  df=self.get("df"),
        )


    def finalize_df(self, from_FTQC: bool, **dfs_BIDS: pd.DataFrame
                    ) -> pd.DataFrame:
        """
        :param paths_BIDS: Dict[str, pd.DataFrame] mapping dtype strings (keys
                           from DTYPE_2_UNIQ_COLS) to pd.DataFrames with 1 row
                           per file of that data type and 1 column (file path)
        :return: pd.DataFrame, valid BidsDB.df with 1 subject session per row
                 and 1 dtype-detail per column 
        """
        # Input parameters to create BidsDBDetailsForType objects for each 
        # dtype depend on whether this BidsDB is FastTrack or Tier1/2
        details = dict(db_COLS=self.COLS, debugging=self.debugging,
                       from_FTQC=from_FTQC)
        if from_FTQC:
            details["col_to_keep"] = self.COLS.QC
            details["drop_NaN_from"] = [self.COLS.hdr]
        else:
            details["col_to_keep"] = self.COLS.fname

        # TODO add explanatory comment
        non_ID_COLS = set()
        details_for_dtype = list()
        for dtype, df in dfs_BIDS.items():
            details_for_dtype.append(BidsDBDetailsForType(
                dtype=dtype, df=df, **details
            ))
            # self.COLS[dtype] = details_for_dtype[-1].COLS.dtype
            non_ID_COLS.update(details_for_dtype[-1].COLS[dtype])

        self.COLS.lazysetdefault("all", lambda: [*self.COLS.ID.sub_ses,
                                                *non_ID_COLS])
        self.COLS.ID.non = pd.Series(list(non_ID_COLS)) 
        self.COLS.ID.non.sort_values(inplace=True)

        # Collapse df so it has 1 row per subject session
        new_dfs = [details.make_final_df() for details in details_for_dtype]

        if self.debugging:
            pdb.set_trace()
        return self.merge_N_dfs(*new_dfs)


    def get_DB_df(self, make_df: Callable, in_fpath: Optional[str] = None,
                  df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        If a BidsDB.df was already created, then use it, else make a new one
        :param in_fpath: Valid path to readable properly-formatted BIDS DB .tsv
        :return: pd.DataFrame, BIDS DB
        """
        if df is None:
            if in_fpath:
                try:
                    df = pd.read_csv(in_fpath, sep="\t",
                                     index_col=self.COLS.ID.sub_ses)
                except (AttributeError, OSError) as e:
                    self.debug_or_raise(e, locals())
            else:
                df = make_df()
        return df


    def get_empty_df(self, *_: Any) -> pd.DataFrame:
        """
        :return: pd.DataFrame, empty but with (some) valid anat and func
                 column names
        """
        return pd.DataFrame(columns=self.COLS.all)


    def merge_2_dfs_on_sub_ses(self, df_L: pd.DataFrame, df_R: pd.DataFrame
                               ) -> pd.DataFrame:
        """
        :param df_L: pd.DataFrame with subject and session ID columns
        :param df_R: pd.DataFrame with subject and session ID columns
        :return: Outer merge of df_L and df_R on subject/session ID columns
        """
        return pd.merge(left=df_L, right=df_R, how="outer",
                        on=self.COLS.ID.sub_ses)
    

    def merge_N_dfs(self, *dfs: pd.DataFrame) -> pd.DataFrame:
        """
        _summary_ 
        :return: pd.DataFrame, _description_
        """  # TODO Implement switch-case (finally) added in Python 3.10
        n_dfs = len(dfs)
        if n_dfs > 2:
            final_df = functools.reduce(self.merge_2_dfs_on_sub_ses, dfs)
        elif n_dfs == 2:
            final_df = self.merge_2_dfs_on_sub_ses(*dfs)
        elif n_dfs == 1:
            final_df = dfs[0]
        else:
            final_df = self.get_empty_df()
        return final_df
    

    def query_get(self, **kwargs: Any) -> pd.DataFrame:
        try:
            return DataFrameQuery(df=self.get("df"), **kwargs)
        except TypeError as e:
            self.debug_or_raise(e, locals())


    def read_df_from(self, *in_paths: str) -> pd.DataFrame:
        """
        :param in_paths: List of strings, each a valid path to a readable .csv 
        :return: pd.DataFrame of all data from the in_paths .csv files
        """
        final_df = None
        if in_paths:
            dfs_to_concat = list()
            for each_csv_path in in_paths:  
                dfs_to_concat.append(pd.read_csv(each_csv_path))
            final_df = pd.concat(dfs_to_concat)
        return final_df
                    

    def save_to(self, bids_DB_fpath: str) -> None:
        """
        Save BidsDB.df object into a spreadsheet to read/use later
        :param bids_DB_fpath: Valid writeable .tsv/.csv file path
        """
        whichsep = "\t" if bids_DB_fpath.endswith(".tsv") else ","
        self.df.to_csv(bids_DB_fpath, index=True, sep=whichsep)


    def set_sub_ses_cols_as_index(self, a_df: pd.DataFrame) -> pd.DataFrame:
        """
        Make the subject and session columns of self.df into its MultiIndex
        :param a_df: pd.DataFrame with subject and session columns
        :return: pd.DataFrame, now with MultiIndex of [subject, session]
        """
        try:
            if a_df.index.names != self.COLS.ID.sub_ses:
                a_df.set_index(self.COLS.ID.sub_ses, inplace=True)
            return a_df
        except (AttributeError, KeyError) as e:
            self.debug_or_raise(e, locals())


    def split_df_by_all_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        :param df: pd.DataFrame with a column called {self.COLS.fname}
        :return: pd.DataFrame with a column for each kind of data in df.fname
        """
        if df.empty:
            new_df = None
        else:
            dtype = df.dtype.unique()[0]
            new_df = pd.DataFrame(
                data=explode_col(df[self.COLS.fname], RGX_SPLIT, dtype),
                columns=self.COLS.ID.sub_ses + DTYPE_2_UNIQ_COLS.get(dtype)
            )
        return new_df
    

    def split_into_dtype_dfs(self, df: pd.DataFrame
                             ) -> Dict[str, pd.DataFrame]:
        """
        :param df: pd.DataFrame, self.df with a "dtype" column
        :return: Dict mapping dtype name strings ('anat', 'func', etc.) to 
                 pd.DataFrames (all rows of each dtype in df)
        """
        dtype2ixs = df.groupby("dtype").indices
        try:  # This assumes numerical indices instead of sub-ses ones
            return {dtype: df.iloc[ixs] for dtype, ixs in dtype2ixs.items()}
        except KeyError as e:
            self.debug_or_raise(e, locals())


class BidsDBToQuery(BidsDB):  # TODO
    def __init__(self, a_DB: BidsDB) -> None:
        self.update(a_DB)
        self.subset = LazyDict({"with_anat": self.query_anat()})
        self.subset["processed"] = self.query_processed_subjects()
        self.subset["unprocessed"] = self.query_unprocessed_subjects()
    

    def get_processed_rows(self) -> pd.DataFrame:
        return self.df[~self.df[self.df.columns[2:]
                                ].isin(['no bids']).any(axis=1)]


    def get_row_with_sub_ses_matching(self, orig_row: pd.Series) -> pd.Series:
        """
        :return: Row of self.df with the same subject and session as orig_row
        """
        return self.get_rows_with_sub_ses(orig_row.get(self.COLS.ID.sub),
                                          orig_row.get(self.COLS.ID.ses)
                                          ).iloc[0]
    

    def get_rows_with_sub_ses(self, subject: str, session: str
                              ) -> pd.DataFrame:
        return self.df[(self.df[self.COLS.ID.sub] == subject) &
                       (self.df[self.COLS.ID.ses] == session)]
        

    def get_session_names(self) -> List[str]:
        return self.df[self.COLS.ID.ses].unique().tolist()


    def query(self, key_name: str) -> pd.DataFrame:
        return self.df[self.subset[key_name]]


    def query_anat(self) -> pd.DataFrame:
        return ~self.df.filter(regex='T[1,2].*').isin([np.nan]).all(axis=1)


    def query_processed_subjects(self) -> pd.DataFrame:
        """
        Filter dataframe to get dataframe of subjects that do not have any
        unprocessed images
        """
        processed_df = self.df[~self.df[self.df.columns[2:]
                                        ].isin(['no bids']).any(axis=1)]
        # Filter again to remove subjects that have need data deleted
        return (~processed_df.isin(
            ['delete (tier1)', 'delete (s3)']
        ).any(axis=1)) & (self.subset["with_anat"])  # Remove subjects without a T1
    

    def query_same_IDs_as(self, otherDB: "BidsDB"):
        return (
            (self.df[self.COLS.ID.sub] == otherDB.df[otherDB.COLS.ID.sub]) &
            (self.df[self.COLS.ID.ses] == otherDB.df[otherDB.COLS.ID.ses])
        )
    

    def query_ses(self, session: str) -> pd.Series:
        return self.df[self.COL.ID.ses] == session
    

    def query_unprocessed_subjects(self) -> pd.Series:
        """
        Check for fully unprocessed subjects
        """
        # Filter dataframe to get dataframe of subjects that are missing one
        # or more modalities
        missing_data_df = self.get_processed_rows()

        # Filter again to remove subjects that have BIDS data somewhere
        return ~missing_data_df.isin(
            ['bids (tier1)', 'delete (tier1)', 'bids (s3)', 'delete (s3)']
        ).any(axis=1)


class FastTrackQCDB(BidsDB):
    def __init__(self, fpath_FTQC: str, dtypes: Iterable[str],
                 in_fpath: Optional[str] = None,  # file_ext: str, 
                 sub_col: Optional[str] = None, ses_col: Optional[str] = None,
                 df: Optional[pd.DataFrame] = None,
                 out_fpath: Optional[str] = None, debugging: bool = False,
                 img2hdr_fpath: Optional[str] = None) -> None:
        """
        :param fpath_FTQC: Valid path to readable abcd_fastqc01.txt file
        :param dtypes: Iterable[str] of data types (DTYPE_2_UNIQ_COLS keys)
        :param in_fpath: Valid path to readable input file to read instead of
                         making the FastTrackQCDB from abcd_fastqc01.txt
        :param sub_col: String naming the subject/participant ID column
        :param ses_col: String naming the session ID column
        :param df: pd.DataFrame, this BIDS DB's contents, if already made
        :param out_fpath: Valid output file path to write the final df to
        :param debugging: True to pause & interact on error or False to crash
        """
        self.debugging = debugging
        self.dtypes = dtypes
        self.ensure_has_COLS(sub=sub_col, ses=ses_col, temp="desc_and_dt",
                             to_split="ftq_series_id", img2hdr_fpath=img2hdr_fpath, 
                             fname="reformatted_fname")
        self.COLS.add_dtype_COLS()
        self.df = self.get_DB_df(lambda: self.make_FTQC_df_from(fpath_FTQC),
                                 in_fpath, df)
        super().__init__(out_fpath=out_fpath, df=self.df, sub_col=sub_col,
                         ses_col=ses_col, debugging=debugging)


    def add_cols_to_df_by_splitting(self, col_to_split: str) -> None:
        """
        Given a column of (file name/path) strings, split it into new columns
        for each detail to save from that file name: subject ID, session ID, 
        image_description, and datetimestamp
        :param col_to_split: String naming the column to split into new cols
        """
        try:
            self.split_df_COL(col_to_split, *self.COLS.ID.sub_ses,
                              self.COLS.temp)
            self.split_df_COL(self.COLS.temp, self.COLS.img_dsc,
                              self.COLS.DT, reverse=True)
            for which_ID in ("sub", "ses"):
                which = self.COLS.ID[which_ID]
                self.df[which] = f"{which_ID}-" + self.df[which]
        except (KeyError, ValueError) as e:
            self.debug_or_raise(e, locals())
    

    def add_run_numbers_to(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add run number by counting up from lowest datetimestamp
        :param df: pd.DataFrame with column names corresponding to these
                   self.COLS keys: ID.sub, ID.ses, hdr, run, and DT.
        :return: pd.DataFrame, now including a column with the run number for
                 every file(/row) in the Fast Track QC spreadsheet.
        """
        # Only add run numbers to ftq_usable == 1 entries
        is_usable = df[self.COLS.QC] == 1.0

        # Every new date-time-stamp is a new run, so count datetimestamps per
        # subject, per session, and per specifier (e.g. task or scan type),
        # and fill missing values with the run number for that date-time-stamp
        self.COLS.ftqc = [*self.COLS.ID.sub_ses, self.COLS.hdr]
        allgroups = df.loc[is_usable].groupby(self.COLS.ftqc)
        df.loc[is_usable, self.COLS.run] = \
            allgroups[self.COLS.DT].cumcount() + 1  # ix 0 -> run num 1
        
        # Entries with ftq_usable != 1 have no run number
        df.loc[~is_usable, self.COLS.run] = np.nan

        # Remove anatomical entries 
        return df


    def make_FTQC_df_from(self, fpath_FTQC: str) -> pd.DataFrame:
        """
        :param fpath_FTQC: Valid path to existing abcd_fastqc01.txt file
        :return: pd.DataFrame, valid BidsDB.df of ftq_series_id values, with
                 1 subject session per row and 1 dtype-detail per column
        """
        # Read abcd_fastqc01.txt file and add col for each relevant detail
        self.df = self.read_FTQC_df_from(fpath_FTQC)
        self.add_cols_to_df_by_splitting(self.COLS.to_split)
        self.rename_df_cols()

        # Add header/dtype columns, and remove any rows representing
        # files with a data type mapped to null in the .JSON
        self.df[self.COLS.hdr] = self.df[self.COLS.img_dsc].apply(
            self.COLS.SWAPPER.to_header_col
        )
        self.df["dtype"] = self.df[self.COLS.hdr
                                   ].apply(self.COLS.SWAPPER.dtype_of.get)

        # Add run numbers to header col
        self.df = self.add_run_numbers_to(self.df)  # Get run numbers
        self.COLS["fname"] = "image_file"
        try:
            self.df = self.finalize_df(**self.split_into_dtype_dfs(self.df),
                                       from_FTQC=True)
            return self.df
        except (AttributeError, KeyError, TypeError, ValueError) as e:
            self.debug_or_raise(e, locals())


    @staticmethod
    def read_FTQC_df_from(fpath_FTQC: str) -> pd.DataFrame:
        """
        :param fpath_FTQC: String, valid readable abcd_fastqc01.txt file path
        :return: pd.DataFrame of data read from fpath_FTQC
        """
        SEP_UNQUOTE = '["]*[,|\t|"]["]*'  # Remove unneeded quotation marks
        return pd.read_csv(fpath_FTQC, encoding="utf-8-sig", sep=SEP_UNQUOTE,
                           engine="python", index_col=False, quotechar='"',
                           skiprows=[1]  # Skip row 2 (column descriptions)
                           ).dropna(how='all', axis=1)


    def rename_df_cols(self) -> None:
        """
        Change column names for good_bad_series_parser, then save to .csv 
        """
        self.df.rename({"ftq_usable": self.COLS.QC, "subjectkey": "pGUID",
                        "visit": "EventName", "interview_age": "SeriesTime",
                        "abcd_compliant": "ABCD_Compliant", "file_source":
                        "image_file", "comments_misc": "SeriesDescription"},
                        axis="columns", inplace=True)


    def split_df_COL(self, col_to_split: str, *cols_to_split_into: str,
                     reverse: bool = False):
        """
        :param col_to_split: String naming the self.df column to split
        :param cols_to_split_into: Iterable[str] naming the new columns to
                                   add to self.df by splitting col_to_split 
        :param reverse: True to start splitting col_to_split at the end;
                        otherwise False to start splitting from the beginning
        """
        n_splits = len(cols_to_split_into) - 1
        ser_to_split = self.df[col_to_split].str
        split = ser_to_split.rsplit if reverse else ser_to_split.split
        self.df[[*cols_to_split_into]] = split('_', n_splits).values.tolist()


class Tier1BidsDB(BidsDB):
    def __init__(self, tier1_dir: str, sub_col: str, ses_col: str,
                 file_ext: str, dtypes: Iterable[str],
                 df: Optional[pd.DataFrame] = None,
                 in_fpath: Optional[str] = None,
                 out_fpath: Optional[str] = None, debugging: bool = False,
                 img2hdr_fpath: Optional[str] = None) -> None:
        """
        :param tier1_dir: String, valid path to existing ABCD BIDS root 
                          directory on tier 1 with "sub-*" subdirectories
        :param sub_col: String naming the subject/participant ID column
        :param ses_col: String naming the session ID column
        :param file_ext: String, the extension of the tier 1 files to collect
        :param dtypes: Iterable[str] of data types (DTYPE_2_UNIQ_COLS keys)
        :param df: pd.DataFrame, this BIDS DB's contents, if already made
        :param out_fpath: Valid output file path to write the final df to
        """
        self.debugging = debugging
        self.dtypes = dtypes
        self.file_ext = file_ext
        self.tier1_dir = tier1_dir

        self.ensure_has_COLS(sub=sub_col, ses=ses_col, fname="NGDR_fpath",
                            tier1_dir="tier1_dirpath",
                            img2hdr_fpath=img2hdr_fpath)
        self.COLS.add_dtype_COLS()

        self.df = self.get_DB_df(lambda: self.finalize_df(
            # Restructure the pd.DataFrame for each dtype into 1 df with a
            # column for each detail we need; turn those details into columns
            # in the final df with 1 row per subject session
            **self.get_BIDS_file_paths_df(), from_FTQC=False
        ), in_fpath, df)

        # Use the newly created dataframe to turn this into a BidsDB object
        super().__init__(df=self.df, debugging=debugging, out_fpath=out_fpath)


    def get_BIDS_file_paths_df(self) -> Dict[str, pd.DataFrame]:
        """
        Collect all files of each dtype in the provided local (tier 1)
        directory path, and put them into one pd.DataFrame per dtype
        :return: dict mapping each dtype to a pd.DataFrame with paths to all
                 local / tier 1 files of that dtype
        """
        df_for = dict()
        for dtype in self.dtypes:  # TODO Only execute glob once by using some kind of "{dtype1} OR {dtype2}" pattern matcher?
            tier1_fpath = build_NGDR_fpath(self.tier1_dir, dtype,
                                           f"*{self.file_ext}")
            paths = glob(tier1_fpath)
            df_for[dtype] = pd.DataFrame({self.COLS.fname: paths})
        return df_for


class S3BidsDB(BidsDB):
    def __init__(self, client: boto3.session.Session.client,
                 buckets: List[str], file_ext: str, dtypes: Iterable[str],
                 in_fpath:  Optional[str] = None,
                 out_fpath: Optional[str] = None,
                 sub_col: Optional[str] = None, ses_col: Optional[str] = None,
                 df: Optional[pd.DataFrame] = None, debugging: bool = False,
                 img2hdr_fpath: Optional[str] = None) -> None:
        """
        :param client: boto3.session.Session.client to access s3 buckets
        :param buckets: List[str] of s3 bucket names to get data from
        :param file_ext: String, the extension of the tier 2 files to collect
        :param dtypes: Iterable[str] of data types (DTYPE_2_UNIQ_COLS keys)
        :param in_fpath: Valid path to readable input file to read instead of
                         making the S3BidsDB by sending requests to the s3 API
        :param out_fpath: Valid output file path to write the final df to
        :param sub_col: String naming the subject/participant ID column
        :param ses_col: String naming the session ID column
        :param df: pd.DataFrame, this BIDS DB's contents, if already made
        :param debugging: True to pause & interact on error or False to crash
        """
        self.debugging = debugging
        self.dtypes = dtypes
        self.client = client
        self.file_ext = file_ext
        self.ensure_has_COLS(sub=sub_col, ses=ses_col,
                             img2hdr_fpath=img2hdr_fpath,
                             fname="s3_file_relative_path")
        self.df = self.get_DB_df(lambda: self.download_and_merge(*buckets),
                                 in_fpath, df)

        # Use the newly created dataframe to turn this into a BidsDB object
        super().__init__(df=self.df, debugging=debugging, out_fpath=out_fpath)


    def download(self, key_name: str, subkey_name: str, Prefix: str,
                 **kwargs: Any) -> List[Dict[str, str]]:
        """
        :param key_name: String, the key mapped to the list of data
                         downloaded from the s3 (either "Contents" or
                         "CommonPrefixes")
        :param subkey_name: String, the key mapped to the specific detail we
                            want to save from the s3 download (either "Key"
                            or "Prefix")
        :param Prefix: String, the s3 path between the bucket name and the
                       parts that vary within the download; i.e. the parent
                       directory path to download from within the bucket
        :return: Subject session detail dicts downloaded from the s3, or None
        """
        try: 
            s3_data = [s3_datum[subkey_name]
                       for page in self.paginate(Prefix=Prefix, **kwargs)
                       for s3_datum in page[key_name]]
        except KeyError:
            s3_data = None
        return s3_data


    def download_1_sub_ses_from(self, bucket: str, subject: str, session: str
                                ) -> pd.DataFrame:
        """
        :param bucket: String naming the s3 bucket to download data from
        :param subject: String, subject ID of participant to download data of
        :param session: String naming the session to download data of
        :return: pd.DataFrame with all data in filenames of that participant
                 session in that bucket: subject ID, session name, rec (if 
                 any), run number (if any), T1 or T2 (if applicable), acq (if
                 any), and task name (if any)
        """
        df_COLS = [*self.COLS.ID.sub_ses, "dtype", self.COLS.fname]
        paths = pd.Series([urlparse.unquote(s3_datum["Key"]) for page in
                           self.paginate(Bucket=bucket,
                                         Prefix=f"{subject}/{session}/")
                           for s3_datum in page["Contents"]])
        df = pd.DataFrame(paths.str.split('/').values.tolist(),
                          columns=df_COLS)
        df = df.groupby("dtype").apply(self.split_df_by_all_dtypes)
        df = df.reset_index(level=1, drop=True).reset_index()
        return df.replace("", np.nan).dropna(axis=1, how="all")
    

    def download_and_merge(self, *buckets: str) -> pd.DataFrame:
        """
        :param buckets: Iterable[str] of accessible s3 bucket names
        :return: pd.DataFrame of all desired data downloaded from buckets
        """
        assert buckets  # assert len(buckets) > 0
        return pd.concat([S3Bucket(self.client, name, self.file_ext,
                                   self.dtypes, *self.COLS.ID.sub_ses,
                                   self.debugging).df for name in buckets])


    def get_BIDS_files_for(self, subj_ID: str, session: Optional[str] = None
                           ) -> List[Dict[str, str]]:
        """
        :param subj_ID: Participant ID, "sub-NDARINV" followed by 8 characters
        :return: List[Dict[str,str]] of each subj_ID file details, or None
        """
        if not session:
            session = self.session
        return self.download("Contents", "Key", f"{subj_ID}/{session}/")


    def paginate(self, **kwargs: Any
                 ) -> boto3.session.botocore.paginate.PageIterator:
        """
        :param kwargs: Dict of keyword arguments for s3client.get_paginator
        :return: Iterator (over s3 bucket objects) yielding pages which act 
                 like nested dicts of strings: Dict[str, Dict[str, Dict[...]]]
        """
        # if not kwargs.get("Bucket")
        kwargs.setdefault("Bucket", self.name)
        return self.client.get_paginator("list_objects_v2").paginate(
            FetchOwner=False, EncodingType="url", ContinuationToken="", 
            StartAfter="", **kwargs
        )


class S3Bucket(S3BidsDB):
    def __init__(self, client: boto3.session.Session.client, bucket_name: str,
                 file_ext: str, dtypes: Iterable[str],
                 sub_col: Optional[str] = None, ses_col: Optional[str] = None,
                 debugging: bool = False, img2hdr_fpath: Optional[str] = None
                 ) -> None:
        """
        :param client: boto3.session.Session.client to access s3 bucket
        :param bucket_name: String naming an accessible s3 bucket
        :param file_ext: String, extension of files to find in the s3 bucket
        :param dtypes: Iterable[str] of data types (DTYPE_2_UNIQ_COLS keys)
        :param sub_col: String naming the subject/participant ID column
        :param ses_col: String naming the session ID column
        :param debugging: True to pause & interact on error or False to crash
        """
        self.debugging = debugging
        self.client = client
        self.dtypes = dtypes
        self.file_ext = file_ext
        self.name = bucket_name
        # self.session = BUCKET2SES[bucket_name]  # Don't assume 1 bucket per session

        with ShowTimeTaken(f"auditing s3://{self.name} bucket"):
            self.ensure_has_COLS(sub=sub_col, ses=ses_col,
                                 fname="s3_file_subpath",
                                 img2hdr_fpath=img2hdr_fpath)
            self.df = self.get_bids_subject_IDs_df()
            self.COLS.add_dtype_COLS()

            # If the bucket has no subjects, then instead of adding content 
            # for each subject session, add an empty session column
            if self.df.empty:  
                self.df[self.COLS.ID.ses] = None
            else:
                self.df = self.finalize_df(**self.get_BIDS_file_paths_df(),
                                           from_FTQC=False)

            self.df = self.set_sub_ses_cols_as_index(self.df)
            super().__init__(client=client, buckets=list(), file_ext=file_ext,
                             dtypes=dtypes, df=self.df, debugging=debugging)


    def get_BIDS_file_paths_df(self) -> Dict[str, pd.DataFrame]:
        """
        :return: Dict mapping a dtype to a pd.DataFrame with data of that type
        """
        with ShowTimeTaken(f"downloading s3://{self.name} paths"):
            # Add self.df column with (URL-unformatted) file name/path strings
            self.df[self.COLS.fname] = self.df[self.COLS.ID.sub
                                            ].apply(self.get_BIDS_files_for)
        with ShowTimeTaken(f"reformatting s3://{self.name} dataframe"):
            all_files = self.df[self.COLS.fname].explode(ignore_index=True
                                                        ).apply(urlparse.unquote)
            
            try:  # Get relevant all_files dataframe subsets (1 per dtype)
                df_all = all_files.loc[all_files.str.endswith(self.file_ext)
                                    ].to_frame()
                df_all["dtype"] = explode_col(df_all[self.COLS.fname],
                                              RGX_SPLIT, "dtype")
                return self.split_into_dtype_dfs(df_all) 
            except (IndexError, KeyError) as e:
                self.debug_or_raise(e, locals())


    def get_bids_subject_IDs_df(self) -> pd.Series:
        """
        :return: pd.Series, a column of all subject IDs in the s3 bucket
        """
        subj_IDs = pd.Series(self.download("CommonPrefixes", "Prefix",
                                           "", Delimiter="/"))
        subj_IDs = (subj_IDs.to_frame() if subj_IDs.empty else
                    subj_IDs.str.extract(RGX_SPLIT.create("subj")).dropna())
        return subj_IDs.rename(columns={0: self.COLS.ID.sub})


class AllBidsDBs(BidsDB):

    def __init__(self, cli_args: Mapping[str, Any], 
                 sub_col: str, ses_col: str,
                 client: boto3.session.Session.client = None) -> None:
        """
        Overarching class to create, contain, and summarize BidsDB objects 
        :param cli_args: Mapping[str, Any] of command-line input arguments
        :param sub_col: String naming the subject/participant ID column
        :param ses_col: String naming the session ID column
        :param client: s3 client to send/receive requests to/from s3 if needed
        """
        self.update(cli_args)  # Save cli_args as attributes for easy access
        self.client = client

        self.ensure_has_COLS(sub=sub_col, ses=ses_col,
                             img2hdr_fpath=self.img2hdr_fpath)
        # self.COLS = LazyDict(sub_col=sub_col, ses_col=ses_col)
        self.DB = list()       # List[BidsDB] created/read during this run
        self.ix_of = dict()    # Dict[str, int]: indices of DBs in self.DB
        self.out_fpaths = {key: self.build_out_fpath(key)  # File path strings
                           for key in self["to_save"]}     # to save DBs to
        
        # self.add_DB input parameters specific to each kind of BidsDB
        self.PARAMS = pd.DataFrame([
            [FastTrackQCDB, "ABCD FastTrack01 QC", "the NDA",
             ["fpath_FTQC"]],
            [Tier1BidsDB, "Tier 1", "the NGDR space on the MSI",
             ["tier1_dir", "file_ext"]],
            [S3BidsDB, "Tier 2", "the AWS s3 buckets " +
             ", ".join(self["buckets"]), ["client", "buckets", "file_ext"]]
        ], columns=["DB", "db_name", "src", "kwargs"], index=WHICH_DBS)


    def __str__(self) -> str:
        """ 
        :return: String representing the contained DB objects' dataframes and
                 name their columns
        """
        name = self.__class__.__name__
        return ("\n\n".join([name] + [db.__repr__() for db in self.DB]))
    __repr__ = __str__


    def add_DB(self, key: str) -> None:
        """
        Given the shorthand name for a BidsDB, create it and store it 
        :param key: String, shorthand naming which BidsDB to make: "ftqc" for
                    FastTrackQCDB, "tier1" for Tier1BidsDB, "s3" for S3BidsDB
        """
        which = self.PARAMS.loc[key]
        with ShowTimeTaken(f"making {which.db_name} DB from {which.src}"):
            self.ix_of[key] = len(self.DB)
            self.DB.append(self.make_DB(key, which.DB, **{
                kwarg: self[kwarg] for kwarg in which.kwargs
            }))


    def build_out_fpath(self, key_DB: str) -> str:
        """
        :param key_DB: String, shorthand naming which BidsDB to make: "ftqc"=
                       FastTrackQCDB, "tier1"=Tier1BidsDB, and "s3"=S3BidsDB
        :return: String, valid output file path
        """
        return os.path.join(self["output"], self.lazyget(
            f"{key_DB}_DB_file", lambda: make_default_out_file_name(key_DB)
        ))
    

    def get_DB(self, key_DB: str) -> BidsDB:
        """
        :param key_DB: String, shorthand name for which BIDS DB to fetch
        :return: BidsDB (already created) whose shorthand name is key_DB
        """
        return self.DB[self.ix_of[key_DB]]
    

    def get_subj_ses_covg(self, row: pd.Series) -> float:
        """
        :param row: pd.Series of floats; each is either 1.0 to mean that a
                    file is covered for a subject session or 0.0 otherwise
        :return: float between 0 and 1; coverage percentage of row
        """
        n_NaNs = self.row_NaN_count.loc[row.name]  # Exclude NaNs
        return (row.sum() - n_NaNs) / (row.shape[0] - n_NaNs)


    def make_DB(self, key: str, initialize_DB: Callable, **kwargs: Any
                ) -> BidsDB:
        """
        :param key: String, shorthand name for which BIDS DB to create:
                    "ftqc"=FastTrackQCDB, "tier1"=Tier1BidsDB, "s3"=S3BidsDB
        :param initialize_DB: Class object to call to create a BidsDB
        :return: BidsDB
        """
        return initialize_DB(in_fpath=self[f"{key}_DB_file"],
                             dtypes=self["dtypes"], sub_col=self.COLS.ID.sub,
                             ses_col=self.COLS.ID.ses,
                             out_fpath=self.out_fpaths.get(key), 
                             debugging=self.debugging,
                             img2hdr_fpath=self.img2hdr_fpath, **kwargs)


    def make_outfile_path(self, uniq_fname_part: str) -> str:
        """
        :param uniq_fname_part: String in output file name identifying what
                                this file will contain specifically
        :return: String, path to the new output file to save
        """
        return os.path.join(self["output"],
                            make_default_out_file_name(uniq_fname_part))
    

    def save_subj_ses_list(self, uniq_fname_part: str,
                           which: pd.Series) -> None:
        """
        Save subject/session ID pairs from self.df into a .tsv file
        :param uniq_fname_part: String in output file name identifying what
                                this file will contain specifically
        :param which: pd.Series[bool]; True to include in saved file else False
        """
        self.df[which].to_csv(self.make_outfile_path(uniq_fname_part),
                              sep="\t", index=False, header=True,
                              columns=self.COLS.ID.sub_ses)  # [self.COLS.sub_col, self.COLS.ses_col])
        

    def save_summary(self, uniq_fname_part: str, which_cols: str = list()
                     ) -> None:
        """
        Save certain columns of the audit into a .tsv file
        :param uniq_fname_part: String in output file name identifying what
                                this file will contain specifically
        :param which_cols: Strings, each naming a column of df_DB to include
                           in the output .tsv file 
        """
        kwargs4save = dict(sep="\t", index=True, header=True)
        if which_cols:
            kwargs4save["columns"] = which_cols
        try:
            fpath = self.make_outfile_path(uniq_fname_part)
            self.df.to_csv(fpath, **kwargs4save)
        except TypeError as e:
            self.debug_or_raise(e, locals())


    def save_all_summary_tsv_files(self) -> None:
        """
        Save summary dataframes to .tsv files 
        """
        self.lazysetdefault("df", self.summarize)  # Get summary df

        # Save summary dataframe(s) of coverage info for all subject sessions
        self.save_summary("audit-summary", ("complete", "coverage_s3",
                                            "coverage_tier1"))
        self.save_summary("audit-full")

        # Save lists of all (in)complete subject sessions included in audit
        self.df.reset_index(inplace=True)
        is_checked_ses = self.df['session'].isin(set(BUCKET2SES.values()))
        are_complete = ((self.df['coverage_s3'] == 1.0) |
                        (self.df['coverage_tier1'] == 1.0))
        cmplt = {"complete": are_complete, "incomplete": ~are_complete}
        for which, completeness in cmplt.items():
            # for completeness in (are_complete, ~are_complete):
            self.save_subj_ses_list(f"{which}-subj-ses",
                                    completeness & is_checked_ses)
            self.save_subj_ses_list(f"no-files-{which}", completeness &
                                    ~self.df["any_files"] & is_checked_ses)


    def summarize(self) -> None:
        with ShowTimeTaken("summarizing audit"):
            self.df = self.make_summary_df()


    def make_summary_df(self) -> pd.DataFrame:
        """
        Get boolified versions of each df, then combine them by
        assigning a separate value for any combination of:
        - "bids" if FastTrackQCDB.df(.ftq_usable==1.0) else "delete"
        - "(tier1)" if Tier1BidsDB.df(.NGDR_fpath) else ""
        - "(s3)" if S3BidsDB.df(.s3_subpath) else ""
        The result will include "bids (s3) (tier1)", "delete (tier1)", etc
        :return: pd.DataFrame mapping each subject session row and specific
                 dtype/detail column to strings summarizing coverage
        """
        self.LOC_KEYS = ("tier1", "s3")
        booled_list = [eachDB.df.fillna(False).applymap(bool)
                       for eachDB in self.DB]

        # Combine all dataframes' indices so each has every subject session
        booled_list = mutual_reindex(*booled_list, fill_value=False)
        self.key_of = invert_dict(self.ix_of)
        self.booled = LazyDict({self.key_of[ix]: booled_list[ix]
                                for ix in range(len(booled_list))})

        # Summarize results; identify subject sessions not matching expectations
        self.COLS.booled = booled_list[0].columns
        for i in range(1, 3):
            self.COLS.booled = \
                self.COLS.booled.intersection(booled_list[i].columns)
        detail_df = self.booled["ftqc"].apply(self.summarize_col)
        any_files_bool_col = detail_df.apply(lambda row: row.dropna().apply(
            lambda cell: (cell["s3"] or cell["tier1"])
        ).any(), axis=1)
        summary_df = detail_df.fillna("").applymap(
            lambda cell: cell if cell=="" else cell["summary"]
        )
        summary_df["any_files"] = any_files_bool_col

        # Calculate coverage/completion percentage(s) for each tier
        self.row_NaN_count = detail_df.isna().sum(axis=1)
        for location in self.LOC_KEYS:
            COVG_LOC = f"coverage_{location}"
            summary_df[COVG_LOC] = detail_df.fillna(1.0).applymap(
                lambda cell: cell if cell==1.0 else cell[COVG_LOC]
            ).apply(self.get_subj_ses_covg, axis=1)
        summary_df["complete"] = summary_df.apply(self.sub_ses_is_complete,
                                                  axis=1)
        return summary_df


    def sub_ses_is_complete(self, sub_ses_row: pd.Series) -> str:
        """
        :param sub_ses_row: a self.df row with the coverage percentage for
                            tiers 1 and 2 for one subject session
        :return: "incomplete" if some files expected from abcd_fastqc01.txt
                 are not present on s3 or tier1 for the subject session; else
                 "complete" plus the tiers that the files are present on
        """
        is_complete_on = list()
        for loc_key in self.LOC_KEYS:
            if sub_ses_row.get(f"coverage_{loc_key}") == 1.0:
                is_complete_on.append(loc_key)

        # If is_complete_on neither or both, say so; if on only 1, name which
        return LazyDict({2: "complete (both)", 0: "incomplete"}).lazyget(
            len(is_complete_on), lambda: f"complete ({is_complete_on[0]})"
        )


    def summarize_col(self, ftqc_col: pd.Series) -> pd.Series:
        """
        :param ftqc_col: pd.Series, Fast Track QC DB df column to summarize
        :return: pd.Series[str] of summary strings describing whether each
                 file is QCd and/or present on tier 1/2
        """
        return (ftqc_col.to_frame(ftqc_col.name).apply(
                    self.summarize_1_cell, axis=1
                ) if ftqc_col.name in self.COLS.booled
                else pd.Series().reindex_like(ftqc_col))
    

    def summarize_1_cell(self, sub_ses_cell: pd.Series) -> Dict[str, Any]:
        """
        :param sub_ses_cell: pd.Series of subject session data to summarize
        :return: Dict[str, Any] summarizing whether the subject session's
                 file(s) is/are QCd and/or present on tier 1/2
        """
        details = {"ftqc": sub_ses_cell[0]}
        for key in self.LOC_KEYS:
            details[key] = self.booled[key].loc[sub_ses_cell.name,
                                                sub_ses_cell.index[0]]
            details[f"coverage_{key}"] = float(details["ftqc"] == details[key])
        details["summary"] = build_summary_str(**details)
        return details
    