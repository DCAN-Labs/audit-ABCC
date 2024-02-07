#!/usr/bin/env python3

"""
ABCC Audit BIDS DB Object Classes
Greg Conan: gregmconan@gmail.com
Created: 2024-01-18
Updated: 2024-02-07
"""
# Standard imports
from datetime import datetime
import functools
from glob import glob
import operator
import pdb
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional
import urllib

# External imports
import boto3
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame

# Local imports
from src.utilities import (
    BIDS_COLUMNS, build_NGDR_fpath, boolify_and_clean_col, debug_or_raise,
    DTYPE_2_UNIQ_COLS, explode_col, get_and_log_time_since,
    get_col_headers_for, ImageTypeIn, is_nothing, LazyDict, log,
    make_col_header, make_col_header_from,
    mutual_reindex, PATH_NGDR, reformat_BIDS_df_col, reformat_pGUID,
    RegexForBidsDetails, run_num_to_str, SESSION_DICT, TaskNames,
)

# Constants:

# Map s3 bucket name to the name/ID of the session it has data from
BUCKET2SES = {"ABCC_year1": "ses-baselineYear1Arm1",
              "ABCC_year2": "ses-2YearFollowUpYArm1",
              "ABCC_year4": "ses-4YearFollowUpYArm1"}

DEFAULT_ID_COLS = ("subject", "session")

# FastQCDB.df.image_description substrings meaning that...
FTQC_DWI = {'QA_ABCD', 'DTI_ABCD'}  # ...dtype="dwi"
FTQC_T2_OUTLIERS = {'fMRI-FM-PA_ABCD-Diffusion-FM-PA_ABCD-T2',
                    "T2_ABCD-T2-NORM"}  # ...header="T2_run-{run}"

RGX_SPLIT = RegexForBidsDetails("anat", "audit", "dtype",
                                "dwi", "fmap", "func")  # , "dwi")  # TODO ?


class FullBidsDB:
    def __init__(self, cli_args: Mapping[str, Any]) -> None:
        self.DB = LazyDict()
        self.DB.dicom = BidsDB(in_paths=cli_args["DB"]) # TODO
        self.DB.tier1 = Tier1BidsDB(tier1_dir=PATH_NGDR,
                                    sub_col=BIDS_COLUMNS[0],
                                    ses_col=BIDS_COLUMNS[1])


class BidsDBColumnNames(LazyDict):


    def __init__(self, prev_COLS:"BidsDBColumnNames" = None,
                 sub:Optional[str] = None, ses:Optional[str] = None,
                 **kwargs: Any) -> None: # , df: pd.DataFrame=None
        """
        :param df: pd.DataFrame that this object represents the columns of
        :param prev_COLS: BidsDBColumnNames object that already exists
        :param sub: String naming the subject/participant ID column
        :param ses: String naming the session ID column
        """
        if prev_COLS:
            self.update(prev_COLS)
        self.update(kwargs)
        self.lazysetdefault("ID", LazyDict)
        # if sub or ses:
        self.add_ID_COLS(sub_col=sub, ses_col=ses)
        if self.get("df") is not None:
            self.create_from_own_df()  # sub_col=sub, ses_col=ses)


    def add_ID_COLS(self, **kwargs: str) -> LazyDict: # sub_col=None, ses_col=None) -> LazyDict:
        # Subject ID and session ID column names
        for i in range(len(DEFAULT_ID_COLS)):
        # for i in range(len(BIDS_COLUMNS)): # ("sub", "ses"):
            col = DEFAULT_ID_COLS[i][:3]
            col_passed_in = kwargs.get(f"{col}_col")
            self.ID[col] = (col_passed_in if col_passed_in else
                            self.lazyget("all", lambda: DEFAULT_ID_COLS)[i])
        self.ID.sub_ses = [self.ID.sub, self.ID.ses]


    def add_dtype_COLS(self) -> None:
        # Column names for details about "func" and "anat" files
        for dtype, extra_cols in DTYPE_2_UNIQ_COLS.items():
            self.ID[dtype] = [*self.ID.sub_ses, *extra_cols]


    def create_from_own_df(self) -> None:  # , sub_col:Optional[str] = None, ses_col:Optional[str] = None) -> LazyDict:
        """
        :param sub_col: String naming the subject/participant ID column
        :param ses_col: String naming the session ID column
        :return: LazyDict, _description_
        """
        # self = self.df.get_or_make_LazyDict_child("COLS")
        self.lazysetdefault("all", lambda: self.df.columns.values)
        # self.add_ID_COLS(sub_col=sub_col, ses_col=ses_col)
        self.task = self.get_subset_with("task-")
        # self.task = [col for col in self.all if col[:5] == "task-"]
        self.ID.lazysetdefault("non", self.get_non_ID_cols)
        # return self
    

    def get_subset_with(self, to_find: str) -> list:
        return [col for col in self.all if re.search(to_find, col)]


    def get_non_ID_cols(self) -> pd.Series:
        """
        :return: pd.Series of all self.df column name strings
                 EXCEPT subject- or session-ID columns. 
        """
        return pd.Series(index=self.df.columns
                         ).drop(self.ID.sub_ses, errors="ignore"
                                ).reset_index().drop(columns=[0])["index"]


    def rename_sub_ses_cols(self, new_sub_col: str, new_ses_col: str) -> None:
        """
        :param new_sub_col: String, subject/participant ID column's new name
        :param new_ses_col: String, session ID column's new name
        """
        self.df.rename(columns={self.ID.sub: new_sub_col,
                                self.ID.ses: new_ses_col}, inplace=True)
        self.add_ID_COLS(sub_col=new_sub_col, ses_col=new_ses_col)
        return self.df


class BidsDB(LazyDict):
    
    # Functions below, constants above
    def __init__(self, in_files=list(), out_fpath: Optional[str] = None,
                 df: Optional[pd.DataFrame] = None,
                 sub_col: Optional[str] = None, ses_col: Optional[str] = None
                 ) -> None:
        self.lazysetdefault("fpath", LazyDict)
        # self.fpath = self.get_or_make_LazyDict_child("fpath")
        # self.fpath.setdefault("in_from", in_paths)
        # self.fpath.setdefault("out_to", out_fpath)

        if df is not None:  # "if df:" wouldn't work because bool(df) raises an exception
            self.df = df
        elif in_files:
            self.df = self.read_df_from(*in_files)
        else:
            print("ERROR: NO DATAFRAME")

        # self.COLS = self.get_COLS_from_own_df(sub_col, ses_col)
        # prev_COLS = self.get_or_make_LazyDict_child("COLS")
        self.COLS = BidsDBColumnNames(
            prev_COLS=self.lazysetdefault("COLS", LazyDict), df=self.df,
            sub=sub_col, ses=ses_col, fname="image_file"
        )
        self.set_sub_ses_cols_as_df_index()

        if out_fpath:
            self.save_to(out_fpath)


    def add_header_col_to(self, **dfs_BIDS: pd.DataFrame
                          ) -> Dict[str, pd.DataFrame]:
        """
        :param dfs_BIDS: Dict mapping data type string ("anat" or "func") to
                         a pd.DataFrame with data of that type
        :return: dict, dfs_BIDS but with a new column in its pd.DataFrame
                 values: the column header in the final self.df
        """
        for dtype, eachdf in dfs_BIDS.items():
            eachdf["header"] = eachdf.apply(
                lambda row: make_col_header_from(row, dtype), axis=1
            )
        return dfs_BIDS
    

    def explode_into_BIDS_df(self, df: pd.DataFrame, dtype: str,
                             *dropNaN_from_cols: str) -> pd.DataFrame:  # col_dropna:Optional[str] = None
                                   
        # Split filenames into 5 new columns
        # dtype_ID_cols = {"anat": COLS.ID.anat, "func": COLS.ID.func}[dtype]
        try:
            if is_nothing(df.get(self.COLS.ID.get(dtype))):
                df[self.COLS.ID.get(dtype)] = explode_col(df[self.COLS.fname],
                                                          RGX_SPLIT, dtype)
            if dropNaN_from_cols:
                # df = df.loc[~df[col_dropna].isna()]
                df.dropna(subset=dropNaN_from_cols, inplace=True)  # , inplace=True)
            assert not is_nothing(df)
        except (AssertionError, AttributeError, KeyError, ValueError) as e:
            debug_or_raise(e, locals())
        return df


    def get_empty_BIDS_DB_DF(self, *_: Any) -> pd.DataFrame:
        return pd.DataFrame(columns=[
            'subject', 'session', 'T1_run-01', 'T2_run-01', 'task-rest_run-01',
            'task-rest_run-02', 'task-rest_run-03', 'task-rest_run-04',
            'task-MID_run-01', 'task-MID_run-02', 'task-SST_run-01',
            'task-SST_run-02', 'task-nback_run-01', 'task-nback_run-02'
        ])


    def make_BIDS_files_dfs(self, **paths_BIDS: pd.DataFrame) -> pd.DataFrame:  # dtypes: Iterable[str], fpart: str
                   
        
        dfs_BIDS = dict()
        # self.COLS.lazysetdefault("dtype", dict)
        non_ID_COLS = list()
        # self.COLS.ID.lazysetdefault("non", list)
        for dtype, pathsdf in paths_BIDS.items():
            dfs_BIDS[dtype] = self.explode_into_BIDS_df(pathsdf, dtype).apply(
                reformat_BIDS_df_col
            )
            try:
                self.COLS[dtype] = get_col_headers_for(dtype, dfs_BIDS[dtype])
            except KeyError as e:
                debug_or_raise(e, locals())
            non_ID_COLS += list(self.COLS[dtype])
        self.COLS.lazysetdefault("all", lambda: [*self.COLS.ID.sub_ses,
                                                 *non_ID_COLS])
        self.COLS.ID.non = pd.Series(non_ID_COLS) 
        # self.COLS.all = self.get_all_col_headers(**dfs_BIDS) 
        dfs_BIDS = self.add_header_col_to(**dfs_BIDS)
        return (self.transform_dfs_to_BIDS_DB(self.COLS.fname, **dfs_BIDS)
                if dfs_BIDS else self.get_empty_BIDS_DB_DF())


    def make_subj_ses_dict(self, a_df: pd.DataFrame, dict_cols: Iterable[str],
                           value_col: str) -> Dict[str, Any]:
        sub_ses_dict = {col: None for col in dict_cols}
        if not a_df.empty:
            for col_ID in self.COLS.ID.sub_ses:
                sub_ses_dict[col_ID] = a_df[col_ID].iloc[0]
            def add_to_sub_ses_dict(row):
                sub_ses_dict[row.get("header")] = row.get(value_col)
            a_df.apply(add_to_sub_ses_dict, axis=1)
        return sub_ses_dict
    

    def read_DB_df_from(self, in_fpath: str) -> pd.DataFrame:
        """ 
        :param in_fpath: str, valid path to existing properly formatted BIDS DB
        :return: pd.DataFrame, _description_
        """
        try:
            return pd.read_csv(in_fpath, sep="\t",
                               index_col=self.COLS.ID.sub_ses)
        except (AttributeError, OSError) as e:
            debug_or_raise(e, locals())


    def set_sub_ses_cols_as_df_index(self):
        # Make the subject and session into the index
        if self.COLS.ID.sub in self.df and self.COLS.ID.ses in self.df:
            self.df.set_index(self.COLS.ID.sub_ses, inplace=True)
    

    def split_into_dtype_dfs(self, df: pd.DataFrame
                             ) -> Dict[str, pd.DataFrame]:
        # return df.groupby("dtype").to_dict
        dtype2ixs = df.groupby("dtype").indices
        try:
            return {dtype: df.iloc[ixs] for dtype, ixs in dtype2ixs.items()}  # This assumes numerical indices instead of sub-ses ones
        except KeyError as e:
            debug_or_raise(e, locals())
        # return {dtype: df[df["dtype"] == dtype] for dtype in self.dtypes}


    def read_df_from(self, *in_paths: str) -> pd.DataFrame:
        final_df = None
        if in_paths:
            dfs_to_concat = list()
            for each_csv_path in in_paths:  
                dfs_to_concat.append(pd.read_csv(each_csv_path))
                # assert tuple(already_uploaded[-1].columns.values.tolist()) == BIDS_COLUMNS
            final_df = pd.concat(dfs_to_concat)
        return final_df
                    

    def save_to(self, bids_DB_fpath: str) -> None:
        whichsep = "\t" if bids_DB_fpath.endswith(".tsv") else ","
        self.df.to_csv(bids_DB_fpath, index=True, sep=whichsep)


    def split_col_to_analyze(self, audit_col: pd.Series) -> pd.DataFrame:
        try: 
            audit_w_no_NaNs = audit_col.dropna()
            ploded = pd.DataFrame(explode_col(audit_w_no_NaNs,  # audit_col.dropna(),
                                              RGX_SPLIT, "audit"),
                                  index=audit_w_no_NaNs.index)
            ploded, _ = mutual_reindex(ploded, audit_col)
            for tier in (1, 2):
                col = boolify_and_clean_col(ploded.pop(tier))
                ploded.insert(tier, col.name, col)
            ploded.rename(columns={0: "operation"}, inplace=True)
        except (KeyError, ValueError) as e:
            debug_or_raise(e, locals())  # TODO
            ploded.drop(ploded.index, inplace=True)
        return ploded


    def transform_dfs_to_BIDS_DB(self, col_to_get: str, **dfs_BIDS:
                                 pd.DataFrame) -> pd.DataFrame:
        """
        :param col_to_get: String naming the column of the current/old self.df
                           to save values from in every column of the
                           final/new self.df
        :param dfs_BIDS: Dict mapping data type string ("anat" or "func") to
                         a pd.DataFrame with data of that type
        :return: pd.DataFrame, final self.df with one subject-session per row
        """
        new_df = dict()
        for dtype, eachdf in dfs_BIDS.items():
            new_df[dtype] = self.transform_1_df_to_BIDS_DB(col_to_get, eachdf,
                                                           dtype)
        if len(new_df) > 2:
            final_df = functools.reduce(self.merge_2_dfs_on_sub_ses,
                                        new_df.values())
        elif len(new_df) == 2:  # ==2:
            final_df = self.merge_2_dfs_on_sub_ses(*new_df.values())
            # final_df = pd.merge(left=new_df["anat"], right=new_df["func"], on=self.COLS.ID.sub_ses, how="outer", sort=True)
            # lambda df_L, df_R: pd.merge(left=df_L, right=df_R, on=self.COLS.ID.sub_ses, how="outer")
        elif len(new_df) == 1:
            final_df = next(dfs_BIDS.values())
        else:
            debug_or_raise(ValueError(), locals())  # TODO
        return final_df


    def merge_2_dfs_on_sub_ses(self, df_L: pd.DataFrame, df_R: pd.DataFrame
                               ) -> pd.DataFrame:
        return pd.merge(left=df_L, right=df_R, on=self.COLS.ID.sub_ses,
                        how="outer")


    def transform_1_df_to_BIDS_DB(self, col_to_get: str, df: pd.DataFrame,
                                  dtype: str) -> pd.DataFrame:
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
        new_df = df.groupby(self.COLS.ID.sub_ses).apply(
            lambda sub_ses_df: self.make_subj_ses_dict(
                sub_ses_df, self.COLS.lazysetdefault(
                    dtype, sub_ses_df["header"].unique
                ), col_to_get
            )
        )
        return pd.DataFrame(new_df.values.tolist(),
                            columns=[*self.COLS.ID.sub_ses,
                                     *self.COLS[dtype]])


class BidsDBToQuery(BidsDB):
    def __init__(self, a_DB: BidsDB) -> None:
        self.update(a_DB)
        self.subset = LazyDict({"with_anat": self.query_anat()})
        self.subset["processed"] = self.query_processed_subjects()
        self.subset["unprocessed"] = self.query_unprocessed_subjects()
        # super().__init__(in_files, out_fpath, df, sub_col, ses_col)
    

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


    def merge(self, otherDB: "BidsDB", keep:Optional[str] = "self"
              ) -> pd.DataFrame:
        return {"self": lambda: self.df.combine_first(otherDB.df),
                "other": lambda: otherDB.df.combine_first(self.df),
                "latest": lambda: self.smart_merge(otherDB)}[keep]()  # TODO


    def query(self, key_name: str) -> pd.DataFrame:
        return self.df[self.subset[key_name]]


    def query_anat(self) -> pd.DataFrame:
        return ~self.df.filter(regex='T[1,2].*').isin([np.nan]).all(axis=1)


    def query_processed_subjects(self) -> pd.DataFrame:
        """
        Filter dataframe to get dataframe of subjects that do not have any unprocessed images
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
        # Filter dataframe to get dataframe of subjects that are missing one or more modalities
        missing_data_df = self.get_processed_rows()
        # Filter again to remove subjects that have BIDS data somewhere
        return ~missing_data_df.isin(
            ['bids (tier1)', 'delete (tier1)', 'bids (s3)', 'delete (s3)']
        ).any(axis=1)


    def smart_merge(self, otherDB: "BidsDB") -> pd.DataFrame:
        self_df, other_df = mutual_reindex(self.df, otherDB.df)
        return self_df.combine(other_df, self.smart_merge_col, overwrite=False)


    def smart_merge_logic(self, row: pd.Series) -> str:
        """
        :param row: pd.Series, _description_
        :return: str, _description_
        """
        str_combo = [row["operation"]]
        if row["tier1"]:
            str_combo.append("(tier1)")
        if row["s3"]:
            str_combo.append("(s3)")
        return " ".join(str_combo)
        

    def smart_merge_col(self, col1: pd.Series, col2: pd.Series) -> pd.Series:
        # if set(col1.dropna().unique()).union(set(col2.dropna().unique())) == {""}:
        ploded1 = self.split_col_to_analyze(col1)
        ploded2 = self.split_col_to_analyze(col2)
        ploded1, ploded2 = mutual_reindex(ploded1, ploded2)
        
        if ploded1.empty or ploded2.empty:
            to_return = col1.combine_first(col2)
        else:
            tier_cols = ploded1.columns[1:3]
            analyzed_combo = ploded1[tier_cols].combine(
                ploded2[tier_cols], operator.or_, overwrite=False
            )  # TODO Remember which to keep??
            to_return = analyzed_combo.apply(self.smart_merge_logic,  axis=1)
        return to_return


class FastTrackQCDB(BidsDB):
 
    def __init__(self, fpath_QC_CSV: str, dtypes: Iterable[str],
                 in_fpath: Optional[str] = None,  # file_ext: str, 
                 sub_col:Optional[str] = None, ses_col:Optional[str] = None,
                 df: Optional[pd.DataFrame] = None,
                 out_fpath:Optional[str] = None) -> None:
        self.dtypes = dtypes
        # self.file_ext = file_ext
        self.TASKS = TaskNames()
        self.COLS = BidsDBColumnNames(
            sub=sub_col, ses=ses_col, temp="desc_and_dt", anat={"T1", "T2"},
            QC="QC", func=self.TASKS.get_all(), fname="reformatted_fname",
            prev_COLS=self.lazysetdefault("COLS", LazyDict)
        )
        self.COLS.add_dtype_COLS()
        if df is not None:
            self.df = df
        elif in_fpath:
            self.df = self.read_DB_df_from(in_fpath)
        else:
            self.df = self.make_FTQC_df_from(fpath_QC_CSV)
        super().__init__(out_fpath=out_fpath,  # in_paths=[fpath_QC_CSV], 
                         df=self.df, sub_col=sub_col, ses_col=ses_col)


    def add_cols_to_df_by_splitting(self, col_to_split: str) -> None:
        try:
            self.df[[self.COLS.ID.sub, self.COLS.ID.ses, self.COLS.temp]] = \
                self.df[col_to_split].str.split('_', 2).values.tolist()
            self.df[['image_description', 'datetimestamp']] = \
                self.df[self.COLS.temp].str.rsplit('_', 1).values.tolist()
            for which_ID in ("sub", "ses"):  # (self.COLS.ID.sub, self.COLS.ID.ses):
                which = self.COLS.ID[which_ID]
                self.df[which] = f"{which_ID}-" + self.df[which]
        except (KeyError, ValueError) as e:
            debug_or_raise(e, locals())


    def make_FTQC_df_from(self, fpath_QC_CSV: str) -> pd.DataFrame:
        """
        _summary_ 
        :param fpath_QC_CSV: str, _description_
        :return: pd.DataFrame, _description_
        """
        self.df = self.read_FTQC_df_from(fpath_QC_CSV)
        # self.df = self.df.loc[self.df["file_source"].str.endswith(self.file_ext)]  # TODO Combine w/ COLS.fpath                            
        self.TASKS = TaskNames()
        self.add_cols_to_df_by_splitting("ftq_series_id")
        self.rename_df_cols()
        self.df[["header", "dtype"]] = self.df.image_description.apply(
            self.img_dsc_to_col_header
        ).values.tolist()
        
        # Add run number by counting up from lowest datetimestamp
        self.COLS.ftqc = [*self.COLS.ID.sub_ses, "header"]
        try:  
            self.df["run"] = self.df.groupby(self.COLS.ftqc).apply(
                lambda df: self.enumerate_runs_in(df["datetimestamp"])
            ).reset_index(level=self.COLS.ftqc, drop=True)
        except TypeError as e:
            debug_or_raise(e, locals())

        # Add run numbers to header col
        self.df.dropna(subset=["header"], inplace=True)
        # pdb.set_trace()
        self.df["header"] = self.df.apply(  # TODO ENSURE THIS GETS RUN NUMBERS PROPERLY
            lambda row: make_col_header(row.get("header"), row.get("run")), axis=1
        )  # lambda row: make_col_header_from(row, row.get("header"))  # lambda row: make_col_header(row["header"], row["run"])
        self.COLS["fname"] = "image_file"  # TODO
        """
        # Separate the dwi data, and remove the irrelevant data
        self.ignorables = self.df.loc[self.df["header"].isna()]
        self.df = self.df.drop(self.ignorables.index)
        self.dwi_df = self.dwi_df.dropna(subset=["dtype"])  # TODO 

        xfm_args = dict()  # {"cols": dict()}
        for dtype in self.dtypes:
            xfm_args[dtype] = self.df[self.df["dtype"] == dtype]
        """
        try:
            # return self.transform_dfs_to_BIDS_DB(self.COLS.QC, **self.
            return self.make_BIDS_files_dfs_no_explode(
                **self.split_into_dtype_dfs(self.df)
            )
        except (AttributeError, KeyError, TypeError, ValueError) as e:
            debug_or_raise(e, locals())
    

    def make_BIDS_files_dfs_no_explode(self, **dfs_BIDS: DataFrame) -> DataFrame:
        self.COLS.ID.non = pd.Series(self.df["header"].unique())
        self.COLS.ID.non.sort_values(inplace=True)
        non_ID_COLS = self.COLS.ID.non.tolist()
        self.COLS.lazysetdefault("all", lambda: [*self.COLS.ID.sub_ses,
                                                 *non_ID_COLS])
        for dtype, eachdf in dfs_BIDS.items():
            self.COLS[dtype] = eachdf["header"].unique()
        # self.COLS.all = self.get_all_col_headers(**dfs_BIDS) 
        # dfs_BIDS = self.add_header_col_to(**dfs_BIDS)
        return (self.transform_dfs_to_BIDS_DB(self.COLS.QC, **dfs_BIDS)
                if dfs_BIDS else self.get_empty_BIDS_DB_DF())


    def enumerate_runs_in(self, ser: pd.Series) -> pd.Series:  # df: pd.DataFrame, COL: str)
        """
        Add run number by counting up from lowest datetimestamp 
        :param ser: pd.Series, _description_ 
        :return: pd.Series, _description_
        """
        uniqs = ser.unique()
        uniqs.sort()
        # if uniqs.shape[0] > 10:
        #     pdb.set_trace()
        dts2run = {uniqs[ix]: ix + 1 for ix in range(uniqs.shape[0])} # enumerate(uniqs, start=1)}
        # runs = np.arange(1, uniqs.shape[0]+1)
        return ser.apply(dts2run.get).astype(int)  # lambda stamp: dts2run[stamp])


    def img_dsc_to_col_header(self, image_description: str) -> list:
        """
        :param image_description: str, _description_
        :return: list with 2 items: specifier str (col header without run num)
                                    and dtype ("func", "anat", "misc"/"dwi",
                                    "fmap", or None)
        """
        if image_description in self.TASKS.inv:
            my_dtype = "func"
            my_col_header = self.TASKS.swap(image_description)
        else:
            split = image_description.split("-")
            parts = set(split)
            anat_header = parts.intersection(self.COLS.anat)
            if anat_header:
                my_dtype = "anat"
                for_header = [anat_header.pop()]
                if "NORM" in parts:
                    for_header.append("NORM")
                my_col_header = "-".join(for_header)
            # parts.remove("ABCD")
            elif parts.intersection(FTQC_DWI):
                my_dtype = "dwi"  # TODO
                my_col_header = "dwi"
            elif "FM" in parts:
                my_dtype = "fmap"
                my_col_header = "-".join(split[1:])
            elif image_description == 'ABCD-Diffusion-QA':
                my_dtype = "dwi"  # TODO
                my_col_header = "dwi"
            else:
                my_dtype = None
                my_col_header = None
        return [my_col_header, my_dtype]


    def compress_fastqc(self):
        self.df.apply(self.compress_fastqc_row, axis=1)


    def compress_fastqc_row(self, row):
        pass # row. TODO?


    def read_FTQC_df_from(self, fpath_QC_CSV: str) -> pd.DataFrame:
        # with open(fpath_QC_CSV) as qc_file:
        SEP_UNQUOTE = '["]*[,|\t|"]["]*' # Remove quotation marks
        return pd.read_csv(
            fpath_QC_CSV, encoding="utf-8-sig", sep=SEP_UNQUOTE,
            engine="python", index_col=False, quotechar='"',
            skiprows=[1]  # Skip row 2 (description)
            # dtype=collections.defaultdict(self.convert_quoted)
        ).dropna(how='all', axis=1)


    def rename_df_cols(self) -> None:
        # Change column names for good_bad_series_parser to use; then save to .csv
        self.df.rename({"ftq_usable": self.COLS.QC, "subjectkey": "pGUID",
                        "visit": "EventName", "interview_age": "SeriesTime",
                        "abcd_compliant": "ABCD_Compliant", "file_source":
                        "image_file", "comments_misc": "SeriesDescription"},
                        axis="columns", inplace=True)


class Tier1BidsDB(BidsDB):
    def __init__(self, tier1_dir: str, sub_col: str, ses_col: str,
                 file_ext: str, dtypes: Iterable[str], #dtypes=("anat", "func"),
                 in_fpath: Optional[str] = None,
                 out_fpath: Optional[str] = None) -> None:
        """
        :param tier1_dir: String, valid path to existing ABCD BIDS root 
                          directory on tier 1 with "sub-*" subdirectories
        :param sub_col: String naming the subject/participant ID column
        :param ses_col: String naming the session ID column
        :param file_ext: String, the extension of the tier 1 files to collect
        :param dtypes: tuple, _description_
        :param out_fpath: str, _description_, defaults to None
        """
        self.dtypes = dtypes
        self.file_ext = file_ext
        self.tier1_dir = tier1_dir

        # Get dataframe column names
        self.COLS = BidsDBColumnNames(
            fname="NGDR_fpath", tier1_dir="tier1_dirpath", sub=sub_col,
            ses=ses_col, prev_COLS=self.lazysetdefault("COLS", LazyDict)
        )  # prev_COLS=self.get("COLS"))
        # self.COLS = self.add_ID_COLS_to(self.get_or_make_LazyDict_child("COLS"), sub_col=sub_col, ses_col=ses_col)
        # self.COLS.fname = "NGDR_fpath"
        # self.COLS.tier1_dir = "tier1_dirpath"
        self.COLS.add_dtype_COLS()

        self.df = self.read_DB_df_from(in_fpath) if in_fpath else (
            # Collect all files of dtypes in tier 1 dir path, turn them into a
            # df with a column for each detail we need, then turn those details
            # into columns in the final df with 1 row per subject session
            self.make_BIDS_files_dfs(**self.get_BIDS_file_paths_df())
        )
        # self.df = self.get_df_for(dtypes, f"*.{file_ext}")

        # self.update_DB()  # TODO

        # Use the newly created dataframe to turn this into a BidsDB object
        super().__init__(df=self.df, out_fpath=out_fpath)  # in_paths=[tier1_dir], 


    def get_BIDS_file_paths_df(self) -> pd.DataFrame:  # Dict[str, List[str]]:
        return {dtype: pd.DataFrame({self.COLS.fname: glob(build_NGDR_fpath(
                    self.tier1_dir, dtype, f"*{self.file_ext}"
                ))}) for dtype in self.dtypes}


    def get_subj_ses_counts_dict(subj_ses_QC_DF: pd.DataFrame
                                 ) -> Dict[str, Any]:  # TODO Does this only do counts?
        """
        :param subj_ses_QC_DF: All df_QC rows for a specific subject and session,
                            but no others. Passed in by DataFrameGroupBy.apply
        :return: dict<str:obj> to add as a row in a new pd.DataFrame
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


    def make_sub_ses_dict_from(self, sub_ses_DB: pd.DataFrame
                               ) -> Dict[str, Any]:
        """
        :param sub_ses_DB: pd.DataFrame from DataFrameGroupBy(subj, session)
                           where the subject and session ID columns each only
                           have one string value  
        :return: dict to become this subject session's row in final self.df
        """
        sub_ses_dict = dict()  # Return value

        # Every subject- and session-ID value should be the same,
        # so just grab the first one for the dict to return
        for col_ID in sub_ses_DB.COLS.ID.sub_ses:
            sub_ses_dict[col_ID] = sub_ses_DB[col_ID].iloc[0]

        def add_to_sub_ses_dict(header: str) -> None:
            """
            Mark a sub_ses_DB row for keeping or for deletion
            :param header: String naming the sub_ses_DB column to modify
            """
            will_delete = (sub_ses_DB[header].empty or not
                           (sub_ses_DB[header] == "no bids").all())
            sub_ses_dict[header] = ("delete" if will_delete else "bids"
                                    ) + " (tier1)"
        
        # For every non-ID column name (to become final self.df column names),
        # note whether to keep that non-ID value for this subject session
        self.COLS.ID.non.apply(add_to_sub_ses_dict)
        return sub_ses_dict


    def update(self, otherDB: BidsDB) -> pd.Series:  #Series<dict> or list<dict>?
        return otherDB.df.groupby(otherDB.COLS.ID.sub_ses
                                  ).apply(self.make_sub_ses_dict_from)


class S3Bucket(BidsDB):

    def __init__(self, client: boto3.session.Session.client, bucket_name: str,
                 file_ext: str, dtypes: Iterable[str],
                 sub_col:Optional[str] = None,
                 ses_col:Optional[str] = None) -> None:
        """
        :param client: boto3.session.Session.client to access s3 buckets
        :param bucketName: String naming an accessible s3 bucket
        """
        self.client = client
        self.dtypes = dtypes
        self.file_ext = file_ext
        self.name = bucket_name
        self.session = BUCKET2SES[bucket_name]
        self.TASKS = TaskNames()
        self.COLS = BidsDBColumnNames(
            sub=sub_col, ses=ses_col, fname="s3_file_subpath",
            func=self.TASKS.get_all(), anat={"T1", "T2"}
        )
        self.df = self.get_bids_subject_IDs_df()  # sub_col)  make_b
        self.COLS.add_dtype_COLS()
        self.df = self.make_BIDS_files_dfs(**self.get_BIDS_file_paths_df())
        self.set_sub_ses_cols_as_df_index()
        super().__init__(df=self.df)
        # self.df = self.transform_df_to_make_BIDS_df()  # sub_col)
        # self.get_all_pages(prefix)
        # self.pages = self.download_pages()


    def download(self, key_name: str, subkey_name: str, Prefix: str,
                 **kwargs: Any) -> List[Dict[str, str]]:
        try: 
            s3_data = [s3_datum[subkey_name] for page in
                       self.client.get_paginator('list_objects_v2').paginate(
                Bucket=self.name, ContinuationToken='', EncodingType='url',
                Prefix=Prefix, FetchOwner=False, 
                StartAfter='', **kwargs
            ) for s3_datum in page[key_name]]
            #    s3_data.extend(page[key_name])  # "Contents"])
        except KeyError as e:
            s3_data = None
        return s3_data
            

    def get_BIDS_file_paths_df(self) -> Dict[str, pd.DataFrame]:
        start = datetime.now()
        self.df[self.COLS.fname] = self.df[self.COLS.ID.sub
                                           ].apply(self.get_BIDS_files_for)
        downloaded_dt = get_and_log_time_since("started checking s3 paths",
                                               start) # get_and_print_time_if(True, start, "started checking s3 paths")
        # all_files = self.df[self.COLS.fname].apply(pd.Series).stack().apply(urllib.parse.unquote)
        all_files = self.df[self.COLS.fname].explode(ignore_index=True).apply(
            urllib.parse.unquote
        )
        get_and_log_time_since("started unstacking BIDS file paths",
                               downloaded_dt)
        # get_and_print_time_if(True, downloaded_dt, "started unstacking BIDS file paths")
        try:
            df_all = all_files.loc[all_files.str.endswith(self.file_ext)
                                   ].to_frame()
            df_all["dtype"] = explode_col(df_all[self.COLS.fname],
                                          RGX_SPLIT, "dtype")
            return self.split_into_dtype_dfs(df_all) 
        except (IndexError, KeyError) as e:
            debug_or_raise(e, locals())


    def get_BIDS_files_for(self, subject_ID: str) -> Dict[str, list]:
        # return pd.Series(s3download(self, "CommonPrefixes", "Prefix", "")).str.extract(FIND.create('subj')).dropna()[0].apply(lambda subj_ID: [s3_datum["Key"] for page in s3bucketyear1.client.get_paginator('list_objects_v2').paginate(Bucket=s3bucketyear1.name, ContinuationToken='', EncodingType='url', Prefix=f"{subj_ID}/{self.session}/", FetchOwner=False, StartAfter='') for s3_datum in page["Contents"]])
        return self.download("Contents", "Key", f"{subject_ID}/{self.session}/")


    def get_bids_subject_IDs_df(self) -> pd.Series:  # , subj_ID_col_name: str) -> Dict[str, list]:
        return pd.Series(self.download(
            "CommonPrefixes", "Prefix", "", Delimiter="/"
        )).str.extract(RGX_SPLIT.create('subj')
                       ).dropna().rename(columns={0: self.COLS.ID.sub})  # subj_ID_col_name}) # [0].reset_index(drop=True)
        # subdir_names = [subject['Prefix'] for subject in self.download("CommonPrefixes", "")]


class S3BidsDB(BidsDB):
    def __init__(self, client: boto3.session.Session.client,
                 buckets:List[str], file_ext: str, dtypes: Iterable[str],
                 in_fpath:  Optional[str] = None,
                 out_fpath: Optional[str] = None,
                 sub_col: Optional[str] = None, ses_col: Optional[str] = None,
                 df: Optional[pd.DataFrame] = None) -> None:
        self.client = client
        self.file_ext = file_ext
        self.TASKS = TaskNames()
        self.COLS = BidsDBColumnNames(
            sub=sub_col, ses=ses_col, fname="s3_file_subpath",
            func=self.TASKS.get_all(), anat={"T1", "T2"}
        )
        if df is not None:
            self.df = df
        elif in_fpath:
            self.df = self.read_DB_df_from(in_fpath)
        else:
            self.buckets = [S3Bucket(client, name, file_ext, dtypes, sub_col,
                                     ses_col) for name in buckets]
            # self.buckets = LazyDict({name: S3Bucket(client, name, file_ext, sub_col, ses_col) for name in buckets})
            # self.df = df if df is not None else self.get_empty_BIDS_DB_DF()
            self.df = pd.concat([bucket.df for bucket in self.buckets])

        # Use the newly created dataframe to turn this into a BidsDB object
        super().__init__(df=self.df, out_fpath=out_fpath)


    def update(self, bidsDB: BidsDB) -> pd.DataFrame:  # cli_args):
        # bids_db = pd.read_csv(cli_args['bids_db_file'])
        df_DB = bidsDB.df
        sub_ses_dicts = list()
        for bucket in self.buckets:  # cli_args['buckets']:
            # bucket = S3Bucket(self.client, bucket_name)
            sub_ses_dicts.append(df_DB.groupby(bidsDB.COLS.ID.sub_ses).apply(
                bucket.make_subj_ses_dict
            ))  # groupby(["subject", "session"])
        # TODO Verify that the same sub/ses doesn't appear in multiple buckets;
        #      and if one does, then combine the multiple subject-session dicts
        return pd.DataFrame(sub_ses_dicts)