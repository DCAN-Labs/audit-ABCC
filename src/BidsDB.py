#!/usr/bin/env python3

"""
ABCC Audit BIDS DB Object Classes
Greg Conan: gregmconan@gmail.com
Created 2024-01-18
Updated 2024-01-18
"""
from datetime import datetime
from glob import glob
import numpy as np
import os
import pandas as pd
from src.utilities import (
    BareObject, BIDS_COLUMNS, get_and_print_time_if, make_ERI_filepath
)


class FullBidsDB:
    def __init__(self, cli_args) -> None:
        self.dicomDB = BidsDB(in_files=cli_args["DB"]) # TODO


class BidsDB:
    def __init__(self, in_files=list(), tier1_dir=None,
                 out_files=list(), df=None) -> None:
        self.fpath = BareObject()
        # self.fpath.in_from = in_files
        self.fpath.out_to = out_files

        if df is not None:
            self.df = df
            self.COLS = self.get_COLS()
        elif in_files:
            self.df = self.read_df_from(in_files)
            self.COLS = self.get_COLS()
        elif tier1_dir:
            self.COLS = self.get_new_ID_col_names()
            self.df = self.make_df_from_tier1_dir(tier1_dir)
        else:
            print("ERROR: NO DATAFRAME")

        self.subset = BareObject()
        self.subset.with_anat = self.query_anat()
        self.subset.complete = self.query_processed_subjects()
        self.subset.unprocessed = self.query_unprocessed_subjects()


    def get_COLS(self):
        COLS = BareObject()
        COLS.all = self.df.columns.values
        COLS.ID = BareObject()
        COLS.ID.sub = COLS.all[0]
        COLS.ID.ses = COLS.all[1]
        COLS.task = [col for col in COLS.all.tolist() if col[:5] == "task-"]

        # Names of all columns in self.df EXCEPT subject- or session-ID cols
        COLS.ID.non = self.df.columns.values.tolist()
        for name_of_ID_col in (COLS.ID.sub, COLS.ID.ses, *BIDS_COLUMNS):
            try: COLS.ID.non.remove(name_of_ID_col)
            except ValueError: pass

        return COLS


    def get_new_ID_col_names(self):
        COLS = BareObject()
        COLS.ID = BareObject()
        COLS.ID.sub = BIDS_COLUMNS[0]
        COLS.ID.ses = BIDS_COLUMNS[1]
        COLS.tier1_dir = "tier1_dirpath"
        return COLS
    

    def get_processed_rows(self):
        return self.df[~self.df[self.df.columns[2:]
                                ].isin(['no bids']).any(axis=1)]


    def get_row_with_sub_ses_matching(self, orig_row):
        """
        :return: pandas.Series, the row of db_ERI with the same subject and
                session as orig_row
        """
        return self.df[
            (self.df[self.COLS.ID.sub] == orig_row.get(self.COLS.ID.sub)) &
            (self.df[self.COLS.ID.ses] == orig_row.get(self.COLS.ID.ses))
        ].iloc[0]
        

    def get_session_names(self):
        return self.df[self.COLS.ID.ses].unique().tolist()


    def get_sub_ses_cols(self):
        """
        :param a_db_df: pandas.DataFrame
        :return: Tuple of 2 strings, the names of the subject and session columns
        """
        return self.COLS.ID.sub, self.COLS.ID.ses # self.df.columns.values[0:1]


    def get_sub_ses_df_from_tier1(tier1_dirpath, sub_col, ses_col):
        path_col = "tier1_dirpath"

        # Get DF of all subjects and their sessions in the NGDR space
        all_sub_ses_NGDR_paths = glob(os.path.join(tier1_dirpath, "sub-*", "ses-*"))
        # NOTE Later we can verify specific files existing instead of just the
        #      session directories existing
        all_sub_ses_NGDR = pd.DataFrame({path_col: [path for path in all_sub_ses_NGDR_paths]})
        all_sub_ses_NGDR[sub_col] = all_sub_ses_NGDR[path_col].apply(
            lambda path: os.path.basename(os.path.dirname(path))
        )
        all_sub_ses_NGDR[ses_col] = all_sub_ses_NGDR[path_col].apply(
            lambda path: os.path.basename(path)
        )
        return all_sub_ses_NGDR


    def make_df_from_tier1_dir(self, tier1_dirpath):

        # Get DF of all subjects and their sessions in the NGDR space
        all_NGDR_paths = glob(os.path.join(tier1_dirpath, "sub-*", "ses-*"))
        # NOTE Later we can verify specific files existing instead of just the
        #      session directories existing
        df_NGDR = pd.DataFrame({self.COLS.tier1_dir:
                                [path for path in all_NGDR_paths]})
        df_NGDR[self.COLS.ID.sub] = df_NGDR[self.COLS.tier1_dir].apply(
            lambda path: os.path.basename(os.path.dirname(path))
        )
        df_NGDR[self.COLS.ID.ses] = df_NGDR[self.COLS.tier1_dir].apply(
            lambda path: os.path.basename(path)
        )
        return df_NGDR


    def rename_sub_ses_cols(self, new_sub_col, new_ses_col):
        self.df = self.df.rename(columns={
            self.COLS.ID.sub: new_sub_col,
            self.COLS.ID.ses: new_ses_col
        })
        self.COLS.ID.sub = new_sub_col
        self.COLS.ID.ses = new_ses_col


    def query_anat(self):
        return self.df.filter(regex='T[1,2].*').isin([np.nan]).all(axis=1)
    

    def query_ses(self, session):
        return self.df[self.COL.ID.ses] == session


    def query_processed_subjects(self):
        """
        Filter dataframe to get dataframe of subjects that do not have any unprocessed images
        """
        processed_df = self.df[~self.df[self.df.columns[2:]
                                        ].isin(['no bids']).any(axis=1)]
        # Filter again to remove subjects that have need data deleted
        return ~processed_df.isin(
            ['delete (tier1)', 'delete (s3)']
        ).any(axis=1) & self.subset.anat  # Remove subjects without a T1
    

    def query_unprocessed_subjects(self):
        """
        Check for fully unprocessed subjects
        """
        # Filter dataframe to get dataframe of subjects that are missing one or more modalities
        missing_data_df = self.get_processed_rows()
        # Filter again to remove subjects that have BIDS data somewhere
        return ~missing_data_df.isin(
            ['bids (tier1)', 'delete (tier1)', 'bids (s3)', 'delete (s3)']
        ).any(axis=1)


    def read_df_from(self, in_files):
        final_df = None
        if in_files:
            dfs_to_concat = list()
            for each_csv_path in in_files:  
                dfs_to_concat.append(pd.read_csv(each_csv_path))
                # assert tuple(already_uploaded[-1].columns.values.tolist()) == BIDS_COLUMNS
            final_df = pd.concat(dfs_to_concat)
        return final_df


class ERIBidsDB(BidsDB):

    # def expects_ERI(sub_ses_row, task_cols):
    #     sub_ses_row.apply()

    def __init__(self, cli_args, in_files=list(), out_files=list(), df=None) -> None:
        super().__init__(in_files, out_files, df)
        self.in_fpaths = in_files
        self.out_dir = cli_args["output"]
        self.out_fpaths = out_files


    def get_col_name(self):
        return f"ERI from {os.path.basename(self.in_fpaths[0])}"


    def get_info(self):
        saved_ERI_tier = dict()
        for tier in (1, 2):
            if os.path.exists(self.get_fpath_tier(tier)):
                saved_ERI_tier[tier] = pd.read_csv()
        return saved_ERI_tier


    def get_fpath_tier(self, tier):
        return os.path.join(self.out_dir, f"ERI_tier{tier}_paths_bids_db.csv")


    def replace_NaNs_for_sub_ses(self, row_ERI_DB_1):
        """
        :return: pandas.Series which replaced as many NaN values as it could in
                row_ERI_DB_1 with values from the same column/row in db_ERI_2.
        """
        return row_ERI_DB_1.combine_first(
            self.get_row_with_sub_ses_matching(row_ERI_DB_1)
        )


    def count_Falses_in(self, row):
        miss_row = self.get_row_with_sub_ses_matching(row)
        miss_row_tasks = miss_row.get(self.COLS.task)
        return len(miss_row_tasks[miss_row_tasks == False])


    def count_ERI_paths_in(self, row):
        row_w_ERI_paths = self.get_row_with_sub_ses_matching(row)
        return row_w_ERI_paths.get(self.COLS.task).count()


    def get_all_missing_ERI_paths_for(self, missing_ERI_list): #, combined_ERI_db, sub_col, ses_col):
        result = list()
        for task_run in missing_ERI_list[2:]:
            result.append(make_ERI_filepath(
                "", missing_ERI_list[0], missing_ERI_list[1],
                *[x.split("-")[-1] for x in task_run.split("_")]
            ))
        return result


    def get_all_task_runs_w_missing_ERI(self, sub_ses_row):
        has_ERI_row = self.replace_NaNs_for_sub_ses(sub_ses_row)
        return [sub_ses_row.get(self.COLS.ID.sub), sub_ses_row.get(self.COLS.ID.ses),
                *has_ERI_row[has_ERI_row==False].index.values.tolist()] # ",".join(


    def save_counts_csv(self, all_ses):
        """
        Save out .csv file with the following columns; the .csv will help for rerunning abcd-dicom2bids:
        1. Subject ID
        2. Session
        3. Number of task runs that this subject session is missing ERI files for
        4. Total number of task runs for this subject session
        """
        MISS = "Missing-ERI"
        sub_ses_cols = self.get_sub_ses_cols()
        counts_ERI = all_ses[MISS][sub_ses_cols].copy() # pd.DataFrame(index=all_sessions["Missing-ERI"].index)
        counts_ERI["Present"] = all_ses[MISS].apply(self.count_ERI_paths_in,
                                                    axis=1)
        counts_ERI["Missing"] = all_ses[MISS].apply(self.count_Falses_in, axis=1)
        counts_ERI["Total"] = counts_ERI["Present"] + counts_ERI["Missing"]
        # print("Total ERI files per subject:\n{}".format(counts_ERI["Total"].iloc[0:5]))
        counts_ERI.to_csv(
            os.path.join(self.out_dir, "sub-ses-missing-ERI-counts.csv"),
            columns=[*sub_ses_cols, "Missing", "Total"], index=False
        )


    def get_BIDS_status_DBs_for(self, session, dcm_DBs, col, combined_ERI_db, sub_ses_df_BIDS_NGDR, is_verbose):
        """
        For each subject-session,
        
            If subject-session completely succeeded abcd-dicom2bids conversion,

                Get whether the subject-session has already been uploaded from the NDA upload files
                - Input: Path to NDA upload submission working directory, list of subject-sessions, path(s) to .csv file(s) with subject-sessions already uploaded to the NDA so those can be excluded from the output list: cli_args["uploaded"]
                - Output: List of subject-sessions to upload to the NDA, which needs to be saved out as a .csv with subject and session column

                Verify that the subject-session exists on the NGDR space
                - Input: NGDR space path(s) where subject-sessions should exist, list of subject-sessions
                - Output: List of subjects to move to the NGDR space, which needs to be a text file mapping (for each subject-session) s3 bucket paths to NGDR paths (one mapping per line, 2 space-separated paths) 
        """
        # sub_ses_cols = [col["sub"], col["ses"]]

        # Split dicoms_db by session and by whether BIDS conversion succeeded
        # session_db = dcm_DBs["with_anat"].loc[dcm_DBs["with_anat"]["session"] == session]
        ses = self.query_ses(session)
        status = {
            "Total":     ses & self.subset.anat,
            "Succeeded": ses & self.subset.complete,
            "Failed":    ses & self.subset.unprocessed,
            "No-anat": dcm_DBs["no_anat"][dcm_DBs["no_anat"]["session"] == session],
        }

        if "uploaded_with_anat" in dcm_DBs:
            status_DB["Already-uploaded"] = dcm_DBs["uploaded_with_anat"][
                dcm_DBs["uploaded_with_anat"]["session"] == session
            ]  # TODO We also want the opposite of this (subject-sessions with anat that have not been uploaded) and, of those, how many succeeded BIDS conversion

        # Check how many subjects should have ERI, but don't
        before_check = datetime.now()
        ses_has_needed_ERI = status_DB["Total"].apply(
            
            lambda row: check_whether_sub_ses_has_ERI(row, combined_ERI_db,
                                                        *sub_ses_cols,
                                                        col["tasks"]), axis=1
        )
        get_and_print_time_if(is_verbose, before_check, "checking "
                                "whether every sub has ERI for ses " + session)
        has_needed_ERI_col = ses_has_needed_ERI.apply(lambda row: all([
                is_truthy(row.get(col)) for col in col["tasks"]
            ]), axis=1)
        status_DB["Has-All-ERI"] = status_DB["Total"][has_needed_ERI_col]
        status_DB["Missing-ERI"] = status_DB["Total"][~has_needed_ERI_col]

        # Identify which subject session task runs specifically have missing ERI
        ses_missing_ERI = status_DB["Missing-ERI"].apply(lambda row:
            get_all_task_runs_w_missing_ERI(row, ses_has_needed_ERI,
                                            *sub_ses_cols), axis=1
        )
        # all_missing_ERI.append(ses_missing_ERI)
        if is_verbose:
            print("{} subjects are missing ERI."
                    .format(len(ses_missing_ERI.index)))

        # For subjects who succeeded with anat, check which made it to the NGDR
        status_DB["NGDR"] = pd.merge(
            status_DB["Succeeded"], sub_ses_df_BIDS_NGDR,
            how="inner", on=sub_ses_cols #
        )
        status_DB["Not yet NGDR"] = pd.merge(
            status_DB["Succeeded"], status_DB["NGDR"],
            how="outer", on=[*sub_ses_cols, *col["non-ID"]]  # TODO This will break with no uploads .csv, fix it
        )

        return status_DB, ses_missing_ERI, ses_has_needed_ERI
