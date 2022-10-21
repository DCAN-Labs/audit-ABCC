#!/bin/usr/env python3

"""
ABCC Audit Main Script 
Greg Conan: gconan@umn.edu
Created 2022-07-06
Updated 2022-10-19
"""
# Standard imports
import argparse
from datetime import datetime
import numpy as np
import os
import pandas as pd
import sys
# import boto3  # TODO Use this for interfacing with s3

# Nonstandard imports
#import fastqc_audit as fastqc  # TODO import from https://github.com/DCAN-Labs/abcd-dicom2bids/blob/audit-dev/src/audit/fastqc_audit.py
#import s3_audit as s3  # TODO import from https://github.com/DCAN-Labs/abcd-dicom2bids/blob/audit-dev/src/audit/s3_audit.py
#import tier1_audit as tier1   # TODO import from https://github.com/DCAN-Labs/abcd-dicom2bids/blob/audit-dev/src/audit/tier1_audit.py
from src.utilities import (
    DICOM2BIDS, get_ERI_filepath, get_sub_ses_df_from_tier1, 
    get_tier1_or_tier2_ERI_db_fname, BIDS_COLUMNS, BIDSPIPELINE,
    PATH_ABCD_BIDS_DB, PATH_DICOM_DB, PATH_NGDR, query_split_by_anat,
    query_processed_subjects, query_has_anat, query_unprocessed_subjects,
    save_to_hash_map_table, valid_output_dir, valid_readable_dir,
    valid_readable_file,
)


# Functions


def main():
    cli_args = _cli([DICOM2BIDS, BIDSPIPELINE])
    subject_lists = dict()
    pd.set_option('display.max_columns', None)

    # DICOM DB FLOW
    dicom_db = get_dicom_db(cli_args)

    # DICOM2BIDS FLOW
    if DICOM2BIDS in cli_args["audit_type"]:
        subject_lists[DICOM2BIDS] = audit_abcd_dicom2bids(dicom_db, cli_args)

    # ABCD-HCP FLOW
    # abcd_bids_db = pd.read_csv(cli_args["pipeline_db"]) # get_abcd_hcp_db(cli_args)
    if BIDSPIPELINE in cli_args["audit_type"]:
        subject_lists[BIDSPIPELINE] = audit_abcd_bids_pipeline(cli_args)

    # AUDIT COUNT TABLE FLOW
    # TODO
    if cli_args["show_counts"]:
        # assert len(cli_args["audit_type"]) > 1
        # TODO Do this for both DICOM2BIDS and BIDSPIPELINE
        for audit_type in cli_args["audit_type"]:
            audit_count_table(cli_args, subject_lists[audit_type], audit_type)


    # DICOM DB FLOW

    # Get BIDS DB info from fastqc_audit, s3_audit, and tier1_audit, 
    # Input: Path to BIDS database file as a .csv (which may or may not exist already)
    # Output: Database as a .csv file containing
    #         - all subjects that could be based on the fast track QC (converted to BIDS)
    #         - whether each file succeeded or not

        # If (A) there's an updated fast-track or (B) fast-track outputs do not already exist; then
            # Run fast-track-QC BIDS DB script (fastqc_audit.py) 

        # If the user specifies that they have new dicom data; then
            # If some of user's new data is on tier1; then
                # Run tier1_audit.py functions
            # If some of user's new data is on s3; then
                # Run s3_audit.py functions


def update_w_sub_ses_ERI_counts(db_ERI, col_ERI, pre_eri_db, non_ID_col_names):
    pre_eri_db.loc[col_ERI] = db_ERI[non_ID_col_names].apply(lambda row: row.count(), axis=1)
    return pre_eri_db


def get_sub_ses_cols_from(a_db_df):
    """
    :param a_db_df: pandas.DataFrame
    :return: Tuple of 2 strings, the names of the subject and session columns
    """
    return a_db_df.columns.values[0], a_db_df.columns.values[1]


def audit_abcd_dicom2bids(dicom_db, cli_args, counts=dict()):
    # DICOM2BIDS FLOW
    sub_col, ses_col = get_sub_ses_cols_from(dicom_db)
    sessions=list(set(dicom_db[ses_col]))

    # Trim down database to only include subjects that have a T1 or T2  # TODO make this comment accurate
    # Input: BIDS DB for that year
    # Output: List of excluded subjects for that year, and BIDS DB with them excluded
    dicom_db_with_anat, dicom_db_no_anat = query_split_by_anat(dicom_db)

    # Get names of every column without identifying information or ERI counts
    non_ID_col_names = get_non_ID_col_names_from(
        dicom_db_with_anat, sub_col, ses_col, *BIDS_COLUMNS,
        *[get_ERI_col_name(fpath) for fpath in cli_args["ERI_DB"]]
    )

    # Get all subject sessions that user already uploaded to the NDA
    # TODO Reorganize so that we count the NGDR statuses before checking uploads
    already_uploaded = list()
    has_needed_ERI = dict()
    if cli_args.get("uploaded"):
        for each_csv_path in cli_args["uploaded"]:  
            already_uploaded.append(pd.read_csv(each_csv_path))
            assert tuple(already_uploaded[-1].columns.values.tolist()
                         ) == BIDS_COLUMNS
        already_uploaded = pd.concat(already_uploaded)  # TODO Maybe we should make this a subset of the dicom_db
        uploaded_with_anat = pd.merge(dicom_db_with_anat, already_uploaded,
                                      how="inner", left_on=[sub_col, ses_col],
                                      right_on=BIDS_COLUMNS, indicator=True)
        # uploaded_with_anat = uploaded_with_anat[uploaded_with_anat["_merge"] == "right_only"]
        # non_ID_col_names = get_non_ID_col_names_from(uploaded_with_anat, sub_col, ses_col, *bids_columns)
        uploaded_with_anat.columns = [sub_col, ses_col, *BIDS_COLUMNS, *non_ID_col_names, "_merge"]
        # non_ID_col_names.remove("_merge")

    # For each year, check/count every subject-session's BIDS conversion status
    ses_DBs = dict()
    sub_ses_df_BIDS_NGDR = get_sub_ses_df_from_tier1(PATH_NGDR, sub_col,
                                                     ses_col)

    # Count ERI files per subject session (TODO FIX & TEST)
    db_ERI = list() # dict()
    for eachfpath in cli_args["ERI_DB"]:
        col_ERI = get_ERI_col_name(eachfpath)
        db_ERI.append(pd.read_csv(eachfpath))  # [col_ERI]
        dicom_db_with_anat = update_w_sub_ses_ERI_counts(  # [get_ERI_col_name(eachfpath)]
                                db_ERI[-1], col_ERI, dicom_db_with_anat, non_ID_col_names
                             )

    combined_ERI_db = db_ERI[0].apply( # dicom_db_with_anat.apply(
        lambda row: get_ERI_for_sub_ses(row, db_ERI[1], sub_col, ses_col), axis=1
    )
    task_cols = [col for col in dicom_db_with_anat.columns.values.tolist()
                 if col[:5] == "task-"]

    all_missing_ERI = list()
    all_missing_ERI_paths = list()
    for session in sessions:
        counts[session] = dict()
        ses_DBs[session] = dict()

        # Split dicoms_db by session and by whether BIDS conversion succeeded
        session_db = dicom_db_with_anat.loc[dicom_db_with_anat["session"] == session]
        ses_DBs[session] = {
            "Total": query_has_anat(session_db),
            "Succeeded": query_processed_subjects(session_db),
            "Failed": query_unprocessed_subjects(session_db),
            "No-anat": dicom_db_no_anat[dicom_db_no_anat["session"] == session],
        }

        if "uploaded_with_anat" in locals():
            ses_DBs[session]["Already-uploaded"] = uploaded_with_anat[
                uploaded_with_anat["session"] == session
            ]  # TODO We also want the opposite of this (subject-sessions with anat that have not been uploaded) and, of those, how many succeeded BIDS conversion

        # Check how many subjects should have ERI, but don't
        before_check = datetime.now()
        has_needed_ERI[session] = ses_DBs[session]["Total"].apply(
            lambda row: check_whether_sub_ses_has_ERI(row, combined_ERI_db,
                                                      sub_col, ses_col,
                                                      task_cols), axis=1
        )
        if cli_args["verbose"]:
            
            print("Took {} to check whether every sub has ERI for ses {}"
                  .format(datetime.now() - before_check, session))           
        has_needed_ERI_col = has_needed_ERI[session].apply(lambda row: all([
                is_truthy(row.get(col)) for col in task_cols
            ]), axis=1)
        ses_DBs[session]["Has-All-ERI"] = ses_DBs[session]["Total"][has_needed_ERI_col]
        ses_DBs[session]["Missing-ERI"] = ses_DBs[session]["Total"][~has_needed_ERI_col]

        # Identify which subject session task runs specifically have missing ERI
        all_missing_ERI.append(ses_DBs[session]["Missing-ERI"].apply(lambda row:
            get_all_task_runs_w_missing_ERI(row, has_needed_ERI[session],
                                            sub_col, ses_col), axis=1
        ))
        if cli_args["verbose"]:
            print("{} subjects are missing ERI."
                  .format(len(all_missing_ERI[-1].index)))
        if not all_missing_ERI[-1].empty:
            all_missing_ERI_paths.append(  # Add this session's missing ERI to total
                all_missing_ERI[-1].apply(get_all_missing_ERI_paths_for)
            )

        # For subjects who succeeded with anat, check which made it to the NGDR
        ses_DBs[session]["NGDR"] = pd.merge(
            ses_DBs[session]["Succeeded"], sub_ses_df_BIDS_NGDR,
            how="inner", on=[sub_col, ses_col] #
        )
        ses_DBs[session]["Not yet NGDR"] = pd.merge(
            ses_DBs[session]["Succeeded"], ses_DBs[session]["NGDR"],
            how="outer", on=[sub_col, ses_col, *non_ID_col_names]  # TODO This will break with no uploads .csv, fix it
        ) 

        # Count how many subjects succeeded vs. failed BIDS conversion
        for convert_status in ses_DBs[session].keys():
            counts[session][convert_status] = len(ses_DBs[session][convert_status])

        # For each subject-session,
        
            # If subject-session completely succeeded abcd-dicom2bids conversion,

                # Get whether the subject-session has already been uploaded from the NDA upload files
                # Input: Path to NDA upload submission working directory, list of subject-sessions, path(s) to .csv file(s) with subject-sessions already uploaded to the NDA so those can be excluded from the output list: cli_args["uploaded"]
                # Output: List of subject-sessions to upload to the NDA, which needs to be saved out as a .csv with subject and session column

                # Verify that the subject-session exists on the NGDR space
                # Input: NGDR space path(s) where subject-sessions should exist, list of subject-sessions
                # Output: List of subjects to move to the NGDR space, which needs to be a text file mapping (for each subject-session) s3 bucket paths to NGDR paths (one mapping per line, 2 space-separated paths)
    # with open(os.path.join(cli_args["output"], "all_missing_ERI_paths.txt"), "w+") as outfile:
    #     outfile.write("\n".join(all_missing_ERI_paths))
    
    # Save lists of missing ERI to .csv files
    pd.concat(all_missing_ERI).to_csv(os.path.join(
            cli_args["output"], "all_missing_ERI_sub-ses-task-runs.csv"
        ), index=False, header=False)
    pd.concat(all_missing_ERI_paths).to_csv(os.path.join(
            cli_args["output"], "all_missing_ERI_paths_sub-ses-task-runs.csv"
        ), index=False, header=False)

    all_sessions = save_run_status_subj_lists(
        sessions, ses_DBs, [sub_col, ses_col], BIDS_COLUMNS, DICOM2BIDS, cli_args
    )

    # Save out .csv file with the following columns; the .csv will help for rerunning abcd-dicom2bids:
    # 1. Subject ID
    # 2. Session
    # 3. Number of task runs that this subject session is missing ERI files for
    # 4. Total number of task runs for this subject session
    counts_ERI = all_sessions["Missing-ERI"][[sub_col, ses_col]].copy() # pd.DataFrame(index=all_sessions["Missing-ERI"].index)
    counts_ERI["Present"] = all_sessions["Missing-ERI"].apply(
        lambda row: count_ERI_paths_in(row, combined_ERI_db, sub_col, ses_col, task_cols), axis=1
    )
    counts_ERI["Missing"] = all_sessions["Missing-ERI"].apply(
        lambda row: count_Falses_in(row, has_needed_ERI, sub_col, ses_col, task_cols), axis=1
    )
    counts_ERI["Total"] = counts_ERI["Present"] + counts_ERI["Missing"]
    print("Total ERI files per subject:\n{}".format(counts_ERI["Total"].iloc[0:5]))
    counts_ERI.to_csv(os.path.join(cli_args["output"],
                                   "sub-ses-missing-ERI-counts.csv"),
                      columns=[sub_col, ses_col, "Missing", "Total"],
                      index=False)

    need_to_copy_to_NGDR = pd.concat([ses_DBs[session]["Not yet NGDR"] for session in sessions])
    outfpath = os.path.join(cli_args["output"], "bids2ngdr_paths_hash_mapping.txt")
    save_to_hash_map_table(need_to_copy_to_NGDR, sub_col, ses_col,
                           cli_args["bids_dir"], PATH_NGDR, outfpath)  # TODO Right now this lists each path twice, so ensure that bids_dir and PATH_NGDR are replaced with their correct (useful) values

            # Else (if subject that at least partly failed abcd-dicom2bids conversion),

                # If subject failed due to timeout,  # NOTE This block will not be worth the effort if it requires reading log files
                    # Add that subject to list of subjects to rerun with more time
                # Else

                    # Add subject to a list of failed subject-sessions to save out as a text file (1 subject-session per line)
    return counts


def count_Falses_in(row, has_ERI_dfs, sub_col, ses_col, task_cols):
    ses = row.get(ses_col)
    miss_row = get_ERI_row_for_sub_ses(row, has_ERI_dfs[ses], sub_col, ses_col)
    miss_row_tasks = miss_row.get(task_cols)
    return len(miss_row_tasks[miss_row_tasks == False])


def count_ERI_paths_in(row, combined_ERI_db, sub_col, ses_col, task_cols):
    row_w_ERI_paths = get_ERI_row_for_sub_ses(row, combined_ERI_db, sub_col, ses_col)
    return row_w_ERI_paths.get(task_cols).count()


def get_all_missing_ERI_paths_for(missing_ERI_list): #, combined_ERI_db, sub_col, ses_col):
    result = list()
    for task_run in missing_ERI_list[2:]:
        result.append(get_ERI_filepath(
            "", missing_ERI_list[0], missing_ERI_list[1],
            *[x.split("-")[-1] for x in task_run.split("_")]
        ))
    return result


def get_all_task_runs_w_missing_ERI(sub_ses_row, has_ERI_df, sub_col, ses_col):
    has_ERI_row = get_ERI_row_for_sub_ses(sub_ses_row, has_ERI_df, sub_col, ses_col)
    return [sub_ses_row.get(sub_col), sub_ses_row.get(ses_col),
                     *has_ERI_row[has_ERI_row==False].index.values.tolist()] # ",".join(


def audit_abcd_bids_pipeline(cli_args): 
    counts = dict()

    # Read in ABCD BIDS Pipeline Database from .csv file path given by user
    pipeline_db = pd.read_csv(cli_args["pipeline_db"], index_col=0)

    # Get pipeline DB column names
    cols = dict()
    sub_ses_cols = get_sub_ses_cols_from(pipeline_db)
    cols["sub"], cols["ses"] = sub_ses_cols
    sessions=list(set(pipeline_db[cols["ses"]]))
    cols["struc"] = pipeline_db.columns.values[2]
    cols["task_runs"] = pipeline_db.columns.values[3:]

    # Pipeline DB Cell Values Guide: 
    # For every stage/task-run in a subject session:
    # - "ok" = Pipeline ran successfully; it made all expected outputs
    # - "NO_ABCD-HCP" = the pipeline has not run yet
    # - "failed" = the pipeline failed
    # - "NO BIDS" = there are no BIDS inputs
    # run_statuses = ["ok", "NO_ABCD-HCP", "failed", "NO BIDS"]
    # run_statuses ={"Successes": "ok", "Not Yet Ran": "NO_ABCD-HCP", "Failures": "failed", "Unable to Run": "NO BIDS"}

    ses_DBs = dict()
    status_DBs = dict()
    for session in sessions:
        counts[session] = dict()
        ses_DBs[session] = pipeline_db[pipeline_db[cols["ses"]] == session]
        status_DBs[session] = dict()
        if cli_args["verbose"]:
            print("Getting run statuses for session {}\n".format(session))
        
        ses_DBs[session]["Overall Status"] = ses_DBs[session].apply(
            lambda row: get_overall_pipeline_status_of_sub_ses(row, cols),
            axis=1
        )
        for run_status in ses_DBs[session]["Overall Status"].unique(): # run_statuses:
            status_DBs[session][run_status] = ses_DBs[session][  # ses_DBs[session][run_status]
                ses_DBs[session]["Overall Status"] == run_status
            ]
            counts[session][run_status] = len(status_DBs[session][run_status])

    # Save out .csv subj lists of pipeline successes, failures, etc.
    save_run_status_subj_lists(
        sessions, status_DBs, sub_ses_cols, BIDS_COLUMNS, BIDSPIPELINE, cli_args
    )

    return counts


def save_run_status_subj_lists(sessions, ses_DBs, cols, header,
                               script_name, cli_args):
    """
    Save a DB of every convert status into its own .csv file 
    :param sessions: List of strings, each naming a session
    :param ses_DBs: _type_, _description_
    :param cols: List/tuple with 2 strings: subject and session column names
    :param header: _type_, _description_
    :param script_name: _type_, _description_
    :param cli_args: _type_, _description_
    :return: _type_, _description_
    """
    all_sessions_status = dict()
    if cli_args["save_lists"]:
        uploads_subject_list_outfpath = os.path.join(
            cli_args["output"], "sub-ses-list_{}_{{}}.csv".format(script_name)
        )
        for convert_status in ses_DBs[sessions[0]].keys():
        # TODO If user provided list of previously-uploaded subject-sessions, 
        #      remove each subject session from this uploads_subject_list
            all_sessions_status[convert_status] = pd.concat([
                ses_DBs[ses][convert_status] for ses in sessions
            ])
            all_sessions_status[convert_status].to_csv( 
                uploads_subject_list_outfpath.format(
                    convert_status.replace(" ", "-")
                ), columns=cols, index=False, header=header
            )
    return all_sessions_status


def get_overall_pipeline_status_of_sub_ses(sub_ses_row, cols): # stage_col_names, sub_col, ses_col):
    """
    For a given row,
    - If a row either has "ok" in every stage OR "ok" in structural and 
      "NO BIDS" in every other stage, then return "Success"
    - If a row has "NO_ABCD-HCP" in all stages, then return "Not Yet Run"
    - If a row has "failed" anywhere, or is a mix of "ok" >1 time(s) and anything else, tell the user it failed
    :param sub_ses_row: pandas.Series, 1 (subject session) row in pipeline_db
    :param cols: Dictionary mapping the role of a column (or group of columns)
                 to the actual name(s) of that column(s) in pipeline_db
    :return: String, the overall status of a subject session row 
    """
    GOOD = "Success"
    HASNT = "Not Yet Run"
    FAIL = "Failure"
    CANT = "Unable To Run"

    # Get whether this subject session has structural data
    struc_status = sub_ses_row.get(cols["struc"])
    overall_status = {
        "ok": GOOD, "NO_ABCD-HCP": HASNT, "failed": FAIL, "NO BIDS": CANT
    }[struc_status]

    # Check the status at each stage
    ix = 0
    while (overall_status not in (FAIL, CANT)) and ix < len(cols["task_runs"]):
        overall_status = infer_overall_status_from(
            sub_ses_row.get(cols["task_runs"][ix]),  # Status of each task run
            overall_status, GOOD, HASNT, FAIL
        )
        ix += 1
    # if struc_status != "ok": sys.exit("{}\n{}\n{}".format(sub_ses_row, struc_status, overall_status))
    return overall_status


def infer_overall_status_from(stage_status, overall_status, GOOD, HASNT, FAIL):
    """
    :param stage_status: String, value of a cell in a pipeline_db stage column
    :param overall_status: String, a subject session row's overall status so
                           far before accounting for stage_status
    :param GOOD: String naming the success condition
    :param HASNT: String naming the "hasn't been tried yet" condition 
    :param FAIL: String naming the failure condition
    :return: String, a subject session row's overall status so far after
             accounting for stage_status
    """
    return consistent_xor_fail(overall_status, {  # NOTE This assumes that a mix of HASNT and GOOD is impossible
        "ok": GOOD, "NO BIDS": GOOD, "NO_ABCD-HCP": HASNT, "failed": FAIL
    }[stage_status], FAIL)


def consistent_xor_fail(overall_status, success, failure):
    """
    Return either success if the stage status is consistent with the previous
    overall status, or failure otherwise
    :param overall_status: String, the overall status so far
    :param success: String naming the success condition
    :param failure: String naming the failure condition
    :return: String, a stage's/task-run's status
    """
    return success if overall_status == success else failure 


    # stage_statuses = [sub_ses_row.get(stage) for stage in stage_col_names]


    # ABCD-HCP FLOW

    # Read in ABCD-HCP audit .csv - get_abcd_hcp_db

    # For every row of the audit .csv (every subject-session),

        # If subject-session completely succeeded, then
        
            # Compare the processed output in the S3 (or tier1) with the derivatives directory
            # If the subject-session is already file-mapped to the derivatives directory

                # Add subject-session to another hash table text file mapping subject-session tier1/s3 derivatives path to the NGDR derivatives path: map_paths_from_storage_to_NGDR

                # Add subject-session to a list to save to a .csv file which will be used to upload to the NDA

            # Else (if file-mapper failed),

                # Add the subject-session to a list of subject-sessions to run file-mapper on

        # Elif subject-session says "no BIDS" in any of the Minimal Processing or DCANBold columns, then
        
            # Add subject-session to a list of anatomical subject-sessions

        # Elif subject-session says "failed" or "NO_ABCD-HCP" in any of the columns, then 
        
            # Add subject-session to a triage list


def audit_count_table(cli_args, counts, audit_type):
    # AUDIT COUNT TABLE FLOW (for Damien)

    # Finally, create a table with subject counts
    # For each session (AKA year),
        # For pipeline in (abcd-dicom2bids, ABCD-HCP), add a row showing how many subjects in session
            # - failed thru pipeline,
            # - succeeded thru pipeline,
            # - succeeded thru pipeline AND are pulled to NGDR,
            # - succeeded thru pipeline AND are pulled to NGDR, AND are uploaded to the NDA
            # - need to be triaged

    counts_df = pd.DataFrame(counts)
    print(counts_df)
    counts_df.to_csv(os.path.join(cli_args["output"],
                                  "counts_{}.csv".format(audit_type)),
                     index=True, header=True)
    """
    print("{} subjects in {}{}: {}"  # TODO Move the "print" functionality to audit count table flow section
            .format(convert_status, session,
                    "" if convert_status[:2].lower() == "no" else " with anat",
                    counts[session][convert_status]))
    """
            # If pipeline == ABCD-HCP, then also add a row showing how many subjects in session
                # - failed to file-map


# More Utility Functions


# def expects_ERI(sub_ses_row, task_cols):
#     sub_ses_row.apply()


def check_whether_sub_ses_has_ERI(row, combined_ERI_DB, sub_col, ses_col, task_cols):
    """
    :param row: pandas.Series, a row of the bids_db
    :param combined_ERI_DB: pandas.DataFrame which has tier 1 and 2 ERI paths
    :param sub_col: String naming the column in combined_ERI_DB, and the label
                    in row, which has the subject ID
    :param ses_col: String naming the column in combined_ERI_DB, and the label
                    in row, which has the session name
    :param task_cols: List of strings, each naming the column in combined_ERI_DB
                      (and label in row) for a given task run with ERI
    :return: pandas.Series where each cell is True iff row has all of the ERI 
             that it should have for that task-run column, or False otherwise
    """
    dummy_row_data = [row.get(sub_col), row.get(ses_col)
                      ] + [True for _ in range(2, len(row.index))]
    result = pd.Series(data=dummy_row_data, index=row.index)
    for task_run, status in row.loc[task_cols].dropna().items():
        if status[:4] == "bids":
            row_ERI = get_ERI_row_for_sub_ses(row, combined_ERI_DB,
                                              sub_col, ses_col)
            result.loc[task_run] = is_truthy(row_ERI.get(task_run))
    return result


def get_ERI_for_sub_ses(row_ERI_DB_1, db_ERI_2, sub_col, ses_col):
    """
    :return: pandas.Series which replaced as many NaN values as it could in
             row_ERI_DB_1 with values from the same column/row in db_ERI_2.
    """
    return row_ERI_DB_1.combine_first(get_ERI_row_for_sub_ses(
        row_ERI_DB_1, db_ERI_2, sub_col, ses_col
    ))


def get_ERI_row_for_sub_ses(orig_row, db_ERI, sub_col, ses_col):
    """
    :return: pandas.Series, the row of db_ERI with the same subject and
             session as orig_row
    """
    return db_ERI[(db_ERI[sub_col] == orig_row.get(sub_col)) &
                  (db_ERI[ses_col] == orig_row.get(ses_col))].iloc[0]


def is_truthy(a_value):
    result = bool(a_value)
    # if not a_value:
    #     result = False
    if result and isinstance(a_value, float):
        result = not np.isnan(a_value)
    return result  # False if not a_value else (not np.isnan(a_value))


def _cli(audit_names): 
    # audits = [audit.split("abcd-")[-1] for audit in audit_names]
    parser = argparse.ArgumentParser()
    msg_csv_files = ("One or more paths to existing readable .csv files which each "
                     "contain a list of subject-session pairs that ")
    msg_valid_path = " Must be a valid path to an existing local directory or s3 bucket."
    
    parser.add_argument(
        "-audit", "--audit-type", nargs="+",
        choices=audit_names, default=audit_names,
        help=("Choose which audit(s) to run, one or both of these: {}"
              .format(audit_names))
    )

    parser.add_argument(  # Only required if running DICOM DB Flow
        "-bids-dir", "--bids-dir", required=False,
        type=valid_readable_dir, # valid_local_or_s3_dirpath, # TODO Make valid_local_or_s3_dirpath function
        help=("Valid path to existing BIDS-formatted input data structure."
              + msg_valid_path)
    )

    parser.add_argument(
        "-counts", "--show-counts", action="store_true",
        help=("Include this flag to show counts of how many subjects pass/"
              "fail/etc. the audit(s) after running the audit(s).")
    )

    parser.add_argument(  # Only required if running ABCD-HCP Flow
        "-deriv", "--derivatives", "--derivatives-dir", 
        help=("Valid path to existing abcd-hcp-pipeline BIDS derivatives "
              "directory.")
    )

    parser.add_argument(
        "-E", "-ERI-DB", "--eri-paths-db", dest="ERI_DB",
        type=valid_readable_file, nargs="+",
        help=("Valid path(s) to existing .csv file(s) which include the paths "
              "(on tier1/MSI or tier2/s3) to EventRelatedInformation.txt "
              "files for each subject session. Include this argument to count "
              "how many subject sessions have ERI paths in the .csv file(s).")
    )

    parser.add_argument(  # Only required if running DICOM DB Flow
        "-ftqc", "--fasttrack-qc", dest="fasttrack_qc", action="store_true",
        help=("Include this flag to generate abcd-dicom2bids database")
    )

    parser.add_argument(  # Only required if running DICOM DB Flow
        "-make-db", "--make-bids-db-from-scratch", dest="make_db",
        action="store_true",
        help=("Include this flag to create a brand-new BIDS database from "
              "from scratch at the ")
    )

    parser.add_argument(
        "-out", "--output", "--output-dir", dest="output", 
        type=valid_output_dir, required=True,
        help=("Valid path to output directory to save output files (including "
              "subject lists and subject counts table .csv files) into.")
    )

    """
    parser.add_argument(  # Only required if running DICOM DB Flow
        "-src", "--sourcedata", type=valid_local_or_s3_dirpath,
        help=("Valid path to existing directory with event-related "
              "information files." + msg_valid_path)
    )
    """  # TODO

    parser.add_argument(
        "-uploaded", "--already-uploaded", dest="uploaded",
        nargs="+", type=valid_readable_file,
        help=msg_csv_files + "were already uploaded to the NDA."
    )

    parser.add_argument(
        "-pldb", "--pipeline-db", "--abcd-bids-db", "--abcd-bids-pipeline-db",
        dest="pipeline_db", nargs="+", type=valid_readable_file,
        default=PATH_ABCD_BIDS_DB,
        help=msg_csv_files + ("are BIDS-formatted and ready to be processed "
                              "through the ABCD BIDS Pipeline (with statuses "
                              "on processing for all the stages).")
    )

    parser.add_argument(
        "-save", "--save-lists", "--save-sub-ses-lists",
        action="store_true", dest="save_lists",
        help=("Include this flag to save out all subject-session lists as "
              ".csv files in the --output directory.")
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help=("Include this flag to print more info to stdout while running")
    )

    return vars(parser.parse_args())


def get_ERI_info(cli_args):
    saved_ERI_tier = dict()
    for tier in (1, 2):
        if os.path.exists(get_tier1_or_tier2_ERI_db_fname(cli_args["output"], tier)):
            saved_ERI_tier[tier] = pd.read_csv()


def get_ERI_col_name(eri_db_fpath):
    return "ERI from {}".format(os.path.basename(eri_db_fpath))
                        

def get_non_ID_col_names_from(a_db, sub_col, ses_col, *bids_columns):
    non_ID_col_names = a_db.columns.values.tolist()
    for name_of_ID_col in (sub_col, ses_col, *bids_columns):
        try: non_ID_col_names.remove(name_of_ID_col)
        except ValueError: pass
    return non_ID_col_names


def get_dicom_db(cli_args):
    # TODO Make this function without hardcoding
    return pd.read_csv(PATH_DICOM_DB)


def save_table_to_text_file(): pass
def save_table_to_csv_file(): pass
def map_paths_from_storage_to_NGDR(tier1_or_s3_in_dirpath, ngdr_out_dirpath, all_subject_sessions):
    # return hash table contents to put in a text file mapping (for each subject-session) s3 bucket paths to NGDR paths (one mapping per line, 2 space-separated paths)
    pass


if __name__ == "__main__":
    main()
