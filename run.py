#!/bin/usr/env python3

"""
ABCC Audit Main Script 
Greg Conan: gconan@umn.edu
Created 2022-07-06
Updated 2022-08-10
"""
# Standard imports
import argparse
from email import header
import os
import pandas as pd
import sys
# import boto3  # TODO Use this for interfacing with s3

# Nonstandard imports
#import fastqc_audit as fastqc  # TODO import from https://github.com/DCAN-Labs/abcd-dicom2bids/blob/audit-dev/src/audit/fastqc_audit.py
#import s3_audit as s3  # TODO import from https://github.com/DCAN-Labs/abcd-dicom2bids/blob/audit-dev/src/audit/s3_audit.py
#import tier1_audit as tier1   # TODO import from https://github.com/DCAN-Labs/abcd-dicom2bids/blob/audit-dev/src/audit/tier1_audit.py
from src.utilities import (
    DICOM2BIDS, dict_has, get_sub_ses_df_from_tier1, 
    get_tier1_or_tier2_ERI_db_fname, HCPIPELINE, PATH_DICOM_DB,
    PATH_NGDR, query_split_by_anat, query_processed_subjects,
    query_has_anat, query_unprocessed_subjects, save_to_hash_map_table,
    valid_output_dir, valid_readable_dir, valid_readable_file,
)


# Functions


def main():
    cli_args = _cli([DICOM2BIDS, HCPIPELINE])
    subject_lists = dict()
    pd.set_option('display.max_columns', None)

    # DICOM DB FLOW
    dicom_db = get_dicom_db(cli_args)

    # DICOM2BIDS FLOW
    subject_lists[DICOM2BIDS] = audit_abcd_dicom2bids(dicom_db, cli_args)

    """ 
    # ABCD-HCP FLOW
    abcd_hcp_db = get_abcd_hcp_db(cli_args)
    subject_lists[HCPIPELINE] = audit_abcd_hcp_pipeline(abcd_hcp_db, cli_args)

    # AUDIT COUNT TABLE FLOW
    if cli_args["show_counts"]:
        assert len(cli_args["audit_type"]) > 1
    """ # TODO


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


def update_w_sub_ses_ERI_counts(eri_db_fpath, pre_eri_db, non_ID_col_names):
    db_ERI = pd.read_csv(eri_db_fpath)
    pre_eri_db[get_ERI_col_name(eri_db_fpath)
               ] = db_ERI[non_ID_col_names].apply(lambda row: row.count(), axis=1)
    return pre_eri_db


def audit_abcd_dicom2bids(dicom_db, cli_args):
    # DICOM2BIDS FLOW
    bids_columns = ["bids_subject_id", "bids_session_id"]
    out_paths = dict()

    sub_col = dicom_db.columns.values[0]  # "subject"
    ses_col = dicom_db.columns.values[1]  # "session"
    sessions=list(set(dicom_db[ses_col]))  # TODO Ensure that this gets the right column name (right now it's just "session")

    # Trim down database to only include subjects that have a T1 or T2  # TODO make this comment accurate
    # Input: BIDS DB for that year
    # Output: List of excluded subjects for that year, and BIDS DB with them excluded
    dicom_db_with_anat, dicom_db_no_anat = query_split_by_anat(dicom_db)

    # Get names of every column without identifying information or ERI counts
    non_ID_col_names = get_non_ID_col_names_from(
        dicom_db_with_anat, sub_col, ses_col, *bids_columns,
        *[get_ERI_col_name(fpath) for fpath in cli_args["ERI_DB"]]
    )

    # Get all subject sessions that user already uploaded to the NDA
    # TODO Reorganize so that we count the NGDR statuses before checking uploads
    already_uploaded = list()
    if dict_has(cli_args, "uploaded"):
        for each_csv_path in cli_args["uploaded"]:  
            already_uploaded.append(pd.read_csv(each_csv_path))
            assert already_uploaded[-1].columns.values.tolist() == bids_columns
        already_uploaded = pd.concat(already_uploaded)  # TODO Maybe we should make this a subset of the dicom_db
        uploaded_with_anat = pd.merge(dicom_db_with_anat, already_uploaded,
                                    how="inner", left_on=[sub_col, ses_col],
                                    right_on=bids_columns, indicator=True)
        # uploaded_with_anat = uploaded_with_anat[uploaded_with_anat["_merge"] == "right_only"]
        # non_ID_col_names = get_non_ID_col_names_from(uploaded_with_anat, sub_col, ses_col, *bids_columns)
        uploaded_with_anat.columns = [sub_col, ses_col, *bids_columns, *non_ID_col_names, "_merge"]
        # non_ID_col_names.remove("_merge")

    # For each year, check/count every subject-session's BIDS conversion status
    counts = dict()
    ses_DBs = dict()
    sub_ses_df_BIDS_NGDR = get_sub_ses_df_from_tier1(PATH_NGDR, sub_col,
                                                    ses_col)

    # Count ERI files per subject session (TODO FIX & TEST)
    # non_ID_cols = [col for col in dicom_db_with_anat.columns.values.tolist() if "task-"]
    for eachfpath in cli_args["ERI_DB"]:
        dicom_db_with_anat = update_w_sub_ses_ERI_counts(  # [get_ERI_col_name(eachfpath)]
                                eachfpath, dicom_db_with_anat, non_ID_col_names
                            )

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

        for eachfpath in cli_args["ERI_DB"]:
            col_ERI = get_ERI_col_name(eachfpath)
            ses_DBs[session][col_ERI] = session_db[session_db[col_ERI] > 0]
            # print(ses_DBs[session][col_ERI])

        # TODO If user provided list of previously-uploaded subject-sessions, 
        #      remove each one from this uploads_subject_list
        # Build paths to .csv files to write subject/session lists into
        out_paths["uploads_subject_list_" + session] = os.path.join(
            cli_args["output"],
            "{}" + "_{}_{}_subject-list.csv".format(DICOM2BIDS, session)
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
        print()
        for convert_status in ses_DBs[session].keys():
            counts[session][convert_status] = len(ses_DBs[session][convert_status])
            print("{} subjects in {}{}: {}"  # TODO Move the "print" functionality to audit count table flow section
                  .format(convert_status, session,
                          "" if convert_status[:2].lower() == "no" else " with anat",
                          counts[session][convert_status]))

            # Save lists of subject-sessions (1 per conversion status)
            ses_DBs[session][convert_status].to_csv(
                out_paths["uploads_subject_list_" + session
                          ].format(convert_status.replace(" ", "_")),
                columns=[sub_col, ses_col], index=False, header=bids_columns
            )

        for eachfpath in cli_args["ERI_DB"]:
            col_ERI = get_ERI_col_name(eachfpath)
            print("Total number of ERI files for {} from {}: {}"
                  .format(session, col_ERI, session_db[col_ERI].sum()))

        # For each subject-session,
        
            # If subject-session completely succeeded abcd-dicom2bids conversion,

                # Get whether the subject-session has already been uploaded from the NDA upload files
                # Input: Path to NDA upload submission working directory, list of subject-sessions, path(s) to .csv file(s) with subject-sessions already uploaded to the NDA so those can be excluded from the output list: cli_args["uploaded"]
                # Output: List of subject-sessions to upload to the NDA, which needs to be saved out as a .csv with subject and session column

                # Verify that the subject-session exists on the NGDR space
                # Input: NGDR space path(s) where subject-sessions should exist, list of subject-sessions
                # Output: List of subjects to move to the NGDR space, which needs to be a text file mapping (for each subject-session) s3 bucket paths to NGDR paths (one mapping per line, 2 space-separated paths)
        
    need_to_copy_to_NGDR = pd.concat([ses_DBs[session]["Not yet NGDR"] for session in sessions])
    outfpath = os.path.join(cli_args["output"], "bids2ngdr_paths_hash_mapping.txt")
    save_to_hash_map_table(need_to_copy_to_NGDR, sub_col, ses_col,
                           cli_args["bids_dir"], PATH_NGDR, outfpath)

            # Else (if subject that at least partly failed abcd-dicom2bids conversion),

                # If subject failed due to timeout,  # NOTE This block will not be worth the effort if it requires reading log files
                    # Add that subject to list of subjects to rerun with more time
                # Else

                    # Add subject to a list of failed subject-sessions to save out as a text file (1 subject-session per line)


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


    # AUDIT COUNT TABLE FLOW (for Damien)

    # Finally, create a table with subject counts
    # For each session (AKA year),
        # For pipeline in (abcd-dicom2bids, ABCD-HCP), add a row showing how many subjects in session
            # - failed thru pipeline,
            # - succeeded thru pipeline,
            # - succeeded thru pipeline AND are pulled to NGDR,
            # - succeeded thru pipeline AND are pulled to NGDR, AND are uploaded to the NDA
            # - need to be triaged

            # If pipeline == ABCD-HCP, then also add a row showing how many subjects in session
                # - failed to file-map


# More Utility Functions
    

def _cli(audit_names): 
    parser = argparse.ArgumentParser()
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
        help=("One or more paths to existing readable .csv files which each "
              "contain a list of subject-session pairs that were already "
              "uploaded to the NDA.")
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
