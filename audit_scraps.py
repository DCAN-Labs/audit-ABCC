import pandas as pd
import os
import glob
import numpy as np

DICOM2BIDS = "abcd-dicom2bids"
HCPIPELINE = "abcd-hcp-pipeline"
PATH_DICOM_DB = "/home/rando149/shared/code/internal/utilities/abcd-dicom2bids/src/audit/ABCD_BIDS_db.csv"
PATH_NGDR = "/spaces/ngdr/ref-data/abcd/nda-3165-2020-09/"

def main():
    
    subject_lists = dict()
    pd.set_option('display.max_columns', None)

    # DICOM DB FLOW
    dicom_db = get_dicom_db(PATH_DICOM_DB)

    # DICOM2BIDS FLOW
    subject_lists[DICOM2BIDS] = audit_abcd_dicom2bids(dicom_db)


def get_dicom_db():
    # TODO Make this function without hardcoding
    return pd.read_csv(PATH_DICOM_DB)



def audit_abcd_dicom2bids(dicom_db):
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

    
    # For each year, check/count every subject-session's BIDS conversion status
    counts = dict()
    ses_DBs = dict()

    sub_ses_df_BIDS_NGDR = get_sub_ses_df_from_tier1(PATH_NGDR, sub_col,
                                                     ses_col)

   
    for session in sessions:
        counts[session] = dict()
        ses_DBs[session] = dict()


        # Split dicoms_db by session and by whether BIDS conversion succeeded
        session_db = dicom_db_with_anat.loc[dicom_db_with_anat["session"] == session]
        ses_DBs[session] = {
            "Total": query_has_anat(session_db),
            "No-anat": dicom_db_no_anat[dicom_db_no_anat["session"] == session],

        }

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
   
def make_has_anat_DICOM_db(dicom_db_with_anat):
    dicom_db_with_anat.to_csv('has_anat_DICOMs.csv')
    
def make_no_anat_DICOM_db(dicom_db_no_anat):
    dicom_db_no_anat.to_csv('no_anat_DICOMs.csv')
    
def make_NGDR_list_db(all_sub_ses_NGDR):
    all_sub_ses_NGDR.to_csv('NGDR_subjects.csv')


def query_split_by_anat(df):
    """
    Filter dataframe for subjects that have at least one anatomical
    """
    filter_cond = df.filter(regex='T[1,2].*').isin([np.nan]).all(axis=1)
    return df[~filter_cond], df[filter_cond]

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

def query_has_anat(df):
    """
    Filter dataframe for subjects that have at least one anatomical
    :param df: pandas.DataFrame
    """
    return df[~df.filter(regex='T[1,2].*').isin([np.nan]).all(axis=1)]
