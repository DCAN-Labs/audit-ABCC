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


def get_dicom_db(file_path):
    # TODO Make this function without hardcoding
    return pd.read_csv(file_path)



def audit_abcd_dicom2bids(dicom_db):
    # DICOM2BIDS FLOW
    bids_columns = ["subject", "session"]
    sub_col = dicom_db.columns.values[0]  # "subject"
    ses_col = dicom_db.columns.values[1]  # "session"
    sessions=list(set(dicom_db[ses_col]))  # TODO Ensure that this gets the right column name (right now it's just "session")

    # Trim down database to only include subjects that have a T1 or T2  # TODO make this comment accurate
    # Input: BIDS DB for that year
    # Output: List of excluded subjects for that year, and BIDS DB with them excluded
    dicom_db_with_anat, dicom_db_no_anat = query_split_by_anat(dicom_db)

   
    for session in sessions:
        # Split dicoms_db by session and by whether BIDS conversion succeeded
        session_db = dicom_db_with_anat.loc[dicom_db_with_anat[ses_col] == session]
        has_anat = query_has_anat(session_db)
        no_anat = dicom_db_no_anat[dicom_db_no_anat[ses_col] == session]

        # Save the session specific data to CSV files
        file_path_has_anat = f'/home/rando149/shared/projects/begim_test/audit_year2_bids_count/{session}_has_anat.csv'
        file_path_no_anat = f'/home/rando149/shared/projects/begim_test/audit_year2_bids_count/{session}_no_anat.csv'

        has_anat.to_csv(file_path_has_anat, index=False)
        no_anat.to_csv(file_path_no_anat, index=False)

    # Saving additional data if required
    all_sub_ses_NGDR = get_sub_ses_df_from_tier1(PATH_NGDR)
    all_sub_ses_NGDR.to_csv('/home/rando149/shared/projects/begim_test/audit_year2_bids_count/NGDR_subjects.csv')

    return {"has_anat": dicom_db_with_anat, "no_anat": dicom_db_no_anat}
   

def query_split_by_anat(df):
    """
    Filter dataframe for subjects that have at least one anatomical
    """
    filter_cond = df.filter(regex='T[1,2].*').isin([np.nan]).all(axis=1)
    return df[~filter_cond], df[filter_cond]

def get_sub_ses_df_from_tier1(tier1_dirpath):
    path_col = "tier1_dirpath"

    # Get DF of all subjects and their sessions in the NGDR space
    all_sub_ses_NGDR_paths = glob.glob(os.path.join(tier1_dirpath, "sub-*", "ses-*"))
    # NOTE Later we can verify specific files existing instead of just the
    #      session directories existing
    all_sub_ses_NGDR = pd.DataFrame({path_col: [path for path in all_sub_ses_NGDR_paths]})
    all_sub_ses_NGDR['subject'] = all_sub_ses_NGDR[path_col].apply(
        lambda path: os.path.basename(os.path.dirname(path))
    )
    all_sub_ses_NGDR['session'] = all_sub_ses_NGDR[path_col].apply(
        lambda path: os.path.basename(path)
    )
    return all_sub_ses_NGDR

def query_has_anat(df):
    """
    Filter dataframe for subjects that have at least one anatomical
    :param df: pandas.DataFrame
    """
    return df[~df.filter(regex='T[1,2].*').isin([np.nan]).all(axis=1)]

if __name__ == "__main__":
    main()
