#! /usr/bin/env python3


import pandas as pd
import csv
import subprocess
import os
import glob
import sys
import argparse

prog_descrip='Create initial database to track ABCD BIDS data processing'

QC_CSV = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "spreadsheets",
    "ABCD_good_and_bad_series_table.csv")
BIDS_DB = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "spreadsheets",
    "ABCD_BIDS_database.csv")


def generate_parser(parser=None):
    if not parser:
        parser = argparse.ArgumentParser(
            description=prog_descrip
        )
    parser.add_argument(
        '-q',
        '--qc-csv',
        dest='qc_csv',
        default=QC_CSV,
        help='Path to the csv file containing aws paths and operator QC info'
    )
    parser.add_argument(
        '-db',
        '--database',
        dest='bids_db_file',
        default=BIDS_DB,
        help='Path to the current BIDS database in csv format '
    )
    return parser

def reformat_pGUID(pguid):
    uid_start = "INV"
    try:
        uid = pguid.split(uid_start, 1)[1]
    except:
        print('WARNING: {} is improperly formatted. Assuming last 8 chars is the uid'.format(pguid))
        uid = pguid[-8:]
    bids_pid = 'sub-NDARINV' + ''.join(uid)
    return bids_pid

def query_anat(sesh_df):
    ## Check if T1_NORM exists and download that instead of just T1
    #   If there is a T1_NORM in the df of good T1s then use it. Else just use good T1
    T1_df = sesh_df[sesh_df['image_description'] == 'ABCD-T1-NORM']
    if T1_df.empty:
        T1_df = sesh_df[sesh_df['image_description'] == 'ABCD-T1']
    num_T1 = T1_df.drop_duplicates(subset='image_file', keep='first').shape[0]

    T2_df = sesh_df[sesh_df['image_description'] == 'ABCD-T2-NORM']
    if T2_df.empty:
        T2_df = sesh_df[sesh_df['image_description'] == 'ABCD-T2']

    num_T2 = T2_df.drop_duplicates(subset='image_file', keep='first').shape[0]
    return (num_T1, num_T2)


def query_func(sesh_df):
    rest_df = sesh_df[sesh_df['image_description'] == 'ABCD-rsfMRI']
    num_rest = rest_df.drop_duplicates(subset='image_file', keep='first').shape[0]
    MID_df = sesh_df[sesh_df['image_description'] == 'ABCD-MID-fMRI']
    num_MID = MID_df.drop_duplicates(subset='image_file', keep='first').shape[0]
    SST_df = sesh_df[sesh_df['image_description'] == 'ABCD-SST-fMRI']
    num_SST = SST_df.drop_duplicates(subset='image_file', keep='first').shape[0]
    nback_df = sesh_df[sesh_df['image_description'] == 'ABCD-nBack-fMRI']
    num_nback = nback_df.drop_duplicates(subset='image_file', keep='first').shape[0]
    return (num_rest, num_MID, num_SST, num_nback)


def reformat_fastqc_spreadsheet(qc_csv):
    """
    Create abcd_fastqc01_reformatted.csv by reformatting the original fastqc01.txt spreadsheet.
    :param cli_args: argparse namespace containing all CLI arguments.
    :return: N/A
    """
    # Import QC data from .csv file
    with open(qc_csv) as qc_file:
        all_qc_data = pd.read_csv(
            qc_file, encoding="utf-8-sig", sep=",|\t", engine="python",
            index_col=False, header=0, skiprows=[1] # Skip row 2 (description)
        )

    # Remove quotes from values and convert int-strings to ints
    all_qc_data = all_qc_data.applymap(lambda x: x.strip('"')).apply(
        lambda x: x.apply(lambda y: int(y) if y.isnumeric() else y)
    )

    # Remove quotes from headers
    new_headers = []
    for header in all_qc_data.columns: # data.columns is your list of headers
        header = header.strip('"') # Remove the quotes off each header
        new_headers.append(header) # Save the new strings without the quotes
    all_qc_data.columns = new_headers # Replace the old headers with the new list
    print(all_qc_data.columns)

    qc_data = fix_split_col(all_qc_data.loc[all_qc_data['ftq_usable'] == 1])

    def get_img_desc(row):
        """
        :param row: pandas.Series with a column called "ftq_series_id"
        :return: String with the image_description of that row
        """
        return row.ftq_series_id.split("_")[2]

    # Add missing column by splitting data from other column
    image_desc_col = qc_data.apply(get_img_desc, axis=1)

    qc_data = qc_data.assign(**{'image_description': image_desc_col.values})

    # Change column names for good_bad_series_parser to use; then save to .csv
    qc_data = qc_data.rename({
        "ftq_usable": "QC", "subjectkey": "pGUID", "visit": "EventName",
        "abcd_compliant": "ABCD_Compliant", "interview_age": "SeriesTime",
        "comments_misc": "SeriesDescription", "file_source": "image_file"
    }, axis="columns")

    return qc_data 

def fix_split_col(qc_df):
    """
    Because qc_df's ftq_notes column contains values with commas, it is split
    into multiple columns on import. This function puts them back together.
    :param qc_df: pandas.DataFrame with all QC data
    :return: pandas.DataFrame which is qc_df, but with the last column(s) fixed
    """
    def trim_end_columns(row):
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

def main():
    parser = generate_parser()
    args = parser.parse_args()

    # Read fastqc spreadsheet into pandas database
    #qc_df = pd.read_csv(args.qc_csv)
    qc_df = reformat_fastqc_spreadsheet(args.qc_csv)

    # Create empty database or read database from file if it exists
    if os.path.exists(args.bids_db_file):
        bids_db = pd.read_csv(args.bids_db_file)
    else:
        bids_db = pd.DataFrame(columns=['subject',
                                        'session',
                                        'T1_run-01',
                                        'T2_run-01',
                                        'task-rest_run-01',
                                        'task-rest_run-02',
                                        'task-rest_run-03',
                                        'task-rest_run-04',
                                        'task-MID_run-01',
                                        'task-MID_run-02',
                                        'task-SST_run-01',
                                        'task-SST_run-02',
                                        'task-nback_run-01',
                                        'task-nback_run-02'])

    possible_status = ['nan', 'no bids']

    session_dict = {'baseline_year_1_arm_1': 'ses-baselineYear1Arm1',
                    '2_year_follow_up_y_arm_1': 'ses-2YearFollowUpYArm1',
                    '4_year_follow_up_y_arm_1': 'ses-4YearFollowUpYArm1'}

    # get list of all unique (subject, session)
    subject_arr = qc_df.pGUID.unique()
    session_arr = qc_df.EventName.unique()

    for pid in subject_arr:
        for sesh in session_arr:
            print('Checking {} {}'.format(pid, sesh))
            pid_df = qc_df[qc_df.pGUID == pid]
            sesh_df = pid_df[pid_df.EventName == sesh]
            if sesh_df.empty:
                break
            num_T1, num_T2 = query_anat(sesh_df[sesh_df['QC'] == 1.0])
            num_rest, num_MID, num_SST, num_nback = query_func(sesh_df[sesh_df['QC'] == 1.0])

            participant_id = reformat_pGUID(pid)
            session_id = session_dict[sesh]

            subject_dict = {}
            subject_dict['subject'] = participant_id
            subject_dict['session'] = session_id

            for i in range(0, num_T1):
                header = 'T1_run-0{}'.format(str(i+1))
                subject_dict[header] = 'no bids'
            for i in range(0, num_T2):
                header = 'T2_run-0{}'.format(str(i+1))
                subject_dict[header] = 'no bids'
            for i in range(0, num_rest):
                header = 'task-rest_run-0{}'.format(str(i+1))
                subject_dict[header] = 'no bids'
            for i in range(0, num_MID):
                header = 'task-MID_run-0{}'.format(str(i+1))
                subject_dict[header] = 'no bids'
            for i in range(0, num_SST):
                header = 'task-SST_run-0{}'.format(str(i+1))
                subject_dict[header] = 'no bids'
            for i in range(0, num_nback):
                header = 'task-nback_run-0{}'.format(str(i+1))
                subject_dict[header] = 'no bids'

            bids_db = bids_db.append(subject_dict, ignore_index=True)
    bids_db.to_csv(args.bids_db_file, index=False)





if __name__ == '__main__':
    main()
