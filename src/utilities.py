#!/usr/bin/env python3

"""
Utilities for ABCC auditing workflow
Originally written by Anders Perrone
Updated by Greg Conan on 2024-01-18
"""
import argparse
from datetime import datetime
import pandas as pd
from glob import glob
import nibabel as nib
import numpy as np
import os

# Constants: Pipeline names, database path, & temporarily hardcoded dirpath
BIDS_COLUMNS = ("bids_subject_id", "bids_session_id")
BIDSPIPELINE = "abcd-bids-pipeline"
DICOM2BIDS = "abcd-dicom2bids"
PATH_ABCD_BIDS_DB = "/home/rando149/shared/projects/ABCC_year2_processing/s3_status_report.csv"
PATH_DICOM_DB = "/home/rando149/shared/code/internal/utilities/abcd-dicom2bids/src/audit/ABCD_BIDS_db.csv"
PATH_NGDR = "/spaces/ngdr/ref-data/abcd/nda-3165-2020-09/"


class BareObject:
    """
    Any function which initalizes this empty object to use as a namespace will
    be able to assign it arbitrary attributes as properties. Used inside other
    class definitions when it is more intuitive than defining a dict.
    """
    pass


def main():
    pd.set_option('display.max_columns', None)


def get_and_print_time_if(will_print, event_time, event_name):
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
        print("\nTime elapsed {}: {}"
              .format(event_name, timestamp - event_time))
    return timestamp 


def get_tier1_or_tier2_ERI_db_fname(parent_dir, tier):
    return os.path.join(parent_dir, "ERI_tier{}_paths_bids_db.csv".format(tier))


def get_variance_of_each_row_of(dtseries_path):
    """
    :param dtseries_path: String, valid path to existing dtseries.nii file
    :return: np.ndarray of floats; each is the variance of a row in the dtseries file
    """
    return load_matrix_from(dtseries_path).var(axis=1)


def is_truthy(a_value):
    result = bool(a_value)
    # if not a_value:
    #     result = False
    if result and isinstance(a_value, float):
        result = not np.isnan(a_value)
    return result  # False if not a_value else (not np.isnan(a_value))


def load_matrix_from(matrix_path):
    """
    :param matrix_path: String, the absolute path to a .gii or .nii matrix file
    :return: numpy.ndarray matrix of data from the matrix file
    """
    return {
        ".gii": lambda x: x.agg_data(),
        ".nii": lambda x: np.array(x.get_fdata().tolist()),
    }[os.path.splitext(matrix_path)[-1]](nib.load(matrix_path))


def make_ERI_filepath(parent, subj_ID, session, task, run):
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


def save_to_hash_map_table(sub_ses_df, subj_col, ses_col, in_dirpath,
                           out_dirpath, outfile_path, subdirnames=list()):
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
    print("Saved hash mapping table of paths to {}".format(outfile_path))


def valid_output_dir(path):
    """
    Try to make a folder for new files at path; throw exception if that fails
    :param path: String which is a valid (not necessarily real) folder path
    :return: String which is a validated absolute path to real writeable folder
    """
    return validate(path, lambda x: os.access(x, os.W_OK),
                    valid_readable_dir, 'Cannot create directory at {}', 
                    lambda y: os.makedirs(y, exist_ok=True))


def valid_readable_dir(path):
    """
    :param path: Parameter to check if it represents a valid directory path
    :return: String representing a valid directory path
    """
    return validate(path, os.path.isdir, valid_readable_file,
                    "Cannot read directory at '{}'")


def valid_readable_file(path):
    """
    Throw exception unless parameter is a valid readable filepath string. Use
    this, not argparse.FileType('r') which leaves an open file handle.
    :param path: Parameter to check if it represents a valid filepath
    :return: String representing a valid filepath
    """
    return validate(path, lambda x: os.access(x, os.R_OK),
                    os.path.abspath, 'Cannot read file at {}')


def validate(to_validate, is_real, make_valid, err_msg, prepare=None):
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


def validate_dtseries(dtseries_path): 
    variances = get_variance_of_each_row_of(dtseries_path)
    # invalid_rows = variances.
    """
    invalid_rows = 0
    for eachvar in variances:
        if not eachvar:
            invalid_rows += 1
    if invalid_rows > THRESHOLD: pass  
    """ # TODO


if __name__ == "__main__":
    main()
