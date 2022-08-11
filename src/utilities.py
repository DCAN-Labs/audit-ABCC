#!/usr/bin/env python3

"""
Utilities for ABCC auditing workflow
Originally written by Anders Perrone
Updated by Greg Conan on 2022-08-10
"""
import argparse
import pandas as pd
from glob import glob
import nibabel as nib
import numpy as np
import os


def main():
    pd.set_option('display.max_columns', None)


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


def load_matrix_from(matrix_path):
    """
    :param matrix_path: String, the absolute path to a .gii or .nii matrix file
    :return: numpy.ndarray matrix of data from the matrix file
    """
    return {
        ".gii": lambda x: x.agg_data(),
        ".nii": lambda x: np.array(x.get_fdata().tolist()),
    }[os.path.splitext(matrix_path)[-1]](nib.load(matrix_path))


def get_variance_of_each_row_of(dtseries_path):
    """
    :param dtseries_path: String, valid path to existing dtseries.nii file
    :return: np.ndarray of floats; each is the variance of a row in the dtseries file
    """
    return load_matrix_from(dtseries_path).var(axis=1)


def query_has_anat(df):
    """
    Filter dataframe for subjects that have at least one anatomical
    :param df: pandas.DataFrame
    """
    return df[~df.filter(regex='T[1,2].*').isin([np.nan]).all(axis=1)]


def query_processed_subjects(df):
    """
    Filter dataframe to get dataframe of subjects that do not have any unprocessed images
    """
    processed_df = df[~df[df.columns[2:]].isin(['no bids']).any(axis=1)]
    # Filter again to remove subjects that have need data deleted
    fully_processed_df = processed_df[~processed_df.isin(['delete (tier1)', 'delete (s3)']).any(axis=1)]
    # Filter again to remove subjects that do not have a T1
    fully_processed_df = query_has_anat(fully_processed_df)
    return fully_processed_df


def query_split_by_anat(df):
    """
    Filter dataframe for subjects that have at least one anatomical
    """
    filter_cond = df.filter(regex='T[1,2].*').isin([np.nan]).all(axis=1)
    return df[~filter_cond], df[filter_cond]
    

def query_unprocessed_subjects(df):
    """
    Check for fully unprocessed subjects
    """
    # Filter dataframe to get dataframe of subjects that are missing one or more modalities
    missing_data_df = df[df[df.columns[2:]].isin(['no bids']).any(axis=1)]
    # Filter again to remove subjects that have BIDS data somewhere
    fully_unprocessed_df = missing_data_df[~missing_data_df.isin(['bids (tier1)', 'delete (tier1)', 'bids (s3)', 'delete (s3)']).any(axis=1)]
    
    return fully_unprocessed_df


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
