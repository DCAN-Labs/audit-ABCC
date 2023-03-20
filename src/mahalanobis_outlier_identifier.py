#!/usr/bin/env python3
# coding: utf-8

"""
Anders Perrone: perr0372@umn.edu
Created: 2021-04-06
Updated: 2022-03-17
"""

# Import standard libraries
import argparse
from glob import glob
import h5py
import nibabel
import numpy as np
import os
import pandas as pd
# import scipy.io
# from scipy.spatial import distance
from sklearn.covariance import EmpiricalCovariance, MinCovDet

VAR_MX = 'variance_matrix'
MAT_VAR = 'matlab_var_name'

#TODO: nan checker - dtseries files
#       - calculate variance of timeseries and check for nan/zero
#       - Save out variance to cifti for visualization
#       - Save report on number of nan/zero and number of subjects

def main():
    cli_args = _cli()
    all_paths = glob(cli_args['data']) # os.listdir(cli_args['parent']) 
    mx_ext = os.path.splitext(all_paths[0])[-1]
    if not dict_has(cli_args, VAR_MX):
        cli_args[VAR_MX] = os.path.join(cli_args['output'], VAR_MX + mx_ext)

    num_subjects = len(all_paths)
    first_mx = load_flat_data_from(all_paths[0], cli_args)
    all_data = np.ndarray(shape=(num_subjects, len(first_mx)))
    all_data[0] = first_mx
    for subj_ix in range(1, num_subjects):
        all_data[subj_ix] = load_flat_data_from(all_paths[subj_ix], cli_args)
        
    print('All data:\n{}'.format(all_data))
    print('all_data.shape=={}'.format(all_data.shape))

    dist = mahalanobis_distance(np.transpose(all_data))
    print('Mahalanobis distances: {}'.format(dist))

    pd.DataFrame(dist).to_csv(os.path.join(cli_args['output'], 'mahalanobis_distances.csv'))


    # TODO Build sheet of subject matrix data by adding each flattened
    # subject's matrix as one column and each row as one cell in the matrix,
    # then feed that sheet into MinCovDet to get the covariance matrix, and
    # use the .mahalanobis function after feeding the sheet in;
    # 
    # Also, once one subject's matrix is loaded in, you can infer the shape
    # of the sheet b/c its X-length is the number of subjects and its
    # Y-length is the total number of cells per matrix
    # 
    # The .mahalanobis function gives one distance value per column (/subject)
    # 
    # For the sheet 2D numpy array, assign the index as each column (/subject) 
        


def _cli():
    """
    :return: Dictionary with all validated command-line arguments from the user
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-mat-var', as_cli_arg(MAT_VAR), default='R', # '--matlab-variable-name', dest=MAT_VAR,
        help=('Name of the variable mapped to the connectivity matrix in each '
              'matrix\'s .mat file. This argument is only needed if you are '
              'getting the Mahalanobis distances of matrices in .mat files.')
    )
    parser.add_argument(
        '-out', '--output', metavar='OUTPUT_DIRECTORY', type=valid_output_dir,
        required=True, help='Directory path to save outputs into.'
    )
    parser.add_argument(
        '-infiles', '--input-data-files', dest='data',
        help=('Valid path to all data files to calculate the Mahalanobis '
              'distances of, using wildcards such as * to represent multiple '
              'files in valid bash/regex/glob format.')
    )
    parser.add_argument(
        '-variance', as_cli_arg(VAR_MX),
        help=('Valid path to variance matrix file. If no file exists at this '
              'path, then the script will create one and save it to this path.')
    )
    return vars(parser.parse_args())


def mahalanobis_distance(all_data):
    """
    :param all_data: np.ndarray, a 2D matrix of numbers
    """
    # fit a Minimum Covariance Determinant (MCD) robust estimator to data
    robust_cov = MinCovDet().fit(all_data) # Get the Mahalanobis distance
    return robust_cov.mahalanobis(all_data) # np.sqrt(robust_cov.mahalanobis(all_data))


def is_readable(filepath):
    """
    :param filepath: String, a path to a file
    :return: True if there is a readable file at filepath; otherwise False
    """
    return os.path.exists(filepath) and os.access(filepath, os.R_OK)


def as_cli_arg(arg_str):
    """
    :param arg_str: String naming a stored argument taken from the command line
    :return: String which is the command-line argument form of arg_str
    """
    return "--" + arg_str.replace("_", "-")


def load_flat_data_from(matrix_path, cli_args):
    """
    :param matrix_path: String which is the absolute path to a matrix file
    :return: numpy.array, 1D list of flattened data from the matrix file
    """
    mx_ext = os.path.splitext(matrix_path)[-1]
    if mx_ext == '.mat':
        with h5py.File(matrix_path, 'r') as infile:
            matrix = dict(infile)
            matrix = matrix[cli_args[MAT_VAR]][:]
    else:
        matrix = nibabel.cifti2.load(matrix_path).get_data().tolist()
    return np.array(matrix).flatten()


def save_to_cifti2(matrix_data, example_file, outfile):
    """
    Save a numpy array into a cifti2 .nii file by importing an arbitrary cifti2
    matrix and saving a copy of it with its data replaced by the data of the
    the new matrix
    :param matrix_data: numpy.ndarray with data to save into cifti2 file
    :param example_file: String, the path to a .nii file with the right format
    :param outfile: String, the path to the output .nii file to save
    :return: N/A
    """
    nii_matrix = nibabel.cifti2.load(example_file)
    nibabel.cifti2.save(nibabel.cifti2.cifti2.Cifti2Image(
        dataobj = matrix_data,
        header = nii_matrix.header,
        nifti_header = nii_matrix.nifti_header,
        file_map = nii_matrix.file_map
    ), outfile)


def dict_has(a_dict, a_key):
    """
    :param a_dict: Dictionary (any)
    :param a_key: Object (any)
    :return: True if and only if a_key is mapped to something truthy in a_dict
    """
    return a_key in a_dict and a_dict[a_key]


def valid_output_dir(path):
    """
    Try to make a folder for new files at path; throw exception if that fails
    :param path: String which is a valid (not necessarily real) folder path
    :return: String which is a validated absolute path to real writeable folder
    """
    return validate(path, lambda x: os.access(x, os.W_OK),
                    valid_readable_dir, "Cannot create directory at {}", 
                    lambda y: os.makedirs(y, exist_ok=True))


def valid_readable_dir(path):
    """
    :param path: Parameter to check if it represents a valid directory path
    :return: String representing a valid directory path
    """
    return validate(path, os.path.isdir, os.path.abspath,
                    "{} is not a valid directory path")


def valid_whole_number(to_validate):
    """
    Throw argparse exception unless to_validate is an integer greater than 0
    :param to_validate: Object to test whether it is an integer greater than 0
    :return: to_validate if it is an integer greater than 0
    """
    return validate(to_validate, lambda x: int(to_validate) > 0, int, 
                    "{} is not a positive integer")


def validate(path, is_real, make_valid, err_msg, prepare=None):
    """
    Parent/base function used by different type validation functions. Raises an
    argparse.ArgumentTypeError if the input path is somehow invalid.
    :param path: String to check if it represents a valid path 
    :param is_real: Function which returns true if and only if path is real
    :param make_valid: Function which returns a fully validated path
    :param err_msg: String to show to user to tell them what is invalid
    :param prepare: Function to create something at path before validation
    :return: path, but fully validated as pointing to the right file or dir
    """
    try:
        if prepare:
            prepare(path)
        assert is_real(path)
        return make_valid(path)
    except (OSError, TypeError, AssertionError, ValueError, 
            argparse.ArgumentTypeError):
        raise argparse.ArgumentTypeError(err_msg.format(path))


if __name__ == '__main__':
    main()
