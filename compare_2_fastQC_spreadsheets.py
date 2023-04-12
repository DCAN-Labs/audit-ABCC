#!/usr/bin/env python3
# coding: utf-8

"""
Compare 2 fastqc01 spreadsheets
Greg Conan: gconan@umn.edu
Created 2022-08-05
Updated 2022-10-18
"""
import argparse
from datetime import datetime
import os
import pandas as pd
import sys

# Constants: Column names in dataframes
COL_SUB = "subjectkey"  # Subject ID column name
COL_SES = "visit"       # Session ID column name


def main():
    scriptname = os.path.basename(sys.argv[0])
    start = get_and_print_timestamp_when(scriptname, "started running")
    cli_args = _cli()
    fastQC01 = {which: pd.read_csv(cli_args[which], sep="\t").drop(index=[0])
                for which in ("old", "new")}
    get_and_print_timestamp_when(scriptname, "imported fastQC01 spreadsheets")

    # Reformat subject ID and session name
    for which in ("old", "new"):
        fastQC01[which][COL_SUB] = fastQC01[which][COL_SUB].apply(reformat_subject_ID)
        fastQC01[which][COL_SES] = fastQC01[which][COL_SES].apply(reformat_ses)

    # Get a set including every specific session name exactly once
    sessions = set(fastQC01["old"][COL_SES])
    sessions.update(set(fastQC01["new"][COL_SES]))

    subjects = dict()
    for ses in sessions:
        subjects[ses] = dict()

        this_ses_dfs = get_old_and_new_dfs_for_session(ses, fastQC01)
        subjects = get_subjects_for_1_ses(ses, this_ses_dfs, subjects)
        save_output_files_for_ses(ses, subjects, cli_args["output_dir"])

    finish = get_and_print_timestamp_when(scriptname, "finished")
    print("Time elapsed since {} started (HH:MM:SS.ssssss): {}"
          .format(scriptname, finish - start))


def reformat_subj_ses(in_arg, prefix): #, *keywords):
    """
    :param in_arg: String to make into a valid subject ID or session name
    :param prefix: String, 'sub-' or 'ses-'
    :return: String, valid subject ID (or session name)
    """
    return in_arg if in_arg[:len(prefix)] == prefix else prefix + in_arg


def reformat_subject_ID(old_subj_ID):
    new_subj_ID = old_subj_ID
    if "_" in old_subj_ID:
        new_subj_ID = "".join(old_subj_ID.split("_"))
    return reformat_subj_ses(new_subj_ID, "sub-")


def reformat_ses(old_ses):  # TODO This function is probably redundant/unneeded
    return reformat_subj_ses(old_ses, "ses-")


def get_and_print_timestamp_when(name, did_what):
    """
    :param name: String naming the function/script which did_what at timestamp
    :param did_what: String describing what the fn/script did at timestamp
    :return: datetime.datetime object representing when the fn/script did_what
    """
    timestamp = datetime.now()
    print("{} {} at {}".format(
        name, did_what, timestamp.strftime("%H:%M:%S on %b %d, %Y")
    ))
    return timestamp

        
def _cli():
    """
    :return: Dictionary containing all command-line input arguments
    """
    msg_pre = "Valid path to readable {} fastqc01 spreadsheet text file."
    parser = argparse.ArgumentParser(
        "Compare 2 ABCD FastQC01 spreadsheets, and for each session, save out "
        "a list of subjects added to the new spreadsheet and a list of "
        "subjects removed from the new spreadsheet."
    )
    parser.add_argument(
        "old", type=valid_readable_file, metavar="OLD_FASTQC01_SPREADSHEET",
        help=msg_pre.format("old")
        
    )
    parser.add_argument(
        "new", type=valid_readable_file, metavar="NEW_FASTQC01_SPREADSHEET",
        help=msg_pre.format("new")
    )
    parser.add_argument(
        "output_dir", type=valid_output_dir, metavar="OUTPUT_DIRECTORY_PATH",
        help="Valid path to directory to save output subject list files into."
    )
    # TODO Add option to also save out a .csv file with the total / all subjects (per session?) 
    return vars(parser.parse_args())


def get_old_and_new_dfs_for_session(ses_name, full_df):
    """
    Filter old and new dataframes to only include subjects with ses_name data
    :param full_df: pandas.DataFrame
    :param ses_name: String naming the session
    :return: Dictionary mapping "old" and "new" to pd.DataFrame of its subjects
    """
    return {which: full_df[which][full_df[which][COL_SES] == ses_name]
            for which in ("old", "new")}


def get_subjects_for_1_ses(ses, this_ses_dfs, subjects):
    """
    :param ses: String naming the session
    :param this_ses_dfs: Dictionary mapping "old" and "new" each to a
                         pd.DataFrame with the old/new subjects of this session
    :param subjects: Dictionary containing all subject/session dataframes
    :return: subjects, but now updated with subjects for this ses
    """ 
    # For old and new dataframes, get set of all subjects in this session
    for which in ("old", "new"):
        subjects[ses][which] = set(this_ses_dfs[which][COL_SUB])

    # Split subjects by whether they were added, kept, or removed
    subjects[ses]["kept"] = (     # Subjects in both dfs
        subjects[ses]["old"].intersection(subjects[ses]["new"])
    )
    subjects[ses]["removed"] = (  # Subjects in old df, but not new df
        subjects[ses]["old"].difference(subjects[ses]["kept"])
    )
    subjects[ses]["added"] = (    # Subjects in new df, but not old df
        subjects[ses]["new"].difference(subjects[ses]["kept"])
    )
    return subjects


def save_output_files_for_ses(ses, subjects, output_dir):
    """
    :param ses: String naming the session
    :param subjects: Dictionary containing all subject/session dataframes
    :param output_dir: String, valid path to existing dir to save outputs into
    """
    saved_msg = "Saved list of {} subjects {} to {}"

    # Tell user how many subjects were removed and were added
    print("Session {}: Removed {} and Added {}".format(
        ses, len(subjects[ses]["removed"]), len(subjects[ses]["added"])
    ))

    # Save list of all subjects in most recent fast-track .csv for this session
    outfpath = os.path.join(output_dir, ses + "_{}_subjects.csv")
    pd.DataFrame(index=list(subjects[ses]["new"]), columns=list()
                 ).to_csv(outfpath.format("new"), index=True, header=False)
    print("\n" + saved_msg.format("all", "in " + ses, outfpath.format("new")))

    # Save removal and addition subject lists into .CSV files in output dir
    for which in ("added", "removed"):

        # Alphabetize subjects and save them into output file
        subjs_to_save = list(subjects[ses][which])
        subjs_to_save.sort()
        to_save = pd.DataFrame(index=subjs_to_save, columns=[COL_SUB])
        to_save[COL_SES] = ses
        to_save.to_csv(outfpath.format(which), index=True, header=False)
        print(saved_msg.format(ses, which, outfpath.format(which)))


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
    return validate(path, os.path.isdir, valid_readable_file,
                    "Cannot read directory at '{}'")


def valid_readable_file(path):
    """
    Throw exception unless parameter is a valid readable filepath string. Use
    this, not argparse.FileType("r") which leaves an open file handle.
    :param path: Parameter to check if it represents a valid filepath
    :return: String representing a valid filepath
    """
    return validate(path, lambda x: os.access(x, os.R_OK),
                    os.path.abspath, "Cannot read file at '{}'")


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


if __name__ == "__main__":
    main()
    
