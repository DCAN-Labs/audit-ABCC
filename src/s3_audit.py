#! /usr/bin/env python3

"""
AWS s3 ABCC Audit Script
Last Updated by Anders Perrone 2022-06-30
Last Updated by Greg Conan 2022-09-27
"""
# Standard imports
import argparse
import boto3
from getpass import getpass
import numpy as np
import os
import pandas as pd
import subprocess as sp
import urllib.parse

# Local custom imports
from utilities import (get_ERI_filepath,
                       get_tier1_or_tier2_ERI_db_fname, PATH_DICOM_DB,
                       PATH_NGDR, valid_output_dir, valid_readable_dir)


def generate_parser(parser=None):
    """ 
    :param parser: argparse.ArgumentParser to add arguments to (optional)
    :return: argparse.ArgumentParser with all command-line arguments from user
    """
    choices_run_modes = ("makeDB", "updateDB")  # TODO Figure out whether to make this script run in modes
    default_buckets = ("ABCC_year1", "ABCC_year2")
    default_host = "https://s3.msi.umn.edu"
    default_out_dir = os.path.join(os.getcwd(), "s3_audit_outputs")
    msg_default = " By default, the value for this argument will be {}"
    prog_descrip = ("Check tier 1 and 2 locations to find the EventRelatedInfo"
                    "rmation.txt files for every subject, session, task, and "
                    "run of some ABCD BIDS data, then save the files' paths.")
    if not parser:
        parser = argparse.ArgumentParser(description=prog_descrip)
    
    parser.add_argument(
        '-b',
        '--buckets',
        nargs='+',
        dest='buckets',
        default=default_buckets,
        help=("List of bucket names containing ABCD BIDS data."
              + msg_default.format(default_buckets))
    )
    parser.add_argument(
        "-bids",
        "--bids-dir",
        default=PATH_NGDR,
        type=valid_readable_dir,
        help=("Valid path to existing BIDS-formatted input data structure. "
              "Must be a valid path to an existing local directory."
              + msg_default.format(PATH_NGDR))
    )
    parser.add_argument(
        '-db',
        '--database',
        dest='bids_db_file',
        default=PATH_DICOM_DB,
        help=("Path to the current BIDS database in .csv format."
              + msg_default.format(PATH_DICOM_DB))
    )
    parser.add_argument(
        "-host",
        "--host",
        default=default_host,
        help=("URL path to AWS host." + msg_default.format(default_host))
    )
    parser.add_argument(
        "-keys",
        "--aws-keys",
        nargs="*",
        default=list(),
        help=("AWS s3 access key and secret key. Optionally, you may choose "
              "to give this flag two arguments, your AWS access key followed "
              "by your AWS secret key. Otherwise, by default this script will "
              "try to infer both keys by running the 's3info' command, and if "
              "that fails then you will be prompted to manually enter both "
              "keys (hidden from the terminal). Enter the word 'manual' to "
              "skip the s3info check and enter keys manually.")
    )
    parser.add_argument(
        "-out",
        "--output",
        type=valid_output_dir,
        default=default_out_dir,
        help=("Valid path to a directory to save output files into. If there "
              "is no directory at this path yet, then one will be created.")
    )
    return parser


def get_s3info():
    """
    :return: Dictionary containing all of the information in the output of the
             "s3info" command in the Unix Bash terminal (via subprocess)
    """
    user_s3info = sp.check_output(("s3info")).decode("utf-8").split("\n")
    aws_s3info = dict()
    for eachline in user_s3info:
        if eachline != "":
            split = eachline.split(":")
            if len(split) > 1:
                split = [x.strip() for x in split]
                aws_s3info[split[0].lower()] = split[-1]
    return aws_s3info
    

def s3_client(access_key, host, secret_key):
    """
    :param access_key: String, 20-character-long AWS access key
    :param host: String, AWS s3 host URL to connect to
    :param secret_key: String, 40-character-long AWS secret key
    :return: boto3.session.Session.client to access s3 buckets
    """
    session = boto3.session.Session()
    client = session.client('s3',endpoint_url=host,
                                 aws_access_key_id=access_key, 
                                 aws_secret_access_key=secret_key)
    return client


def s3_get_bids_subjects(client, bucketName, prefix=""):
    # TODO Test that this function still works after my edits
    paginator = client.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(
        Bucket=bucketName, Delimiter='/', Prefix=prefix, EncodingType='url',
        ContinuationToken='', FetchOwner=False, StartAfter=''
    )
    page_sub_ses_ERI = list()
    for page in page_iterator:
        print(s3_get_objects(client, bucketName, prefix.split("/")[0]))
        print(prefix)
        new_objects = [item["Key"] for item in page["CommonPrefixes"]["Contents"]]
        print("Added {}".format(new_objects[-1]))
        page_sub_ses_ERI.extend(new_objects)
        no_children = False
    if no_children:
        print(f"No children in {prefix}")
    return page_sub_ses_ERI


def s3_get_bids_sessions(client, bucketName, prefix):
    """ 
    :param client: boto3.session.Session.client to access s3 buckets
    :param bucketName: String naming an accessible s3 bucket
    :param host: String, URL path to AWS host
    :param prefix: String naming the (s3 equivalent of a) parent directory
    :return: List of strings, the names of all sessions w/ data matching prefixs
    """  # TODO Test that this still works after my edits
    s3_data = s3_get_objects(client, bucketName, prefix,
                             Delimiter='/', MaxKeys=1000)
    bids_sessions = [item['Prefix'].split('/')[1] for item in s3_data['CommonPrefixes'] if 'ses' in item['Prefix'].split('/')[1]]
    return bids_sessions


def s3_get_bids_anats(s3_data):  # TODO Test that this still works after my edits
    try:
        anats_T = dict()
        for t in range(1, 3):
            anats_T[t] = list()
        for obj in s3_data['Contents']:
            key = obj['Key']
            for t in range(1, 3):
                if key.endswith(f"_T{t}w.nii.gz"):
                    anats_T[t].append(key)
        all_anats = list()
        for t in range(1, 3):
            for i in range(0, len(anats_T[t])):
                all_anats.append("T{}_run-0{}".format(t, i+1))
    except KeyError:
        all_anats = None
    return all_anats


def s3_get_objects(client, bucketName, prefix, **kwargs):
    """
    :param client: boto3.session.Session.client to access s3 buckets
    :param bucketName: String naming an accessible s3 bucket
    :param prefix: String naming the (s3 equivalent of a) parent directory
    :return: List of s3 file objects fetched via list_objects_v2
    """
    try:
        s3_data = list()
        paginator = client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=bucketName, Prefix=prefix,
                                           StartAfter='', ContinuationToken='',
                                           EncodingType='url', FetchOwner=False,
                                           **kwargs)
        for page in page_iterator:
            s3_data.extend(page["Contents"])
    except KeyError:
        s3_data = None
    return s3_data


def s3_get_bids_funcs(s3_data):  # TODO Test that this still works after my edits
    suffix='_bold.nii.gz' # looking for functional nifti files
    try:
        funcs = []
        for obj in s3_data['Contents']:
            key = obj['Key'] 
            if 'func' in key and key.endswith(suffix):
                # figure out functional basename
                task = get_functional_basename_of(key, "task-")
                if not task:
                    raise ValueError('this is not a BIDS folder. Exiting.')
                func = [f"task-{task}"]
                run = get_functional_basename_of(key, "run-")
                if not run:
                    run = "01"
                acq = get_functional_basename_of(key, "acq-")
                if acq:
                    func.append(f"acq-{acq}")
                funcs.append("_".join([*func, f"run-{run}"]))
        funcs = list(set(funcs))
        return funcs
    except KeyError:
        return


def get_functional_basename_of(key, base):
    """
    :param key: String, the full name of a subject session file
    :param base: String, the part of key to split on; e.g. "acq-" or "run-"
    :return: String, everything in key that's between base and the next "_"
    """
    try:
        result = key.split(base)[1].split('_')[0]
    except (IndexError, TypeError, ValueError):
        result=""
    return result


def s3_get_anat_and_func_imgs(client, bucket, subject, session):
    print('Checking S3 for {} {}'.format(subject, session))
    s3_data = s3_get_objects(client, bucket, "/".join([subject, session, ""]))
    return {"anat": s3_get_bids_anats(s3_data),
            "func": s3_get_bids_funcs(s3_data)}


def validate_aws_keys_args(user_input, parser):
    """
    Raise error if the user did not give a valid AWS key pair
    :param user_input: List which should have 2 strings, the first a valid
                       AWS access key and the second a valid AWS secret key
    :param parser: argparse.ArgumentParser to raise error if anything's wrong
    :return: user_input, but only if user_input is a valid AWS key pair
    """
    err_msg = None
    if len(user_input) != 2:
        err_msg = ("Please give exactly two keys, your access key and "
                    "your secret key (in that order).")  
    elif len(user_input[0]) != 20: 
        err_msg = "Your AWS access key must be 20 characters long."
    elif len(user_input[1]) != 40:
        err_msg = "Your AWS secret key must be 40 characters long."
    if err_msg:
        parser.error(err_msg)
    return user_input


def get_AWS_credential(cred_name, cli_args):
    """
    If AWS credential was a CLI arg, return it; otherwise prompt user for it 
    :param cred_name: String naming which credential (username or password)
    :param input_fn: Function to get the credential from the user
    :param cli_args: Dictionary containing all command-line arguments from user
    :return: String, user's NDA credential
    """
    return cli_args[cred_name] if cli_args.get(cred_name) else getpass(
        "Enter your AWS S3 {}: ".format(cred_name)
    )


def find_ERI_on_s3_for(row, client, bucket, mappings):
    """
    _summary_ 
    :param row: pandas.Series
    :param client: boto3.session.Session.client to access s3 buckets
    :param bucket: String, name of a valid accessible AWS s3 bucket
    :param mappings: Dictionary mapping bucket names to session names
    :return: List of ERI text file s3 objects if any exist 
    """
    session = row.get("session") 
    try:
        assert mappings[bucket] == session
        subject = row.get("subject")
        all_objects = s3_get_bids_subjects(client, bucket, "sourcedata/{}/{}/func/".format(subject, session))
    except (AssertionError, KeyError):
        print(f"No s3 ERI data for {session} in {bucket}")
        all_objects = None
    return all_objects


def find_and_save_all_ERI_paths(cli_args, client):
    """
    Get the paths to EventRelatedInformation.txt files for as many subject
    sessions in the BIDS DB as possible, on tier 1 (MSI) or 2 (s3), and save
    out a .csv file containing all ERI txt file paths for each subject-session-
    task-run. Assumes there's a BIDS database .csv at cli_args["bids_db_file"].
    :param cli_args: Dictionary containing all command-line arguments from user
    :param client: boto3.session.Session.client to access s3 buckets
    """
    # Local variables: BIDS DB DF and column names to check for ERI files of
    bids_db = pd.read_csv(cli_args["bids_db_file"])
    cols = [c for c in bids_db.columns.values.tolist() if "run-" in c] # c.split("_")[0][:5] == "task-" or c.split("_")[0][0] == "T"]
    bids_db[cols] = np.nan  # Clear out BIDS DB values; we only need rows/cols

    # Update DB with ERI paths on tier1/bids_db
    bids_db = bids_db.apply(lambda row: find_ERI_on_tier1_for(
                                cli_args["bids_dir"], row, cols
                            ), axis=1)
    save_out_bids_db_to_csv(bids_db, cli_args["output"], 1)

    # Update DB with ERI paths on tier2/s3
    all_objects = dict()
    all_ERI_paths = dict()
    for bucketName in cli_args["buckets"]:
        all_ERI_paths[bucketName] = list()
        all_objects[bucketName] = ["s3://{}/{}".format(bucketName, urllib.parse.unquote(obj["Key"])) for obj in
                                   s3_get_objects(client, bucketName,
                                                  "sourcedata")]
        for each_ERI_URL in all_objects[bucketName]:
            fname = each_ERI_URL.rsplit("/", 1)[-1]
            subject, session, task, run, _, _ = fname.split("_")
            sub_ses_ix = int(bids_db[(bids_db["subject"] == subject) &
                                     (bids_db["session"] == session)].index[0])
            for col in cols:  # TODO Make this more efficient by using Series.apply instead of a triply(!) nested for loop
                split = col.split("_")
                if split[0] == task and split[1] == run:
                    bids_db.iloc[sub_ses_ix][col] = each_ERI_URL
    save_out_bids_db_to_csv(bids_db, cli_args["output"], 2)


def save_out_bids_db_to_csv(bids_db, out_dir, tier):
    """
    Save all ERI file paths in a pd.DataFrame into a .csv file
    :param bids_db: pandas.DataFrame containing ERI file paths
    :param out_dir: String, valid path to existing dir to save .csv files into
    :param tier: Int, the tier number (either 1 for MSI or 2 for s3)
    """
    outfile_path = get_tier1_or_tier2_ERI_db_fname(out_dir, tier)
    bids_db.to_csv(outfile_path, index=False)
    print(f"Saved tier {tier} ERI .txt file paths to {outfile_path}")


def find_ERI_on_tier1_for(bids_dir, row, cols):
    result = row
    subject = row.get("subject")
    session = row.get("session")
    for col in cols:
        task, run = [c.split("-")[-1] for c in col.split("_")]
        path = get_ERI_filepath(bids_dir, subject, session, task, run)
        if os.path.exists(path):
            result.loc[col] = path
    return result


def s3_get_subject_dict(sub_ses_imgs, subject, session, assign_to_header):
    subject_dict = {'subject': subject, 'session': session}
    for img_type in sub_ses_imgs.keys():
        for header in sub_ses_imgs[img_type]:
            subject_dict[header] = assign_to_header
    return subject_dict


def get_and_validate_AWS_s3_credentials(cli_args, parser):
    """
    Get AWS credentials, either from s3info or from the user entering them
    :param cli_args: Dictionary, all command-line arguments from user
    :param parser: argparse.ArgumentParser to raise error if anything's invalid
    :return: Dictionary mapping 
    """
    key_names = ("access key", "secret key")  # AWS s3 key names
    aws_creds = dict()  # Return value
    aws_keys = cli_args.get("aws_keys")
    if aws_keys == ["manual"]:
        aws_keys = list()
    elif aws_keys:
        for i in range(len(key_names)):
            aws_creds[key_names[i]] = aws_keys[i]
    else:
        try:
            aws_creds = get_s3info()
            aws_keys = [aws_creds.get("access key"),
                        aws_creds.get("secret key")]
        except (sp.CalledProcessError, ValueError):
            pass
    for each_key in key_names:
        if not aws_creds.get(each_key):
            aws_creds[each_key] = get_AWS_credential(each_key, cli_args)
            aws_keys.append(aws_creds[each_key])
    validate_aws_keys_args(aws_keys, parser)
    return aws_creds


def main():
    # Get (and validate) command-line arguments and AWS s3 credentials from user
    parser = generate_parser()
    cli_args = vars(parser.parse_args())
    aws_creds = get_and_validate_AWS_s3_credentials(cli_args, parser)

    # Get all ERI paths on MSI and s3, then save them into .csv files
    client = s3_client(aws_creds["access key"], cli_args["host"], aws_creds["secret key"])
    find_and_save_all_ERI_paths(cli_args, client)


if __name__ == "__main__":
    main()