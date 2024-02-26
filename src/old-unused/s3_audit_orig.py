#! /usr/bin/env python3

import boto3
import pandas as pd
import numpy as np
import argparse
import os

HOST='https://s3.msi.umn.edu'
BUCKETS=['ABCC_year1','ABCC_year2']

prog_descrip='Check s3 locations for ABCD BIDS data and update the database accordingly'
BIDS_DB = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "spreadsheets",
    "ABCD_BIDS_database.csv")

def generate_parser(parser=None):
    if not parser:
        parser = argparse.ArgumentParser(
            description=prog_descrip
        )
    parser.add_argument(
        '-db',
        '--database',
        dest='bids_db_file',
        default=BIDS_DB,
        help='Path to the current BIDS database in csv format '
    )
    parser.add_argument(
        '-b',
        '--buckets',
        nargs='+',
        dest='buckets',
        default=BUCKETS,
        help='List of bucket names containing ABCD BIDS data (e.g. --buckets ABCC_year1 ABCC_year2)'
    )
    parser.add_argument(
        '--access-key',
        dest='access_key',
        help='Access key to s3 buckets'
    )
    parser.add_argument(
        '--secret-key',
        dest='secret_key',
        help='Secret key to s3 buckets'
    )
    return parser

def s3_client(access_key,host,secret_key):
    session = boto3.session.Session()
    client = session.client('s3',endpoint_url=host,
                                 aws_access_key_id=access_key, 
                                 aws_secret_access_key=secret_key)
    return client

def s3_get_bids_subjects(access_key,bucketName,host,prefix,secret_key):
    # prefix = ABCC_year2/
    client = s3_client(access_key=access_key,host=host,secret_key=secret_key)
    paginator = client.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucketName,Delimiter='/',Prefix=prefix,EncodingType='url',ContinuationToken='',
                                             FetchOwner=False,
                                             StartAfter='')
    get_data = client.list_objects_v2(Bucket=bucketName,Delimiter='/',EncodingType='url',
                                            Prefix=prefix,
                                             MaxKeys=1000,
                                             ContinuationToken='',
                                             FetchOwner=False,
                                             StartAfter='')
    bids_subjects = []
    for page in page_iterator:
        page_bids_subjects = ['sub-'+item['Prefix'].split('sub-')[1].strip('/') for item in page['CommonPrefixes'] if 'sub' in item['Prefix']]
        bids_subjects.extend(page_bids_subjects)
    return bids_subjects

def s3_get_bids_sessions(access_key,bucketName,host,prefix,secret_key):
    # prefix = subject ID
    client = s3_client(access_key=access_key,host=host,secret_key=secret_key)
    get_data = client.list_objects_v2(Bucket=bucketName,Delimiter='/',EncodingType='url',
                                          MaxKeys=1000,
                                          Prefix=prefix,
                                          ContinuationToken='',
                                          FetchOwner=False,
                                          StartAfter='')
    bids_sessions = [item['Prefix'].split('/')[1] for item in get_data['CommonPrefixes'] if 'ses' in item['Prefix'].split('/')[1]]
    return bids_sessions

def s3_get_bids_anats(access_key,bucketName,host,prefix,secret_key):
    client = s3_client(access_key=access_key,host=host,secret_key=secret_key)
    try:
        get_data = client.list_objects_v2(Bucket=bucketName,EncodingType='url',
                                          Prefix=prefix,
                                          ContinuationToken='',
                                          FetchOwner=False,
                                          StartAfter='')   
    except KeyError:
        return
    try:
        anats = []
        T1s = []
        T2s = []
        for obj in get_data['Contents']:
            key = obj['Key']
            if key.endswith('_T1w.nii.gz'):
                T1s.append(key)
            if key.endswith('_T2w.nii.gz'):
                T2s.append(key)
        for i in range(0,len(T1s)):
            bn = 'T1_run-0{}'.format(i+1)
            anats.append(bn)
        for i in range(0, len(T2s)):
            bn = 'T2_run-0{}'.format(i+1)
            anats.append(bn)
        return anats
    except KeyError:
        return

def s3_get_bids_funcs(access_key,bucketName,host,prefix,secret_key):
    client = s3_client(access_key=access_key,host=host,secret_key=secret_key)
    suffix='_bold.nii.gz' # looking for functional nifti files
    try:
        get_data = client.list_objects_v2(Bucket=bucketName,EncodingType='url',
                                          Prefix=prefix,
                                          ContinuationToken='',
                                          FetchOwner=False,
                                          StartAfter='')   
    except KeyError:
        return
    try:
        funcs = []
        for obj in get_data['Contents']:
            key = obj['Key'] 
            if 'func' in key and key.endswith(suffix):
                # figure out functional basename
                try:
                    task = key.split('task-')[1].split('_')[0]
                except:
                    raise Exception('this is not a BIDS folder. Exiting.')
                try:
                    run = key.split('run-')[1].split('_')[0]
                except:
                    run=''
                try:
                    acq = key.split('acq-')[1].split('_')[0]
                except:
                    acq=''
                if not run:
                    if not acq:
                        funcs.append('task-'+task+'_run-01')
                    else:
                        funcs.append('task-'+task+'_acq-'+acq+'_run-01')
                else:
                    if not acq:
                        funcs.append('task-'+task+'_run-'+run)
                    else:
                        funcs.append('task-'+task+'_acq-'+acq+'_run-'+run)
        funcs = list(set(funcs))
        return funcs
    except KeyError:
        return



def main():

    parser = generate_parser()
    args = parser.parse_args()
    ACCESS_KEY = args.access_key
    SECRET_KEY = args.secret_key

    def make_new_db():

        s3_bids_db = pd.DataFrame(columns=['subject', 'session'])

        prefix = ''

        client = s3_client(ACCESS_KEY, HOST, SECRET_KEY)
        bids_subjects = s3_get_bids_subjects(ACCESS_KEY, BUCKET, HOST, prefix, SECRET_KEY)

        for subject in bids_subjects:
            prefix = '/'.join([subject, ''])
            bids_sessions = s3_get_bids_sessions(ACCESS_KEY, BUCKET, HOST, prefix, SECRET_KEY)
            for session in bids_sessions:
                print('Checking S3 for {} {}'.format(subject, session))
                subject_dict = {'subject': subject, 'session': session}
                prefix = '/'.join([subject, session, ''])
                anats = s3_get_bids_anats(ACCESS_KEY, BUCKET, HOST, prefix, SECRET_KEY)
                for header in anats:
                    subject_dict[header] = 'bids'
                funcs = s3_get_bids_funcs(ACCESS_KEY, BUCKET, HOST, prefix, SECRET_KEY)
                for header in funcs:
                    subject_dict[header] = 'bids'
                s3_bids_db = s3_bids_db.append(subject_dict, ignore_index=True)
                
        s3_bids_db.to_csv('s3_bids_db.csv', index=False)

    def update_db(bids_db_path, buckets):
        bids_db = pd.read_csv(bids_db_path)

        for BUCKET in buckets:

            print('Checking {}'.format(BUCKET))
            
            prefix = ''

            client = s3_client(ACCESS_KEY, HOST, SECRET_KEY)

            bids_subjects = s3_get_bids_subjects(ACCESS_KEY, BUCKET, HOST, prefix, SECRET_KEY)

            print('Found {} subjects'.format(len(bids_subjects)))

            for subject in bids_subjects:
                prefix = '/'.join([subject, ''])
                bids_sessions = s3_get_bids_sessions(ACCESS_KEY, BUCKET, HOST, prefix, SECRET_KEY)
                for session in bids_sessions:
                    print('Checking S3 for {} {}'.format(subject, session))
                    subject_dict = {'subject': subject, 'session': session}
                    prefix = '/'.join([subject, session, ''])
                    anats = s3_get_bids_anats(ACCESS_KEY, BUCKET, HOST, prefix, SECRET_KEY)
                    funcs = s3_get_bids_funcs(ACCESS_KEY, BUCKET, HOST, prefix, SECRET_KEY)
                    subject_db = bids_db[(bids_db['subject'] == subject) & (bids_db['session'] == session)]
                    if subject_db.empty:
                        print('ERROR: Subject {}/{} in s3 not in fastqc spreadhseet'.format(subject, session))
                        subject_dict = {'subject': subject, 'session': session}
                        for header in anats:
                            subject_dict[header] = 'delete (s3)'
                        for header in funcs:
                            subject_dict[header] = 'delete (s3)'
                        bids_db = bids_db.append(subject_dict, ignore_index=True)
                    elif len(subject_db) == 1:
                        for header in anats + funcs:
                            try:
                                if (subject_db[header] == 'no bids').all():
                                    subject_db[header] = 'bids (s3)'
                                elif (subject_db[header] == 'bids (tier1)').all():
                                    subject_db[header] = 'bids (tier1) (s3)'
                            except:
                                subject_db[header] = 'delete (s3)'
                                bids_db[header] = np.nan
                        bids_db.loc[(bids_db['subject'] == subject) & (bids_db['session'] == session)] = subject_db
                    else:
                        print('ERROR: Multiple entries for {} {}'.format(subject, session))

        bids_db.to_csv(bids_db_path, index=False)
         
    update_db(args.bids_db_file, args.buckets)

if __name__ == "__main__":
    main()


