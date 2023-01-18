#!/usr/bin/env python3

import os
import glob
import pandas as pd
import argparse

prog_descrip='Check tier1 locations for ABCD BIDS data and update the database accordingly'
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
    return parser

def tier1_get_bids_anats(bids_dir, subject, session):
    anat_paths = glob.glob(os.path.join(bids_dir, subject, session, 'anat/*.json'))
    anats = []
    T1s = []
    T2s = []
    for anat in anat_paths:
        if anat.endswith('_T1w.json'):
            T1s.append(anat)
        if anat.endswith('_T2w.json'):
            T2s.append(anat)
    for i in range(0,len(T1s)):
        bn = 'T1_run-0{}'.format(i+1)
        anats.append(bn)
    for i in range(0, len(T2s)):
        bn = 'T2_run-0{}'.format(i+1)
        anats.append(bn)
    return anats

def tier1_get_bids_funcs(bids_dir, subject, session):
    func_paths = glob.glob(os.path.join(bids_dir, subject, session, 'func/*.json'))
    try:
        funcs = []
        for func in func_paths:
            # figure out functional basename
            try:
                task = func.split('task-')[1].split('_')[0]
            except:
                raise Exception('this is not a BIDS folder. Exiting.')
            try:
                run = func.split('run-')[1].split('_')[0]
            except:
                run=''
            try:
                acq = func.split('acq-')[1].split('_')[0]
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

    def make_new_db():
        tier1_bids_db = pd.DataFrame(columns=['subject', 'session'])

        bids_dir = '/spaces/ngdr/ref-data/abcd/nda-3165-2020-09'
        for subject_path in glob.glob(os.path.join(bids_dir, 'sub-*')):
            subject = os.path.basename(subject_path)
            for session in os.listdir(subject_path):
                print('Checking tier1 for {} {}'.format(subject, session))
                anats = tier1_get_bids_anats(bids_dir, subject, session)
                funcs = tier1_get_bids_funcs(bids_dir, subject, session)
                subject_dict = {'subject': subject, 'session': session}
                for header in anats:
                    subject_dict[header] = 'bids'
                for header in funcs:
                    subject_dict[header] = 'bids'
                tier1_bids_db = tier1_bids_db.append(subject_dict, ignore_index=True)

        tier1_bids_db.to_csv('tier1_bids_db.csv', index=False)

    def update_db(bids_db_path):

        bids_db = pd.read_csv(bids_db_path) 

        bids_dir = '/spaces/ngdr/ref-data/abcd/nda-3165-2020-09'
        for subject_path in glob.glob(os.path.join(bids_dir, 'sub-*')):
            subject = os.path.basename(subject_path)
            for session in os.listdir(subject_path):
                print('Checking tier1 for {} {}'.format(subject, session))
                anats = tier1_get_bids_anats(bids_dir, subject, session)
                funcs = tier1_get_bids_funcs(bids_dir, subject, session)
                subject_db = bids_db[(bids_db['subject'] == subject) & (bids_db['session'] == session)]
                if subject_db.empty:
                    print('ERROR: Subject {}/{} in tier1 not in fastqc spreadhseet'.format(subject, session))
                    subject_dict = {'subject': subject, 'session': session}
                    for header in anats:
                        subject_dict[header] = 'delete (tier1)'
                    for header in funcs:
                        subject_dict[header] = 'delete (tier1)'
                    bids_db = bids_db.append(subject_dict, ignore_index=True)
                elif len(subject_db) == 1:
                    for header in anats:
                        if (subject_db[header] == 'no bids').all():
                            subject_db[header] = 'bids (tier1)'
                        else:
                             subject_db[header] = 'delete (tier1)'
                    for header in funcs:
                        if (subject_db[header] == 'no bids').all():
                            subject_db[header] = 'bids (tier1)'
                        else:
                             subject_db[header] = 'delete (tier1)'
            
                    bids_db.loc[(bids_db['subject'] == subject) & (bids_db['session'] == session)] = subject_db
                else:
                    print('ERROR: Multiple entries for {} {}'.format(subject, session))

                    

        bids_db.to_csv(bids_db_path, index=False)

    update_db(args.bids_db_file)



    return

if __name__ == '__main__':
    main()
        







