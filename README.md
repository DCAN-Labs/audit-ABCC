# Temporary notes on contents of output files:

- `s3_BIDS_DB_<DATE>.tsv`: Lists files present for each modality on S3
- `audit-full_BIDS_DB_<DATE>.tsv`: lists QC for each potential file and whether it's present on s3 and NGDR or not
- `audit-summary_BIDS_DB_<DATE>.tsv`: summarizes the files. DETAILS FOR THIS FILE:

> Header named `complete` includes following value options:
>- `incomplete`: incomplete on both tier1 and tier2
>- `compplete`: complete on both tier1 and tier2
>- `complete (s3)`: complete on only tier2/s3
>- `complete (tier1)`: complete on only tier1/NGDR
>
>Header names `coverage_s3` and `coverage_tier1` include values that are the percentage of expected files present on s3 and tier1 (i.e. NGDR) respectively



# Potentially useful files: 
- `complete-subj-ses_BIDS_DB_2024-08-30_19-15-35.tsv`: list of complete subjects/sessions (have all expected files)
- `incomplete-subj-ses_BIDS_DB_2024-08-30_19-15-35.tsv`: list of incomplete subjects/sessions (missing some or all expected files)


# Other
- `no-files-complete_BIDS_DB_2024-08-30_19-15-35.tsv` : list of subjects/sessions missing all expected files from s3, but all expected files present on NGDR??
- `no-files-incomplete_BIDS_DB_2024-08-30_19-15-35.tsv`: list of subjects/sessions missing all expected files from s3, and incomplete on NGDR as well???
- `tier1_BIDS_DB_2024-08-30_18-27-53.tsv`
- `ftqc_BIDS_DB_2024-08-30_18-27-53.tsv`
