#!/usr/bin/env Disorderspredwp3
"""
Fetch original data from UK Biobank by using ukbb_parser
========================================================

This code fetch original data for different populations
from UK Biobank by using ukbb_parser.

All categories, feild IDs defined on UkbbParams class on 

# Author:  Kimia Nazarzadeh <k.nazarzadeh@fz-juelich.de>

"""

import os
import sys
# import argparse
import pandas as pd
from datalad_wrappers import UKBFetcher

# inserting the modules directory at
# from arguments_input import parse_args
from func_fetch_ukb_csv import fetch_ukb_csv
from ukbb_parser_parameters import UkbbParams

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())


###############################################################################
# Parse, add and return the arguments by function parse_args.
# Define motor, population and mri status to run the code:
filename = sys.argv[0]
motor = sys.argv[1]
population = sys.argv[2]
mri_status = sys.argv[3]

###############################################################################
# Initial the constant variables
# ishealthy variable for determining the population:
    # 1 --> Healthy controls 
    # 0 --> diseases (here stroke or parkinson)
if population in "healthy" :
    ishealthy=1
else:
    ishealthy=0
    
# ----- mri : binary
# The binary values of MRI data status for subjects with/without MRI
    # 0 --> All data (subjects non-MRI+MRI all together )
    # 1 --> MRI data (subjects with MRI data only)
if mri_status == "mri":
    mri = 1
elif mri_status == "nonmri":
    mri = 0
###############################################################################
# Module1: Load healthy/Stroke/Stroke types parameters (Class UkbbParams)
# from ukbb_params.py file
# ---------------------------------#
# Module1: Load healthy/Stroke/Stroke types parameters (Class UkbbParams)
# ---------------------------------#
# ----- create ukbb_params class variable ----------------------------------#
# inputs are:
    # motor
    # ishealthy
    # population
    # mri
ukbb_params = UkbbParams(motor, ishealthy, population, mri)
# ----- Create ukbb_parser command string ----------------------------------#
#calling the function: get_ukbb_parser_cmd() on ukbb_params.py/UkbbParams class

data_field_list = ukbb_params.get_ukbb_parser_cmd()

# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################
#Create folders and file names to save the fetched data on Juseless
# ----------- On Juseless ----------#
# ----- The main folder is the directory folder to save all sub-folders
# ----------------------------------#
ukbb_parser_out_folder = os.path.join(
    "/data",
    "project",
    "stroke_ukb",
    "knazarzadeh",
    "project_hgsprediction",
    "data_hgs",
    f"{population}",
    "original_data",
    )

# ----- make the directory folder if it has not created before
if(not os.path.isdir(ukbb_parser_out_folder)):
    os.mkdir(ukbb_parser_out_folder)

############################################################################### 
# ---------------------------------#
# ----- UKB database source for datalad
# ---------------------------------#
datalad_database_source = "ria+http://ukb.ds.inm7.de#~bids"
# ----- Folder of the analysis log files
logs_folder = os.path.join(
    ukbb_parser_out_folder,
    f'{motor}_analysis_logs')

# ----- make the directory folder if it has not created before
if(not os.path.isdir(logs_folder)):
    os.mkdir(logs_folder)

log_filename = os.path.join(logs_folder,
        f'log_{motor}_ukbb_parser.txt')

# ----- make the directory folder if it has not created before
if(os.path.isfile(log_filename)):
    os.remove(log_filename)

###############################################################################
# ----------------------------------------------------------------------------#
# ----- mri : binary
# The binary values of MRI data status for subjects with/without MRI
    # 0 --> All data (subjects non-MRI+MRI data all together)
    # 1 --> MRI data (subjects with MRI data only)
if mri == 0:
        # sub_population_folder: 
        # for sub folder of MRI or all subjects(non-MRI+MRI subjects together)
    sub_population_folder = f"all_{population}"

elif mri == 1:
        # sub_population_folder: 
        # for sub folder of MRI or all subjects(non-MRI+MRI subjects together)
    sub_population_folder = f"mri_{population}"

# ----- Define the output folder to save .csv files
# based on sub_population_folder and the main population directory folder
ukbb_parser_out_folder = os.path.join(ukbb_parser_out_folder, sub_population_folder)

# ----- make the directory folder if it has not created before
if(not os.path.isdir(ukbb_parser_out_folder)):
        os.mkdir(ukbb_parser_out_folder)

# ----- Define the output .csv file name and join to the directory
ukbb_parser_out_prefix = f"{sub_population_folder}"
ukbb_parser_out_file = os.path.join(
        ukbb_parser_out_folder,
        f'{ukbb_parser_out_prefix}.csv'
        )
# print('Done!')
# embed(globals(), locals()) #
###############################################################################
# ---------------------------------#
# UKBFetcher toolkit to make everything ready for fetching
# ---------------------------------#
# from merged_toolkit.preprocessing.datalad_wrappers import UKBFetcher
with UKBFetcher(
                repo = datalad_database_source,
                log_filename = log_filename,
        ) as myUKBFetcher:
        database_folder = myUKBFetcher.directory
        myUKBFetcher.log(f'Database folder: {database_folder}\n')

        # ---------------------------------------#
        # Module2: Get the behavioral data of UKB subjects
        # using ukbb_parser
        # ---------------------------------------#
        # The .tsv filename on UK Biobank
        UKB_tsv_filename = os.path.join(
            database_folder,
            'ukb670018.tsv'
            # 'ukb668954.tsv'
        )
        # The old file: 'ukb45132.tsv'
        # print('Done!')
        # embed(globals(), locals()) # --> In order to put a break point
# ------------------------------
        # Use fetch_ukb_csv function to fetch data from UK Biobank 
        # by using ukbb_parser command
        # if the file has not created before, call fetch_ukb_csv function
        # to fetch data and make file for it
        if(not os.path.isfile(ukbb_parser_out_file)):
            ukb_data_csv, UKB_out_file_full = fetch_ukb_csv(
                database_folder,
                UKB_tsv_filename,
                ukbb_parser_out_prefix,
                ukbb_parser_out_folder,
                data_field_list
            )
            # read the output in .csv format
            ukb_csv = pd.read_csv(ukbb_parser_out_file, sep=',')
        # if the file has created before, read it in .csv format
        else:
            # read the output in .csv format
            ukb_csv = pd.read_csv(ukbb_parser_out_file, sep=',')
###############################################################################


print("===== Done! =====")
embed(globals(), locals())
