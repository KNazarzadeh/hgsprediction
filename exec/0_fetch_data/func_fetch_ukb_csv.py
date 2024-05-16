# --------------------------------#
# Import necessary libraries
# --------------------------------#
# from . import hipsnp
import shutil
from os.path import expanduser
import datalad.api as dl
import os
import numpy as np
import pandas as pd
from ptpython.repl import embed


def fetch_ukb_csv(
        database_folder,
        UKB_tsv_filename,
        ukbb_parser_out_prefix,
        ukbb_parser_out_folder,
        data_field_list,
):
    """ Extract the list of UKB subject IDs with neuroimaging data

    Parameters
    -----------
    UKB_tsv_filename : str or path
        file path to the UKB's big csv file (~11GB)
    kbb_out_prefix : str
        A prefix for the output csv file as a sub-table of the UKB csv file
    ukbb_parser_out_folder: str or path
        folder name to save the output file
    ukbb_parser_DF_code : str
        The desired Data-Filed to be extracted from UK Biobank
    ukbb_parser_DF_exclude : str
        The desired Data-Filed to be excluded from UK Biobank Data

    Returns
    --------
    ukb_data_csv : pd.DataFrame
        A dataframe including the list of UKB subjects with neuroimaging
        data
    UKB_out_file_full : str or path
        full address of the output csv file including the list of UKB 
        subjects with fMRI data
    see also:
        - https://github.com/USC-IGC/ukbb_parser

    """

    # sanity check
    ukbb_parser = shutil.which('ukbb_parser')

    if ukbb_parser is None:
        print('ukbb_parser is not available')
        raise

    UKB_out_file = f"{ukbb_parser_out_prefix}"

    UKB_out_file_full = os.path.join(
        ukbb_parser_out_folder, f'{UKB_out_file}.csv'
    )

    if not os.path.isfile(UKB_out_file_full):

        curr_dir = os.getcwd()
        os.chdir(database_folder)
        dl.get(UKB_tsv_filename)

        # Parse subjects, data fields and categories
        if not os.path.exists(ukbb_parser_out_folder):
            os.makedirs(ukbb_parser_out_folder)
        
        # Extract requested info from the ukbb csv fil
        ukbb_cmd = (
            f"ukbb_parser parse --incsv {UKB_tsv_filename} --out {UKB_out_file}"
            f"{data_field_list}"
        )
        # print("===== Done! =====")
        # embed(globals(), locals())
        os.system(ukbb_cmd)  # Run the ukbb_parser command

        # Move the output files of ukbb_parser from the current directory
        # to a desired location
        UKB_mv_cmd = f"mv {UKB_out_file}* {ukbb_parser_out_folder}"
        os.system(UKB_mv_cmd)

        ukb_data_csv = pd.read_csv(UKB_out_file_full, sep=',')

        # dl.drop(UKB_tsv_filename)
        os.chdir(curr_dir)

        UKB_tsv_filename = os.path.join(database_folder, UKB_tsv_filename)

    else:

        ukb_data_csv = pd.read_csv(UKB_out_file_full, sep=',')
        UKB_tsv_filename = os.path.join(database_folder, UKB_tsv_filename)

    return ukb_data_csv, UKB_out_file_full
