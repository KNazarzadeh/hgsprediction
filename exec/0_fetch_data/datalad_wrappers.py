#!/usr/bin/env python3
# --------------------------------- #

# --------------------------------#
# For debugging only!
# --------------------------------#

from ptpython.repl import embed # For debugging using ptpython
# print('OKKKK')
# embed(globals(), locals()) # --> In order to put a break point

# --------------------------------#
# Import necessary libraries
# --------------------------------#
# from . import hipsnp
import shutil
from os.path import expanduser
import warnings
import tempfile
import datalad.api as dl
import datalad
import os
import numpy as np
import pandas as pd
import time


# --------------------------------- #
home = expanduser("~")
get_exc = datalad.support.exceptions.IncompleteResultsError


class Fetcher:
    """ A generic class to wrap up some common datalad functionalities
    This can be used to create custom wrappers for specific datasets like UKB.
    When implementing a wrapper for a specific dataset make sure to initialise
    the Fetcher class with sensible default values for that dataset in the init
    method and then make any required additions.
    For example, Fetcher classes will need a dataset name (initialise the
    self.dataset_name attribute). This is the name of the root folder of the
    dataset.
    Attributes
    -----------
    self.directory : str or path
        absolute path leading up to the datalad dataset
    self.repo : str
        remote address for the datalad dataset to clone
    self.files : list of str or list of paths
        files the fetcher has fetched so far
        (relative paths from self.directory)
    """

    def __init__(
        self, repo=None, 
        directory=None, 
        log_filename="Fetcher_log.txt",
        HPC_name="Juseless",
        Subj_ID=None
    ):
        """
        Parameters
        ----------
        repo : str
            link to remote of the datalad dataset
        directory : str or path
            directory in which to clone the dataset
        """

        self.directory = directory
        self.repo = repo
        self.log_filename = log_filename
        self.HPC_name = HPC_name
        self.Subj_ID = Subj_ID

    def log(self, log_str):
        """
        Write a string to the input log file
        Parameters:
        ---------
        log_filename : str or path
            name of the log file
        log_str : str
            string to be written in the log file
        Returns
        --------
            None
        """

        assert isinstance(log_str, str), (
            "Please provide input to the logging function as a string!"
        )
        try:
            with open(self.log_filename, "a+") as f:
                f.write(log_str)
        except FileNotFoundError:
            print("Logfile does not exist!")
        finally:
            print(log_str)

    def clone_repo(self, repo=None, directory=None):
        """ Clone a remote-hosted datalad dataset.
        The purpose of this method is to easily create a temp folder for the
        dataset to be installed if no other directory is specified, clone the
        repo and keep track of where the datalad dataset is installed by
        updating the self.directory attribute (if no specific directory is
        provided)
        Parameters
        ----------
        repo : str
            remote address from which clone. If not provided, fetcher will use
            the address at self.repo (from the init method). Raises an Error if
            no remote address has been provided during initialisation or use of
            the method.
        directory : str or path
            if datalad should not be cloned into self.directory then you can
            provide another directory here to specifiy where to install it
        """

        if self.repo is None:
            self.repo = repo
            if repo is None:
                raise ValueError(
                    "Which repo should be cloned? Please provide a link!"
                )

        if self.directory is None:

            self.log("No Directory was specified. Making tmp directory\n")
            if(self.HPC_name=='Juseless'):
                self.directory = tempfile.mkdtemp()
            else:
                self.directory = tempfile.mkdtemp(
                    dir='/dev/shm', 
                    prefix=self.Subj_ID)

        dl.install(self.directory, source=self.repo)
        self.dataset = dl.Dataset(self.directory)

    def get_custom_file(self, path, tries=50, wait_period=1):
        """ Get a file from a datalad dataset with an arbitrary path from that
        dataset. For more details on the datalad.api.get function check here:
        https://docs.datalad.org/en/stable/generated/datalad.api.get.html
        The purpose of this function is to easily get the file, and keep track
        of the files that have been downloaded so far, so that these can be
        easily removed later.
        Parameters
        -----------
        path : str or path
            path of file to get relative from self.directory (the directory
            in which the datalad dataset was installed)
        tries : int
            how many times fetcher should try to get a file
        wait_period : int
            how long to wait until trying again
        """

        if self.directory is None:
            raise ValueError(
                "Please clone the repository first. Then you can get files!"
            )

        self.log("-----------------------------\n")
        self.log(
            f"DATALAD GETTING \n{path} \nIN DIRECTORY: \n{self.directory}\n"
        )

        assert tries > 0, "You have to at least try once!"
        tried = 0
        while tried < tries:
            try:
                self.dataset.get(path)
            except get_exc:
                tried += 1
                time.sleep(wait_period)
            break

        self.log("DATALAD DONE GETTING FILE\n")

        return os.path.join(self.dataset.pathobj, path)

    def drop_and_remove(self):
        """ Drop all files and remove this dataset
        This method should be called at the end of every processing routine to
        clean up the temporary directory if keeping the dataset installed is
        undesired.
        """

        self.log("-----------------------------\n")
        self.log("Removing Dataset\n")
        self.log("-----------------------------\n")
        dl.remove(dataset=self.directory, recursive=True)

    def __enter__(self):
        """ Enter method for context manager
        """
        self.clone_repo()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """ Exit method for context manager
        """
        self.drop_and_remove()

# --------------------------------- #
class UKBFetcher(Fetcher):
    def __init__(
        self,
        repo=(
            "ria+http://ukb.ds.inm7.de#~bids"
        ),
        log_filename=None,
        *args,
        **kwargs
    ):
        """ Initialise JuImaGen
        Parameters
        ----------
        repo : str
            link to remote of the datalad dataset
        directory : str or path
            directory in which to clone the dataset
        Attributes
        ----------
        self.TR : float
            time resolution in seconds

        """

        Fetcher.__init__(
            self,
            repo=repo,
            log_filename=log_filename,
            *args,
            **kwargs
        )
        self.TR = 0.72

    def img_subs_only(
            self,
            UKB_tsv_filename,
            ukbb_parser_out_prefix,
            ukbb_parser_out_folder,
            ukbb_parser_DF_code=None,
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
        Returns
        --------
        img_subs : pd.DataFrame
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

        UKB_out_file = f"{ukbb_parser_out_prefix}_img_subs_only"

        UKB_out_file_full = os.path.join(
            ukbb_parser_out_folder, f'{UKB_out_file}.csv'
        )
        if not os.path.isfile(UKB_out_file_full):

            database_folder = self.directory
            curr_dir = os.getcwd()
            os.chdir(database_folder)

            dl.get(UKB_tsv_filename)

            # Parse subjects, data fields and categories
            if not os.path.exists(ukbb_parser_out_folder):
                os.makedirs(ukbb_parser_out_folder)

            # Make the cmd string for extracting the desired Data-Field using
            # ukbb_parser
            if(not ukbb_parser_DF_code):
                data_field_list = ""
            else:
                data_field_list = ''
                for n_code in range(len(ukbb_parser_DF_code)):

                    data_field_list = f'{data_field_list} --inhdr {ukbb_parser_DF_code[n_code]}'

            # Extract requested info from the ukbb csv fil
            ukbb_cmd = (
                f"ukbb_parser parse --incsv {UKB_tsv_filename} -o {UKB_out_file}"
                f"{data_field_list} --img_subs_only --long_names"
            )
            # print("===== Done! =====")
            # embed(globals(), locals())
            self.log(
                "** ukbb_paser command fro extraction of UKB data fields: \n"
                f"{ukbb_cmd} \n"
                "**----------------------------\n"
            )

            os.system(ukbb_cmd)  # Run the ukbb_parser command

            # Move the output files of ukbb_parser from the current directory
            # to a desired location
            UKB_mv_cmd = f"mv {UKB_out_file}* {ukbb_parser_out_folder}"
            os.system(UKB_mv_cmd)

            img_subs = pd.read_csv(UKB_out_file_full, sep=',')

            dl.drop(UKB_tsv_filename)
            os.chdir(curr_dir)

            UKB_tsv_filename = os.path.join(self.directory, UKB_tsv_filename)

            self.log("-----------------------------\n")
            self.log(f"** UKB .tsv file (4.3 GB): {UKB_tsv_filename}\n")
            self.log("** This file has now been processed by ukbb_parser")
            self.log("** and has been dropped by datalad successfully!")
            self.log(f"The ukbb_parser output file: {UKB_out_file_full}")

        else:

            img_subs = pd.read_csv(UKB_out_file_full, sep=',')
            UKB_tsv_filename = os.path.join(self.directory, UKB_tsv_filename)

            self.log("------------------------------\n")
            self.log(f"** UKB .tsv file (11 GB): {UKB_tsv_filename}\n")
            self.log("** The desired UKB Data field has been already ")
            self.log("** extracted by ukbb_parser.")
            self.log(f"** The ukbb_parser output file: {UKB_out_file_full}")

        return img_subs, UKB_out_file_full
