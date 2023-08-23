#!/usr/bin/env Disorderspredwp3

"""
Compute Target, Calculate and Add new columns based on corresponding Field-IDs and conditions

# Author:  Kimia Nazarzadeh <k.nazarzadeh@fz-juelich.de>

"""

import numpy as np
import pandas as pd

from ptpython.repl import embed

###############################################################################
class StrokeExtractTarget:
    def __init__(self, 
                 df, 
                 mri_status,
                 stroke_cohort, 
                 visit_session,
                 target):
        """Preprocess data, Calculate and Add new columns to dataframe

        Parameters
        ----------
        df : dataframe
            The dataframe that desired to analysis
        """
        self.df = df
        self.mri_status = mri_status
        self.stroke_cohort = stroke_cohort
        self.visit_session = visit_session
        self.target = target
        
        assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
        assert isinstance(mri_status, str), "mri_status must be a string!"
        assert isinstance(stroke_cohort, str), "stroke_cohort must be a string!"
        assert isinstance(visit_session, str), "visit_session must be a string!"
        assert isinstance(target, str), "target must be a string!"

        if visit_session == "1":
            self.session_column = f"1st_{stroke_cohort}_session"
        elif visit_session == "2":
            self.session_column = f"2nd_{stroke_cohort}_session"
        elif visit_session == "3":
            self.session_column = f"3rd_{stroke_cohort}_session"
        elif visit_session == "4":
            self.session_column = f"4th_{stroke_cohort}_session"
###############################################################################
# This class extract all required targets from data:
    def extract_target(self):
        
        # Assign corresponding session number from the Class:
        session_column = self.session_column
        target = self.target
        
        assert isinstance(session_column, str), "session_column must be a string!"
        target_list = []
        
        if target == "hgs_L+R":
            target_list = self.extract_sum_hgs()
                
        elif target == "hgs_left":
            target_list = self.extract_left_hgs()

        elif target == "hgs_right":
            target_list = self.extract_right_hgs()
            
        elif target == "hgs_dominant":
            target_list = self.extract_dominant_hgs()
            
        elif target == "hgs_nondominant":
            target_list = self.extract_nondominant_hgs()
            
        elif target == "hgs_LI":
            target_list = self.extract_laterality_index_hgs()
            
        elif target == "hgs_L-R":
            target_list = self.extract_sub_hgs()        
        
        return target_list
###############################################################################
    def extract_sum_hgs(self):
        """Calculate sum of Handgrips
        and add "hgs(L+R)" column to dataframe

        Parameters
        ----------
        df : dataframe
            The dataframe that desired to analysis

        Return
        ----------
        df : dataframe
            with calculating extra column for: (HGS Left + HGS Right)
        """
        # Assign corresponding session number from the Class:
        session_column = self.session_column
        

        assert isinstance(session_column, str), "session_column must be a string!"
        # -----------------------------------------------------------
        substring_to_remove = "session"
        # Add a new column 'new_column'
        hgs_sum = session_column.replace(substring_to_remove, "hgs_L+R")

        return hgs_sum

###############################################################################
    def extract_left_hgs(self):
        """Calculate right and add "hgs(left)" column to dataframe

        Parameters
        ----------
        df : dataframe
            The dataframe that desired to analysis

        Return
        ----------
        df : dataframe
            with calculating extra column for:
            HGS(Left)
        """
        # Assign corresponding session number from the Class:
        session_column = self.session_column
        

        assert isinstance(session_column, str), "session_column must be a string!"
        # -----------------------------------------------------------
        substring_to_remove = "session"
        # Add a new column 'new_column'
        hgs_left = session_column.replace(substring_to_remove, "hgs_left")
  
        return hgs_left

###############################################################################
    def extract_right_hgs(self):
        """Calculate right and add "hgs(right)" column to dataframe

        Parameters
        ----------
        df : dataframe
            The dataframe that desired to analysis

        Return
        ----------
        df : dataframe
            with calculating extra column for:
            HGS(Right)
        """
        # Assign corresponding session number from the Class:
        session_column = self.session_column
        

        assert isinstance(session_column, str), "session_column must be a string!"
        # -----------------------------------------------------------
        substring_to_remove = "session"
        # Add a new column 'new_column'
        hgs_right = session_column.replace(substring_to_remove, "hgs_right")

        return hgs_right

###############################################################################
    def extract_sub_hgs(self):
        """Calculate subtraction of Handgrips
        and add "hgs(L-R)" column to dataframe

        Parameters
        ----------
        df : dataframe
            The dataframe that desired to analysis

        Return
        ----------
        df : dataframe
            with calculating extra column for: (HGS Left - HGS Right)
        """
        # Assign corresponding session number from the Class:
        session_column = self.session_column
        

        assert isinstance(session_column, str), "session_column must be a string!"
        # -----------------------------------------------------------
        substring_to_remove = "session"
        # Add a new column 'new_column'
        hgs_sub = session_column.replace(substring_to_remove, "hgs_L-R")

        return hgs_sub

###############################################################################
    def extract_laterality_index_hgs(self):
        """Calculate Laterality Index and add "hgs(LI)" column to dataframe

        Parameters
        ----------
        df : dataframe
            The dataframe that desired to analysis

        Return
        ----------
        df : dataframe
            with calculating extra column for:
            HGS(Left - Right)/HGS(Left + Right)
        """
        # Assign corresponding session number from the Class:
        session_column = self.session_column
        

        assert isinstance(session_column, str), "session_column must be a string!"
        # -----------------------------------------------------------
        substring_to_remove = "session"
        hgs_LI = session_column.replace(substring_to_remove, "hgs_LI")

        return hgs_LI

###############################################################################
    def extract_dominant_hgs(self):
        """Calculate dominant handgrip
        and add "hgs_dominant" column to dataframe

        Parameters
        ----------
        df : dataframe
            The dataframe that desired to analysis

        Return
        ----------
        df : dataframe
            with extra column for: Dominant hand Handgrip strength
        """
        # Assign corresponding session number from the Class:
        session_column = self.session_column
        

        assert isinstance(session_column, str), "session_column must be a string!"
        # -----------------------------------------------------------
        substring_to_remove = "session"
        # Add a new column 'new_column'
        hgs_dominant = session_column.replace(substring_to_remove, "hgs_dominant")
                
        return hgs_dominant
    
    ###############################################################################
    def extract_nondominant_hgs(self):
        """Calculate dominant handgrip
        and add "hgs_dominant" column to dataframe

        Parameters
        ----------
        df : dataframe
            The dataframe that desired to analysis

        Return
        ----------
        df : dataframe
            with extra column for: Dominant hand Handgrip strength
        """
        # Assign corresponding session number from the Class:
        session_column = self.session_column
        

        assert isinstance(session_column, str), "session_column must be a string!"
        # -----------------------------------------------------------
        substring_to_remove = "session"
        # Add a new column 'new_column'
        hgs_nondominant = session_column.replace(substring_to_remove, "hgs_nondominant")
                
        return hgs_nondominant