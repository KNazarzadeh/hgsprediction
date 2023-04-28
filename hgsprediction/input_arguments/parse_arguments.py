#!/usr/bin/env Disorderspredwp3
"""
Parse and check validate input arguments to fetch data.
=======================================================

This module provides a convenient interface to handle
command-line arguments. It parse, add and check arguments
and display help and errors.

# Author:  Kimia Nazarzadeh <k.nazarzadeh@fz-juelich.de>

"""

import argparse


###############################################################################
# Check validation of arguments and give an error if they failed.
def validate_args(args):
    """Check that arguments are valid.

    Fail and print errors if arguments are not valid.

    Parameters
    ----------
    args

    """
    available_motor = [
        "handgrip_strength",
        "hgs",
        "handgrip"
    ]
    available_population = [
        "stroke",
        "healthy",
        "parkinson",
    ]
    available_mri_status = [
        "mri",
        "nonmri",
    ]
    
    available_feature = [
        "anthropometric",
        "anthropometric+gender",
        "behavioural",
        "behavioural+gender",
        "anthropometric+behavioural",
        "anthropometric+behavioural+gender",
        "all",
    ]
    
    available_target = [
        "L+R",
        "dominant",
        "nondominant",
    ]
    
    available_confound = [
        0,
        1,
    ]
    
    available_gender = [
        "female",
        "male",
        "both",
    ]
    
    available_model = [
        "linear_svm",
        "rf",
    ]
    
    if args.motor not in available_motor:
        print("Invalid Motor Type!")
        print("please choose Motor type from the list:\n",
              available_motor)
    if args.population not in available_population:
        print("Invalid Population Name!")
        print("please choose Population name from the list:\n",
              available_population)
    if args.mri_status not in available_mri_status:
        print("Invalid MRI Status!")
        print("please choose MRI status from the list:\n",
              available_mri_status)
    if args.feature_type not in available_feature:
        print("Invalid Features type!")
        print("please choose Features type from the list:\n",
              available_feature)
    if args.target not in available_target:
        print("Invalid Target!")
        print("please choose Target from the list:\n",
              available_target)
    if args.gender not in available_gender:
        print("Invalid Gender!")
        print("please choose Gender from the binary list:\n",
              available_gender)
    if args.model not in available_model:
        print("Invalid Model!")
        print("please choose Model from the binary list:\n",
              available_model)
    if args.confound_status not in available_confound:
        print("Invalid Confound Status!")
        print("please choose Confound status from the binary list:\n",
              available_confound)

###############################################################################
# Parse, add and return the arguments.
def parse_args():
    """Parse arguments for fetch data."""

    # Create the parser
    parser = argparse.ArgumentParser(
        description="Define Motor, Population and MRI status."
    )
    # Add an argument
    # Add motor type argument:
    parser.add_argument("motor",
                        type=str.lower,
                        # choices=["handgrip_strength",
                        #         "hgs",],
                        help="Motor type (str).")
    # Add population name argument:
    parser.add_argument("population",
                        type=str.lower,
                        # choices=["healthy",
                        #         "stroke",
                        #         "parkinson",],
                        help="Population name (str).")
    # Add MRI status argument:
    parser.add_argument("mri_status",
                        type=str.lower,
                        # choices=["mri",
                        #         "nonmri",],
                        help="MRI status (str).")
    # Add Features type argument:
    parser.add_argument("feature_type",
                        type=str.lower,
                        # choices=["anthropometric",
                        # "anthropometric+gender",
                        # "behavioural",
                        # "behavioural+gender",
                        # "anthropometric+behavioural",
                        # "anthropometric+behavioural+gender",],
                        help="Features type (str).")
    # Add Target argument:
    parser.add_argument("target",
                        type=str,
                        # choices=["L+R",
                        #         "dominant",
                        #         "nondominant",],
                        help="Confound status (int).")
    # Add Gender argument:
    parser.add_argument("gender",
                        type=str.lower,
                        # choices=["female",
                        #         "male",
                        # "both"],
                        help="Gender (str).")
    # Add Model argument:
    parser.add_argument("model",
                        type=str.lower,
                        # choices=["linear_svm",
                        #         "rf",
                        # ],
                        help="Model (str).")
    # Add Confound status argument:
    parser.add_argument("confound_status",
                        type=int,
                        # choices=[0,
                        #         1,],
                        help="Confound status (int).")
    # Add Repeat numbers:
    parser.add_argument("repeat_number",
                        type=int,
                        # choices=[0,
                        #         1,],
                        help="Repeat Number (int).")
    # Add Fold numbers:
    parser.add_argument("fold_number",
                        type=int,
                        # choices=[0,
                        #         1,],
                        help="Fold Number (int).")
    
    # Parse the argument
    args = parser.parse_args()
    # Check validate of arguments
    validate_args(args)

    return args
