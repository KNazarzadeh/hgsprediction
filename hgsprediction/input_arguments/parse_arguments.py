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
import sys

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
        "handgrip",
        "gripstrength",
        "grip_strength",
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
        "anthropometrics",
        "anthropometrics_gender",
        "anthropometrics_age",
        "behavioral",
        "behavioral_gender",
        "anthropometrics_behavioral",
        "anthropometrics_behavioral_gender",
    ]
    
    available_target = [
        "hgs_L+R",
        "hgs_dominant",
        "hgs_nondominant",
        "hgs_left",
        "hgs_right",
        "hgs_LI",
        "hgs_L-R"
    ]
    
    available_confound = [
        0, # no confound removal
        1, # with confound removal
    ]
    
    available_gender = [
        "female",
        "male",
        "both_gender",
    ]
    
    available_model = [
        "linear_svm",
        "random_forest",
    ]
    
    if args.motor not in available_motor:
        print("Invalid Motor Type!")
        print("please choose Motor type from the list:\n",
              available_motor)
        sys.exit()
    if args.population not in available_population:
        print("Invalid Population Name!")
        print("please choose Population name from the list:\n",
              available_population)
        sys.exit()
    if args.mri_status not in available_mri_status:
        print("Invalid MRI Status!")
        print("please choose MRI status from the list:\n",
              available_mri_status)
        sys.exit()
        
    if args.feature_type not in available_feature:
        print("Invalid Features type!")
        print("please choose Features type from the list:\n",
              available_feature,"\n anthropometrics --> anthropometric features"
                                "\n anthropometrics_gender --> anthropometrics with gender features"
                                "\n anthropometrics_age --> anthropometrics with age features"
                                "\n behavioral --> behavioral features"
                                "\n behavioral_gender --> behavioural with gender features"
                                "\n anthropometrics_behavioral --> anthropometrics with behavioral features"
                                "\n anthropometrics_behavioral_gender --> anthropometrics with behavioral and gender features")
        sys.exit()
    if args.target not in available_target:
        print("Invalid Target!")
        print("please choose Target from the target list:\n",
              available_target, "\n hgs_L+R for Left+Right HGS \n hgs_dominant for dominant HGS \n hgs_nondominant for non-dominant HGS \
                   \n hgs_left for Left HGS  \n hgs_right for Right HGS  \n hgs_L-R for Left-Right HGS")
        sys.exit()
    if args.gender not in available_gender:
        print("Invalid Gender!")
        print("please choose Gender from the gender list:\n",
              available_gender)
        sys.exit()
    if args.model_name not in available_model:
        print("Invalid Model!")
        print("please choose Model from the model list:\n",
              available_model, "\n linear_svm for Linear SVM \n random_forest for Random Forest")
        sys.exit()
    if args.confound_status not in available_confound:
        print("Invalid Confound Status!")
        print("please choose Confound status from the binary list:\n",
              available_confound, "\n 0 for without confound removal \n 1 for with confound removal")
        sys.exit()
        
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
                        # choices=["anthropometrics",
                                    # "anthropometrics_gender",
                                    # "anthropometrics_age"
                                    # "behavioural",
                                    # "behavioural_gender",
                                    # "anthropometrics_behavioural",
                                    # "anthropometrics_behavioural_gender",
                                    # "all",],
                        help="Features type (str).")
    # Add Target argument:
    parser.add_argument("target",
                        type=str,
                        # choices=["hgs_L+R",
                        #         "hgs_dominant",
                        #         "hgs_nondominant",
                        #         "hgs_left",
                        #         "hgs_right",
                        #         "hgs_L-R",],
                        help="Confound status (int).")
    # Add Gender argument:
    parser.add_argument("gender",
                        type=str.lower,
                        # choices=["female",
                        #         "male",
                        # "both_gender"],
                        help="Gender (str).")
    # Add Model argument:
    parser.add_argument("model_name",
                        type=str.lower,
                        # choices=["linear_svm",
                        #         "random_forest",
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

###############################################################################
def input_arguments(args):
    # Define motor, population and mri status to run the code:
    motor = args.motor
    population = args.population
    mri_status = args.mri_status
    feature_type = args.feature_type
    target = args.target
    gender = args.gender
    model_name = args.model_name
    confound_status = args.confound_status
    cv_repeats_number = args.repeat_number
    cv_folds_number = args.fold_number

    # Print all input
    print("================== Inputs ==================")
    if motor == "hgs":
        print("Motor = handgrip strength")
    else:
        print("Motor =", motor)
    print("Population =", population)
    print("MRI status =", mri_status)
    print("Feature type =", feature_type)
    print("Target =", target)
    if gender == "both_gender":
        print("Gender = both genders")
    else:
        print("Gender =", gender)
    if model_name == "random_forest":
        print("Model = random forest")
    elif model_name == "linear_svm":
        print("Model =", "Linear SVM")
    if confound_status == 0:
        print("Confound_status = Without Confound Removal")
    else:
        print("Confound_status = With Confound Removal")

    print("CV Repeat Numbers =", cv_repeats_number)
    print("CV Fold Numbers = ", cv_folds_number)

    print("============================================")

    return motor, population, mri_status, feature_type, target, gender, model_name, confound_status, cv_repeats_number, cv_folds_number