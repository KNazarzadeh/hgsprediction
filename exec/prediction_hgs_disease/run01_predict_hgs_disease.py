#!/home/knazarzadeh/miniconda3/envs/disorderspredwp3/bin/python3

"""
Predict the motor based on specific features on populations data.
Motor is Handgrip strength (1 phase).

# Author:  Kimia Nazarzadeh <k.nazarzadeh@fz-juelich.de>

# License: AGPL

"""

import pandas as pd
import numpy as np
from hgsprediction.input_arguments import parse_args
from hgsprediction.extract_features import ExtractFeatures

from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score
from julearn import run_cross_validation

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())


###############################################################################
# Parse the input arguments by function parse_args.
args = parse_args()
# Define the following parameters to run the code:
# motor, population name, mri status and confound removal status, 
# Type of feature, target and gender 
# Model, the number of repeats and folds for run_cross_validation
# motor is hgs or handgrip_strength(str)
motor = args.motor
# populations are: healthy, stroke or parkinson(str)
population = args.population
# MRI status: mri or nonmri(str)
mri_status = args.mri_status
# Type of features:(str)
# cognitive, cognitive+gnder
# bodysize, bodysize+gender
# bodysize+cognitive, bodysize+cognitive+gender
feature_type = args.feature_type
# Target(str): L+R(for HGS(Left +Right)), dominant_hgs or nondominant_hgs
target = args.target
# Type of genders: both (female+male), female and male
gender = args.gender
# Type of models(str): linear_svm, random forest(rf)
model = args.model
# 0 means without confound removal(int)
# 1 means with confound removal(int)
confound_status = args.confound_status
# Number of repeats for run_cross_validation(int)
n_repeats = args.repeat_number
# Number of folds for run_cross_validation(int)
n_folds = args.fold_number
###############################################################################
# Print summary of all inputs
print("================== Inputs ==================")
# print Motor type
if motor == "hgs":
    print("Motor = handgrip strength")
else:
    print("Motor =", motor)
# print Population type
print("Population =", population)
# print MRI status
print("MRI status =", mri_status)
# print Feature type
print("Feature type =", feature_type)
# print Target type
print("Target =", target)
# print Gender type
if gender == "both":
    print("Gender = both genders")
else:
    print("Gender =", gender)
# print Model type
if model == "rf":
    print("Model = random forest")
else:
    print("Model =", model)
# print Confound status 
if confound_status == 0:
    print("Confound status = Without Confound Removal")
else:
    print("Confound status = With Confound Removal")
# print Number of repeats for run_cross_validation
print("Repeat Numbers =", n_repeats)
# print Number of folds for run_cross_validation
print("Fold Numbers = ", n_folds)
print("============================================")

###############################################################################