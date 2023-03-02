#!/usr/bin/env Disorderspredwp3
"""
Predict the motor based on specific features on populations data.
Motor is Handgrip strength (1 phase).

# Author:  Kimia Nazarzadeh <k.nazarzadeh@fz-juelich.de>

# License: AGPL

"""

import pandas as pd
from hgsprediction.input_arguments import parse_args
from hgsprediction.load_data import load_original_data_per_session
from hgsprediction.preprocess import PreprocessData


from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())


###############################################################################
# Parse, add and return the arguments by function parse_args.
args = parse_args()
# Define motor, population and mri status to run the code:
motor = args.motor
population = args.population
mri_status = args.mri_status
feature_type = args.feature_type
target = args.target
gender = args.gender
model = args.model
confound_status = args.confound_status

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
if gender == "both":
    print("Gender = both genders")
else:
    print("Gender =", gender)
if model == "rf":
    print("Model = random forest")
else:
    print("Model =", model)
if confound_status == 0:
    print("Confound_status = Without Confound Removal")
else:
    print("Confound_status = With Confound Removal")

print("============================================")

###############################################################################
# Read CSV file from Juseless
###############################################################################
# Read CSV file from Juseless
data_loaded = load_original_data_per_session(motor, population, mri_status, session=0)

data_hgs = check_hgs_availability_per_session(data_loaded, session=0)

extract_features = ExtractFeatures(data_hgs, motor, population)
extracted_data = extract_features.extract_features()



