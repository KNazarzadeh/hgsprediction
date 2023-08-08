import os
import pandas as pd
import numpy as np
import sys

from hgsprediction.input_arguments import parse_args, input_arguments
from hgsprediction.load_imaging_data import load_imaging_data
from hgsprediction.load_trained_model import load_best_model_trained

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())


filename = sys.argv[0]
population = sys.argv[1]

img_type = sys.argv[1]
neuroanatomy = sys.argv[2]

###############################################################################
# Parse, add and return the arguments by function parse_args.
args = parse_args()
motor, population, mri_status, feature_type, target, gender, model_name, \
    confound_status, cv_repeats_number, cv_folds_number = input_arguments(args)
    
###############################################################################
best_model_trained = load_best_model_trained(
                                population,
                                gender,
                                feature_type,
                                target,
                                confound_status,
                                model_name,
                                cv_repeats_number,
                                cv_folds_number,
                            )
print(best_model_trained)

###############################################################################
img_df = load_imaging_data(img_type,neuroanatomy)

print("===== Done! =====")
embed(globals(), locals())