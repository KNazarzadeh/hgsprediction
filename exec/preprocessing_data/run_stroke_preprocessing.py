


import pandas as pd
from sys import argv
import numpy as np
from hgsprediction.load_data import load_original_data
from hgsprediction.data_preprocessing import StrokePreprocessor
from hgsprediction.save_data import save_prepared_disease_data

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################
filename = argv[0]
motor = argv[1]
population = argv[2]
mri_status = argv[3]

df_original = load_original_data(motor=motor, population=population, mri_status=mri_status)

###############################################################################

prepare_data = StrokePreprocessor(df_original)
df_available_disease_dates = prepare_data.remove_missing_stroke_dates(df_original)
df_available_hgs = prepare_data.remove_missing_hgs(df_available_disease_dates)



print("===== Done! =====")
embed(globals(), locals())