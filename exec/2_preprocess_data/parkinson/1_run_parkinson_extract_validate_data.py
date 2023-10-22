import pandas as pd
import sys

from hgsprediction.load_data import parkinson_load_data
from hgsprediction.data_preprocessing import parkinson_data_preprocessor
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################
filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]

df_original = parkinson_load_data.load_original_data(population=population, mri_status=mri_status)

data_processor = parkinson_data_preprocessor.ParkinsonMainDataPreprocessor(df_original)
df = data_processor.remove_missing_hgs(df_original)

print("===== Done! =====")
embed(globals(), locals())