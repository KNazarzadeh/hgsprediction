
import pandas as pd
import sys

from hgsprediction.load_data import stroke_load_data
from hgsprediction.data_preprocessing import stroke_data_preprocessor

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################
filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]

df_original = stroke_load_data.load_original_data(population=population, mri_status=mri_status)
###############################################################################
print("===== Done! =====")
embed(globals(), locals())

data_processor = stroke_data_preprocessor.StrokeMainDataPreprocessor(df_original)
df = data_processor.remove_missing_stroke_dates(df_original)
df = data_processor.remove_missing_hgs(df)
###############################################################################

df = data_processor.define_stroke_type(df)
df = data_processor.define_followup_days(df)
###############################################################################
df_post_stroke = data_processor.extract_post_stroke_df(df)
df_pre_stroke = data_processor.extract_pre_stroke_df(df)
df_longitudinal_sroke = data_processor.extract_longitudinal_stroke_df(df)
###############################################################################
df_all_post_stroke_visits = data_processor.extract_all_post_stroke_visits(df)
df_all_pre_stroke_visits = data_processor.extract_all_pre_stroke_visits(df)

