
import pandas as pd
import sys

from hgsprediction.load_data import stroke_load_data
from hgsprediction.data_preprocessing import stroke_data_preprocessor
from hgsprediction.save_data import save_preprocessed_data

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################
filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]


df_original = stroke_load_data.load_original_data(population=population, mri_status=mri_status)

###############################################################################

data_processor = stroke_data_preprocessor.StrokeMainDataPreprocessor(df_original)
df = data_processor.remove_missing_stroke_dates(df_original)
df = data_processor.remove_missing_hgs(df)

###############################################################################
df = data_processor.define_stroke_type(df)
df = data_processor.define_followup_days(df)
###############################################################################
df_preprocessed = data_processor.preprocess_stroke_df(df)
###############################################################################
df_post_stroke = data_processor.extract_post_stroke_df(df_preprocessed)
df_pre_stroke = data_processor.extract_pre_stroke_df(df_preprocessed)
df_longitudinal_stroke = data_processor.extract_longitudinal_stroke_df(df_preprocessed)

###############################################################################
save_preprocessed_data(df_preprocessed, population, mri_status, stroke_cohort="original_preprocessed")

save_preprocessed_data(df_post_stroke, population, mri_status, stroke_cohort="post-stroke")
save_preprocessed_data(df_pre_stroke, population, mri_status, stroke_cohort="pre-stroke")
save_preprocessed_data(df_longitudinal_stroke, population, mri_status, stroke_cohort="longitudinal-sroke")

print("===== Done! =====")
embed(globals(), locals())



