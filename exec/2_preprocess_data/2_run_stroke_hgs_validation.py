
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
stroke_cohort = sys.argv[3]
visit_session = sys.argv[4]
 
df = stroke_load_data.load_preprocessed_data(population, mri_status, stroke_cohort)
stroke_cohort = "post-stroke"
df = df[df["1st_post-stroke_session"]>=0]
###############################################################################
data_processor = stroke_data_preprocessor.StrokeValidateDataPreprocessor(df, 
                                                            mri_status,
                                                            stroke_cohort, 
                                                            visit_session)

# print("===== Done! =====")
# embed(globals(), locals())
hgs_validated_df = data_processor.validate_handgrips(df)

print("===== Done! =====")
embed(globals(), locals())
