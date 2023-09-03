
import pandas as pd
import sys

from hgsprediction.load_data import stroke_load_data
from hgsprediction.data_preprocessing import stroke_data_preprocessor
from hgsprediction.save_data import stroke_save_data

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################
filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
stroke_cohort = sys.argv[3]
visit_session = sys.argv[4]
 
df = stroke_load_data.load_main_preprocessed_data(population, mri_status)

###############################################################################
data_processor = stroke_data_preprocessor.StrokeValidateDataPreprocessor(df, 
                                                            mri_status,
                                                            stroke_cohort, 
                                                            visit_session)
extracted_pre_post_data, session_column = data_processor.extract_data(df)
# Remove all columns with all NaN values
extracted_pre_post_data = data_processor.remove_nan_columns(extracted_pre_post_data)

extracted_pre_post_data_female = extracted_pre_post_data[extracted_pre_post_data["31-0.0"]==0.0]
extracted_pre_post_data_male = extracted_pre_post_data[extracted_pre_post_data["31-0.0"]==1.0]


hgs_validated_df, session_column = data_processor.validate_handgrips(extracted_pre_post_data)
hgs_validated_df_female, session_column = data_processor.validate_handgrips(extracted_pre_post_data_female)
hgs_validated_df_male, session_column = data_processor.validate_handgrips(extracted_pre_post_data_male)

print("===== Done! =====")
embed(globals(), locals())

stroke_save_data.save_validated_hgs_data(hgs_validated_df, population, mri_status, session_column, "both_gender")
stroke_save_data.save_validated_hgs_data(hgs_validated_df_female, population, mri_status, session_column, "female")
stroke_save_data.save_validated_hgs_data(hgs_validated_df_male, population, mri_status, session_column, "male")

stroke_save_data.save_original_extracted_pre_post_data(extracted_pre_post_data, population, mri_status, session_column, "both_gender")
stroke_save_data.save_original_extracted_pre_post_data(extracted_pre_post_data_female, population, mri_status, session_column, "female")
stroke_save_data.save_original_extracted_pre_post_data(extracted_pre_post_data_male, population, mri_status, session_column, "male")

print("===== Done! =====")
embed(globals(), locals())