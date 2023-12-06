"""Provide imports for the hgsprediction package."""
############### HEALTHY FUNCTIONS ###############
# Load Data
from .load_data import healthy_load_data
# Preprocessing Data
from .data_preprocessing import HealthyDataPreprocessor
# Features Computing
from .compute_features import healthy_compute_features
# Target Computing
from .compute_target import healthy_compute_target
from .compute_extra_column import healthy_compute_extra_column
# Save Data
from .save_data import healthy_save_data

from .extract_data import healthy_extract_data

from .save_results.healthy import save_hgs_predicted_results, save_spearman_correlation_results, save_trained_model_results

from .load_results import load_trained_models
from .load_results.healthy import load_hgs_predicted_results, load_spearman_correlation_results

from .save_plot.save_correlations_plot import healthy_save_correlations_plot
from .plots.plot_correlations import healthy_plot_hgs_correlations
############### STROKE FUNCTIONS ###############
# Load Stroke Data
from .load_data import stroke_load_data
# Data Preprocessing
from .data_preprocessing import stroke_data_preprocessor
# Features Computing
from .compute_features import stroke_compute_features
# Target Computing
from .compute_target import stroke_compute_target
# Save Stroke Data
from .save_data import stroke_save_data

from .extract_data import stroke_extract_data
from .compute_extra_column import stroke_compute_extra_column
from .predict_hgs import calculate_spearman_hgs_correlation
from .load_results.stroke import load_hgs_predicted_results, load_hgs_predicted_results_mri_records_sessions_only
from .plots import plot_correlations
from .save_plot import save_correlations_plot, save_correlations_plot_mri_records_sessions_only

from .save_results.stroke_save_spearman_correlation_results import stroke_save_spearman_correlation_results,\
    stroke_save_spearman_correlation_results_mri_records_sessions_only

from .save_results.stroke_save_hgs_predicted_results import stroke_save_hgs_predicted_results,\
    stroke_save_hgs_predicted_results_mri_records_sessions_only
                  
from .old_define_features import stroke_define_features



from .predict_hgs import predict_hgs

from .predict_hgs import calculate_spearman_hgs_correlation_on_brain_correlations


# from hgsprediction.input_arguments import parse_args, input_arguments
# from hgsprediction.load_imaging_data import load_imaging_data
# from hgsprediction.prepare_stroke.prepare_stroke_data import prepare_stroke



################################## PARKINSON DISEASE ##################################
from .load_data import parkinson_load_data
from .data_preprocessing import parkinson_data_preprocessor
from .save_data import parkinson_save_data
from .compute_features import parkinson_compute_features
from .extract_data import parkinson_extract_data
from .save_results.parkinson_save_spearman_correlation_results import parkinson_save_spearman_correlation_results
from .save_results.parkinson_save_hgs_predicted_results import parkinson_save_hgs_predicted_results
from .load_results.healthy import load_hgs_predicted_results

################################## Major Depressive DISOREDR (MDD) ##################################
from .load_data import depression_load_data
from .data_preprocessing import depression_data_preprocessor
from .save_data import depression_save_data
from .compute_features import depression_compute_features
from .extract_data import depression_extract_data
from .save_results.depression_save_spearman_correlation_results import depression_save_spearman_correlation_results
from .save_results.depression_save_hgs_predicted_results import depression_save_hgs_predicted_results
from .load_results.healthy import load_hgs_predicted_results

# # ************************************************************************************************** #
# """LOAD TRAIN SET"""
# # Load Primary Train set (after binning and splitting to Train and test)
# # from .load_data.load_healthy import load_primary_train_set_df
# # Load Processed Train set (after data validation, feature engineering)
# from .load_data.load_healthy import load_preprocessed_train_df
# ############################################################################
# ############################################################################
# ############################################################################
# from .data_preprocessing import healthy_data_preprocessor, DataPreprocessor







# from .preprocess import PreprocessData
# # Parse and check validate input arguments to fetch data
# from .input_arguments import parse_arguments
# #
# from .load_data import load_hgs_data_per_session, load_original_data_per_session, load_original_data
# from .prepare_data.prepare_disease import PrepareDisease
# from .save_data.save_disease import save_prepared_disease_data
# # from .load_data.load_disease import load_prepared_data
# from .preprocess.check_hgs_availability_healthy import check_hgs_availability
# # from define_targets import define_targets
# from .compute_target import healthy_compute_target

# from .extract_features import features_extractor
# from .extract_target import target_extractor
# ############################################################################
# ########################  IMPORT SAVE FUNCTIONS  ###########################
# ############################################################################
# from .save_results import save_extracted_nonmri_data, \
#                           save_best_model_trained, \
#                           save_scores_trained
# ############################################################################
# ##################  IMPORT LOAD BEST MODEL TRAINED FUNCTION  ###############
# ############################################################################
# from .load_trained_model import load_best_model_trained
# ############################################################################
# from .load_data.load_healthy import load_mri_data, \
#                                     load_mri_data_for_anthropometrics
# ############################################################################
# from .save_results import save_extracted_mri_data, \
#                           save_tested_mri_data

# from .load_imaging_data.load_brain_imaging_data import load_imaging_data

# from .data_preprocessing import DataPreprocessor

# from .LinearSVRHeuristicC_zscore import LinearSVR, LinearSVRHeuristicC_zscore
# from .data_extraction import data_extractor, run_data_extraction

# ############################################################################
# # Remove columns that all values are NaN
# from .prediction_preparing import remove_nan_columns

# # Run feature extraction
# from .features_extraction import features_extractor

# from .prepare_stroke.prepare_stroke_data import prepare_stroke

# ########################################################################################################################################################
# ########################################################################################################################################################
# ########################################################################################################################################################
# # STROKE
# from .data_preprocessing import stroke_data_preprocessor

# # PLOTS
# # from .plots.make_plot import create_regplot

# from .load_data.load_healthy.load_train_data import load_primary_mri_df, load_primary_nonmri_train_set_df, load_preprocessed_train_df


# from .compute_features import Features