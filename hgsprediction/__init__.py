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

from .save_results.healthy import save_hgs_predicted_results, save_correlation_results, save_trained_model_results, save_corrected_prediction_correlation_results, save_corrected_prediction_results

from .load_results import load_trained_models
from .load_results import load_corrected_prediction_results, load_corrected_prediction_correlation_results
from .load_results.healthy import load_hgs_predicted_results
# , load_correlation_results

from .save_plot.save_correlations_plot import healthy_save_correlations_plot
from .plots.plot_correlations import healthy_plot_hgs_correlations


############## Disorders ##################
# Load Stroke Data
from .load_data import disorder_load_data
# Save Stroke Data
from .save_data import disorder_save_data

from .compute_features import disorder_compute_features
# Target Computing
from .compute_target import disorder_compute_target
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
from .predict_hgs import calculate_pearson_hgs_correlation
from .load_results.stroke import load_hgs_predicted_results, load_hgs_predicted_results_mri_records_sessions_only
from .plots import plot_correlations
from .save_plot import save_correlations_plot, save_correlations_plot_mri_records_sessions_only

from .save_results.stroke_save_spearman_correlation_results import stroke_save_spearman_correlation_results,\
    stroke_save_spearman_correlation_results_mri_records_sessions_only

from .save_results.stroke_save_hgs_predicted_results import stroke_save_hgs_predicted_results,\
    stroke_save_hgs_predicted_results_mri_records_sessions_only
                  
from .old_define_features import stroke_define_features



from .predict_hgs import predict_hgs

from .predict_hgs import calculate_brain_hgs
from .predict_hgs import calculate_t_valuesGMV_HGS

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

## ************************************************************************************************** #

from .save_data import save_multi_samplesize_training_data
from .save_results import save_multi_samples_trained_model_results
## ************************************************************************************************** #

from .load_data import load_multi_samplesize_training_data
from .load_results.load_multi_samples_trained_models_results import load_scores_trained


from .save_results.brain_save_correlates_results import *


from .load_results.load_brain_correlates_results import *

from .load_results.load_trained_model_results import *

from .prediction_corrector_model import *

from .save_results.healthy.save_corrected_prediction_results import *
from .save_results.healthy.save_corrected_prediction_correlation_results import *

# from load_results.load_corrected_prediction_correlation_results import load_corrected_prediction_correlation_results
# from load_results.load_corrected_prediction_results import load_corrected_prediction_results

from .extract_data import disorder_extract_data

from .save_results.save_disorder_hgs_predicted_results import save_disorder_hgs_predicted_results

from .load_results.load_hgs_predicted_results import load_hgs_predicted_results

from .save_results.save_zscore_results import save_zscore_results
from .load_results.load_zscore_results import load_zscore_results

from .load_results.load_disorder_hgs_predicted_results import load_disorder_hgs_predicted_results
from .load_results.load_disorder_corrected_prediction_results import load_disorder_corrected_prediction_results
from .save_results.save_disorder_corrected_prediction_results import save_disorder_corrected_prediction_results
from .save_results.save_disorder_corrected_prediction_correlation_results import save_disorder_corrected_prediction_correlation_results
from .load_results.load_disorder_corrected_prediction_correlation_results import load_disorder_corrected_prediction_correlation_results
from .save_results.save_disorder_matched_samples_results import save_disorder_matched_samples_results

from .load_results.load_disorder_matched_samples_results import load_disorder_matched_samples_results
from .save_results.save_disorder_matched_samples_correlation_results import save_disorder_matched_control_samples_correlation_results