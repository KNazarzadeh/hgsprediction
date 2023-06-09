# Load trained models
"""
The function loads model trained on Non-MRI Healthy data
with different parameters:
    motor
    population
    mri status
    feature
    target
    gender
    model
    confound status
    repeat numbers
    fold numbers
"""

import os
import pickle

###############################################################################
def load_trained_models(
    motor,
    population,
    mri_status,
    feature,
    target,
    gender,
    model,
    confound_status,
    n_repeats,
    n_folds,
):
    """
    The function constructs a file path based on the provided parameters to locate the trained model file.
    It determines the appropriate file path based on the gender specified.
    
    Parameters
    ----------
    motor: str
        A string representing the motor type or category:
        - "hgs": for Handgrip strength
    population: str
        A string representing the population or sample:
        - "healthy"
        - "stroke
    mri_status: str
        A string representing the MRI status:
        - "mri": for MRI data
        - "nonmri": for Non-MRI data
    feature: str
        A string representing the feature type"
        - "anthropometric"
        - "anthropometric+age"
        - "behavioural"
        - "anthropometric+behavioural"
        - "anthropometric+gender"
        - "behavioural+gender"
        - "anthropometric+behavioural+gender"
    target: str
        A string representing the target type:
        - L+R: for (Left HGS + right HGS)
        - L-R: for (Left HGS - right HGS)
        - LI: for Laterality Index=(Left HGS - right HGS)/(Left HGS + right HGS)
    gender: str
        A string representing the gender:
        - "both": for both genders(males and females)
        - "male": for only males
        - "female: for only females
    model: str
        A string representing the model type:
        - "linear_svm": for Linear SVM
        - "rf": for Random Forest
    confound_status: binarry
        An integer (0 or 1) indicating the confound status:
        - 0: without confound removal
        - 1: with confound removal
    n_repeats: int
        An integer representing the number of repeats.
    n_folds: int
        An integer representing the number of folds.
    Return
    ----------
    model_trained : pickle format
        The trained model object loaded from the specified file path.
    """
    if confound_status == 0:
        confound = "without_confound_removal"
    else:
        confound = "with_confound_removal"

    if model == "rf":
        model_name = "random_forest"
    if model == "svm":
        model_name = "linear_svm"
        
    if "+" in feature:
            feature_type = feature.replace("+", "_")
    else:
        feature_type = feature

    if target == "L+R":
        target_label = "L_plus_R"
    else:
        target_label = target

    open_folder_path = os.path.join(
            "/data",
            "project",
            "stroke_ukb",
            "knazarzadeh",
            "GIT_repositories",
            "motor_ukb",
            "results",
            f"{motor}_prediction",
            f"results_{population}",
            f"results_{mri_status}",
            f"results_{gender}_genders",
            f"{feature_type}_features",
            f"{target_label}_target",
            f"{model_name}",
            f"{confound}",
            f"{n_repeats}_repeats_{n_folds}_folds",
            f"results_csv",
            "model_trained",
        )

    # Define the csv file path to save
    if gender == "both":
        open_file_path = os.path.join(
            open_folder_path,
            f"main_model_trained_{mri_status}_{population}_{gender}_genders_{feature_type}_{target_label}_{model_name}_{confound}_{n_repeats}_repeats_{n_folds}_folds.pkl")
        with open(open_file_path, 'rb') as f:
            model_trained = pickle.load(f)
    
    if gender == "female":
        # Define the csv file path to save
        open_file_path_female = os.path.join(
            open_folder_path,
            f"model_trained_female_{mri_status}_{population}_{gender}_genders_{feature_type}_{target_label}_{model_name}_{confound}_{n_repeats}_repeats_{n_folds}_folds.pkl")
        with open(open_file_path_female, 'rb') as f:
            model_trained = pickle.load(f)
            
    if gender == "male":
        # Define the csv file path to save
        open_file_path_male = os.path.join(
            open_folder_path,
            f"model_trained_male_{mri_status}_{population}_{gender}_genders_{feature_type}_{target_label}_{model_name}_{confound}_{n_repeats}_repeats_{n_folds}_folds.pkl")

        with open(open_file_path_male, 'rb') as f:
            model_trained = pickle.load(f)

    return model_trained
