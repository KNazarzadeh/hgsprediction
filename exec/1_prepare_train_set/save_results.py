#!/usr/bin/env python3
"""Save data in different locations.
    Based on different purposes for saving in different files.
"""

import os
import numpy as np
import pandas as pd
import pickle
###############################################################################
def save_prediction_scores_results(
    dataframe,
    population,
    mri_status,
    confound_status,
    gender,
    feature_type,
    target,
    model,
    n_repeats,
    n_folds,
    output_type,
):
    """
    Save results to csv file.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame that should be save in specific folder.
    motor : str
        Name of the motor which to be analyse.
    population: str
        Name of the population which to  to be analyse
    mri_status: str
        MRI status which data to be  to be analyse.
    """
    if confound_status == 0:
        confound = "without_confound_removal"
    else:
        confound = "with_confound_removal"
    if model == "rf":
        model = "random_forest"
    if model == "svm":
        model = "linear_svm"
        
    if "+" in feature_type:
        feature_type = feature_type.replace("+", "_")
    if target == "L+R":
        target_label = "L_plus_R"
    else:
        target_label = target

    save_folder_path = os.path.join(
        "/data",
        "project",
        "stroke_ukb",
        "knazarzadeh",
        "GIT_repositories",
        "motor_ukb",
        "results",
        "hgs_prediction",
        f"results_{population}",
        f"results_{mri_status}",
        f"results_{gender}_genders",
        f"{feature_type}_features",
        f"{target_label}_target",
        f"{model}",
        f"{confound}",
        f"{n_repeats}_repeats_{n_folds}_folds",
        f"results_csv",
    )
    
    if(not os.path.isdir(save_folder_path)):
        os.makedirs(save_folder_path)

    # Define the csv file path to save
    save_file_path = os.path.join(
        save_folder_path,
        f"{output_type}_{mri_status}_{population}_{gender}_genders_{feature_type}_{target_label}_{model}_{confound}_{n_repeats}_repeats_{n_folds}_folds.csv")
    # Save the dataframe to csv file path
    dataframe.to_csv(save_file_path, sep=',', float_format='%.6f')

###############################################################################
def save_prediction_models_results(
    dataframe,
    population,
    mri_status,
    confound_status,
    gender,
    feature_type,
    target,
    model,
    n_repeats,
    n_folds,
    output_type,
):
    """
    Save results to csv file.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame that should be save in specific folder.
    motor : str
        Name of the motor which to be analyse.
    population: str
        Name of the population which to  to be analyse
    mri_status: str
        MRI status which data to be  to be analyse.
    """
    if confound_status == 0:
        confound = "without_confound_removal"
    else:
        confound = "with_confound_removal"
    if model == "rf":
        model = "random_forest"
    if model == "svm":
        model = "linear_svm"
        
    if "+" in feature_type:
        feature_type = feature_type.replace("+", "_")
    if target == "L+R":
        target_label = "L_plus_R"
    else:
        target_label = target

    save_folder_path = os.path.join(
        "/data",
        "project",
        "stroke_ukb",
        "knazarzadeh",
        "GIT_repositories",
        "motor_ukb",
        "results",
        "hgs_prediction",
        f"results_{population}",
        f"results_{mri_status}",
        f"results_{gender}_genders",
        f"{feature_type}_features",
        f"{target_label}_target",
        f"{model}",
        f"{confound}",
        f"{n_repeats}_repeats_{n_folds}_folds",
        f"results_csv",
    )
    
    if(not os.path.isdir(save_folder_path)):
        os.makedirs(save_folder_path)

    # Define the csv file path to save
    save_file_path = os.path.join(
        save_folder_path,
        f"{output_type}_{mri_status}_{population}_{gender}_genders_{feature_type}_{target_label}_{model}_{confound}_{n_repeats}_repeats_{n_folds}_folds.csv")
    # Save the dataframe to csv file path
    dataframe.to_csv(save_file_path, sep=',', index=False)
    
###############################################################################
def save_validation_ypred_results(
    dataframe,
    population,
    mri_status,
    confound_status,
    gender,
    feature_type,
    target,
    model,
    n_repeats,
    n_folds,
    output_type,
    idx,
):
    """
    Save results to csv file.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame that should be save in specific folder.
    motor : str
        Name of the motor which to be analyse.
    population: str
        Name of the population which to  to be analyse
    mri_status: str
        MRI status which data to be  to be analyse.
    """
    if confound_status == 0:
        confound = "without_confound_removal"
    else:
        confound = "with_confound_removal"
    if model == "rf":
        model = "random_forest"
    if model == "svm":
        model = "linear_svm"
        
    if "+" in feature_type:
        feature_type = feature_type.replace("+", "_")
    if target == "L+R":
        target_label = "L_plus_R"
    else:
        target_label = target

    save_folder_path = os.path.join(
        "/data",
        "project",
        "stroke_ukb",
        "knazarzadeh",
        "GIT_repositories",
        "motor_ukb",
        "results",
        "hgs_prediction",
        f"results_{population}",
        f"results_{mri_status}",
        f"results_{gender}_genders",
        f"{feature_type}_features",
        f"{target_label}_target",
        f"{model}",
        f"{confound}",
        f"{n_repeats}_repeats_{n_folds}_folds",
        f"results_csv",
        f"{output_type}",
    )
    
    if(not os.path.isdir(save_folder_path)):
        os.makedirs(save_folder_path)

    # Define the csv file path to save
    save_file_path = os.path.join(
        save_folder_path,
        "validation_ypred_df_id_"+str(idx)+".csv")
    # Save the dataframe to csv file path
    dataframe.to_csv(save_file_path, sep=',', index=False)
    
###############################################################################
def save_samesample_validation_ypred_results(
    dataframe,
    population,
    mri_status,
    confound_status,
    gender,
    feature_type,
    target,
    model,
    n_repeats,
    n_folds,
    output_type,
    samplesize,
    idx,
):
    """
    Save results to csv file.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame that should be save in specific folder.
    motor : str
        Name of the motor which to be analyse.
    population: str
        Name of the population which to  to be analyse
    mri_status: str
        MRI status which data to be  to be analyse.
    """
    if confound_status == 0:
        confound = "without_confound_removal"
    else:
        confound = "with_confound_removal"
    if model == "rf":
        model = "random_forest"
    if model == "svm":
        model = "linear_svm"
        
    if "+" in feature_type:
        feature_type = feature_type.replace("+", "_")
    if target == "L+R":
        target_label = "L_plus_R"
    else:
        target_label = target

    save_folder_path = os.path.join(
        "/data",
        "project",
        "stroke_ukb",
        "knazarzadeh",
        "GIT_repositories",
        "motor_ukb",
        "results",
        "hgs_prediction",
        f"results_{population}",
        f"results_{mri_status}",
        f"results_{gender}_genders",
        f"{feature_type}_features",
        f"{target_label}_target",
        f"{model}",
        f"{confound}",
        f"{n_repeats}_repeats_{n_folds}_folds",
        f"results_samples_{samplesize}",
        f"results_csv",
        f"{output_type}",
    )
    
    if(not os.path.isdir(save_folder_path)):
        os.makedirs(save_folder_path)

    # Define the csv file path to save
    save_file_path = os.path.join(
        save_folder_path,
        "validation_ypred_df_id_"+str(idx)+".csv")
    # Save the dataframe to csv file path
    dataframe.to_csv(save_file_path, sep=',', index=False)
    
###############################################################################
def save_features_dataframe_results(
    dataframe,
    population,
    mri_status,
    confound_status,
    gender,
    feature_type,
    target,
    model,
    n_repeats,
    n_folds,
):
    """
    Save results to csv file.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame that should be save in specific folder.
    motor : str
        Name of the motor which to be analyse.
    population: str
        Name of the population which to  to be analyse
    mri_status: str
        MRI status which data to be  to be analyse.
    """
    if confound_status == 0:
        confound = "without_confound_removal"
    else:
        confound = "with_confound_removal"
    if model == "rf":
        model = "random_forest"
    if model == "svm":
        model = "linear_svm"
        
    if "+" in feature_type:
        feature_type = feature_type.replace("+", "_")
    if target == "L+R":
        target_label = "L_plus_R"
    else:
        target_label = target

    save_folder_path = os.path.join(
        "/data",
        "project",
        "stroke_ukb",
        "knazarzadeh",
        "GIT_repositories",
        "motor_ukb",
        "results",
        "hgs_prediction",
        f"results_{population}",
        f"results_{mri_status}",
        f"results_{gender}_genders",
        f"{feature_type}_features",
        f"{target_label}_target",
        f"{model}",
        f"{confound}",
        f"{n_repeats}_repeats_{n_folds}_folds",
        f"results_same_samples_csv",
    )
    
    if(not os.path.isdir(save_folder_path)):
        os.makedirs(save_folder_path)

    # Define the csv file path to save
    save_file_path = os.path.join(
        save_folder_path,
        f"dataframe_total_{mri_status}_{population}_{gender}_genders_{feature_type}_{target_label}_{model}_{confound}_{n_repeats}_repeats_{n_folds}_folds.csv")
    # Save the dataframe to csv file path
    dataframe.to_csv(save_file_path, sep=',')
    
###############################################################################
def save_count_features_results(
    dataframe,
    gender,
    feature_type,
):
    """
    Save results to csv file.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame that should be save in specific folder.
    motor : str
        Name of the motor which to be analyse.
    population: str
        Name of the population which to  to be analyse
    mri_status: str
        MRI status which data to be  to be analyse.
    """
    # if confound_status == 0:
    #     confound = "without_confound_removal"
    # else:
    #     confound = "with_confound_removal"
    # if model == "rf":
    #     model = "random_forest"
    # if model == "svm":
    #     model = "linear_svm"
        
    if "+" in feature_type:
        feature_type = feature_type.replace("+", "_")
    # if target == "L+R":
    #     target_label = "L_plus_R"
    # else:
    #     target_label = target

    save_folder_path = os.path.join(
        "/data",
        "project",
        "stroke_ukb",
        "knazarzadeh",
        "GIT_repositories",
        "motor_ukb",
        "results",
        "hgs_prediction",
        f"{gender}",
        f"{feature_type}",
    )
    
    if(not os.path.isdir(save_folder_path)):
        os.makedirs(save_folder_path)

    # Define the csv file path to save
    save_file_path = os.path.join(
        save_folder_path,
        f"{feature_type}.csv")
    # Save the dataframe to csv file path
    dataframe.to_csv(save_file_path, sep=',')
###############################################################################
def save_same_samplesize_r2_results(
    dataframe,
    population,
    mri_status,
    confound_status,
    gender,
    feature_type,
    target,
    model,
    n_repeats,
    n_folds,
    output_type,
):
    """
    Save results to csv file.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame that should be save in specific folder.
    motor : str
        Name of the motor which to be analyse.
    population: str
        Name of the population which to  to be analyse
    mri_status: str
        MRI status which data to be  to be analyse.
    """
    if confound_status == 0:
        confound = "without_confound_removal"
    else:
        confound = "with_confound_removal"
    if model == "rf":
        model = "random_forest"
    if model == "svm":
        model = "linear_svm"
        
    if "+" in feature_type:
        feature_type = feature_type.replace("+", "_")
    if target == "L+R":
        target_label = "L_plus_R"
    else:
        target_label = target

    save_folder_path = os.path.join(
        "/data",
        "project",
        "stroke_ukb",
        "knazarzadeh",
        "GIT_repositories",
        "motor_ukb",
        "results",
        "hgs_prediction",
        f"results_{population}",
        f"results_{mri_status}",
        f"results_{gender}_genders",
        f"{feature_type}_features",
        f"{target_label}_target",
        f"{model}",
        f"{confound}",
        f"{n_repeats}_repeats_{n_folds}_folds",
        f"results_samples_samesize",
    )
    
    if(not os.path.isdir(save_folder_path)):
        os.makedirs(save_folder_path)

    # Define the csv file path to save
    save_file_path = os.path.join(
        save_folder_path,
        f"{output_type}_{mri_status}_{population}_{gender}_genders_{feature_type}_{target_label}_{model}_{confound}_{n_repeats}_repeats_{n_folds}_folds.csv")
    # Save the dataframe to csv file path
    dataframe.to_csv(save_file_path, sep=',', float_format='%.6f')


###############################################################################
def save_bodysize_samplesize_results(
    dataframe,
    population,
    mri_status,
    confound_status,
    gender,
    feature_type,
    target,
    model,
    n_repeats,
    n_folds,
    output_type,
    samplesize,
):
    """
    Save results to csv file.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame that should be save in specific folder.
    motor : str
        Name of the motor which to be analyse.
    population: str
        Name of the population which to  to be analyse
    mri_status: str
        MRI status which data to be  to be analyse.
    """
    if confound_status == 0:
        confound = "without_confound_removal"
    else:
        confound = "with_confound_removal"
    if model == "rf":
        model = "random_forest"
    if model == "svm":
        model = "linear_svm"
        
    if "+" in feature_type:
        feature_type = feature_type.replace("+", "_")
    if target == "L+R":
        target_label = "L_plus_R"
    else:
        target_label = target

    save_folder_path = os.path.join(
        "/data",
        "project",
        "stroke_ukb",
        "knazarzadeh",
        "GIT_repositories",
        "motor_ukb",
        "results",
        "hgs_prediction",
        f"results_{population}",
        f"results_{mri_status}",
        f"results_{gender}_genders",
        f"{feature_type}_features",
        f"{target_label}_target",
        f"{model}",
        f"{confound}",
        f"{n_repeats}_repeats_{n_folds}_folds",
        f"results_samples_{samplesize}_csv",
    )
    
    if(not os.path.isdir(save_folder_path)):
        os.makedirs(save_folder_path)

    # Define the csv file path to save
    save_file_path = os.path.join(
        save_folder_path,
        f"{output_type}_{samplesize}_{mri_status}_{population}_{gender}_genders_{feature_type}_{target_label}_{model}_{confound}_{n_repeats}_repeats_{n_folds}_folds.csv")
    # Save the dataframe to csv file path
    dataframe.to_csv(save_file_path, sep=',', float_format='%.6f')
    
###############################################################################
def save_data_summary(
    df,
    population,
    mri_status,
):
    """
    Save results to csv file.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame that should be save in specific folder.
    motor : str
        Name of the motor which to be analyse.
    population: str
        Name of the population which to  to be analyse
    mri_status: str
        MRI status which data to be  to be analyse.
    """
    save_folder_path = os.path.join(
        "/data",
        "project",
        "stroke_ukb",
        "knazarzadeh",
        "GIT_repositories",
        "motor_ukb",
        "data_ukb",
        "data_hgs",
        f"{population}",
        "preprocessed_data",
        "binned_data",
        f"{mri_status}",
    )
    if(not os.path.isdir(save_folder_path)):
        os.makedirs(save_folder_path)

    # Define the csv file path to save
    save_file_path = os.path.join(
        save_folder_path,
        f"data_summary_{mri_status}_{population}.csv")
    # Save the dataframe to csv file path
    df.to_csv(save_file_path, sep=',', index=False)


###############################################################################
def save_binned_df(
    df,
    population,
    mri_status,
):
    """
    Save results to csv file.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame that should be save in specific folder.
    motor : str
        Name of the motor which to be analyse.
    population: str
        Name of the population which to  to be analyse
    mri_status: str
        MRI status which data to be  to be analyse.
    """
    save_folder_path = os.path.join(
        "/data",
        "project",
        "stroke_ukb",
        "knazarzadeh",
        "GIT_repositories",
        "motor_ukb",
        "data_ukb",
        "data_hgs",
        f"{population}",
        "preprocessed_data",
        "binned_data",
        f"{mri_status}",
    )
    if(not os.path.isdir(save_folder_path)):
        os.makedirs(save_folder_path)

    # Define the csv file path to save
    save_file_path = os.path.join(
        save_folder_path,
        f"binned_data_{mri_status}_{population}.csv")
    # Save the dataframe to csv file path
    df.to_csv(save_file_path, sep=',', index=False)

    return df
    
############################################################################### 
def save_train_set_df(
    df_train,
    population,
    mri_status,
):
    """
    Save results to csv file.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame that should be save in specific folder.
    motor : str
        Name of the motor which to be analyse.
    population: str
        Name of the population which to  to be analyse
    mri_status: str
        MRI status which data to be  to be analyse.
    """
    save_folder_path = os.path.join(
        "/data",
        "project",
        "stroke_ukb",
        "knazarzadeh",
        "GIT_repositories",
        "motor_ukb",
        "data_ukb",
        "data_hgs",
        f"{population}",
        "preprocessed_data",
        "binned_data",
        f"{mri_status}",
    )
    if(not os.path.isdir(save_folder_path)):
        os.makedirs(save_folder_path)

    # Define the csv file path to save
    save_file_path = os.path.join(
        save_folder_path,
        f"train_set_{mri_status}_{population}.csv")
    # Save the dataframe to csv file path
    df_train.to_csv(save_file_path, sep=',', index=False)
        
    # Define the csv file path to save
    df_train_female = df_train[df_train['31-0.0'] == 0]
    save_file_path = os.path.join(
        save_folder_path,
        f"train_set_female_{mri_status}_{population}.csv")
    df_train_female.to_csv(save_file_path, sep=',', index=False)
    
    df_train_male = df_train[df_train['31-0.0'] == 1]
    save_file_path = os.path.join(
        save_folder_path,
        f"train_set_male_{mri_status}_{population}.csv")
    df_train_male.to_csv(save_file_path, sep=',', index=False)

############################################################################### 
def save_test_set_df(
    df_test,
    population,
    mri_status,
):
    """
    Save results to csv file.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame that should be save in specific folder.
    motor : str
        Name of the motor which to be analyse.
    population: str
        Name of the population which to  to be analyse
    mri_status: str
        MRI status which data to be  to be analyse.
    """
    save_folder_path = os.path.join(
        "/data",
        "project",
        "stroke_ukb",
        "knazarzadeh",
        "GIT_repositories",
        "motor_ukb",
        "data_ukb",
        "data_hgs",
        f"{population}",
        "preprocessed_data",
        "binned_data",
        f"{mri_status}",
    )
    
    if(not os.path.isdir(save_folder_path)):
        os.makedirs(save_folder_path)

    # Define the csv file path to save
    save_file_path = os.path.join(
        save_folder_path,
        f"test_set_{mri_status}_{population}.csv")
    # Save the dataframe to csv file path
    df_test.to_csv(save_file_path, sep=',', index=False)
    
    df_test_female = df_test[df_test['31-0.0']==0]
    save_file_path = os.path.join(
        save_folder_path,
        f"test_set_female_{mri_status}_{population}.csv")
    df_test_female.to_csv(save_file_path, sep=',', index=False)

    df_test_male = df_test[df_test['31-0.0']==1]
    save_file_path = os.path.join(
        save_folder_path,
        f"test_set_male_{mri_status}_{population}.csv")
    df_test_male.to_csv(save_file_path, sep=',', index=False)


###############################################################################
def save_preprocessed_train_df(
    df,
    population,
    mri_status,
):
    """
    Save results to csv file.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame that should be save in specific folder.
    motor : str
        Name of the motor which to be analyse.
    population: str
        Name of the population which to  to be analyse
    mri_status: str
        MRI status which data to be  to be analyse.
    """
    save_folder_path = os.path.join(
        "/data",
        "project",
        "stroke_ukb",
        "knazarzadeh",
        "GIT_repositories",
        "motor_ukb",
        "data_ukb",
        "data_hgs",
        f"{population}",
        "preprocessed_data",
        "binned_data",
        f"{mri_status}",
        "train_set",
    )
    
    if(not os.path.isdir(save_folder_path)):
        os.makedirs(save_folder_path)

    # Define the csv file path to save
    save_file_path = os.path.join(
        save_folder_path,
        f"preprocessed_train_{mri_status}_{population}.csv")
    # Save the dataframe to csv file path
    df.to_csv(save_file_path, sep=',', index=False)
    
###############################################################################
def save_train_features_target_df(
    dataframe,
    population,
    mri_status,
    confound_status,
    gender,
    feature_type,
    target,
    model,
    n_repeats,
    n_folds,
):
    """
    Save results to csv file.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame that should be save in specific folder.
    motor : str
        Name of the motor which to be analyse.
    population: str
        Name of the population which to  to be analyse
    mri_status: str
        MRI status which data to be  to be analyse.
    """
    if confound_status == 0:
        confound = "without_confound_removal"
    else:
        confound = "with_confound_removal"
    if model == "rf":
        model = "random_forest"
    if model == "svm":
        model = "linear_svm"
        
    if "+" in feature_type:
        feature_type = feature_type.replace("+", "_")
    if target == "L+R":
        target_label = "L_plus_R"
    else:
        target_label = target

    save_folder_path = os.path.join(
        "/data",
        "project",
        "stroke_ukb",
        "knazarzadeh",
        "GIT_repositories",
        "motor_ukb",
        "results",
        "hgs_prediction",
        f"results_{population}",
        f"results_{mri_status}",
        f"results_{gender}_genders",
        f"{feature_type}_features",
        f"{target_label}_target",
        f"{model}",
        f"{confound}",
        f"{n_repeats}_repeats_{n_folds}_folds",
        f"train_set",
    )
    
    if(not os.path.isdir(save_folder_path)):
        os.makedirs(save_folder_path)

    # Define the csv file path to save
    save_file_path = os.path.join(
        save_folder_path,
        f"train_set_{mri_status}_{population}_{gender}_genders_{feature_type}_{target_label}_{model}_{confound}_{n_repeats}_repeats_{n_folds}_folds.csv")
    
    # Save the dataframe to csv file path
    dataframe.to_csv(save_file_path, sep=',', index=False)
    
    
###############################################################################
def save_samesamples_train_features_target_df(
    dataframe,
    population,
    mri_status,
    confound_status,
    gender,
    feature_type,
    target,
    model,
    n_repeats,
    n_folds,
    samplesize,
):
    """
    Save results to csv file.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame that should be save in specific folder.
    motor : str
        Name of the motor which to be analyse.
    population: str
        Name of the population which to  to be analyse
    mri_status: str
        MRI status which data to be  to be analyse.
    """
    if confound_status == 0:
        confound = "without_confound_removal"
    else:
        confound = "with_confound_removal"
    if model == "rf":
        model = "random_forest"
    if model == "svm":
        model = "linear_svm"
        
    if "+" in feature_type:
        feature_type = feature_type.replace("+", "_")
    if target == "L+R":
        target_label = "L_plus_R"
    else:
        target_label = target

    save_folder_path = os.path.join(
        "/data",
        "project",
        "stroke_ukb",
        "knazarzadeh",
        "GIT_repositories",
        "motor_ukb",
        "results",
        "hgs_prediction",
        f"results_{population}",
        f"results_{mri_status}",
        f"results_{gender}_genders",
        f"{feature_type}_features",
        f"{target_label}_target",
        f"{model}",
        f"{confound}",
        f"{n_repeats}_repeats_{n_folds}_folds",
        f"results_samples_{samplesize}",
        f"train_set",
    )
    
    if(not os.path.isdir(save_folder_path)):
        os.makedirs(save_folder_path)

    # Define the csv file path to save
    save_file_path = os.path.join(
        save_folder_path,
        f"train_set_{mri_status}_{population}_{gender}_genders_{feature_type}_{target_label}_{model}_{confound}_{n_repeats}_repeats_{n_folds}_folds.csv")
    # Save the dataframe to csv file path
    dataframe.to_csv(save_file_path, sep=',', index=False)
    
###############################################################################
def save_samesample_prediction_scores_results(
    dataframe,
    population,
    mri_status,
    confound_status,
    gender,
    feature_type,
    target,
    model,
    n_repeats,
    n_folds,
    output_type,
    samplesize,
):
    """
    Save results to csv file.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame that should be save in specific folder.
    motor : str
        Name of the motor which to be analyse.
    population: str
        Name of the population which to  to be analyse
    mri_status: str
        MRI status which data to be  to be analyse.
    """
    if confound_status == 0:
        confound = "without_confound_removal"
    else:
        confound = "with_confound_removal"
    if model == "rf":
        model = "random_forest"
    if model == "svm":
        model = "linear_svm"
        
    if "+" in feature_type:
        feature_type = feature_type.replace("+", "_")
    if target == "L+R":
        target_label = "L_plus_R"
    else:
        target_label = target

    save_folder_path = os.path.join(
        "/data",
        "project",
        "stroke_ukb",
        "knazarzadeh",
        "GIT_repositories",
        "motor_ukb",
        "results",
        "hgs_prediction",
        f"results_{population}",
        f"results_{mri_status}",
        f"results_{gender}_genders",
        f"{feature_type}_features",
        f"{target_label}_target",
        f"{model}",
        f"{confound}",
        f"{n_repeats}_repeats_{n_folds}_folds",
        f"results_samples_{samplesize}",
        f"results_csv",
    )
    
    if(not os.path.isdir(save_folder_path)):
        os.makedirs(save_folder_path)

    # Define the csv file path to save
    save_file_path = os.path.join(
        save_folder_path,
        f"{output_type}_{mri_status}_{population}_{gender}_genders_{feature_type}_{target_label}_{model}_{confound}_{n_repeats}_repeats_{n_folds}_folds.csv")
    # Save the dataframe to csv file path
    dataframe.to_csv(save_file_path, sep=',', float_format='%.6f')

###############################################################################
def save_samesample_prediction_models_results(
    dataframe,
    population,
    mri_status,
    confound_status,
    gender,
    feature_type,
    target,
    model,
    n_repeats,
    n_folds,
    output_type,
    samplesize,
):
    """
    Save results to csv file.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame that should be save in specific folder.
    motor : str
        Name of the motor which to be analyse.
    population: str
        Name of the population which to  to be analyse
    mri_status: str
        MRI status which data to be  to be analyse.
    """
    if confound_status == 0:
        confound = "without_confound_removal"
    else:
        confound = "with_confound_removal"
    if model == "rf":
        model = "random_forest"
    if model == "svm":
        model = "linear_svm"
        
    if "+" in feature_type:
        feature_type = feature_type.replace("+", "_")
    if target == "L+R":
        target_label = "L_plus_R"
    else:
        target_label = target

    save_folder_path = os.path.join(
        "/data",
        "project",
        "stroke_ukb",
        "knazarzadeh",
        "GIT_repositories",
        "motor_ukb",
        "results",
        "hgs_prediction",
        f"results_{population}",
        f"results_{mri_status}",
        f"results_{gender}_genders",
        f"{feature_type}_features",
        f"{target_label}_target",
        f"{model}",
        f"{confound}",
        f"{n_repeats}_repeats_{n_folds}_folds",
        f"results_samples_{samplesize}",
        f"results_csv",
    )
    
    if(not os.path.isdir(save_folder_path)):
        os.makedirs(save_folder_path)

    # Define the csv file path to save
    save_file_path = os.path.join(
        save_folder_path,
        f"{output_type}_{mri_status}_{population}_{gender}_genders_{feature_type}_{target_label}_{model}_{confound}_{n_repeats}_repeats_{n_folds}_folds.csv")
    # Save the dataframe to csv file path
    dataframe.to_csv(save_file_path, sep=',', index=False)
    
###############################################################################
def save_model_trained(
    model_trained,
    model_trained_female,
    model_trained_male,
    population,
    mri_status,
    confound_status,
    gender,
    feature_type,
    target,
    model_name,
    n_repeats,
    n_folds,
):
    if confound_status == 0:
        confound = "without_confound_removal"
    else:
        confound = "with_confound_removal"
    if model_name == "rf":
        model_name = "random_forest"
    if model_name == "svm":
        model_name = "linear_svm"
    if "+" in feature_type:
            feature_type = feature_type.replace("+", "_")
    if target == "L+R":
        target_label = "L_plus_R"
    else:
        target_label = target
    # Assuming that you have already trained and instantiated the model as `model`
    save_folder_path = os.path.join(
            "/data",
            "project",
            "stroke_ukb",
            "knazarzadeh",
            "GIT_repositories",
            "motor_ukb",
            "results",
            "hgs_prediction",
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
        
    if(not os.path.isdir(save_folder_path)):
        os.makedirs(save_folder_path)

    # Define the csv file path to save
    save_file_path = os.path.join(
        save_folder_path,
        f"main_model_trained_{mri_status}_{population}_{gender}_genders_{feature_type}_{target_label}_{model_name}_{confound}_{n_repeats}_repeats_{n_folds}_folds.pkl")
    # Save the model to disk
    with open(save_file_path, 'wb') as f:
        pickle.dump(model_trained, f)
    # Define the csv file path to save
    save_file_path = os.path.join(
        save_folder_path,
        f"model_trained_female_{mri_status}_{population}_{gender}_genders_{feature_type}_{target_label}_{model_name}_{confound}_{n_repeats}_repeats_{n_folds}_folds.pkl")
    # Save the model to disk
    with open(save_file_path, 'wb') as f:
        pickle.dump(model_trained_female, f)
        # Define the csv file path to save
    save_file_path = os.path.join(
        save_folder_path,
        f"model_trained_male_{mri_status}_{population}_{gender}_genders_{feature_type}_{target_label}_{model_name}_{confound}_{n_repeats}_repeats_{n_folds}_folds.pkl")
    # Save the model to disk
    with open(save_file_path, 'wb') as f:
        pickle.dump(model_trained_male, f)
###############################################################################
def save_model_trained_bodysize_samplesize(
    model_trained,
    model_trained_female,
    model_trained_male,
    population,
    mri_status,
    confound_status,
    gender,
    feature_type,
    target,
    model_name,
    n_repeats,
    n_folds,
    samplesize,
):
    if confound_status == 0:
        confound = "without_confound_removal"
    else:
        confound = "with_confound_removal"
    if model_name == "rf":
        model_name = "random_forest"
    if model_name == "svm":
        model_name = "linear_svm"
    if "+" in feature_type:
            feature_type = feature_type.replace("+", "_")
    if target == "L+R":
        target_label = "L_plus_R"
    else:
        target_label = target
    # Assuming that you have already trained and instantiated the model as `model`
    save_folder_path = os.path.join(
            "/data",
            "project",
            "stroke_ukb",
            "knazarzadeh",
            "GIT_repositories",
            "motor_ukb",
            "results",
            "hgs_prediction",
            f"results_{population}",
            f"results_{mri_status}",
            f"results_{gender}_genders",
            f"{feature_type}_features",
            f"{target_label}_target",
            f"{model_name}",
            f"{confound}",
            f"{n_repeats}_repeats_{n_folds}_folds",
            f"results_samples_{samplesize}",
            f"results_csv",
            "model_trained",
        )
        
    if(not os.path.isdir(save_folder_path)):
        os.makedirs(save_folder_path)

    # Define the csv file path to save
    save_file_path = os.path.join(
        save_folder_path,
        f"main_model_trained_{mri_status}_{population}_{gender}_genders_{feature_type}_{target_label}_{model_name}_{confound}_{n_repeats}_repeats_{n_folds}_folds.pkl")
    # Save the model to disk
    with open(save_file_path, 'wb') as f:
        pickle.dump(model_trained, f)
    # Define the csv file path to save
    save_file_path = os.path.join(
        save_folder_path,
        f"model_trained_female_{mri_status}_{population}_{gender}_genders_{feature_type}_{target_label}_{model_name}_{confound}_{n_repeats}_repeats_{n_folds}_folds.pkl")
    # Save the model to disk
    with open(save_file_path, 'wb') as f:
        pickle.dump(model_trained_female, f)
        # Define the csv file path to save
    save_file_path = os.path.join(
        save_folder_path,
        f"model_trained_male_{mri_status}_{population}_{gender}_genders_{feature_type}_{target_label}_{model_name}_{confound}_{n_repeats}_repeats_{n_folds}_folds.pkl")
    # Save the model to disk
    with open(save_file_path, 'wb') as f:
        pickle.dump(model_trained_male, f)