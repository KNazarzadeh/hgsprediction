#!/home/knazarzadeh/miniconda3/envs/disorderspredwp3/bin/python3

import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from hgsprediction.input_arguments import parse_args
from hgsprediction.extract_features import ExtractFeatures

from ptpython.repl import embed
print("============================ Done! ============================")
embed(globals(), locals()) 

###############################################################################
# Parse the input arguments by function parse_args.
args = parse_args()
# Define the following parameters to run the code:
# motor, population name, mri status and confound removal status, 
# Type of feature, target and gender 
# Model, the number of repeats and folds for run_cross_validation
# motor is hgs or handgrip_strength(str)
motor = args.motor
# populations are: healthy, stroke or parkinson(str)
population = args.population
# MRI status: mri or nonmri(str)
mri_status = args.mri_status
# Type of features:(str)
# cognitive, cognitive+gnder
# bodysize, bodysize+gender
# bodysize+cognitive, bodysize+cognitive+gender
feature_type = args.feature_type
# Target(str): L+R(for HGS(Left +Right)), dominant_hgs or nondominant_hgs
target = args.target
# Type of genders: both (female+male), female and male
gender = args.gender
# Type of models(str): linear_svm, random forest(rf)
model = args.model
# 0 means without confound removal(int)
# 1 means with confound removal(int)
confound_status = args.confound_status
# Number of repeats for run_cross_validation(int)
n_repeats = args.repeat_number
# Number of folds for run_cross_validation(int)
n_folds = args.fold_number
###############################################################################
# Print summary of all inputs
print("================== Inputs ==================")
# print Motor type
if motor == "hgs":
    print("Motor = handgrip strength")
else:
    print("Motor =", motor)
# print Population type
print("Population =", population)
# print MRI status
print("MRI status =", mri_status)
# print Feature type
print("Feature type =", feature_type)
# print Target type
print("Target =", target)
# print Gender type
if gender == "both":
    print("Gender = both genders")
else:
    print("Gender =", gender)
# print Model type
if model == "rf":
    print("Model = random forest")
else:
    print("Model =", model)
# print Confound status 
if confound_status == 0:
    print("Confound status = Without Confound Removal")
else:
    print("Confound status = With Confound Removal")
# print Number of repeats for run_cross_validation
print("Repeat Numbers =", n_repeats)
# print Number of folds for run_cross_validation
print("Fold Numbers = ", n_folds)
print("============================================")

###############################################################################
samplesize = input("Enter the samplesize:")

if confound_status == 0:
    confound = "without_confound_removal"
else:
    confound = "with_confound_removal"

if model == "rf":
    model = "random_forest"
    model_label = "Random Forest"
if model == "svm":
    model = "linear_svm"
    model_label = "Linear SVM"

if target == "L+R":
    target_label = "L_plus_R"
    target_title = "HGS(Left+Right)"
else:
    target_label = target
    if target == "dominant_hgs":
        target_title = "Dominant HGS"
    elif target == "nondominant_hgs":
        target_title = "non-Dominant HGS"

###############################################################################
plot_folder_path = os.path.join(
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
    f"results_r2_violine_plots",       
    f"results_{gender}_genders",
    f"{target_label}_target",
    f"{model}",
    f"{confound}",
    f"{n_repeats}_repeats_{n_folds}_folds",
    f"{feature_type}",
    f"{samplesize}",
    "results_plots",
)

if(not os.path.isdir(plot_folder_path)):
    os.makedirs(plot_folder_path)

# Define the csv file path to save
plot_file_path = os.path.join(
    plot_folder_path,
    f"violineplot_{gender}_gender_{mri_status}_{population}_{feature_type}_feature_{target_label}_{model}_{confound}_{samplesize}_{n_repeats}_repeats_{n_folds}_folds")

###############################################################################
        
list_features = [
                "cognitive",
                "cognitive_gender",
                "bodysize",
                "bodysize_gender",
                "bodysize_cognitive",
                "bodysize_cognitive_gender"
]
 
concat_df = pd.DataFrame()
both_gender_df = pd.DataFrame()
for i, item in enumerate(list_features):
    df_tmp = load_r2_results(population,
                            mri_status,
                            confound,
                            gender,
                            feature_type,
                            target_label,
                            model,
                            n_repeats,
                            n_folds,
                            samplesize,)
    
    df_tmp = df_tmp.drop("Repeats", axis=1)
    for col in df_tmp.columns:
        concat_df = pd.concat([concat_df, df_tmp.loc[:, col]])
    concat_df.loc[:, "feature_type"] = item
    both_gender_df = pd.concat([both_gender_df, concat_df], axis=0)
    concat_df = pd.DataFrame()
both_gender_df = both_gender_df.rename(columns={0: "r2_score"})
both_gender_df = both_gender_df.reset_index()
        
fig = plt.figure(figsize=(50,20))
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
axes = sns.set_style("whitegrid")
axes = sns.set_palette("Set2")
sns.violinplot(data=both_gender_df, x="feature_type", y="r2_score", linewidth=4)
# set labels
plt.xlabel("Features", fontsize=40, fontweight="bold")
plt.ylabel("R2 score", fontsize=40, fontweight="bold")
plt.ylim(bottom=-0.4, top=0.8)
plt.tick_params(axis='both', labelsize=30)
plt.suptitle(f'$R^2$ scores on different features, \n Sample size(N=2,065) \n Repeats={n_repeats}, Folds={n_folds} \n Model={model_label} , Target={target_title}',
            fontsize=30, y=1, fontweight="bold")
plt.show()
plot_save_path = plot_file_path + ".png"
plt.savefig(os.path.join(plot_folder_path, plot_save_path))
plt.close()

###############################################################################
gender = "female_male"
# Define the csv file path to save
plot_file_path = os.path.join(
    plot_folder_path,
    f"violineplot_{gender}_gender_{mri_status}_{population}_{feature_type}_feature_{target_label}_{model}_{confound}_{samplesize}_{n_repeats}_repeats_{n_folds}_folds")


list_features = [
                "cognitive",
                "bodysize",
                "bodysize_cognitive",
]

concat_female_df = pd.DataFrame()
concat_male_df = pd.DataFrame()
all_female_df = pd.DataFrame()
all_male_df = pd.DataFrame()

for i, item in enumerate(list_features):
    # Female
    df_female_tmp = load_r2_results(population,
                                    mri_status,
                                    confound,
                                    gender="female",
                                    feature_type,
                                    target_label,
                                    model,
                                    n_repeats,
                                    n_folds,
                                    samplesize,)

    df_female_tmp = df_female_tmp.drop("Repeats", axis=1)
    for col in df_female_tmp.columns:
        concat_female_df = pd.concat([concat_female_df, df_female_tmp.loc[:, col]])
    concat_female_df.loc[:, "feature_type"] = item
    concat_female_df.loc[:, "gender"] = "female"
    
    # Male
    df_male_tmp = load_r2_results(population,
                                mri_status,
                                confound,
                                gender="male",
                                feature_type,
                                target_label,
                                model,
                                n_repeats,
                                n_folds,
                                samplesize,)
    
    df_male_tmp = df_male_tmp.drop("Repeats", axis=1)
    for col in df_male_tmp.columns:
        concat_male_df = pd.concat([concat_male_df, df_male_tmp.loc[:, col]])
    concat_male_df.loc[:, "feature_type"] = item
    concat_male_df.loc[:, "gender"] = "male" 
    all_female_df = pd.concat([all_female_df, concat_female_df], axis=0)
    all_male_df = pd.concat([all_male_df, concat_male_df], axis=0)
    concat_female_df = pd.DataFrame()
    concat_male_df = pd.DataFrame()
    
all_female_df = all_female_df.rename(columns={0: "r2_score"})
all_female_df = all_female_df.reset_index()
all_male_df = all_male_df.rename(columns={0: "r2_score"})
all_male_df = all_male_df.reset_index()


all_df = pd.concat([all_female_df, all_male_df], axis=0)
all_df = all_df.reset_index()

fig = plt.figure(figsize=(50,20))
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
axes = sns.set_style("whitegrid")
sns.violinplot(data=all_df, x="feature_type", y="r2_score", hue="gender", linewidth=4, 
               palette={"female": "mediumvioletred", "male": "lightskyblue"})
# set labels
plt.xlabel("Features", fontsize=40, fontweight="bold")
plt.ylabel("R2 score", fontsize=40, fontweight="bold")
plt.ylim(bottom=-0.4, top=0.8)
plt.tick_params(axis='both', labelsize=30)
plt.suptitle(f'$R^2$ scores on different features \n Female(N=1,147) - Male(N=918) \n Repeats={n_repeats}, Folds={n_folds} \n Model={model_label} , Target={target_title}',
            fontsize=30, y=1, fontweight="bold")
plt.legend(loc='upper left', fontsize=30)
plt.show()
plot_save_path = plot_file_path + ".png"
plt.savefig(os.path.join(plot_folder_path, plot_save_path))
plt.close() 

################################################################################
