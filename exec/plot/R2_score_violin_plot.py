#!/home/knazarzadeh/miniconda3/envs/disorderspredwp3/bin/python3

import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from hgsprediction.input_arguments import parse_args
from hgsprediction.

from ptpython.repl import embed


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
if model == "svm":
    model = "linear_svm"

if target == "L+R":
    target_label = "L_plus_R"
else:
    target_label = target


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
    f"violinplot_{mri_status}_{population}_{gender}_genders_{feature_type}_feature_{target_label}_{model}_{confound}_{samplesize}_{n_repeats}_repeats_{n_folds}_folds")

###############################################################################
list_features = [
                "cognitive",
                "cognitive+gender",
                "bodysize",
                "bodysize+gender",
                "bodysize+cognitive",
                "bodysize+cognitive+gender"
]
 
concat_df = pd.DataFrame()
all_df = pd.DataFrame()
for i, item in enumerate(list_features):
    if "+" in item:
        item = item.replace("+", "_")
    df_tmp = load_r2_results(population,
                            mri_status,
                            confound,
                            gender,
                            item,
                            target_label,
                            model,
                            n_repeats=10,
                            n_folds=5)
    
    df_tmp = df_tmp.drop("Repeats", axis=1)
    for col in df_tmp.columns:
        concat_df = pd.concat([concat_df, df_tmp.loc[:, col]])
    concat_df.loc[:, "feature_type"] = item
    # print("============================ Done! ============================")
    # embed(globals(), locals())  
    all_df = pd.concat([all_df, concat_df], axis=0)
    concat_df = pd.DataFrame()
all_df = all_df.rename(columns={0: "r2_score"})
all_df = all_df.reset_index()
# print("============================ Done! ============================")
# embed(globals(), locals()) 
if model == "linear_svm":
    model = "Linear SVM"
elif model == "random_forest":
    model = "Random Forest"
if target == "L+R":
        target = "HGS(Left+Right)"
elif target == "dominant_hgs":
        target = "Dominant HGS"
elif target == "nondominant_hgs":
        target = "non-Dominant HGS"
        
fig = plt.figure(figsize=(50,20))
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
axes = sns.set_style("whitegrid")
axes = sns.set_palette("Set2")
sns.violinplot(data=all_df, x="feature_type", y="r2_score", linewidth=4)
# set labels
plt.xlabel("Features", fontsize=40, fontweight="bold")
plt.ylabel("R2 score", fontsize=40, fontweight="bold")
plt.ylim(bottom=-0.4, top=0.8)
plt.tick_params(axis='both', labelsize=30)
plt.suptitle(f'$R^2$ scores on different features, \n Sample size(N=2,065) \n Repeats={n_repeats}, Folds={n_folds} \n Model={model} , Target={target}',
            fontsize=30, y=1, fontweight="bold")
plt.show()
plot_save_path = plot_file_path + ".png"
plt.savefig(os.path.join(plot_folder_path, plot_save_path))
# plt.savefig(".png")
plt.close()

print("============================ Done! ============================")
embed(globals(), locals())  
###############################################################################
if confound_status == 0:
        confound = "without_confound_removal"
else:
    confound = "with_confound_removal"
if model == "rf":
    model = "random_forest"
if model == "svm":
    model = "linear_svm"

if target == "L+R":
    target_label = "L_plus_R"
else:
    target_label = target
    
plot_folder_path = os.path.join(
    "/data",
    "project",
    "stroke_ukb",
    "knazarzadeh",
    "GIT_repositories",
    "motor_ukb",
    "results",
    "hgs_prediction",
    f"results_r2_violine_plots",
    f"results_{population}",
    f"results_{mri_status}",        
    f"results_{gender}_genders",
    f"{target_label}_target",
    f"{model}",
    f"{confound}",
    f"{n_repeats}_repeats_{n_folds}_folds",
)

if(not os.path.isdir(plot_folder_path)):
    os.makedirs(plot_folder_path)

# Define the csv file path to save
plot_file_path = os.path.join(
    plot_folder_path,
    f"violineplot_gender_seperation_{mri_status}_{population}_{gender}_genders_{target_label}_{model}_{confound}_{n_repeats}_repeats_{n_folds}_folds")

if gender == "both":
    list_features = [
                    "cognitive",
                    # "cognitive+gender",
                    "bodysize",
                    # "bodysize+gender",
                    "bodysize+cognitive",
                    # "bodysize+cognitive+gender"
    ]

    concat_female_df = pd.DataFrame()
    concat_male_df = pd.DataFrame()
    all_female_df = pd.DataFrame()
    all_male_df = pd.DataFrame()

    for i, item in enumerate(list_features):
        if "+" in item:
            item = item.replace("+", "_")
        df_female_tmp, df_male_tmp = load_r2_genders_results(population,
                                                                mri_status,
                                                                confound,
                                                                gender,
                                                                item,
                                                                target_label,
                                                                model,
                                                                n_repeats=10,
                                                                n_folds=5)
        
        df_female_tmp = df_female_tmp.drop("Repeats", axis=1)
        for col in df_female_tmp.columns:
            concat_female_df = pd.concat([concat_female_df, df_female_tmp.loc[:, col]])
        concat_female_df.loc[:, "feature_type"] = item
        concat_female_df.loc[:, "gender"] = "female"
        
        # Male
        df_male_tmp = df_male_tmp.drop("Repeats", axis=1)
        for col in df_male_tmp.columns:
            concat_male_df = pd.concat([concat_male_df, df_male_tmp.loc[:, col]])
        concat_male_df.loc[:, "feature_type"] = item
        concat_male_df.loc[:, "gender"] = "male"
        # print("============================ Done! ============================")
        # embed(globals(), locals())  
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
# print("============================ Done! ============================")
# embed(globals(), locals()) 
if model == "linear_svm":
    model = "Linear SVM"
elif model =="random_forest":
    model = "Random Forest"
if target == "L+R":
    target = "HGS(Left+Right)"
elif target == "dominant_hgs":
    target = "Dominant HGS"
elif target == "nondominant_hgs":
    target = "non-Dominant HGS"

fig = plt.figure(figsize=(50,20))
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
axes = sns.set_style("whitegrid")
# axes = sns.set_palette("Set2")
sns.violinplot(data=all_df, x="feature_type", y="r2_score", hue="gender", linewidth=4, 
               palette={"female": "mediumvioletred", "male": "lightskyblue"})
# sns.violinplot(data=all_df, x="feature_type", y="r2_score", hue="gender", split=True, linewidth=4, 
#                palette={"female": "lightgreen", "male": "lightblue"})
# sns.violinplot(data=all_df, x="feature_type_male", y="r2_score", linewidth=4, color="lightblue")
# set labels
plt.xlabel("Features", fontsize=40, fontweight="bold")
plt.ylabel("R2 score", fontsize=40, fontweight="bold")
plt.ylim(bottom=-0.4, top=0.8)
plt.tick_params(axis='both', labelsize=30)
plt.suptitle(f'$R^2$ scores on different features \n Female(N=1,147) - Male(N=918) \n Repeats={n_repeats}, Folds={n_folds} \n Model={model} , Target={target}',
            fontsize=30, y=1, fontweight="bold")
plt.legend(loc='upper left', fontsize=30)
plt.show()
plot_save_path = plot_file_path + "_withoutgender_gender_seperation.png"
plt.savefig(os.path.join(plot_folder_path, plot_save_path))
plt.savefig(".png")
plt.close()

print("============================ Done! ============================")
embed(globals(), locals())  

################################################################################
