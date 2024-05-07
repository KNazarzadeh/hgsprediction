
import os
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score 
from scipy.stats import pearsonr
from statsmodels.stats.multitest import fdrcorrection

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

file_path = "/data/project/stroke_ukb/knazarzadeh/GIT_repositories/motor_ukb/results/hgs_prediction/results_healthy/results_nonmri/results_both_genders/cognitive_features/L_plus_R_target/linear_svm/without_confound_removal/10_repeats_5_folds/train_set/train_set_nonmri_healthy_both_genders_cognitive_L_plus_R_linear_svm_without_confound_removal_10_repeats_5_folds.csv"

df = pd.read_csv(file_path, sep=',')

df = df.drop(columns="index")

df = df.rename(columns={"eid":"SubjectID"})

df = df.set_index("SubjectID")

df.loc[:, "hgs_left-0.0"] = df.loc[:, "46-0.0"]
df.loc[:, "hgs_right-0.0"] = df.loc[:, "47-0.0"]

df = df.rename(columns={"20016-0.0":"Fluid intelligence",
                        "20023-0.0":"Reaction time",
                        "4282-0.0":"Numeric memory",
                        "20156-0.0":"Trail making: part A",
                        "20157-0.0":"Trail making: part B",
                        "399-0.1":"Pairs matching 1: error made",
                        "399-0.2":"Pairs matching 2: error made",
                        "400-0.1":"Pairs matching 1: time",
                        "400-0.2":"Pairs matching 2: time",
                        "20018-0.0":"Prospective memory",
                        "20159-0.0":"Symbol-digit matching: corrected",
                        "20195-0.0":"Symbol-digit matching: attempted",
                        "4526-0.0":"Happiness",
                        "4559-0.0":"Satisfaction: family relationship",
                        "4537-0.0":"Satisfaction: job/work",
                        "4548-0.0":"Satisfaction: health",
                        "4570-0.0":"Satisfaction:friendship",
                        "4581-0.0":"Satisfaction: financial situation",
                        "neuroticism_score":"Neuroticism", 
                        "anxiety_score":"Anxiety symptom", 
                        "depression_score":"Depression symptom",
                        "CIDI_score":"CIDI-depression",
                        "20458-0.0":"Happiness: general",
                        "20459-0.0":"Happiness with own health",
                        "20460-0.0":"Belief that life is meaningful"})

cognitive_features = ["Fluid intelligence",
                    "Reaction time",
                    "Numeric memory",
                    "Trail making: part A",
                    "Trail making: part B",
                    "Pairs matching 1: error made",
                    "Pairs matching 2: error made",
                    "Pairs matching 1: time",
                    "Pairs matching 2: time",
                    "Prospective memory",
                    "Symbol-digit matching: corrected",
                    "Symbol-digit matching: attempted",
                    ]

depression_anxiety_features = [
                    "Neuroticism", 
                    "Anxiety symptom", 
                    "Depression symptom",
                    "CIDI-depression",
                    ]

life_satisfaction_features = [
                    "Happiness",
                    "Satisfaction: family relationship",
                    "Satisfaction: job/work",
                    "Satisfaction: health",
                    "Satisfaction:friendship",
                    "Satisfaction: financial situation",
                    ]

well_being_features = [
                    "Happiness: general",
                    "Happiness with own health",
                    "Belief that life is meaningful"
                    ]


features = cognitive_features + depression_anxiety_features + life_satisfaction_features + well_being_features

# n_significance = pd.DataFrame(columns=["hgs(L+R)", "hgs_left", "hgs_right"])
# n_significance_female = pd.DataFrame(columns=["hgs(L+R)", "hgs_left", "hgs_right"])
# n_significance_male = pd.DataFrame(columns=["hgs(L+R)", "hgs_left", "hgs_right"])
n_significance =[]
n_significance_female =[]
n_significance_male = []

for target in ["hgs(L+R)", "hgs_left", "hgs_right"]:
    x = df[f'{target}-0.0']

    # Collect all p-values and corresponding cognitive features
    df_significant = pd.DataFrame(columns=["feature_name", "p-values"])
    p_values = []

    for i, y in enumerate(features):
        _, p_value = pearsonr(x, df[y])
        p_values.append(p_value)
        df_significant.loc[i, "feature_name"]=y
        df_significant.loc[i,"p-values"]= p_value

    # Apply FDR correction to all p-values
    _, fdr_corrected_p_values = fdrcorrection(p_values)
    print("===== Done! =====")
    embed(globals(), locals())
    # Calculate significance (-log10(FDR corrected P-value)) for each corrected p-value
    significances = [-np.log10(p_val) for p_val in fdr_corrected_p_values]

    # Add FDR corrected p-values and significance to the DataFrame
    df_significant["fdr_corrected_p-values"] = fdr_corrected_p_values
    df_significant["significance"] = significances

    idx = df_significant[df_significant['feature_name'].isin(cognitive_features)].index
    df_significant.loc[idx, "cognitive_type"] = "Cognitive function"
    idx = df_significant[df_significant['feature_name'].isin(depression_anxiety_features)].index
    df_significant.loc[idx, "cognitive_type"] = "Depression/Anxiety"
    idx = df_significant[df_significant['feature_name'].isin(life_satisfaction_features)].index
    df_significant.loc[idx, "cognitive_type"] = "Life satisfaction"
    idx = df_significant[df_significant['feature_name'].isin(well_being_features)].index
    df_significant.loc[idx, "cognitive_type"] = "Subjective well-being"

    # n_significance[target] = len(df_significant[df_significant['significance']>1.3])
    n_significance.append(len(df_significant[df_significant['significance']>1.3]))
    
    df_sorted = df_significant.sort_values(by='significance', ascending=False)

    custom_paltte = ["#eb0917", "#86AD21", "#5ACACA", "#B382D6"]
    # Plot the significance values using Seaborn
    plt.figure(figsize=(20,30))
    plt.rcParams.update({"font.weight": "bold", 
                        "axes.labelweight": "bold",
                        "ytick.labelsize": 25,
                        "xtick.labelsize": 25})
    ax = sns.barplot(x='significance', y='feature_name', data=df_sorted, hue="cognitive_type", palette=custom_paltte, width=0.5)
    # Add bar labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', padding=3, fontsize=25, color='black')

    plt.xlabel('-log(p-value)', weight="bold", fontsize=30)
    plt.ylabel('')
    plt.xticks(range(0, 25, 5))

    plt.title(f'non-MRI Controls (N={len(df)})', weight="bold", fontsize=30)

    # Place legend outside the plot
    plt.legend(fontsize='20', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    plt.show()
    plt.savefig(f"both_gender_cognitive_{target}.png")
    plt.close()


###############################################################################
###############################################################################
df_female = df[df['31-0.0'] == 0]

for target in ["hgs(L+R)", "hgs_left", "hgs_right"]:
    x = df_female[f'{target}-0.0']

    # Collect all p-values and corresponding cognitive features
    df_significant_female = pd.DataFrame(columns=["feature_name", "p-values"])
    p_values = []

    for i, y in enumerate(features):
        _, p_value = pearsonr(x, df_female[y])
        p_values.append(p_value)
        df_significant_female.loc[i, "feature_name"]=y
        df_significant_female.loc[i,"p-values"]= p_value


    # Apply FDR correction to all p-values
    _, fdr_corrected_p_values = fdrcorrection(p_values)

    # Calculate significance (-log10(FDR corrected P-value)) for each corrected p-value
    significances_female = [-np.log10(p_val) for p_val in fdr_corrected_p_values]

    # Add FDR corrected p-values and significance to the DataFrame
    df_significant_female["fdr_corrected_p-values"] = fdr_corrected_p_values
    df_significant_female["significance"] = significances_female

    idx = df_significant_female[df_significant_female['feature_name'].isin(cognitive_features)].index
    df_significant_female.loc[idx, "cognitive_type"] = "Cognitive function"
    idx = df_significant_female[df_significant_female['feature_name'].isin(depression_anxiety_features)].index
    df_significant_female.loc[idx, "cognitive_type"] = "Depression/Anxiety"
    idx = df_significant_female[df_significant_female['feature_name'].isin(life_satisfaction_features)].index
    df_significant_female.loc[idx, "cognitive_type"] = "Life satisfaction"
    idx = df_significant_female[df_significant_female['feature_name'].isin(well_being_features)].index
    df_significant_female.loc[idx, "cognitive_type"] = "Subjective well-being"

    # n_significance_female[target] = len(df_significant_female[df_significant_female['significance']>1.3])
    n_significance_female.append(len(df_significant_female[df_significant_female['significance']>1.3]))

    df_sorted_female = df_significant_female.sort_values(by='significance', ascending=False)

    # Plot the significance values using Seaborn
    plt.figure(figsize=(20,30))
    plt.rcParams.update({"font.weight": "bold", 
                        "axes.labelweight": "bold",
                        "ytick.labelsize": 25,
                        "xtick.labelsize": 25})
    ax = sns.barplot(x='significance', y='feature_name', data=df_sorted_female, hue="cognitive_type", palette=custom_paltte, width=0.5)
    # Add bar labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', padding=3, fontsize=25, color='black')

    plt.xlabel('-log(p-value)', weight="bold", fontsize=30)
    plt.ylabel('')
    plt.xticks(range(0, 25, 5))

    plt.title(f'non-MRI Controls-Females(N={len(df_female)})', weight="bold", fontsize=30)

    # Place legend outside the plot
    plt.legend(fontsize='20', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    plt.show()
    plt.savefig(f"female_cognitive_{target}.png")
    plt.close()


###############################################################################
###############################################################################
df_male = df[df['31-0.0']== 1]

for target in ["hgs(L+R)", "hgs_left", "hgs_right"]:
    x = df_male[f'{target}-0.0']

    # Collect all p-values and corresponding cognitive features
    df_significant_male = pd.DataFrame(columns=["feature_name", "p-values"])
    p_values = []

    for i, y in enumerate(features):
        _, p_value = pearsonr(x, df_male[y])
        p_values.append(p_value)
        df_significant_male.loc[i, "feature_name"]=y
        df_significant_male.loc[i,"p-values"]= p_value


    # Apply FDR correction to all p-values
    _, fdr_corrected_p_values = fdrcorrection(p_values)

    # Calculate significance (-log10(FDR corrected P-value)) for each corrected p-value
    significances_male = [-np.log10(p_val) for p_val in fdr_corrected_p_values]

    # Add FDR corrected p-values and significance to the DataFrame
    df_significant_male["fdr_corrected_p-values"] = fdr_corrected_p_values
    df_significant_male["significance"] = significances_male

    idx = df_significant_male[df_significant_male['feature_name'].isin(cognitive_features)].index
    df_significant_male.loc[idx, "cognitive_type"] = "Cognitive function"
    idx = df_significant_male[df_significant_male['feature_name'].isin(depression_anxiety_features)].index
    df_significant_male.loc[idx, "cognitive_type"] = "Depression/Anxiety"
    idx = df_significant_male[df_significant_male['feature_name'].isin(life_satisfaction_features)].index
    df_significant_male.loc[idx, "cognitive_type"] = "Life satisfaction"
    idx = df_significant_male[df_significant_male['feature_name'].isin(well_being_features)].index
    df_significant_male.loc[idx, "cognitive_type"] = "Subjective well-being"

    # n_significance_male[target]  = len(df_significant_male[df_significant_male['significance']>1.3])
    n_significance_male.append(len(df_significant_male[df_significant_male['significance']>1.3]))

    df_sorted_male = df_significant_male.sort_values(by='significance', ascending=False)

    # Plot the significance values using Seaborn
    plt.figure(figsize=(20,30))
    plt.rcParams.update({"font.weight": "bold", 
                        "axes.labelweight": "bold",
                        "ytick.labelsize": 25,
                        "xtick.labelsize": 25})
    ax = sns.barplot(x='significance', y='feature_name', data=df_sorted_male, hue="cognitive_type", palette=custom_paltte, width=0.5)
    # Add bar labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', padding=3, fontsize=25, color='black')

    plt.xlabel('-log(p-value)', weight="bold", fontsize=30)
    plt.ylabel('')
    plt.xticks(range(0, 25, 5))

    plt.title(f'non-MRI Controls-Males(N={len(df_male)})', weight="bold", fontsize=30)

    # Place legend outside the plot
    plt.legend(fontsize='20', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    plt.show()
    plt.savefig(f"male_cognitive_{target}.png")
    plt.close()

###############################################################################
###############################################################################
    
print("===== Done! =====")
embed(globals(), locals())
