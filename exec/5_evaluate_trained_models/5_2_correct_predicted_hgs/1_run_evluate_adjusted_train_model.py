import os
import pandas as pd
import numpy as np
import sys

from hgsprediction.load_results import load_trained_models
from hgsprediction.load_data import healthy_load_data
from hgsprediction.extract_data import healthy_extract_data
from hgsprediction.define_features import define_features
from hgsprediction.predict_hgs import predict_hgs
from hgsprediction.predict_hgs import calculate_pearson_hgs_correlation
from hgsprediction.save_results.healthy import save_correlation_results, \
                                               save_hgs_predicted_results
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################
filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
feature_type = sys.argv[3]
target = sys.argv[4]
model_name = sys.argv[5]
session = sys.argv[6]
confound_status = sys.argv[7]
n_repeats = sys.argv[8]
n_folds = sys.argv[9]

###############################################################################
female_best_model_trained = load_trained_models.load_best_model_trained(
                                "healthy",
                                "nonmri",
                                int(confound_status),
                                "female",
                                feature_type,
                                target,
                                "linear_svm",
                                n_repeats,
                                n_folds,
                            )

print(female_best_model_trained)

male_best_model_trained = load_trained_models.load_best_model_trained(
                                "healthy",
                                "nonmri",
                                int(confound_status),
                                "male",
                                feature_type,
                                target,
                                "linear_svm",
                                n_repeats,
                                n_folds,
                            )
print(male_best_model_trained)

##############################################################################
# load data
df = healthy_load_data.load_preprocessed_data(population, mri_status, session, "both_gender")

features, extend_features = define_features(feature_type)

data_extracted = healthy_extract_data.extract_data(df, features, extend_features, feature_type, target, mri_status, session)

X = features
y = target

df_female = data_extracted[data_extracted["gender"] == 0]
df_male = data_extracted[data_extracted["gender"] == 1]

df_female = predict_hgs(df_female, X, y, female_best_model_trained, target)
df_male = predict_hgs(df_male, X, y, male_best_model_trained, target)

print(df_female)
print(df_male)

df_both_gender = pd.concat([df_female, df_male], axis=0)
print(df_both_gender)

def adjust_prediction(df, target):
    df_corrected = pd.DataFrame()
    df_correlations = pd.DataFrame(columns=["r_values_true_raw_predicted", "r2_values_true_raw_predicted",
                                            "r_values_true_raw_delta", "r2_values_true_raw_delta",
                                            "r_values_true_corrected_predicted", "r2_values_true_corrected_predicted",
                                            "r_values_true_corrected_delta", "r2_values_true_corrected_delta"])

    model = LinearRegression()
    model.fit(df.loc[:, f"{target}"].values.reshape(-1, 1), df.loc[:, f"{target}_predicted"])
    slope = model.coef_[0]
    intercept = model.intercept_
    df.loc[:, "corrected_predicted_hgs"] = (df.loc[:, f"{target}_predicted"] - intercept) / slope
    df.loc[:, "corrected_delta_hgs"] =  df.loc[:, f"{target}"] - df.loc[:, "corrected_predicted_hgs"]
    
    r_values_true_raw_predicted = pearsonr(df.loc[:, f"{target}"],df.loc[:,f"{target}_predicted"])[0]
    r2_values_true_raw_predicted = r2_score(df.loc[:, f"{target}"],df.loc[:,f"{target}_predicted"])

    r_values_true_raw_delta = pearsonr(df.loc[:, f"{target}"],df.loc[:,f"{target}_delta(true-predicted)"])[0]
    r2_values_true_raw_delta = r2_score(df.loc[:, f"{target}"],df.loc[:,f"{target}_delta(true-predicted)"])

    r_values_true_corrected_predicted = pearsonr(df.loc[:, f"{target}"],df.loc[:,"corrected_predicted_hgs"])[0]
    r2_values_true_corrected_predicted = r2_score(df.loc[:, f"{target}"],df.loc[:,"corrected_predicted_hgs"])

    r_values_true_corrected_delta = pearsonr(df.loc[:, f"{target}"],df.loc[:,"corrected_delta_hgs"])[0]
    r2_values_true_corrected_delta = r2_score(df.loc[:, f"{target}"],df.loc[:,"corrected_delta_hgs"])

    df_correlations.loc[0, "r_values_true_raw_predicted"] = r_values_true_raw_predicted
    df_correlations.loc[0, "r2_values_true_raw_predicted"] = r2_values_true_raw_predicted
    df_correlations.loc[0, "r_values_true_raw_delta"] = r_values_true_raw_delta
    df_correlations.loc[0, "r2_values_true_raw_delta"] = r2_values_true_raw_delta
    df_correlations.loc[0, "r_values_true_corrected_predicted"] = r_values_true_corrected_predicted
    df_correlations.loc[0, "r2_values_true_corrected_predicted"] = r2_values_true_corrected_predicted
    df_correlations.loc[0, "r_values_true_corrected_delta"] = r_values_true_corrected_delta
    df_correlations.loc[0, "r2_values_true_corrected_delta"] = r2_values_true_corrected_delta

    df_corrected = pd.concat([df_corrected, df], axis=0)

    # df_correlations = df_correlations.set_index("cv_fold")
    return df_corrected, df_correlations

###############################################################################

df_female_corrected, df_female_correlations = adjust_prediction(df_female, target)
df_male_corrected, df_male_correlations = adjust_prediction(df_male, target)



print("===== Done! =====")
embed(globals(), locals())
###############################################################################
# Predicted vs True HGS
# Raw predicted HGS vs True HGS
# Corrected predicted HGS vs True HGS
fig, axes = plt.subplots(2, 1, figsize=(25, 25))

plt.rcParams.update({"font.weight": "bold", 
                     "axes.labelweight": "bold",
                     "ytick.labelsize": 40,
                     "xtick.labelsize": 40,
                     })

sns.set_style("whitegrid", {'axes.grid' : False})

for i in range(2):
    ax = axes[i]
    
    if i == 0:
        sns.regplot(data=df_female_corrected, x=f"{target}", y=f"{target}_predicted", color='lightcoral', marker="$\circ$", scatter_kws={'s': 50, 'linewidths': 12}, line_kws={"color": "red"}, ax=ax)
        sns.regplot(data=df_male_corrected, x=f"{target}", y=f"{target}_predicted", color='#069AF3', marker="$\circ$", scatter_kws={'s': 50, 'linewidths': 12}, line_kws={"color": "blue"}, ax=ax)
        ax.set_ylabel("Raw predicted HGS", fontsize=40, fontweight="bold")
        ax.set_xlabel("")            
        # ax.set_title(f"Fold:{fold}", fontsize=40, fontweight="bold")            
        
        r_text_female = f"r:{df_female_correlations.loc[0, 'r_values_true_raw_predicted']:.3f}\nR2:{df_female_correlations.loc[0, 'r2_values_true_raw_predicted']:.3f}"
        r_text_male = f"r:{df_male_correlations.loc[0, 'r_values_true_raw_predicted']:.3f}\nR2:{df_male_correlations.loc[0, 'r2_values_true_raw_predicted']:.3f}"
        ax.annotate(r_text_female, xy=(0.05, 0.9), xycoords='axes fraction', fontsize=30, fontweight="bold", color='red')
        ax.annotate(r_text_male, xy=(0.05, 0.8), xycoords='axes fraction', fontsize=30, fontweight="bold", color='#069AF3')
    elif i == 1:
        sns.regplot(data=df_female_corrected, x=f"{target}", y=f"corrected_predicted_hgs", color='lightcoral', marker="$\circ$", scatter_kws={'s': 50, 'linewidths': 5}, line_kws={"color": "red"}, ax=ax)
        sns.regplot(data=df_male_corrected, x=f"{target}", y=f"corrected_predicted_hgs", color='#069AF3', marker="$\circ$", scatter_kws={'s': 50, 'linewidths': 5}, line_kws={"color": "blue"}, ax=ax)
        ax.set_ylabel("Corrected predicted HGS", fontsize=40, fontweight="bold")
        ax.set_xlabel("True HGS", fontsize=40, fontweight="bold")

        r_text_female = f"r:{df_female_correlations.loc[0, 'r_values_true_corrected_predicted']:.3f}\nR2:{df_female_correlations.loc[0, 'r2_values_true_corrected_predicted']:.3f}"
        r_text_male = f"r:{df_male_correlations.loc[0, 'r_values_true_corrected_predicted']:.3f}\nR2:{df_male_correlations.loc[0, 'r2_values_true_corrected_predicted']:.3f}"
        ax.annotate(r_text_female, xy=(0.05, 0.9), xycoords='axes fraction', fontsize=30, fontweight="bold", color='red')
        ax.annotate(r_text_male, xy=(0.05, 0.8), xycoords='axes fraction', fontsize=30, fontweight="bold", color='#069AF3')
    
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    ax.plot([xmin, xmax], [ymin, ymax], color='darkgrey', linestyle='--', linewidth=12)

plt.tight_layout()
plt.show()
plt.savefig(f"true_predicted.png")
plt.close()

###############################################################################
# Delta vs True HGS
# Raw delta HGS vs True HGS
# Corrected delta HGS vs True HGS
fig, axes = plt.subplots(2, 1, figsize=(25, 25))

plt.rcParams.update({"font.weight": "bold", 
                     "axes.labelweight": "bold",
                     "ytick.labelsize": 40,
                     "xtick.labelsize": 40,
                     })

sns.set_style("whitegrid", {'axes.grid' : False})

for i in range(2):
    ax = axes[i]
    
    if i == 0:
        sns.regplot(data=df_female_corrected, x=f"{target}", y=f"{target}_delta(true-predicted)", color='lightcoral', marker="$\circ$", scatter_kws={'s': 50, 'linewidths': 12}, line_kws={"color": "red"}, ax=ax)
        sns.regplot(data=df_male_corrected, x=f"{target}", y=f"{target}_delta(true-predicted)", color='#069AF3', marker="$\circ$", scatter_kws={'s': 50, 'linewidths': 12}, line_kws={"color": "blue"}, ax=ax)
        ax.set_ylabel("Raw delta HGS", fontsize=40, fontweight="bold")
        ax.set_xlabel("")                        
        
        r_text_female = f"r:{df_female_correlations.loc[0, 'r_values_true_raw_delta']:.3f}\nR2:{df_female_correlations.loc[0, 'r2_values_true_raw_delta']:.3f}"
        r_text_male = f"r:{df_male_correlations.loc[0, 'r_values_true_raw_delta']:.3f}\nR2:{df_male_correlations.loc[0, 'r2_values_true_raw_delta']:.3f}"
        ax.annotate(r_text_female, xy=(0.05, 0.9), xycoords='axes fraction', fontsize=30, fontweight="bold", color='red')
        ax.annotate(r_text_male, xy=(0.05, 0.8), xycoords='axes fraction', fontsize=30, fontweight="bold", color='#069AF3')
    elif i == 1:
        sns.regplot(data=df_female_corrected, x=f"{target}", y=f"corrected_delta_hgs", color='lightcoral', marker="$\circ$", scatter_kws={'s': 50, 'linewidths': 5}, line_kws={"color": "red"}, ax=ax)
        sns.regplot(data=df_male_corrected, x=f"{target}", y=f"corrected_delta_hgs", color='#069AF3', marker="$\circ$", scatter_kws={'s': 50, 'linewidths': 5}, line_kws={"color": "blue"}, ax=ax)
        ax.set_ylabel("Corrected delta HGS", fontsize=40, fontweight="bold")
        ax.set_xlabel("True HGS", fontsize=40, fontweight="bold")

        r_text_female = f"r:{df_female_correlations.loc[0, 'r_values_true_corrected_delta']:.3f}\nR2:{df_female_correlations.loc[0, 'r2_values_true_corrected_delta']:.3f}"
        r_text_male = f"r:{df_male_correlations.loc[0, 'r_values_true_corrected_delta']:.3f}\nR2:{df_male_correlations.loc[0, 'r2_values_true_corrected_delta']:.3f}"
        ax.annotate(r_text_female, xy=(0.05, 0.9), xycoords='axes fraction', fontsize=30, fontweight="bold", color='red')
        ax.annotate(r_text_male, xy=(0.05, 0.8), xycoords='axes fraction', fontsize=30, fontweight="bold", color='#069AF3')
    
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    ax.plot([xmin, xmax], [ymin, ymax], color='darkgrey', linestyle='--', linewidth=12)

plt.tight_layout()
plt.show()
plt.savefig(f"true_delta.png")
plt.close()