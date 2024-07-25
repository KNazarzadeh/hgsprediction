
import sys
import os
import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
#--------------------------------------------------------------------------#
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################
# Parse, add and return the arguments by function parse_args.
###############################################################################
filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
feature_type = sys.argv[3]
target = sys.argv[4]
model_name = sys.argv[5]
confound_status = sys.argv[6]
n_repeats = sys.argv[7]
n_folds = sys.argv[8]
data_set = sys.argv[9]
gender = sys.argv[10]   

###############################################################################
for session in ["0", "1"]:
    folder_path = os.path.join(
        "/data",
        "project",
        "stroke_ukb",
        "knazarzadeh",
        "project_hgsprediction",
        "results_hgsprediction",
        "test_retest_reliability",
        f"{population}",
        f"{mri_status}",
        f"{feature_type}",
        f"{target}",
        f"longitudinal_session_ukb",
        f"{session}_session_ukb",
        "extracted_data_by_feature_and_target",
        )

    if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)

    # Define the csv file path to save
    file_path = os.path.join(
    folder_path,
    f"{gender}_extracted_data_by_feature_and_target.csv")

    # Save the dataframe to csv file path
    if session == "0":
        df_session_0 = pd.read_csv(file_path, sep=',', index_col=0)
    elif session == "1":
        # Save the dataframe to csv file path
        df_session_1 = pd.read_csv(file_path, sep=',', index_col=0)


###############################################################################
# Function to calculate the Concordance correlation coefficient (CCC):
# CCC = 2⋅Cov(X,Y)/​Var(X)+Var(Y)+(Xˉ−Yˉ)^2
def concordance_correlation_coefficient(test_session_0, retest_session_1):
    """Concordance correlation coefficient."""
    # Raw data
    dct = {
        'test_session_0': test_session_0,
        'retest_session_1': retest_session_1
    }
    df = pd.DataFrame(dct)
    # Remove NaNs
    df = df.dropna()
    # Pearson product-moment correlation coefficients
    test_session_0 = df['test_session_0']
    retest_session_1 = df['retest_session_1']
    cor = np.corrcoef(test_session_0, retest_session_1)[0][1]
    # Means
    mean_true = np.mean(test_session_0)
    mean_pred = np.mean(retest_session_1)
    # Population variances
    var_true = np.var(test_session_0)
    var_pred = np.var(retest_session_1)
    # Population standard deviations
    sd_true = np.std(test_session_0)
    sd_pred = np.std(retest_session_1)
    # Calculate CCC
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred)**2

    ccc = numerator / denominator
    
    return ccc

###############################################################################
print("\ngender: ", gender)
print("----------------------------------------------------------")
# Calculate CCC between corrected delta hgs and raw hgs for each session 0
test_session_0 = df_session_0[f"{target}_predicted"]
retest_session_1 = df_session_1[f"{target}_predicted"]

ccc_value = concordance_correlation_coefficient(test_session_0, retest_session_1)
print(f"Concordance Correlation Coefficient for predicted values without bias-adjustment: {ccc_value:.2f}")
###############################################################################
# Calculate CCC between corrected delta hgs and raw hgs for each session 0
test_session_0 = df_session_0[f"{target}_corrected_predicted"]
retest_session_1 = df_session_1[f"{target}_corrected_predicted"]

ccc_value = concordance_correlation_coefficient(test_session_0, retest_session_1)
print(f"Concordance Correlation Coefficient for predicted values with bias-adjustment: {ccc_value:.2f}")
###############################################################################
# Calculate CCC between corrected delta hgs and raw hgs for each session 0
test_session_0 = df_session_0[f"{target}_delta(true-predicted)"]
retest_session_1 = df_session_1[f"{target}_delta(true-predicted)"]

ccc_value = concordance_correlation_coefficient(test_session_0, retest_session_1)
print(f"\nConcordance Correlation Coefficient for delta values without bias-adjustment: {ccc_value:.2f}")
###############################################################################
# Calculate CCC between corrected delta hgs and raw hgs for each session 0
test_session_0 = df_session_0[f"{target}_corrected_delta(true-predicted)"]
retest_session_1 = df_session_1[f"{target}_corrected_delta(true-predicted)"]

ccc_value = concordance_correlation_coefficient(test_session_0, retest_session_1)
print(f"Concordance Correlation Coefficient for delta values with bias-adjustment: {ccc_value:.2f}")
###############################################################################
age_differ = df_session_1.loc[:, "age"] - df_session_0.loc[:, "age"]
min_age = min(age_differ)
max_age = max(age_differ)

###############################################################################
print("----------------------------------------------------------")

y_true_session_0 = df_session_0[f"{target}"]
y_pred_session_0 = df_session_0[f"{target}_predicted"]
mae_ses_0_predicted = mean_absolute_error(y_true_session_0, y_pred_session_0)
print(f"MAE for predicted values of session 0 without bias-adjustment: {mae_ses_0_predicted:.3f}")

y_true_session_1 = df_session_1[f"{target}"]
y_pred_session_1 = df_session_1[f"{target}_predicted"]
mae_ses_1_predicted = mean_absolute_error(y_true_session_1, y_pred_session_1)
print(f"MAE for predicted values of session 1 without bias-adjustment: {mae_ses_1_predicted:.3f}")

y_true_session_0 = df_session_0[f"{target}"]
y_pred_session_0 = df_session_0[f"{target}_corrected_predicted"]
mae_ses_0_corrected_predicted = mean_absolute_error(y_true_session_0, y_pred_session_0)
print(f"\nMAE for predicted values of session 0 with bias-adjustment: {mae_ses_0_corrected_predicted:.3f}")

y_true_session_1 = df_session_1[f"{target}"]
y_pred_session_1 = df_session_1[f"{target}_corrected_predicted"]
mae_ses_1_corrected_predicted = mean_absolute_error(y_true_session_1, y_pred_session_1)
print(f"MAE for predicted values of session 1 with bias-adjustment: {mae_ses_1_corrected_predicted:.3f}")

print("===== Done! =====")
embed(globals(), locals())

###############################################################################
palette_male = sns.color_palette("Blues")
palette_female = sns.color_palette("Reds")
color_female = palette_female[5]
color_male = palette_male[5]

# palette_female = sns.cubehelix_palette()
# color_female = palette_female[1]
# color_male = palette_male[2]
# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################

sns.set_style("white")

fig, ax = plt.subplots(1, 2, figsize=(10, 10))

# Plot female and male data for target vs. delta (true-predicted) in the third subplot
sns.regplot(data=df_male, x=f"age", y=f"{target}_delta(true-predicted)", color=color_male, marker="o", scatter_kws={'s': 50,'edgecolor': 'black'}, line_kws={"color": color_male}, ax=ax[0])
sns.regplot(data=df_female, x=f"age", y=f"{target}_delta(true-predicted)", color=color_female, marker="o", scatter_kws={'s': 50,'edgecolor': 'black'}, line_kws={"color": color_female}, ax=ax[0])

# Plot female and male data for target vs. corrected delta (true-predicted) in the fourth subplot
sns.regplot(data=df_male, x=f"age", y=f"{target}_corrected_delta(true-predicted)", color=color_male, marker="o", scatter_kws={'s': 50,'edgecolor': 'black'}, line_kws={"color": color_male}, ax=ax[1])
sns.regplot(data=df_female, x=f"age", y=f"{target}_corrected_delta(true-predicted)", color=color_female, marker="o", scatter_kws={'s': 50,'edgecolor': 'black'}, line_kws={"color": color_female}, ax=ax[1])

#-----------------------------------------------------------#
ax[0].set_ylabel("Delta HGS", fontsize=16)
ax[1].set_ylabel("Delta adjusted HGS", fontsize=16)

ax[0].set_xlabel("Age", fontsize=16) 
ax[1].set_xlabel("Age", fontsize=16) 

#-----------------------------------------------------------#                   
ax[0].set_title(f"Without bias-adjustment", fontsize=16, fontweight="bold")            
ax[1].set_title(f"With bias-adjustment", fontsize=16, fontweight="bold")

#-----------------------------------------------------------#
ax[0].set_box_aspect(1)
ax[1].set_box_aspect(1)
#-----------------------------------------------------------#
# Iterate over each subplot to change the font size for tick labels
for axis in ax.flatten():
    axis.tick_params(axis='both', labelsize=12, direction='out', length=5)
#-----------------------------------------------------------#
# Get x and y limits for the first subplot first row
xmin00, xmax00 = ax[0].get_xlim()
ymin00, ymax00 = ax[0].get_ylim()
# Get x and y limits for the second subplot first row
xmin01, xmax01 = ax[1].get_xlim()
ymin01, ymax01 = ax[1].get_ylim()
# Find the common y-axis limits
ymin_0 = min(ymin00, ymin01)
ymax_0 = max(ymax00, ymax01)
# Set the y-ticks step value
ystep0_value = 20
# Calculate the range for y-ticks
yticks0_range = np.arange(math.floor(ymin_0 / 10) * 10, math.ceil(ymax_0 / 10) * 10 + 10, ystep0_value)
# Set the y-ticks for both subplots
ax[0].set_yticks(yticks0_range)
ax[1].set_yticks(yticks0_range)
#-----------------------------------------------------------#
# Get x and y limits for the first subplot first row
xmin00, xmax00 = ax[0].get_xlim()
ymin00, ymax00 = ax[0].get_ylim()
# Get x and y limits for the second subplot first row
xmin01, xmax01 = ax[1].get_xlim()
ymin01, ymax01 = ax[1].get_ylim()
#-----------------------------------------------------------#
# Plot a dark grey dashed line with width 3 from (xmin00, ymin00) to (xmax00, ymax00) on the first subplot
ax[0].plot([xmin00, xmax00], [ymin00, ymax00], color='darkgrey', linestyle='--', linewidth=3)
# Plot a dark grey dashed line with width 3 from (xmin01, ymin01) to (xmax01, ymax01) on the second subplot
ax[1].plot([xmin01, xmax01], [ymin01, ymax01], color='darkgrey', linestyle='--', linewidth=3)
#-----------------------------------------------------------#
plt.tight_layout()
#-----------------------------------------------------------#
plt.show()
plt.savefig(plot_file)
plt.close()

print("===== Done! =====")
embed(globals(), locals())

x_female=df_female["age"]
y_female=df_female[f"{target}_delta(true-predicted)"]

female_corr_delta = pearsonr(y_female, x_female)
#-----------------------------------------------------------#
x_female=df_female["age"]
y_female=df_female[f"{target}_corrected_delta(true-predicted)"]

female_corr_adjusted_delta = pearsonr(y_female, x_female)
#-----------------------------------------------------------#
x_male=df_male["age"]
y_male=df_male[f"{target}_delta(true-predicted)"]

male_corr_delta = pearsonr(y_male, x_male)

x_male=df_male["age"]
y_male=df_male[f"{target}_corrected_delta(true-predicted)"]

male_corr_adjusted_delta = pearsonr(y_male, x_male)