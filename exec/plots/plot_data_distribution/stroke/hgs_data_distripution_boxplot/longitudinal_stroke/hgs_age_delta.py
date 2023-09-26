import sys
import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from hgsprediction.load_data import stroke_load_data
from hgsprediction.load_results import stroke
import statsmodels.api as sm
from scipy.stats import spearmanr

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
model_name = sys.argv[3]
feature_type = sys.argv[4]
target = sys.argv[5]

stroke_cohort = "longitudinal-stroke"
session_column = f"1st_{stroke_cohort}_session"
df_longitudinal = stroke.load_hgs_predicted_results(population, mri_status, session_column, model_name, feature_type, target, "both_gender")

df_longitudinal[f"delta_post-pre_{target}_actual"] = df_longitudinal[f"1st_post-stroke_{target}_actual"]-df_longitudinal[f"1st_pre-stroke_{target}_actual"]
df_longitudinal[f"delta_post-pre_{target}_predicted"] = df_longitudinal[f"1st_post-stroke_{target}_predicted"]-df_longitudinal[f"1st_pre-stroke_{target}_predicted"]
df_longitudinal[f"delta_post-pre_{target}_(actual-predicted)"] = df_longitudinal[f"delta_post-pre_{target}_actual"]-df_longitudinal[f"delta_post-pre_{target}_predicted"]


df_female = df_longitudinal[df_longitudinal["gender"]==0]
df_male = df_longitudinal[df_longitudinal["gender"]==1]
print("===== Done! =====")
embed(globals(), locals())
if target == "hgs_L+R":
    target_string = target.replace("L", "Left").replace("R", "Right")
    target_string = target_string.replace("hgs_", "")
    target_string = "HGS(" + target_string + ")"
elif target in ["hgs_left", "hgs_right"]:
    target_string = target.replace("hgs_", "")
    target_string = target_string.capitalize()
    target_string = "HGS(" + target_string + ")"

# Create a figure and axis
custom_palette = sns.color_palette(['#800080', '#000080'])  # You can use any hex color codes you prefer
x_values = ["age", "bmi", "height", "waist_to_hip_ratio"]
for xval in x_values:
    fig, ax = plt.subplots(1,2, figsize=(15,10))
    ax[0].set_box_aspect(1)
    ax[1].set_box_aspect(1)
    sns.regplot(data=df_female, x=f"1st_pre-stroke_{xval}", y=f"delta_post-pre_{target}_(actual-predicted)", color="#800080", label="Female", ax=ax[0])
    sns.regplot(data=df_male, x=f"1st_pre-stroke_{xval}", y=f"delta_post-pre_{target}_(actual-predicted)", color='#000080', label="Male", ax=ax[0])
    sns.regplot(data=df_female, x=f"1st_post-stroke_{xval}", y=f"delta_post-pre_{target}_(actual-predicted)", color="#800080", label="Female", ax=ax[1])
    sns.regplot(data=df_male, x=f"1st_post-stroke_{xval}", y=f"delta_post-pre_{target}_(actual-predicted)", color="#000080", label="Male", ax=ax[1])
    
    # Set labels and title
    ax[0].set_xlabel(f"{xval.capitalize()} Pre-stroke", fontsize=20, fontweight="bold")
    ax[0].set_ylabel(f"delta actual-delta predicted {target_string}", fontsize=20, fontweight="bold")
    ax[1].set_xlabel(f"{xval.capitalize()} Post-stroke", fontsize=20, fontweight="bold")
    ax[1].set_ylabel("")

    ymin = min(ax[0].get_ylim()[0], ax[1].get_ylim()[0])
    ymax = max(ax[0].get_ylim()[1], ax[1].get_ylim()[1])
    xmin = min(ax[0].get_xlim()[0], ax[1].get_xlim()[0])
    xmax = max(ax[0].get_xlim()[1], ax[1].get_xlim()[1])
    
    if xval == "waist_to_hip_ratio":
        ax[0].set_xticks(np.arange(round(xmin/.05) * .05, (round(xmax/.05) * .05)+ .05, .05))
        ax[1].set_xticks(np.arange(round(xmin/.05) * .05, (round(xmax/.05) * .05)+ .05, .05))
        ax[0].set_yticks(np.arange(round(ymin/5) * 5, (round(ymax/5) * 5)+ 5, 5))
        ax[1].set_yticks(np.arange(round(ymin/5) * 5, (round(ymax/5) * 5)+ 5, 5))  
    else:
        ax[0].set_xticks(np.arange(round(xmin/5) * 5, (round(xmax/5) * 5)+ 5, 5))
        ax[1].set_xticks(np.arange(round(xmin/5) * 5, (round(xmax/5) * 5)+ 5, 5))
        ax[0].set_yticks(np.arange(round(ymin/5) * 5, (round(ymax/5) * 5)+ 5, 5))
        ax[1].set_yticks(np.arange(round(ymin/5) * 5, (round(ymax/5) * 5)+ 5, 5))   

    ax[0].tick_params(axis='y', labelsize=14 )  # Adjust the font size (12 in this example)
    ax[1].tick_params(axis='y', labelsize=14)  # Adjust the font size (12 in this example)

    ax[0].tick_params(axis='x', labelsize=14)  # Adjust the font size (12 in this example)
    ax[1].tick_params(axis='x', labelsize=14)  # Adjust the font size (12 in this example)

    fig.suptitle(f"Correlation between (delta actual-delta predicted) vs {xval.capitalize()}\nTarget={target_string}", fontsize=20, fontweight="bold")
    # Add a text box with inside text at coordinates (x, y)
    if target == "hgs_L+R":
        y_position = 6
    else:
        y_position = 3

    if xval == "waist_to_hip_ratio":
        x_text = max(ax[0].get_xticks()) - .07 # X-coordinate
        y_text = max(ax[0].get_yticks()) - y_position  # Y-coordinate
    elif xval == "bmi":
        x_text = max(ax[0].get_xticks()) - 5 # X-coordinate
        y_text = max(ax[0].get_yticks()) - y_position  # Y-coordinate
    else:
        x_text = max(ax[0].get_xticks()) - 10 # X-coordinate
        y_text = max(ax[0].get_yticks()) - y_position  # Y-coordinate
    text0 = 'r_female= ' + str(format(spearmanr(df_female[f"delta_post-pre_{target}_(actual-predicted)"], df_female[f"1st_pre-stroke_{xval}"])[0], '.3f')) + "\n" + \
            'r_male= ' + str(format(spearmanr(df_male[f"delta_post-pre_{target}_(actual-predicted)"], df_male[f"1st_pre-stroke_{xval}"])[0], '.3f'))
    text1 = 'r_female= ' + str(format(spearmanr(df_female[f"delta_post-pre_{target}_(actual-predicted)"], df_female[f"1st_post-stroke_{xval}"])[0], '.3f')) + "\n" + \
            'r_male= ' + str(format(spearmanr(df_male[f"delta_post-pre_{target}_(actual-predicted)"], df_male[f"1st_post-stroke_{xval}"])[0], '.3f'))

    ax[0].text(x_text, y_text, text0, bbox=dict(facecolor='white', alpha=.7))
    ax[1].text(x_text, y_text, text1, bbox=dict(facecolor='white', alpha=.7))        
    
    # Show the legend outside the axes
    legend = plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # Show the plot
    plt.show()
    plt.savefig(f"delta_hgs_{target}_{xval}.png")
print("===== Done! =====")
embed(globals(), locals())

# def calculate_quadrilateral_area(x1, y1, x2, y2, x3, y3, x4, y4):
#     area = 0.5 * abs(x1*y2 + x2*y3 + x3*y4 + x4*y1 - y1*x2 - y2*x3 - y3*x4 - y4*x1)
#     return area

