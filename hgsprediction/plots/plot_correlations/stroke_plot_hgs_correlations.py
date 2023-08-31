
import pandas as pd
import numpy as np

from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())
# Create the actual HGS vs predicted HGS plot for females and fefemales separately
def plot_hgs_correlations(df, x, y, stroke_cohort, feature_type, target, gender):
    
    if y == "actual-predicted":
        y = "hgs_" + "(" + y + ")"
        # words = y.replace("hgs_", "").split('-')
        # capitalized_words = [word.capitalize() for word in words]
        y_label = "(Actual-Predicted)" + " HGS"
    else:
        y = "hgs_" + y
        y_label = y.replace("hgs_", "").capitalize() + " HGS"
##############################################################################
    if x == "actual-predicted":
        x = "hgs_" + "(" + x + ")"
        words = x.split('-')
        capitalized_words = [word.capitalize() for word in words]
        x_label = '-'.join(capitalized_words) + " HGS"
    elif x == "years":
        x = x
        x_label = stroke_cohort.capitalize() + " " + x.capitalize()
    elif x in ["actual", "predicted"]:
        x = "hgs_" + x
        x_label = x.replace("hgs_", "").capitalize() + " HGS"
##############################################################################
    df_female = df[df["gender"] == 0]
    df_male = df[df["gender"] == 1]
##############################################################################
    fig, ax = plt.subplots(figsize=(20,10))
    sns.set_context("poster")
    ax.set_box_aspect(1)
    if gender == "both_gender":
        sns.regplot(data=df, x=x, y=y, ax=ax, line_kws={"color": "grey"}, scatter=False)
        sns.scatterplot(data=df, x=x, y=y, hue="gender", palette=['red', 'blue'])
    elif gender == "female":
        sns.regplot(data=df, x=x, y=y, ax=ax, color="red", line_kws={"color": "grey"})
    elif gender == "male":
        sns.regplot(data=df, x=x, y=y, ax=ax, color="blue", line_kws={"color": "grey"})

    xmax = np.max(ax.get_xlim())
    ymax = np.max(ax.get_ylim())
    xmin = np.min(ax.get_xlim())
    ymin = np.min(ax.get_ylim())
    # Adjust 'labelsize' as needed
    ax.tick_params(axis='both', labelsize=20)  

    ax.set_xlabel(f"{x_label}", fontweight="bold", fontsize=20)
    ax.set_ylabel(f"{y_label}", fontweight="bold", fontsize=20)
    
    text = 'r = ' + str(format(spearmanr(df[y], df[x])[0], '.3f'))
    ax.text(xmin + 0.5 * xmin, ymax + 0.01 * ymax, text, verticalalignment='top',
            horizontalalignment='left', fontsize=20, fontweight="bold")
    
    if gender == "both_gender":
        legend = ax.legend(title="Gender", loc="lower right", fontsize='xx-small')
        # Modify individual legend labels
        new_legend_labels = ['Female', 'Male']
        for text, label in zip(legend.get_texts(), new_legend_labels):
            text.set_text(label)
    # Plot regression line
    plt.plot([xmin+1, xmax- 1], [ymin, ymax], 'k--')
    
    target = target.replace('_', ' ').upper()
    feature_type = feature_type.replace('_', ' and ').capitalize()
    
    if gender == "both_gender":
        ax.set_title(f"{y_label} vs {x_label} \n Features={feature_type}, Target={target} \n (N={len(df)})-(Females={len(df_female)}, Males={len(df_male)})", fontsize=15, fontweight="bold", y=1)
    elif gender == "female":
        ax.set_title(f"{y_label} vs {x_label} \n Features={feature_type}, Target={target} (Females={len(df)})", fontsize=15, fontweight="bold", y=1)
    
    elif gender == "male":
        ax.set_title(f"{y_label} vs {x_label} \n Features={feature_type}, Target={target} (Males={len(df)})", fontsize=15, fontweight="bold", y=1)
        
    return plt