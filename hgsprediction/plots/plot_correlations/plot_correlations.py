
import pandas as pd
import numpy as np

from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# Create the actual HGS vs predicted HGS plot for females and fefemales separately
def plot_hgs_correlations(data, x, y, x_label, y_label, target, gender):
    
    fig, ax = plt.subplots(figsize=(20,10))
    sns.set_context("poster")
    ax.set_box_aspect(1)
    if gender == "both_gender":
        sns.regplot(data=data, x=x, y=y, ax=ax, line_kws={"color": "grey"}, scatter=False)
        sns.scatterplot(data=data, x=x, y=y, hue="gender", palette=['red', 'blue'])
    elif gender == "female":
        sns.regplot(data=data, x=x, y=y, ax=ax, color="red", line_kws={"color": "grey"})
    elif gender == "male":
        sns.regplot(data=data, x=x, y=y, ax=ax, color="blue", line_kws={"color": "grey"})

    ax.tick_params(axis='both')    
    xmax = np.max(ax.get_xlim())
    ymax = np.max(ax.get_ylim())
    xmin = np.min(ax.get_xlim())
    ymin = np.min(ax.get_ylim())
    
    ax.set_xlabel(f"{x_label}", fontweight="bold")
    ax.set_ylabel(f"{y_label}", fontweight="bold")
    
    text = 'r = ' + str(format(spearmanr(data[y], data[x])[0], '.3f'))
    ax.text(xmin + 0.5 * xmin, ymax + 0.01 * ymax, text, verticalalignment='top',
            horizontalalignment='left', fontsize=20, fontweight="bold")
    
    legend = ax.legend(title="Gender", loc="lower right")
    # Modify individual legend labels
    new_legend_labels = ['Female', 'Male']
    for text, label in zip(legend.get_texts(), new_legend_labels):
        text.set_text(label)
    # Plot regression line    
    plt.plot([xmin+1, xmax- 1], [ymin, ymax], 'k--')
    
    target = target.replace('_', ' ').upper()

    ax.set_title(f"{y_label} vs {x_label} - Target={target} (Females={len(data_extracted_female)}, Males={len(data_extracted_male)})", fontsize=15, fontweight="bold", y=1)
    
    return plt

def plot_hgs_years_correlations(data, x, y, x_label, y_label, target, gender):
    
    fig, ax = plt.subplots(figsize=(20,10))
    sns.set_context("poster")
    ax.set_box_aspect(1)
    if gender == "both_gender":
        sns.regplot(data=data, x=x, y=y, ax=ax, line_kws={"color": "grey"}, scatter=False)
        sns.scatterplot(data=data, x=x, y=y, hue="gender", palette=['red', 'blue'])
    elif gender == "female":
        sns.regplot(data=data, x=x, y=y, ax=ax, color="red", line_kws={"color": "grey"}, )
    elif gender == "male":
        sns.regplot(data=data, x=x, y=y, ax=ax, color="blue", line_kws={"color": "grey"}, )

    ax.tick_params(axis='both',
                   labelsize=20,
                   labelsize=20,
                   )    
    xmax = np.max(ax.get_xlim())
    ymax = np.max(ax.get_ylim())
    xmin = np.min(ax.get_xlim())
    ymin = np.min(ax.get_ylim())
    
    ax.set_xlabel(f"{x_label}", fontweight="bold")
    ax.set_ylabel(f"{y_label}", fontweight="bold")
    
    text = 'r = ' + str(format(spearmanr(data[y], data[x])[0], '.3f'))
    ax.text(xmin + 0.5 * xmin, ymax + 0.01 * ymax, text, verticalalignment='top',
            horizontalalignment='left', fontsize=20, fontweight="bold")
    
    legend = ax.legend(title="Gender", loc="lower right")
    # Modify individual legend labels
    new_legend_labels = ['Female', 'Male']
    for text, label in zip(legend.get_texts(), new_legend_labels):
        text.set_text(label)
    # Plot regression line    
    plt.plot([xmin+1, xmax- 1], [ymin, ymax], 'k--')
    
    target = target.replace('_', ' ').upper()

    ax.set_title(f"{y_label} vs {x_label} - Target={target} (Females={len(data_extracted_female)}, Males={len(data_extracted_male)})", fontsize=15, fontweight="bold", y=1)
    
    return plt