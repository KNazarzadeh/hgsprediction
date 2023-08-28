
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

def plot_correlation_hgs(df,
                   x,
                   y,
                   stroke_type,
                   motor,
                   population,
                   target,
                   features,
                   gender,
):
    ###### Prepare and calculate variables for plotting ######
    # condition on x and calculate new column to dataframe
    if x == "years":
        df["years"] = df["days"]/365
    # Capitalize the first character
    if y == "actual-predicted":
        # Split the string using "-"
        parts = y.split("-")
        # Capitalize the first character of each part
        capitalized_parts = [part.capitalize() for part in parts]
        # Join the parts back together
        title = "-".join(capitalized_parts)
    else:
        # Capitalize the first character of target
        title = y.capitalize()
        y_label = y.capitalize() + " " + motor.upper()
    # Capitalize the first character of stroke_type
    if stroke_type == "post-stroke":
        stroke_type = stroke_type.capitalize()
    # Add the "hgs" to first of y to be same as column name
    y = motor + "_" + y
    
    ###### extract female and male dataframes ######
    df_male = df[df['31-0.0']==1.0]
    df_female = df[df['31-0.0']==0.0]
    ###### Calculate correlation ######
    corr_both_genders = spearmanr(df[y], df["years"])[0]
    corr_male = spearmanr(df_male[y], df_male["years"])[0]
    corr_female = spearmanr(df_female[y], df_female["years"])[0]
    ###### Set folder to save plot ######
    folder_path = os.path.join("data",
                               "project",
                               "stroke_ukb",
                               "knazarzadeh",
                               "project_hgsprediction",
                               "plots",
                               f"{population}",
                               "correlation",
                               f"{stroke_type}",
                               f"{gender}",
                               f"{features}",
                               f"{target}",
                               f"{y}",
                               f"{x}")
    ###### Creat plot for female and male ######
    fig = plt.figure(figsize=(15,15))
    # Set plot parameters
    sns.set_context("poster")
    ax = fig.add_subplot(111)
    plt.tick_params(axis='both', labelsize=30)
    # create regplots for correlation
    sns.regplot(x=df_male['years'], y=df_male[y], scatter_kws={"color": "blue"}, line_kws={"color": "grey"})
    sns.regplot(x=df_female['years'], y=df_female[y], scatter_kws={"color": "red"}, line_kws={"color": "grey"})
    # Set plot to square
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    # Add two text annotations to the corner
    text_male = 'r= ' + str(format(corr_male, '.3f'))
    text_female = 'r= ' + str(format(corr_female, '.3f'))
    # Get x and y axis lims
    xmax = np.max(ax.get_xlim())
    ymax = np.max(ax.get_ylim())
    xmin = np.min(ax.get_xlim())
    ymin = np.min(ax.get_ylim())
    # Add texts to plot
    plt.text(xmax - 0.15 * xmax, ymax - 0.005 * ymax, text_male, verticalalignment='top',
            horizontalalignment='right', fontsize=24, fontweight="bold", color="blue")
    plt.text(xmax - 0.15 * xmax, ymax - 0.065 * ymax,text_female, verticalalignment='top',
            horizontalalignment='right', fontsize=24, fontweight="bold", color="red")
    # Plot regression line
    plt.plot([xmin+1, xmax- 1], [ymin, ymax], 'k--')
    # Put a legend to the right of the current axis
    # with setting title
    legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Gender")
    # Modify individual legend labels
    new_legend_labels = ['Female', 'Male']
    for text, label in zip(legend.get_texts(), new_legend_labels):
        text.set_text(label)
    # set x and y labels
    plt.xlabel(f"{stroke_type} {x}", fontsize=30, fontweight="bold")
    plt.ylabel(y_label, fontsize=30, fontweight="bold")
    # Set title
    plt.title(f"{title} {motor.upper()} vs {stroke_type} {x} \n Target:{target} \n Features:{features}", fontsize=25, fontweight="bold", y=1.05)
    # Adjust layout to prevent clipping of legend
    plt.tight_layout()
    plt.show()
    # Save figure
    plt.savefig(os.path.join(folder_path, f"{stroke_type}_{x}_{target}_{title}_gender_separation_correlation.png"))
    plt.close()

    ###### Creat plot for both genders together ######   
    fig = plt.figure(figsize=(15,15))
    # Set plot parameters
    sns.set_context("poster")
    ax = fig.add_subplot(111)
    plt.tick_params(axis='both', labelsize=30)
    # create regplots for correlation    
    sns.regplot(data=df, x=x, y=y, line_kws={"color": "grey"}, scatter=False)
    sns.scatterplot(data=df, x=x, y=y, hue="31-0.0", palette=['red', 'blue'])
    # Set plot to square    
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    # Add two text annotations to the corner
    text = 'r= ' + str(format(corr_both_genders, '.3f'))
    # Get x and y axis lims
    xmax = np.max(ax.get_xlim())
    ymax = np.max(ax.get_ylim())
    xmin = np.min(ax.get_xlim())
    ymin = np.min(ax.get_ylim())
    # Add texts to plot    
    plt.text(xmax - 0.15 * xmax, ymax - 0.005 * ymax, text, verticalalignment='top',
            horizontalalignment='right', fontsize=24, fontweight="bold", color="black")
    # Plot regression line    
    plt.plot([xmin+1, xmax- 1], [ymin, ymax], 'k--')
    # Put a legend to the right of the current axis
    # with setting title
    legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Gender")
    # Modify individual legend labels
    new_legend_labels = ['Female', 'Male']
    for text, label in zip(legend.get_texts(), new_legend_labels):
        text.set_text(label)
    # set x and y labels
    plt.xlabel(f"{stroke_type} {x}", fontsize=30, fontweight="bold")
    plt.ylabel(y_label, fontsize=30, fontweight="bold")
    # Set title
    plt.title(f"{title} {motor.upper()} vs {stroke_type} {x} \n Target:{target} \n Features:{features}", fontsize=25, fontweight="bold", y=1.05)
    
    # Adjust layout to prevent clipping of legend
    plt.tight_layout()
    plt.show()
    # Save figure
    plt.savefig(os.path.join(folder_path, f"{stroke_type}_{x}_{target}_{title}_both_genders_together_correlation.png"))
    plt.close()

###############################################################################################################################################################
