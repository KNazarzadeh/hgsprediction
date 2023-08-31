
import pandas as pd
import numpy as np

from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

from hgsprediction.save_plot.save_correlations_plot.healthy_save_correlations_plot import save_correlations_plot


from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())


# Create the actual HGS vs predicted HGS plot for females and fefemales separately
def plot_hgs_correlations_scatter_plot(df, 
                        x,
                        y,
                        population,
                        mri_status,
                        model_name,
                        feature_type,
                        target,
):
    if y == "actual-predicted":
        y = "hgs_" + "(" + y + ")"
        # words = y.replace("hgs_", "").split('-')
        # capitalized_words = [word.capitalize() for word in words]
        y_label = "(Actual-Predicted)" + " HGS"
    else:
        y = "hgs_" + y
        y_label = y.replace("hgs_", "").capitalize() + " HGS"
##############################################################################
    if x in ["actual", "predicted"]:
        x = "hgs_" + x
        x_label = x.replace("hgs_", "").capitalize() + " HGS"
##############################################################################
    target_label = target.replace('_', ' ').upper()
    feature_label = feature_type.replace('_', ' and ').capitalize()
##############################################################################    
    df_female = df[df["gender"] == 0]
    df_male = df[df["gender"] == 1]
##############################################################################
    fig, ax = plt.subplots(figsize=(20,10))
    # Adjust 'labelsize' as needed
    plt.tick_params(axis='both', labelsize=20) 
    sns.set_context("poster")
    ax.set_box_aspect(1)
    sns.regplot(data=df, x=x, y=y, ax=ax, line_kws={"color": "grey"}, scatter=False)
    sns.scatterplot(data=df, x=x, y=y, hue="gender", palette=['red', 'blue'])

    xmax = np.max( ax.get_xlim())
    ymax = np.max( ax.get_ylim())
    xmin = np.min( ax.get_xlim())
    ymin = np.min( ax.get_ylim())
    
    ax.set_xlabel(f"{x_label}", fontweight="bold", fontsize=20)
    ax.set_ylabel(f"{y_label}", fontweight="bold", fontsize=20)
    
    text = 'r = ' + str(format(spearmanr(df[y], df[x])[0], '.3f'))
    ax.text(xmin + 0.5 * xmin, ymax + 0.01 * ymax, text, verticalalignment='top',
            horizontalalignment='left', fontsize=20, fontweight="bold")
    
    legend = ax.legend(title="Gender", loc="lower right", fontsize='xx-small')
    # Modify individual legend labels
    new_legend_labels = ['Female', 'Male']
    for text, label in zip(legend.get_texts(), new_legend_labels):
        text.set_text(label)
    # Plot regression line
    plt.plot([xmin+1, xmax- 1], [ymin, ymax], 'k--')
       
    ax.set_title(f"{y_label} vs {x_label} \n Features={feature_label}, Target={target_label} \n (N={len(df)})-(Females={len(df_female)}, Males={len(df_male)})", fontsize=15, fontweight="bold", y=1)

    file_path = save_correlations_plot(
        x,
        y,
        population,
        mri_status,
        model_name,
        feature_type,
        target,
        "both_gender")

    plt.show()
    plt.savefig(file_path)
    plt.close()
    print("===== Done! =====")
    embed(globals(), locals())
##############################################################################
    fig, ax = plt.subplots(figsize=(20,10))
    # Adjust 'labelsize' as needed
    plt.tick_params(axis='both', labelsize=20) 
    sns.set_context("poster")
    ax.set_box_aspect(1)        
    sns.regplot(data=df_female, x=x, y=y, ax=ax, color="red", line_kws={"color": "grey"})

    xmax = np.max(ax.get_xlim())
    ymax = np.max(ax.get_ylim())
    xmin = np.min(ax.get_xlim())
    ymin = np.min(ax.get_ylim())

    ax.set_xlabel(f"{x_label}", fontweight="bold", fontsize=20)
    ax.set_ylabel(f"{y_label}", fontweight="bold", fontsize=20)
    
    text = 'r = ' + str(format(spearmanr(df_female[y], df_female[x])[0], '.3f'))
    ax.text(xmin + 0.5 * xmin, ymax + 0.01 * ymax, text, verticalalignment='top',
            horizontalalignment='left', fontsize=20, fontweight="bold")

    # Plot regression line
    plt.plot([xmin+1, xmax- 1], [ymin, ymax], 'k--')

    ax.set_title(f"{y_label} vs {x_label} \n Features={feature_label}, Target={target_label} (Females={len(df_female)})", fontsize=15, fontweight="bold", y=1)
    
    file_path = save_correlations_plot(
            x,
            y,
            population,
            mri_status,
            model_name,
            feature_type,
            target,
            "female")

    plt.show()
    plt.savefig(file_path)
    plt.close()
##############################################################################
    fig, ax = plt.subplots(figsize=(20,10))
    # Adjust 'labelsize' as needed
    plt.tick_params(axis='both', labelsize=20) 
    sns.set_context("poster")
    ax.set_box_aspect(1)        
    sns.regplot(data=df_male, x=x, y=y, ax=ax, color="blue", line_kws={"color": "grey"})

    xmax = np.max(ax.get_xlim())
    ymax = np.max(ax.get_ylim())
    xmin = np.min(ax.get_xlim())
    ymin = np.min(ax.get_ylim())

    ax.set_xlabel(f"{x_label}", fontweight="bold", fontsize=20)
    ax.set_ylabel(f"{y_label}", fontweight="bold", fontsize=20)
    
    text = 'r = ' + str(format(spearmanr(df_male[y], df_male[x])[0], '.3f'))
    ax.text(xmin + 0.5 * xmin, ymax + 0.01 * ymax, text, verticalalignment='top',
            horizontalalignment='left', fontsize=20, fontweight="bold")

    # Plot regression line
    plt.plot([xmin+1, xmax- 1], [ymin, ymax], 'k--')

    ax.set_title(f"{y_label} vs {x_label} \n Features={feature_label}, Target={target_label} (Males={len(df_male)})", fontsize=15, fontweight="bold", y=1)

    file_path = save_correlations_plot(
            x,
            y,
            population,
            mri_status,
            model_name,
            feature_type,
            target,
            "male")

    plt.show()
    plt.savefig(file_path)
    plt.close()
##############################################################################
##############################################################################
# Create the actual HGS vs predicted HGS plot for females and fefemales separately
def plot_hgs_correlations_hexbin_plot(df, 
                        x,
                        y,
                        population,
                        mri_status,
                        model_name,
                        feature_type,
                        target,
):
    if y == "actual-predicted":
        y = "hgs_" + "(" + y + ")"
        # words = y.replace("hgs_", "").split('-')
        # capitalized_words = [word.capitalize() for word in words]
        y_label = "(Actual-Predicted)" + " HGS"
    else:
        y = "hgs_" + y
        y_label = y.replace("hgs_", "").capitalize() + " HGS"
##############################################################################
    if x in ["actual", "predicted"]:
        x = "hgs_" + x
        x_label = x.replace("hgs_", "").capitalize() + " HGS"
##############################################################################
    target_label = target.replace('_', ' ').upper()
    feature_label = feature_type.replace('_', ' and ').capitalize()
##############################################################################    
    df_female = df[df["gender"] == 0]
    df_male = df[df["gender"] == 1]
##############################################################################
    fig, axs=plt.subplots(2,2,figsize=(8,6), gridspec_kw={'hspace': 0, 
                                                            'wspace': 0,
                                                            'width_ratios': [5, 1],
                                                            'height_ratios': [1, 5]})

    # Upper part charts
    sns.jointplot(df_male, x=x, y=y, kind='hex', gridsize=20, color='blue', ax=axs[0,0])
    sns.jointplot(df_female, x=x, y=y, kind='hex', gridsize=20, color='red', ax=axs[0,0])

    axs[0,0].axis("off")
    axs[0,1].axis("off")
    axs[1,1].axis("off")

    # Right part charts
    sns.distplot(data[data.Gender=="Male"].Peso, bins=10, ax=axs[1,1], color="LightBlue", vertical=True)
    sns.distplot(data[data.Gender!="Male"].Peso, bins=10, ax=axs[1,1], color="lightcoral", vertical=True)

    # Linear regression for the middle plot
    sns.regplot(x="Altura", y="Peso", data=data[data.Gender=="Male"], color='blue', marker='+', ax=axs[1,0], scatter=False)
    sns.regplot(x="Altura", y="Peso", data=data[data.Gender!="Male"], color='magenta', marker='+', ax=axs[1,0], scatter=False)

    # KDE middle part
    sns.kdeplot(data[data.Gender=="Male"].Altura,data[data.Gender=="Male"].Peso, 
                shade=True,shade_lowest=False, cmap="Blues", ax=axs[1,0])
    sns.kdeplot(data[data.Gender!="Male"].Altura,data[data.Gender!="Male"].Peso, 
                shade=True,shade_lowest=False, cmap="Reds", ax=axs[1,0])
    file_path = save_correlations_plot(
        x,
        y,
        population,
        mri_status,
        model_name,
        feature_type,
        target,
        "both_gender")

    plt.show()
    plt.savefig(file_path)
    plt.close()
    print("===== Done! =====")
    embed(globals(), locals())
    
    
    fig, ax = plt.subplots()
    # # Adjust 'labelsize' as needed
    # plt.tick_params(axis='both', labelsize=20)
    sns.set(style="whitegrid")
 
    # sns.set_context("poster")
    # ax.set_box_aspect(1)
    # Create the hexbin plot
    # Create the hexbin plots for male and female
    jp1 = sns.jointplot(data=df_female, x=x, y=y, kind='hex', gridsize=20, color='red', ax=ax)
      
    jp2 = sns.jointplot(data=df_male, x=x, y=y, kind='hex', gridsize=20, color='blue', ax=ax)
    # Get the axes of the hexbin plots
    # hexplot = sns.jointplot(data=df, x=x, y=y, kind='hex', hue="gender", gridsize=20)
    # Add regression line
    # sns.regplot(data=df, x=x, y=y, ax=ax, line_kws={"color": "grey"}, scatter=False)
    # g = sns.jointplot(data=df_female, x=x, y=y, kind="hex", color="red")
    # g.jointplot(data=df_female, x=x, y=y, kind="hex", color="blue")
    # Adjust layout for both plots
    # plt.tight_layout()
    # Extract components from the first jointplot
    # Adjust 'labelsize' as needed
    # Remove the overlapping space
    plt.subplots_adjust(wspace=0, hspace=0)
        

    xmax = np.max( ax.get_xlim())
    ymax = np.max( ax.get_ylim())
    xmin = np.min( ax.get_xlim())
    ymin = np.min( ax.get_ylim())
    
    ax.set_xlabel(f"{x_label}", fontweight="bold", fontsize=20)
    ax.set_ylabel(f"{y_label}", fontweight="bold", fontsize=20)
    
    text = 'r = ' + str(format(spearmanr(df[y], df[x])[0], '.3f'))
    ax.text(xmin + 0.5 * xmin, ymax + 0.01 * ymax, text, verticalalignment='top',
            horizontalalignment='left', fontsize=20, fontweight="bold")
    
    legend = ax.legend(title="Gender", loc="lower right", fontsize='xx-small')
    # Modify individual legend labels
    new_legend_labels = ['Female', 'Male']
    for text, label in zip(legend.get_texts(), new_legend_labels):
        text.set_text(label)
    # Plot regression line
    # plt.plot([xmin+1, xmax- 1], [ymin, ymax], 'k--')
       
    fig.suptitle(f"{y_label} vs {x_label} \n Features={feature_label}, Target={target_label} \n (N={len(df)})-(Females={len(df_female)}, Males={len(df_male)})", y=1.5)

    file_path = save_correlations_plot(
        x,
        y,
        population,
        mri_status,
        model_name,
        feature_type,
        target,
        "both_gender")

    plt.show()
    plt.savefig(file_path)
    plt.close()
    print("===== Done! =====")
    embed(globals(), locals())
##############################################################################
    fig, ax = plt.subplots(figsize=(20,10))
    # Adjust 'labelsize' as needed
    plt.tick_params(axis='both', labelsize=20) 
    sns.set_context("poster")
    ax.set_box_aspect(1)
    # Create the hexbin plot
    hexplot = sns.jointplot(data=df_female, x=x, y=y, kind='hex', gridsize=20)
    # Add regression line
    ax = hexplot.ax_joint
    sns.regplot(data=df_female, x=x, y=y, ax=ax, scatter=False, color='grey')    
    # sns.regplot(data=df_female, x=x, y=y, ax=ax, color="red", line_kws={"color": "grey"})

    xmax = np.max(ax.get_xlim())
    ymax = np.max(ax.get_ylim())
    xmin = np.min(ax.get_xlim())
    ymin = np.min(ax.get_ylim())

    ax.set_xlabel(f"{x_label}", fontweight="bold", fontsize=20)
    ax.set_ylabel(f"{y_label}", fontweight="bold", fontsize=20)
    
    text = 'r = ' + str(format(spearmanr(df_female[y], df_female[x])[0], '.3f'))
    ax.text(xmin + 0.5 * xmin, ymax + 0.01 * ymax, text, verticalalignment='top',
            horizontalalignment='left', fontsize=20, fontweight="bold")

    # Plot regression line
    # plt.plot([xmin+1, xmax- 1], [ymin, ymax], 'k--')

    ax.set_title(f"{y_label} vs {x_label} \n Features={feature_label}, Target={target_label} (Females={len(df_female)})", fontsize=15, fontweight="bold", y=1)
    
    file_path = save_correlations_plot(
            x,
            y,
            population,
            mri_status,
            model_name,
            feature_type,
            target,
            "female")

    plt.show()
    plt.savefig(file_path)
    plt.close()
    print("===== Done! =====")
    embed(globals(), locals())
##############################################################################
    fig, ax = plt.subplots(figsize=(20,10))
    # Adjust 'labelsize' as needed
    plt.tick_params(axis='both', labelsize=20) 
    sns.set_context("poster")
    ax.set_box_aspect(1)        
    sns.regplot(data=df_male, x=x, y=y, ax=ax, color="blue", line_kws={"color": "grey"})

    xmax = np.max(ax.get_xlim())
    ymax = np.max(ax.get_ylim())
    xmin = np.min(ax.get_xlim())
    ymin = np.min(ax.get_ylim())

    ax.set_xlabel(f"{x_label}", fontweight="bold", fontsize=20)
    ax.set_ylabel(f"{y_label}", fontweight="bold", fontsize=20)
    
    text = 'r = ' + str(format(spearmanr(df_male[y], df_male[x])[0], '.3f'))
    ax.text(xmin + 0.5 * xmin, ymax + 0.01 * ymax, text, verticalalignment='top',
            horizontalalignment='left', fontsize=20, fontweight="bold")

    # Plot regression line
    plt.plot([xmin+1, xmax- 1], [ymin, ymax], 'k--')

    ax.set_title(f"{y_label} vs {x_label} \n Features={feature_label}, Target={target_label} (Males={len(df_male)})", fontsize=15, fontweight="bold", y=1)

    file_path = save_correlations_plot(
            x,
            y,
            population,
            mri_status,
            model_name,
            feature_type,
            target,
            "male")

    plt.show()
    plt.savefig(file_path)
    plt.close()
