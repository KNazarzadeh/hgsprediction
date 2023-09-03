
import pandas as pd
import numpy as np

from scipy.stats import spearmanr
from PIL import Image
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
    # print("===== Done! =====")
    # embed(globals(), locals())
##############################################################################
    # Assuming you have defined 'x' and 'y' as the columns you want to plot
    # Assuming df_male and df_female are your dataframes
    # Calculate axis limits based on both df_male and df_female
    alpha_value = 1

    # Create jointplots for df_male and df_female
    # j1 = sns.jointplot(data=df_male, x=x, y=y, kind='hex', gridsize=40, color='LightBlue', marginal_kws={"color": 'blue', "alpha": alpha_value})
    # j1.ax_marg_x.set_xlim(j1.ax_joint.get_xlim())
    # j1.ax_marg_y.set_ylim(j1.ax_joint.get_ylim())

    # j2 = sns.jointplot(data=df_female, x=x, y=y, kind='hex', gridsize=40, color='red', marginal_kws={"color": 'red', "alpha": alpha_value})
    # j2.ax_marg_x.set_xlim(j2.ax_joint.get_xlim())
    # j2.ax_marg_y.set_ylim(j2.ax_joint.get_ylim())

    # Create a JointGrid and set the axis limits
    # g = sns.JointGrid(x=None, y=None, data=None)
    # x_min = min(df_male[x].min(), df_female[x].min())
    # x_max = max(df_male[x].max(), df_female[x].max())
    # y_min = min(df_male[y].min(), df_female[y].min())
    # y_max = max(df_male[y].max(), df_female[y].max())

    # ax_min = min(x_min, y_min)
    # ax_max = max(x_max, y_max)
    
    # j1.ax_marg_x.set_xlim(ax_min, ax_max)
    # j2.ax_marg_x.set_xlim(ax_min, ax_max)
    # j1.ax_marg_y.set_ylim(ax_min, ax_max)
    # j2.ax_marg_y.set_ylim(ax_min, ax_max)
    # g.ax_marg_x.set_xlim(ax_min, ax_max)
    # g.ax_marg_y.set_ylim(ax_min, ax_max)

    # Set the joint plot for the JointGrid
    # g.ax_joint = j1.ax_joint

    # # Add the second jointplot to the JointGrid at the specified position
    # g.fig.add_subplot(j2.ax_joint.get_geometry())
    # g.fig.add_subplot(j2.ax_marg_x.get_geometry())
    # g.fig.add_subplot(j2.ax_marg_y.get_geometry())

    # Show the combined plot
    # plt.figure()
    # plt.show()
    # j1.savefig("h1.png")
    # j2.savefig("h2.png")
    # #read all the plots, merge and save the result 
    # h1m = Image.open("h1.png")
    # h2m = Image.open("h2.png")
    # h1m.paste(h2m, (0, 0), h2m)
    # h1m.save('j_hex_tr.png')

    # Image.open('hex_tr.png')
    # Image.open('j_hex_tr.png')

    # background = 'hex_tr.png'
    # layer_list = ['j_hex_tr.png']
    # bg1 = merge_plot(background, layer_list)
    # bg1.save('bg_hex6.png')
    # alpha_value = 0.2
    # # Create a jointplot for df_male with blue hexbins
    # j1 = sns.jointplot(data=df_male, x=x, y=y, kind='hex', gridsize=40, color='LightBlue', marginal_kws={"color": 'blue', "alpha": alpha_value})
    # j1.ax_marg_x.set_xlim(j1.ax_joint.get_xlim())
    # j1.ax_marg_y.set_ylim(j1.ax_joint.get_ylim())
    # j2 = sns.jointplot(data=df_female, x=x, y=y, kind='hex', gridsize=40, color='red', marginal_kws={"color": 'red', "alpha": alpha_value})
    # j2.ax_marg_x.set_xlim(j2.ax_joint.get_xlim())
    # j2.ax_marg_y.set_ylim(j2.ax_joint.get_ylim())
    # g = sns.JointGrid(x=None, y=None, data=None)
    # x_min = min(df_male[x].min(), df_female[x].min())
    # x_max = max(df_male[x].max(), df_female[x].max())
    # y_min = min(df_male[y].min(), df_female[y].min())
    # y_max = max(df_male[y].max(), df_female[y].max())
    
    # ax_min = min(x_min, y_min)
    # ax_max = max(x_max, y_max)
    # g.ax_marg_x.set_xlim(ax_min, ax_max)
    # g.ax_marg_y.set_ylim(ax_min, ax_max)
    # g.ax_joint = j1.ax_joint
    # g.ax_marg_x = j1.ax_marg_x
    # g.ax_marg_y = j1.ax_marg_y
    # g.plot_joint(j2.ax_joint)
    # g.ax_marg_y = [j1.ax_marg_y, j2.ax_marg_y]
    from PIL import Image, ImageDraw, ImageFilter
    # x_min = min(df_male[x].min(), df_female[x].min())
    # x_max = max(df_male[x].max(), df_female[x].max())
    # y_min = min(df_male[y].min(), df_female[y].min())
    # y_max = max(df_male[y].max(), df_female[y].max())

    # Sample data
    # Create a JointGrid
    g = sns.JointGrid(data=df, x=x, y=y)

    # Define custom plot functions for the joint plot
    def plot_func1(data, x, y, **kwargs):
        sns.jointplot(df_male, x, y, kind="hex", color="blue", **kwargs)

    def plot_func2(data, x, y, **kwargs):
        sns.jointplot(df_female, x, y, kind="hex", color="red", **kwargs)

    # Use plot_joint to create the first plot
    g.plot_joint(plot_func1)

    # Create a second plot using plot_joint in the same grid
    g.plot_joint(plot_func2)

    # Add marginal histograms
    sns.kdeplot(data=df, x=x, ax=g.ax_marg_x, color="blue", hue="gender")
    sns.kdeplot(data=df, y=y, ax=g.ax_marg_y, color="red", hue="gender")
    plt.show()
    print("===== Done! =====")
    embed(globals(), locals())
    # Create and save the first joint plot
    j1 = sns.jointplot(data=df_male, x=x, y=y, kind='hex', gridsize=40, color='blue', marginal_kws={"color": 'blue', "alpha": alpha_value})
    xmin1, xmax1 = j1.ax_marg_x.get_xlim()
    ymin1, ymax1 = j1.ax_marg_y.get_ylim()
    j1.ax_joint.set_facecolor("none")
    j1.ax_marg_x.set_facecolor("none")
    j1.ax_marg_y.set_facecolor("none")
    # j1.savefig("h1.png", transparent=True)
    # Loop through the patches and set face color based on index
    # for patch in j1.ax_marg_x.patches:
    #     patch.set_facecolor(df_male[x])
    #     patch.set_facecolor(df_female[x])   
    # print("===== Done! =====")
    # embed(globals(), locals())
    
    # # Create and save the second joint plot
    j2 = sns.jointplot(data=df_female, x=x, y=y, kind='hex', gridsize=40, color='lightcoral', marginal_kws={"color": 'lightcoral', "alpha": alpha_value})
    # xmin2, xmax2 = j2.ax_marg_x.get_xlim()
    # ymin2, ymax2 = j2.ax_marg_y.get_ylim()
    # # j2.ax_marg_x.set_xlim(ax_min, ax_max)
    # # j2.ax_marg_y.set_ylim(ax_min, ax_max)
    # j2.ax_joint.set_facecolor("none")
    # j2.ax_marg_x.set_facecolor("none")
    # j2.ax_marg_y.set_facecolor("none")

    # ax_xmin = min(xmin1, xmin2)
    # ax_xmax = max(xmax1, xmax2)
    # ax_ymin = min(ymin1, ymin2)
    # ax_ymax = max(ymax1, ymax2)
    
    # j1.ax_marg_x.set_xlim(ax_xmin, ax_xmax)
    # j1.ax_marg_y.set_ylim(ax_ymin, ax_ymax)
    # j2.ax_marg_x.set_xlim(ax_xmin, ax_xmax)
    # j2.ax_marg_y.set_ylim(ax_ymin, ax_ymax)
    # j1.savefig("h1.png", transparent=True)
    # j2.savefig("h2.png", transparent=True)

    # # Open the saved images
    # h1m = Image.open("h1.png")
    # h2m = Image.open("h2.png")
    # # mask_im = Image.new("L", h2m.size, 0)
    # # mask_im_blur = mask_im.filter(ImageFilter.GaussianBlur(10))
    # # mask_im_blur.save('x1.jpg', quality=95)

    # # back_im = h1m.copy()
    # # back_im.paste(h2m, (0, 0), mask_im_blur)
    # # back_im.save('xxx.png')

    # h2m.paste(h1m, (0, 0), h1m)
    # # j_hex_tr.plot(j_hex_tr.get_xlim(), j_hex_tr.get_ylim(), ls="--", color='b')
    # h2m.save('j_hex_tr.png', facecolor="white")
    # Merge the images
    # merged_image = Image.new("RGB", (h1m.width + h2m.width, h1m.height))
    # merged_image.paste(h1m, (0, 0))
    # merged_image.paste(h2m, (h1m.width, 0))

    # # Save the merged image
    # merged_image.save('j_hex_tr.png')

    # # Close the original images
    # h1m.close()
    # h2m.close()
    print("===== Done! =====")
    embed(globals(), locals())
# Now you can use 'j_hex_tr.png' as the merged image.


    file_path = save_correlations_plot(
            x,
            y,
            population,
            mri_status,
            model_name,
            feature_type,
            target,
            "both_gender")

    # plt.show()
    plt.savefig(file_path)
    # plt.close()
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
