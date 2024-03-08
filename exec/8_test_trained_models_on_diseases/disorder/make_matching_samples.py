import math
import sys
import numpy as np
import pandas as pd
import os

from hgsprediction.load_results.load_hgs_predicted_results import load_hgs_predicted_results
from hgsprediction.load_results.load_zscore_results import load_zscore_results
from hgsprediction.load_results.load_disorder_hgs_predicted_results import load_disorder_hgs_predicted_results
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())
pd.options.mode.chained_assignment = None  # 'None' suppresses the warning

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
disorder_cohort = sys.argv[11]
visit_session = sys.argv[12]

##############################################################################
df_zscore_healthy_mri_female = load_zscore_results(
    "healthy",
    "mri",
    model_name,
    feature_type,
    target,
    "female",
    session,
    confound_status,
    n_repeats,
    n_folds,
)

df_zscore_healthy_mri_male = load_hgs_predicted_results(
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    "male",
    session,
    confound_status,
    n_repeats,
    n_folds,
)
 
###############################################################################
disorder_cohort = f"{disorder_cohort}-{population}"
if visit_session == "1":
    session_column = f"1st_{disorder_cohort}_session"

df_disorder_female = load_disorder_hgs_predicted_results(
            population,
            mri_status,
            session_column,
            model_name,
            feature_type,
            target,
            "female",
            )

df_disorder_male = load_disorder_hgs_predicted_results(
            population,
            mri_status,
            session_column,
            model_name,
            feature_type,
            target,
            "male",
            )

    df_stroke = stroke.load_hgs_predicted_results("stroke", mri_status, session_column, model_name, feature_type, target, "both_gender")
    df_stroke.loc[:, "disease"] = 1
    # df_stroke = df_stroke.drop(index=1872273)

    df_pre_stroke = df_stroke.loc[:, ["gender", "1st_pre-stroke_age", "1st_pre-stroke_bmi",  "1st_pre-stroke_height",  "1st_pre-stroke_waist_to_hip_ratio", f"1st_pre-stroke_{target}", "disease"]]
    df_pre_stroke.rename(columns={"1st_pre-stroke_age":"age", "1st_pre-stroke_bmi":"bmi",  "1st_pre-stroke_height":"height",  "1st_pre-stroke_waist_to_hip_ratio":"waist_to_hip_ratio", 
                                "1st_pre-stroke_handedness":"handedness", f"1st_pre-stroke_{target}":f"{target}"}, inplace=True)

    df_post_stroke = df_stroke.loc[:, ["gender", "1st_post-stroke_age", "1st_post-stroke_bmi",  "1st_post-stroke_height",  "1st_post-stroke_waist_to_hip_ratio", f"1st_post-stroke_{target}", "disease"]]
    df_post_stroke.rename(columns={"1st_post-stroke_age":"age", "1st_post-stroke_bmi":"bmi",  "1st_post-stroke_height":"height",  "1st_post-stroke_waist_to_hip_ratio":"waist_to_hip_ratio",
                                "1st_post-stroke_handedness":"handedness", f"1st_post-stroke_{target}":f"{target}"}, inplace=True)
    ###############################################################################
    df_pre = pd.concat([df_healthy, df_pre_stroke], axis=0)
    df_pre.insert(0, "index", df_pre.index)

    df_post = pd.concat([df_healthy, df_post_stroke], axis=0)
    df_post.insert(0, "index", df_post.index)
    print("===== Done! =====")
    embed(globals(), locals())
    ##############################################################################
    # Define the covariates you want to use for matching
    # covariates = ["age", "bmi",  "height",  "waist_to_hip_ratio", f"{target}"]
    covariates = ["bmi", "height", "waist_to_hip_ratio", "age"]
    X = covariates
    y = target
    custom_palette = sns.color_palette(['#a851ab', '#005c95'])  # You can use any hex color codes you prefer
    fig, ax = plt.subplots(2,2, figsize=(12,12))
    for stroke_cohort in ["pre-stroke", "post-stroke"]:
        if stroke_cohort == "pre-stroke":
            df = df_pre.copy()
            df_stroke = df_pre[df_pre['disease']==1]
        elif stroke_cohort == "post-stroke":
            df = df_post.copy()
            df_stroke = df_post[df_post['disease']==1]
    ##############################################################################
        control_samples_female = pd.DataFrame()
        control_samples_male = pd.DataFrame()
        for gender in ["Female", "Male"]:
            if gender == "Female":
                data = df[df["gender"]==0]
                df_female_stroke = df_stroke[df_stroke["gender"]==0]
                print(gender)
            elif gender == "Male":    
                data = df[df["gender"]==1]
                df_male_stroke = df_stroke[df_stroke["gender"]==1]

            # Fit a logistic regression model to estimate propensity scores
            propensity_model = LogisticRegression()
            propensity_model.fit(data[covariates], data['disease'])
            propensity_scores = propensity_model.predict_proba(data[covariates])[:, 1]
            data['propensity_scores'] = propensity_scores

            # Create a DataFrame for diseaseed and control groups
            disease_group = data[data['disease'] == 1]
            control_group = data[data['disease'] == 0]

            matched_pairs = pd.DataFrame({'disease_index': disease_group.index})
            matched_data = pd.DataFrame()
            matched_patients = pd.DataFrame()
            matched_controls = pd.DataFrame()
            unmatched_controls = pd.DataFrame()
            unmatched_patients = pd.DataFrame()
            # Define the range of k from 1 to n
            n = 10 # You can change this to the desired value of n
            for k in range(1, n + 1):
                # Fit a Nearest Neighbors model on the control group with the current k
                knn = NearestNeighbors(n_neighbors=k)
                knn.fit(control_group[covariates])

                # Find the k nearest neighbors for each diseased unit
                distances, indices = knn.kneighbors(disease_group[covariates])

                matched_pairs_tmp = pd.DataFrame({
                    f'control_index_{k}': control_group.index[indices[:,k-1].flatten()],
                    f'distance_{k}': distances[:,k-1].flatten(),
                    f'propensity_score_{k}': propensity_scores[indices[:,k-1].flatten()]
                })
                
                matched_pairs = pd.concat([matched_pairs, matched_pairs_tmp], axis=1)
        
                matched_data_tmp = disease_group.reset_index(drop=True).join(control_group.iloc[indices[:,k-1].flatten()].reset_index(drop=True), lsuffix="_disease", rsuffix="_control")
                matched_data_tmp['distance'] = matched_pairs[f'distance_{k}'].values
                # matched_data = matched_data.append(matched_data_tmp)
                matched_data = pd.concat([matched_data, matched_data_tmp], axis=0)
                if gender == "Female":
                    control_samples_female = pd.concat([control_samples_female, control_group.iloc[indices[:,k-1].flatten()]], axis=0)
                    df_female = control_samples_female.copy()
                    df_female = predict_hgs(df_female, X, y, female_best_model_trained, target)
                    df_female_stroke = predict_hgs(df_female_stroke, X, y, female_best_model_trained, target)
                    corr_female_control = spearmanr(df_female[f"{target}_predicted"], df_female[f"{target}_actual"])[0]
                    corr_female_stroke = spearmanr(df_female_stroke[f"{target}_predicted"], df_female_stroke[f"{target}_actual"])[0]
                elif gender == "Male":                
                    control_samples_male = pd.concat([control_samples_male, control_group.iloc[indices[:,k-1].flatten()]], axis=0)
                    df_male = control_samples_male.copy()
                    df_male = predict_hgs(df_male, X, y, male_best_model_trained, target)
                    df_male_stroke = predict_hgs(df_male_stroke, X, y, male_best_model_trained, target)
                    corr_male_control = spearmanr(df_male[f"{target}_predicted"], df_male[f"{target}_actual"])[0]
                    corr_male_stroke = spearmanr(df_male_stroke[f"{target}_predicted"], df_male_stroke[f"{target}_actual"])[0]
            print(matched_data)
            print(matched_pairs)
        df_both_gender = pd.concat([df_female, df_male], axis=0)
        df_both_stroke = pd.concat([df_female_stroke, df_male_stroke], axis=0)
        corr_control = spearmanr(df_both_gender[f"{target}_predicted"], df_both_gender[f"{target}_actual"])[0]
        corr_stroke = spearmanr(df_both_stroke[f"{target}_predicted"], df_both_stroke[f"{target}_actual"])[0]
        text_control = 'r= ' + str(format(corr_control, '.3f'))
        text_stroke = 'r= ' + str(format(corr_stroke, '.3f'))
        text_control_female = 'r= ' + str(format(corr_female_control, '.3f'))
        text_stroke_female = 'r= ' + str(format(corr_female_stroke, '.3f'))
        text_control_male = 'r= ' + str(format(corr_male_control, '.3f'))
        text_stroke_male = 'r= ' + str(format(corr_male_stroke, '.3f'))
        print(df_both_gender)
        print(df_both_stroke)
        df_both_gender = df_both_gender.drop(columns=f"{target}")
        df_both_gender.rename(columns={f'{target}_actual':"actual", f"{target}_predicted":"predicted", f"{target}_(actual-predicted)": "delta"}, inplace=True)
        df_both_stroke = df_both_stroke.drop(columns=f"{target}")
        df_both_stroke.rename(columns={f'{target}_actual':"actual", f"{target}_predicted":"predicted", f"{target}_(actual-predicted)": "delta"}, inplace=True)
        if target == "hgs_L+R":
            if stroke_cohort == "pre-stroke":
                df_l_r_pre = df_both_gender
                df_l_r_pre['hgs_target'] = "HGS L+R"
                df_l_r_stroke_pre = df_both_stroke                
                df_l_r_stroke_pre['hgs_target'] = "HGS L+R"                
            elif stroke_cohort == "post-stroke":
                df_l_r_post = df_both_gender
                df_l_r_post['hgs_target'] = "HGS L+R"
                df_l_r_stroke_post = df_both_stroke                
                df_l_r_stroke_post['hgs_target'] = "HGS L+R"                 
        elif target == "hgs_left":
            if stroke_cohort == "pre-stroke":
                df_left_pre = df_both_gender
                df_left_pre['hgs_target'] = "HGS Left"
                df_left_stroke_pre = df_both_stroke                
                df_left_stroke_pre['hgs_target'] = "HGS Left"                 
            elif stroke_cohort == "post-stroke":
                df_left_post = df_both_gender
                df_left_post['hgs_target'] = "HGS Left"
                df_left_stroke_post = df_both_stroke                
                df_left_stroke_post['hgs_target'] = "HGS Left"                  
        elif target == "hgs_right":
            if stroke_cohort == "pre-stroke":
                df_right_pre = df_both_gender
                df_right_pre['hgs_target'] = "HGS Right"
                df_right_stroke_pre = df_both_stroke                
                df_right_stroke_pre['hgs_target'] = "HGS Right"                  
            elif stroke_cohort == "post-stroke":
                df_right_post = df_both_gender
                df_right_post['hgs_target'] = "HGS Right"
                df_right_stroke_post = df_both_stroke                
                df_right_stroke_post['hgs_target'] = "HGS Right"    

    ##############################################################################
    ##############################################################################
    
df_both_pre = pd.concat([df_left_pre, df_right_pre, df_l_r_pre])
df_both_pre['stroke_cohort'] = "pre"
df_both_post = pd.concat([df_left_post, df_right_post, df_l_r_post])
df_both_post['stroke_cohort'] = "post"

df = pd.concat([df_both_pre, df_both_post])

df_both_stroke_pre = pd.concat([df_left_stroke_pre, df_right_stroke_pre, df_l_r_stroke_pre])
df_both_stroke_pre['stroke_cohort'] = "pre"
df_both_stroke_post = pd.concat([df_left_stroke_post, df_right_stroke_post, df_l_r_stroke_post])
df_both_stroke_post['stroke_cohort'] = "post"

df_stroke_together = pd.concat([df_both_stroke_pre, df_both_stroke_post])

###############################################################################
def add_median_labels(ax, fmt='.3f'):
    xticks_positios_array = []
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
    lines_per_box = int(len(lines) / len(boxes))
    for median in lines[4:len(lines):lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if (median.get_xdata()[1] - median.get_xdata()[0]) == 0 else y
        text = ax.text(x, y, f'{value:{fmt}}', ha='center', va='center',  color='white', fontsize=12)
                    #    fontweight='bold',
        # create median-colored border around white text for contrast
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground=median.get_color()),
            path_effects.Normal(),
        ])
        xticks_positios_array.append(x)
    return xticks_positios_array
###############################################################################
###############################################################################
print("===== Done! =====")
embed(globals(), locals())
df_anova=pd.concat([df,df_stroke_together])
a = df_anova[["disease", "gender", "delta", "hgs_target", "stroke_cohort"]]
b = a[a["hgs_target"]=="HGS L+R"]
b = b.rename(columns={"disease":"group", "stroke_cohort":"disease_time"})
b["group"].replace(0, "healthy", inplace=True)
b["group"].replace(1, "stroke", inplace=True)
b["gender"].replace(0, "female", inplace=True)
b["gender"].replace(1, "male", inplace=True)
formula = 'delta ~ C(group) + C(disease_time) + C(hgs_target) + C(gender) + C(group):C(disease_time) + C(group):C(hgs_target) + C(group):C(gender) + C(disease_time):C(hgs_target) + C(disease_time):C(gender) + C(hgs_target):C(gender) + C(group):C(disease_time):C(hgs_target) + C(group):C(disease_time):C(gender) + C(group):C(hgs_target):C(gender) + C(disease_time):C(hgs_target):C(gender) + C(group):C(disease_time):C(hgs_target):C(gender)'
model = ols(formula, b).fit()
anova_results = sm.stats.anova_lm(model, typ=2)

print(anova_results)
print("===== Done! =====")
embed(globals(), locals())
# Define a palette for hgs_target
# Create a dictionary for mapping gender to colors and labels
# Define palettes for hgs_target for Female and Male
female_palette = {'HGS Left': 'lightcoral', 'HGS Right': 'darkred'}
male_palette = {'HGS Left': 'lightblue', 'HGS Right': 'darkblue'}
# Create a point plot
plt.figure(figsize=(12, 8))
g = sns.catplot(
    data=b[b["gender"]=="female"], x="disease_time", y="delta", hue="hgs_target", col="group",
    capsize=.2, palette=female_palette, errorbar="se",
    kind="point", height=6, aspect=.75,
)
g.despine(left=True)
plt.show()
plt.savefig("anova_delta_gender_female.png")
plt.figure(figsize=(12, 8))
g = sns.catplot(
    data=b[b["gender"]=="male"], x="disease_time", y="delta", hue="hgs_target", col="group",
    capsize=.2, palette=male_palette, errorbar="se",
    kind="point", height=6, aspect=.75,
)
g.despine(left=True)
plt.show()
plt.savefig("anova_delta_gender_male.png")
# Create a point plot
plt.figure(figsize=(12, 8))
g = sns.catplot(
    data=b[b["gender"]=="female"], x="group", y="delta", hue="hgs_target", col="disease_time",
    capsize=.2, palette=female_palette, errorbar="se",
    kind="point", height=6, aspect=.75,
)
g.despine(left=True)
plt.show()
plt.savefig("anova_group_xaxis_delta_gender_female.png")

plt.figure(figsize=(12, 8))
g = sns.catplot(
    data=b[b["gender"]=="male"], x="group", y="delta", hue="hgs_target", col="disease_time",
    capsize=.2, palette=male_palette, errorbar="se",
    kind="point", height=6, aspect=.75,
)
g.despine(left=True)
plt.show()
plt.savefig("anova_group_xaxis_delta_gender_male.png")


# Perform post-hoc tests on significant interactions (Tukey's HSD)
import statsmodels.stats.multicomp as mc
interaction_groups =  b.disease_time.astype(str) + "_"+ b.group.astype(str)+ "_" + b.hgs_target.astype(str)
# interaction_groups =  b.disease_time.astype(str) + "_" + b.group.astype(str) + "_" + b.hgs_target.astype(str) + "_" + b.gender.astype(str)
comp = mc.MultiComparison(b["delta"], interaction_groups)
post_hoc_res = comp.tukeyhsd()
print(post_hoc_res.summary())

from statsmodels.graphics.factorplots import interaction_plot
fig = interaction_plot(x=b['group'], trace=b['hgs_target'], response=b['delta'], 
    colors=['#4c061d','#d17a22', '#b4c292'])
plt.show()
plt.savefig("interaction_plot.png")
###############################################################################
###############################################################################
df["hgs_target_stroke_cohort"] = df["hgs_target"] + "-" +df["stroke_cohort"]
df_stroke_together["hgs_target_stroke_cohort"] = df_stroke_together["hgs_target"] + "-" +df_stroke_together["stroke_cohort"]

df_healthy_anova = [df_left_pre["delta"], df_left_post["delta"], df_right_pre["delta"], df_right_post["delta"], df_l_r_pre["delta"], df_l_r_post["delta"]]
df_stroke_anova = [df_left_stroke_pre["delta"], df_left_stroke_post["delta"], df_right_stroke_pre["delta"], df_right_stroke_post["delta"], df_l_r_stroke_pre["delta"], df_l_r_stroke_post["delta"]]

# Perform ANOVA
_, p_value = stats.f_oneway(*df_healthy_anova, *df_stroke_anova)
print(p_value)
# Define significance level (alpha)
alpha = 0.05

# Check if p-value is less than alpha
if p_value < alpha:
    print("ANOVA results: There are significant differences in HGS across the groups.")
else:
    print("ANOVA results: There are no significant differences in HGS across the groups.")

###############################################################################
###############################################################################
df_healthy_pre_anova = [df_left_pre["delta"], df_right_pre["delta"], df_l_r_pre["delta"]]
df_healthy_post_anova = [df_left_post["delta"], df_right_post["delta"], df_l_r_post["delta"]]
df_stroke_pre_anova = [df_left_stroke_pre["delta"], df_right_stroke_pre["delta"], df_l_r_stroke_pre["delta"]]
df_stroke_post_anova = [df_left_stroke_post["delta"], df_right_stroke_post["delta"], df_l_r_stroke_post["delta"]]

# Perform two-way ANOVA for "pre" groups (healthy and stroke)
_, p_value_pre = stats.f_oneway(*df_healthy_pre_anova, *df_stroke_pre_anova)

# Perform two-way ANOVA for "post" groups (healthy and stroke)
_, p_value_post = stats.f_oneway(*df_healthy_post_anova, *df_stroke_post_anova)

print(p_value_pre)
print(p_value_post)
# Define significance level (alpha)
alpha = 0.05

# Check if p-values are less than alpha for both "pre" and "post" groups
if p_value_pre < alpha:
    print("ANOVA results for 'pre' groups: There are significant differences between healthy and stroke.")
else:
    print("ANOVA results for 'pre' groups: There are no significant differences between healthy and stroke.")

if p_value_post < alpha:
    print("ANOVA results for 'post' groups: There are significant differences between healthy and stroke.")
else:
    print("ANOVA results for 'post' groups: There are no significant differences between healthy and stroke.")

###############################################################################
###############################################################################




###############################################################################
###############################################################################
df_main = pd.concat([df, df_stroke_together])
for y_axis in ["actual", "predicted", "delta"]:
    melted_df = pd.melt(df_main, id_vars=["hgs_target_stroke_cohort", "disease"], value_vars=y_axis, var_name="variable", ignore_index=False)
    # Initialize a list to store the test results
    results = pd.DataFrame(columns=["hgs_target_stroke_cohort", "ranksum_stat", "ranksum_p_value", f"max_sample_{y_axis}", f"max_stroke_{y_axis}"])
    for i, hgs_target_stroke_cohort in enumerate(["HGS Left-pre", "HGS Left-post", "HGS Right-pre", "HGS Right-post", "HGS L+R-pre", "HGS L+R-post"]):
        tmp = melted_df[melted_df["hgs_target_stroke_cohort"]== hgs_target_stroke_cohort]
        tmp_samples = tmp[tmp["disease"]==0]
        tmp_stroke = tmp[tmp["disease"]==1]
        stat, p_value = ranksums(tmp_samples["value"], tmp_stroke["value"])
        print(tmp)
        print(stat, p_value)
        results.loc[i, "hgs_target_stroke_cohort"] = hgs_target_stroke_cohort
        results.loc[i, "ranksum_stat"] = stat
        results.loc[i, "ranksum_p_value"] = p_value
        results.loc[i, f"max_sample_{y_axis}"] = tmp_samples["value"].max()
        results.loc[i, f"max_stroke_{y_axis}"] = tmp_stroke["value"].max()

    # Define a custom palette with two blue colors
    custom_palette = sns.color_palette(['#95CADB', '#008ECC'])  # You can use any hex color codes you prefer
    plt.figure(figsize=(18, 10))  # Adjust the figure size if needed
    sns.set(style="whitegrid")
    # Define the order in which you want the x-axis categories
    x_order = ['HGS Left-pre', 'HGS Left-post', 'HGS Right-pre', 'HGS Right-post', 'HGS L+R-pre', 'HGS L+R-post']
    ax = sns.boxplot(data=melted_df, x="hgs_target_stroke_cohort", y="value", hue="disease", order=x_order, palette=custom_palette)   
    # Add labels and title
    plt.xlabel("HGS targets", fontsize=20, fontweight="bold")
    plt.ylabel(f"HGS {y_axis.capitalize()} values", fontsize=20, fontweight="bold")
    plt.title(f"Matching samples from controls vs Stroke HGS {y_axis.capitalize()} values", fontsize=15, fontweight="bold")

    ymin, ymax = plt.ylim()
    plt.yticks(range(math.floor(ymin/10)*10, math.ceil(ymax/10)*10+10, 10), fontsize=18, weight='bold')
    plt.xticks(fontsize=18, weight='bold')
    legend = plt.legend(loc="upper left", prop={'size': 16, 'weight': 'bold'})
    legend.set_title("Samples", {'size': 16, 'weight': 'bold'})
    # Modify individual legend labels
    legend.get_texts()[0].set_text(f"Matching samples from controls(N={len(df_both_gender)})")
    legend.get_texts()[1].set_text(f"Stroke(N={len(df_stroke)})")

    plt.tight_layout()

    xticks_positios_array = add_median_labels(ax)

    for i, x_box_pos in enumerate(np.arange(0,11,2)):
        x1 = xticks_positios_array[x_box_pos]
        x2 = xticks_positios_array[x_box_pos+1]
        y, h, col = results.loc[i, f"max_sample_{y_axis}"] + 2, 2, 'k'
        plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
        plt.text((x1+x2)*.5, y+h, f"p={results.loc[i, 'ranksum_p_value']:.3f}", ha='center', va='bottom', fontsize=14, weight='bold',  color=col)

    plt.show()
    plt.savefig(f"boxplot_1_to_{n}_samples_{session_column}_{y_axis}_{population}_{feature_type}_hgs_both_gender_controls_Stroke.png")
    plt.close()
###############################################################################

    melted_df_female = pd.melt(df_main[df_main["gender"]==0], id_vars=["hgs_target_stroke_cohort", "disease"], value_vars=y_axis, var_name="variable", ignore_index=False)
    results_female = pd.DataFrame(columns=["hgs_target_stroke_cohort", "ranksum_stat", "ranksum_p_value", f"max_sample_{y_axis}", f"max_stroke_{y_axis}"])
    for i, hgs_target_stroke_cohort in enumerate(["HGS Left-pre", "HGS Left-post", "HGS Right-pre", "HGS Right-post", "HGS L+R-pre", "HGS L+R-post"]):
        tmp = melted_df_female[melted_df_female["hgs_target_stroke_cohort"]== hgs_target_stroke_cohort]
        tmp_samples = tmp[tmp["disease"]==0]
        tmp_stroke = tmp[tmp["disease"]==1]
        stat, p_value = ranksums(tmp_samples["value"], tmp_stroke["value"])
        print(tmp)
        print(stat, p_value)
        results_female.loc[i, "hgs_target_stroke_cohort"] = hgs_target_stroke_cohort
        results_female.loc[i, "ranksum_stat"] = stat
        results_female.loc[i, "ranksum_p_value"] = p_value
        results_female.loc[i, f"max_sample_{y_axis}"] = tmp_samples["value"].max()
        results_female.loc[i, f"max_stroke_{y_axis}"] = tmp_stroke["value"].max()
    custom_palette = sns.color_palette(['#ca96cc', '#a851ab'])  # You can use any hex color codes you prefer
    plt.figure(figsize=(18, 10))  # Adjust the figure size if needed
    sns.set(style="whitegrid")
    x_order = ['HGS Left-pre', 'HGS Left-post', 'HGS Right-pre', 'HGS Right-post', 'HGS L+R-pre', 'HGS L+R-post']
    ax = sns.boxplot(data=melted_df_female, x="hgs_target_stroke_cohort", y="value", hue="disease", order=x_order, palette=custom_palette)    
    # Add labels and title
    plt.xlabel("HGS targets", fontsize=20, fontweight="bold")
    plt.ylabel(f"HGS {y_axis.capitalize()} values", fontsize=20, fontweight="bold")
    plt.title(f"Matching samples from controls vs Stroke HGS {y_axis.capitalize()} values - Females", fontsize=15, fontweight="bold")

    ymin, ymax = plt.ylim()
    plt.yticks(range(math.floor(ymin/10)*10, math.ceil(ymax/10)*10+10, 10), fontsize=18, weight='bold')
    plt.xticks(fontsize=18, weight='bold')
    
    legend = plt.legend(loc="upper left", prop={'size': 16, 'weight': 'bold'})
    legend.set_title("Samples", {'size': 16, 'weight': 'bold'})    # Modify individual legend labels
    female_matching_samples_number = len(df_both_gender[df_both_gender["gender"]==0])
    female_stroke_number = len(df_both_stroke[df_both_stroke["gender"]==0])
    legend.get_texts()[0].set_text(f"Matching samples from controls Female(N={female_matching_samples_number})")
    legend.get_texts()[1].set_text(f"Stroke Female(N={female_stroke_number})")

    plt.tight_layout()

    xticks_positios_array = add_median_labels(ax)

    for i, x_box_pos in enumerate(np.arange(0,11,2)):
        x1 = xticks_positios_array[x_box_pos]
        x2 = xticks_positios_array[x_box_pos+1]
        y, h, col = results_female.loc[i, f"max_sample_{y_axis}"] + 2, 2, 'k'
        plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
        plt.text((x1+x2)*.5, y+h, f"p={results_female.loc[i, 'ranksum_p_value']:.3f}", ha='center', va='bottom', fontsize=14, weight='bold', color=col)

    plt.show()
    plt.savefig(f"boxplot_1_to_{n}_samples_{session_column}_{y_axis}_{population}_{feature_type}_hgs_separate_gender_separated_Stroke_Female.png")
    plt.close()

###############################################################################

    melted_df_male = pd.melt(df_main[df_main["gender"]==1], id_vars=["hgs_target_stroke_cohort", "disease"], value_vars=y_axis, var_name="variable", ignore_index=False)
    results_male = pd.DataFrame(columns=["hgs_target_stroke_cohort", "ranksum_stat", "ranksum_p_value", f"max_sample_{y_axis}", f"max_stroke_{y_axis}"])
    for i, hgs_target_stroke_cohort in enumerate(["HGS Left-pre", "HGS Left-post", "HGS Right-pre", "HGS Right-post", "HGS L+R-pre", "HGS L+R-post"]):
        tmp = melted_df_male[melted_df_male["hgs_target_stroke_cohort"]== hgs_target_stroke_cohort]
        tmp_samples = tmp[tmp["disease"]==0]
        tmp_stroke = tmp[tmp["disease"]==1]
        stat, p_value = ranksums(tmp_samples["value"], tmp_stroke["value"])
        print(tmp)
        print(stat, p_value)
        results_male.loc[i, "hgs_target_stroke_cohort"] = hgs_target_stroke_cohort
        results_male.loc[i, "ranksum_stat"] = stat
        results_male.loc[i, "ranksum_p_value"] = p_value
        results_male.loc[i, f"max_sample_{y_axis}"] = tmp_samples["value"].max()
        results_male.loc[i, f"max_stroke_{y_axis}"] = tmp_stroke["value"].max()

    custom_palette = sns.color_palette(['#669dbf', '#005c95'])  # You can use any hex color codes you prefer
    plt.figure(figsize=(18, 10))  # Adjust the figure size if needed
    sns.set(style="whitegrid")
    x_order = ['HGS Left-pre', 'HGS Left-post', 'HGS Right-pre', 'HGS Right-post', 'HGS L+R-pre', 'HGS L+R-post']
    ax = sns.boxplot(data=melted_df_male, x="hgs_target_stroke_cohort", y="value", hue="disease", order=x_order, palette=custom_palette)    
    # Add labels and title
    plt.xlabel("HGS targets", fontsize=20, fontweight="bold")
    plt.ylabel(f"HGS {y_axis.capitalize()} values", fontsize=20, fontweight="bold")
    plt.title(f"Matching samples from controls vs Stroke HGS {y_axis.capitalize()} values - Males", fontsize=15, fontweight="bold")

    ymin, ymax = plt.ylim()
    plt.yticks(range(math.floor(ymin/10)*10, math.ceil(ymax/10)*10+10, 10), fontsize=18, weight='bold')
    plt.xticks(fontsize=18, weight='bold')
    legend = plt.legend(loc="upper left", prop={'size': 16, 'weight': 'bold'})
    legend.set_title("Samples", {'size': 16, 'weight': 'bold'})
    male_matching_samples_number = len(df_both_gender[df_both_gender["gender"]==1])
    male_stroke_number = len(df_both_stroke[df_both_stroke["gender"]==1])
    # Modify individual legend labels
    legend.get_texts()[0].set_text(f"Matching samples from controls Male(N={male_matching_samples_number})")
    legend.get_texts()[1].set_text(f"Stroke Male(N={male_stroke_number})")

    plt.tight_layout()

    xticks_positios_array = add_median_labels(ax)

    for i, x_box_pos in enumerate(np.arange(0,11,2)):
        x1 = xticks_positios_array[x_box_pos]
        x2 = xticks_positios_array[x_box_pos+1]
        y, h, col = results_male.loc[i, f"max_sample_{y_axis}"] + 2, 2, 'k'
        plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
        plt.text((x1+x2)*.5, y+h, f"p={results_male.loc[i, 'ranksum_p_value']:.3f}", ha='center', va='bottom', fontsize=14, weight='bold', color=col)

    plt.show()
    plt.savefig(f"boxplot_1_to_{n}_samples_{session_column}_{y_axis}_{population}_{feature_type}_hgs_separate_gender_separated_Stroke_Male.png")
    plt.close()


print("===== Done! =====")
embed(globals(), locals())