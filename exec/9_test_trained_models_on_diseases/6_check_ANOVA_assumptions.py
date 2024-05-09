import math
import sys
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM

from hgsprediction.load_results.load_prepared_data_for_anova import load_prepare_data_for_anova
from scipy.stats import levene, shapiro, kstest
from scipy import stats
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################
filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
feature_type = sys.argv[3]
model_name = sys.argv[4]
confound_status = sys.argv[5]
n_repeats = sys.argv[6]
n_folds = sys.argv[7]
disorder_cohort = sys.argv[8]
visit_session = sys.argv[9]
n_samples = sys.argv[10]
target = sys.argv[11]
anova_target = sys.argv[12]
##############################################################################
disorder_cohort = f"{disorder_cohort}-{population}"
if visit_session == "1":
    session_column = f"1st_{disorder_cohort}_session"
##############################################################################
# Load data for ANOVA
df = load_prepare_data_for_anova(
    population,
    mri_status,
    session_column,
    model_name,
    feature_type,
    target,
    confound_status,
    n_repeats,
    n_folds,
    n_samples,
)
df["gender"].replace(0, "female", inplace=True)
df["gender"].replace(1, "male", inplace=True)

df_female = df[df["gender"]=="female"]
df_male = df[df["gender"]=="male"]

df_control = df[df['group']=='control']
df_disorder = df[df['group']==f'{population}']
# print("===== Done! =====")
# embed(globals(), locals())
##############################################################################
male_pre_controls = df_male[df_male["time_point"]=="pre-control"][anova_target]

female_pre_controls = df_female[df_female["time_point"]=="pre-control"][anova_target]

male_post_controls = df_male[df_male["time_point"]=="post-control"][anova_target]

female_post_controls = df_female[df_female["time_point"]=="post-control"][anova_target]

male_pre_patients = df_male[df_male["time_point"]==f"pre-{population}"][anova_target]

female_pre_patients = df_female[df_female["time_point"]==f"pre-{population}"][anova_target]

male_post_patients = df_male[df_male["time_point"]==f"post-{population}"][anova_target]

female_post_patients = df_female[df_female["time_point"]==f"post-{population}"][anova_target]

##############################################################################
# Check Homogeneity of Variance to use gender as a main factor in ANOVA:
# Null and Alternative Hypotheses:
#     Null Hypothesis (H0): All group variances are equal.
#     Alternative Hypothesis (H1): At least one group has a variance that is different from the others.
# Interpreting the p-value:
#     p-value ≤ α (commonly 0.05): If the p-value is less than or equal to the significance level (typically 0.05), 
# you reject the null hypothesis. This suggests that there is evidence that 
# at least one group’s variance significantly differs from the others, indicating that the assumption of homogeneity of variances is violated.
#     p-value > α: If the p-value is greater than the significance level, you fail to reject the null hypothesis. 
# This indicates that there is no sufficient evidence to suggest the variances are different, supporting the assumption of equal variances among the groups.
######################
# Between male controls and female controls (pre-time_point)
stat_male_female_pre_controls, p_value_male_female_pre_controls = levene(male_pre_controls, female_pre_controls)
if p_value_male_female_pre_controls < .05:
    print(f"Levene's test between male and female controls (pre-time_point) is rejected (The variances are different): (p-value = {p_value_male_female_pre_controls:.6f})")
else:
    print(f"Levene's test between male and female controls (pre-time_point) is met (The variances are equal): (p-value = {p_value_male_female_pre_controls:.6f})")

# Between male controls and female controls (post-time_point)
stat_male_female_post_controls, p_value_male_female_post_controls = levene(male_post_controls, female_post_controls)
if p_value_male_female_post_controls < .05:
    print(f"Levene's test between male and female controls (post-time_point) is rejected (The variances are different): (p-value = {p_value_male_female_post_controls:.6f})")
else:
    print(f"Levene's test between male and female controls (post-time_point) is met (The variances are equal): (p-value = {p_value_male_female_post_controls:.6f})")
    
# Between male patients and female patients (pre-time_point)
stat_male_female_pre_patients, p_value_male_female_pre_patients = levene(male_pre_patients, female_pre_patients)
if p_value_male_female_pre_patients < .05:
    print(f"Levene's test between male and female patients (pre-time_point) is rejected (The variances are different): (p-value = {p_value_male_female_pre_patients:.6f})")
else:
    print(f"Levene's test between male and female patients (pre-time_point) is met (The variances are equal): (p-value = {p_value_male_female_pre_patients:.6f})")

# Between male patients and female patients (post-time_point)
stat_male_female_post_patients, p_value_male_female_post_patients = levene(male_post_patients, female_post_patients)
if p_value_male_female_post_patients < .05:
    print(f"Levene's test between male and female patients (post-time_point) is rejected (The variances are different): (p-value = {p_value_male_female_post_patients:.6f})")
else:
    print(f"Levene's test between male and female patients (post-time_point) is met (The variances are equal): (p-value = {p_value_male_female_post_patients:.6f})")
# Between pre-time_point and post-time_point for male controls
stat_male_pre_controls_patients, p_value_male_pre_controls_patients = levene(male_pre_controls, male_pre_patients)
if p_value_male_pre_controls_patients < .05:
    print(f"Levene's test between male controls and male patients (pre-time_point) is rejected (The variances are different): (p-value = {p_value_male_pre_controls_patients:.6f})")
else:
    print(f"Levene's test between male controls and male patients (pre-time_point) is met (The variances are equal): (p-value = {p_value_male_pre_controls_patients:.6f})")

# Between pre-time_point and post-time_point for female controls
stat_female_pre_controls_patients, p_value_female_pre_controls_patients = levene(female_pre_controls, female_pre_patients)
if p_value_female_pre_controls_patients < .05:
    print(f"Levene's test between female controls and female patients (pre-time_point) is rejected (The variances are different): (p-value = {p_value_female_pre_controls_patients:.6f})")
else:
    print(f"Levene's test between female controls and female patients (pre-time_point) is met (The variances are equal): (p-value = {p_value_female_pre_controls_patients:.6f})")
    
# Between pre-time_point and post-time_point for male patients
stat_male_post_controls_patients, p_value_male_post_controls_patients = levene(male_post_controls, male_post_patients)
if p_value_male_post_controls_patients < .05:
    print(f"Levene's test between male controls and male patients (post-time_point) is rejected (The variances are different): (p-value = {p_value_male_post_controls_patients:.6f})")
else:
    print(f"Levene's test between male controls and male patients (post-time_point) is met (The variances are equal): (p-value = {p_value_male_post_controls_patients:.6f})")
    
# Between pre-time_point and post-time_point for female patients
stat_female_post_controls_patients, p_value_female_post_controls_patients = levene(female_post_controls, female_post_patients)
if p_value_female_post_controls_patients < .05:
    print(f"Levene's test between female controls and female patients (post-time_point) is rejected (The variances are different): (p-value = {p_value_female_post_controls_patients:.6f})")
else:
    print(f"Levene's test between female controls and female patients (post-time_point) is met (The variances are equal): (p-value = {p_value_female_post_controls_patients:.6f})")
####################
if (p_value_male_female_pre_controls <= .05 or p_value_male_female_post_controls <= .05 or p_value_male_female_pre_patients <= .05 or p_value_male_female_post_patients <= .05 or
    p_value_male_pre_controls_patients <= .05 or p_value_female_pre_controls_patients <= .05 or p_value_male_post_controls_patients <= .05 or p_value_female_post_controls_patients <= .05):
    print("Levene's test is rejected (The variances are different)")
    print(f"********** Gender cannot be used as a main factor for ANOVA in {population} **********")
elif (p_value_male_female_pre_controls > .05 and p_value_male_female_post_controls > .05 and p_value_male_female_pre_patients > .05 and p_value_male_female_post_patients > .05 and
    p_value_male_pre_controls_patients > .05 and p_value_female_pre_controls_patients > .05 and p_value_male_post_controls_patients > .05 and p_value_female_post_controls_patients > .05):
    print("Levene's test is met in all data  (The variances are equal)")
    print(f"********** Gender can be used as a main factor for ANOVA in {population} **********")
# Check each p-value and print if it is less than 0.05
# List of p-values with descriptive names
p_values = [
    ("male vs female, pre-controls", p_value_male_female_pre_controls),
    ("male vs female, post-controls", p_value_male_female_post_controls),
    ("male vs female, pre-patients", p_value_male_female_pre_patients),
    ("male vs female, post-patients", p_value_male_female_post_patients),
    ("male, pre-controls vs pre-patients", p_value_male_pre_controls_patients),
    ("female, pre-controls vs pre-patients", p_value_female_pre_controls_patients),
    ("male, post-controls vs post-patients", p_value_male_post_controls_patients),
    ("female, post-controls vs post-patients", p_value_female_post_controls_patients)
]
# Check each p-value and print if it is less than 0.05
for description, p_value in p_values:
    if p_value < 0.05:
        print(f"Levene's test for {description} is rejected: (p-value = {p_value:.6f})")
    else:
        print(f"Levene's test for {description} is met: (p-value = {p_value:.6f})")
    
##############################################################################
# Check Normality:
# Don'T use shapiro wilk for larger datasets only for small database
# Use kolmogrov smirnov for larger datasets
# Perform Shapiro-Wilk tests and print results
# Null Hypothesis (H0): The data is drawn from a normal distribution.
# Alternative Hypothesis (H1): The data is not drawn from a normal distribution.
# Interpreting the p-value:
#     p-value ≤ α (commonly 0.05): If the p-value is less than or equal to the significance level (typically 0.05), 
# you reject the null hypothesis. This means there is sufficient evidence to say that the data does not follow a normal distribution.
#     p-value > α: If the p-value is greater than the significance level, you fail to reject the null hypothesis. 
# This suggests that there is not enough evidence to conclude that the data distribution is different from normal.
if len(df_disorder[df_disorder['time_point']==f'pre-{population}']) < 30:
    #  Use shapiro wilk for small database
    # For male controls (pre-time_point)
    stat_male_pre_controls, p_value_male_pre_controls = shapiro(male_pre_controls)
    if p_value <= .05:
        print(f"Shapiro-Wilk test for male controls (pre-time_point) is rejected (The data  does not look normal): (p-value = {p_value_male_pre_controls:.6f}")    
    else:
        print(f"Shapiro-Wilk test for male controls (pre-time_point) is met (The data  does not look normal): (p-value = {p_value_male_pre_controls:.6f}")

    # For male controls (post-time_point)
    stat_male_post_controls, p_value_male_post_controls = shapiro(male_post_controls)
    if p_value <= .05:
        print(f"Shapiro-Wilk test for male controls (post-time_point) is rejected (The data  does not look normal): (p-value = {p_value_male_post_controls:.6f}")    
    else:
        print(f"Shapiro-Wilk test for male controls (post-time_point) is met (The data  does not look normal): (p-value = {p_value_male_post_controls:.6f}")

    # For female controls (pre-time_point)
    stat_female_pre_controls, p_value_female_pre_controls = shapiro(female_pre_controls)
    if p_value <= .05:
        print(f"Shapiro-Wilk test for female controls (pre-time_point) is rejected (The data  does not look normal): (p-value = {p_value_female_pre_controls:.6f}")    
    else:
        print(f"Shapiro-Wilk test for female controls (pre-time_point) is met (The data  does not look normal): (p-value = {p_value_female_pre_controls:.6f}")
        
    # For female controls (post-time_point)
    stat_female_post_controls, p_value_female_post_controls = shapiro(female_post_controls)
    if p_value <= .05:
        print(f"Shapiro-Wilk test for female controls (post-time_point) is rejected (The data  does not look normal): (p-value = {p_value_female_post_controls:.6f}")
        
    else:
        print(f"Shapiro-Wilk test for female controls (post-time_point) is met (The data  does not look normal): (p-value = {p_value_female_post_controls:.6f}")

    # For male patients (pre-time_point)
    stat_male_pre_patients, p_value_male_pre_patients = shapiro(male_pre_patients)
    if p_value <= .05:
        print(f"Shapiro-Wilk test for male patients (pre-time_point) is rejected (The data  does not look normal): (p-value = {p_value_male_pre_patients:.6f}")    
    else:
        print(f"Shapiro-Wilk test for male patients (pre-time_point) is met (The data  does not look normal): (p-value = {p_value_male_pre_patients:.6f}")
        
    # For male patients (post-time_point)
    stat_male_post_patients, p_value_male_post_patients = shapiro(male_post_patients)
    if p_value <= .05:
        print(f"Shapiro-Wilk test for male patients (post-time_point) is rejected (The data  does not look normal): (p-value = {p_value_male_post_patients:.6f}")    
    else:
        print(f"Shapiro-Wilk test for male patients (post-time_point) is met (The data  does not look normal): (p-value = {p_value_male_post_patients:.6f}")

    # For female patients (pre-time_point)
    stat_female_pre_patients, p_value_female_pre_patients = shapiro(female_pre_patients)
    if p_value <= .05:
        print(f"Shapiro-Wilk test for male patients (pre-time_point) is rejected (The data  does not look normal): (p-value = {p_value_female_pre_patients:.6f}")    
    else:
        print(f"Shapiro-Wilk test for female patients (pre-time_point) is met (The data  does not look normal): (p-value = {p_value_female_pre_patients:.6f}")

    # For female patients (post-time_point)

    stat_female_post_patients, p_value_female_post_patients = shapiro(female_post_patients)
    if p_value <= .05:
        print(f"Shapiro-Wilk test for female patients (post-time_point) is rejected (The data  does not look normal): (p-value = {p_value_female_post_patients:.6f}")
    else:
        print(f"Shapiro-Wilk test for female patients (post-time_point) is met (The data  does not look normal): (p-value = {p_value_female_post_patients:.6f}")
        
    ####################
    if (p_value_male_pre_controls <= .05 or p_value_male_post_controls <= .05 or p_value_male_pre_patients <= .05 or p_value_male_post_patients <= .05 or
        p_value_female_pre_controls <= .05 or p_value_female_post_controls <= .05 or p_value_female_pre_patients <= .05 or p_value_female_post_patients <= .05):
        print(f"********** ANOVA cannot be used as Shapiro-Wilk's test is rejected (The data  does not look normal) **********")
    elif (p_value_male_pre_controls > .05 or p_value_male_post_controls > .05 or p_value_male_pre_patients > .05 or p_value_male_post_patients > .05 or
        p_value_female_pre_controls > .05 or p_value_female_post_controls > .05 or p_value_female_pre_patients > .05 or p_value_female_post_patients > .05):
        print(f"********** ANOVA can be used as Shapiro-Wilk's test is met in all data  (The data  does not look normal) **********")

    # Check each p-value and print if it is less than 0.05
    # List of p-values with descriptive names
    p_values = [
        ("male pre-controls", p_value_male_pre_controls),
        ("male post-controls", p_value_male_post_controls),
        ("male pre-patients", p_value_male_pre_patients),
        ("male post-patients", p_value_male_post_patients),
        ("female, pre-controls", p_value_female_pre_controls),
        ("female, post-controls", p_value_female_post_controls),
        ("female, pre-patients", p_value_female_pre_patients),
        ("female, post-patients", p_value_female_post_patients)
    ]
    # Check each p-value and print if it is less than 0.05
    for description, p_value in p_values:
        if p_value <= 0.05:
            print(f"Shapiro-Wilk's test for {description} is rejected: (p-value = {p_value:.6f})")
        else:
            print(f"Shapiro-Wilk's test for {description} is met: (p-value = {p_value:.6f})")
    ###############################################################################
elif len(df_disorder[df_disorder['time_point']==f'pre-{population}']) > 30:
    # Use kolmogrov smirnov (KS) for larger datasets
    # Hypotheses:
    #     Null Hypothesis (H0): The sample data comes from a normal distribution.
    #     Alternative Hypothesis (H1): The sample data does not come from a normal distribution.

    stat_male_pre_controls, p_value_male_pre_controls = kstest(male_pre_controls, "norm")
    if p_value <= .05:
        print(f"Kolmogorov-Smirnov test for male controls (pre-time_point) is rejected (The data  does not look normal): (p-value = {p_value_male_pre_controls:.6f}")    
    else:
        print(f"Kolmogorov-Smirnov test for male controls (pre-time_point) is met (The data  does not look normal): (p-value = {p_value_male_pre_controls:.6f}")

    # For male controls (post-time_point)
    stat_male_post_controls, p_value_male_post_controls = kstest(male_post_controls, stats.norm.cdf)
    if p_value <= .05:
        print(f"Kolmogorov-Smirnov test for male controls (post-time_point) is rejected (The data  does not look normal): (p-value = {p_value_male_post_controls:.6f}")    
    else:
        print(f"Kolmogorov-Smirnov test for male controls (post-time_point) is met (The data  does not look normal): (p-value = {p_value_male_post_controls:.6f}")

    # For female controls (pre-time_point)
    stat_female_pre_controls, p_value_female_pre_controls = kstest(female_pre_controls, stats.norm.cdf)
    if p_value <= .05:
        print(f"Kolmogorov-Smirnov test for female controls (pre-time_point) is rejected (The data  does not look normal): (p-value = {p_value_female_pre_controls:.6f}")    
    else:
        print(f"Kolmogorov-Smirnov test for female controls (pre-time_point) is met (The data  does not look normal): (p-value = {p_value_female_pre_controls:.6f}")
        
    # For female controls (post-time_point)
    stat_female_post_controls, p_value_female_post_controls = kstest(female_post_controls, stats.norm.cdf)
    if p_value <= .05:
        print(f"Kolmogorov-Smirnov test for female controls (post-time_point) is rejected (The data  does not look normal): (p-value = {p_value_female_post_controls:.6f}")
        
    else:
        print(f"Kolmogorov-Smirnov test for female controls (post-time_point) is met (The data  does not look normal): (p-value = {p_value_female_post_controls:.6f}")

    # For male patients (pre-time_point)
    stat_male_pre_patients, p_value_male_pre_patients = kstest(male_pre_patients, stats.norm.cdf)
    if p_value <= .05:
        print(f"Kolmogorov-Smirnov test for male patients (pre-time_point) is rejected (The data  does not look normal): (p-value = {p_value_male_pre_patients:.6f}")    
    else:
        print(f"Kolmogorov-Smirnov test for male patients (pre-time_point) is met (The data  does not look normal): (p-value = {p_value_male_pre_patients:.6f}")
        
    # For male patients (post-time_point)
    stat_male_post_patients, p_value_male_post_patients = kstest(male_post_patients, stats.norm.cdf)
    if p_value <= .05:
        print(f"Kolmogorov-Smirnov test for male patients (post-time_point) is rejected (The data  does not look normal): (p-value = {p_value_male_post_patients:.6f}")    
    else:
        print(f"Kolmogorov-Smirnov test for male patients (post-time_point) is met (The data  does not look normal): (p-value = {p_value_male_post_patients:.6f}")

    # For female patients (pre-time_point)
    stat_female_pre_patients, p_value_female_pre_patients = kstest(female_pre_patients, stats.norm.cdf)
    if p_value <= .05:
        print(f"Kolmogorov-Smirnov test for male patients (pre-time_point) is rejected (The data  does not look normal): (p-value = {p_value_female_pre_patients:.6f}")    
    else:
        print(f"Kolmogorov-Smirnov test for female patients (pre-time_point) is met (The data  does not look normal): (p-value = {p_value_female_pre_patients:.6f}")

    # For female patients (post-time_point)
    stat_female_post_patients, p_value_female_post_patients = shapiro(female_post_patients)
    if p_value <= .05:
        print(f"Kolmogorov-Smirnov test for female patients (post-time_point) is rejected (The data  does not look normal): (p-value = {p_value_female_post_patients:.6f}")
    else:
        print(f"Kolmogorov-Smirnov test for female patients (post-time_point) is met (The data  does not look normal): (p-value = {p_value_female_post_patients:.6f}")
        
        
    ####################
    if (p_value_male_pre_controls <= .05 or p_value_male_post_controls <= .05 or p_value_male_pre_patients <= .05 or p_value_male_post_patients <= .05 or
        p_value_female_pre_controls <= .05 or p_value_female_post_controls <= .05 or p_value_female_pre_patients <= .05 or p_value_female_post_patients <= .05):
        print(f"********** ANOVA cannot be used as Kolmogorov-Smirnov's test is rejected (The data  does not look normal) **********")
    elif (p_value_male_pre_controls > .05 or p_value_male_post_controls > .05 or p_value_male_pre_patients > .05 or p_value_male_post_patients > .05 or
        p_value_female_pre_controls > .05 or p_value_female_post_controls > .05 or p_value_female_pre_patients > .05 or p_value_female_post_patients > .05):
        print(f"********** ANOVA can be used as Kolmogorov-Smirnov's test is met in all data  (The data  does not look normal) **********")

    # Check each p-value and print if it is less than 0.05
    # List of p-values with descriptive names
    p_values = [
        ("male pre-controls", p_value_male_pre_controls),
        ("male post-controls", p_value_male_post_controls),
        ("male pre-patients", p_value_male_pre_patients),
        ("male post-patients", p_value_male_post_patients),
        ("female, pre-controls", p_value_female_pre_controls),
        ("female, post-controls", p_value_female_post_controls),
        ("female, pre-patients", p_value_female_pre_patients),
        ("female, post-patients", p_value_female_post_patients)
    ]
    # Check each p-value and print if it is less than 0.05
    for description, p_value in p_values:
        if p_value <= 0.05:
            print(f"Kolmogorov-Smirnov's test for {description} is rejected: (p-value = {p_value:.6f})")
        else:
            print(f"Kolmogorov-Smirnov's test for {description} is met: (p-value = {p_value:.6f})")
###############################################################################
##############################################################################
# Check Homogeneity of Variance to use ANOVA without Gender factor:
# Controls (pre-time_point) and (post-time_point):
df_control_pre = df_control[df_control['time_point']=='pre-control'][anova_target]
df_control_post = df_control[df_control['time_point']=='post-control'][anova_target]

# Patients (pre-time_point) and (post-time_point):
df_disorder_pre = df_disorder[df_disorder['time_point']==f'pre-{population}'][anova_target]
df_disorder_post = df_disorder[df_disorder['time_point']==f'post-{population}'][anova_target]

# Between controls (pre-time_point) and patients (pre-time_point):
stat_pre_control_patients, p_value_pre_control_patients = levene(df_control_pre, df_disorder_pre)
if p_value_pre_control_patients <= .05:
    print(f"Levene's test between controls and patients (pre-time_point) is rejected (The variances are different): (p-value = {p_value_pre_control_patients:.6f})")
else:
    print(f"Levene's test between controls and patients (pre-time_point) is met (The variances are equal): (p-value = {p_value_pre_control_patients:.6f})")
    
# Between controls (pre-time_point) and patients (pre-time_point):
stat_post_control_patients, p_value_post_control_patients = levene(df_control_post, df_disorder_post)
if p_value_post_control_patients <= .05:
    print(f"Levene's test between controls and patients (post-time_point) is rejected (The variances are different): (p-value = {p_value_post_control_patients:.6f})")
else:
    print(f"Levene's test between controls and patients (post-time_point) is met (The variances are equal): (p-value = {p_value_post_control_patients:.6f})")

####################
if (p_value_pre_control_patients <= .05 or p_value_post_control_patients <= .05):
    print("Levene's test is rejected (The variances are different)")
    print(f"********** ANOVA cannot be used between controls and patients groups without/ignoring Gender factor in {population} **********")
elif (p_value_pre_control_patients > .05 and p_value_post_control_patients > .05):
    print("Levene's test is met in all data (The variances are equal)")
    print(f"********** ANOVA can be used between controls and patients groups without/ignoring Gender factor in {population} **********")  
# Check each p-value and print if it is less than 0.05
# List of p-values with descriptive names
p_values = [
    ("controls vs patients, pre-controls", p_value_pre_control_patients),
    ("controls vs patients, post-controls", p_value_post_control_patients),
]
# Check each p-value and print if it is less than 0.05
for description, p_value in p_values:
    if p_value < 0.05:
        print(f"Levene's test for {description} is rejected: (p-value = {p_value:.6f})")
    else:
        print(f"Levene's test for {description} is met: (p-value = {p_value:.6f})")

# print("===== Done! End =====")
# embed(globals(), locals())
##############################################################################
# Check Normality:
# Don'T use shapiro wilk for larger datasets only for small database
# Use kolmogrov smirnov for larger datasets
# Perform Shapiro-Wilk tests and print results
# Null Hypothesis (H0): The data is drawn from a normal distribution.
# Alternative Hypothesis (H1): The data is not drawn from a normal distribution.
# Interpreting the p-value:
#     p-value ≤ α (commonly 0.05): If the p-value is less than or equal to the significance level (typically 0.05), 
# you reject the null hypothesis. This means there is sufficient evidence to say that the data does not follow a normal distribution.
#     p-value > α: If the p-value is greater than the significance level, you fail to reject the null hypothesis. 
# This suggests that there is not enough evidence to conclude that the data distribution is different from normal.
###############################
if len(df_disorder[df_disorder['time_point']==f'pre-{population}']) < 30:
    # Check Normality:
    # Perform Shapiro-Wilk test tests and print results
    # For Controls (pre-time_point):
    stat_control_pre, p_value_control_pre = shapiro(df_control_pre)
    if p_value <= .05:
        print(f"Shapiro-Wilk test for controls (pre-time_point) is rejected (The data  does not look normal): (p-value = {p_value_control_pre:.6f}")
    else:
        print(f"Shapiro-Wilk test for controls (pre-time_point) is met (The data looks normal): (p-value = {p_value_control_pre:.6f}")
        
    # For Controls (post-time_point):
    stat_control_post, p_value_control_post = shapiro(df_control_post)
    if p_value <= .05:
        print(f"Shapiro-Wilk test for controls (post-time_point) is rejected (The data  does not look normal): (p-value = {p_value_control_post:.6f}")
    else:
        print(f"Shapiro-Wilk test for controls (post-time_point) is met (The data looks normal): (p-value = {p_value_control_post:.6f}")
    
    # For Controls (post-time_point):
    stat_disorder_pre, p_value_disorder_pre = shapiro(df_disorder_pre)
    if p_value <= .05:
        print(f"Shapiro-Wilk test for patients (pre-time_point) is rejected (The data  does not look normal): (p-value = {p_value_disorder_pre:.6f}")
    else:
        print(f"Shapiro-Wilk test for patients (pre-time_point) is met (The data looks normal): (p-value = {p_value_disorder_pre:.6f}")

    stat_disorder_post, p_value_disorder_post = shapiro(df_disorder_post)
    if p_value <= .05:
            print(f"Shapiro-Wilk test for patients (post-time_point) is rejected (The data  does not look normal): (p-value = {p_value_disorder_post:.6f}")
    else:
        print(f"Shapiro-Wilk test for patients (post-time_point) is met (The data looks normal): (p-value = {p_value_disorder_post:.6f}")
     
     # Check each p-value and print if it is less than 0.05
    # List of p-values with descriptive names
    p_values = [
        ("pre-controls", p_value_control_pre),
        ("post-controls", p_value_control_post),
        ("pre-patients", p_value_disorder_pre),
        ("post-patients", p_value_disorder_post),
    ]
    # Check each p-value and print if it is less than 0.05
    for description, p_value in p_values:
        if p_value <= 0.05:
            print(f"Shapiro-Wilk's test for {description} is rejected: (p-value = {p_value:.6f})")
        else:
            print(f"Shapiro-Wilk's test for {description} is met: (p-value = {p_value:.6f})")  

elif len(df_disorder[df_disorder['time_point']==f'pre-{population}']) > 30:
    # Check Normality:
    # Perform Kolmogorov-Smirnov test tests and print results
    # For Controls (pre-time_point):
    stat_control_pre, p_value_control_pre = kstest(stats.norm.rvs(df_control_pre), stats.norm.cdf)
    if p_value <= .05:
        print(f"Kolmogorov-Smirnov test for controls (pre-time_point) is rejected (The data  does not look normal): (p-value = {p_value_control_pre:.6f}")
    else:
        print(f"Kolmogorov-Smirnov test for controls (pre-time_point) is met (The data looks normal): (p-value = {p_value_control_pre:.6f}")
        
    # For Controls (post-time_point):
    stat_control_post, p_value_control_post = kstest(stats.norm.rvs(df_control_post), stats.norm.cdf)
    if p_value <= .05:
        print(f"Kolmogorov-Smirnov test for controls (post-time_point) is rejected (The data  does not look normal): (p-value = {p_value_control_post:.6f}")
    else:
        print(f"Kolmogorov-Smirnov test for controls (post-time_point) is met (The data looks normal): (p-value = {p_value_control_post:.6f}")
    
    # For Controls (post-time_point):
    stat_disorder_pre, p_value_disorder_pre = kstest(stats.norm.rvs(df_disorder_pre), stats.norm.cdf)
    if p_value <= .05:
        print(f"Kolmogorov-Smirnov test for patients (pre-time_point) is rejected (The data  does not look normal): (p-value = {p_value_disorder_pre:.6f}")
    else:
        print(f"Kolmogorov-Smirnov test for patients (pre-time_point) is met (The data  looks normal): (p-value = {p_value_disorder_pre:.6f}")

    stat_disorder_post, p_value_disorder_post = kstest(stats.norm.rvs(df_disorder_post), stats.norm.cdf)
    if p_value <= .05:
            print(f"Kolmogorov-Smirnov test for patients (post-time_point) is rejected (The data  does not look normal): (p-value = {p_value_disorder_post:.6f}")
    else:
        print(f"Kolmogorov-Smirnov test for patients (post-time_point) is met (The data  looks normal): (p-value = {p_value_disorder_post:.6f}")
     
     # Check each p-value and print if it is less than 0.05
    # List of p-values with descriptive names
    p_values = [
        ("pre-controls", p_value_control_pre),
        ("post-controls", p_value_control_post),
        ("pre-patients", p_value_disorder_pre),
        ("post-patients", p_value_disorder_post),
    ]
    # Check each p-value and print if it is less than 0.05
    for description, p_value in p_values:
        if p_value <= 0.05:
            print(f"Kolmogorov-Smirnov's test for {description} is rejected: (p-value = {p_value:.6f})")
        else:
            print(f"Kolmogorov-Smirnov's test for {description} is met: (p-value = {p_value:.6f})")  

print("===== Done! End =====")
embed(globals(), locals())