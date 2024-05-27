import math
import sys
import os
import numpy as np
import pandas as pd
from hgsprediction.load_data.load_healthy_extracted_data_by_features import load_extracted_data_by_features

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())


filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
feature_type = sys.argv[3]
target = sys.argv[4]
confound_status = sys.argv[5]
gender = sys.argv[6]


df = load_extracted_data_by_features(
    population,
    mri_status,
    confound_status,
    gender,
    feature_type,
    target,
)


print("===== Done! =====")
embed(globals(), locals())
