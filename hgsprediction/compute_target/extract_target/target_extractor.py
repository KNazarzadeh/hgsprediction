#!/usr/bin/env Disorderspredwp3
"""Extract the target from populations data."""

# Authors: Kimia Nazarzadeh <k.nazarzadeh@fz-juelich.de>

import pandas as pd
# from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

def target_extractor(df, target):

    # filter_col = [col for col in df.columns if f"{target}" in col][0]
    # parts = filter_col.split('-')
    # if len(parts) == 2 and parts[1] in ["0.0"]:
    #     target = parts[0]
    
    target = [col for col in df.columns if f"{target}" in col][0]
    
    return target