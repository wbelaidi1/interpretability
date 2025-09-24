import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

from utils import (
    log_col,
    grd,
    emp_len,
    grd_sub,
    grade_dico,
    emp_lenght_dico,
    sub_grade_dico,
    col_to_encode,
)


def load_and_preprocess_data():
    df = pd.read_csv("data/data.csv", index_col=0)

    df = df.rename({"loan duration": "loan_duration"})

    for col in log_col:
        df[f"log_{col}"] = np.log1p(df[col])

    df[f"{grd}_encoded"] = df[grd].map(grade_dico)
    df[f"{emp_len}_encoded"] = df[emp_len].map(emp_lenght_dico)
    df[f"{grd_sub}_encoded"] = df[grd_sub].map(sub_grade_dico)

    lb = LabelEncoder()
    for col in col_to_encode:
        df[f"{col}_encoded"] = lb.fit_transform(df[col])

    df = df.drop(columns=log_col + col_to_encode + [grd, grd_sub, emp_len])

    return df
