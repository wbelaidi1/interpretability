numerical_col = [
    "issue_d",
    "loan duration",
    "bc_util",
    "delinq_2yrs",
    "dti",
    "funded_amnt",
    "inq_last_6mths",
    "int_rate",
    "mo_sin_rcnt_rev_tl_op",
    "mo_sin_rcnt_tl",
    "mort_acc",
    "mths_since_recent_bc",
    "num_actv_bc_tl",
    "num_bc_tl",
    "num_il_tl",
    "num_rev_accts",
    "open_acc",
    "pub_rec",
    "pub_rec_bankruptcies",
    "revol_util",
    "tax_liens",
    "Pct_afro_american",
    "log_annual_inc",
    "log_avg_cur_bal",
    "log_bc_open_to_buy",
    "log_fico_range_high",
    "log_mo_sin_old_rev_tl_op",
    "log_revol_bal",
    "grade_encoded",
    "emp_length_encoded",
    "sub_grade_encoded",
]
tgt = "target"
first_pred = ["Predictions", "Predicted probabilities"]
ohe = ["home_ownership"]
tgt_encoding = ["emp_title", "purpose", "zip_code"]
grd = "grade"
grd_sub = "sub_grade"
emp_len = "emp_length"
log_col = [
    "annual_inc",
    "avg_cur_bal",
    "bc_open_to_buy",
    "fico_range_high",
    "mo_sin_old_rev_tl_op",
    "revol_bal",
]

grade_dico = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 6}

emp_lenght_dico = {
    "< 1 year": 0,
    "1 year": 1,
    "2 years": 2,
    "3 years": 3,
    "4 years": 4,
    "5 years": 5,
    "6 years": 6,
    "7 years": 7,
    "8 years": 8,
    "9 years": 9,
    "10+ years": 9,
}

sub_grade_dico = {
    "A1": 1,
    "A2": 2,
    "A3": 3,
    "A4": 4,
    "A5": 5,
    "B1": 6,
    "B2": 7,
    "B3": 8,
    "B4": 9,
    "B5": 10,
    "C1": 11,
    "C2": 12,
    "C3": 13,
    "C4": 14,
    "C5": 15,
    "D1": 16,
    "D2": 17,
    "D3": 18,
    "D4": 19,
    "D5": 20,
    "E1": 21,
    "E2": 22,
    "E3": 23,
    "E4": 24,
    "E5": 25,
    "F1": 26,
    "F2": 27,
    "F3": 28,
    "F4": 29,
    "F5": 30,
    "G1": 31,
    "G2": 32,
    "G3": 33,
    "G4": 34,
    "G5": 35,
}

best_params = {
    "max_depth": 4,
    "min_child_weight": 9,
    "gamma": 0.00790867295350553,
    "subsample": 0.8596454064316353,
    "colsample_bytree": 0.7619123188731923,
    "learning_rate": 0.03373184445484679,
    "n_estimators": 1000,
    "reg_alpha": 0.8981291134257546,
    "reg_lambda": 0.4813558491685588,
}
