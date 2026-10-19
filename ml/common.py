"""Shared data generation + feature engineering (used by training AND serving to avoid skew)."""
import numpy as np, pandas as pd

TARGET = "readmitted_30d"
DISPOSITIONS = ["Home", "Home Health", "SNF", "Rehab", "Other"]
DX_CATEGORIES = ["Circulatory", "Respiratory", "Digestive", "Injury", "Endocrine", "Infectious", "Neoplasms", "Other"]


def generate_dataset(n=12000, seed=42):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame()
    df["age"] = rng.randint(18, 95, n)
    df["gender"] = rng.choice(["Male", "Female"], n, p=[0.48, 0.52])
    df["length_of_stay"] = rng.poisson(4.5, n).clip(1, 40)
    df["emergency_admission"] = rng.binomial(1, 0.55, n)
    df["num_diagnoses"] = rng.poisson(5.5, n).clip(1, 20)
    df["num_procedures"] = rng.poisson(1.8, n).clip(0, 12)
    df["num_medications"] = rng.poisson(8.0, n).clip(0, 40)
    df["prior_admissions_1yr"] = rng.poisson(0.9, n).clip(0, 15)
    df["prior_ed_visits_1yr"] = rng.poisson(1.4, n).clip(0, 20)
    df["comorbidity_score"] = rng.poisson(2.2, n).clip(0, 12)
    df["abnormal_creatinine"] = rng.binomial(1, 0.18, n)
    df["abnormal_hba1c"] = rng.binomial(1, 0.22, n)
    df["abnormal_hemoglobin"] = rng.binomial(1, 0.25, n)
    df["discharge_disposition"] = rng.choice(DISPOSITIONS, n, p=[0.55, 0.15, 0.15, 0.08, 0.07])
    df["primary_dx_category"] = rng.choice(DX_CATEGORIES, n, p=[0.22, 0.14, 0.12, 0.10, 0.09, 0.08, 0.07, 0.18])
    risk = ((df.age - 50) / 40 * 0.6 + df.length_of_stay / 15 * 0.5 + df.num_diagnoses / 10 * 0.7
            + df.num_medications / 20 * 0.5 + df.prior_admissions_1yr * 0.55 + df.prior_ed_visits_1yr * 0.25
            + df.comorbidity_score * 0.35 + df.emergency_admission * 0.45 + df.abnormal_creatinine * 0.4
            + df.abnormal_hemoglobin * 0.3 + df.discharge_disposition.isin(["SNF", "Rehab"]).astype(int) * 0.5
            + rng.normal(0, 0.7, n))
    prob = 1 / (1 + np.exp(-(risk - 2.2)))
    df[TARGET] = (rng.rand(n) < prob).astype(int)
    for c in ["comorbidity_score", "num_medications"]:
        df.loc[rng.rand(n) < 0.025, c] = np.nan
    return df


def engineer(df, comorb_median=2.0):
    d = df.copy()
    d["age_group"] = pd.cut(d["age"], [0, 40, 60, 75, 120], labels=["Young", "Middle", "Senior", "Elderly"]).astype(str)
    d["high_utiliser"] = ((d.prior_admissions_1yr >= 2) | (d.prior_ed_visits_1yr >= 3)).astype(int)
    d["polypharmacy"] = (d.num_medications.fillna(0) >= 10).astype(int)
    d["complex_case"] = ((d.num_diagnoses >= 7) & (d.num_procedures >= 2)).astype(int)
    d["lab_abnormal_count"] = d.abnormal_creatinine + d.abnormal_hba1c + d.abnormal_hemoglobin
    d["risk_composite"] = (d.prior_admissions_1yr * 0.6 + d.comorbidity_score.fillna(comorb_median) * 0.4
                           + d.length_of_stay * 0.15 + d.emergency_admission * 0.5 + d.lab_abnormal_count * 0.35)
    return d
