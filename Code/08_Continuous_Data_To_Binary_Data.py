# 08_Continuous_Data_To_Binary_Data.py

import os
import pandas as pd
import numpy as np

# -----------------------------
# 0. Output directory
# -----------------------------
output_dir = r"<output_path>"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# 1. Load raw Excel dataset
# -----------------------------
df = pd.read_excel(r"<file_path>")

# -----------------------------
# 1b. Replace empty/blank cells with NaN
# -----------------------------
df = df.replace(r'^\s*$', np.nan, regex=True)

# -----------------------------
# 2. Harmonize column types
# -----------------------------
binary_cols = ["HLA-B27","ANA","Anti-Ro","Anti-La","Anti-dsDNA","Anti-Sm"]

# 1. Replace empty strings or pure whitespace with NaN
df[binary_cols] = df[binary_cols].replace(r'^\s*$', np.nan, regex=True)

# 2. Map "Positive" -> 1, "Negative" -> 0, keep NaN for missing
df[binary_cols] = df[binary_cols].apply(lambda x: x.str.strip().str.lower().map({'positive':1,'negative':0}))

# Now df[binary_cols] contains 1, 0, or NaN

continuous_cols = ["Age","ESR","CRP","RF","Anti-CCP","C3","C4"]
for col in continuous_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['Gender'] = df['Gender'].str.strip().str.lower().map({'male':'M','female':'F'})
df['Gender'] = df['Gender'].astype('category')
df['Disease'] = df['Disease'].astype('category')

# -----------------------------
# 3. Apply clinical cutoffs (binary abnormal flags)
# -----------------------------
# ESR
df['ESR_abn'] = np.where(
    df['ESR'].isna(),
    np.nan,
    np.where(
        ((df['Gender']=='M') & (df['ESR']>15)) |
        ((df['Gender']=='F') & (df['ESR']>20)),
        1, 0
    )
)

# CRP
df['CRP_abn'] = np.where(
    df['CRP'].isna(),
    np.nan,
    np.where(df['CRP']>3.0, 1, 0)
)

# RF
df['RF_abn'] = np.where(
    df['RF'].isna(),
    np.nan,
    np.where(df['RF']>3.0, 1, 0)
)

# Anti-CCP
df['AntiCCP_abn'] = np.where(
    df['Anti-CCP'].isna(),
    np.nan,
    np.where(df['Anti-CCP']>20, 1, 0)
)

# C3
df['C3_abn'] = np.where(
    df['C3'].isna(),
    np.nan,
    np.where(
        ((df['Gender']=='M') & ((df['C3']<90) | (df['C3']>180))) |
        ((df['Gender']=='F') & ((df['C3']<88) | (df['C3']>206))),
        1, 0
    )
)

# C4
df['C4_abn'] = np.where(
    df['C4'].isna(),
    np.nan,
    np.where(
        ((df['Gender']=='M') & ((df['C4']<12) | (df['C4']>72))) |
        ((df['Gender']=='F') & ((df['C4']<13) | (df['C4']>75))),
        1, 0
    )
)

# -----------------------------
# 4. Missingness report
# -----------------------------
missing_pct = df.isna().mean() * 100
miss_report = pd.DataFrame({'Variable': missing_pct.index, 'Percent_Missing': missing_pct.values})
miss_report.to_excel(os.path.join(output_dir,"missingness_report.xlsx"), index=False)
print(f"Missingness report saved to {output_dir}")

# -----------------------------
# 5. Demographics summary
# -----------------------------
demo_summary = df.groupby('Disease', observed=True).agg(
    N=('Age', 'count'),
    Age_median=('Age', 'median'),
    Age_IQR=('Age', lambda x: x.quantile(0.75) - x.quantile(0.25)),
    Male_pct=('Gender', lambda x: (x=='M').mean()*100),
    Female_pct=('Gender', lambda x: (x=='F').mean()*100)
).reset_index()
demo_summary.to_excel(os.path.join(output_dir,"demographics_summary.xlsx"), index=False)

# -----------------------------
# 6. Marker prevalence summary
# -----------------------------
marker_cols = binary_cols + ['ESR_abn','CRP_abn','RF_abn','AntiCCP_abn','C3_abn','C4_abn']
prevalence = df.groupby('Disease', observed=True)[marker_cols].mean().multiply(100).reset_index()
prevalence.to_excel(os.path.join(output_dir,"marker_prevalence.xlsx"), index=False)

# -----------------------------
# 7. Save cleaned dataset + separate male/female
# -----------------------------
df.to_excel(os.path.join(output_dir,"cleaned_dataset.xlsx"), index=False)

df_male = df[df['Gender']=='M'].copy()
df_female = df[df['Gender']=='F'].copy()

df_male.to_excel(os.path.join(output_dir,"cleaned_dataset_male.xlsx"), index=False)
df_female.to_excel(os.path.join(output_dir,"cleaned_dataset_female.xlsx"), index=False)

print(f"Cleaned dataset saved to {output_dir}")
print("Male and Female datasets saved separately.")

