import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns


input_folder = 'dataset_csv'
output_folder = 'output/master_eda_visuals'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


rename_map = {
    'BPQ020': 'High_BP',
    'PAD810Q': 'Vigorous_Activity_Freq',
    'AUQ054': 'Hearing_Difficulty',
    'WHD020': 'Current_Weight(lbs)',
    'WHD050': 'Weight_1yr_Ago(lbs)',
    'PAD820': 'Vigorous_Activity_Duration',
    'KIQ022': 'Weak_Kidneys',
    'PAD790Q': 'Sedentary_Time(minutes)',
    'IMQ011': 'HepA_Vaccine',
    'SLD012': 'Sleep_Hours',
    'INDFMMPI': 'Poverty_Index',
    'WHD010': 'Height(inches)'
}


categorical_vars = ['High_BP', 'Hearing_Difficulty', 'Weak_Kidneys', 'HepA_Vaccine', 'Vigorous_Activity_Freq']
continuous_vars = ['Current_Weight(lbs)', 'Sleep_Hours', 'Poverty_Index', 'Height(inches)']


# PART 1: MAIN PREDICTIVE DATASET ANALYSIS
print("--- Part 1: Analyzing Predictor Variables (Merged Data) ---")

# 1. Load Target (Diabetes) for Merging
try:
    target_path = glob.glob(os.path.join(input_folder, '*Diabetes*.csv'))[0]
    df = pd.read_csv(target_path, usecols=['SEQN', 'DIQ010'])
    df = df[df['DIQ010'].isin([1, 2])] # Keep only Yes/No
    df['Condition'] = df['DIQ010'].replace({1: 'Diabetic', 2: 'Healthy'})
    df = df[['SEQN', 'Condition']]
except IndexError:
    print(" Error: Could not find Diabetes file (DIQ).")
    exit()

# 2. Merge Predictors
all_files = glob.glob(os.path.join(input_folder, "*.csv"))
for col_code, col_name in rename_map.items():
    for filepath in all_files:
        if "DIQ" in filepath: continue
        if col_code in pd.read_csv(filepath, nrows=0).columns:
            df_temp = pd.read_csv(filepath, usecols=['SEQN', col_code])
            df = pd.merge(df, df_temp, on='SEQN', how='left')
            df = df.rename(columns={col_code: col_name})
            break


df_plot = df.copy()
for col in continuous_vars:
    if col in df_plot.columns:
        df_plot[col] = df_plot[col].replace({7777: np.nan, 9999: np.nan})
for col in categorical_vars:
    if col in df_plot.columns:
        df_plot[col] = df_plot[col].replace({7: np.nan, 9: np.nan, 77: np.nan, 99: np.nan})

print(f"Merged Data Shape: {df.shape}")

# --- PLOT 1: CONTINUOUS KDE GRID ---
print("Generating 1. Continuous KDE Grid...")
valid_cont_vars = [v for v in continuous_vars if v in df_plot.columns]
if valid_cont_vars:
    rows = (len(valid_cont_vars) // 2) + (len(valid_cont_vars) % 2)
    fig, axes = plt.subplots(rows, 2, figsize=(14, 5 * rows))
    axes = axes.flatten()
    for i, col in enumerate(valid_cont_vars):
        sns.kdeplot(data=df_plot[df_plot['Condition']=='Healthy'], x=col, label='Healthy', shade=True, color='blue', ax=axes[i])
        sns.kdeplot(data=df_plot[df_plot['Condition']=='Diabetic'], x=col, label='Diabetic', shade=True, color='red', ax=axes[i])
        axes[i].set_title(f'Distribution of {col}')
        axes[i].legend()
    for j in range(i + 1, len(axes)): axes[j].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '1_Predictors_KDE_Grid.png'))
    plt.close()

# --- PLOT 1.5: CONTINUOUS VIOLIN GRID 
print("Generating 1.5. Continuous Violin Grid...")
if valid_cont_vars:
    rows = (len(valid_cont_vars) // 2) + (len(valid_cont_vars) % 2)
    fig, axes = plt.subplots(rows, 2, figsize=(14, 5 * rows))
    axes = axes.flatten()
    for i, col in enumerate(valid_cont_vars):
        sns.violinplot(x='Condition', y=col, data=df_plot, split=True, palette='muted', ax=axes[i])
        axes[i].set_title(f'Violin Plot: {col}')
        axes[i].set_xlabel('')
    for j in range(i + 1, len(axes)): axes[j].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '1.5_Predictors_Violin_Grid.png'))
    plt.close()

# --- PLOT 2: CATEGORICAL BAR GRID ---
print("Generating 2. Categorical Bar Grid...")
valid_cat_vars = [v for v in categorical_vars if v in df_plot.columns]
if valid_cat_vars:
    cols = 3
    rows = (len(valid_cat_vars) // cols) + (1 if len(valid_cat_vars) % cols > 0 else 0)
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    axes = axes.flatten()
    for i, col in enumerate(valid_cat_vars):
        temp_df = df_plot.copy()
        temp_df = temp_df[temp_df[col].isin([1, 2])]
        temp_df[col] = temp_df[col].replace({1: 'Yes', 2: 'No'})
        props = (temp_df.groupby('Condition')[col].value_counts(normalize=True).rename('pct').mul(100).reset_index())
        sns.barplot(x='Condition', y='pct', hue=col, data=props, palette='Set2', ax=axes[i])
        axes[i].set_title(f'{col}')
        axes[i].set_ylim(0, 100)
        axes[i].set_ylabel('% within Group')
        for p in axes[i].patches:
            if p.get_height() > 0:
                axes[i].text(p.get_x() + p.get_width()/2, p.get_height()+1, f'{p.get_height():.1f}%', ha='center', fontsize=9)
    for j in range(i+1, len(axes)): axes[j].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '2_Predictors_Categorical_Grid.png'))
    plt.close()

# --- PLOT 3: ROBUST CORRELATION MATRIX 
print("Generating 3. Correlation Matrix (Robust)...")

# 1. Create a fresh copy for correlation to avoid pollution from string replacements
df_corr = df.copy()

# 2. Add numeric condition
df_corr['Diabetes'] = df_corr['Condition'].replace({'Diabetic': 1, 'Healthy': 0})

# 3. Define the EXACT list of columns we want to see
cols_to_corr = ['Diabetes'] + list(rename_map.values())

# 4. FORCE NUMERIC CONVERSION (This fixes missing columns)
for col in cols_to_corr:
    if col in df_corr.columns:
        df_corr[col] = pd.to_numeric(df_corr[col], errors='coerce')

# 5. Filter to only columns that exist
valid_corr_cols = [c for c in cols_to_corr if c in df_corr.columns]

# 6. Plot
plt.figure(figsize=(14, 12))
sns.heatmap(df_corr[valid_corr_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Complete Feature Correlation Matrix')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, '3_Correlation_Matrix.png'))
plt.close()

# =========================================
# PART 2: DEEP DIVE INTO DIABETES.CSV
# =========================================
print("\n--- Part 2: Deep Dive into Diabetes Target File ---")

# Reload Diabetes File specifically to get all hidden columns
df_dia = pd.read_csv(target_path)
dia_rename = {
    'DIQ010': 'Status',
    'DID040': 'Age_Diagnosis',
    'DIQ160': 'Prediabetes',
    'DIQ050': 'Taking_Insulin',
    'DIQ070': 'Taking_Pills'
}
df_dia = df_dia.rename(columns=dia_rename)

# Map Values for Plotting
df_dia['Status_Label'] = df_dia['Status'].replace({1: 'Yes', 2: 'No', 3: 'Borderline', 9: 'Unknown'})
yes_no_map = {1: 'Yes', 2: 'No', 7: 'Refused', 9: 'Unknown'}
for c in ['Prediabetes', 'Taking_Insulin', 'Taking_Pills']:
    if c in df_dia.columns:
        df_dia[c] = df_dia[c].replace(yes_no_map)

# --- PLOT 4: TARGET IMBALANCE (Pie + Bar) ---
print("Generating 4. Target Status Distribution...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
counts = df_dia['Status_Label'].value_counts()
axes[0].pie(counts, labels=counts.index, autopct='%1.1f%%', colors=sns.color_palette('pastel'), startangle=90)
axes[0].set_title('Diabetes Status (Includes Borderline)')
sns.countplot(x='Status_Label', data=df_dia, palette='pastel', ax=axes[1])
for p in axes[1].patches:
    axes[1].annotate(f'{int(p.get_height())}', (p.get_x() + 0.35, p.get_height() + 50))
axes[1].set_title('Patient Counts')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, '4_Target_DeepDive.png'))
plt.close()

# --- PLOT 5: AGE OF DIAGNOSIS (Histogram) ---
print("Generating 5. Age of Diagnosis...")
plt.figure(figsize=(10, 6))
sns.histplot(df_dia['Age_Diagnosis'].dropna(), kde=True, bins=30, color='purple')
plt.title('At what Age are people diagnosed?')
plt.xlabel('Age (Years)')
plt.axvline(df_dia['Age_Diagnosis'].median(), color='red', linestyle='--', label=f'Median: {df_dia["Age_Diagnosis"].median()}')
plt.legend()
plt.savefig(os.path.join(output_folder, '5_Age_Diagnosis.png'))
plt.close()

# --- PLOT 6: TREATMENT ANALYSIS (Insulin vs Pills) ---
print("Generating 6. Treatment Analysis...")
diabetics_only = df_dia[df_dia['Status_Label'] == 'Yes']
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.countplot(x='Taking_Insulin', data=diabetics_only, ax=axes[0], palette='Set2')
axes[0].set_title('Insulin Use (Diabetics Only)')
sns.countplot(x='Taking_Pills', data=diabetics_only, ax=axes[1], palette='Set2')
axes[1].set_title('Pill Use (Diabetics Only)')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, '6_Treatment_Analysis.png'))
plt.close()

# --- PLOT 7: PREDIABETES AWARENESS ---
print("Generating 7. Prediabetes Awareness...")
non_diabetics = df_dia[df_dia['Status_Label'] == 'No']
plt.figure(figsize=(8, 6))
sns.countplot(x='Prediabetes', data=non_diabetics, palette='coolwarm')
plt.title('Prediabetes Awareness (Among "Healthy" Patients)')
plt.savefig(os.path.join(output_folder, '7_Prediabetes_Awareness.png'))
plt.close()

print(f"\n SUCCESS! All 7 Visual sets saved in '{output_folder}'.")