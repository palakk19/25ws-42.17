import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
from sklearn.feature_selection import mutual_info_classif

# --- CONFIGURATION ---
input_folder = 'dataset_csv'
output_folder = 'output/final_summary_plots'
top_n_to_plot = 15

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 1. LOAD TARGET (DIABETES)
print("--- Step 1: Loading Target ---")
try:
    target_path = glob.glob(os.path.join(input_folder, '*Diabetes*.csv'))[0]
    df_target = pd.read_csv(target_path, usecols=['SEQN', 'DIQ010'])
    
    # Strict Binary: 1=Yes, 2=No. Drop everything else (Borderline, Refused).
    df_target = df_target[df_target['DIQ010'].isin([1, 2])]
    df_target['Diabetes'] = df_target['DIQ010'].replace({2: 0}) # 1=Yes, 0=No
    df_target = df_target[['SEQN', 'Diabetes']]
except IndexError:
    print(" ERROR: Could not find DIQ file.")
    exit()

# 2. SCANNING LOOP
print("\n--- Step 2: Rigorous Scanning (Using Mutual Information) ---")
candidates = []
all_files = glob.glob(os.path.join(input_folder, "*.csv"))

for filepath in all_files:
    filename = os.path.basename(filepath)
    # if "DIQ" in filename: continue 

    try:
        df_scan = pd.read_csv(filepath)
        if 'SEQN' not in df_scan.columns: continue
        
        merged = pd.merge(df_target, df_scan, on='SEQN', how='inner')
        if len(merged) < 500: continue 

        for col in merged.columns:
            if col in ['SEQN', 'Diabetes', 'DIQ010']: continue
            if not pd.api.types.is_numeric_dtype(merged[col]): continue

            # --- STRICT CLEANING LOGIC ---
            unique_count = merged[col].nunique()
            
            # TYPE A: CATEGORICAL (< 15 unique values)
            if unique_count < 15:
                # Drop garbage codes (7, 9, 77, 99)
                valid_data = merged[~merged[col].isin([7, 9, 77, 99])]
                is_categorical = True
                
            # TYPE B: CONTINUOUS (> 15 unique values)
            else:
                # Drop garbage codes (like 9999)
                valid_data = merged[merged[col] < 7777]
                
                # --- NEW: Outlier Capping for cleaner plots ---
                # Cap extremely high values at the 99th percentile
                limit = valid_data[col].quantile(0.99)
                valid_data = valid_data[valid_data[col] <= limit]
                is_categorical = False
            
            if len(valid_data) < 100: continue

            # --- CALCULATE MUTUAL INFORMATION ---
            X = valid_data[[col]]
            y = valid_data['Diabetes']
            
            # Discrete_features=True handles categorical data correctly
            mi_score = mutual_info_classif(X, y, discrete_features=[is_categorical], random_state=42)[0]
            
            if mi_score > 0.005: 
                candidates.append({
                    'File': filename,
                    'Column': col,
                    'Score': mi_score,
                    'Is_Cat': is_categorical,
                    'Data': valid_data[[col, 'Diabetes']]
                })
    except Exception as e:
        continue



# 3. RANKING
if not candidates:
    print("No significant variables found.")
    exit()

candidates_df = pd.DataFrame(candidates)
top_picks = candidates_df.sort_values(by='Score', ascending=False).head(top_n_to_plot).reset_index(drop=True)

print(f"\n--- Top {len(top_picks)} Predictors Found ---")


codebook = {
    'HUQ010': 'General_Health',
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


top_picks['Description'] = top_picks['Column'].apply(lambda x: codebook.get(x, x))

#  PLOT 1: THE LEADERBOARD
plt.figure(figsize=(12, 8)) 
sns.barplot(x='Score', y='Description', data=top_picks, palette='viridis')

plt.title(f'Top {top_n_to_plot} Predictors by Mutual Information Score', fontsize=14)
plt.xlabel('Mutual Information Score (Higher indicates stronger predictive power)', fontsize=12)
plt.ylabel('Variable Description (Questionnaire Item)', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=11) 

plt.tight_layout()
plt.savefig(os.path.join(output_folder, '1_Leaderboard_Scores_Readable.png'))
print(" Saved readable leaderboard plot.")


# --- PLOT 2: THE EVIDENCE GRID 
rows = 5
cols = 3
fig, axes = plt.subplots(rows, cols, figsize=(15, 22))
axes = axes.flatten()

for i, ax in enumerate(axes):
    if i < len(top_picks):
        row = top_picks.iloc[i]
        col_name = row['Column']
        desc = row['Description'] 
        file_name = row['File']
        data = row['Data']
        is_cat = row['Is_Cat']
        
        if is_cat:
            ct = pd.crosstab(data[col_name], data['Diabetes'], normalize='index')
            ct.plot(kind='bar', stacked=True, color=['green', 'red'], ax=ax, legend=False)
            ax.set_ylabel("Risk Proportion")
        else:
            sns.boxplot(x='Diabetes', y=col_name, data=data, palette="Set2", ax=ax)
            ax.set_ylabel(col_name) 

        from textwrap import wrap
        title_text = f"#{i+1}: {desc}\nCode: {col_name} ({file_name})"
        ax.set_title("\n".join(wrap(title_text, 40)), fontsize=9)
        
        ax.set_xlabel("") 
        
    else:
        ax.axis('off')

handles, labels = ax.get_legend_handles_labels()
if handles:
    fig.legend(handles, ['Healthy', 'Diabetic'], loc='upper center', ncol=2, fontsize=12)

plt.tight_layout()
plt.subplots_adjust(top=0.92, hspace=0.4) 
plt.savefig(os.path.join(output_folder, '2_Evidence_Grid_Readable.png'))
print(" Saved readable evidence grid.")