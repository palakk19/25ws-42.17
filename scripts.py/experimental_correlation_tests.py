import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay, classification_report, accuracy_score, f1_score

# --- IMPORT MODELS ---
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


input_folder = 'dataset_csv'
output_folder = 'output/final_rigorous_project'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


rename_map = {
    'BPQ020': 'High_BP',
    'PAD810Q': 'Vigorous_Activity_Freq',
    'AUQ054': 'Hearing_Difficulty',
    'WHD020': 'Current_Weight(lbs)',
    
    'PAD820': 'Vigorous_Activity_Duration',
    'KIQ022': 'Weak_Kidneys',
    
    'IMQ011': 'HepA_Vaccine',
    'SLD012': 'Sleep_Hours',
    'INDFMMPI': 'Poverty_Index',
    'WHD010': 'Height(inches)'
}
target_columns = list(rename_map.keys())


# STEP 1: LOAD & MERGE
print("--- Step 1: Loading Data ---")
try:
    target_path = glob.glob(os.path.join(input_folder, '*Diabetes*.csv'))[0]
    df = pd.read_csv(target_path, usecols=['SEQN', 'DIQ010'])
    df = df[df['DIQ010'].isin([1, 2])]
    df['Diabetes'] = df['DIQ010'].replace({2: 0})
    df = df[['SEQN', 'Diabetes']]
except:
    print(" Critical Error: DIQ (Target) file not found.")
    exit()

all_files = glob.glob(os.path.join(input_folder, "*.csv"))
for col in target_columns:
    for filepath in all_files:
        # if "DIQ" in filepath: continue
        if col in pd.read_csv(filepath, nrows=0).columns:
            df_temp = pd.read_csv(filepath, usecols=['SEQN', col])
            df = pd.merge(df, df_temp, on='SEQN', how='left')
            break

df = df.rename(columns=rename_map)
print(f"Initial Shape: {df.shape}")


# STEP 2: VISUALIZATION & CLEANING
print("\n--- Step 2: Rigorous Cleaning & Visualization ---")

# 1. Drop Rows with No Target first
df = df.dropna(subset=['Diabetes'])

# 2. CALCULATE & PLOT MISSINGNESS
threshold_col = 0.60 # 60% Cutoff
missing_pct = df.isnull().mean() * 100
plot_data = missing_pct.drop(['SEQN', 'Diabetes'], errors='ignore').sort_values(ascending=False)

plt.figure(figsize=(12, 6))
colors = ['red' if x > (threshold_col * 100) else 'steelblue' for x in plot_data.values]
sns.barplot(x=plot_data.index, y=plot_data.values, palette=colors)
plt.axhline(threshold_col * 100, color='black', linestyle='--', linewidth=2, label=f'Drop Cutoff ({threshold_col*100}%)')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Percentage of Missing Values')
plt.title('Missing Data Analysis: Red Bars will be Dropped')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, '1_Missing_Data_RedZone.png'))
plt.show() 

# 3. ACTUALLY DROP THE COLUMNS
cols_to_drop = plot_data[plot_data > (threshold_col * 100)].index.tolist()
if cols_to_drop:
    print(f" Dropping {len(cols_to_drop)} columns (> {threshold_col*100}% missing): {cols_to_drop}")
    df = df.drop(columns=cols_to_drop)
else:
    print(" No columns dropped.")

# 4. DROP PATIENTS (> 80% Missing Answers) 
threshold_row = 0.80
rows_before = len(df)
df['missing_pct'] = df.isnull().mean(axis=1)
df = df[df['missing_pct'] <= threshold_row].drop(columns=['missing_pct'])
rows_dropped = rows_before - len(df)

if rows_dropped > 0:
    print(f" Dropped {rows_dropped} patients (> {threshold_row*100}% missing answers).")
else:
    print(f"No patients were dropped (All had < {threshold_row*100}% missing data).")

# 5. PRE-CLEAN GARBAGE CODES (Before KNN)
cols_numeric = [c for c in df.columns if c not in ['SEQN', 'Diabetes']]
categorical_cols = [] 

for col in cols_numeric:
    if df[col].nunique() < 15: # Categorical
        categorical_cols.append(col)
        df[col] = df[col].replace({7: np.nan, 9: np.nan, 77: np.nan, 99: np.nan})
    else: # Continuous
        df[col] = df[col].replace({7777: np.nan, 9999: np.nan})

# 6. RUN KNN IMPUTATION
print(f"• Running KNN Imputation on {len(df)} patients...")
X_missing = df[cols_numeric]
target = df[['SEQN', 'Diabetes']]

imputer = KNNImputer(n_neighbors=5)
X_imputed_array = imputer.fit_transform(X_missing)
X_imputed = pd.DataFrame(X_imputed_array, columns=cols_numeric, index=df.index)

# Round Categorical columns back to integers
for col in categorical_cols:
    if col in X_imputed.columns:
        X_imputed[col] = X_imputed[col].round()

df = pd.concat([target, X_imputed], axis=1)
print(f"• Final Cleaned Shape: {df.shape}")


# STEP 3: TRAIN & GENERATE REPORTS
print("\n--- Step 3: Training Models ---")
X = df.drop(['SEQN', 'Diabetes'], axis=1)
y = df['Diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN Classifier": KNeighborsClassifier(n_neighbors=5),
    "SVM (Linear)": SVC(kernel="linear", probability=True),
    "Decision Tree": DecisionTreeClassifier(max_depth=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "LDA": LinearDiscriminantAnalysis()
}

results = []
print(f"{'MODEL NAME':<25} | {'ACC':<6} | {'AUC':<6} | {'F1-SCORE'}")
print("-" * 60)

for name, model in models.items():
    model.fit(X_train_sc, y_train)
    
    preds = model.predict(X_test_sc)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test_sc)[:, 1]
    else:
        probs = model.decision_function(X_test_sc)
    
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    fpr, tpr, _ = roc_curve(y_test, probs)
    score_auc = auc(fpr, tpr)
    
    results.append({
        "Name": name, "AUC": score_auc, "Accuracy": acc, "F1": f1,
        "Model": model, "Preds": preds, "FPR": fpr, "TPR": tpr
    })
    
    print(f"{name:<25} | {acc:.3f}  | {score_auc:.3f}  | {f1:.3f}")


# STEP 4: DETAILED CLASSIFICATION REPORTS

print("\n" + "="*60)
print("DETAILED CLASSIFICATION REPORTS")
print("="*60)

for res in results:
    print(f"\n🔹 MODEL: {res['Name']}")
    print("-" * 30)
    print(classification_report(y_test, res['Preds'], target_names=['Healthy', 'Diabetes']))


# STEP 5: VISUALIZATIONS


# A. ROC CURVES
plt.figure(figsize=(10, 7))
for res in results:
    plt.plot(res['FPR'], res['TPR'], label=f"{res['Name']} (AUC={res['AUC']:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve Comparison (KNN Imputed Data)')
plt.legend(loc='lower right')
plt.savefig(os.path.join(output_folder, '2_Final_ROC.png'))
plt.show()

# B. CONFUSION MATRICES GRID
print("\nGenerating Confusion Matrix Grid...")
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for i, res in enumerate(results):
    ConfusionMatrixDisplay.from_predictions(y_test, res['Preds'], ax=axes[i], cmap='Blues', display_labels=['No', 'Yes'])
    axes[i].set_title(f"{res['Name']}\n(AUC: {res['AUC']:.2f})")
    axes[i].grid(False)

axes[7].axis('off') 
plt.tight_layout()
plt.savefig(os.path.join(output_folder, '3_Confusion_Matrix_Grid.png'))
plt.show()


# STEP 6: FINAL VERDICT
results.sort(key=lambda x: x['AUC'], reverse=True)
print("\n" + "="*40)
print(f"🏆 BEST MODEL: {results[0]['Name']} (AUC={results[0]['AUC']:.3f})")
print(f"🥈 RUNNER UP: {results[1]['Name']} (AUC={results[1]['AUC']:.3f})")
print("="*40)