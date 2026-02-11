import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_curve, auc, ConfusionMatrixDisplay, 
                             classification_report, accuracy_score)

# --- CORE MODELS & ENSEMBLES ---
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                              StackingClassifier)

# --- XAI EXTRAS ---
import shap
import lime
import lime.lime_tabular

# --- CONFIG ---
input_folder = 'dataset_csv'
output_folder = 'output/final_rigorous_project'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

rename_map = {
    'BPQ020': 'High_BP', 'PAD810Q': 'Vigorous_Activity_Freq',
    'AUQ054': 'Hearing_Difficulty', 'WHD020': 'Current_Weight(lbs)', 'WHD050': 'Weight_1yr_Ago(lbs)',
    'PAD820': 'Vigorous_Activity_Duration', 'KIQ022': 'Weak_Kidneys', 'PAD790Q': 'Sedentary_Time(minutes)',
    'IMQ011': 'HepA_Vaccine', 'SLD012': 'Sleep_Hours', 'INDFMMPI': 'Poverty_Index', 'WHD010': 'Height(inches)'
}
target_columns = list(rename_map.keys())

# --- STEP 1: LOAD & MERGE ---
print("--- Step 1: Loading Data ---")
try:
    # Logic assumes a file with 'Diabetes' in the name exists
    target_files = glob.glob(os.path.join(input_folder, '*Diabetes*.csv'))
    if not target_files:
        raise FileNotFoundError("No Diabetes CSV found in input folder.")
    
    df_target = pd.read_csv(target_files[0], usecols=['SEQN', 'DIQ010'])
    df_target = df_target[df_target['DIQ010'].isin([1, 2])]
    df_target['Diabetes'] = df_target['DIQ010'].replace({2: 0})
    df = df_target[['SEQN', 'Diabetes']]
except Exception as e:
    print(f" Critical Error: {e}")
    exit()

all_files = glob.glob(os.path.join(input_folder, "*.csv"))
for col in target_columns:
    for filepath in all_files:
        col_check = pd.read_csv(filepath, nrows=0).columns
        if col in col_check:
            df_temp = pd.read_csv(filepath, usecols=['SEQN', col])
            df = pd.merge(df, df_temp, on='SEQN', how='left')
            break
df = df.rename(columns=rename_map)

# --- STEP 2: CLEANING & IMPUTATION ---
print("\n--- Step 2: Cleaning & Imputation ---")
df = df.dropna(subset=['Diabetes'])
cols_numeric = [c for c in df.columns if c not in ['SEQN', 'Diabetes']]
categorical_cols = [col for col in cols_numeric if df[col].nunique() < 15]

# Cleaning special NHANES codes (7: Refused, 9: Don't know)
for col in cols_numeric:
    if col in categorical_cols:
        df[col] = df[col].replace({7: np.nan, 9: np.nan, 77: np.nan, 99: np.nan})
    else:
        df[col] = df[col].replace({7777: np.nan, 9999: np.nan})

imputer = KNNImputer(n_neighbors=5)
X_imputed = pd.DataFrame(imputer.fit_transform(df[cols_numeric]), columns=cols_numeric, index=df.index)
for col in categorical_cols: 
    X_imputed[col] = X_imputed[col].round()
df = pd.concat([df[['SEQN', 'Diabetes']], X_imputed], axis=1)

# --- STEP 3: PREP & TRAIN ---
X = df.drop(['SEQN', 'Diabetes'], axis=1)
y = df['Diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# Define Base Learners for Ensemble
base_learners = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(random_state=42)),
    ('lr', LogisticRegression(max_iter=1000))
]

# FIX: Added Stacking Ensemble to the models dictionary
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN Classifier": KNeighborsClassifier(n_neighbors=5),
    "SVM (Linear)": SVC(kernel="linear", probability=True),
    "Decision Tree": DecisionTreeClassifier(max_depth=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "LDA": LinearDiscriminantAnalysis(),
    "Stacking Ensemble": StackingClassifier(estimators=base_learners, final_estimator=LogisticRegression())
}

# --- STEP 4: EVALUATION LOOP ---
results = []
print(f"\n{'MODEL NAME':<20} | {'ACC':<6} | {'AUC':<6}")
print("-" * 35)

for name, model in models.items():
    model.fit(X_train_sc, y_train)
    preds = model.predict(X_test_sc)
    probs = model.predict_proba(X_test_sc)[:, 1]
    
    acc = accuracy_score(y_test, preds)
    fpr, tpr, _ = roc_curve(y_test, probs)
    score_auc = auc(fpr, tpr)
    
    results.append({
        "Name": name, "AUC": score_auc, "Accuracy": acc, 
        "Model": model, "FPR": fpr, "TPR": tpr, "Preds": preds
    })
    print(f"{name:<20} | {acc:.3f}  | {score_auc:.3f}")

    # STEP 4: DETAILED CLASSIFICATION REPORTS

print("\n" + "="*60)
print("DETAILED CLASSIFICATION REPORTS")
print("="*60)

for res in results:
    print(f"\n🔹 MODEL: {res['Name']}")
    print("-" * 30)
    print(classification_report(y_test, res['Preds'], target_names=['Healthy', 'Diabetes']))

# --- STEP 5: VISUALIZATIONS ---

# 1. Consolidated ROC Curve
plt.figure(figsize=(10, 7))
for res in results:
    plt.plot(res['FPR'], res['TPR'], label=f"{res['Name']} (AUC={res['AUC']:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison (All Models)')
plt.legend(loc='lower right')
plt.savefig(os.path.join(output_folder, 'Final_ROC_Comparison.png'))
plt.show()

# 2. Confusion Matrix Grid (Dynamic sizing)
n_models = len(results)
rows = (n_models + 2) // 3
fig, axes = plt.subplots(rows, 3, figsize=(18, 5 * rows))
axes = axes.flatten()
for i, res in enumerate(results):
    ConfusionMatrixDisplay.from_predictions(y_test, res['Preds'], ax=axes[i], cmap='Blues', display_labels=['No', 'Yes'])
    axes[i].set_title(f"{res['Name']}")
    axes[i].grid(False)

for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'Confusion_Matrix_Grid.png'))
plt.show()

# --- STEP 6: INTERPRETABILITY ---
print("\n--- Step 6: Interpretability ---")

# SHAP (Global) using Random Forest
rf_model = models["Random Forest"]
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test_sc)

# Logic to handle different SHAP output formats (binary classification)
if isinstance(shap_values, list):
    # Older versions return [neg_class_values, pos_class_values]
    values_to_plot = shap_values[1]
elif len(shap_values.shape) == 3:
    # Some versions return (samples, features, classes)
    values_to_plot = shap_values[:, :, 1]
else:
    values_to_plot = shap_values

plt.figure(figsize=(10, 6))
shap.summary_plot(
    values_to_plot, 
    X_test_sc, 
    feature_names=X.columns.tolist(), 
    show=False
)
plt.title("SHAP Global Feature Importance (Random Forest)")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'SHAP_Summary.png'))
plt.show()

# LIME (Local) using Stacking Ensemble
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_sc,
    feature_names=X.columns.tolist(),
    class_names=['Healthy', 'Diabetes'],
    mode='classification'
)

# Explain the first instance of the test set
exp = lime_explainer.explain_instance(
    X_test_sc[0], 
    models["Stacking Ensemble"].predict_proba, 
    num_features=5
)
exp.as_pyplot_figure()
plt.title("LIME: Local Explanation (Stacking Model)")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'LIME_Explanation.png'))
plt.show()

print(f"\nAll outputs saved to: {output_folder}")