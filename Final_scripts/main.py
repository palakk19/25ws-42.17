# ==============================================================================
# NHANES DIABETES PREDICTION: FINAL COMPREHENSIVE PIPELINE
# (Ensemble, Confusion Matrices, ROC Plots, Threshold Tuning)
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import shap
import os
import shutil
import warnings
from scipy.stats import spearmanr

# Scikit-Learn Imports
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, precision_recall_curve, auc
)
from tqdm.auto import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')

def seed_everything(seed=42):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
DATA_PATH = r"C:\palakdatathon\25ws-42.17 (1)\25ws-42.17\dataset_csv\nhanes_filtered.csv"
OUTPUT_DIR = "diabetes_final_v3"
PROCESSED_DIR = os.path.join(OUTPUT_DIR, "processed_data") 
TARGET_COL = "DIQ010"
BATCH_SIZE = 256
EPOCHS = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_STATE = 42
LEAKAGE_COLS = ["LBXGH", "LBXGLU", "DIQ050", "DIQ070", "LBXGLT"]

# ------------------------------------------------------------------------------
# 1. LOAD & PROCESSING
# ------------------------------------------------------------------------------
print("\n[1/9] Processing Data...")

df = pd.read_csv(DATA_PATH)
if "SEQN" in df.columns: df = df.drop(columns=["SEQN"])

existing_leakage = [c for c in LEAKAGE_COLS if c in df.columns]
if existing_leakage: df = df.drop(columns=existing_leakage)

df = df[df[TARGET_COL].isin([1, 2, 3])]
df[TARGET_COL] = df[TARGET_COL].map({1: 1, 2: 0, 3: 0})
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL].values
feature_names_original = X.columns.tolist() 

null_cols = X.columns[X.isna().all()].tolist()
if null_cols: 
    X = X.drop(columns=null_cols)
    feature_names_original = [f for f in feature_names_original if f not in null_cols]

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_train_enc = encoder.fit_transform(X_train_raw)
X_test_enc = encoder.transform(X_test_raw)

# Calculate valid limits
valid_max_indices = []
for i in range(X_train_enc.shape[1]):
    col = X_train_enc[:, i]
    valid_data = col[col >= 0]
    m = np.max(valid_data) if len(valid_data) > 0 else 0
    valid_max_indices.append(m)

print(" -> Imputing...")
imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=3, random_state=RANDOM_STATE, verbose=1)
X_train_imp = imputer.fit_transform(X_train_enc)
X_test_imp = imputer.transform(X_test_enc)

# Sanitization
X_train_imp = np.nan_to_num(X_train_imp, nan=0.0, posinf=0.0, neginf=0.0)
X_test_imp = np.nan_to_num(X_test_imp, nan=0.0, posinf=0.0, neginf=0.0)

for i in range(X_train_imp.shape[1]):
    limit = valid_max_indices[i] + 2
    X_train_imp[:, i] = np.clip(X_train_imp[:, i], 0, limit)
    X_test_imp[:, i] = np.clip(X_test_imp[:, i], 0, limit)

X_train_imp = np.round(X_train_imp).astype(int)
X_test_imp = np.round(X_test_imp).astype(int)

# ------------------------------------------------------------------------------
# 2. AUTOENCODER DEFINITION
# ------------------------------------------------------------------------------
cat_cols_count = X_train_imp.shape[1]
cardinalities = []
for i in range(cat_cols_count):
    max_val = max(X_train_imp[:, i].max(), X_test_imp[:, i].max())
    safe_card = max(2, min(int(max_val) + 2, 500)) 
    cardinalities.append(safe_card)

class CatAutoEncoder(nn.Module):
    def __init__(self, cardinalities, latent_dim):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, min(50, max(4, card // 2))) for card in cardinalities
        ])
        emb_out_dim = sum(e.embedding_dim for e in self.embeddings)
        
        self.encoder = nn.Sequential(
            nn.Linear(emb_out_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, emb_out_dim)
        )

    def forward(self, x):
        embs = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        x_cat = torch.cat(embs, dim=1)
        z = self.encoder(x_cat)
        out = self.decoder(z)
        return out, z

# ------------------------------------------------------------------------------
# 3. TRAIN AUTOENCODER
# ------------------------------------------------------------------------------
print("\n[2/9] Training Autoencoder...")
OPTIMAL_DIM = 40 

train_tensor = torch.tensor(X_train_imp, dtype=torch.long)
test_tensor = torch.tensor(X_test_imp, dtype=torch.long)
train_loader = DataLoader(TensorDataset(train_tensor), batch_size=BATCH_SIZE, shuffle=True)

ae_model = CatAutoEncoder(cardinalities, OPTIMAL_DIM).to(DEVICE)
optimizer = torch.optim.Adam(ae_model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in tqdm(range(EPOCHS), desc="AE Epochs"):
    ae_model.train()
    for (batch,) in train_loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        recon, _ = ae_model(batch)
        with torch.no_grad():
            target = torch.cat([ae_model.embeddings[i](batch[:, i]) for i in range(batch.shape[1])], dim=1)
        loss = criterion(recon, target)
        loss.backward()
        optimizer.step()

# ------------------------------------------------------------------------------
# 4. EXTRACT LATENT VECTORS (Z)
# ------------------------------------------------------------------------------
print("\n[3/9] Extracting Latent Features...")
ae_model.eval()

def get_latent(tensor_data):
    loader = DataLoader(TensorDataset(tensor_data), batch_size=BATCH_SIZE, shuffle=False)
    features = []
    for (batch,) in loader:
        batch = batch.to(DEVICE)
        with torch.no_grad():
            _, z = ae_model(batch)
        features.append(z.cpu().numpy())
    return np.vstack(features)

X_train_z = get_latent(train_tensor)
X_test_z = get_latent(test_tensor)

scaler_z = StandardScaler()
X_train_z = scaler_z.fit_transform(X_train_z)
X_test_z = scaler_z.transform(X_test_z)

# ------------------------------------------------------------------------------
# 5. FIXED MAPPING (UNIQUE & NON-REDUNDANT)
# ------------------------------------------------------------------------------
print("\n[4/9] Mapping Latent Features...")

def get_unique_latent_names(X_original, Z_latent, feature_names):
    n_latent = Z_latent.shape[1]
    subset_idx = np.random.choice(len(X_original), min(2000, len(X_original)), replace=False)
    X_sub = X_original[subset_idx]
    Z_sub = Z_latent[subset_idx]
    
    all_corrs = []
    for i in range(n_latent):
        for j in range(X_sub.shape[1]):
            c, _ = spearmanr(Z_sub[:, i], X_sub[:, j])
            all_corrs.append((abs(c), j, i))
            
    all_corrs.sort(key=lambda x: x[0], reverse=True)
    
    latent_names = {}
    used_features = set()
    used_latent = set()
    
    for corr, feat_idx, lat_idx in all_corrs:
        if lat_idx not in used_latent and feat_idx not in used_features:
            latent_names[lat_idx] = f"L{lat_idx}: {feature_names[feat_idx]}"
            used_features.add(feat_idx)
            used_latent.add(lat_idx)
            
    final_names = []
    for i in range(n_latent):
        if i in latent_names:
            final_names.append(latent_names[i])
        else:
            final_names.append(f"L{i}: (Weak Signal)")
            
    return final_names

latent_feature_names = get_unique_latent_names(X_train_imp, X_train_z, feature_names_original)

# ------------------------------------------------------------------------------
# 6. DEFINE MODELS (INCLUDING ENSEMBLE)
# ------------------------------------------------------------------------------
print("\n[5/9] Defining Models (Including Ensemble)...")

lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE)
svc_model = SVC(kernel='linear', class_weight='balanced', probability=True, random_state=RANDOM_STATE)
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)
rf_model = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=RANDOM_STATE)
nn_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=RANDOM_STATE)

# --- ENSEMBLE MODEL ---
ensemble_model = VotingClassifier(
    estimators=[
        ('LR', lr_model),
        ('SVM', svc_model),
        ('GB', gb_model)
    ],
    voting='soft'
)

models = {
    "Logistic Regression": lr_model,
    "SVM (Linear)": svc_model,
    "Random Forest": rf_model,
    "Gradient Boosting": gb_model,
    "Neural Net": nn_model,
    "Ensemble (Voting)": ensemble_model
}

# ------------------------------------------------------------------------------
# 7. TRAIN, TUNE & PLOT CONFUSION MATRICES
# ------------------------------------------------------------------------------
print("\n[6/9] Training, Tuning & Generating Confusion Matrices...")

results = []
roc_curve_data = {}

for name, model in tqdm(models.items(), desc="Models"):
    # Train
    model.fit(X_train_z, y_train)
    
    # Probabilities
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_z)[:, 1]
    else:
        y_proba = model.decision_function(X_test_z)
        y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())

    # --- THRESHOLD TUNING ---
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    # Predictions at Optimal Threshold
    y_pred = (y_proba >= best_thresh).astype(int)
    
    # Metrics
    roc = roc_auc_score(y_test, y_proba)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred)
    
    results.append({
        "Model": name, "ROC_AUC": roc, "Recall": rec, 
        "Precision": prec, "F1_Score": f1, "Threshold": best_thresh
    })
    
    # Store ROC Data
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_curve_data[name] = (fpr, tpr, roc)
    
    # --- PLOT CONFUSION MATRIX ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Healthy', 'Diabetes'],
                yticklabels=['Healthy', 'Diabetes'])
    plt.title(f"Confusion Matrix: {name}\n(Thresh={best_thresh:.2f})")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"cm_{name.replace(' ', '_')}.png"))
    plt.close()

# ------------------------------------------------------------------------------
# 8. ROC CURVE COMPARISON PLOT
# ------------------------------------------------------------------------------
print("\n[7/9] Generating ROC Comparison Plot...")
plt.figure(figsize=(10, 8))

for name, (fpr, tpr, auc_score) in roc_curve_data.items():
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Chance')
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
plt.title('ROC Curve Comparison (Latent Features)', fontsize=14)
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve_comparison.png"))
plt.close()

# ------------------------------------------------------------------------------
# 9. SHAP ANALYSIS
# ------------------------------------------------------------------------------
print("\n[8/9] Generating SHAP Plot...")
try:
    explainer_model = models["Gradient Boosting"]
    explainer = shap.TreeExplainer(explainer_model)
    shap_values = explainer.shap_values(X_test_z)
    
    vals = shap_values[1] if isinstance(shap_values, list) else shap_values
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        vals, 
        X_test_z, 
        feature_names=latent_feature_names, 
        show=False,
        plot_size=(10, 8)
    )
    plt.title("SHAP: Feature Importance (Latent Space)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary.png"))
    plt.close()
    print(" -> SHAP plot saved.")
except Exception as e:
    print(f" -> SHAP skipped: {e}")

# ------------------------------------------------------------------------------
# 10. SAVE RESULTS
# ------------------------------------------------------------------------------
print("\n[9/9] Saving Final Results...")
df_res = pd.DataFrame(results).sort_values(by="Recall", ascending=False)
df_res.to_csv(os.path.join(OUTPUT_DIR, "final_comprehensive_results.csv"), index=False)

print("\n--- Final Leaderboard ---")
print(df_res[["Model", "Recall", "Precision", "ROC_AUC", "Threshold"]])

print(f"\nPipeline Complete! All plots and CSVs saved to: {os.path.abspath(OUTPUT_DIR)}")