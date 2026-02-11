import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os

# --- 1. SETTINGS ---
file_path = r'C:\Users\palak\Documents\datathon\25ws-42.17\dataset_csv\combined_nhanes_data.csv'
target_col = 'DIQ010'

# --- 2. LOAD DATA ---
if not os.path.exists(file_path):
    print(f"Error: Could not find {file_path}.")
else:
    df = pd.read_csv(file_path)
    
    # --- 3. SELECT ATTRIBUTES ---
    potential_cols = [col for col in df.columns if col not in ['SEQN', target_col]]
    num_to_pick = min(len(potential_cols), 9)
    random_cols = random.sample(potential_cols, num_to_pick)
    selected_cols = [target_col] + random_cols

    # --- 4. PLOTTING ---
    # Increased figsize height and enabled constrained_layout
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(16, 30), constrained_layout=True)
    axes = axes.flatten()

    print(f"Plotting frequencies for: {selected_cols}")

    for i, col in enumerate(selected_cols):
        plot_data = df[col].dropna()
        unique_vals = plot_data.nunique()
        
        # Use a distinctive color palette for each plot
        if unique_vals < 15:
            sns.countplot(x=plot_data, ax=axes[i], palette="magma")
        else:
            sns.histplot(plot_data, bins=30, ax=axes[i], kde=True, color="#2ecc71")
            
        # Refined font sizes to prevent overlap
        axes[i].set_title(f"Frequency: {col}", fontsize=14, fontweight='bold', pad=10)
        axes[i].set_xlabel("Value", fontsize=12)
        axes[i].set_ylabel("Count / Frequency", fontsize=12)
        axes[i].tick_params(axis='both', which='major', labelsize=10)

    # --- 5. EXPORT ---
    output_image = 'nhanes_attribute_plots_fixed.png'
    plt.savefig(output_image, dpi=300) # Higher DPI for clearer text
    print(f"Success! Plot saved as {output_image}")
    plt.show()