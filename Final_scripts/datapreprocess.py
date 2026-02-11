import pandas as pd
import os

def combine_nhanes_folder(base_path):
    # --- 1. CONFIGURATION ---
    # Update this if your diabetes file has a different name
    diabetes_filename = 'diabetes.csv' 
    target_col = 'DIQ010'
    output_filename = 'combined_nhanes_data.csv'

    # Construct the full path to the diabetes file
    diabetes_path = os.path.join(base_path, diabetes_filename)

    if not os.path.exists(diabetes_path):
        print(f"Error: Could not find the base file {diabetes_filename} in {base_path}")
        return

    # --- 2. LOAD AND FILTER THE BASE FILE ---
    print(f"Reading base diabetes file: {diabetes_filename}")
    df_combined = pd.read_csv(diabetes_path)

    # Filter for diabetes values 1, 2, or 3 and remove NaNs
    valid_values = [1, 2, 3]
    df_combined = df_combined[df_combined[target_col].isin(valid_values)].dropna(subset=[target_col])
    
    print(f"Base patients identified: {len(df_combined)}")

    # --- 3. LOOP THROUGH THE FOLDER AND MERGE ---
    for file in os.listdir(base_path):
        # We only want CSVs, and we skip the diabetes file (already loaded) and the output file
        if file.endswith('.csv') and file != diabetes_filename and file != output_filename:
            file_path = os.path.join(base_path, file)
            print(f"Merging: {file}")
            
            try:
                df_temp = pd.read_csv(file_path)
                
                # Standard NHANES merge on SEQN (Sequence Number)
                # 'left' merge ensures we only keep the patients from our filtered diabetes list
                df_combined = pd.merge(df_combined, df_temp, on='SEQN', how='left')
            except Exception as e:
                print(f"Could not merge {file}: {e}")

    # --- 4. EXPORT ---
    save_path = os.path.join(base_path, output_filename)
    df_combined.to_csv(save_path, index=False)
    
    print("-" * 30)
    print(f"Success! Combined dataset saved to: {save_path}")
    print(f"Final dimensions: {df_combined.shape}")

# --- EXECUTION ---
folder_path = r'C:\Users\palak\Documents\datathon\25ws-42.17\dataset_csv'
combine_nhanes_folder(folder_path)