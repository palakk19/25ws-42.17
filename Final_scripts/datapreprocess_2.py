import pandas as pd
import numpy as np
import re

def clean_nhanes_data(file_path, output_path):
    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path)

    # --- 1. FIX FLOATING POINT ERRORS (TINY EXPONENTIALS) ---
    print("Fixing floating point errors...")
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Identify values that are practically zero (e.g., 1.4e-15) and snap them to 0.0
    # The threshold 1e-8 is generally safe for NHANES data
    for col in numeric_cols:
        # Create a mask for tiny non-zero values
        mask = (df[col].abs() < 1e-9) & (df[col] != 0)
        if mask.any():
            df.loc[mask, col] = 0.0

    # --- 2. HANDLE TIME COLUMNS (HH:MM -> Decimal Hours) ---
    print("Converting time columns...")
    
    # Regex to identify HH:MM format (e.g., "14:30", "09:45")
    time_pattern = re.compile(r'^\d{1,2}:\d{2}$')

    for col in df.columns:
        # Check if column is object/string type
        if df[col].dtype == 'object':
            # Check the first valid value to see if it looks like a time
            first_valid = df[col].dropna().astype(str).iloc[0] if not df[col].dropna().empty else ""
            
            if time_pattern.match(first_valid):
                print(f" -> Converting column '{col}' to 24-hr decimal format.")
                
                def convert_time(val):
                    if pd.isna(val) or not isinstance(val, str): return np.nan
                    try:
                        h, m = map(int, val.split(':'))
                        return h + (m / 60.0) # Converts 14:30 to 14.5
                    except:
                        return np.nan
                
                df[col] = df[col].apply(convert_time)

    # --- 3. REPLACE INVALID CODES (7, 9, 7777, 9999) ---
    print("Replacing invalid codes (7, 9, 7777, 9999) with NaN...")
    
    for col in df.columns:
        # Skip non-numeric columns for this step
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
            
        # -- Heuristic to detect Categorical vs Continuous --
        # If a column has few unique values (e.g. < 20), we treat it as Categorical.
        # This is because continuous data (like weight) usually has hundreds of unique values.
        unique_count = df[col].nunique()
        max_val = df[col].max()
        
        if unique_count < 25 and max_val < 20:
            # CATEGORICAL CASE: Replace 7 and 9
            # (Note: Be careful if 7 is a valid value, like "Days per week")
            mask = df[col].isin([7, 9])
            if mask.any():
                df.loc[mask, col] = np.nan
                
        else:
            # CONTINUOUS CASE: Replace 7777, 9999 (and typically 77, 99 if high values)
            # We look for exact matches to the common NHANES missing codes
            invalid_continuous = [7777, 9999, 77777, 99999]
            mask = df[col].isin(invalid_continuous)
            if mask.any():
                df.loc[mask, col] = np.nan

    # --- 4. SAVE ---
    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to: {output_path}")
    print(f"Final shape: {df.shape}")

# --- EXECUTION ---
input_csv = r'C:\Users\palak\Documents\datathon\25ws-42.17\dataset_csv\combined_nhanes_data.csv'
output_csv = r'C:\Users\palak\Documents\datathon\25ws-42.17\dataset_csv\nhanes_cleaned.csv'

clean_nhanes_data(input_csv, output_csv)


#---Drop 

# --- 1. SETTINGS ---
input_path = r'C:\Users\palak\Documents\datathon\25ws-42.17\dataset_csv\nhanes_cleaned.csv'
output_path = r'C:\Users\palak\Documents\datathon\25ws-42.17\dataset_csv\nhanes_filtered.csv'

# --- 2. LOAD DATA ---
df = pd.read_csv(input_path)

# Store shape before
shape_before = df.shape

# --- 3. DROP ROWS WITH > 60% MISSING VALUES ---
# Logic: If we want to drop rows with more than 60% missing, 
# we must KEEP rows that have at least 40% non-missing values.
# thresh = minimum number of NON-NaN values required to keep the row.
total_columns = len(df.columns)
min_non_nan_required = int(0.25 * total_columns)

df_filtered = df.dropna(thresh=min_non_nan_required)

# Store shape after
shape_after = df_filtered.shape

# --- 4. RESULTS & SAVING ---
print(f"--- Dataset Filtering Results ---")
print(f"Shape BEFORE filtering: {shape_before}")
print(f"Shape AFTER filtering:  {shape_after}")
print(f"Rows removed:           {shape_before[0] - shape_after[0]}")
print(f"Total columns:          {total_columns}")
print(f"Required data points per row: {min_non_nan_required}")

# Save the final file
df_filtered.to_csv(output_path, index=False)
print(f"\nFinal dataset saved to: {output_path}")

