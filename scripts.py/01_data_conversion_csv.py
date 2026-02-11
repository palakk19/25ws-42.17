import pandas as pd
import glob
import os

# --- CONFIGURATION ---
# We tell Python exactly where to look based on your structure
source_folder = os.path.join("data", "raw")  # Looks in data/raw
output_folder = "dataset_csv"                # Where to save the CSVs

# 1. Check if the source folder actually exists
if not os.path.exists(source_folder):
    print(f" ERROR: Python cannot find the folder: {source_folder}")
    print(f"   Current working directory is: {os.getcwd()}")
    print("   Make sure you are running this script from the project root!")
else:
    # 2. Create the output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created new folder: {output_folder}")

    # 3. Find all XPT files in data/raw
    # We look for both lowercase .xpt and uppercase .XPT
    search_path = os.path.join(source_folder, "*.XPT")
    xpt_files = glob.glob(search_path) + glob.glob(os.path.join(source_folder, "*.xpt"))

    if not xpt_files:
        print(f" No .XPT files found in {source_folder}")
    else:
        print(f" Found {len(xpt_files)} files in '{source_folder}'\n")

        # 4. Loop and Convert
        for file_path in xpt_files:
            try:
                file_name = os.path.basename(file_path)
                print(f"   Converting {file_name}...", end=" ")

                # Read SAS file
                df = pd.read_sas(file_path)

                # Fix weird text formatting (byte strings)
                for col in df.select_dtypes([object]):
                    try:
                        df[col] = df[col].str.decode('utf-8')
                    except AttributeError:
                        pass

                # Save as CSV in the output folder
                csv_name = os.path.splitext(file_name)[0] + ".csv"
                save_path = os.path.join(output_folder, csv_name)
                
                df.to_csv(save_path, index=False)
                print(f"Done.")

            except Exception as e:
                print(f"Failed: {e}")

        print(f"\n Success! Open the '{output_folder}' folder to see your CSVs.")