import os
import re
import glob
import pandas as pd
import numpy as np

def parse_filename_for_coordinates_and_id(filename):
    """
    Given a file path like:
      "IDVG [PL_beforeAnneal _ 0 0 60 0  (60) ; 3_20_2025 5_18_38 PM].csv"
    parse out the x-coordinate, y-coordinate, and device id from the substring
    after the second underscore.

    Returns:
      (x, y, device_id) as (float, float, int) if found,
      otherwise (None, None, None).
    """
    # 1. Strip directory and extension
    base = os.path.splitext(os.path.basename(filename))[0]
    # Example base: "IDVG [PL_beforeAnneal _ 0 0 60 0  (60) ; 3_20_2025 5_18_38 PM]"

    # 2. Split by '_' at most twice
    parts = base.split('_', 2)
    if len(parts) < 3:
        return None, None, None

    remainder = parts[2]
    # 3. Split on whitespace and extract numeric tokens using regex
    tokens = remainder.split()
    numeric_tokens = []
    for t in tokens:
        # Match a number (integer or float) with optional sign
        if re.match(r'^-?\d+(\.\d+)?$', t):
            numeric_tokens.append(t)
    # We expect at least 3 numeric tokens: x, y, device_id
    if len(numeric_tokens) < 3:
        return None, None, None

    try:
        x = float(numeric_tokens[0])
        y = float(numeric_tokens[1])
        device_id = int(float(numeric_tokens[2]))
    except Exception:
        return None, None, None

    return x, y, device_id

def read_data_row_267_reorder(filename, na_values=["######", "####", "---"], delimiter=","):
    """
    Reads the CSV so that row 267 (1-based) is used as the header (skiprows=266).
    Renames columns by partial matching to ["VG", "VD", "IG", "ID"].
    Reorders final DataFrame to have columns [VG, VD, IG, ID],
    filling any missing column with NaN.
    """
    desired_signals_map = {
        "vg": "VG",
        "vd": "VD",
        "ig": "IG",
        "id": "ID"
    }
    # Read file with row 267 as header (skip first 266 rows)
    df_raw = pd.read_csv(
        filename,
        skiprows=266,
        header=0,
        delimiter=delimiter,
        na_values=na_values
    )

    # Rename columns by partial, case-insensitive matching
    rename_map = {}
    for col in df_raw.columns:
        col_lower = col.lower()
        matched_name = None
        for pattern, std_name in desired_signals_map.items():
            if pattern in col_lower:
                matched_name = std_name
                break
        if matched_name:
            rename_map[col] = matched_name
        else:
            rename_map[col] = col  # keep original if no match

    df_raw.rename(columns=rename_map, inplace=True)

    # Convert all columns to numeric if possible (non-numeric become NaN)
    for c in df_raw.columns:
        df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")

    # Ensure the final DataFrame has exactly these columns, filling missing ones with NaN
    final_order = ["VG", "VD", "IG", "ID"]
    for col in final_order:
        if col not in df_raw.columns:
            df_raw[col] = np.nan

    df_final = df_raw[final_order].copy()
    df_final.dropna(how="all", inplace=True)  # optionally drop rows that are completely NaN

    return df_final

def process_all_csv_files(folder):
    """
    1. Finds all CSV files in the folder.
    2. Parses each file name to extract (x, y, device_id).
    3. Reads each file (row 267 as header), extracting columns [VG, VD, IG, ID].
    4. Returns a list of dictionaries sorted by device_id.
    
    Each dictionary has:
      - "filename": file name
      - "x": x-coordinate (or None)
      - "y": y-coordinate (or None)
      - "device_id": device id (or None)
      - "data": DataFrame with columns [VG, VD, IG, ID]
    """
    all_data = []
    for filepath in glob.glob(os.path.join(folder, "*.csv")):
        x, y, dev_id = parse_filename_for_coordinates_and_id(filepath)
        df = read_data_row_267_reorder(filepath)
        entry = {
            "filename": os.path.basename(filepath),
            "x": x,
            "y": y,
            "device_id": dev_id,
            "data": df
        }
        all_data.append(entry)
    
    # Sort the list by device_id; if device_id is None, sort it at the end
    all_data.sort(key=lambda d: d["device_id"] if d["device_id"] is not None else float('inf'))
    return all_data

# ------------------ Main Block ------------------
if __name__ == "__main__":
    # Folder where the CSV files are located
    folder_path = r"D:\PL,"  # adjust to your folder location
    
    results = process_all_csv_files(folder_path)
    
    # Display summary for each file processed
    for entry in results:
        print(f"File: {entry['filename']}")
        print(f"  Device ID: {entry['device_id']}, (x={entry['x']}, y={entry['y']})")
        print(f"  Data shape: {entry['data'].shape}")
        print(f"  Head of DataFrame:\n{entry['data'].head()}\n")
