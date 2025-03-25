import os
import re
import glob
import numpy as np
import pandas as pd

# =========================
# File Parsing and Data Read
# =========================

def parse_filename_for_coordinates_and_id(filename):
    """
    Given a file path like:
      "IDVG [PL_beforeAnneal _ 0 0 60 0  (60) ; 3_20_2025 5_18_38 PM].csv"
    parse out the x-coordinate, y-coordinate, and device id from the substring
    after the second underscore.

    Returns:
      (x, y, device_id) as (float, float, int) if found;
      otherwise (None, None, None).
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    # Split by '_' at most twice
    parts = base.split('_', 2)
    if len(parts) < 3:
        return None, None, None
    remainder = parts[2]
    tokens = remainder.split()
    numeric_tokens = []
    for t in tokens:
        if re.match(r'^-?\d+(\.\d+)?$', t):
            numeric_tokens.append(t)
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
    Reorders final DataFrame to have these four columns,
    filling any missing column with NaN.
    """
    desired_signals_map = {
        "vg": "VG",
        "vd": "VD",
        "ig": "IG",
        "id": "ID"
    }
    df_raw = pd.read_csv(
        filename,
        skiprows=266,  # so that line 267 is used as header
        header=0,
        delimiter=delimiter,
        na_values=na_values
    )
    # Rename columns by partial (case-insensitive) match
    rename_map = {}
    for col in df_raw.columns:
        col_lower = col.lower()
        matched_name = None
        for pattern, std_name in desired_signals_map.items():
            if pattern in col_lower:
                matched_name = std_name
                break
        rename_map[col] = matched_name if matched_name else col
    df_raw.rename(columns=rename_map, inplace=True)
    # Convert all columns to numeric if possible
    for c in df_raw.columns:
        df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")
    # Ensure the DataFrame has exactly these columns
    final_order = ["VG", "VD", "IG", "ID"]
    for col in final_order:
        if col not in df_raw.columns:
            df_raw[col] = np.nan
    df_final = df_raw[final_order].copy()
    df_final.dropna(how="all", inplace=True)
    return df_final

def fill_column_with_repeating_sequence(df, column_name, sequence, repeat_count):
    """
    If the specified column is missing or entirely NaN, fill it with a repeating sequence.

    Parameters:
      df : pandas.DataFrame
         The DataFrame to modify.
      column_name : str
         The name of the column to check and fill.
      sequence : list
         The sequence of values to repeat (e.g., [0.1, 1, 1.9, 2.8]).
      repeat_count : int
         The number of consecutive rows each sequence value should occupy.
         
    Assumes total rows are equal to len(sequence)*repeat_count or more.
    """
    if column_name not in df.columns or df[column_name].isna().all():
        total_rows = len(df)
        fill_values = []
        for value in sequence:
            fill_values.extend([value] * repeat_count)
        # Extend if needed
        while len(fill_values) < total_rows:
            fill_values.extend(fill_values)
        df[column_name] = fill_values[:total_rows]
    return df

def process_all_csv_files(folder):
    """
    Processes all CSV files in the given folder:
      - Parses filename for x, y, and device_id.
      - Reads each file using row 267 as header, extracting [VG, VD, IG, ID].
      - Fills missing columns (e.g. VD) using a repeating sequence.
    
    Returns a list of dictionaries sorted by device_id, where each dictionary contains:
      - "filename": the file name
      - "x", "y": coordinates (or None)
      - "device_id": device id (or None)
      - "data": the DataFrame with columns [VG, VD, IG, ID]
    """
    all_data = []
    for filepath in glob.glob(os.path.join(folder, "*.csv")):
        x, y, dev_id = parse_filename_for_coordinates_and_id(filepath)
        df = read_data_row_267_reorder(filepath)
        # If VD is missing, fill with known VDD sequence ([0.1, 1, 1.9, 2.8] with 162 rows each)
        df = fill_column_with_repeating_sequence(df, "VD", [0.1, 1, 1.9, 2.8], 162)
        entry = {
            "filename": os.path.basename(filepath),
            "x": x,
            "y": y,
            "device_id": dev_id,
            "data": df
        }
        all_data.append(entry)
    # Sort by device_id (None values at the end)
    all_data.sort(key=lambda d: d["device_id"] if d["device_id"] is not None else float('inf'))
    return all_data

# =========================
# FoM Calculations
# =========================

def normalize_current(df, device_width=10):
    """
    Converts ID and IG from A to µA/µm given the device width in µm.
    """
    conversion_factor = 1e6 / device_width  # from A to µA, normalized per µm
    df["ID_norm"] = df["ID"] * conversion_factor
    df["IG_norm"] = df["IG"] * conversion_factor
    return df

def compute_SS(df):
    """
    Computes the subthreshold slope (SS) from the ID_norm vs VG curve.
    Uses finite differences on log10(ID_norm) for ID_norm values above a threshold.
    Then discards any SS values less than 60 mV/dec (i.e. 0.06 V/dec) and returns
    the minimal SS among the remaining values.
    If no SS values pass the threshold, returns np.nan.
    """
    threshold = 1e-3  # µA/µm threshold to avoid log(0) issues
    df_valid = df[df["ID_norm"] > threshold].copy()
    if len(df_valid) < 2:
        return np.nan
    VG = df_valid["VG"].values
    ID_norm = df_valid["ID_norm"].values
    logID = np.log10(ID_norm)
    SS_list = []
    for i in range(len(VG) - 1):
        dVG = VG[i+1] - VG[i]
        dlogI = logID[i+1] - logID[i]
        if dlogI != 0:
            SS_list.append(dVG / dlogI)
    if not SS_list:
        return np.nan
    # Filter out SS values less than 0.06 V/dec (i.e. 60 mV/dec)
    valid_ss = [ss for ss in SS_list if ss >= 0.06]
    if valid_ss:
        return min(valid_ss)
    else:
        return np.nan

def compute_VTH_and_gm(df, vdd):
    """
    Computes the threshold voltage (VTH) and maximum transconductance (gm_max)
    from the ID_norm vs VG curve using the linear extrapolation method.
    
    The method:
      - Computes gm = d(ID_norm)/d(VG)
      - Finds the index where gm is maximum
      - Estimates the intercept (Vcross) from:
             0 = ID_max + gm_max*(VG_intercept - VG_max)
             => VG_intercept = VG_max - (ID_max / gm_max)
      - Defines VTH = VG_intercept - 0.5 * vdd
      
    Returns (VTH, gm_max).
    """
    VG = df["VG"].values
    ID_norm = df["ID_norm"].values
    if len(VG) < 2:
        return np.nan, np.nan
    gm = np.gradient(ID_norm, VG)
    i_max = np.argmax(gm)
    gm_max = gm[i_max]
    VG_max = VG[i_max]
    ID_max = ID_norm[i_max]
    if gm_max == 0:
        return np.nan, np.nan
    Vcross = VG_max - (ID_max / gm_max)
    VTH = Vcross - 0.5 * vdd
    return VTH, gm_max

def compute_hysteresis(df, vdd):
    """
    For a bi-directional sweep (assumed to be split evenly),
    computes hysteresis as the absolute difference between VTH values from the
    forward and reverse sweeps.
    
    Returns the hysteresis (in V) or NaN if data is insufficient.
    """
    n = len(df)
    if n % 2 != 0 or n < 2:
        return np.nan
    half = n // 2
    df_forward = df.iloc[:half].copy()
    df_reverse = df.iloc[half:].copy()
    VTH_fwd, _ = compute_VTH_and_gm(df_forward, vdd)
    VTH_rev, _ = compute_VTH_and_gm(df_reverse, vdd)
    return abs(VTH_fwd - VTH_rev)

def compute_device_FoMs(df, device_width=10):
    """
    Given a DataFrame with columns [VG, VD, IG, ID] for one device,
    this function:
      - Normalizes currents (ID and IG) from A to µA/µm.
      - Filters data by VDD (using the filled VD column) for VDD = 0.1 V and 1 V.
      - Computes:
          FoM1: Minimum subthreshold slope (SS) from the ID–VG curve at VDD = 0.1 V.
          FoM2: Threshold voltage (VTH) from the linear extrapolation method at VDD = 0.1 V.
          FoM3: Hysteresis (absolute difference in VTH from forward and reverse sweeps) at VDD = 0.1 V.
          FoM4: Maximum transconductance (Gm_max) at VDD = 1 V.
    Returns a dictionary of these FoMs.
    """
    # Normalize currents
    df = normalize_current(df, device_width)
    
    # Filter data for the two VDD values
    df_vdd_0_1 = df[df["VD"] == 0.1].copy()
    df_vdd_1 = df[df["VD"] == 1.0].copy()
    
    # FoM1: SS (in V/dec) from the ID–VG curve at VDD = 0.1 V.
    FoM1_SS = compute_SS(df_vdd_0_1)
    
    # FoM2: VTH from linear extrapolation at VDD = 0.1 V.
    VTH, gm_max = compute_VTH_and_gm(df_vdd_0_1, 0.1)
    FoM2_VTH = VTH
    
    # FoM3: Hysteresis (|VTH_forward - VTH_reverse|) at VDD = 0.1 V.
    FoM3_hysteresis = compute_hysteresis(df_vdd_0_1, 0.1)
    
    # FoM4: Gm_max at VDD = 1 V.
    _, gm_max_vdd1 = compute_VTH_and_gm(df_vdd_1, 1.0)
    FoM4_gm_max = gm_max_vdd1
    
    return {
        "SS_min (V/dec)": FoM1_SS,
        "VTH (V)": FoM2_VTH,
        "Hysteresis (V)": FoM3_hysteresis,
        "Gm_max @1V (µA/µm per V)": FoM4_gm_max
    }

# =========================
# Main Processing Block
# =========================

if __name__ == "__main__":
    # Set the folder containing your CSV files
    folder_path = r"D:\PL,\PL1\Testing"  # adjust as needed
    # Optional: set the device width in µm (default 10 µm)
    device_width = 10

    # Process all CSV files into a list of device entries
    results = process_all_csv_files(folder_path)
    
    # For each device, compute and display FoMs (using the appropriate VDD slices)
    for entry in results:
        print(f"File: {entry['filename']}")
        print(f"  Device ID: {entry['device_id']}, (x={entry['x']}, y={entry['y']})")
        print(f"  Data shape: {entry['data'].shape}")
        # Compute FoMs for this device
        FoMs = compute_device_FoMs(entry["data"], device_width)
        print("  Figures-of-Merit:")
        for key, value in FoMs.items():
            print(f"    {key}: {value}")
        print("\n")
