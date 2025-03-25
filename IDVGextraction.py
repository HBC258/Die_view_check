import os
import re
import glob
import numpy as np
import pandas as pd

# -------------------------
# 1) Parsing & Data Reading
# -------------------------

def parse_filename_for_coordinates_and_id(filename):
    base = os.path.splitext(os.path.basename(filename))[0]
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

def read_data_with_header(filename, header_row=267, na_values=["######", "####", "---"], delimiter=","):
    skip_rows = header_row - 1
    desired_signals_map = {"vg": "VG", "vd": "VD", "ig": "IG", "id": "ID"}
    df_raw = pd.read_csv(
        filename,
        skiprows=skip_rows,
        header=0,
        delimiter=delimiter,
        na_values=na_values
    )
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

    final_order = ["VG", "VD", "IG", "ID"]
    for col in final_order:
        if col not in df_raw.columns:
            df_raw[col] = np.nan
    df_final = df_raw[final_order].copy()

    # Convert columns to numeric to avoid multiplication errors
    for col in final_order:
        df_final[col] = pd.to_numeric(df_final[col], errors='coerce')

    df_final.dropna(how="all", inplace=True)
    return df_final

def fill_column_with_repeating_sequence(df, column_name, sequence, repeat_count):
    if column_name not in df.columns or df[column_name].isna().all():
        total_rows = len(df)
        fill_values = []
        for value in sequence:
            fill_values.extend([value] * repeat_count)
        while len(fill_values) < total_rows:
            fill_values.extend(fill_values)
        df[column_name] = fill_values[:total_rows]
    return df

def process_all_csv_files(folder, header_row=267):
    all_data = []
    for filepath in glob.glob(os.path.join(folder, "*.csv")):
        x, y, dev_id = parse_filename_for_coordinates_and_id(filepath)
        df = read_data_with_header(filepath, header_row=header_row)
        df = fill_column_with_repeating_sequence(df, "VD", [0.1, 1, 1.9, 2.8], 162)
        entry = {
            "filename": os.path.basename(filepath),
            "x": x,
            "y": y,
            "device_id": dev_id,
            "data": df
        }
        all_data.append(entry)
    all_data.sort(key=lambda d: d["device_id"] if d["device_id"] is not None else float('inf'))
    return all_data

# -------------------------
# 2) Checking "Bad" Data
# -------------------------

def is_bad_data(df):
    """
    Returns True if the DataFrame is incomplete or invalid.
    You can customize the conditions here.

    For example:
    - Fewer than 10 rows
    - ID column has too many NaNs
    - etc.
    """
    if len(df) < 10:
        return True  # too few data points

    # Check how many NaNs are in ID column
    num_nans = df["ID"].isna().sum()
    if num_nans > 0.8 * len(df):
        return True  # more than 80% are NaN

    # (Optional) Add more conditions if needed

    return False

# -------------------------
# 3) FoM Calculations
# -------------------------

def normalize_current(df, device_width=10):
    conversion_factor = 1e6 / device_width
    # Force columns to numeric again, just to be safe
    df["ID"] = pd.to_numeric(df["ID"], errors='coerce')
    df["IG"] = pd.to_numeric(df["IG"], errors='coerce')

    df["ID_norm"] = df["ID"] * conversion_factor
    df["IG_norm"] = df["IG"] * conversion_factor
    return df

def compute_SS(df):
    """
    Computes the subthreshold slope (SS) from the ID_norm vs VG curve.
    Uses finite differences on log10(ID_norm) for ID_norm values above a threshold.
    Then discards any SS values less than 0.06 V/dec (60 mV/dec) and returns the minimal SS among the remaining values.
    If no SS values pass the threshold, returns np.nan.
    """
    threshold = 1e-3  # µA/µm threshold to avoid log(0)
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
    # Filter out SS values less than 0.06 V/dec (60 mV/dec)
    valid_ss = [ss for ss in SS_list if ss >= 0.06]
    if valid_ss:
        return min(valid_ss)
    else:
        return np.nan

def compute_VTH_and_gm(df, vdd):
    """
    Computes the threshold voltage (VTH) and maximum transconductance (gm_max)
    from the ID_norm vs VG curve using linear extrapolation.
    
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
    gm_max = gm[i_max-1]
    VG_max = VG[i_max]
    ID_max = ID_norm[i_max]
    if gm_max == 0:
        return np.nan, np.nan
    Vcross = VG_max - (ID_max / gm_max)
    VTH = Vcross - 0.5 * vdd
    return VTH, gm_max

def compute_hysteresis(df, threshold_current=1e-6):
    """
    Computes hysteresis for a dual-sweep measurement at VDD = 0.1 V.
    
    The function checks if VG exhibits a dual sweep pattern (e.g. forward sweep from -2 to 2, and backward from 2 to -2).
    It splits the data at the turning point (the maximum VG value) into forward and backward segments.
    
    For each segment, it finds the two consecutive points where ID_norm crosses the threshold_current.
    Linear interpolation is then used to calculate the precise VG value at which ID_norm equals the threshold.
    
    The hysteresis is the absolute difference between the interpolated VG values from the forward and backward sweeps.
    If the dual-sweep condition is not met or the threshold crossing cannot be determined, returns np.nan.
    """
    VG = df["VG"].values
    ID_norm = df["ID_norm"].values
    n = len(VG)
    if n < 2:
        return np.nan
    # Identify turning point: maximum VG (assumed as the split point)
    i_max = np.argmax(VG)
    if i_max == 0 or i_max == n - 1:
        return np.nan  # not a dual sweep
    
    # Split the data into forward (from start to i_max) and backward (from i_max to end) segments.
    forward_df = df.iloc[:i_max+1].copy()
    backward_df = df.iloc[i_max+1:].copy()
    
    def interpolate_crossing(VG_arr, ID_arr, threshold):
        """
        Returns the interpolated VG value where ID_arr crosses the threshold,
        by scanning for a pair of consecutive points where the crossing occurs.
        """
        for i in range(len(VG_arr) - 1):
            if ID_arr[i] < threshold and ID_arr[i+1] >= threshold:
                VG1, VG2 = VG_arr[i], VG_arr[i+1]
                I1, I2 = ID_arr[i], ID_arr[i+1]
                if I2 == I1:
                    return VG1  # avoid division by zero
                VG_cross = VG1 + (threshold - I1) * (VG2 - VG1) / (I2 - I1)
                return VG_cross
        return None

    # For the forward sweep (assumed to be increasing in VG)
    VG_forward = forward_df["VG"].values
    ID_forward = forward_df["ID_norm"].values
    VG_cross_forward = interpolate_crossing(VG_forward, ID_forward, threshold_current)
    
    # For the backward sweep, sort by VG in ascending order for interpolation.
    backward_sorted = backward_df.sort_values(by="VG").reset_index(drop=True)
    VG_backward = backward_sorted["VG"].values
    ID_backward = backward_sorted["ID_norm"].values
    VG_cross_backward = interpolate_crossing(VG_backward, ID_backward, threshold_current)
    
    if VG_cross_forward is None or VG_cross_backward is None:
        return np.nan
    
    return abs(VG_cross_backward - VG_cross_forward)


def compute_device_FoMs(df, device_width=10):
    """
    If the data is "bad," return all NaNs. Otherwise, compute FoMs.
    """
    # 1) Check if data is "bad"
    if is_bad_data(df):
        return {
            "SS_min (V/dec)": np.nan,
            "VTH (V)": np.nan,
            "Hysteresis (V)": np.nan,
            "Gm_max @1V (µA/µm per V)": np.nan
        }

    # 2) Otherwise proceed as normal
    df = normalize_current(df, device_width)

    df_vdd_0_1 = df[df["VD"] == 0.1].copy()
    df_vdd_1 = df[df["VD"] == 1.0].copy()

    FoM1_SS = compute_SS(df_vdd_0_1)
    VTH, gm_max = compute_VTH_and_gm(df_vdd_0_1, 0.1)
    FoM2_VTH = VTH
    FoM3_hysteresis = compute_hysteresis(df_vdd_0_1, threshold_current=1e-4)
    _, gm_max_vdd1 = compute_VTH_and_gm(df_vdd_1, 1.0)
    FoM4_gm_max = gm_max_vdd1

    return {
        "SS_min (V/dec)": FoM1_SS,
        "VTH (V)": FoM2_VTH,
        "Hysteresis (V)": FoM3_hysteresis,
        "Gm_max @1V (µA/µm per V)": FoM4_gm_max
    }

# -------------------------
# 4) Main Processing
# -------------------------

if __name__ == "__main__":
    folder_path = r"D:\PL,\PL1"  # adjust as needed
    header_row_input = 267
    device_width = 10

    results = process_all_csv_files(folder_path, header_row=header_row_input)

    for entry in results:
        print(f"File: {entry['filename']}")
        print(f"  Device ID: {entry['device_id']}, (x={entry['x']}, y={entry['y']})")
        print(f"  Data shape: {entry['data'].shape}")
        FoMs = compute_device_FoMs(entry["data"], device_width)
        print("  Figures-of-Merit:")
        for key, value in FoMs.items():
            print(f"    {key}: {value}")
        print()
