# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 15:57:14 2025
Updated: 2025-12-02 by GPT-5 mini
Full integration: match files by current density in filename
@author: tpham + GPT5
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from datetime import datetime
import re

# ===== USER SETTINGS =====
CELL_AREA_CM2 = 25                     # Cell area in cm²
VF_COLUMNS = ('Vf', 'Vf1', 'Vf2')      # AE channels to check
FIT_POINTS = 5                          # Number of points for EIS linear fit
LAST_SECONDS = 300                      # Last X seconds for GSTATIC averaging
HEADER_MAP = {                          # New names for dataframe
    'Vf': 'Cell Potential (V)',
    'Vf1': 'Anode vs. RHE (V)',
    'Vf2': 'Cathode vs. RHE (V)'
}
INPUT_FOLDER = r"Y:\5900\HydrogenTechFuelCellsGroup\CO2R\Nhan P\Experiments\CO2 Cell Testing\TS2\2NP53_Break in at 0p3slpm\2NP53_2_Rinse 2_10h Conditioning 400mA_Break in 6"
RE = 0  # Reference electrode vs. RHE

# ===== PARSE LOGS =====
PARSE_LOG_GSTATIC = []
PARSE_LOG_GSTATIC_PARSED = []
PARSE_LOG_EIS = []
PARSE_LOG_EIS_PARSED = []

# ===== HELPER FUNCTIONS =====
def read_dta_table(file_path, start_marker):
    try:
        with open(file_path, 'r', encoding='latin1') as f:
            lines = f.readlines()
    except Exception:
        return pd.DataFrame()

    start_index = None
    for i, line in enumerate(lines):
        if start_marker in line:
            start_index = i + 1
            break

    if start_index is None or start_index + 1 >= len(lines):
        return pd.DataFrame()

    try:
        header = lines[start_index].strip().split('\t')
        data_lines = lines[start_index + 2:]
        data = [l.strip().split('\t') for l in data_lines if l.strip() != '']
        if not data:
            return pd.DataFrame(columns=header)
        df = pd.DataFrame(data, columns=header).apply(pd.to_numeric, errors='coerce')
        return df
    except Exception:
        return pd.DataFrame()

def parse_cd_from_filename(file):
    """Extract numeric current density before 'mA' in the filename."""
    match = re.search(r'(\d+)mA', file)
    return int(match.group(1)) if match else np.nan

# ===== STEP 1: Process GSTATIC =====
def process_gstatic(folder):
    results = []
    for file in os.listdir(folder):
        if not (file.startswith('PWRGSTATIC') and file.endswith('.DTA')):
            continue
        file_path = os.path.join(folder, file)
        df = read_dta_table(file_path, 'CURVE')

        if df.empty:
            PARSE_LOG_GSTATIC.append({'File': file, 'Reason': 'Unable to parse table', 'Timestamp': datetime.now()})
            continue

        missing_required = {'T', 'Im'} - set(df.columns)
        if missing_required:
            PARSE_LOG_GSTATIC.append({'File': file, 'Reason': f"Missing required columns: {missing_required}", 'Timestamp': datetime.now()})
            continue

        max_time = df['T'].max()
        df_last = df[df['T'] >= max_time - LAST_SECONDS]
        if df_last.empty:
            PARSE_LOG_GSTATIC.append({'File': file, 'Reason': f"No rows in last {LAST_SECONDS}s", 'Timestamp': datetime.now()})
            continue

        avg_current_density = round(abs(df_last['Im'].mean() / CELL_AREA_CM2) * 1000)
        cd_file = parse_cd_from_filename(file)

        row = {'File': file, 'Current Density (mA/cm²)': avg_current_density, 'CD_file': cd_file}

        for col in VF_COLUMNS:
            header_name = HEADER_MAP.get(col, col)
            std_name = header_name + ' Std'
            if col in df_last.columns:
                mean_val = np.nanmean(df_last[col].values)
                std_val = np.nanstd(df_last[col].values, ddof=1) if df_last.shape[0] > 1 else 0.0
                if col in ('Vf1', 'Vf2') and not np.isnan(mean_val):
                    mean_val -= RE
                if np.isnan(mean_val) or np.isnan(std_val):
                    mean_val = 0.0
                    std_val = 0.0
                mean_val = abs(mean_val)
            else:
                mean_val = 0.0
                std_val = 0.0
            row[header_name] = mean_val
            row[std_name] = std_val

        results.append(row)
        PARSE_LOG_GSTATIC_PARSED.append({'File': file, 'Current Density (mA/cm²)': avg_current_density, 'CD_file': cd_file, 'Rows in last window': len(df_last), 'Timestamp': datetime.now()})

    df_out = pd.DataFrame(results)
    output_path = os.path.join(folder, 'output')
    os.makedirs(output_path, exist_ok=True)
    gstatic_excel = os.path.join(output_path, 'GSTATIC_results.xlsx')
    df_out.to_excel(gstatic_excel, index=False)

    # Plot
    try:
        plt.figure(figsize=(8,6))
        x = df_out['CD_file']
        for col in VF_COLUMNS:
            header = HEADER_MAP.get(col, col)
            if header in df_out.columns:
                y = df_out[header]
                yerr = df_out.get(header + ' Std', None)
                plt.errorbar(x, y, yerr=yerr, fmt='o-', label=header)
        plt.xlabel('Current Density from filename (mA/cm²)')
        plt.ylabel('Potential (V)')
        plt.legend()
        plt.grid(True)
        plt.title(os.path.basename(folder))
        plt.savefig(os.path.join(output_path, f"{os.path.basename(folder)}_GSTATIC_plot.png"))
        plt.close()
    except Exception as e:
        PARSE_LOG_GSTATIC.append({'File':'PLOTTING','Reason':f"Plotting exception: {e}",'Timestamp':datetime.now()})

    return gstatic_excel

# ===== STEP 2: Process EIS =====
def process_eis(folder):
    def process_group(file_list):
        results = []
        for file in file_list:
            file_path = os.path.join(folder, file)
            df = read_dta_table(file_path, 'ZCURVE')
            if df.empty:
                PARSE_LOG_EIS.append({'File': file, 'Reason': 'Unable to parse ZCURVE', 'Timestamp': datetime.now()})
                continue
            required_cols = {'Zreal','Zimag','Idc','Freq'}
            missing = required_cols - set(df.columns)
            if missing:
                PARSE_LOG_EIS.append({'File':file,'Reason':f"Missing columns: {missing}",'Timestamp':datetime.now()})
                continue
            current_density = round(abs(df['Idc'].mean()/CELL_AREA_CM2*1000))
            cd_file = parse_cd_from_filename(file)
            df_filtered = df[(df['Freq']>=10000)&(df['Freq']<=100000)]
            if len(df_filtered)<2:
                hfr = np.nan
            else:
                slope, intercept, r_value, _, _ = linregress(df_filtered['Zreal'], df_filtered['Zimag'])
                hfr = -intercept/slope*CELL_AREA_CM2 if slope!=0 else np.nan
            results.append({'File':file,'Current Density (mA/cm²)':current_density,'CD_file':cd_file,'HFR (Ohm·cm²)':hfr,'R²':r_value**2 if len(df_filtered)>=2 else np.nan})
            PARSE_LOG_EIS_PARSED.append({'File':file,'Current Density (mA/cm²)':current_density,'CD_file':cd_file,'Timestamp':datetime.now()})
        return pd.DataFrame(results)

    full_cell_files, anode_files, cathode_files = [], [], []
    for file in os.listdir(folder):
        if file.startswith('PWRGEIS') and not file.endswith('Raw.DTA'):
            if file.endswith('AECH1.DTA'):
                anode_files.append(file)
            elif file.endswith('AECH2.DTA'):
                cathode_files.append(file)
            else:
                full_cell_files.append(file)

    df_full = process_group(full_cell_files)
    df_anode = process_group(anode_files)
    df_cathode = process_group(cathode_files)

    output_path = os.path.join(folder,'output')
    os.makedirs(output_path, exist_ok=True)
    eis_excel = os.path.join(output_path,'EIS_results.xlsx')
    with pd.ExcelWriter(eis_excel) as writer:
        if not df_full.empty: df_full.to_excel(writer, sheet_name='Full Cell', index=False)
        if not df_anode.empty: df_anode.to_excel(writer, sheet_name='Anode', index=False)
        if not df_cathode.empty: df_cathode.to_excel(writer, sheet_name='Cathode', index=False)

    return eis_excel

# ===== STEP 3: Merge Results =====
def merge_results(gstatic_file, eis_file):
    # Read GSTATIC results
    df1 = pd.read_excel(gstatic_file)
    if 'CD_file' not in df1.columns:
        raise KeyError("GSTATIC results missing 'CD_file' column.")

    # Read all sheets from EIS workbook and concatenate
    try:
        eis_sheets = pd.read_excel(eis_file, sheet_name=None)
        df2 = pd.concat(eis_sheets.values(), ignore_index=True) if eis_sheets else pd.DataFrame()
    except Exception:
        # Fallback: try reading first sheet only
        df2 = pd.read_excel(eis_file)

    # Merge on CD_file
    if 'CD_file' not in df2.columns:
        raise KeyError("EIS results missing 'CD_file' column.")
    merged = pd.merge(df1, df2, on='CD_file', how='inner')

    # Sort by CD_file ascending
    merged = merged.sort_values(by='CD_file').reset_index(drop=True)

    merged_excel = os.path.join(os.path.dirname(gstatic_file), 'Merged_results.xlsx')
    merged.to_excel(merged_excel, index=False)
    return merged_excel


# ===== STEP 4: HFR-free and plot =====
def add_hfr_free_and_plot(merged_file,folder):
    df = pd.read_excel(merged_file)
    if 'HFR (Ohm·cm²)' not in df.columns:
        df['HFR (Ohm·cm²)'] = np.nan
    df['HFR-free potential (V)'] = df['Cell Potential (V)'] - df['CD_file']*df['HFR (Ohm·cm²)']/1000
    output_path = os.path.join(folder,'output'); os.makedirs(output_path,exist_ok=True)
    updated_merged_excel = os.path.join(output_path,'Merged_results_HFRfree.xlsx')
    df.to_excel(updated_merged_excel,index=False)

    # Plot
    try:
        plt.figure(figsize=(8,6))
        x = df['CD_file']
        for col in VF_COLUMNS:
            header = HEADER_MAP.get(col,col)
            if header in df.columns:
                y = df[header]
                yerr = df.get(header+' Std',None)
                plt.errorbar(x,y,yerr=yerr,fmt='o-',label=header)
        plt.plot(x,df['HFR-free potential (V)'],marker='s',linestyle='--',label='HFR-free Cell Potential (V)')
        plt.xlabel('Current Density from filename (mA/cm²)')
        plt.ylabel('Potential (V)')
        plt.legend(); plt.grid(True)
        plt.title(os.path.basename(folder)+' — HFR-corrected')
        plt.savefig(os.path.join(output_path,f"{os.path.basename(folder)}_HFRfree_plot.png"))
        plt.close()
    except Exception as e:
        PARSE_LOG_GSTATIC.append({'File':'HFR_PLOTTING','Reason':f"Exception: {e}",'Timestamp':datetime.now()})

    print("HFR-free potential added and plotted. File saved at:",updated_merged_excel)
    return updated_merged_excel

# ===== WRITE PARSE REPORT =====
def write_parse_report(folder):
    output_path = os.path.join(folder,'output'); os.makedirs(output_path,exist_ok=True)
    report_path = os.path.join(output_path,'Parse_Report.xlsx')
    with pd.ExcelWriter(report_path) as writer:
        pd.DataFrame(PARSE_LOG_GSTATIC or [{'File':'','Reason':'','Timestamp':''}]).to_excel(writer,sheet_name='GSTATIC_Skipped',index=False)
        pd.DataFrame(PARSE_LOG_GSTATIC_PARSED or [{'File':'','Current Density (mA/cm²)':'','CD_file':'','Rows in last window':'','Timestamp':''}]).to_excel(writer,sheet_name='GSTATIC_Parsed',index=False)
        pd.DataFrame(PARSE_LOG_EIS or [{'File':'','Reason':'','Timestamp':''}]).to_excel(writer,sheet_name='EIS_Skipped',index=False)
        pd.DataFrame(PARSE_LOG_EIS_PARSED or [{'File':'','Current Density (mA/cm²)':'','CD_file':'','Timestamp':''}]).to_excel(writer,sheet_name='EIS_Parsed',index=False)
    print(f"Parse report saved to: {report_path}")
    return report_path

# ===== MAIN =====
if __name__ == "__main__":
    print("Starting processing for folder:", INPUT_FOLDER)
    gstatic_path = process_gstatic(INPUT_FOLDER)
    eis_path = process_eis(INPUT_FOLDER)
    parse_report_path = write_parse_report(INPUT_FOLDER)
    merged_path = merge_results(gstatic_path,eis_path)
    print("Processing complete. Merged file saved at:", merged_path)
    add_hfr_free_and_plot(merged_path,INPUT_FOLDER)
    print("Done. Parse report:", parse_report_path)