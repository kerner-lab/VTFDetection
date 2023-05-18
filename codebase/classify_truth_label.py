import glob
import os
import shutil
from copy import deepcopy
from datetime import datetime
from glob import iglob
from itertools import chain, product

import numpy as np
import openpyxl
import pandas as pd
import rasterio
from affine import Affine
from dateutil import parser
from openpyxl import Workbook
from pyproj import Proj, transform
from sklearn.cluster import KMeans


def formatDate_classify(fname, vol, base_path):
    """
    Extract the date from the filename and format it.
    Args:
        fname (str): Filename.
        vol (str): Volcano name.
        base_path (str): Base path of the filename.
    Returns:
        datetime: Formatted date.
    """
    # Extract date from filename in the correct format
    date_str = fname.split(base_path + vol + "/AST_08_003", 1)[1][0:2] + '-' + \
        fname.split(base_path + vol + "/AST_08_003", 1)[1][2:4] + '-' + \
        fname.split(base_path + vol + "/AST_08_003", 1)[1][4:8]
    DT = parser.parse(date_str)
    return DT


def process_file_list(file_list, res_path, table_path):
    """
    Process the list of files and create a merged DataFrame.
    Args:
        file_list (list): List of file paths.
        res_path (str): Path to the result files.
        table_path (str): Path to the processed xlsx file.
    Returns:
        DataFrame: Merged DataFrame.
    """
    df = pd.DataFrame(columns=['Date', 'Ground_Truth'])
    df_dateInfo = pd.DataFrame(columns=['Date', 'fname', 'Volcano'])
    df_list_date = []
    df_list_th = []

    vol_before = ""
    new_vol_flag = False

    for fname in file_list:
        print("File:", fname)
        str1 = fname.split(res_path)[1]
        vol = str1.split("/")[0]

        if vol != vol_before:
            new_vol_flag = True
        try:
            formatted_date = formatDate_classify(fname, vol, res_path)
            split_str = fname.split("/")
            df_dateInfo.loc[-1] = [formatted_date, fname, vol]
            df_dateInfo.index = df_dateInfo.index + 1

        except Exception:
            pass

        if new_vol_flag:
            new_vol_flag = False
            print(vol)
            vol_before = vol
            wb = openpyxl.load_workbook(table_path)
            title = ""

            for sheet in wb.worksheets:
                if sheet.title.lower() == vol.lower():
                    title = sheet.title
            print("Volcano:", title)
            try:
                print(table_path)
                data_table = pd.read_excel(table_path, sheet_name=[title], engine='openpyxl')
                df_list_date.append(list(data_table.get(title).get('Date')))
                df_list_th.append(list(data_table.get(title).get('Thermal Anomaly (Y/N)')))

            except Exception:
                print("--------Exception at file:", fname)
                pass

    df_list_date_fl = list(chain.from_iterable(df_list_date))
    df_list_th_fl = list(chain.from_iterable(df_list_th))

    df['Date'] = df_list_date_fl
    df['Ground_Truth'] = df_list_th_fl
    df['Ground_Truth'].replace(
        ['No Thermal Feature', 'No Feature', 'N', 'Y', 'No Thermal Features',
         'Maybe (checking currently)', 'N '],
        [0, 0, 0, 1, 0, 1, 0], inplace=True)

    col_name = 'Date'
    merged_df = pd.DataFrame()
    merged_df = pd.merge(df, df_dateInfo, on=col_name, how='left')
    merged_df = merged_df.dropna(subset=['fname'])

    merged_df.to_csv("/home/amohan62/vtfs/merged_df.csv")

    print(merged_df['Ground_Truth'].value_counts())
    print("Merged DF Shape:", merged_df.shape)

    return merged_df


def copy_files_to_directories(merged_df):
    """
    Copy files to respective directories based on Ground_Truth values.
    Args:
        merged_df (DataFrame): Merged DataFrame.
    """
    for i in range(0, merged_df.shape[0]):
        if merged_df.iloc[i]['Ground_Truth'] == 1 and merged_df.iloc[i]['fname'] == merged_df.iloc[i]['fname']:
            table_path = "_yes_instances_all/"
            if not os.path.exists(table_path):
                os.mkdir(table_path)
            split_str = merged_df.iloc[i]['fname'].split("/")
            extracted_fname = split_str[len(split_str) - 3]
            print(extracted_fname)

            shutil.copy(merged_df.iloc[i]['fname'], table_path + extracted_fname)

        elif merged_df.iloc[i]['Ground_Truth'] == 0 and merged_df.iloc[i]['fname'] == merged_df.iloc[i]['fname']:
            table_path = "_no_instances_all/"
            if not os.path.exists(table_path):
                os.mkdir(table_path)
            split_str = merged_df.iloc[i]['fname'].split("/")
            extracted_fname = split_str[len(split_str) - 3]
            print(extracted_fname)

            shutil.copy(merged_df.iloc[i]['fname'], table_path + extracted_fname)


def main():
    """
    Main function to execute the workflow.
    """
    # Path to the processed xlsx
    table_path = "/home/amohan62/vtfs/Merged_tables_processed.xlsx"
    res_path = "/home/amohan62/vtfs/results_all/lrx_mod/5/"
    rootdir_glob = res_path + '**'

    file_list = [f for f in iglob(rootdir_glob, recursive=True) if os.path.isfile(f) and f.endswith(".tif")]

    merged_df = process_file_list(file_list, res_path, table_path)
    copy_files_to_directories(merged_df)


if __name__ == "__main__":
    main()
