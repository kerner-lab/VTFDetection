import glob
import os
import shutil
from copy import deepcopy
from datetime import datetime
from glob import iglob
from itertools import chain, product

import geopandas as gpd
import numpy as np
import openpyxl
import pandas as pd
import rasterio
import skimage.measure
from affine import Affine
from openpyxl import Workbook
from pyproj import Proj, transform
from rasterio.warp import transform
from shapely.geometry import Polygon
from sklearn.cluster import KMeans


def calculate_mean_top_n(arr, n):
    """
    Calculate the mean of the top n values in a given array.
    Args:
        arr (ndarray): Input array.
        n (int): Number of top values to consider.
    Returns:
        float: Mean of the top n values.
    """
    sorted_list = sorted(arr.flatten(), reverse=True)[:n]
    mean = np.mean(sorted_list)
    return mean


def calculate_score(arr, b_inner=2, top_n=6, bottom_n=50):
    """
    Calculate the score based on the mean of the top n values and the mean of the bottom n values.
    Args:
        arr (ndarray): Input array.
        b_inner (int): Inner boundary for calculation.
        top_n (int): Number of top values to consider for mean calculation.
        bottom_n (int): Number of bottom values to consider for mean calculation.
    Returns:
        float: Score calculated based on the means.
    """
    sum_b = []
    for i0 in range(0, arr.shape[0]-1):
        for j0 in range(0, arr.shape[1]-1):
            if (i0 + b_inner < arr.shape[0] and i0 - b_inner >= 0 and j0 + b_inner < arr.shape[1] and j0 - b_inner >= 0):
                inner_sum = np.nansum(arr[i0-b_inner:i0+b_inner, j0-b_inner:j0+b_inner])
                sum_b.append([inner_sum, i0, j0])
            
    mean_top_n = calculate_mean_top_n(np.array(sum_b)[:, 0], top_n)
    mean_bottom_n = calculate_mean_top_n(np.array(sum_b)[:, 0], bottom_n)
    score = 1 - (mean_bottom_n / mean_top_n)
    
    return score


def convert_pixel_to_geo_coordinates(pixel_coords, transform, crs):
    """
    Convert pixel coordinates to geographic coordinates.
    Args:
        pixel_coords (tuple): Pixel coordinates (x, y).
        transform (Affine): Affine transformation.
        crs (CRS): Coordinate Reference System.
    Returns:
        tuple: Geographic coordinates (longitude, latitude).
    """
    x_geo, y_geo = transform * pixel_coords
    crs_decimal_degrees = rasterio.crs.CRS.from_epsg(4326)
    x_dd, y_dd = rasterio.warp.transform(crs, crs_decimal_degrees, [x_geo], [y_geo])
    return x_dd[0], y_dd[0]


def calculate_intersection(convex_hulls):
    """
    Calculate the intersection of multiple convex hulls.
    Args:
        convex_hulls (GeoDataFrame): GeoDataFrame containing convex hull geometries.
    Returns:
        Polygon: Intersection polygon.
    """
    intersection = convex_hulls.geometry.unary_union.convex_hull
    return intersection


def get_decision_score(csv_path, output_csv_path):
    """
    Get decision scores and save the results to a CSV file.
    Args:
        csv_path (str): Path to the input CSV file.
        output_csv_path (str): Path to save the output CSV file.
    """
    df = pd.read_csv(csv_path)
    df_pred_list = []
    df_intersection = []
    df_area = []

    for i in range(df.shape[0]):
        print(df.iloc[i]['fname'])
        split_str = df.iloc[i]['fname'].split("/")
        
        if df.iloc[i]["Ground_Truth"] == 0:
            dst_path = "_no_instances_all/" + split_str[len(split_str)-3]
        else:
            dst_path = "_yes_instances_all/" + split_str[len(split_str)-3]
            
        try:
            with rasterio.open(dst_path) as r:
                print(dst_path)
                arr = r.read(1)
                print(arr.shape)

                score = calculate_score(arr)
                df_pred_list.append(score)

                transform = r.transform
                crs = r.crs
                geo_list_1 = []
                geo_list_2 = []

                for k in range(len(sorted_list)):
                    i1 = sorted_list[k][1]
                    j1 = sorted_list[k][2]

                    polygon_indices = [
                        [i1-b_inner, j1], [i1, j1-b_inner],
                        [i1, j1+b_inner], [i1+b_inner, j1]
                    ]

                    for l in polygon_indices:
                        i2 = l[0]
                        j2 = l[1]
                        geo_coords = convert_pixel_to_geo_coordinates((j2, i2), transform, crs)
                        geo_list_1.append(geo_coords)

                    points = geo_list_1
                    gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy([p[0] for p in points], [p[1] for p in points]))

                    convex_hull = gdf.geometry.unary_union.convex_hull
                    geo_list_2.append(convex_hull)
                    geo_list_1 = []

                print("len ", len(geo_list_2))
                print(geo_list_2)

                gdf = gpd.GeoDataFrame(geometry=geo_list_2)
                intersection = calculate_intersection(gdf)

                print("\nIntersection: ", intersection)
                print("Polygon: ", str(intersection))
                
                df.loc[i, 'Intersection'] = str(intersection)
                area_intersection = intersection.area.sum()
                print("Area: ", area_intersection)
                df_area.append(area_intersection)
                df.loc[i, 'Area'] = area_intersection
                df.loc[i, 'Prediction'] = 1 - (score / (score + mean))
                print(df.shape)
                geo_list_2 = []

        except Exception:
            pass

    print("df list ", df_pred_list)
    print("pred: ", df.shape)

    df.to_csv(output_csv_path)



input_csv_path = '/home/amohan62/vtfs/merged_df.csv'
output_csv_path = '/home/amohan62/vtfs/merged_df_pred4.csv'
get_decision_score(input_csv_path, output_csv_path)
