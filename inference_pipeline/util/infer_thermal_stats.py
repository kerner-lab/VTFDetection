import tifffile
import numpy as np
import matplotlib.pyplot as plt
from rasterio import open as rio_open
import rasterio
from rasterio.warp import transform
from rasterio.transform import Affine 
import numpy as np
import pandas as pd
import os
from rasterio.windows import Window
from dateutil import parser
from rasterio.transform import rowcol

def formatDate_classify(fname):
    file_type = "/all" # cloud-free or all
    base_path =  fname.split(file_type)[0] #'/scratch/amohan62/915volcano_data/'
    date_str = fname.split(base_path + file_type+"/AST_08_003",1)[1][0:2] + '-' + fname.split(base_path+file_type+"/AST_08_003",1)[1][2:4] + '-' + fname.split(base_path +file_type+"/AST_08_003",1)[1][4:8]
    DT = parser.parse(date_str)
    return DT

def infer_thermal_stats(lrx_filename,original_path,predicted,volcano):
    # Open the raster file
    image_a = lrx_filename
    image_b = original_path
    date = formatDate_classify(original_path)
    new_row = {'Date': date, 'Volcano': volcano,'Volcanic Thermal Anomaly (Y/N)': 'N', 'Maximum (K)': '', 'Mean (Background Temperature) (K)': '','Standard Deviation': '', 'Max Temp Above Background (K)' : '', 'Filename' : original_path}
    if(predicted == 1):
        try:
            with rasterio.open(image_b) as src:
                crs = src.crs
                transform = src.transform
                image = src.read(1)
                row,col = np.unravel_index(np.argmax(image), image.shape)

            with rasterio.open(image_b) as src:
                sorted_scores_df = lrx_filename.rsplit('/', 1)[0] + "/selections-lrx.csv"
                df = pd.read_csv(sorted_scores_df, header=None, names=["Index", "Value1", "RowCol", "Value2"])
                crs = src.crs
                # transform = src.transform
                # x_geo, y_geo = lon, lat
                image = src.read(1)
                def get_patch(image, row, col, size=10):
                    half_size = size // 2
                    return image[row - half_size:row + half_size + 1, col - half_size:col + half_size + 1]

                # Initialize the list to store pixel values
                pixel_values = []

                # Iterate through each row in the DataFrame
                i=0
                for _, row in df.iterrows():
                    i+=1
                    # Extract row and col values from the RowCol column
                    row_col = row['RowCol']
                    row, col = map(int, row_col.split('-'))
                    image = image/10.0
                    with rasterio.open(image_a) as lrx_src:
                        lrx_crs = lrx_src.crs
                    
                    x_raster, y_raster = lrx_src.transform * (col, row)
                    lon, lat = rasterio.warp.transform(lrx_crs, rasterio.crs.CRS.from_epsg(4326), [x_raster], [y_raster])
                    x_raster, y_raster = rasterio.warp.transform(rasterio.crs.CRS.from_epsg(4326), crs, lon, lat)
                    row, col = rowcol(src.transform,x_raster[0],y_raster[0] )
                    
                    # Get the pixel value at the specified row and col
                    #pixel_value = image[row, col]
                    
                    # Get a 10x10 patch around the pixel
                    patch = get_patch(image, row, col)
                    pixel_value = np.max(patch)
                    
                    # Calculate the average value of the patch
                    patch_avg = np.mean(patch)

                    value_at_mapped_coord = np.max(patch)
                    bg_arr = patch[patch != np.max(patch)]
                    # new_row = {'Date': date, 'Volcano': volcano,'Volcanic Thermal Anomaly (Y/N)': 'Y', 'Maximum (K)': value_at_mapped_coord, 'Mean (Background Temperature) (K)': np.mean(bg_arr),'Standard Deviation':np.std(patch), 'Max Temp Above Backround (K)' : np.max(patch) - np.min(patch)}
                    new_row = {'Date': date, 'Volcano': volcano,'Volcanic Thermal Anomaly (Y/N)': 'Y', 'Maximum (K)': value_at_mapped_coord, 'Mean (Background Temperature) (K)': np.mean(bg_arr),'Standard Deviation':np.std(patch), 'Max Temp Above Background (K)' : np.max(patch) - np.mean(bg_arr), 'Filename': original_path}
                    return new_row
        except Exception as e:
            print(f"exception : {e}")
            pass
    return new_row

