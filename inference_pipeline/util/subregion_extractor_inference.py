import pandas as pd
from os import path
from rasterio import open as rio_open
import rasterio
from rasterio.warp import transform
from rasterio.transform import Affine 
import numpy as np
import os
from rasterio.windows import Window
def get_image_coordinates(path, lat, lon):
    
    with rasterio.open(path) as src:
        crs = src.crs
        transform = src.transform
        x_geo, y_geo = lon, lat

        # Reproject geographic coordinates to match raster CRS
        x_raster, y_raster = rasterio.warp.transform(rasterio.crs.CRS.from_epsg(4326), crs, [x_geo], [y_geo])

        # Convert reprojected coordinates to pixel coordinates
        col, row = ~transform * (x_raster[0], y_raster[0])
        #lon, lat = pyproj.Transformer.from_crs("EPSG:4326", src.crs).transform(lon, lat)
        # Convert to integer values (row and column indices are usually integers)
        col = int(col)
        row = int(row)
        # If the coordinates fall outside the raster, handle as needed
        if row < 0 or row >= src.height or col < 0 or col >= src.width:
            raise ValueError("Given coordinates are outside the raster extent.")

    return row,col

def extract_patch(image_array, center_x, center_y, patch_size):
    """
    Extract a fixed-size patch (patch_size x patch_size) centered at (center_x, center_y)
    from the input image_array. If the patch extends beyond the image boundaries,
    pad the patch with zeros.

    Parameters:
    - image_array: 2D NumPy array representing the image.
    - center_x: x-coordinate of the center pixel.
    - center_y: y-coordinate of the center pixel.
    - patch_size: Size of the square patch to extract 

    Returns:
    - patch: Extracted patch from the image.
    """
    half_patch_size = patch_size // 2

    # Calculate the bounding box for the patch
    top = max(center_x - half_patch_size, 0)
    bottom = min(center_x + half_patch_size, image_array.shape[0] - 1)
    left = max(center_y - half_patch_size, 0)
    right = min(center_y + half_patch_size, image_array.shape[1] - 1)

    top_pad = max(0, half_patch_size - center_x)
    bottom_pad = max(0, center_x + half_patch_size - image_array.shape[0] + 1)
    left_pad = max(0, half_patch_size - center_y)
    right_pad = max(0, center_y + half_patch_size - image_array.shape[1] + 1)

    # Pad the image and extract the patch
    padded_image = np.pad(image_array, ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant', constant_values=np.average(image_array))
    patch = padded_image[top:bottom + top_pad + bottom_pad, left:right + left_pad + right_pad]

    return patch

def extract_single_subregion(fname, lat, lon, volcano, results_folder):


  # Open the tif image
  with rio_open(fname) as src:
    # Get image metadata
    transform = src.transform
    crs = src.crs
    data = src.read(1)  # assuming a single band image

    row, col = get_image_coordinates(fname, lat, lon)

    patch_size = 200
    half_patch_size = 100
    min_row = max(row - half_patch_size, 0)
    min_col = max(col - half_patch_size, 0)
    max_row = min(row + half_patch_size, src.height)
    max_col = min(col + half_patch_size, src.width)
    patch_width = max_col - min_col
    patch_height = max_row - min_row

    # Read the patch from the TIF file
    patch = src.read(window=Window.from_slices((min_row, max_row), (min_col, max_col)))


    # window = Window(col - half_patch_size, row - half_patch_size, patch_size, patch_size)

    # # Read the patch from the TIF file
    # patch = src.read(window=window)

    metadata = src.meta.copy()
    metadata['width'], metadata['height'] = patch_width, patch_height
    metadata['transform'] = rasterio.windows.transform(Window(min_col, min_row, patch_width, patch_height), src.transform)
    
    basename = path.basename(fname).split("/")[-1]
    
    output_filename = path.join(results_folder, basename)
    with rasterio.open(output_filename, 'w', **metadata) as dst:
        dst.write(patch)


def extract_subregion(folder_path, lat, lon, volcano, results_folder):

    for filename in os.listdir(folder_path):
        try:
            extract_single_subregion(folder_path+filename, lat, lon, volcano, results_folder)
        
        except Exception as e:
            pass
        