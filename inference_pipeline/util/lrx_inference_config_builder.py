import glob
import os
import yaml
from itertools import product
from copy import deepcopy
import shutil

algs = [
    {'lrx': {
        'inner_window': 3,
        'outer_window': 5,
        'bands': 1
    }}
]

conv_patches = [5]

base = {
    'data_loader': {
        'name': 'raster_patches',
        'params': {
            'stride': 1,
            'nodata': 2000
        }
    },
    'zscore_normalization': False,
    'features': {
        'flattened_pixel_values': {
            'normalize_pixels': False
        }
    },
    'top_n': 10,
    'outlier_detection': {},
    'results': {
        'save_scores': {},
        'reshape_raster': {
            'colormap': 'magma',
            'data_format': 'patches'
        }
    }
}

def delete_all_contents(directory):
    if os.path.exists(directory):
        # Iterate over the contents of the directory
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                pass
    else:
        print(f'The directory {directory} does not exist.')


def build_configs(folder_path,lrx_folder_path,volcano):
    configs_directory = f"configs_lrx_inference/{volcano}"
    delete_all_contents(configs_directory)
    data = glob.glob(folder_path+"*")
    patches = [5]
    for i, params in enumerate(product(patches, data, algs)):
        patch_size, volcano_path, alg = params
        volcano = os.path.basename(volcano_path)
        tif_path = os.path.join(volcano_path)
        shp_path = os.path.join(volcano_path)
        config = deepcopy(base)
        config['out_dir'] = os.path.join(lrx_folder_path, volcano)
        config['data_loader']['params']['patch_size'] = patch_size
        config['data_to_fit'] = tif_path
        config['data_to_score'] = tif_path
        config['outlier_detection'].update(alg)
        config['results']['reshape_raster'].update({
            'raster_path': tif_path,
            'patch_size': patch_size
        })
        os.makedirs(configs_directory, exist_ok=True)
        with open(f'{configs_directory}/{i}.yml', 'w') as f:
            f.write(yaml.dump(config))


if __name__ == '__main__':
    main()
