import torch
import torch.nn as nn
import tifffile
import torch.nn.functional as F
from skimage.transform import resize
from util.lrx_inference_config_builder import build_configs
from util.subregion_extractor_inference import extract_subregion
from util.infer_thermal_stats import infer_thermal_stats
import subprocess
import pandas as pd
import re
import glob
from concurrent.futures import ThreadPoolExecutor
import os
import shutil
import argparse

class CNNClassifier(nn.Module):
    def __init__(self, output_channels):
        super(CNNClassifier, self).__init__()
        self.output_channels = output_channels
        self.conv1 = nn.Conv2d(1, output_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(output_channels, output_channels*2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(output_channels*2, output_channels*4, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(output_channels*4*25*25, 128)
        self.fc2 = nn.Linear(128, 1)

        self.bn1 = nn.BatchNorm2d(output_channels)
        self.bn2 = nn.BatchNorm2d(output_channels*2)
        self.bn3 = nn.BatchNorm2d(output_channels*4)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        x = x.view(-1, self.output_channels*4*25*25)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

output_channels = 16
model = CNNClassifier(output_channels)
model = model.to(torch.double)
state_dict = torch.load('model/model_618.pth', map_location=torch.device('cpu'))
new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)
model.eval()

def preprocess_image(image):
    image = resize(image, (200,200))
    image = (image - image.min()) / (image.max() - image.min())
    return image

def run_dora(volcano):
    file_list = sorted(os.listdir(f'configs_lrx_inference/{volcano}'))
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for file in file_list[2:]:
            print(f"Running LRX for {file}")
            file_path = os.path.join(f'configs_lrx_inference/{volcano}', file)
            futures.append(executor.submit(subprocess.run, ['dora_exp', file_path]))
            break
        for future in futures:
            future.result()

    print("All subprocesses completed LRX processing")

def lrx_process(raw_folder_path, lrx_folder_path, volcano, configs_dir):
    if configs_dir is None:
        configs_dir = f"configs_lrx_inference"
    configs_dir = f"{configs_dir}/{volcano}"

    os.makedirs(configs_dir, exist_ok=True)
    build_configs(raw_folder_path, lrx_folder_path, volcano)
    run_dora(volcano)

def split_into_words(input_string):
    words = re.split(r'\W+', input_string)
    words = [word for word in words if word]
    return words

def subregion_extractor(raw_folder_path, cropped_lrx_dir, gvp_volcano_name):
    volcano = raw_folder_path.split("/")[-2]
    volcanic_word_list = volcano.split("_")
    df = pd.read_csv("util/gvp_volcanic_coordinates.csv")
    if gvp_volcano_name is None:
        df['Volcano Name'] = df['Volcano Name'].astype(str)
        df['Volcano Name Words'] = df['Volcano Name'].apply(split_into_words)
        filtered_df = df[df['Volcano Name Words'].apply(lambda x: all(word in x for word in volcanic_word_list))]
    else:
        filtered_df = df[df['Volcano Name'] == gvp_volcano_name]
    lat = filtered_df.iloc[0]['Latitude']
    lon = filtered_df.iloc[0]['Longitude']
    extract_subregion(raw_folder_path+"all/", lat, lon, volcano, cropped_lrx_dir)

def delete_all_contents(directory):
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                else:
                    os.remove(file_path)
            except Exception as e:
                pass
    else:
        print(f'The directory {directory} does not exist.')

def predict(raw_folder_path, output_path, gvp_volcano_name, cropped_lrx_dir=None, lrx_folder_path=None, configs_dir=None):
    volcano = raw_folder_path.split("/")[-2]
    if cropped_lrx_dir is None:
        cropped_lrx_dir = f"extracted_subregions_inference"
    if lrx_folder_path is None:
        lrx_folder_path = f"lrx_for_inference"

    cropped_lrx_dir = f"{cropped_lrx_dir}/{volcano}"
    lrx_folder_path = f"{lrx_folder_path}/{volcano}/"
    
    os.makedirs(cropped_lrx_dir, exist_ok=True)
    os.makedirs(lrx_folder_path, exist_ok=True)
    delete_all_contents(cropped_lrx_dir)   
    subregion_extractor(raw_folder_path, cropped_lrx_dir, gvp_volcano_name)
    delete_all_contents(lrx_folder_path)
    lrx_process(cropped_lrx_dir+"/", lrx_folder_path, volcano, configs_dir)
    result = []
    df = pd.DataFrame()
    df["Date"] = ""
    df["Volcano"] = ""
    df["Volcanic Thermal Anomaly (Y/N)"] = ""
    df['Maximum (K)'] = ""
    df['Mean (Background Temperature) (K)'] = ""
    df['Standard Deviation'] = ""
    df['Max Temp Above Backround (K)'] = ""
    i = 0
    for lrx_image_path in os.listdir(lrx_folder_path):
        print(F"Inferring: {lrx_image_path}")
        try:
            lrx_filename = lrx_folder_path+lrx_image_path + "/lrx-bands=1-inner_window=3-outer_window=5/scores_raster_lrx.tif"
            file = tifffile.imread(lrx_filename)
            
            image = preprocess_image(file)
            
            image = torch.from_numpy(image)
            image = image.unsqueeze(0)
            image = image.to(dtype=torch.double)
            with torch.no_grad():
                output = model(image)
            
            probs = torch.sigmoid(output)
            predicted = (probs > 0.5).int()
            row = infer_thermal_stats(lrx_filename, raw_folder_path+"all/"+lrx_image_path, predicted.item(), volcano)
            df = df.append(row, ignore_index=True)

        except Exception as e:
            pass

    print(f"CSV has been exported to: {output_path}")
    volcano = raw_folder_path.split("/")[-2]
    df.to_csv(output_path)
    return

def main():
    parser = argparse.ArgumentParser(description="Process an image and output the result.")
    parser.add_argument('--input', help="Path to the input image file")
    parser.add_argument('--output', help="Path to save the processed image file")
    parser.add_argument('--gvp_volcano_name', help="[Optional] Name of the volcano in the GVP database")
    parser.add_argument('--cropped_lrx_dir', help="[Optional] Path to the cropped LRX directory")
    parser.add_argument('--lrx_folder_path', help="[Optional] Path to the LRX folder")
    parser.add_argument('--configs_dir', help="[Optional] Path to the LRX configs directory")
    
    
    args = parser.parse_args()
    predict(args.input, args.output, args.gvp_volcano_name, args.cropped_lrx_dir, args.lrx_folder_path)

if __name__ == '__main__':
    main()