import os
import subprocess
import argparse

parser = argparse.ArgumentParser(description="Run the VTF detection pipeline for all vocanoes in a specific volcanic region.")
parser.add_argument("--region", type=str, required=True,
                    help="Volcanic region to process (e.g., Antarctica, Africa_and_Red_Sea, Alaska, etc.)")

args = parser.parse_args()
volcanic_region = args.region

# volcanic_region = "Antarctica" 
#["Antarctica", "Africa_and_Red_Sea", "Alaska", "Atlantic_Ocean", "Canada_and_Western_USA", 
# "Hawaii_and_Pacific_Ocean", "Iceland_and_Arctic_Ocean", "Indonesia", "Japan_Taiwan_Marianas",
# "Kamchatka_and_Mainland_Asia", "Kuril_Islands", "Mediterranean_and_Western_Asia",
# "Melanesia_and_Australia", "Mexico_and_Central_America","Middle_East_and_Indian_Ocean"
# "New_Zealand_to_Fiji", "Philippines_and_SE_Asia", "South_America", "West_Indies"]
base_dir = os.path.join('../data/CloudFree_ASTER_files', volcanic_region)
output_base_dir = os.path.join('./results', volcanic_region)

current_directory = os.getcwd()
print(current_directory)

# Ensure output directory exists
os.makedirs(output_base_dir, exist_ok=True)

# Loop through all directories within volcanic region
for folder in os.listdir(base_dir):
    folder_str = os.path.join(folder, '')
    folder_path = os.path.join(base_dir, folder_str)
    if os.path.isdir(folder_path):
        output_file = os.path.join(output_base_dir, f"{folder}_res.csv")
        if not os.path.exists(output_file):
            cmd = [
                'python3', 'vtf_app.py',
                f'--input={folder_path}',
                f'--output={output_file}'
            ]
            print(f'Running: {" ".join(cmd)}')
            subprocess.run(cmd, check=True)
        else:
            print(f'Skipping {folder}, results already exist.')

print('Processing complete!')