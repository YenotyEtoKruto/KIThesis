"""
This script is used to unzip the individual AHN files from a folder
"""
import zipfile
import os

input_folder = "E:/AHN-zipped"
output_folder = "D:/AHN"

for file_name in os.listdir(input_folder):
    if file_name.endswith(".zip"):
        base_name = os.path.splitext(file_name)[0]
        expected_tif = os.path.join(output_folder, base_name + ".tif")

        if os.path.exists(expected_tif):
            continue

        zip_path = os.path.join(input_folder, file_name)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_folder)
            print(f"Extracted {file_name}")

print("Done unpacking all zip files.")
