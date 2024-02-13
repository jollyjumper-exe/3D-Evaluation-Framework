import sys
import os
import argparse
import json
import shutil
import glob

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'scripts'))
sys.path.append(module_path)
import export_camera_path as export_cp
import remove_from_zip as remove_zip

#Command-line Arguments
parser = argparse.ArgumentParser(description='A script with command-line arguments.')
parser.add_argument('-scene', type=str, help='Specify the scale value.')
parser.add_argument('-index', type=int, help='Specify the index that should be removed.')

args = parser.parse_args()
scene = args.scene
index = args.index

input_folder = f'input/{scene}'
raw_folder = f'{input_folder}/raw'
working_folder = f'{input_folder}/working/{index}'
eval_folder = f'{working_folder}/eval'

# create necessary folders and files
if not os.path.exists(working_folder):
    os.makedirs(working_folder)

if not os.path.exists(eval_folder):
    os.makedirs(eval_folder)

raw_info_path = f'{raw_folder}/info.json'
# Open the JSON file
with open(raw_info_path, 'r') as file:
    data = json.load(file)

for i, matrix in enumerate(data['matrices'], start=0):
    # Extract the key and the value
    _, m = next(iter(matrix.items()))
    if i == index: break
    i = None

export_cp.export(m, f'{eval_folder}/camerapath.json')

zip_file = glob.glob(os.path.join(raw_folder, '*.zip'))[0]
file_name = os.path.basename(zip_file)
new_zip_file = os.path.join(working_folder, file_name)
shutil.copy(zip_file, new_zip_file)

remove_zip.process(new_zip_file, index, eval_folder)