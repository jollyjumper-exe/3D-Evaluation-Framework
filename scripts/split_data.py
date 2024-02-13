import sys
import os
import json
import shutil
import glob

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'scripts'))
sys.path.append(module_path)
import export_camera_path as export_cp
import remove_from_zip as remove_zip

def split_data(scene, index):
    input_folder = f'input/{scene}'
    raw_folder = f'{input_folder}/raw'
    working_folder = f'{input_folder}/working/{index}'
    image_folder = f'images/{scene}/original'

    # create necessary folders and files
    if not os.path.exists(working_folder):
        os.makedirs(working_folder)
    
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    raw_info_path = f'{raw_folder}/info.json'
    # Open the JSON file
    with open(raw_info_path, 'r') as file:
        data = json.load(file)

    for i, matrix in enumerate(data['matrices'], start=0):
        # Extract the key and the value
        _, m = next(iter(matrix.items()))
        if i == index: break
        i = None

    export_cp.export(m, f'{working_folder}/camerapath.json')

    zip_file = glob.glob(os.path.join(raw_folder, '*.zip'))[0]
    file_name = os.path.basename(zip_file)
    new_zip_file = os.path.join(working_folder, file_name)
    shutil.copy(zip_file, new_zip_file)

    
    remove_zip.process(new_zip_file, index, image_folder)