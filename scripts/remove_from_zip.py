import zipfile
import os
import shutil

def process(zip_file, index, destination):
    # Get Subfolder
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall("temp_extracted")

    matching_folders = []
    for root, dirs, files in os.walk("temp_extracted"):
        for dir_name in dirs:
            if dir_name.endswith('-poly'):
                matching_folders.append(os.path.join(root, dir_name))
    
    poly_folder = matching_folders[0]

    subfolder_name = f'{poly_folder}/keyframes/images'

    subfolder_path = os.path.join(subfolder_name)

    files_in_subfolder = sorted(os.listdir(subfolder_path))

    if index >= 0 and index < len(files_in_subfolder):
        file_to_delete = os.path.join(subfolder_path, files_in_subfolder[index])

        if os.path.exists(file_to_delete):
            shutil.copy(file_to_delete, os.path.join(destination, f'{index}.jpg'))
            os.remove(file_to_delete)
            print(f"Deleted {file_to_delete}")
    
    # Delete Files
            
    subfolder_names = [f'{poly_folder}/keyframes/images']
    subfolder_names.append(f'{poly_folder}/keyframes/cameras')
    subfolder_names.append(f'{poly_folder}/keyframes/corrected_images')
    subfolder_names.append(f'{poly_folder}/keyframes/corrected_cameras')
    subfolder_names.append(f'{poly_folder}/keyframes/confidence')
    subfolder_names.append(f'{poly_folder}/keyframes/depth')

    for subfolder_name in subfolder_names: 
        subfolder_path = os.path.join(subfolder_name)

        files_in_subfolder = sorted(os.listdir(subfolder_path))

        if index >= 0 and index < len(files_in_subfolder):
            file_to_delete = os.path.join(subfolder_path, files_in_subfolder[index])

            if os.path.exists(file_to_delete):
                os.remove(file_to_delete)
                print(f"Deleted {file_to_delete}")
    
    with zipfile.ZipFile(zip_file, 'w') as zip_ref:
        for root, dirs, files in os.walk("temp_extracted"):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, "temp_extracted")
                zip_ref.write(file_path, arcname)

        print("Zip file updated")
    
    # Clean up extracted files
    shutil.rmtree("temp_extracted")

    
def move_nth_file_in_subfolder(zip_file_path, subfolder_name, file_index, destination=None):
    # Extract the contents of the zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall("temp_extracted")

    # Construct the path to the subfolder
    subfolder_path = os.path.join("temp_extracted", subfolder_name)

    # Get a list of all files in the subfolder
    files_in_subfolder = sorted(os.listdir(subfolder_path))

    # Check if the file index is valid
    if file_index >= 0 and file_index < len(files_in_subfolder):
        # Construct the path to the file to be deleted
        file_to_delete = os.path.join(subfolder_path, files_in_subfolder[file_index])

        # Check if the file exists
        if os.path.exists(file_to_delete):
            # Delete the file
            if destination != None : shutil.copy(file_to_delete, os.path.join(destination, 'image.jpg'))
            os.remove(file_to_delete)
            print(f"Deleted {file_to_delete}")

            # Re-create the zip file
            with zipfile.ZipFile(zip_file_path, 'w') as zip_ref:
                for root, dirs, files in os.walk("temp_extracted"):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, "temp_extracted")
                        zip_ref.write(file_path, arcname)

            print("Zip file updated")
        else:
            print(f"File {file_to_delete} not found")
    else:
        print(f"Invalid file index: {file_index}")

    # Clean up extracted files
    shutil.rmtree("temp_extracted")


def find_folder_with_suffix_in_zip(zip_file_path, suffix):
    # Extract the contents of the zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall("temp_extracted")

    # Search for folders with the specified suffix in the extracted contents
    matching_folders = []
    for root, dirs, files in os.walk("temp_extracted"):
        for dir_name in dirs:
            if dir_name.endswith(suffix):
                matching_folders.append(os.path.join(root, dir_name))

     # Clean up extracted files
    shutil.rmtree("temp_extracted")

    # If there is exactly one matching folder, return its name, otherwise return None
    if len(matching_folders) == 1:
        return matching_folders[0]
    else:
        return None
