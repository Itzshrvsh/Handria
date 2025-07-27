import os

# Path to the dataset folder
dataset_path = "dataset1"

for folder_name in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder_name)

    # Only target folders A to Z
    if os.path.isdir(folder_path) and folder_name.islower() and len(folder_name) == 1:
        new_folder_name = folder_name + "_lower"
        new_folder_path = os.path.join(dataset_path, new_folder_name)
        os.rename(folder_path, new_folder_path)
        print(f"Renamed: {folder_name} -> {new_folder_name}")
