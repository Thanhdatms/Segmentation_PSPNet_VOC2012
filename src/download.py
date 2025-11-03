import kagglehub
import os
import shutil

path = kagglehub.dataset_download("huanghanchina/pascal-voc-2012")
print("Downloaded dataset path:", path)

target_dir = os.path.join(os.getcwd(), "./datasets", "pascal-voc-2012")
os.makedirs(target_dir, exist_ok=True)

if os.path.exists(target_dir):
    shutil.copytree(path, target_dir, dirs_exist_ok=True)
    print("✅ Dataset successfully copied to:", target_dir)
else:
    print("❌ Target directory not found or could not be created.")
