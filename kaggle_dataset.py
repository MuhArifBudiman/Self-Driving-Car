import kagglehub
import os

cwd = os.getcwd()

# Download latest version
path = kagglehub.dataset_download("shuvoalok/cityscapes", path=cwd)


print("Path to dataset files:", path)
# print(cwd)
