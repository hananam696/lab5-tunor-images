import argparse
import os
import pandas as pd
from skimage import io
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--input_images", type=str)
parser.add_argument("--output_features", type=str)
args = parser.parse_args()

image_folder = args.input_images
features = []

for img_name in os.listdir(image_folder):
    path = os.path.join(image_folder, img_name)
    try:
        img = io.imread(path)
        features.append({
            "image": img_name,
            "mean": np.mean(img),
            "std": np.std(img),
        })
    except Exception as e:
        print("Skipping:", path, "-", str(e))

df = pd.DataFrame(features)

os.makedirs(args.output_features, exist_ok=True)
df.to_parquet(os.path.join(args.output_features, "features.parquet"))
