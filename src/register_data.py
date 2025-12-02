# src/register_data.py

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Data
import os

# -------- Azure ML Workspace info --------
SUBSCRIPTION_ID = "a00dcbea-fd05-4973-82dc-120208b60116"
RESOURCE_GROUP = "rg-60104758"
WORKSPACE_NAME = "lab5_60104758"
DATA_NAME = "tumor_images_raw"
LOCAL_DATASET = "./brain_tumor_dataset"  # fixed path
# ----------------------------------------

# Sanity check: make sure Python can see the folders
print("Local dataset exists:", os.path.exists(LOCAL_DATASET))
print("Yes folder files:", os.listdir(os.path.join(LOCAL_DATASET, "yes")))
print("No folder files:", os.listdir(os.path.join(LOCAL_DATASET, "no")))

# Connect to Azure ML
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id=SUBSCRIPTION_ID,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WORKSPACE_NAME
)

# Define the data asset using local path
data_asset = Data(
    name=DATA_NAME,
    type="uri_folder",
    path=LOCAL_DATASET,  # <-- correct parameter
    description="Raw MRI tumor images (yes/no)",
    tags={"layer": "bronze"}
)

# Register (this will automatically upload local files)
registered_data = ml_client.data.create_or_update(data_asset)
print(f"Data asset registered: {registered_data.name}, version: {registered_data.version}")
