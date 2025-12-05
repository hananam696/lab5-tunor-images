# train_model.py
import os
import json
import joblib
import pandas as pd
import pyarrow.parquet as pq
from io import BytesIO
from azure.identity import DefaultAzureCredential
from azure.storage.filedatalake import DataLakeServiceClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model

# -------------------------------
# ADLS connection via Managed Identity
# -------------------------------
account_name = "tumorimages60104758"
credential = DefaultAzureCredential()
service_client = DataLakeServiceClient(
    account_url=f"https://{account_name}.dfs.core.windows.net",
    credential=credential
)

gold_container = "gold"
gold_client = service_client.get_file_system_client(gold_container)

# -------------------------------
# Helper to download parquet from ADLS
# -------------------------------
def download_parquet(path):
    file_client = gold_client.get_file_client(path)
    pq_bytes = BytesIO(file_client.download_file().readall())
    return pd.read_parquet(pq_bytes)

# -------------------------------
# Load train/test data
# -------------------------------
train = download_parquet("data_splitting/train.parquet")
test = download_parquet("data_splitting/test.parquet")
print(f"Train: {train.shape}, Test: {test.shape}")

# -------------------------------
# Load selected features
# -------------------------------
features_file = gold_client.get_file_client("feature_selection/selected_features.json")
selected_features = json.loads(features_file.download_file().readall())

# Keep only features that exist in train DataFrame
if isinstance(selected_features, list):
    feature_cols = [f for f in selected_features if f in train.columns]
else:
    feature_cols = [f for f in selected_features.get("selected_features", []) if f in train.columns]

# Fallback if nothing matches
if not feature_cols:
    feature_cols = train.columns.drop(["filename", "label"]).tolist()
    print("Warning: No GA-selected features found, using all columns in train DataFrame except filename and label")

# -------------------------------
# Train Model
# -------------------------------
X_train = train[feature_cols]
y_train = train["label"]
X_test = test[feature_cols]
y_test = test["label"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Model trained. Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(cm)

# -------------------------------
# Save metrics to ADLS
# -------------------------------
metrics = {
    "accuracy": accuracy,
    "confusion_matrix": cm.tolist()
}

metrics_client = gold_client.get_file_client("model_metrics/metrics.json")
metrics_client.upload_data(json.dumps(metrics), overwrite=True)
print("Metrics saved to 'gold/model_metrics/metrics.json'")

# -------------------------------
# Register model in Azure ML
# -------------------------------
subscription_id = "a00dcbea-fd05-4973-82dc-120208b60116"
resource_group = "rg-60104758"
workspace_name = "lab5-60104758"

ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

# Save the trained model locally
model_file = "tumor_rf_model.pkl"
joblib.dump(model, model_file)

# Define model name
model_name = "tumor_rf_model"

# Register in Azure ML
registered_model = Model(
    path=model_file,
    name=model_name,
    type="custom_model",
    description="RandomForest model trained on MRI tumor features"
)
ml_client.models.create_or_update(registered_model)
print(f"Model registered in Azure ML: {model_name}")
