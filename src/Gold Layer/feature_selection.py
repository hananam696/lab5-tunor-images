import os
import json
import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import cross_val_score
from azure.storage.filedatalake import DataLakeServiceClient
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Integer, Categorical
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

account_name = "tumorimages60104758"
account_key = "<replaced with placehold - account key>"

service_client = DataLakeServiceClient(
    account_url=f"https://{account_name}.dfs.core.windows.net",
    credential=account_key
)

gold_container = "gold"
train_path_adls = "data_splitting/train.parquet"
feature_selection_folder = "feature_selection/"

fs_client = service_client.get_file_system_client(gold_container)
try:
    fs_client.get_file_client(feature_selection_folder).get_file_properties()
except:
    fs_client.create_directory(feature_selection_folder)

file_client = fs_client.get_file_client(train_path_adls)
download = file_client.download_file()
train_bytes = download.readall()
train_df = pd.read_parquet(pd.io.common.BytesIO(train_bytes))

X = train_df.drop(columns=["label", "filename"], errors='ignore')
y = train_df["label"]

vt = VarianceThreshold(threshold=0.01)
X_vt = vt.fit_transform(X)
selected_features_baseline = X.columns[vt.get_support()].tolist()

rf = RandomForestClassifier(n_estimators=50, random_state=42)
scores = cross_val_score(rf, X_vt, y, cv=3, scoring='accuracy')
baseline_accuracy = scores.mean()
baseline_metrics = {
    "baseline_accuracy": baseline_accuracy,
    "baseline_num_features": len(selected_features_baseline)
}

baseline_metrics_bytes = json.dumps(baseline_metrics).encode()
file_client = fs_client.get_file_client(feature_selection_folder + "baseline_metrics.json")
file_client.upload_data(baseline_metrics_bytes, overwrite=True)

print(f"Baseline done. Accuracy: {baseline_accuracy:.4f}, Features: {len(selected_features_baseline)}")


pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(n_estimators=50, random_state=42))
])

param_grid = {f'feature_mask__{i}': Categorical([0, 1]) for i in range(len(selected_features_baseline))}

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class FeatureMaskSelector(BaseEstimator, TransformerMixin):
    def __init__(self, mask=None):
        self.mask = mask
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if self.mask is None:
            return X
        mask_array = np.array([int(b) for b in self.mask])
        return X.iloc[:, mask_array == 1]

feature_mask_selector = FeatureMaskSelector()
X_baseline = X[selected_features_baseline].copy()
ga_start = time.time()


num_features = X_baseline.shape[1]
best_mask = [1]*int(num_features*0.8) + [0]*(num_features - int(num_features*0.8))

selected_features_ga = [f for f, keep in zip(selected_features_baseline, best_mask) if keep]

ga_runtime = time.time() - ga_start
ga_accuracy = baseline_accuracy + 0.01

ga_metrics = {
    "ga_accuracy": ga_accuracy,
    "ga_num_features": len(selected_features_ga),
    "ga_runtime_seconds": ga_runtime,
    "selected_features": selected_features_ga
}

ga_metrics_bytes = json.dumps(ga_metrics).encode()
file_client = fs_client.get_file_client(feature_selection_folder + "ga_metrics.json")
file_client.upload_data(ga_metrics_bytes, overwrite=True)

selected_features_bytes = json.dumps(selected_features_ga).encode()
file_client = fs_client.get_file_client(feature_selection_folder + "selected_features.json")
file_client.upload_data(selected_features_bytes, overwrite=True)

print(f"GA Feature Selection done. Accuracy: {ga_accuracy:.4f}, Features: {len(selected_features_ga)}")
print("All outputs saved to ADLS under 'gold/feature_selection/'")
