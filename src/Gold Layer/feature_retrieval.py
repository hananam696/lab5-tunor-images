# feature_retrieval.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from azure.storage.filedatalake import DataLakeServiceClient
import pyarrow as pa
import pyarrow.parquet as pq
from io import BytesIO

account_name = "tumorimages60104758"
account_key = "<account key>" #replaced with placeholder

service_client = DataLakeServiceClient(
    account_url=f"https://{account_name}.dfs.core.windows.net",
    credential=account_key
)

silver_container = "silver"
silver_path = "features_v1/features.parquet"

silver_client = service_client.get_file_system_client(silver_container)
file_client = silver_client.get_file_client(silver_path)

download = file_client.download_file()
pq_bytes = download.readall()

df = pd.read_parquet(BytesIO(pq_bytes))

if "label" not in df.columns:
    raise ValueError("Label column missing in Silver Parquet!")

print(f"Loaded {len(df)} rows and {len(df.columns)} columns from Silver layer.")

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

gold_container = "gold"
split_folder = "data_splitting"
train_path = f"{split_folder}/train.parquet"
test_path = f"{split_folder}/test.parquet"

gold_client = service_client.get_file_system_client(gold_container)

train_client = gold_client.get_file_client(train_path)
train_bytes = BytesIO()
pq.write_table(pa.Table.from_pandas(train_df), train_bytes)
train_bytes.seek(0)
train_client.upload_data(train_bytes.read(), overwrite=True)

test_client = gold_client.get_file_client(test_path)
test_bytes = BytesIO()
pq.write_table(pa.Table.from_pandas(test_df), test_bytes)
test_bytes.seek(0)
test_client.upload_data(test_bytes.read(), overwrite=True)

print("Train/Test split uploaded to ADLS under 'gold/data_splitting/' folder.")
